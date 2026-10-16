"""Export a D-FINE checkpoint to an ONNX file that ppe_detection can load.

This script runs in a SEPARATE export environment (PyTorch + onnx + onnxsim),
never in the application environment. It:

1. Checks that the D-FINE repository is at the audited commit.
2. Checks the checkpoint's SHA-256 against the audited value (COCO checkpoints)
   or prints it for the record (your own fine-tuned checkpoints).
3. Exports with the same logic as D-FINE's ``tools/deployment/export_onnx.py``
   (see ``trace_and_export``), then runs onnx.checker and onnxsim as it does.
   (Never use ``tools/deployment/export_yolo_w_nms.py``: it imports the
   AGPL-licensed ``ultralytics`` package.)
4. Prepends the image preprocessing (BGR->RGB, resize to 640x640, scale to 0-1)
   to the graph as ONNX operators, so it runs on the GPU together with the
   network. The finished model takes one raw ``uint8`` BGR frame of any size.
5. Embeds the class names and the input contract into the ONNX metadata, so the
   application never has to guess class indices.
6. Optionally (``--verify-image``) checks the network against PyTorch, and the
   in-graph preprocessing against the same resize done in PyTorch.
7. Prints the SHA-256 of the result, to be added to ``models/SHA256SUMS``.

Example (COCO D-FINE-N, from the project folder)::

    python scripts/export_dfine_onnx.py --dfine-repo ../D-FINE --variant n \
        --checkpoint ../weights/dfine_n_coco.pth --output models/dfine_n_coco.onnx

Example (your own PPE fine-tune)::

    python scripts/export_dfine_onnx.py --dfine-repo training/D-FINE \
        --config training/work/dfine_s_ppe.yml \
        --checkpoint training/work/runs/dfine_s_ppe/best_stg2.pth \
        --labels-json training/work/data/labels.json --output models/dfine_s_ppe.onnx

(``training/train.py`` prints this command with your paths.) The licence stored in a
fine-tune's metadata is "not declared" unless ``--license`` is given: the base weights are
Apache-2.0, but the licence of the fine-tuned weights is the owner's decision and has to take
the training data's terms into account.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

AUDITED_DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"

# Audited COCO-only checkpoints from https://github.com/Peterande/storage/releases/tag/dfinev1.0
AUDITED_CHECKPOINTS = {
    "n": {
        "file": "dfine_n_coco.pth",
        "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth",
        "sha256": "41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c",
        "config": "configs/dfine/dfine_hgnetv2_n_coco.yml",
    },
    "s": {
        "file": "dfine_s_coco.pth",
        "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        "sha256": "48a6c8cc43eb57186843f752e2e8461ddd3326e0d3c575e71e6e960844683e89",
        "config": "configs/dfine/dfine_hgnetv2_s_coco.yml",
    },
}

INPUT_SIZE = 640  # D-FINE's exporter traces at 640x640; its val transform is Resize([640, 640]).
FORMAT_VERSION = "2"  # 2 = single uint8 BGR "frame" input with in-graph preprocessing


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    """The D-FINE commit: from git, or from the DFINE_COMMIT.txt marker that
    training/setup_dfine.py writes when it downloads the source archive."""
    marker = repo / "DFINE_COMMIT.txt"
    if (repo / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    raise SystemExit(f"Cannot tell which D-FINE commit {repo} is: no .git folder and no {marker.name}.")


def coco_labels(repo: Path) -> list[str]:
    """Read the COCO class names in D-FINE's contiguous label order.

    In deploy mode D-FINE's postprocessor returns contiguous indices 0..79 (no
    COCO category-id remapping), and ``mscoco_category2label`` enumerates
    ``mscoco_category2name`` in insertion order, so index i -> i-th name.
    """
    sys.path.insert(0, str(repo))
    from src.data.dataset.coco_dataset import mscoco_category2name  # type: ignore

    names = list(mscoco_category2name.values())
    if len(names) != 80:
        raise SystemExit(f"Expected 80 COCO classes, found {len(names)}")
    return names


def build_deploy_model(repo: Path, config: str, checkpoint: Path):
    """Build D-FINE's deploy-mode model + postprocessor, as its exporter does.

    Mirrors ``tools/deployment/export_onnx.py`` at commit 956d170, except that
    the checkpoint is loaded with ``weights_only=True`` so no pickled code runs.
    """
    import os

    import torch
    from torch import nn

    sys.path.insert(0, str(repo))
    previous_cwd = os.getcwd()
    os.chdir(repo)  # D-FINE configs use paths relative to the repo root
    try:
        from src.core import YAMLConfig  # type: ignore

        cfg = YAMLConfig(config, resume=str(checkpoint))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # never download backbone weights
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        state = state["ema"]["module"] if "ema" in state else state["model"]
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)

        return Model().eval()
    finally:
        os.chdir(previous_cwd)


def trace_and_export(model, output: Path, batch: int) -> None:
    """Export with the upstream settings: names, dynamic batch axis, opset 16, folding.

    The tracing batch defaults to 2 instead of the upstream 32 (32 needs far more
    RAM). It must stay above 1: the decoder only traces its per-batch anchor
    ``repeat`` when ``memory.shape[0] > 1``.
    """
    import torch

    data = torch.rand(batch, 3, INPUT_SIZE, INPUT_SIZE)
    size = torch.tensor([[INPUT_SIZE, INPUT_SIZE]] * batch)
    with torch.no_grad():
        _ = model(data, size)
        torch.onnx.export(
            model,
            (data, size),
            str(output),
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
            opset_version=16,
            verbose=False,
            do_constant_folding=True,
            dynamo=False,  # the TorchScript exporter the upstream script was written for
        )


def verify_parity(model, onnx_path: Path, images: list[Path], min_score: float = 0.3) -> None:
    """Compare PyTorch and ONNX Runtime outputs on real images, batch 1 and batch 2.

    Both sides get identical input tensors (PIL stretch to 640x640, RGB, scale 1/255),
    so this checks the exported network itself. Fails if confident detections differ.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image

    tensors, sizes = [], []
    for path in images:
        image = Image.open(path).convert("RGB")
        sizes.append([image.width, image.height])
        resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        tensors.append(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    batch = np.stack(tensors)
    size = np.asarray(sizes, dtype=np.int64)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(batch), torch.from_numpy(size))]
    singles = [
        session.run(None, {"images": batch[i : i + 1], "orig_target_sizes": size[i : i + 1]})
        for i in range(len(images))
    ]
    runs = {
        "batch": session.run(None, {"images": batch, "orig_target_sizes": size}),
        "single": [np.concatenate([run[k] for run in singles]) for k in range(3)],
    }

    for name, (labels, boxes, scores) in runs.items():
        for i, path in enumerate(images):
            keep_ref = ref[2][i] > min_score
            keep = scores[i] > min_score
            if keep_ref.sum() != keep.sum():
                raise SystemExit(f"{name}: {path.name}: {keep.sum()} vs {keep_ref.sum()} detections")
            order_ref, order = np.argsort(-ref[2][i][keep_ref]), np.argsort(-scores[i][keep])
            same_labels = np.array_equal(ref[0][i][keep_ref][order_ref], labels[i][keep][order])
            box_err = np.abs(ref[1][i][keep_ref][order_ref] - boxes[i][keep][order]).max(initial=0.0)
            score_err = np.abs(ref[2][i][keep_ref][order_ref] - scores[i][keep][order]).max(initial=0.0)
            # Also compare the 10 highest scores whatever their value, so a model that is not yet
            # confident (early in training) is still checked.
            top_err = np.abs(np.sort(ref[2][i])[::-1][:10] - np.sort(scores[i])[::-1][:10]).max(initial=0.0)
            print(
                f"parity {name:8} {path.name}: {int(keep.sum())} detections > {min_score}, "
                f"labels equal={same_labels}, max box diff={box_err:.3f}px, max score diff={score_err:.5f}, "
                f"top-10 score diff={top_err:.5f}"
            )
            if not same_labels or box_err > 1.0 or score_err > 1e-3 or top_err > 1e-3:
                raise SystemExit("ONNX output does not match PyTorch.")


def add_preprocessing(network):
    """Return ``network`` with preprocessing operators in front of it.

    New input ``frame``: uint8 [height, width, 3] in OpenCV's BGR order, any size.
    The added operators cast to float, reorder BGR->RGB, transpose to NCHW, resize
    to 640x640 (bilinear, half-pixel centres, as OpenCV INTER_LINEAR and PyTorch
    ``interpolate(align_corners=False)``), scale to 0-1, and compute
    ``orig_target_sizes`` = (width, height) from the frame's shape, so the boxes
    come out in the frame's own pixels. All of these run on the execution provider
    (CUDA on a GPU) together with the network.
    """
    from onnx import TensorProto, compose, helper

    opset = next(o.version for o in network.opset_import if o.domain in ("", "ai.onnx"))
    nodes = [
        helper.make_node("Cast", ["frame"], ["pp_float"], to=TensorProto.FLOAT),
        helper.make_node("Gather", ["pp_float", "pp_bgr_to_rgb"], ["pp_rgb"], axis=2),
        helper.make_node("Transpose", ["pp_rgb"], ["pp_chw"], perm=[2, 0, 1]),
        helper.make_node("Unsqueeze", ["pp_chw", "pp_axis0"], ["pp_nchw"]),
        helper.make_node(
            "Resize", ["pp_nchw", "", "", "pp_size"], ["pp_resized"],
            mode="linear", coordinate_transformation_mode="half_pixel",
        ),
        helper.make_node("Mul", ["pp_resized", "pp_inv255"], ["pp_images"]),
        helper.make_node("Shape", ["frame"], ["pp_hw3"]),
        helper.make_node("Gather", ["pp_hw3", "pp_wh_index"], ["pp_wh"], axis=0),
        helper.make_node("Unsqueeze", ["pp_wh", "pp_axis0"], ["pp_sizes"]),
    ]
    constants = [
        helper.make_tensor("pp_bgr_to_rgb", TensorProto.INT64, [3], [2, 1, 0]),
        helper.make_tensor("pp_axis0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("pp_size", TensorProto.INT64, [4], [1, 3, INPUT_SIZE, INPUT_SIZE]),
        helper.make_tensor("pp_inv255", TensorProto.FLOAT, [], [1.0 / 255.0]),
        helper.make_tensor("pp_wh_index", TensorProto.INT64, [2], [1, 0]),
    ]
    graph = helper.make_graph(
        nodes,
        "preprocess",
        [helper.make_tensor_value_info("frame", TensorProto.UINT8, ["height", "width", 3])],
        [
            helper.make_tensor_value_info("pp_images", TensorProto.FLOAT, [1, 3, INPUT_SIZE, INPUT_SIZE]),
            helper.make_tensor_value_info("pp_sizes", TensorProto.INT64, [1, 2]),
        ],
        constants,
    )
    pre = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    pre.ir_version = network.ir_version
    return compose.merge_models(pre, network, io_map=[("pp_images", "images"), ("pp_sizes", "orig_target_sizes")])


def verify_preprocessing(network_path: Path, fused_path: Path, images: list[Path], min_score: float = 0.3) -> None:
    """The fused model on a raw BGR frame must equal the bare network fed the same
    resize computed by PyTorch (bilinear, align_corners=False, no antialias)."""
    import numpy as np
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from PIL import Image

    network = ort.InferenceSession(str(network_path), providers=["CPUExecutionProvider"])
    fused = ort.InferenceSession(str(fused_path), providers=["CPUExecutionProvider"])
    for path in images:
        bgr = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])
        height, width = bgr.shape[:2]
        rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1)[None].float()
        ref_images = F.interpolate(rgb, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False) / 255.0
        ref = network.run(None, {"images": ref_images.numpy(), "orig_target_sizes": np.array([[width, height]])})
        out = fused.run(None, {"frame": bgr})
        keep_ref, keep = ref[2][0] > min_score, out[2][0] > min_score
        same = keep_ref.sum() == keep.sum() and np.array_equal(
            ref[0][0][keep_ref][np.argsort(-ref[2][0][keep_ref])], out[0][0][keep][np.argsort(-out[2][0][keep])]
        )
        score_err = np.abs(np.sort(ref[2][0][keep_ref]) - np.sort(out[2][0][keep])).max(initial=0.0) if same else 1.0
        print(f"in-graph preprocessing {path.name}: {int(keep.sum())} detections > {min_score}, "
              f"labels equal={same}, max score diff={score_err:.5f}")
        if not same or score_err > 1e-3:
            raise SystemExit("In-graph preprocessing does not match the PyTorch reference.")


def simplify_and_load(path: Path, batch: int):
    """Run onnx.checker and onnxsim, as the upstream exporter does by default."""
    import onnx
    import onnxsim

    onnx.checker.check_model(onnx.load(str(path)))
    shapes = {"images": [batch, 3, INPUT_SIZE, INPUT_SIZE], "orig_target_sizes": [batch, 2]}
    simplified, ok = onnxsim.simplify(str(path), test_input_shapes=shapes)
    if not ok:
        raise SystemExit("onnxsim could not validate the simplified model.")
    return simplified


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dfine-repo", type=Path, required=True, help="Path to a D-FINE clone")
    parser.add_argument("--checkpoint", type=Path, required=True, help="D-FINE .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx path")
    parser.add_argument("--variant", choices=sorted(AUDITED_CHECKPOINTS), help="Audited COCO variant")
    parser.add_argument("--config", help="D-FINE config (relative to the repo) for non-COCO models")
    parser.add_argument("--labels-json", type=Path, help="JSON list of class names, index order")
    parser.add_argument(
        "--license",
        help="Licence recorded in the model's metadata (COCO variants: Apache-2.0; fine-tunes: 'not declared' "
        "unless given)",
    )
    parser.add_argument(
        "--allow-other-commit", action="store_true", help="Allow a D-FINE commit other than the audited one"
    )
    parser.add_argument(
        "--trace-batch", type=int, default=2, help="Batch size of the tracing input (upstream uses 32; must be >1)"
    )
    parser.add_argument(
        "--verify-image",
        type=Path,
        action="append",
        default=[],
        help="Image to compare PyTorch vs ONNX Runtime outputs on (repeatable; use 2+)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trace_batch < 2:
        raise SystemExit("--trace-batch must be at least 2 (see trace_and_export).")
    repo = args.dfine_repo.resolve()
    checkpoint = args.checkpoint.resolve()

    commit = git_commit(repo)
    if commit != AUDITED_DFINE_COMMIT and not args.allow_other_commit:
        raise SystemExit(f"D-FINE repo is at {commit}, audited commit is {AUDITED_DFINE_COMMIT}.")

    checkpoint_sha = sha256_of(checkpoint)
    if args.variant:
        audited = AUDITED_CHECKPOINTS[args.variant]
        if checkpoint_sha != audited["sha256"]:
            raise SystemExit(
                f"Checkpoint SHA-256 {checkpoint_sha} does not match the audited "
                f"{audited['file']} ({audited['sha256']})."
            )
        config = audited["config"]
        config_label = config
        labels = coco_labels(repo)
        source = audited["url"]
    else:
        if not (args.config and args.labels_json):
            raise SystemExit("Non-COCO exports need --config and --labels-json.")
        # A config path given relative to the current folder (e.g. training/work/...) must survive
        # the chdir into the D-FINE repo; otherwise it is taken as relative to the repo, as before.
        config = str(Path(args.config).resolve()) if Path(args.config).is_file() else args.config
        config_label = Path(args.config).name
        labels = json.loads(args.labels_json.read_text(encoding="utf-8"))
        if not (isinstance(labels, list) and all(isinstance(x, str) and x for x in labels)):
            raise SystemExit("--labels-json must contain a JSON list of non-empty strings.")
        source = str(checkpoint.name)

    import onnx  # imported late so --help works without the export dependencies

    with tempfile.TemporaryDirectory() as tmp:
        raw_onnx = Path(tmp) / "model.onnx"
        network_onnx = Path(tmp) / "network.onnx"
        torch_model = build_deploy_model(repo, config, checkpoint)
        trace_and_export(torch_model, raw_onnx, args.trace_batch)
        network = simplify_and_load(raw_onnx, args.trace_batch)
        onnx.save(network, str(network_onnx))
        if args.verify_image:
            verify_parity(torch_model, network_onnx, args.verify_image)  # network vs PyTorch, batch 1 and 2
        model = add_preprocessing(network)
        onnx.checker.check_model(model)
        if args.verify_image:
            fused_onnx = Path(tmp) / "fused.onnx"
            onnx.save(model, str(fused_onnx))
            verify_preprocessing(network_onnx, fused_onnx, args.verify_image)

    metadata = {
        "ppe_detection.format_version": FORMAT_VERSION,
        "ppe_detection.model_family": "dfine",
        "ppe_detection.labels": json.dumps(labels),
        "ppe_detection.input_width": str(INPUT_SIZE),
        "ppe_detection.input_height": str(INPUT_SIZE),
        "ppe_detection.input": "frame: uint8 [height, width, 3], BGR (OpenCV order)",
        "ppe_detection.preprocessing": "in-graph: BGR->RGB, bilinear stretch to 640x640 (half_pixel), x/255",
        "ppe_detection.source_checkpoint": source,
        "ppe_detection.source_checkpoint_sha256": checkpoint_sha,
        "ppe_detection.dfine_commit": commit,
        "ppe_detection.dfine_config": config_label,
        "ppe_detection.license": args.license or ("Apache-2.0" if args.variant else "not declared"),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"Wrote {args.output} ({len(labels)} classes)")
    print(f"{sha256_of(args.output)}  {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
