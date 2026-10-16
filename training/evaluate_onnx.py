"""Measure an exported PPE model's accuracy on the test split, exactly as the app runs it.

    python training\\evaluate_onnx.py --model models\\dfine_s_ppe.onnx
    python training\\evaluate_onnx.py --model models\\dfine_s_ppe.onnx --conf 0.4

It runs the ONNX file through the app's own detector code (``ppe_detection.detector``) on every
test image, then reports:

* COCO-style AP (IoU 0.50:0.95) and AP50 (IoU 0.50), overall and **per class**;
* at the app's confidence threshold (``--conf``, default 0.5, IoU 0.5): precision, recall and
  missed/false boxes per class, which is what you will see on screen.

It writes ``training\\work\\eval_<model>.json`` and ``training\\work\\record_<model>.md``: the
``models\\SHA256SUMS`` line and a ``MODEL_LICENSES.md`` section, filled in with the dataset's
source and licence lines, the checkpoint lineage and these measured numbers. Review that text and
paste it in only when you accept the model; the app will not load the model before that (unless
``--allow-unverified-model`` is passed).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parent
PROJECT = TRAINING.parent
sys.path.insert(0, str(PROJECT))

from ppe_detection.detector import DFineDetector  # noqa: E402


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def operating_point(gt: dict, results: list[dict], names: list[str], conf: float) -> dict[str, dict[str, float]]:
    """Per-class TP/FP/FN at ``conf`` with greedy IoU>=0.5 matching (highest score first)."""
    truth = defaultdict(list)
    for ann in gt["annotations"]:
        truth[(ann["image_id"], ann["category_id"])].append(ann["bbox"])
    preds = defaultdict(list)
    for det in results:
        if det["score"] >= conf:
            preds[(det["image_id"], det["category_id"])].append(det)
    table = {}
    for k, name in enumerate(names):
        tp = fp = fn = 0
        image_ids = {i for (i, c) in truth if c == k} | {i for (i, c) in preds if c == k}
        for image_id in image_ids:
            boxes = list(truth.get((image_id, k), []))
            used = [False] * len(boxes)
            for det in sorted(preds.get((image_id, k), []), key=lambda d: -d["score"]):
                best, best_iou = -1, 0.5
                for j, box in enumerate(boxes):
                    if not used[j] and iou(det["bbox"], box) >= best_iou:
                        best, best_iou = j, iou(det["bbox"], box)
                if best >= 0:
                    used[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += used.count(False)
        table[name] = {
            "instances": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "missed": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        }
    return table


def coco_ap(gt: dict, results: list[dict], names: list[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Overall and per-class AP (0.50:0.95) and AP50 with faster-coco-eval."""
    from faster_coco_eval import COCO, COCOeval_faster

    logging.getLogger("faster_coco_eval").setLevel(logging.ERROR)
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    if not results:
        empty = {"AP": 0.0, "AP50": 0.0}
        return empty, {n: dict(empty) for n in names}
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]  # [IoU thresholds, recall points, classes, areas, max dets]
    per_class = {}
    for k, name in enumerate(names):
        p_all = precision[:, :, k, 0, -1]
        p_50 = precision[0, :, k, 0, -1]
        per_class[name] = {
            "AP": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan"),
            "AP50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else float("nan"),
        }
    return {"AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1])}, per_class


def audited_checkpoints() -> dict:
    """The audited COCO checkpoint table from scripts/export_dfine_onnx.py."""
    spec = importlib.util.spec_from_file_location("export_dfine_onnx", PROJECT / "scripts" / "export_dfine_onnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AUDITED_CHECKPOINTS


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float) -> str:
    return "  n/a" if math.isnan(value) else f"{value:5.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="exported ONNX model")
    parser.add_argument("--data", type=Path, default=TRAINING / "work" / "data", help="output of prepare_dataset.py")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold used in the app")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="ONNX Runtime device")
    args = parser.parse_args(argv)

    data = args.data.resolve()
    split = args.split if (data / f"{args.split}.json").is_file() else "val"
    if split != args.split:
        print("! No test split: evaluating on the validation split, which also selected the best checkpoint "
              "during training, so these numbers will be optimistic.")
    gt = json.loads((data / f"{split}.json").read_text(encoding="utf-8"))
    info = json.loads((data / "dataset_info.json").read_text(encoding="utf-8"))
    names = json.loads((data / "labels.json").read_text(encoding="utf-8"))
    folder = Path(info["image_folders"][split])

    detector = DFineDetector(args.model.resolve(), device=args.device, allow_unverified=True)
    if list(detector.labels) != names:
        raise SystemExit(f"The model's classes {list(detector.labels)} differ from the dataset's {names}.")

    from PIL import Image

    results = []
    for n, image in enumerate(gt["images"], start=1):
        rgb = np.asarray(Image.open(folder / image["file_name"]).convert("RGB"))
        frame = np.ascontiguousarray(rgb[:, :, ::-1])  # BGR, as the camera/OpenCV delivers it
        for det in detector.detect(frame, conf=0.001):
            x1, y1, x2, y2 = det.box
            results.append({"image_id": image["id"], "category_id": det.class_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": det.score})
        if n % 50 == 0:
            print(f"  {n}/{len(gt['images'])} images")

    overall, per_class_ap = coco_ap(gt, results, names)
    at_conf = operating_point(gt, results, names, args.conf)

    print(f"\n{args.model.name} on the {split} split: {len(gt['images'])} images, "
          f"{len(gt['annotations'])} boxes. Overall AP {overall['AP']:.3f}, AP50 {overall['AP50']:.3f}")
    print(f"\n  {'class':16} {'boxes':>6} {'AP':>6} {'AP50':>6}   at conf {args.conf}: "
          f"{'precision':>9} {'recall':>7} {'missed':>7} {'false':>6}")
    for name in names:
        ap, op = per_class_ap[name], at_conf[name]
        print(f"  {name:16} {op['instances']:6d} {fmt(ap['AP']):>6} {fmt(ap['AP50']):>6}   "
              f"{'':9}{fmt(op['precision']):>9} {fmt(op['recall']):>7} {op['missed']:7d} {op['false_positives']:6d}")

    report = {"model": str(args.model), "split": split, "conf": args.conf, "overall": overall,
              "per_class_ap": per_class_ap, "at_conf": at_conf}
    stem = args.model.stem
    (TRAINING / "work" / f"eval_{stem}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    meta = detector.info.metadata
    variant = re.search(r"dfine_([a-z])_", meta.get("ppe_detection.dfine_config", ""))
    audited = audited_checkpoints().get(variant.group(1)) if variant else None
    start = (f"`{audited['file']}` (D-FINE COCO, Apache-2.0), SHA-256 `{audited['sha256']}`" if audited
             else "**fill in: the D-FINE COCO checkpoint training started from**")
    digest = sha256_of(args.model)
    licence_lines = [f"`[{f}]` {line}" for f, lines in info.get("licence_lines", {}).items() for line in lines]
    rows = "\n".join(
        f"| {n} | {at_conf[n]['instances']} | {fmt(per_class_ap[n]['AP'])} | {fmt(per_class_ap[n]['AP50'])} | "
        f"{fmt(at_conf[n]['precision'])} | {fmt(at_conf[n]['recall'])} |"
        for n in names
    )
    record = f"""## {stem} - fill in, review, then paste into MODEL_LICENSES.md

Line for `models/SHA256SUMS` (add only after you accept this model):

```text
{digest}  {args.model.name}
```

| Field | Value |
| --- | --- |
| Model | D-FINE fine-tuned on PPE classes: {', '.join(names)} |
| Artifact | `models/{args.model.name}`, {args.model.stat().st_size:,} bytes, SHA-256 `{digest}` |
| Starting weights | {start} |
| Exported checkpoint | `{meta.get('ppe_detection.source_checkpoint', '?')}`, SHA-256 `{meta.get('ppe_detection.source_checkpoint_sha256', '?')}` |
| Code | D-FINE commit `{meta.get('ppe_detection.dfine_commit', '?')}` (Apache-2.0), config `{meta.get('ppe_detection.dfine_config', '?')}` |
| Training data | `{info['dataset_folder']}` ({info['format']} export). Licence/source lines found: {'; '.join(licence_lines) or '**none found: record the dataset licence and source here**'} |
| Licence of these weights | **To be decided and filled in by the model's owner.** Base weights and code are Apache-2.0; the training data's terms above (for example CC BY 4.0 attribution) apply as well. Metadata says: "{meta.get('ppe_detection.license', '?')}" |
| Split sizes (images) | {', '.join(f"{s}: {c['images']}" for s, c in info['counts'].items())} |
| Evaluation | {split} split, {len(gt['images'])} images: AP {overall['AP']:.3f}, AP50 {overall['AP50']:.3f} |

| Class | Boxes | AP | AP50 | Precision @ {args.conf} | Recall @ {args.conf} |
| --- | --- | --- | --- | --- | --- |
{rows}
"""
    record_path = TRAINING / "work" / f"record_{stem}.md"
    record_path.write_text(record, encoding="utf-8")
    print(f"\nSaved {TRAINING / 'work' / f'eval_{stem}.json'} and {record_path}")
    print("Review the per-class numbers (especially Hardhat vs NO-Hardhat) before approving the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
