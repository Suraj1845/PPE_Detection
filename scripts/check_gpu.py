"""Show where every operation of the model really runs (GPU or CPU), using ONNX Runtime's profiler.

Run it once on each machine after installing requirements.txt:

    python scripts/check_gpu.py                       # D-FINE-N on GPU 0, frames from Samples/ppe1.mp4
    python scripts/check_gpu.py --model models/dfine_s_coco.onnx --frames 100

It loads the model exactly as the application does (same approval check, same provider
selection, no silent CPU fallback), runs it on real frames, and prints:

* the execution provider the session uses,
* how many model operations ran on each provider and their share of the kernel time,
* which operation types (if any) ONNX Runtime kept on the CPU,
* the average time per frame (frame upload + preprocessing + network + top-300 selection).

ONNX Runtime deliberately keeps a few tiny shape-calculation operations on the CPU even with
CUDA; they show up in the list but take a negligible share of the time. Exit code 0 means the
GPU provider ran at least 95 % of the kernel time, 1 means it did not, 3 means the model or GPU
could not be loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ppe_detection import PROJECT_ROOT  # noqa: E402
from ppe_detection.config import DEFAULT_MODEL, resolve_path  # noqa: E402
from ppe_detection.detector import PROVIDERS, DFineDetector, ModelError, _import_onnxruntime  # noqa: E402

GPU_SHARE_REQUIRED = 0.95


def load_frames(source: Path | None, count: int) -> list[np.ndarray]:
    """Frames from a video or image, or a synthetic 1280x720 frame if none is available."""
    if source is not None and source.is_file():
        capture = cv2.VideoCapture(str(source))
        frames = []
        try:
            while len(frames) < count:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        if frames:
            return frames
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)]


def profiled_session_factory(profile_dir: Path):
    def factory(path: str, providers):
        ort = _import_onnxruntime()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_dir / "ppe_profile")
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    return factory


def summarise(profile_file: Path) -> tuple[dict[str, float], dict[str, set[str]], Counter]:
    """Kernel time (microseconds) per provider, node names per provider, CPU op types."""
    events = json.loads(profile_file.read_text(encoding="utf-8"))
    time_us: dict[str, float] = defaultdict(float)
    nodes: dict[str, set[str]] = defaultdict(set)
    cpu_ops: Counter = Counter()
    for event in events:
        args = event.get("args", {})
        if event.get("cat") != "Node" or not event.get("name", "").endswith("_kernel_time") or "provider" not in args:
            continue
        provider = args["provider"]
        time_us[provider] += float(event.get("dur", 0))
        if event["name"] not in nodes[provider]:
            nodes[provider].add(event["name"])
            if provider == PROVIDERS["cpu"]:
                cpu_ops[args.get("op_name", "?")] += 1
    return time_us, nodes, cpu_ops


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda", choices=["cuda", "directml", "cpu"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "Samples" / "ppe1.mp4")
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args(argv)

    frames = load_frames(args.source, args.frames + 5)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            detector = DFineDetector(
                resolve_path(args.model, PROJECT_ROOT),
                device=args.device,
                gpu_id=args.gpu_id,
                session_factory=profiled_session_factory(Path(tmp)),
            )
        except ModelError as exc:
            print(f"FAILED to load on {args.device}: {exc}")
            return 3
        for frame in frames[:5]:  # warm-up (CUDA kernel selection, memory arenas)
            detector.detect(frame, 0.5)
        timed = (frames * (args.frames // len(frames) + 1))[: args.frames]
        start = time.perf_counter()
        for frame in timed:
            detector.detect(frame, 0.5)
        per_frame_ms = (time.perf_counter() - start) * 1000.0 / len(timed)
        profile_file = Path(detector._session.end_profiling())
        time_us, nodes, cpu_ops = summarise(profile_file)

    total = sum(time_us.values()) or 1.0
    provider = detector.info.provider
    print(f"Model: {detector.info.path.name}   session provider: {provider}   frame size: "
          f"{timed[0].shape[1]}x{timed[0].shape[0]}")
    for name in sorted(time_us, key=time_us.get, reverse=True):
        print(f"  {name:28} {len(nodes[name]):4d} operations  {100.0 * time_us[name] / total:6.2f} % of kernel time")
    if cpu_ops:
        listed = ", ".join(f"{op} x{n}" for op, n in cpu_ops.most_common())
        print(f"  operations kept on the CPU: {listed}")
    print(f"Average per frame over {len(timed)} frames: {per_frame_ms:.1f} ms "
          "(upload + in-graph preprocessing + network + top-300 selection)")

    if args.device == "cpu":
        return 0
    gpu_share = time_us.get(provider, 0.0) / total if provider != PROVIDERS["cpu"] else 0.0
    verdict = "OK" if gpu_share >= GPU_SHARE_REQUIRED else "NOT ENOUGH ON THE GPU"
    print(f"{verdict}: {100.0 * gpu_share:.1f} % of kernel time on {provider} (required {GPU_SHARE_REQUIRED:.0%}).")
    return 0 if gpu_share >= GPU_SHARE_REQUIRED else 1


if __name__ == "__main__":
    raise SystemExit(main())
