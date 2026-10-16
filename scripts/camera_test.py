"""Measure what frame rate a webcam really delivers in each capture mode (no model involved).

    python scripts/camera_test.py              # camera 0
    python scripts/camera_test.py --index 1 --frames 60

For each backend (Windows: Media Foundation and DirectShow), with and without MJPG, and at
1280x720 and 640x480, it requests 30 FPS, reads frames and prints the resolution and frame rate
actually delivered. Use the fastest row's settings with the app, e.g.
``python -m ppe_detection --source 0 --backend msmf --fourcc MJPG``.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

MODES = ((1280, 720), (640, 480))


def measure(index: int, api: int, fourcc: str | None, width: int, height: int, frames: int) -> str:
    capture = cv2.VideoCapture(index, api)
    try:
        if not capture.isOpened():
            return "could not open"
        if fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(5):  # let exposure and buffers settle
            capture.read()
        start = time.perf_counter()
        got = sum(1 for _ in range(frames) if capture.read()[0])
        elapsed = time.perf_counter() - start
        if got == 0:
            return "opened but delivered no frames"
        actual = f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        return f"got {actual:>9}  {got / elapsed:5.1f} fps"
    finally:
        capture.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--index", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--frames", type=int, default=30, help="frames to time per mode (default: 30)")
    args = parser.parse_args(argv)

    if sys.platform.startswith("win"):
        backends = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}
    elif sys.platform == "darwin":
        backends = {"avfoundation": cv2.CAP_AVFOUNDATION}
    else:
        backends = {"v4l2": cv2.CAP_V4L2}
    print(f"Camera {args.index}, requesting 30 FPS in every mode:")
    for name, api in backends.items():
        for fourcc in (None, "MJPG"):
            for width, height in MODES:
                result = measure(args.index, api, fourcc, width, height, args.frames)
                print(f"  --backend {name:5} --fourcc {fourcc or '(none)':6} {width}x{height} -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
