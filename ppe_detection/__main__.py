"""Command-line entry point: ``python -m ppe_detection --source 0``.

Exit codes: 0 success, 1 unexpected error, 2 invalid settings (including a missing input file),
3 model problem, 4 camera/video/image/output problem, 130 interrupted with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import __version__
from .camera import FrameSource, OutputWriter, SourceError, list_cameras, open_source
from .compliance import FrameAssessment, PpeSchema, Status, assess, undetectable
from .config import (
    BACKENDS,
    DEVICES,
    IMAGE_SUFFIXES,
    AppConfig,
    ConfigError,
    SourceKind,
    build_config,
    load_toml,
    parse_source,
)
from .detector import DFineDetector, ModelError
from .visualization import draw_frame, draw_status_bar, status_bar_height

log = logging.getLogger("ppe_detection")

WINDOW_TITLE = "PPE Detection  (q / Esc to quit)"
QUIT_KEYS = frozenset({ord("q"), ord("Q"), 27})  # 27 = Escape
EXIT_OK, EXIT_UNEXPECTED, EXIT_CONFIG, EXIT_MODEL, EXIT_SOURCE, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection",
        description="Real-time PPE detection from a webcam, video or image with a D-FINE ONNX model.",
        epilog="Examples:\n"
        "  python -m ppe_detection --source 0\n"
        "  python -m ppe_detection --source Samples/ppe1.mp4\n"
        "  python -m ppe_detection --source 0 --conf 0.50\n"
        "  python -m ppe_detection --list-cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", help="camera index (0, 1, ...) or a video/image file path (default: 0)")
    parser.add_argument("--model", type=Path, help="ONNX model, relative to the project folder")
    parser.add_argument("--conf", type=float, help="minimum confidence 0-1 (default: 0.5)")
    parser.add_argument("--width", type=int, help="requested camera width (default: 1280)")
    parser.add_argument("--height", type=int, help="requested camera height (default: 720)")
    parser.add_argument(
        "--device", choices=DEVICES,
        help="cuda (default: NVIDIA GPU, required), directml (other Windows GPUs), cpu (explicit CPU run), "
        "auto (best available, may fall back to CPU)",
    )
    parser.add_argument("--gpu-id", type=int, help="GPU index for cuda/directml (default: 0)")
    parser.add_argument(
        "--backend", choices=BACKENDS,
        help="camera capture backend (default: auto = Media Foundation, then DirectShow on Windows)",
    )
    parser.add_argument("--fourcc", help="camera pixel format to request, e.g. MJPG (optional)")
    parser.add_argument("--fps", type=int, help="frame rate to request from the camera (default: 30)")
    parser.add_argument("--output", type=Path, help="save annotated result (.mp4, .avi or an image file)")
    parser.add_argument("--no-display", action="store_true", help="do not open a window (use with --output)")
    parser.add_argument("--classes", help="comma-separated class names to show, or 'all'")
    parser.add_argument("--required-ppe", help="comma-separated items a person must wear (default: hardhat,vest)")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--allow-unverified-model", action="store_true", help="run a model not in models/SHA256SUMS")
    parser.add_argument("--config", type=Path, help="TOML settings file; command-line options override it")
    parser.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    parser.add_argument("--max-camera-index", type=int, default=6, help="indices probed by --list-cameras")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--version", action="version", version=f"ppe_detection {__version__}")
    return parser


def _split(value: str | None) -> tuple[str, ...] | None:
    return None if value is None else tuple(part.strip() for part in value.split(",") if part.strip())


def config_from_args(args: argparse.Namespace) -> AppConfig:
    base = load_toml(args.config) if args.config else None
    return build_config(
        base,
        source=args.source,
        model=args.model,
        conf=args.conf,
        width=args.width,
        height=args.height,
        device=args.device,
        gpu_id=args.gpu_id,
        backend=args.backend,
        output=args.output,
        display=False if args.no_display else None,
        classes=_split(args.classes),
        required_ppe=_split(args.required_ppe),
        max_frames=args.max_frames,
        allow_unverified_model=True if args.allow_unverified_model else None,
        fourcc=args.fourcc,
        fps=args.fps,
    )


# ------------------------------------------------------------------- display


class CvDisplay:
    """The OpenCV window. ``show`` returns False when the user asks to quit."""

    def __init__(self) -> None:
        self._open = False

    def _ensure_window(self, frame: np.ndarray) -> None:
        if self._open:
            return
        try:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, min(frame.shape[1], 1600), min(frame.shape[0], 900))
        except cv2.error as exc:
            raise SourceError(
                "Could not open a display window (is this a headless OpenCV build or a session without a "
                f"desktop?). Use --no-display with --output instead. OpenCV said: {exc}"
            ) from exc
        self._open = True

    def _closed_by_user(self) -> bool:
        try:
            return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def show(self, frame: np.ndarray, wait_for_key: bool = False) -> bool:
        self._ensure_window(frame)
        cv2.imshow(WINDOW_TITLE, frame)
        while True:
            key = cv2.waitKey(100 if wait_for_key else 1) & 0xFF
            if key in QUIT_KEYS or (wait_for_key and key != 0xFF) or self._closed_by_user():
                return False
            if not wait_for_key:
                return True

    def close(self) -> None:
        if self._open:
            self._open = False
            try:
                cv2.destroyWindow(WINDOW_TITLE)
                cv2.waitKey(1)  # let the GUI thread process the close on Windows
            except cv2.error:
                pass


# --------------------------------------------------------------------- run


@dataclass
class RunStats:
    frames: int = 0
    seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def resolve_class_filter(
    requested: tuple[str, ...] | None, labels: Sequence[str], schema: PpeSchema
) -> frozenset[str] | None:
    """Exact class names to show, or None for all of them."""
    if requested is None:
        # A PPE model shows everything; a general model (e.g. COCO) shows only people.
        return None if schema.ppe_supported or not schema.person_labels else frozenset(schema.person_labels)
    if any(name.lower() == "all" for name in requested):
        return None
    by_lower = {label.lower(): label for label in labels}
    unknown = [name for name in requested if name.lower() not in by_lower]
    if unknown:
        raise ConfigError(f"Unknown class name(s) {unknown}. This model's classes are: {', '.join(labels)}.")
    return frozenset(by_lower[name.lower()] for name in requested)


def status_lines(fps: float, inference_ms: float, provider: str, assessment: FrameAssessment, shown: int) -> list[str]:
    lines = [f"FPS {fps:5.1f} | inference {inference_ms:5.1f} ms | {provider.replace('ExecutionProvider', '')}"]
    if assessment.ppe_supported:
        lines.append(
            f"People {len(assessment.persons)}: {assessment.count(Status.VIOLATION)} violation, "
            f"{assessment.count(Status.COMPLIANT)} OK, {assessment.count(Status.UNKNOWN)} unknown"
            + (f" | unattributed PPE boxes {len(assessment.unattributed)}" if assessment.unattributed else "")
        )
    else:
        lines.append(f"Detections shown: {shown}")
    return lines


def model_warning(detector: Any, schema: PpeSchema, class_filter: frozenset[str] | None) -> str | None:
    parts = []
    if not detector.info.approved:
        parts.append("UNVERIFIED MODEL")
    if not schema.ppe_supported:
        shown = ", ".join(sorted(class_filter)) if class_filter else "all classes"
        parts.append(f"PPE model not installed - showing {shown} only, no PPE compliance assessed")
    return " | ".join(parts) or None


def run(
    config: AppConfig,
    detector_factory: Callable[..., Any] = DFineDetector,
    source_opener: Callable[..., FrameSource] = open_source,
    writer_factory: Callable[..., Any] = OutputWriter,
    display_factory: Callable[[], Any] = CvDisplay,
) -> RunStats:
    """Load the model once, then process frames until the input ends or the user quits."""
    spec = parse_source(config.source, config.base_dir)
    output = None if config.output is None else config.output.expanduser().resolve()
    if output is not None:
        is_image_output = output.suffix.lower() in IMAGE_SUFFIXES
        if is_image_output != (spec.kind is SourceKind.IMAGE):
            raise ConfigError("Use an image --output (.jpg/.png) for image input and .mp4/.avi for video or camera.")

    detector = detector_factory(
        config.model_path, device=config.device, allow_unverified=config.allow_unverified_model, gpu_id=config.gpu_id
    )
    schema = PpeSchema.from_labels(detector.labels)
    if schema.ppe_supported and undetectable(schema, config.required_ppe):
        raise ConfigError(
            f"This model cannot detect {', '.join(undetectable(schema, config.required_ppe))}, so nobody could be "
            f"confirmed compliant. Set --required-ppe to items it detects: {', '.join(sorted(schema.categories))}."
        )
    class_filter = resolve_class_filter(config.classes, detector.labels, schema)
    warning = model_warning(detector, schema, class_filter)
    if warning:
        log.warning(warning)

    stats = RunStats()
    display = display_factory() if config.display else None
    writer = None
    started = time.perf_counter()
    fps_ema = 0.0
    try:
        with source_opener(
            spec, config.width, config.height, config.backend, fourcc=config.fourcc, fps=config.fps
        ) as source:
            log.info("Reading %s at %dx%d.", source.description, *source.frame_size)
            if output is not None:
                writer = writer_factory(output, source.fps)
            last = time.perf_counter()
            while config.max_frames is None or stats.frames < config.max_frames:
                frame = source.read()
                if frame is None:
                    break
                # Assess on every detection; --classes only limits what is drawn, so hiding a class
                # can never hide a violation from the person-level assessment.
                detections = detector.detect(frame, config.conf)
                assessment = assess(detections, schema, config.required_ppe, config.min_item_overlap)
                shown = [d for d in detections if class_filter is None or d.label in class_filter]
                now = time.perf_counter()
                instant = 1.0 / max(now - last, 1e-6)
                fps_ema = instant if stats.frames == 0 else 0.9 * fps_ema + 0.1 * instant
                last = now
                provider = detector.info.provider
                lines = status_lines(fps_ema, detector.last_inference_ms, provider, assessment, len(shown))
                draw_frame(frame, shown, schema, assessment, status_bar_height(frame, len(lines) + bool(warning)))
                draw_status_bar(frame, lines, warning)
                stats.frames += 1
                if writer is not None:
                    writer.write(frame)
                if display is not None and not display.show(frame, wait_for_key=spec.kind is SourceKind.IMAGE):
                    break
    finally:
        stats.seconds = time.perf_counter() - started
        try:
            if writer is not None:
                writer.release()
        finally:
            if display is not None:
                display.close()
    if output is not None and stats.frames:
        log.info("Saved %s", output)
    log.info("Processed %d frame(s) in %.1f s (%.1f FPS overall).", stats.frames, stats.seconds, stats.fps)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    try:
        if args.list_cameras:
            config = config_from_args(args)
            cameras = list_cameras(args.max_camera_index, config.backend)
            if not cameras:
                print(f"No camera delivered a frame on indices 0-{args.max_camera_index - 1}.")
                return EXIT_SOURCE
            for cam in cameras:
                fps = f", {cam.fps:.0f} fps reported" if cam.fps else ""
                print(f"camera {cam.index}: {cam.width}x{cam.height} via {cam.backend}{fps}")
            return EXIT_OK
        run(config_from_args(args))
        return EXIT_OK
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG
    except ModelError as exc:
        log.error("Model problem: %s", exc)
        return EXIT_MODEL
    except SourceError as exc:
        log.error("Input/output problem: %s", exc)
        return EXIT_SOURCE
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return EXIT_INTERRUPTED
    except Exception:  # noqa: BLE001 - last resort: report with traceback, distinct exit code
        log.exception("Unexpected error. Please report it with the traceback above.")
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
