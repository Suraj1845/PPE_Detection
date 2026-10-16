"""Frame sources (camera, video file, image file), camera discovery and output recording.

Every source and writer is a context manager, so the camera, files and
writers are released even when an exception is raised in the main loop.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import numpy as np

from .config import IMAGE_SUFFIXES, SourceKind, SourceSpec

log = logging.getLogger(__name__)

CaptureFactory = Callable[..., Any]  # cv2.VideoCapture-compatible constructor

_BACKEND_NAMES = {
    "any": "CAP_ANY",
    "dshow": "CAP_DSHOW",
    "msmf": "CAP_MSMF",
    "v4l2": "CAP_V4L2",
    "avfoundation": "CAP_AVFOUNDATION",
}
_FIRST_FRAME_ATTEMPTS = 10  # some cameras return a few empty frames right after opening
_MAX_CONSECUTIVE_CAMERA_FAILURES = 30


class SourceError(RuntimeError):
    """Raised when a camera, video or image cannot be opened or stops delivering frames."""


def backend_candidates(backend: str, platform: str = sys.platform) -> list[tuple[str, int]]:
    """Return ``(name, cv2 constant)`` pairs to try, in order, for a ``--backend`` value."""
    if backend == "auto":
        if platform.startswith("win"):
            # Media Foundation first: on the tested webcam it gave 30 FPS at 1280x720 where DirectShow
            # gave 5. The generic backend is left out on Windows: it only adds FFmpeg device probing
            # (noisy warnings) and real webcams are covered by these two.
            names = ["msmf", "dshow"]
        elif platform == "darwin":
            names = ["avfoundation", "any"]
        else:
            names = ["v4l2", "any"]
    else:
        names = [backend]
    candidates = []
    for name in names:
        constant = getattr(cv2, _BACKEND_NAMES[name], None)
        if constant is not None:
            candidates.append((name, int(constant)))
    if not candidates:
        raise SourceError(f"OpenCV in this environment has no {backend!r} capture backend.")
    return candidates


class FrameSource:
    """Common interface: ``read()`` returns a BGR frame, or ``None`` at the end of the input."""

    kind: SourceKind
    description: str = ""
    fps: float = 0.0
    frame_size: tuple[int, int] = (0, 0)  # (width, height)

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def release(self) -> None:
        """Release the underlying resource. Safe to call more than once."""

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class CaptureSource(FrameSource):
    """A camera or video file read through ``cv2.VideoCapture``."""

    def __init__(self, capture: Any, kind: SourceKind, description: str, first_frame: np.ndarray) -> None:
        self._capture = capture
        self._pending: np.ndarray | None = first_frame
        self._failures = 0
        self.kind = kind
        self.description = description
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 0.0
        self.frame_size = (int(first_frame.shape[1]), int(first_frame.shape[0]))

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        while True:
            ok, frame = self._capture.read()
            if ok and frame is not None and frame.size:
                self._failures = 0
                return frame
            if self.kind is not SourceKind.CAMERA:
                return None  # end of the video file
            self._failures += 1
            if self._failures >= _MAX_CONSECUTIVE_CAMERA_FAILURES:
                raise SourceError(
                    f"{self.description} stopped delivering frames ({self._failures} failed reads in a "
                    "row). Check the cable, or whether another application took the camera."
                )

    def release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            finally:
                self._capture = None


class ImageSource(FrameSource):
    """A single image file, returned once."""

    def __init__(self, path: Path) -> None:
        data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if frame is None:
            raise SourceError(f"Could not read image {path}: the file is empty, corrupt or not an image.")
        self._frame: np.ndarray | None = frame
        self.kind = SourceKind.IMAGE
        self.description = str(path)
        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))

    def read(self) -> np.ndarray | None:
        frame, self._frame = self._frame, None
        return frame

    def release(self) -> None:
        self._frame = None


def _read_first_frame(capture: Any) -> np.ndarray | None:
    for _ in range(_FIRST_FRAME_ATTEMPTS):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            return frame
    return None


def _safe_release(capture: Any) -> None:
    try:
        capture.release()
    except Exception:  # noqa: BLE001 - best effort while reporting the original problem
        log.debug("Ignoring error while releasing a capture", exc_info=True)


def open_camera(
    index: int,
    width: int,
    height: int,
    backend: str = "auto",
    capture_factory: CaptureFactory = cv2.VideoCapture,
    fourcc: str | None = None,
    fps: int | None = 30,
) -> CaptureSource:
    """Open camera ``index``, request ``width`` x ``height`` at ``fps`` and check that frames arrive.

    ``fourcc`` (e.g. ``"MJPG"``) optionally requests a compressed pixel format, which lets
    many USB webcams deliver higher resolutions at full frame rate through DirectShow.
    """
    tried = []
    for name, constant in backend_candidates(backend):
        capture = capture_factory(index, constant)
        if not capture.isOpened():
            _safe_release(capture)
            tried.append(f"{name}: could not open")
            continue
        if fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))  # must precede the size
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            capture.set(cv2.CAP_PROP_FPS, fps)  # some webcams stay at a slow default unless asked
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prefer the newest frame over a backlog
        frame = _read_first_frame(capture)
        if frame is None:
            _safe_release(capture)
            tried.append(f"{name}: opened but returned no frames")
            continue
        source = CaptureSource(capture, SourceKind.CAMERA, f"camera {index} ({name})", frame)
        actual_w, actual_h = source.frame_size
        if (actual_w, actual_h) != (width, height):
            log.warning("Camera %d delivers %dx%d (requested %dx%d).", index, actual_w, actual_h, width, height)
        return source
    raise SourceError(
        f"Could not open camera {index} ({'; '.join(tried)}). Check that the camera is connected, "
        "not in use by another application (Teams, Zoom, the Camera app), and allowed in "
        "Windows Settings > Privacy & security > Camera. Run with --list-cameras to see available indices."
    )


def open_video(path: Path, capture_factory: CaptureFactory = cv2.VideoCapture) -> CaptureSource:
    """Open a video file and check that its first frame decodes."""
    capture = capture_factory(str(path))
    if not capture.isOpened():
        _safe_release(capture)
        raise SourceError(
            f"Could not open video {path}: unsupported format, missing codec or corrupt file. On Windows, "
            "a folder or file name with non-English characters can also cause this; try a plain path."
        )
    frame = _read_first_frame(capture)
    if frame is None:
        _safe_release(capture)
        raise SourceError(f"Video {path} opened but no frame could be decoded.")
    return CaptureSource(capture, SourceKind.VIDEO, str(path), frame)


def open_source(
    spec: SourceSpec,
    width: int,
    height: int,
    backend: str = "auto",
    capture_factory: CaptureFactory = cv2.VideoCapture,
    fourcc: str | None = None,
    fps: int | None = 30,
) -> FrameSource:
    """Open whichever kind of source ``spec`` describes."""
    if spec.kind is SourceKind.CAMERA:
        assert spec.camera_index is not None
        return open_camera(spec.camera_index, width, height, backend, capture_factory, fourcc, fps)
    assert spec.path is not None
    if spec.kind is SourceKind.IMAGE:
        return ImageSource(spec.path)
    return open_video(spec.path, capture_factory)


@dataclass(frozen=True)
class CameraInfo:
    index: int
    backend: str
    width: int
    height: int
    fps: float


def list_cameras(
    max_index: int = 6,
    backend: str = "auto",
    capture_factory: CaptureFactory = cv2.VideoCapture,
) -> list[CameraInfo]:
    """Probe camera indices ``0..max_index-1`` and return the ones that deliver a frame.

    Each index is tried with the same backends, in the same order, as :func:`open_camera`.
    OpenCV's own logging is silenced while probing, because missing indices are expected
    and would otherwise print one internal error line each.
    """
    if max_index < 1:
        raise SourceError(f"--max-camera-index must be at least 1, got {max_index}.")
    candidates = backend_candidates(backend)
    cv_logging = cv2.utils.logging
    previous_level = cv_logging.getLogLevel()
    cv_logging.setLogLevel(cv_logging.LOG_LEVEL_SILENT)
    found = []
    try:
        for index in range(max_index):
            for name, constant in candidates:
                capture = capture_factory(index, constant)
                try:
                    frame = _read_first_frame(capture) if capture.isOpened() else None
                    if frame is not None:
                        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                        found.append(CameraInfo(index, name, int(frame.shape[1]), int(frame.shape[0]), fps))
                        break
                finally:
                    _safe_release(capture)
    finally:
        cv_logging.setLogLevel(previous_level)
    return found


class OutputWriter:
    """Writes annotated frames to a video (.mp4/.avi) or a single image file.

    The video writer is created on the first frame, when the frame size is known.
    """

    def __init__(self, path: Path, fps: float, writer_factory: Callable[..., Any] = cv2.VideoWriter) -> None:
        self.path = path
        self.fps = fps if fps > 0 else 30.0
        self._writer_factory = writer_factory
        self._writer: Any = None
        self._is_image = path.suffix.lower() in IMAGE_SUFFIXES
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SourceError(f"Cannot create the output folder {path.parent}: {exc}") from exc

    def write(self, frame: np.ndarray) -> None:
        if self._is_image:
            ok, encoded = cv2.imencode(self.path.suffix, frame)
            if not ok:
                raise SourceError(f"Could not encode the output image {self.path}.")
            try:
                encoded.tofile(str(self.path))
            except OSError as exc:
                raise SourceError(f"Cannot write the output image {self.path}: {exc}") from exc
            return
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if self.path.suffix.lower() == ".mp4" else "MJPG"))
            height, width = frame.shape[:2]
            self._writer = self._writer_factory(str(self.path), fourcc, self.fps, (width, height))
            if not self._writer.isOpened():
                self._writer = None
                raise SourceError(f"Could not create output video {self.path}. Try an .avi file name.")
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            finally:
                self._writer = None

    def __enter__(self) -> OutputWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
