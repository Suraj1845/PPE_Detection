"""Camera, video and image sources with mocked cv2.VideoCapture (no webcam needed)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import FakeCapture, blank

from ppe_detection.camera import (
    ImageSource,
    OutputWriter,
    SourceError,
    backend_candidates,
    list_cameras,
    open_camera,
    open_video,
)
from ppe_detection.config import SourceKind


def factory(*captures: FakeCapture):
    """A VideoCapture stand-in returning the given captures in order and recording the calls."""
    queue = list(captures)
    calls = []

    def make(*args):
        calls.append(args)
        return queue.pop(0)

    make.calls = calls
    return make


def test_camera_that_cannot_open_gives_helpful_error_and_is_released():
    caps = [FakeCapture(opened=False), FakeCapture(opened=False), FakeCapture(opened=False)]
    with pytest.raises(SourceError) as info:
        open_camera(3, 1280, 720, backend="auto", capture_factory=factory(*caps))
    message = str(info.value)
    assert "camera 3" in message and "--list-cameras" in message and "Privacy" in message
    assert all(c.released == 1 for c in caps[: len(backend_candidates("auto"))])


def test_camera_that_opens_but_gives_no_frames_is_rejected():
    cap = FakeCapture(opened=True, frames=[])
    with pytest.raises(SourceError, match="returned no frames"):
        open_camera(0, 640, 480, backend="any", capture_factory=factory(cap))
    assert cap.released == 1


def test_camera_resolution_is_requested_and_first_frame_is_kept():
    first = blank(640, 480)
    cap = FakeCapture(frames=[first, blank(640, 480)])
    make = factory(cap)
    source = open_camera(1, 1280, 720, backend="any", capture_factory=make)
    assert make.calls[0] == (1, cv2.CAP_ANY)
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280 and cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert source.kind is SourceKind.CAMERA and source.frame_size == (640, 480)
    assert source.read() is first


def test_camera_that_stops_delivering_frames_raises():
    cap = FakeCapture(frames=[blank()])
    source = open_camera(0, 320, 240, backend="any", capture_factory=factory(cap))
    source.read()  # the buffered first frame
    with pytest.raises(SourceError, match="stopped delivering frames"):
        source.read()


def test_video_ends_cleanly_and_release_is_idempotent():
    cap = FakeCapture(frames=[blank(), blank()])
    with open_video("clip.mp4", capture_factory=factory(cap)) as source:
        assert source.kind is SourceKind.VIDEO and source.fps == 25.0
        assert source.read() is not None and source.read() is not None
        assert source.read() is None
    source.release()
    assert cap.released == 1


def test_source_is_released_when_the_loop_raises():
    cap = FakeCapture(frames=[blank()])
    with pytest.raises(RuntimeError, match="boom"):
        with open_video("clip.mp4", capture_factory=factory(cap)):
            raise RuntimeError("boom")
    assert cap.released == 1


def test_unreadable_video_errors():
    with pytest.raises(SourceError, match="Could not open video"):
        open_video("x.mp4", capture_factory=factory(FakeCapture(opened=False)))
    with pytest.raises(SourceError, match="no frame could be decoded"):
        open_video("x.mp4", capture_factory=factory(FakeCapture(frames=[])))


def test_image_source(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), blank(40, 30))
    with ImageSource(path) as source:
        assert source.frame_size == (40, 30)
        assert source.read() is not None and source.read() is None
    corrupt = tmp_path / "broken.jpg"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(SourceError, match="corrupt or not an image"):
        ImageSource(corrupt)


def test_list_cameras_reports_working_indices_and_releases_all():
    caps = [FakeCapture(frames=[blank(640, 480)]), FakeCapture(opened=False), FakeCapture(frames=[blank(1280, 720)])]
    found = list_cameras(3, backend="any", capture_factory=factory(*caps))
    assert [(c.index, c.width, c.height) for c in found] == [(0, 640, 480), (2, 1280, 720)]
    assert all(c.released == 1 for c in caps)


def test_backend_order_on_windows():
    # Media Foundation first (30 FPS at 1280x720 on the tested webcam vs 5 with DirectShow);
    # no generic backend, which only adds FFmpeg probing noise on Windows.
    names = [name for name, _ in backend_candidates("auto", platform="win32")]
    assert names == ["msmf", "dshow"]


class FakeWriter:
    def __init__(self, path, fourcc, fps, size, opened=True):
        self.args, self.opened, self.frames, self.released = (path, fourcc, fps, size), opened, 0, 0

    def isOpened(self):  # noqa: N802
        return self.opened

    def write(self, frame):
        self.frames += 1

    def release(self):
        self.released += 1


def test_output_writer_creates_video_lazily_and_releases(tmp_path):
    made = []

    def make(*args):
        made.append(FakeWriter(*args))
        return made[-1]

    with OutputWriter(tmp_path / "out" / "result.mp4", fps=0, writer_factory=make) as writer:
        writer.write(blank(320, 240))
        writer.write(blank(320, 240))
    assert made[0].args[2] == 30.0 and made[0].args[3] == (320, 240)  # fps fallback, (w, h)
    assert made[0].frames == 2 and made[0].released == 1


def test_output_writer_failure_is_reported(tmp_path):
    writer = OutputWriter(tmp_path / "x.mp4", 25, writer_factory=lambda *a: FakeWriter(*a, opened=False))
    with pytest.raises(SourceError, match="Could not create output video"):
        writer.write(blank())


def test_output_image(tmp_path):
    path = tmp_path / "annotated.jpg"
    OutputWriter(path, fps=0).write(np.full((20, 30, 3), 128, np.uint8))
    assert cv2.imread(str(path)).shape == (20, 30, 3)


def test_fourcc_is_requested_before_the_resolution():
    order = []

    class Recording(FakeCapture):
        def set(self, prop, value):
            order.append(prop)
            return super().set(prop, value)

    cap = Recording(frames=[blank()])
    open_camera(0, 1280, 720, backend="any", capture_factory=factory(cap), fourcc="MJPG", fps=25)
    assert order[:2] == [cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH]
    assert cap.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert cap.props[cv2.CAP_PROP_FPS] == 25  # frame rate is requested too


def test_list_cameras_falls_back_to_the_next_backend_like_open_camera():
    # Windows order: msmf fails for index 0, dshow works; index 1 absent on both.
    caps = [FakeCapture(opened=False), FakeCapture(frames=[blank(640, 480)]),
            FakeCapture(opened=False), FakeCapture(opened=False)]
    import ppe_detection.camera as camera_module

    original = camera_module.backend_candidates
    try:
        camera_module.backend_candidates = lambda backend: original(backend, platform="win32")
        found = list_cameras(2, backend="auto", capture_factory=factory(*caps))
    finally:
        camera_module.backend_candidates = original
    assert [(c.index, c.backend) for c in found] == [(0, "dshow")]
    assert all(c.released == 1 for c in caps)


def test_list_cameras_rejects_non_positive_range():
    with pytest.raises(SourceError, match="at least 1"):
        list_cameras(0)


def test_output_folder_that_cannot_be_created_is_a_source_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(SourceError, match="Cannot create the output folder"):
        OutputWriter(blocker / "sub" / "out.mp4", fps=25)
