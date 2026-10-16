"""Command-line parsing, exit codes and the main loop's resource cleanup (all mocked)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import PPE_LABELS, blank

import ppe_detection.__main__ as cli
from ppe_detection import PROJECT_ROOT
from ppe_detection.camera import CameraInfo, FrameSource, SourceError
from ppe_detection.compliance import PpeSchema
from ppe_detection.config import AppConfig, ConfigError, SourceKind, build_config
from ppe_detection.detector import Detection, ModelError, ModelInfo


def parse(*argv: str):
    return cli.config_from_args(cli.build_parser().parse_args(list(argv)))


# ------------------------------------------------------------------ parsing


def test_documented_examples_parse():
    assert parse("--source", "0").source == "0"
    assert parse("--source", "Samples/ppe1.mp4").source == "Samples/ppe1.mp4"
    config = parse("--source", "0", "--conf", "0.50")
    assert config.conf == 0.5 and config.display is True


def test_all_options_map_to_config():
    config = parse(
        "--source", "1", "--model", "models/dfine_s_coco.onnx", "--conf", "0.4", "--width", "640",
        "--height", "480", "--device", "cpu", "--gpu-id", "1", "--backend", "dshow",
        "--output", "out.mp4", "--no-display",
        "--classes", "Person, NO-Hardhat", "--required-ppe", "hardhat", "--max-frames", "5",
        "--allow-unverified-model",
    )
    assert config.model_path == PROJECT_ROOT / "models" / "dfine_s_coco.onnx"
    assert (config.width, config.height, config.device, config.backend) == (640, 480, "cpu", "dshow")
    assert config.gpu_id == 1
    assert config.display is False and config.output == Path("out.mp4")
    assert config.classes == ("Person", "NO-Hardhat") and config.required_ppe == ("hardhat",)
    assert config.max_frames == 5 and config.allow_unverified_model is True


def test_argparse_rejects_bad_choices():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--device", "gpu"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--conf", "high"])


def test_main_exit_codes_for_bad_settings_and_missing_files(tmp_path, caplog):
    assert cli.main(["--conf", "2"]) == cli.EXIT_CONFIG
    assert cli.main(["--source", str(tmp_path / "missing.mp4")]) == cli.EXIT_CONFIG
    assert "Input file not found" in caplog.text
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert cli.main(["--source", str(video), "--model", str(tmp_path / "none.onnx"), "--no-display"]) == cli.EXIT_MODEL
    assert "Model file not found" in caplog.text


def test_main_reports_camera_failure(monkeypatch):
    def failing_run(config):
        raise SourceError("Could not open camera 7")

    monkeypatch.setattr(cli, "run", failing_run)
    assert cli.main(["--source", "7"]) == cli.EXIT_SOURCE


def test_list_cameras(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_cameras", lambda n, backend: [CameraInfo(0, "dshow", 1280, 720, 30.0)])
    assert cli.main(["--list-cameras"]) == cli.EXIT_OK
    assert "camera 0: 1280x720 via dshow" in capsys.readouterr().out
    monkeypatch.setattr(cli, "list_cameras", lambda n, backend: [])
    assert cli.main(["--list-cameras"]) == cli.EXIT_SOURCE


def test_class_filter_resolution():
    ppe = PpeSchema.from_labels(PPE_LABELS)
    coco_labels = ("person", "bicycle", "car")
    coco = PpeSchema.from_labels(coco_labels)
    assert cli.resolve_class_filter(None, PPE_LABELS, ppe) is None
    assert cli.resolve_class_filter(None, coco_labels, coco) == {"person"}
    assert cli.resolve_class_filter(("all",), coco_labels, coco) is None
    assert cli.resolve_class_filter(("PERSON", "car"), coco_labels, coco) == {"person", "car"}
    with pytest.raises(ConfigError, match="Unknown class"):
        cli.resolve_class_filter(("helmet",), coco_labels, coco)


# ------------------------------------------------------------ run() and cleanup


class FakeDetector:
    instances = 0

    def __init__(self, path, device="cuda", allow_unverified=False, fail_on_frame=None, gpu_id=0):
        FakeDetector.instances += 1
        self.labels = PPE_LABELS
        self.info = ModelInfo(Path(path), "0" * 64, True, PPE_LABELS, (640, 640), {}, "CPUExecutionProvider")
        self.last_inference_ms = 1.0
        self.calls = 0
        self.fail_on_frame = fail_on_frame

    def detect(self, frame, conf, class_filter=None):
        self.calls += 1
        if self.calls == self.fail_on_frame:
            raise RuntimeError("inference exploded")
        return [Detection(5, "Person", 0.9, (10, 10, 100, 200)), Detection(2, "NO-Hardhat", 0.8, (30, 10, 80, 50))]


class FakeSource(FrameSource):
    def __init__(self, frames: int):
        self.kind, self.description, self.fps, self.frame_size = SourceKind.CAMERA, "fake camera", 30.0, (320, 240)
        self._left = frames
        self.released = 0

    def read(self):
        if self._left == 0:
            return None
        self._left -= 1
        return blank()

    def release(self):
        self.released += 1


class FakeDisplay:
    def __init__(self, quit_after: int | None = None):
        self.shown, self.closed, self.quit_after = 0, 0, quit_after

    def show(self, frame, wait_for_key=False):
        self.shown += 1
        return self.quit_after is None or self.shown < self.quit_after

    def close(self):
        self.closed += 1


class RecordingWriter:
    last = None

    def __init__(self, path, fps):
        self.frames, self.released = 0, 0
        RecordingWriter.last = self

    def write(self, frame: np.ndarray):
        self.frames += 1

    def release(self):
        self.released += 1


def run_with(config: AppConfig, source: FakeSource, display: FakeDisplay, detector_kwargs=None):
    return cli.run(
        config,
        detector_factory=lambda *a, **k: FakeDetector(*a, **k, **(detector_kwargs or {})),
        source_opener=lambda *a, **k: source,
        writer_factory=RecordingWriter,
        display_factory=lambda: display,
    )


def test_model_is_loaded_once_and_all_frames_processed(tmp_path):
    FakeDetector.instances = 0
    source, display = FakeSource(frames=5), FakeDisplay()
    stats = run_with(build_config(output=tmp_path / "o.mp4"), source, display)
    assert stats.frames == 5 and FakeDetector.instances == 1
    assert RecordingWriter.last.frames == 5 and RecordingWriter.last.released == 1
    assert source.released == 1 and display.closed == 1


def test_quit_key_stops_the_loop():
    source, display = FakeSource(frames=100), FakeDisplay(quit_after=3)
    assert run_with(build_config(), source, display).frames == 3
    assert source.released == 1 and display.closed == 1


def test_resources_are_released_when_inference_raises(tmp_path):
    source, display = FakeSource(frames=10), FakeDisplay()
    with pytest.raises(RuntimeError, match="exploded"):
        run_with(build_config(output=tmp_path / "o.mp4"), source, display, {"fail_on_frame": 3})
    assert source.released == 1 and display.closed == 1 and RecordingWriter.last.released == 1


def test_max_frames_and_image_output_rules(tmp_path):
    source = FakeSource(frames=50)
    assert run_with(build_config(max_frames=4, display=False), source, FakeDisplay()).frames == 4
    with pytest.raises(ConfigError, match="image --output"):
        run_with(build_config(output=tmp_path / "x.jpg"), FakeSource(1), FakeDisplay())


def test_model_errors_surface_before_any_source_is_opened():
    opened = []

    def bad_detector(*args, **kwargs):
        raise ModelError("broken")

    with pytest.raises(ModelError):
        cli.run(build_config(), detector_factory=bad_detector, source_opener=lambda *a, **k: opened.append(1))
    assert opened == []


# ------------------------------------------------------------- review follow-ups


def test_hidden_classes_still_count_for_compliance(monkeypatch):
    """--classes Person must not hide the NO-Hardhat that makes the person a violation."""
    drawn = []
    def record(frame, shown, schema, assessment, top):
        drawn.append((shown, assessment))

    monkeypatch.setattr(cli, "draw_frame", record)
    run_with(build_config(classes=("Person",), max_frames=1), FakeSource(1), FakeDisplay())
    shown, assessment = drawn[0]
    assert [d.label for d in shown] == ["Person"]
    assert assessment.persons[0].status.value == "violation"


def test_required_item_the_model_cannot_detect_is_rejected():
    class HardhatOnly(FakeDetector):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.labels = ("Person", "Hardhat", "NO-Hardhat")

    with pytest.raises(ConfigError, match="cannot detect vest"):
        cli.run(build_config(), detector_factory=HardhatOnly, source_opener=lambda *a, **k: FakeSource(1))


def test_ctrl_c_releases_resources_and_exits_130(monkeypatch, tmp_path):
    source, display = FakeSource(frames=10), FakeDisplay()

    class Interrupting(FakeDetector):
        def detect(self, frame, conf, class_filter=None):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli.run(build_config(output=tmp_path / "o.mp4"), detector_factory=Interrupting,
                source_opener=lambda *a, **k: source, writer_factory=RecordingWriter, display_factory=lambda: display)
    assert source.released == 1 and display.closed == 1 and RecordingWriter.last.released == 1

    def interrupted(config):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", interrupted)
    assert cli.main([]) == cli.EXIT_INTERRUPTED


def test_unexpected_errors_exit_1_with_a_message(monkeypatch, caplog):
    def broken(config):
        raise ZeroDivisionError("bug")

    monkeypatch.setattr(cli, "run", broken)
    assert cli.main([]) == cli.EXIT_UNEXPECTED
    assert "Unexpected error" in caplog.text


def test_fourcc_reaches_the_camera_opener():
    received = {}

    def opener(*args, **kwargs):
        received.update(kwargs)
        return FakeSource(1)

    cli.run(build_config(fourcc="MJPG", fps=25, display=False), detector_factory=FakeDetector, source_opener=opener)
    assert received == {"fourcc": "MJPG", "fps": 25}


# --------------------------------------------- the real OpenCV window, cv2 mocked


class FakeGui:
    """Replaces the cv2 GUI calls used by CvDisplay."""

    def __init__(self, keys, visible=1.0):
        self.keys, self.visible, self.events = list(keys), visible, []

    def install(self, monkeypatch):
        monkeypatch.setattr(cli.cv2, "namedWindow", lambda *a: self.events.append("open"))
        monkeypatch.setattr(cli.cv2, "resizeWindow", lambda *a: None)
        monkeypatch.setattr(cli.cv2, "imshow", lambda *a: self.events.append("show"))
        monkeypatch.setattr(cli.cv2, "waitKey", lambda delay: self.keys.pop(0) if self.keys else -1)
        monkeypatch.setattr(cli.cv2, "getWindowProperty", lambda *a: self.visible)
        monkeypatch.setattr(cli.cv2, "destroyWindow", lambda *a: self.events.append("destroy"))
        return self


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_quit_keys_stop_the_display(monkeypatch, key):
    gui = FakeGui([-1, key]).install(monkeypatch)
    display = cli.CvDisplay()
    assert display.show(blank()) is True  # no key pressed
    assert display.show(blank()) is False  # q / Q / Esc
    display.close()
    display.close()  # idempotent
    assert gui.events.count("open") == 1 and gui.events.count("destroy") == 1


def test_closing_the_window_stops_the_display(monkeypatch):
    FakeGui([-1], visible=0.0).install(monkeypatch)
    assert cli.CvDisplay().show(blank()) is False


def test_image_mode_waits_until_any_key(monkeypatch):
    gui = FakeGui([-1, -1, ord("x")]).install(monkeypatch)
    assert cli.CvDisplay().show(blank(), wait_for_key=True) is False
    assert gui.keys == []  # it polled until the key arrived


def test_missing_gui_support_is_a_clear_error(monkeypatch):
    def no_gui(*a):
        raise cli.cv2.error("The function is not implemented")

    monkeypatch.setattr(cli.cv2, "namedWindow", no_gui)
    with pytest.raises(SourceError, match="--no-display"):
        cli.CvDisplay().show(blank())
