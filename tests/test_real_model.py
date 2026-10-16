"""REAL model inference tests (marker ``model``). No webcam is needed.

They run the ONNX files in models/ on frames from Samples/ and are skipped if
either is missing. Run only these with ``pytest -m model``; skip them with
``pytest -m "not model"``.

The CPU tests run everywhere (onnxruntime-gpu also contains the CPU provider).
The GPU tests (marker ``gpu``) need a working CUDA GPU and are skipped otherwise;
they check that the model really runs on CUDA and gives the same detections as the CPU.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from conftest import SAMPLES

import ppe_detection.__main__ as cli
from ppe_detection import PROJECT_ROOT
from ppe_detection.detector import DFineDetector, ModelError, read_checksums, sha256_file

pytestmark = pytest.mark.model

MODELS = PROJECT_ROOT / "models"
MODEL_FILES = ["dfine_n_coco.onnx", "dfine_s_coco.onnx"]
# COCO names in D-FINE's contiguous label order (src/data/dataset/coco_dataset.py at 956d170).
COCO_SPOT_CHECK = {0: "person", 1: "bicycle", 2: "car", 11: "stop sign", 79: "toothbrush"}


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.relative_to(PROJECT_ROOT)} not present")
    return path


def first_frame(video: Path, at_seconds: float = 4.0):
    capture = cv2.VideoCapture(str(require(video)))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok, f"could not decode {video}"
    return frame


@pytest.fixture(scope="module", params=MODEL_FILES)
def detector(request) -> DFineDetector:
    return DFineDetector(require(MODELS / request.param), device="cpu")


def test_shipped_models_match_recorded_checksums():
    approved = read_checksums()
    for name in MODEL_FILES:
        assert sha256_file(require(MODELS / name)) == approved[name]


def test_real_metadata_class_mapping(detector):
    assert len(detector.labels) == 80
    for index, name in COCO_SPOT_CHECK.items():
        assert detector.labels[index] == name
    assert detector.info.metadata["ppe_detection.license"] == "Apache-2.0"
    assert detector.info.provider == "CPUExecutionProvider"


@pytest.mark.parametrize("video", ["ppe1.mp4", "ppe2.mp4", "ppe3.mp4"])
def test_real_inference_finds_people_in_sample_videos(detector, video):
    frame = first_frame(SAMPLES / video)
    people = detector.detect(frame, conf=0.5, class_filter=frozenset({"person"}))
    assert people, "expected at least one person at confidence >= 0.5"
    height, width = frame.shape[:2]
    for person in people:
        x1, y1, x2, y2 = person.box
        assert 0 <= x1 < x2 < width and 0 <= y1 < y2 < height
        assert person.area > 0.01 * width * height  # a real person, not a speck


def test_real_boxes_land_on_the_person(detector):
    """Reference from a visually inspected frame: in ppe1.mp4 at 4 s one worker stands just
    right of centre and fills most of the frame height. Catches scaling/axis-order bugs."""
    frame = first_frame(SAMPLES / "ppe1.mp4")
    height, width = frame.shape[:2]
    best = max(detector.detect(frame, 0.5, frozenset({"person"})), key=lambda d: d.score)
    x1, y1, x2, y2 = best.box
    assert 0.40 * width < (x1 + x2) / 2 < 0.75 * width
    assert (y2 - y1) > 0.60 * height


def test_real_inference_is_deterministic(detector):
    frame = first_frame(SAMPLES / "ppe2.mp4")
    assert detector.detect(frame, 0.3) == detector.detect(frame.copy(), 0.3)


def test_cli_end_to_end_on_sample_video(tmp_path):
    require(SAMPLES / "ppe1.mp4")
    out = tmp_path / "annotated.mp4"
    code = cli.main(["--source", str(SAMPLES / "ppe1.mp4"), "--device", "cpu", "--no-display", "--max-frames", "8",
                     "--output", str(out)])
    assert code == cli.EXIT_OK
    capture = cv2.VideoCapture(str(out))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    finally:
        capture.release()


def test_cli_end_to_end_on_image(tmp_path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), first_frame(SAMPLES / "ppe3.mp4"))
    out = tmp_path / "annotated.png"
    assert cli.main(["--source", str(image), "--device", "cpu", "--no-display", "--output", str(out)]) == cli.EXIT_OK
    assert cv2.imread(str(out)).shape == cv2.imread(str(image)).shape


# ----------------------------------------------------------------------- GPU (CUDA)


@pytest.fixture(scope="module", params=MODEL_FILES)
def gpu_and_cpu(request) -> tuple[DFineDetector, DFineDetector]:
    path = require(MODELS / request.param)
    try:
        gpu = DFineDetector(path, device="cuda")
    except ModelError as exc:
        pytest.skip(f"no working CUDA GPU here: {exc}")
    return gpu, DFineDetector(path, device="cpu")


@pytest.mark.gpu
def test_gpu_session_really_uses_cuda(gpu_and_cpu):
    gpu, _ = gpu_and_cpu
    assert gpu.info.provider == "CUDAExecutionProvider"


@pytest.mark.gpu
@pytest.mark.parametrize("video", ["ppe1.mp4", "ppe2.mp4", "ppe3.mp4"])
def test_gpu_and_cpu_find_the_same_people(gpu_and_cpu, video):
    gpu, cpu = gpu_and_cpu
    frame = first_frame(SAMPLES / video)
    people = frozenset({"person"})
    on_gpu, on_cpu = gpu.detect(frame, 0.5, people), cpu.detect(frame, 0.5, people)
    assert len(on_gpu) == len(on_cpu) > 0
    for a, b in zip(sorted(on_gpu, key=lambda d: d.box), sorted(on_cpu, key=lambda d: d.box), strict=True):
        assert max(abs(p - q) for p, q in zip(a.box, b.box, strict=True)) <= 3  # pixels
        assert abs(a.score - b.score) < 0.02


@pytest.mark.gpu
def test_gpu_runs_at_least_95_percent_of_the_model(gpu_and_cpu):
    """Profiles real frames: at least 95 % of kernel time must be on CUDA (scripts/check_gpu.py)."""
    import importlib.util

    gpu, _ = gpu_and_cpu
    spec = importlib.util.spec_from_file_location("check_gpu", PROJECT_ROOT / "scripts" / "check_gpu.py")
    check_gpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_gpu)
    assert check_gpu.main(["--model", str(gpu.info.path), "--frames", "20"]) == 0


@pytest.mark.gpu
def test_cli_end_to_end_on_gpu(gpu_and_cpu, tmp_path):
    out = tmp_path / "gpu.mp4"
    code = cli.main(["--source", str(SAMPLES / "ppe2.mp4"), "--no-display", "--max-frames", "8", "--output", str(out)])
    assert code == cli.EXIT_OK and out.stat().st_size > 0
