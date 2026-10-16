"""Detector unit tests with a mocked ONNX Runtime session (no real model needed)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from conftest import PPE_LABELS, FakeSession, blank, dfine_metadata

from ppe_detection.detector import (
    PROVIDERS,
    DFineDetector,
    ModelError,
    check_approval,
    choose_providers,
    decode_outputs,
    parse_metadata,
    read_checksums,
)


def outputs(rows: list[tuple[int, float, tuple[float, float, float, float]]], k: int = 300):
    """Build D-FINE-shaped outputs: labels [1,K], boxes [1,K,4], scores [1,K]."""
    labels = np.zeros((1, k), dtype=np.int64)
    boxes = np.zeros((1, k, 4), dtype=np.float32)
    scores = np.zeros((1, k), dtype=np.float32)
    for i, (label, score, box) in enumerate(rows):
        labels[0, i], scores[0, i], boxes[0, i] = label, score, box
    return labels, boxes, scores


# ------------------------------------------------------------- class mapping


def test_labels_come_from_metadata_in_index_order():
    labels, size = parse_metadata(dfine_metadata())
    assert labels == PPE_LABELS
    assert labels.index("NO-Safety Vest") == 4 and labels.index("Person") == 5
    assert size == (640, 640)


def test_decoded_class_names_follow_metadata_indices():
    labels, boxes, scores = outputs([(2, 0.9, (10, 10, 50, 50)), (7, 0.8, (60, 60, 120, 140))])
    dets = decode_outputs(labels[0], boxes[0], scores[0], 0.5, (640, 480), PPE_LABELS)
    assert [(d.class_id, d.label) for d in dets] == [(2, "NO-Hardhat"), (7, "Safety Vest")]


def test_out_of_range_class_index_is_an_error_not_a_guess():
    labels, boxes, scores = outputs([(10, 0.9, (10, 10, 50, 50))])
    with pytest.raises(ModelError, match="class index 10"):
        decode_outputs(labels[0], boxes[0], scores[0], 0.5, (640, 480), PPE_LABELS)


@pytest.mark.parametrize(
    ("meta", "message"),
    [
        ({k: v for k, v in dfine_metadata().items() if not k.endswith("labels")}, "missing"),
        (dfine_metadata(labels=()), "non-empty"),
        (dfine_metadata(labels=("A", "A")), "duplicate"),
        (dfine_metadata(model_family="yolo"), "family"),
        (dfine_metadata(preprocessing="cpu: letterbox"), "preprocessing"),
        (dfine_metadata(format_version="1"), "Re-export"),
        (dfine_metadata(input_width="abc"), "malformed"),
    ],
)
def test_bad_metadata_is_rejected(meta, message):
    with pytest.raises(ModelError, match=message):
        parse_metadata(meta)


# ------------------------------------------------- output parsing / filtering


def test_confidence_threshold_filters_and_sorts():
    labels, boxes, scores = outputs(
        [(0, 0.49, (0, 0, 10, 10)), (1, 0.50, (0, 0, 10, 10)), (5, 0.95, (0, 0, 10, 10)), (6, 0.7, (0, 0, 10, 10))]
    )
    dets = decode_outputs(labels[0], boxes[0], scores[0], 0.5, (100, 100), PPE_LABELS)
    assert [round(d.score, 2) for d in dets] == [0.95, 0.7, 0.5]  # 0.49 dropped, >= threshold kept


def test_class_filter_keeps_only_named_classes():
    labels, boxes, scores = outputs([(5, 0.9, (0, 0, 10, 10)), (8, 0.9, (0, 0, 10, 10))])
    dets = decode_outputs(labels[0], boxes[0], scores[0], 0.5, (100, 100), PPE_LABELS, frozenset({"Person"}))
    assert [d.label for d in dets] == ["Person"]


def test_unexpected_output_shapes_are_rejected():
    with pytest.raises(ModelError, match="shapes"):
        decode_outputs(np.zeros(5, np.int64), np.zeros((5, 3)), np.zeros(5), 0.5, (10, 10), PPE_LABELS)


# --------------------------------------------- bounding-box coordinate handling


def test_boxes_are_rounded_and_clipped_to_the_frame():
    labels, boxes, scores = outputs([(5, 0.9, (-12.4, 10.6, 700.2, 485.0))])
    (det,) = decode_outputs(labels[0], boxes[0], scores[0], 0.5, (640, 480), PPE_LABELS)
    assert det.box == (0, 11, 639, 479)


def test_degenerate_boxes_are_dropped():
    labels, boxes, scores = outputs([(5, 0.9, (50, 50, 50, 80)), (5, 0.9, (700, 10, 800, 50))])
    assert decode_outputs(labels[0], boxes[0], scores[0], 0.5, (640, 480), PPE_LABELS) == []


def test_detect_hands_the_raw_uint8_frame_to_the_model(approved_model):
    """No CPU preprocessing: the exact OpenCV frame goes to ONNX Runtime; the model does the rest on the GPU."""
    model, sums = approved_model
    session = FakeSession(outputs=outputs([(5, 0.9, (10, 20, 300, 230))]), providers=["CUDAExecutionProvider"])
    det = DFineDetector(model, checksums_file=sums, session_factory=lambda p, prov: session,
                        available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    frame = blank(320, 240)
    (result,) = det.detect(frame, 0.5)
    fed = session.calls[-1]
    assert list(fed) == ["frame"]
    assert fed["frame"] is frame  # same array: no resize, colour conversion or float copy on the CPU
    assert result.label == "Person" and result.box == (10, 20, 300, 230)


def test_non_uint8_or_non_bgr_frames_are_rejected(approved_model):
    model, sums = approved_model
    det = DFineDetector(model, device="cpu", checksums_file=sums, session_factory=lambda p, prov: FakeSession(),
                        available_providers=lambda: ["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="uint8"):
        det.detect(blank().astype(np.float32), 0.5)
    with pytest.raises(ValueError, match="uint8"):
        det.detect(blank()[:, :, 0], 0.5)


def test_gpu_index_is_passed_to_the_gpu_provider_only(approved_model):
    model, sums = approved_model
    seen = []

    def factory(path, providers):
        seen.extend(providers)
        return FakeSession(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    DFineDetector(model, checksums_file=sums, session_factory=factory, gpu_id=1,
                  available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert seen == [("CUDAExecutionProvider", {"device_id": "1"}), ("CPUExecutionProvider", {})]


# --------------------------------------------- invalid / missing model files


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelError, match="not found"):
        DFineDetector(tmp_path / "nope.onnx")


def test_pytorch_checkpoint_is_refused(tmp_path):
    pt = tmp_path / "ppe.pt"
    pt.write_bytes(b"x")
    with pytest.raises(ModelError, match="PyTorch checkpoint"):
        DFineDetector(pt)


def test_empty_model_file(tmp_path):
    empty = tmp_path / "empty.onnx"
    empty.touch()
    with pytest.raises(ModelError, match="empty"):
        DFineDetector(empty)


def test_corrupt_onnx_file_gives_model_error(tmp_path):
    """Uses the real ONNX Runtime parser on a garbage file (no model download)."""
    bad = tmp_path / "corrupt.onnx"
    bad.write_bytes(b"\x00\x01garbage" * 100)
    with pytest.raises(ModelError, match="corrupt or not a valid ONNX model"):
        DFineDetector(bad, device="cpu", allow_unverified=True, checksums_file=tmp_path / "none")


def test_unlisted_model_needs_explicit_permission(tmp_path, approved_model):
    model, _ = approved_model
    with pytest.raises(ModelError, match="not an approved model"):
        DFineDetector(model, checksums_file=tmp_path / "missing_sums")


def test_modified_model_is_detected(approved_model):
    model, sums = approved_model
    sums.write_text(f"{'0' * 64}  {model.name}\n", encoding="utf-8")
    with pytest.raises(ModelError, match="corrupt or has been modified"):
        DFineDetector(model, checksums_file=sums)


def test_unverified_model_runs_only_with_flag_and_is_marked(tmp_path):
    model = tmp_path / "x.onnx"
    model.write_bytes(b"abc")
    det = DFineDetector(
        model, device="cpu", allow_unverified=True, checksums_file=tmp_path / "none",
        session_factory=lambda p, prov: FakeSession(), available_providers=lambda: ["CPUExecutionProvider"],
    )
    assert det.info.approved is False


def test_incompatible_io_names(approved_model):
    model, sums = approved_model
    for session in (FakeSession(output_names=("output0",)), FakeSession(input_names=("images", "orig_target_sizes")),
                    FakeSession(input_type="tensor(float)")):
        with pytest.raises(ModelError, match="Incompatible model|expected uint8"):
            DFineDetector(model, device="cpu", checksums_file=sums, session_factory=lambda p, prov, s=session: s,
                          available_providers=lambda: ["CPUExecutionProvider"])


def test_read_checksums_rejects_malformed_lines(tmp_path):
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("# comment\n\nnot-a-hash file.onnx\n", encoding="utf-8")
    with pytest.raises(ModelError, match="line 3"):
        read_checksums(sums)


def test_check_approval_accepts_listed_hash_under_any_name(tmp_path):
    digest = hashlib.sha256(b"m").hexdigest()
    assert check_approval(Path("renamed.onnx"), digest, {"orig.onnx": digest}, allow_unverified=False)


# ------------------------------------------------------------- providers


def test_gpu_is_the_default_and_never_silently_replaced_by_the_cpu(approved_model):
    model, sums = approved_model
    cpu_only = lambda: ["CPUExecutionProvider"]  # noqa: E731 - e.g. requirements-cpu.txt installed
    with pytest.raises(ModelError, match="--device cuda needs CUDAExecutionProvider"):
        DFineDetector(model, checksums_file=sums, session_factory=lambda p, prov: FakeSession(),
                      available_providers=cpu_only)


def test_auto_prefers_gpu_providers_that_exist():
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert choose_providers("auto", available) == [PROVIDERS["cuda"], PROVIDERS["cpu"]]
    assert choose_providers("auto", ["CPUExecutionProvider"]) == [PROVIDERS["cpu"]]


def test_explicit_gpu_request_without_provider_fails_clearly():
    with pytest.raises(ModelError, match="offers only CPUExecutionProvider"):
        choose_providers("cuda", ["CPUExecutionProvider"])


def test_actual_provider_is_reported_and_silent_cpu_fallback_is_caught(approved_model):
    model, sums = approved_model
    gpu_failed = FakeSession(providers=["CPUExecutionProvider"])  # ORT fell back to CPU
    det = DFineDetector(model, device="auto", checksums_file=sums, session_factory=lambda p, prov: gpu_failed,
                        available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    assert det.info.provider == "CPUExecutionProvider"
    with pytest.raises(ModelError, match="failed to initialise"):
        DFineDetector(model, device="directml", checksums_file=sums, session_factory=lambda p, prov: gpu_failed,
                      available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])


def test_checksum_file_with_byte_order_mark(tmp_path):
    sums = tmp_path / "SHA256SUMS"
    sums.write_bytes(b"\xef\xbb\xbf" + f"{'a' * 64}  model.onnx\n".encode())
    assert read_checksums(sums) == {"model.onnx": "a" * 64}


def test_blocked_onnxruntime_import_becomes_a_model_error(monkeypatch):
    import sys

    from ppe_detection import detector as detector_module

    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # makes `import onnxruntime` raise ImportError
    with pytest.raises(ModelError, match="WinError 4551"):
        detector_module._available_providers()
