"""Shared test doubles. Nothing here needs a webcam or a model download."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ppe_detection import PROJECT_ROOT

#: The original project's 10-class order (verified in ppe.pt's metadata during the audit).
PPE_LABELS = (
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
)
SAMPLES = PROJECT_ROOT / "Samples"


def dfine_metadata(labels: tuple[str, ...] = PPE_LABELS, **overrides: str) -> dict[str, str]:
    meta = {
        "ppe_detection.format_version": "2",
        "ppe_detection.model_family": "dfine",
        "ppe_detection.labels": json.dumps(list(labels)),
        "ppe_detection.input_width": "640",
        "ppe_detection.input_height": "640",
        "ppe_detection.input": "frame: uint8 [height, width, 3], BGR (OpenCV order)",
        "ppe_detection.preprocessing": "in-graph: BGR->RGB, bilinear stretch to 640x640 (half_pixel), x/255",
    }
    meta.update({f"ppe_detection.{k}": v for k, v in overrides.items()})
    return meta


@dataclass
class _Port:
    name: str
    shape: list[Any] = field(default_factory=list)
    type: str = "tensor(float)"


class FakeSession:
    """Mimics the parts of onnxruntime.InferenceSession the detector uses."""

    def __init__(
        self,
        meta: dict[str, str] | None = None,
        outputs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        providers: list[str] | None = None,
        input_names: tuple[str, ...] = ("frame",),
        output_names: tuple[str, ...] = ("labels", "boxes", "scores"),
        input_type: str = "tensor(uint8)",
    ) -> None:
        self.meta = dfine_metadata() if meta is None else meta
        self.outputs = outputs
        self.providers = providers or ["CPUExecutionProvider"]
        self.input_names, self.output_names, self.input_type = input_names, output_names, input_type
        self.calls: list[dict[str, np.ndarray]] = []
        self.providers_requested: list[Any] = []

    def get_inputs(self) -> list[_Port]:
        return [_Port(name, ["height", "width", 3], self.input_type) for name in self.input_names]

    def get_outputs(self) -> list[_Port]:
        return [_Port(name) for name in self.output_names]

    def get_providers(self) -> list[str]:
        return self.providers

    def get_modelmeta(self) -> Any:
        return type("Meta", (), {"custom_metadata_map": self.meta})()

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(feed)
        assert self.outputs is not None, "FakeSession.outputs not set"
        return list(self.outputs)


@pytest.fixture
def approved_model(tmp_path: Path) -> tuple[Path, Path]:
    """A dummy .onnx file plus a SHA256SUMS that approves it (content is never parsed by FakeSession)."""
    import hashlib

    model = tmp_path / "fake.onnx"
    model.write_bytes(b"not really onnx, the session is faked")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{hashlib.sha256(model.read_bytes()).hexdigest()}  fake.onnx\n", encoding="utf-8")
    return model, sums


class FakeCapture:
    """Mimics cv2.VideoCapture: ``frames`` are returned in order, then reads fail."""

    def __init__(self, opened: bool = True, frames: list[np.ndarray] | None = None, fps: float = 25.0) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.fps = fps
        self.released = 0
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV naming
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        import cv2

        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self.released += 1


def blank(width: int = 320, height: int = 240) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)
