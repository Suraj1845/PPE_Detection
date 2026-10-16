"""D-FINE ONNX model loading, GPU inference and output decoding with ONNX Runtime.

The decoder is written for the exact interface produced by ``scripts/export_dfine_onnx.py``
(model format version 2):

* input   ``frame`` uint8 [height, width, 3]: the camera frame exactly as OpenCV delivers it (BGR).
  The model itself converts BGR->RGB, resizes to 640x640, scales to 0-1 and computes the frame
  size, so all preprocessing runs on the GPU together with the network.
* outputs ``labels`` int64 [1, 300], ``boxes`` float32 [1, 300, 4] (x1, y1, x2, y2 in frame
  pixels) and ``scores`` float32 [1, 300] (sigmoid confidences), with the top-300 selection
  and box scaling also done inside the model.

The only per-frame CPU work left here is handing the frame to ONNX Runtime (which copies it to
the GPU) and turning the few confident outputs into :class:`Detection` objects.

D-FINE predicts one box per query and needs no non-maximum suppression, so
there is no IoU threshold to tune. Class names come from the model's own
metadata, never from a hard-coded list.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import PROJECT_ROOT

log = logging.getLogger(__name__)

CHECKSUMS_FILE = PROJECT_ROOT / "models" / "SHA256SUMS"
META_PREFIX = "ppe_detection."
SUPPORTED_FORMAT_VERSION = "2"
INPUT_NAME = "frame"
OUTPUT_NAMES = ("labels", "boxes", "scores")
PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


class ModelError(RuntimeError):
    """Raised when a model file is missing, corrupt, unapproved or incompatible."""


@dataclass(frozen=True)
class Detection:
    """One detected object, in pixel coordinates of the original frame."""

    class_id: int
    label: str
    score: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    sha256: str
    approved: bool
    labels: tuple[str, ...]
    input_size: tuple[int, int]  # (width, height)
    metadata: Mapping[str, str]
    provider: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path = CHECKSUMS_FILE) -> dict[str, str]:
    """Parse ``<sha256>  <filename>`` lines into ``{filename: sha256}``."""
    approved: dict[str, str] = {}
    if not path.is_file():
        return approved
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ModelError(f"{path.name} line {line_no} is not '<sha256>  <filename>': {line!r}")
        approved[parts[1].lstrip("*")] = parts[0].lower()
    return approved


def check_approval(path: Path, digest: str, approved: Mapping[str, str], allow_unverified: bool) -> bool:
    """Return True if ``digest`` is an approved model; raise or warn otherwise."""
    if digest in approved.values():
        return True
    if path.name in approved:
        raise ModelError(
            f"{path.name} does not match the SHA-256 recorded in models/SHA256SUMS "
            f"(expected {approved[path.name]}, got {digest}). The file is corrupt or has been modified."
        )
    if not allow_unverified:
        raise ModelError(
            f"{path.name} is not an approved model (its SHA-256 {digest} is not in models/SHA256SUMS). "
            "Verify its licence and provenance, record it in MODEL_LICENSES.md and SHA256SUMS, or pass "
            "--allow-unverified-model to run it anyway for testing."
        )
    log.warning("Running UNVERIFIED model %s (SHA-256 %s) because --allow-unverified-model was given.", path, digest)
    return False


def choose_providers(device: str, available: Iterable[str]) -> list[str]:
    """Pick ONNX Runtime execution providers for ``device`` from those actually available."""
    available = list(available)
    if device == "auto":
        chosen = [PROVIDERS[d] for d in ("cuda", "directml") if PROVIDERS[d] in available]
    else:
        wanted = PROVIDERS.get(device)
        if wanted is None:
            raise ModelError(f"Unknown device {device!r}; use auto, cpu, cuda or directml.")
        if wanted not in available:
            raise ModelError(
                f"--device {device} needs {wanted}, but this ONNX Runtime installation offers only "
                f"{', '.join(available) or 'nothing'}. Install the matching requirements file "
                "(see README) or use --device auto."
            )
        chosen = [wanted] if wanted != PROVIDERS["cpu"] else []
    return chosen + [PROVIDERS["cpu"]]


def parse_metadata(meta: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[int, int]]:
    """Validate the ``ppe_detection.*`` metadata and return (labels, (width, height))."""

    def get(key: str) -> str:
        value = meta.get(META_PREFIX + key)
        if value is None:
            raise ModelError(
                f"Model metadata is missing '{META_PREFIX}{key}'. Export the model with "
                "scripts/export_dfine_onnx.py so its class names and preprocessing are recorded."
            )
        return value

    if get("format_version") != SUPPORTED_FORMAT_VERSION:
        raise ModelError(
            f"Unsupported model format version {get('format_version')!r} (this app needs {SUPPORTED_FORMAT_VERSION}, "
            "a model with in-graph GPU preprocessing). Re-export it with scripts/export_dfine_onnx.py."
        )
    if get("model_family") != "dfine":
        raise ModelError(f"Unsupported model family {get('model_family')!r}; this app decodes D-FINE models.")
    if not get("preprocessing").startswith("in-graph"):
        raise ModelError(f"Unsupported preprocessing {get('preprocessing')!r}; expected in-graph preprocessing.")
    try:
        labels = json.loads(get("labels"))
        width, height = int(get("input_width")), int(get("input_height"))
    except (ValueError, TypeError) as exc:
        raise ModelError(f"Model metadata is malformed: {exc}") from exc
    if not (isinstance(labels, list) and labels and all(isinstance(x, str) and x for x in labels)):
        raise ModelError("Model metadata 'labels' must be a non-empty JSON list of class names.")
    if len(set(labels)) != len(labels):
        raise ModelError("Model metadata 'labels' contains duplicate class names.")
    if width <= 0 or height <= 0:
        raise ModelError(f"Model metadata has an invalid input size {width}x{height}.")
    return tuple(labels), (width, height)


def decode_outputs(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    conf: float,
    frame_size: tuple[int, int],
    names: tuple[str, ...],
    class_filter: frozenset[str] | None = None,
) -> list[Detection]:
    """Turn one image's D-FINE outputs into :class:`Detection` objects.

    ``labels`` [K], ``boxes`` [K, 4] and ``scores`` [K] are the batch-0 slices.
    Boxes are already in frame pixels (the model scales them by
    ``orig_target_sizes``); they are clipped to the frame and rounded.
    """
    if labels.ndim != 1 or scores.shape != labels.shape or boxes.shape != (labels.shape[0], 4):
        raise ModelError(
            f"Unexpected model output shapes labels={labels.shape}, boxes={boxes.shape}, scores={scores.shape}."
        )
    width, height = frame_size
    keep = np.flatnonzero(scores >= conf)
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    detections: list[Detection] = []
    for i in keep:
        class_id = int(labels[i])
        if not 0 <= class_id < len(names):
            raise ModelError(f"Model produced class index {class_id}, but its metadata lists {len(names)} classes.")
        name = names[class_id]
        if class_filter is not None and name not in class_filter:
            continue
        x1, y1, x2, y2 = boxes[i]
        box = (
            int(round(min(max(float(x1), 0.0), width - 1))),
            int(round(min(max(float(y1), 0.0), height - 1))),
            int(round(min(max(float(x2), 0.0), width - 1))),
            int(round(min(max(float(y2), 0.0), height - 1))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue  # degenerate after clipping
        detections.append(Detection(class_id, name, float(scores[i]), box))
    return detections


def _import_onnxruntime() -> Any:
    """Import ONNX Runtime, turning DLL/import failures into a ModelError with advice."""
    try:
        import onnxruntime as ort
    except (ImportError, OSError) as exc:
        raise ModelError(
            f"ONNX Runtime could not be loaded ({exc}). Check that exactly one of requirements.txt (GPU), "
            "requirements-cpu.txt or requirements-directml.txt is installed in this .venv. On managed "
            "Windows machines, 'WinError 4551' means an Application Control policy blocked one of its DLLs; "
            "ask IT to allow it."
        ) from exc
    # Errors only: ONNX Runtime 1.30 otherwise prints a harmless warning about its optional
    # plugin-device registry before loading CUDA. Real provider failures are still caught below.
    ort.set_default_logger_severity(3)
    return ort


def provider_options(providers: list[str], gpu_id: int) -> list[tuple[str, dict[str, str]]]:
    """Attach the GPU index to the GPU providers (ONNX Runtime's ``device_id`` option)."""
    gpu = {PROVIDERS["cuda"], PROVIDERS["directml"]}
    return [(name, {"device_id": str(gpu_id)} if name in gpu else {}) for name in providers]


def _default_session_factory(path: str, providers: list[tuple[str, dict[str, str]]]) -> Any:
    ort = _import_onnxruntime()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3  # errors only; provider fallback is reported by this module
    return ort.InferenceSession(path, sess_options=options, providers=providers)


def _available_providers() -> list[str]:
    ort = _import_onnxruntime()
    return list(ort.get_available_providers())


def _preload_cuda_dlls() -> None:
    """Let ONNX Runtime find CUDA/cuDNN DLLs installed from the nvidia-* pip packages."""
    ort = _import_onnxruntime()
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception as exc:  # noqa: BLE001 - provider check below reports the outcome
            log.warning("Could not preload CUDA/cuDNN DLLs: %s", exc)


class DFineDetector:
    """Loads a D-FINE ONNX model once and runs it on BGR frames."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        allow_unverified: bool = False,
        checksums_file: Path = CHECKSUMS_FILE,
        session_factory: Callable[[str, list[Any]], Any] = _default_session_factory,
        available_providers: Callable[[], list[str]] = _available_providers,
        gpu_id: int = 0,
    ) -> None:
        path = Path(model_path)
        if path.suffix.lower() in {".pt", ".pth"}:
            raise ModelError(
                f"{path.name} is a PyTorch checkpoint. This application only runs approved ONNX models; "
                "ppe.pt in particular is UNVERIFIED and Ultralytics-derived (see MODEL_LICENSES.md)."
            )
        if not path.is_file():
            raise ModelError(f"Model file not found: {path}. See models/README.md for how to obtain it.")
        if path.stat().st_size == 0:
            raise ModelError(f"Model file is empty: {path}")

        digest = sha256_file(path)
        approved = check_approval(path, digest, read_checksums(checksums_file), allow_unverified)

        providers = choose_providers(device, available_providers())
        if providers[0] == PROVIDERS["cuda"]:
            _preload_cuda_dlls()
        try:
            self._session = session_factory(str(path), provider_options(providers, gpu_id))
        except ModelError:
            raise
        except Exception as exc:  # onnxruntime raises several pybind exception types
            raise ModelError(
                f"Could not create an ONNX Runtime session for {path.name} with {', '.join(providers)}: {exc}. "
                "The file may be corrupt or not a valid ONNX model, or the execution provider failed to start "
                "(try --device cpu)."
            ) from exc

        actual = list(self._session.get_providers())
        if providers[0] not in actual:
            message = (
                f"{providers[0]} failed to initialise on GPU {gpu_id} (missing or blocked GPU DLLs, or an old "
                f"driver?); ONNX Runtime fell back to {actual[0]}."
            )
            if device != "auto":
                raise ModelError(message + " Use --device cpu to run on the CPU deliberately.")
            log.warning(message)

        self._check_io()
        meta = dict(self._session.get_modelmeta().custom_metadata_map)
        labels, input_size = parse_metadata(meta)
        self.info = ModelInfo(path, digest, approved, labels, input_size, meta, actual[0])
        self.last_inference_ms = 0.0
        where = f"GPU {gpu_id} via {actual[0]}" if actual[0] != PROVIDERS["cpu"] else "the CPU"
        log.info("Loaded %s on %s (%d classes; preprocessing runs inside the model).", path.name, where, len(labels))

    def _check_io(self) -> None:
        inputs = self._session.get_inputs()
        outputs = tuple(o.name for o in self._session.get_outputs())
        names = tuple(i.name for i in inputs)
        if names != (INPUT_NAME,) or sorted(outputs) != sorted(OUTPUT_NAMES):
            raise ModelError(
                f"Incompatible model: inputs {names} / outputs {outputs}; expected input ('{INPUT_NAME}',) and "
                f"outputs {OUTPUT_NAMES} from scripts/export_dfine_onnx.py (format version 2)."
            )
        frame = inputs[0]
        if getattr(frame, "type", "tensor(uint8)") != "tensor(uint8)" or len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ModelError(
                f"Model input 'frame' is {getattr(frame, 'type', '?')} {list(frame.shape)}; expected uint8 [H, W, 3]."
            )
        self._output_order = list(OUTPUT_NAMES)

    @property
    def labels(self) -> tuple[str, ...]:
        return self.info.labels

    def detect(
        self, frame: np.ndarray, conf: float, class_filter: frozenset[str] | None = None
    ) -> list[Detection]:
        """Run the model on one BGR uint8 frame and return detections at or above ``conf``.

        The frame is passed as-is (no CPU resize, colour conversion or float copy);
        ONNX Runtime uploads it and the model preprocesses it on the GPU.
        """
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected a BGR uint8 image of shape [H, W, 3], got {frame.dtype} {frame.shape}.")
        frame = np.ascontiguousarray(frame)  # no copy for frames from OpenCV, which are contiguous
        start = time.perf_counter()
        labels, boxes, scores = self._session.run(self._output_order, {INPUT_NAME: frame})
        self.last_inference_ms = (time.perf_counter() - start) * 1000.0
        return decode_outputs(
            labels[0], boxes[0], scores[0], conf, (frame.shape[1], frame.shape[0]), self.labels, class_filter
        )
