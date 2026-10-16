"""Application settings: defaults, validation, optional TOML file and path resolution."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import PROJECT_ROOT

DEFAULT_MODEL = Path("models/dfine_n_coco.onnx")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_OUTPUT_SUFFIXES = frozenset({".mp4", ".avi"})
OUTPUT_SUFFIXES = IMAGE_SUFFIXES | VIDEO_OUTPUT_SUFFIXES
DEVICES = ("cuda", "directml", "cpu", "auto")  # cuda is the default; cpu only when asked for
BACKENDS = ("auto", "any", "dshow", "msmf", "v4l2", "avfoundation")


class ConfigError(ValueError):
    """Raised when a setting is missing, out of range or inconsistent."""


class SourceKind(StrEnum):
    CAMERA = "camera"
    VIDEO = "video"
    IMAGE = "image"


@dataclass(frozen=True)
class SourceSpec:
    """Where frames come from: a camera index, a video file or an image file."""

    kind: SourceKind
    camera_index: int | None = None
    path: Path | None = None

    def describe(self) -> str:
        return f"camera {self.camera_index}" if self.kind is SourceKind.CAMERA else str(self.path)


@dataclass(frozen=True)
class AppConfig:
    """All run-time settings. Build with :func:`build_config`, which validates them."""

    source: str = "0"
    model: Path = DEFAULT_MODEL
    conf: float = 0.5
    width: int = 1280
    height: int = 720
    device: str = "cuda"
    gpu_id: int = 0
    backend: str = "auto"
    output: Path | None = None
    display: bool = True
    classes: tuple[str, ...] | None = None  # None = model default; ("all",) = every class
    allow_unverified_model: bool = False
    max_frames: int | None = None
    # Person-to-PPE association (see compliance.py)
    min_item_overlap: float = 0.5
    required_ppe: tuple[str, ...] = ("hardhat", "vest")
    fourcc: str | None = None  # e.g. "MJPG": camera pixel format request (optional)
    fps: int = 30  # frame rate requested from the camera
    # Where relative file *sources* are resolved from when they are not found under the CWD.
    base_dir: Path = field(default=PROJECT_ROOT, compare=False)

    @property
    def model_path(self) -> Path:
        """Absolute model path. Relative model paths are always relative to the project folder;
        ``load_toml`` turns paths written in a config file into absolute paths first."""
        return resolve_path(self.model, PROJECT_ROOT)


def resolve_path(path: Path | str, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` against ``base_dir`` (the project or config folder), not the CWD."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def parse_source(source: str | int, base_dir: Path = PROJECT_ROOT) -> SourceSpec:
    """Turn ``--source`` into a :class:`SourceSpec`.

    A non-negative integer is a camera index. Anything else is a file path:
    an existing path relative to the current folder is used as typed (what a
    user expects from ``--source Samples/ppe1.mp4``); otherwise it is resolved
    against ``base_dir``. Image suffixes select image mode.
    """
    text = str(source).strip()
    if not text:
        raise ConfigError("--source is empty. Use a camera index such as 0 or a file path.")
    if text.lstrip("+-").isdigit():
        index = int(text)
        if index < 0:
            raise ConfigError(f"Camera index must be 0 or greater, got {index}.")
        return SourceSpec(SourceKind.CAMERA, camera_index=index)

    typed = Path(text).expanduser()
    path = typed.resolve() if typed.is_absolute() or typed.exists() else resolve_path(typed, base_dir)
    if not path.is_file():
        raise ConfigError(f"Input file not found: {text} (looked for {path}).")
    kind = SourceKind.IMAGE if path.suffix.lower() in IMAGE_SUFFIXES else SourceKind.VIDEO
    return SourceSpec(kind, path=path)


def validate(config: AppConfig) -> AppConfig:
    """Check every setting and return the config unchanged, or raise :class:`ConfigError`."""
    if not 0.0 < config.conf <= 1.0:
        raise ConfigError(f"--conf must be greater than 0 and at most 1, got {config.conf}.")
    if config.width <= 0 or config.height <= 0:
        raise ConfigError(f"--width and --height must be positive, got {config.width}x{config.height}.")
    if config.width > 7680 or config.height > 4320:
        raise ConfigError(f"Requested resolution {config.width}x{config.height} is larger than 8K.")
    if config.gpu_id < 0:
        raise ConfigError(f"--gpu-id must be 0 or greater, got {config.gpu_id}.")
    if config.device not in DEVICES:
        raise ConfigError(f"--device must be one of {', '.join(DEVICES)}, got {config.device!r}.")
    if config.backend not in BACKENDS:
        raise ConfigError(f"--backend must be one of {', '.join(BACKENDS)}, got {config.backend!r}.")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ConfigError(f"--max-frames must be positive, got {config.max_frames}.")
    if not 0.0 < config.min_item_overlap <= 1.0:
        raise ConfigError(f"min_item_overlap must be in (0, 1], got {config.min_item_overlap}.")
    if not config.required_ppe:
        raise ConfigError("--required-ppe is empty; list at least one of hardhat, vest, mask.")
    unknown = set(config.required_ppe) - {"hardhat", "vest", "mask"}
    if unknown:
        raise ConfigError(f"required_ppe may contain hardhat, vest, mask; unknown: {sorted(unknown)}.")
    if config.classes is not None and (not config.classes or not all(name.strip() for name in config.classes)):
        raise ConfigError("--classes is empty or contains an empty class name; use 'all' to show every class.")
    if not 1 <= config.fps <= 240:
        raise ConfigError(f"--fps must be between 1 and 240, got {config.fps}.")
    if config.fourcc is not None and (len(config.fourcc) != 4 or not config.fourcc.isascii()):
        raise ConfigError(f"--fourcc must be exactly 4 ASCII characters such as MJPG, got {config.fourcc!r}.")
    if config.model_path.suffix.lower() != ".onnx":
        raise ConfigError(
            f"Model must be an .onnx file, got {config.model_path.name}. PyTorch checkpoints such as "
            "ppe.pt are not loaded by this application (see MODEL_LICENSES.md)."
        )
    if config.output is not None and config.output.suffix.lower() not in OUTPUT_SUFFIXES:
        raise ConfigError(
            f"--output must end in one of {', '.join(sorted(OUTPUT_SUFFIXES))}, got {config.output.name!r}."
        )
    return config


#: Accepted TOML value types per setting (bool is excluded from numbers on purpose).
_TOML_TYPES: dict[str, tuple[type, ...]] = {
    "source": (str, int),
    "model": (str,),
    "conf": (int, float),
    "width": (int,),
    "height": (int,),
    "device": (str,),
    "gpu_id": (int,),
    "backend": (str,),
    "output": (str,),
    "display": (bool,),
    "classes": (list,),
    "allow_unverified_model": (bool,),
    "max_frames": (int,),
    "min_item_overlap": (int, float),
    "required_ppe": (list,),
    "fourcc": (str,),
    "fps": (int,),
}


def load_toml(path: Path) -> AppConfig:
    """Load settings from a TOML file.

    Relative ``model``, ``output`` and file ``source`` paths in it are resolved against the
    file's folder. Settings it does not mention keep their defaults (e.g. the project's model).
    """
    path = path.expanduser().resolve()
    try:
        # utf-8-sig: tolerate the byte-order mark some Windows editors write.
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc

    unknown = set(data) - set(_TOML_TYPES)
    if unknown:
        raise ConfigError(f"Unknown setting(s) in {path.name}: {', '.join(sorted(unknown))}.")
    for key, value in data.items():
        allowed = _TOML_TYPES[key]
        if (isinstance(value, bool) and bool not in allowed) or not isinstance(value, allowed):
            names = " or ".join(t.__name__ for t in allowed)
            raise ConfigError(f"{path.name}: '{key}' must be {names}, got {type(value).__name__} {value!r}.")
        if key in ("classes", "required_ppe") and not all(isinstance(x, str) for x in value):
            raise ConfigError(f"{path.name}: '{key}' must be a list of strings.")

    values: dict[str, Any] = dict(data)
    for key in ("model", "output"):
        if key in values:
            values[key] = resolve_path(values[key], path.parent)
    for key in ("classes", "required_ppe"):
        if key in values:
            values[key] = tuple(values[key])
    if "source" in values:
        values["source"] = str(values["source"])
    return AppConfig(base_dir=path.parent, **values)


def build_config(base: AppConfig | None = None, **overrides: Any) -> AppConfig:
    """Apply non-``None`` overrides (e.g. from the CLI) on top of ``base`` and validate."""
    config = base or AppConfig()
    changes = {key: value for key, value in overrides.items() if value is not None}
    return validate(replace(config, **changes))
