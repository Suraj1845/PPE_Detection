"""Real-time PPE detection with D-FINE ONNX models and ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

__version__ = "2.0.0"

#: The project folder (the parent of this package). Relative model paths are
#: resolved against it, never against the terminal's current directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
