"""Download D-FINE's source code at the audited commit and its COCO starting weights.

    python training\\setup_dfine.py

Creates (inside training\\, all git-ignored):

* ``training\\D-FINE\\``      D-FINE source at commit 956d170 (Apache-2.0), with a DFINE_COMMIT.txt marker
* ``training\\weights\\``     dfine_n_coco.pth and dfine_s_coco.pth, each checked against its audited SHA-256

Uses ``git`` if it is installed, otherwise GitHub's source archive for that exact commit.
Safe to run again: anything already present and correct is kept.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
PROJECT = TRAINING.parent
DFINE_DIR = TRAINING / "D-FINE"
WEIGHTS_DIR = TRAINING / "weights"
COMMIT_MARKER = "DFINE_COMMIT.txt"


def audited() -> tuple[str, dict]:
    """Commit and checkpoint table from scripts/export_dfine_onnx.py (single source of truth)."""
    spec = importlib.util.spec_from_file_location("export_dfine_onnx", PROJECT / "scripts" / "export_dfine_onnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AUDITED_DFINE_COMMIT, module.AUDITED_CHECKPOINTS


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str) -> bytes:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def fetch_source(commit: str) -> None:
    marker = DFINE_DIR / COMMIT_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == commit:
        print(f"D-FINE source already at {commit[:7]}: {DFINE_DIR}")
        return
    if DFINE_DIR.exists():
        shutil.rmtree(DFINE_DIR)
    if shutil.which("git"):
        print("Cloning D-FINE with git ...")
        subprocess.run(["git", "clone", "--quiet", "https://github.com/Peterande/D-FINE", str(DFINE_DIR)], check=True)
        subprocess.run(["git", "-C", str(DFINE_DIR), "checkout", "--quiet", commit], check=True)
    else:
        print("git not found; downloading the source archive for the commit ...")
        data = download(f"https://github.com/Peterande/D-FINE/archive/{commit}.zip")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            top = archive.namelist()[0].split("/")[0]
            archive.extractall(TRAINING)
        (TRAINING / top).rename(DFINE_DIR)
    for required in ("train.py", "configs/dfine/custom/dfine_hgnetv2_s_custom.yml", "src/core/yaml_config.py"):
        if not (DFINE_DIR / required).is_file():
            raise SystemExit(f"D-FINE source looks incomplete: {required} is missing.")
    marker.write_text(commit + "\n", encoding="utf-8")
    print(f"D-FINE source ready at {commit[:7]}: {DFINE_DIR}")


def fetch_weights(checkpoints: dict) -> None:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for variant, info in sorted(checkpoints.items()):
        path = WEIGHTS_DIR / info["file"]
        if path.is_file() and sha256_of(path) == info["sha256"]:
            print(f"{info['file']}: already present, SHA-256 OK")
            continue
        data = download(info["url"])
        if hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise SystemExit(f"{info['file']}: downloaded file does not match the audited SHA-256; not saved.")
        path.write_bytes(data)
        print(f"{info['file']}: saved, SHA-256 OK (D-FINE-{variant.upper()} COCO weights, Apache-2.0)")


def main() -> int:
    commit, checkpoints = audited()
    fetch_source(commit)
    fetch_weights(checkpoints)
    print("\nNext: python training\\prepare_dataset.py --dataset <your dataset folder>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
