"""Tests for the training kit's pure-Python parts (no PyTorch, no GPU, no downloads).

The dataset-conversion tests need Pillow and PyYAML, which are in training/requirements-train.txt
but not in the app's requirements; they are skipped when those are missing.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import PPE_LABELS

from ppe_detection import PROJECT_ROOT

TRAINING = PROJECT_ROOT / "training"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"training_{name}", TRAINING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ train.py


def test_learning_rates_scale_with_batch_size():
    train = load("train")
    optimizer = {
        "type": "AdamW",
        "lr": 0.0004,
        "params": [{"params": "^(?=.*backbone)(?!.*bn).*$", "lr": 0.0002}, {"params": "norm", "weight_decay": 0.0}],
    }
    scaled = train.scale_optimizer(optimizer, 8 / 64)
    assert scaled["lr"] == pytest.approx(0.00005)
    assert scaled["params"][0]["lr"] == pytest.approx(0.000025)
    assert "lr" not in scaled["params"][1]
    assert optimizer["lr"] == 0.0004  # the original is not modified


# --------------------------------------------------------------- evaluate_onnx.py


def coco(boxes: list[tuple[int, int, list[float]]]) -> dict:
    """Ground truth: (image_id, class_id, [x, y, w, h])."""
    images = sorted({image_id for image_id, _, _ in boxes})
    return {
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": 640, "height": 480} for i in images],
        "annotations": [
            {"id": n, "image_id": i, "category_id": c, "bbox": b, "area": b[2] * b[3], "iscrowd": 0}
            for n, (i, c, b) in enumerate(boxes, start=1)
        ],
        "categories": [{"id": k, "name": name} for k, name in enumerate(PPE_LABELS)],
    }


def test_operating_point_counts_hits_misses_and_false_alarms():
    evaluate = load("evaluate_onnx")
    hardhat, vest = PPE_LABELS.index("Hardhat"), PPE_LABELS.index("Safety Vest")
    gt = coco([(0, hardhat, [10, 10, 50, 50]), (0, hardhat, [200, 10, 50, 50]), (1, vest, [10, 10, 100, 100])])
    results = [
        {"image_id": 0, "category_id": hardhat, "bbox": [12, 12, 50, 50], "score": 0.9},  # hit
        {"image_id": 0, "category_id": hardhat, "bbox": [14, 14, 50, 50], "score": 0.8},  # duplicate: false
        {"image_id": 0, "category_id": hardhat, "bbox": [400, 400, 50, 50], "score": 0.3},  # below conf: ignored
        {"image_id": 1, "category_id": hardhat, "bbox": [10, 10, 100, 100], "score": 0.9},  # wrong class: false
    ]
    table = evaluate.operating_point(gt, results, list(PPE_LABELS), conf=0.5)
    assert table["Hardhat"] == {
        "instances": 2, "true_positives": 1, "false_positives": 2, "missed": 1,
        "precision": pytest.approx(1 / 3), "recall": 0.5,
    }
    assert table["Safety Vest"]["missed"] == 1 and table["Safety Vest"]["recall"] == 0.0
    assert math.isnan(table["Mask"]["recall"])


def test_coco_ap_is_one_for_perfect_predictions_and_lower_for_misses():
    pytest.importorskip("faster_coco_eval")
    evaluate = load("evaluate_onnx")
    gt = coco([(i, k % 3, [10.0 + k * 60, 20.0, 50.0, 80.0]) for i in range(4) for k in range(3)])
    perfect = [
        {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
        for a in gt["annotations"]
    ]
    overall, per_class = evaluate.coco_ap(gt, perfect, list(PPE_LABELS))
    assert overall["AP"] == pytest.approx(1.0) and per_class["Hardhat"]["AP50"] == pytest.approx(1.0)
    assert math.isnan(per_class["vehicle"]["AP"])  # no boxes of that class
    overall, _ = evaluate.coco_ap(gt, perfect[::2], list(PPE_LABELS))
    assert 0.3 < overall["AP"] < 0.7


# ------------------------------------------------------------ prepare_dataset.py


def write_image(path: Path, width: int = 64, height: int = 48) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((height, width, 3), np.uint8)).save(path)


def test_yolo_dataset_is_converted_with_clipping_and_polygons(tmp_path: Path):
    pytest.importorskip("PIL")
    pytest.importorskip("yaml")
    prepare = load("prepare_dataset")
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: ['Hardhat', 'Person']\nnc: 2\n", encoding="utf-8")
    (dataset / "README.roboflow.txt").write_text("License: CC BY 4.0\nunrelated line\n", encoding="utf-8")
    for split in ("train", "valid"):
        write_image(dataset / split / "images" / "a.jpg")
        labels = dataset / split / "labels"
        labels.mkdir(parents=True)
        (labels / "a.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n"            # box: 16,12 32x24
            "1 0.1 0.1 0.9 0.1 0.9 0.9\n"    # polygon -> box 6.4,4.8 51.2x38.4
            "1 0.99 0.5 0.2 0.2\n"           # straddles the right edge: clipped
            "7 0.5 0.5 0.1 0.1\n"            # unknown class: dropped
            "0 1.5 1.5 0.1 0.1\n",           # outside the image: dropped
            encoding="utf-8",
        )
    out = tmp_path / "out"
    assert prepare.main(["--dataset", str(dataset), "--out", str(out)]) == 0

    train = json.loads((out / "train.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in train["categories"]] == ["Hardhat", "Person"]
    boxes = [(a["category_id"], a["bbox"]) for a in train["annotations"]]
    assert boxes[0] == (0, [16.0, 12.0, 32.0, 24.0])
    assert boxes[1] == (1, [6.4, 4.8, 51.2, 38.4])
    assert boxes[2][1][0] + boxes[2][1][2] == pytest.approx(64.0)  # clipped at the right edge
    assert len(boxes) == 3
    assert min(a["id"] for a in train["annotations"]) == 1  # COCO evaluators treat id 0 as "unmatched"
    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["licence_lines"] == {"README.roboflow.txt": ["License: CC BY 4.0"]}
    assert info["issues"]["boxes with an unknown class id (dropped)"] == 2
    assert "test" not in info["counts"]


def test_roboflow_coco_dataset_drops_placeholder_category_and_remaps_ids(tmp_path: Path):
    pytest.importorskip("PIL")
    prepare = load("prepare_dataset")
    dataset = tmp_path / "ds"
    categories = [{"id": 0, "name": "ppe"}, {"id": 1, "name": "Hardhat"}, {"id": 2, "name": "Person"}]
    for split in ("train", "valid", "test"):
        write_image(dataset / split / "a.jpg")
        (dataset / split / "_annotations.coco.json").write_text(json.dumps({
            "images": [{"id": 5, "file_name": "a.jpg", "width": 64, "height": 48},
                       {"id": 6, "file_name": "missing.jpg", "width": 64, "height": 48}],
            "annotations": [{"id": 1, "image_id": 5, "category_id": 2, "bbox": [1, 2, 10, 10]},
                            {"id": 2, "image_id": 5, "category_id": 1, "bbox": [20, 2, 10, 10]},
                            {"id": 3, "image_id": 6, "category_id": 1, "bbox": [20, 2, 10, 10]}],
            "categories": categories,
        }), encoding="utf-8")
    out = tmp_path / "out"
    assert prepare.main(["--dataset", str(dataset), "--out", str(out)]) == 0

    assert json.loads((out / "labels.json").read_text(encoding="utf-8")) == ["Hardhat", "Person"]
    test = json.loads((out / "test.json").read_text(encoding="utf-8"))
    assert [img["id"] for img in test["images"]] == [0]
    assert sorted((a["category_id"], a["bbox"][0]) for a in test["annotations"]) == [(0, 20.0), (1, 1.0)]
    info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["format"] == "COCO" and info["image_folders"]["val"].endswith("valid")
