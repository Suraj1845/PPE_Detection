"""Check a PPE dataset and convert it into the COCO format D-FINE trains on.

    python training\\prepare_dataset.py --dataset C:\\Users\\ATL\\Desktop\\ppe-data\\construction-site-safety

Accepts the two layouts Roboflow (and most tools) export:

* YOLO:  data.yaml + {train,valid,test}/images + {train,valid,test}/labels (*.txt)
* COCO:  {train,valid,test}/_annotations.coco.json next to the images

Writes ``training\\work\\data\\`` (git-ignored): ``train.json``, ``val.json``, ``test.json`` (COCO,
class ids 0..N-1 in the dataset's own class order, which is what D-FINE expects), ``labels.json``
(class names in that order) and ``dataset_info.json`` (source, licence text found, counts).
Images are not copied; the JSON files point at them where they are.

It also reports instances per class and split, dropped/clipped boxes, and which classes the app
will use for person-level PPE checks, and prints the licence lines it finds so you can record them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
PROJECT = TRAINING.parent
sys.path.insert(0, str(PROJECT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": ("train",), "val": ("valid", "val", "validation"), "test": ("test",)}
LICENSE_FILES = ("README.roboflow.txt", "README.dataset.txt", "README.md", "README.txt", "LICENSE", "LICENSE.txt")
MIN_BOX = 1.0  # pixels


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size  # (width, height)


def clip_box(x: float, y: float, w: float, h: float, width: int, height: int) -> list[float] | None:
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
        return None
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def find_split_dir(dataset: Path, split: str) -> Path | None:
    for name in SPLITS[split]:
        if (dataset / name).is_dir():
            return dataset / name
    return None


# ------------------------------------------------------------------------- YOLO


def yolo_names(dataset: Path) -> list[str]:
    import yaml

    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8-sig"))
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not (isinstance(names, list) and names and all(isinstance(n, str) for n in names)):
        raise SystemExit("data.yaml has no usable 'names' list.")
    return names


def read_yolo_split(split_dir: Path, num_classes: int, stats: Counter) -> tuple[list[dict], list[dict]]:
    images_dir, labels_dir = split_dir / "images", split_dir / "labels"
    if not images_dir.is_dir():
        images_dir, labels_dir = split_dir, split_dir
    images, annotations = [], []
    for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        width, height = image_size(image_path)
        image_id = len(images)
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.is_file():
            stats["images without a label file"] += 1
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
            if not 0 <= cls < num_classes:
                stats["boxes with an unknown class id (dropped)"] += 1
                continue
            if len(values) == 4:  # cx cy w h, normalised
                cx, cy, w, h = values
                x, y, w, h = (cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height
            elif len(values) >= 6 and len(values) % 2 == 0:  # polygon: x1 y1 x2 y2 ... normalised
                xs, ys = values[0::2], values[1::2]
                x, y = min(xs) * width, min(ys) * height
                w, h = (max(xs) - min(xs)) * width, (max(ys) - min(ys)) * height
                stats["polygon labels converted to boxes"] += 1
            else:
                stats["malformed label lines (dropped)"] += 1
                continue
            box = clip_box(x, y, w, h, width, height)
            if box is None:
                stats["boxes too small after clipping (dropped)"] += 1
                continue
            annotations.append({"image_id": image_id, "category_id": cls, "bbox": box})
    return images, annotations


# ------------------------------------------------------------------------- COCO


def read_coco_split(split_dir: Path, stats: Counter) -> tuple[list[dict], list[dict], list[dict]]:
    ann_file = split_dir / "_annotations.coco.json"
    data = json.loads(ann_file.read_text(encoding="utf-8-sig"))
    by_id = {img["id"]: img for img in data["images"]}
    images, id_map = [], {}
    for img in data["images"]:
        if not (split_dir / img["file_name"]).is_file():
            stats["images listed but missing on disk (dropped)"] += 1
            continue
        id_map[img["id"]] = len(images)
        images.append({"id": len(images), "file_name": img["file_name"], "width": img["width"], "height": img["height"]})
    annotations = []
    for ann in data.get("annotations", []):
        if ann["image_id"] not in id_map:
            continue
        img = by_id[ann["image_id"]]
        box = clip_box(*ann["bbox"], img["width"], img["height"])
        if box is None:
            stats["boxes too small after clipping (dropped)"] += 1
            continue
        annotations.append({"image_id": id_map[ann["image_id"]], "category_id": ann["category_id"], "bbox": box})
    return images, annotations, data["categories"]


# ------------------------------------------------------------------------ common


def find_licence(dataset: Path) -> dict:
    """Collect lines that mention a licence or source URL from the files datasets ship with."""
    found: dict[str, list[str]] = {}
    for name in LICENSE_FILES + ("data.yaml",):
        path = dataset / name
        if path.is_file():
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                if re.search(r"licen[cs]e|https?://|provided by|cc by|creative commons", line, re.IGNORECASE)
            ]
            if lines:
                found[name] = lines[:12]
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="dataset folder (YOLO or COCO layout)")
    parser.add_argument("--out", type=Path, default=TRAINING / "work" / "data", help="output folder")
    args = parser.parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset}")

    split_dirs = {split: find_split_dir(dataset, split) for split in SPLITS}
    if split_dirs["train"] is None or split_dirs["val"] is None:
        raise SystemExit(f"{dataset} needs at least 'train' and 'valid' (or 'val') sub-folders.")
    is_yolo = (dataset / "data.yaml").is_file()
    is_coco = (split_dirs["train"] / "_annotations.coco.json").is_file()
    if not (is_yolo or is_coco):
        raise SystemExit("Unrecognised layout: expected data.yaml (YOLO) or train/_annotations.coco.json (COCO).")

    stats: Counter = Counter()
    raw: dict[str, tuple[list[dict], list[dict], Path]] = {}
    if is_coco:
        categories = None
        for split, split_dir in split_dirs.items():
            if split_dir is not None:
                images, anns, cats = read_coco_split(split_dir, stats)
                categories = categories or cats
                raw[split] = (images, anns, split_dir)
        used = {a["category_id"] for _, anns, _ in raw.values() for a in anns}
        # Roboflow adds a placeholder super-category with no boxes; keep only classes that are used.
        kept = [c for c in sorted(categories, key=lambda c: c["id"]) if c["id"] in used]
        for c in categories:
            if c["id"] not in used:
                print(f"note: category {c['id']} '{c['name']}' has no boxes in any split; left out")
        names = [c["name"] for c in kept]
        remap = {c["id"]: i for i, c in enumerate(kept)}
        for _, anns, _ in raw.values():
            for a in anns:
                a["category_id"] = remap[a["category_id"]]
        source_format = "COCO"
    else:
        names = yolo_names(dataset)
        for split, split_dir in split_dirs.items():
            if split_dir is not None:
                images, anns = read_yolo_split(split_dir, len(names), stats)
                image_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
                raw[split] = (images, anns, image_dir)
        source_format = "YOLO"

    if len(set(names)) != len(names):
        raise SystemExit(f"Duplicate class names in the dataset: {names}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    for split, (images, anns, _) in raw.items():
        for i, ann in enumerate(anns, start=1):  # COCO evaluators treat annotation id 0 as "unmatched"
            ann.update(id=i, area=round(ann["bbox"][2] * ann["bbox"][3], 2), iscrowd=0)
        coco = {
            "images": images,
            "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in enumerate(names)],
        }
        (out / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
        per_class = Counter(names[a["category_id"]] for a in anns)
        counts[split] = {"images": len(images), **{n: per_class.get(n, 0) for n in names}}

    (out / "labels.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    licence = find_licence(dataset)
    info = {
        "dataset_folder": str(dataset),
        "format": source_format,
        "classes": names,
        "image_folders": {split: str(image_dir) for split, (_, _, image_dir) in raw.items()},
        "counts": counts,
        "issues": dict(stats),
        "licence_lines": licence,
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- report
    print(f"\nDataset: {dataset} ({source_format} layout), {len(names)} classes")
    splits = list(counts)
    print(f"  {'class':18}" + "".join(f"{s:>9}" for s in splits))
    print(f"  {'(images)':18}" + "".join(f"{counts[s]['images']:>9}" for s in splits))
    for n in names:
        print(f"  {n:18}" + "".join(f"{counts[s][n]:>9}" for s in splits))
    for issue, n in sorted(stats.items()):
        print(f"  ! {issue}: {n}")
    for split in ("train", "val"):
        empty = [n for n in names if counts.get(split, {}).get(n, 0) == 0]
        if empty:
            print(f"  ! no '{split}' boxes for: {', '.join(empty)}")
    if "test" not in counts:
        print("  ! no test split: evaluation will fall back to the validation split, which also chose the best checkpoint")

    from ppe_detection.compliance import PpeSchema

    schema = PpeSchema.from_labels(names)
    if schema.ppe_supported:
        print(f"\nApp PPE rules: person class {sorted(schema.person_labels)}, PPE classes "
              f"{sorted(schema.item_roles)} -> per-person checks for {sorted(schema.categories)}")
    else:
        print("\n! The app's per-person PPE checks need a 'Person' class and PPE classes named like "
              "Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest / Mask / NO-Mask. With these names the app "
              "will only draw object-level boxes.")

    if licence:
        print("\nLicence/source lines found (record these in MODEL_LICENSES.md):")
        for file, lines in licence.items():
            for line in lines:
                print(f"  [{file}] {line}")
    else:
        print("\n! No licence information found in the dataset folder. Find the dataset's licence and source "
              "before using a model trained on it.")
    print(f"\nWrote {out}\nNext: python training\\train.py --variant s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
