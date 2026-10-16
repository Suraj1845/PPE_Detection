# Model licences and provenance

Every AI model this application can load is listed here. Only models whose SHA-256 appears in
[`models/SHA256SUMS`](models/SHA256SUMS) load by default; anything else needs
`--allow-unverified-model` and is flagged as **UNVERIFIED** on screen.

Audit date: 2026-10-15. The full audit report, with every source opened, was delivered separately
(Phase 1); the evidence needed to re-check each claim is repeated below.

**Important: no approved model detects PPE yet.** The two approved models are general-purpose
COCO detectors. The application uses them to show **people only** and states on screen that PPE
compliance is not assessed. Compliant PPE detection needs a D-FINE model fine-tuned in-house on a
licence-approved PPE dataset (see [Adding a PPE model](#adding-a-ppe-model)).

## Summary

| Artifact | Status | Declared licence | Detects |
| --- | --- | --- | --- |
| `models/dfine_n_coco.onnx` | **Approved** (default) | Apache-2.0 (basis and caveats below) | 80 COCO classes; app shows `person` |
| `models/dfine_s_coco.onnx` | **Approved** | Apache-2.0 (basis and caveats below) | 80 COCO classes; app shows `person` |
| `ppe.pt` (project root) | **UNVERIFIED - not used** | None in file; Ultralytics lineage (GPL-3.0 / AGPL-3.0) | 10 PPE classes |

No tracking model or any other AI model is used. Associating PPE with a person
([`ppe_detection/compliance.py`](ppe_detection/compliance.py)) is plain box geometry, not a model.

## D-FINE-N (COCO) - `models/dfine_n_coco.onnx`

| Field | Value |
| --- | --- |
| Model | D-FINE-N, HGNetV2-B0 backbone, 4M parameters (per the D-FINE README), COCO-trained |
| Architecture / code | D-FINE, [github.com/Peterande/D-FINE](https://github.com/Peterande/D-FINE), commit `956d1709314c2c6a4df6f34de232054578a7449f` (2026-08-19). LICENSE file: Apache-2.0 |
| Source checkpoint | `dfine_n_coco.pth`, [release `dfinev1.0` of Peterande/storage](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth), 15,489,558 bytes, SHA-256 `41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c`. Contains only tensors (pickle globals checked; loaded with `weights_only=True`) |
| Artifact | `models/dfine_n_coco.onnx`, 15,489,380 bytes, SHA-256 `567fa8997efd52d463adcf0570395a86cef0b79ad69f735f27cce23ed64239a5` (format version 2: preprocessing built in) |
| Declared licence | **Apache-2.0**, recorded in the ONNX metadata key `ppe_detection.license` |
| Basis for the licence | (1) The D-FINE repository is Apache-2.0 and its README's licence note excludes only Objects365-trained checkpoints (`*_obj365`, `*_obj2coco`) from being assumed "commercially cleared under the D-FINE license", implying the COCO-only checkpoints fall under it. (2) The Hugging Face Transformers team's port of the same model, [ustc-community/dfine-nano-coco](https://huggingface.co/ustc-community/dfine-nano-coco), is tagged `apache-2.0` (the small port's full record is below) |
| Caveats | The original authors publish no per-file licence for the checkpoint. The ustc-community organisation (not verified on Hugging Face) also tags its Objects365 ports Apache-2.0, contradicting the authors, so that tag carries little independent weight. Optional: ask the D-FINE authors to confirm in a GitHub issue |
| Training data | COCO 2017 train. [COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm): annotations CC BY 4.0; "The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use." The backbone was initialised from `PPHGNetV2_B0_stage1.pth` (SHA-256 `70a372e8cbc59b34c5da2943261ecb633faf304a58e7e05461a27bd8d8b7f3d1`), an ImageNet-pretrained PaddleClas backbone; ImageNet's terms are research-oriented. This image-provenance risk is shared by almost every pretrained detector |
| Classes | 80 COCO names in D-FINE's contiguous label order (from `src/data/dataset/coco_dataset.py` at the commit above), stored in the ONNX metadata key `ppe_detection.labels`. Index 0 = `person`. Verified by test (`tests/test_real_model.py`) |
| Conversion | `scripts/export_dfine_onnx.py`: the logic of D-FINE's `tools/deployment/export_onnx.py` (deploy-mode model + postprocessor, opset 16, dynamic batch, constant folding), PyTorch 2.11.0 TorchScript exporter (`dynamo=False`), `onnx` 1.23.2 checker, `onnxsim` 0.8.1. Differences from upstream: tracing batch 2 instead of 32 (memory), `weights_only=True`. Then nine standard ONNX operators are prepended (Cast, Gather, Transpose, Unsqueeze, Resize, Mul, Shape, Gather, Unsqueeze) so the model takes a raw `uint8` BGR frame and does BGR to RGB, the bilinear 640x640 resize, the 0-1 scaling and the frame-size computation itself, on the GPU. These operators contain no learned weights. Never use D-FINE's `export_yolo_w_nms.py`, which imports `ultralytics` (AGPL-3.0) |
| Conversion check | (1) Network: PyTorch vs ONNX Runtime on three sample-video frames, batch 1 and batch 2: identical labels, max box difference 0.001 px, max score difference < 0.00001. (2) Built-in preprocessing: the finished model on raw frames vs the network fed PyTorch's bilinear resize (`align_corners=False`): identical confident detections, max score difference < 0.00001. Re-running the export reproduced the identical SHA-256 |
| Runtime | ONNX Runtime (MIT); `onnxruntime-gpu` 1.30.0 with the CUDA provider by default. Tested here with ONNX Runtime 1.30.0 on the CPU provider only (no GPU in the test environment). Input `frame` uint8 [height, width, 3] (BGR, any size); outputs `labels` int64 [1,300], `boxes` float32 [1,300,4] (x1,y1,x2,y2 in frame pixels), `scores` float32 [1,300]. No NMS. No PyTorch or Ultralytics needed |

## D-FINE-S (COCO) - `models/dfine_s_coco.onnx`

Identical to D-FINE-N above except:

| Field | Value |
| --- | --- |
| Model | D-FINE-S, HGNetV2-B0 backbone, 10M parameters (per the D-FINE README) |
| Source checkpoint | `dfine_s_coco.pth`, [release `dfinev1.0`](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth), 41,841,422 bytes, SHA-256 `48a6c8cc43eb57186843f752e2e8461ddd3326e0d3c575e71e6e960844683e89` |
| Artifact | `models/dfine_s_coco.onnx`, 41,744,021 bytes, SHA-256 `3af4bf499e26f254bda0538d1c075b156893828b40781e1f72a1ced7ad7c1deb` (format version 2: preprocessing built in) |
| Hugging Face port | [ustc-community/dfine-small-coco](https://huggingface.co/ustc-community/dfine-small-coco) @ `f79e65b5fbb33ceb9d3ebba042955d7410c608f8`, tagged `apache-2.0`, `datasets: coco` |
| Conversion check | Same two checks as D-FINE-N: identical labels, max box difference < 0.001 px, max score difference < 0.00001 |

## `ppe.pt` - UNVERIFIED, kept for investigation only

The original project's checkpoint is preserved unchanged in the project root but is **not** listed in
`SHA256SUMS`. The application refuses `.pt`/`.pth` files outright.

| Field | Finding |
| --- | --- |
| File | `ppe.pt`, 87,634,504 bytes, SHA-256 `2ec3a84353574e4beec4f9cfd3952c43bcf2406508dd8b581594b257987d4f13` |
| Contents | Pickled `ultralytics.nn.tasks.DetectionModel` (YOLOv8-L layout, about 43.7M FP16 values), Ultralytics 8.0.30, saved 2023-02-06; fine-tuned from Ultralytics `yolov8l.pt`; 50 epochs at 640 px on `.../Datasets/ConstructionSafety/data.yaml` (a Google Colab path) |
| Classes | 0 Hardhat, 1 Mask, 2 NO-Hardhat, 3 NO-Mask, 4 NO-Safety Vest, 5 Person, 6 Safety Cone, 7 Safety Vest, 8 machinery, 9 vehicle (read from the checkpoint; matches the old `main.py`) |
| Source | Byte-identical to `ppe.pt` in [Vinayakmane47/PPE_detection_YOLO](https://github.com/Vinayakmane47/PPE_detection_YOLO) (commit `30784f9ecd2ca6a249c0f9e72717feec32900937`). That repo's MIT licence covers its author's code; who trained the weights is not established |
| Licence | None in the file. Ultralytics was GPL-3.0 at v8.0.30, and [Ultralytics states](https://www.ultralytics.com/license) "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default" |
| Dataset | Probably Roboflow [Construction Site Safety v27](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27) (CC BY 4.0); not proven |
| Runtime | Would require the `ultralytics` package (AGPL-3.0) and loads by unpickling Python objects |
| Decision | Not MIT/Apache-2.0 compliant on the evidence available. Do not use or redistribute |

The Hugging Face candidate `innovativeskipper/ppe-compliance-yolov8` (revision
`a427495f771c65c328ae3db83e2ef2d24d7f2813`) was also rejected: it is tagged MIT, but its own card
calls it a "Fine-tuned YOLOv8n model" and it ships only an Ultralytics `best.pt`.

## Adding a PPE model

The training kit in [`training/`](training/README.md) does steps 2-4 and drafts step 5.

1. Choose a PPE dataset and record its licence (e.g. CC BY 4.0 requires attribution in the model card).
2. Fine-tune D-FINE from `dfine_n_coco.pth` or `dfine_s_coco.pth` with D-FINE's custom-dataset configs.
   Use class names from the table in `ppe_detection/compliance.py` (`Hardhat`, `NO-Hardhat`, `Mask`,
   `NO-Mask`, `Safety Vest`, `NO-Safety Vest`, `Person`) so person-level assessment switches on.
3. Export with `scripts/export_dfine_onnx.py --config ... --labels-json ... --verify-image ...`.
4. Evaluate it on a held-out split and record the measured metrics.
5. Decide the licence of the fine-tuned weights (base weights and code are Apache-2.0; the training
   data's terms also apply), add a section to this file (dataset, licence, checksums, metrics), then
   add its SHA-256 to `models/SHA256SUMS`. Only then does it load without `--allow-unverified-model`.
