# Training a PPE model (D-FINE fine-tune)

The app ships D-FINE models trained on COCO: they find people, not hardhats or vests. This kit
fine-tunes D-FINE on a PPE dataset on your own GPU and turns the result into a model the app can
load. Everything stays Apache-2.0 / MIT / BSD; no Ultralytics or other AGPL code is involved.

```text
dataset folder ──prepare_dataset.py──► training\work\data      (COCO json, class list, licence lines)
COCO weights   ──train.py (D-FINE)───► training\work\runs\...  (best_stg2.pth / best_stg1.pth)
checkpoint     ──export_dfine_onnx.py► models\dfine_s_ppe.onnx (same format as the shipped models)
ONNX model     ──evaluate_onnx.py────► accuracy per class + a ready-to-review licence record
```

All commands are for **PowerShell**, run from the project folder. Training has its **own
environment** (`training\.venv`, with PyTorch). Keep the app's `.venv` for running the app.

## 0. Before you start: the dataset and its licence

You need a PPE dataset with the same class names the app understands: `Person`, `Hardhat`,
`NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Mask`, `NO-Mask` (extra classes such as
`Safety Cone`, `machinery` and `vehicle` are fine).

The suggested one is Roboflow Universe **"Construction Site Safety"** (version 27, those 10 classes,
which match the original `ppe.pt`). It is licensed **CC BY 4.0**: you may use it commercially, but
you must credit it, and **whether Addsoft accepts CC BY 4.0 data is your decision**. Download it
(a free Roboflow account is needed) as **YOLOv8** or **COCO JSON**, and unzip it, for example to
`C:\Users\ATL\Desktop\ppe-data\construction-site-safety`. The folder should contain `train`,
`valid` and `test` sub-folders (plus `data.yaml` for YOLO).

Your own dataset works too, in either of those two layouts.

## 1. Create the training environment (once)

```powershell
cd C:\Users\ATL\Desktop\ppe-detection
deactivate                                   # only if the app's (.venv) is active
py -3.12 -m venv training\.venv --prompt ppe-train
training\.venv\Scripts\Activate.ps1          # the prompt now starts with (ppe-train)
python -m pip install --upgrade pip
python -m pip install -r training\requirements-train.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `2.11.0+cu128 True NVIDIA GeForce RTX 5070`. The install downloads
about 3 GB (PyTorch with its own CUDA libraries). Use `python -m pip`, not `pip`, because your
Application Control policy blocks `pip.exe`.

(Command Prompt: activate with `training\.venv\Scripts\activate.bat`; the other commands are the same.)

## 2. Download D-FINE and its starting weights (once)

```powershell
python training\setup_dfine.py
```

This puts D-FINE's source at the audited commit `956d170` in `training\D-FINE` and the COCO
weights in `training\weights`, each checked against its recorded SHA-256 (a mismatching download
is refused, not saved). It uses `git` if installed, otherwise GitHub's zip for that commit.

## 3. Check and convert the dataset

```powershell
python training\prepare_dataset.py --dataset C:\Users\ATL\Desktop\ppe-data\construction-site-safety
```

It prints a table of boxes per class and split, any problems (missing labels, boxes outside the
image, unknown class ids), whether the app's per-person PPE checks will switch on with these class
names, and the licence lines it found in the dataset's README. Read that output before training.

## 4. Train

First a 2-epoch trial run, to check everything works on your machine (a few minutes):

```powershell
python training\train.py --variant s --epochs 2
```

Then the real run (D-FINE-S, 100 epochs, batch 8):

```powershell
python training\train.py --variant s
```

- The time per epoch is printed; multiply by 100 for the total. Leave the PC on.
- **CUDA out of memory**: use `--batch 4`.
- **Errors mentioning pickling or worker processes**: add `--workers 0`.
- **Interrupted** (closed window, reboot): `python training\train.py --variant s --resume`.
- Training curves: `python -m tensorboard.main --logdir training\work\runs`, then open
  http://localhost:6006.
- `--variant n` trains the smaller, faster D-FINE-N instead.

D-FINE checks accuracy on the `valid` split after every epoch and keeps the best checkpoint. When
it finishes, `train.py` prints the exact export command for your paths.

## 5. Export to the app's format

Copy the command `train.py` printed. It looks like this:

```powershell
python scripts\export_dfine_onnx.py --dfine-repo training\D-FINE --config "...\training\work\dfine_s_ppe.yml" --checkpoint "...\best_stg2.pth" --labels-json "...\training\work\data\labels.json" --output "...\models\dfine_s_ppe.onnx" --verify-image "...jpg" --verify-image "...jpg"
```

It writes `models\dfine_s_ppe.onnx` and checks on two test images that the ONNX model gives the same
boxes and scores as PyTorch. It stops with an error if they differ.

## 6. Measure it

```powershell
python training\evaluate_onnx.py --model models\dfine_s_ppe.onnx --device cuda
```

This runs the ONNX model through the app's own detector code on every `test` image and prints,
per class, AP / AP50 and, at the app's confidence threshold (0.5), precision, recall, missed and
false boxes. `--conf 0.4` shows another threshold. Pay most attention to `Hardhat` vs `NO-Hardhat`
and `Safety Vest` vs `NO-Safety Vest`: those decide violations.

It also writes `training\work\record_dfine_s_ppe.md`: the `SHA256SUMS` line and a draft
`MODEL_LICENSES.md` section with the dataset, lineage and the measured numbers.

## 7. Try it in the app (before approving)

Switch to the app's environment:

```powershell
deactivate
.venv\Scripts\Activate.ps1
python -m ppe_detection --model models\dfine_s_ppe.onnx --allow-unverified-model --source 0
```

With a PPE model the status bar shows per-person results (violation / compliant / unknown). Test
with and without a hardhat and a vest, in your real lighting and camera angles.

## 8. Approve it

When you accept the model:

1. Decide the licence of the fine-tuned weights (base weights and code are Apache-2.0; the
   dataset's terms, e.g. CC BY 4.0 attribution, also apply) and fill it in the draft record.
2. Paste the reviewed record from `training\work\record_dfine_s_ppe.md` into `MODEL_LICENSES.md`.
3. Add its `SHA256SUMS` line to `models\SHA256SUMS`.
4. If you use git, add `!models/dfine_s_ppe.onnx` to `.gitignore` next to the two COCO models.

From then on it loads without `--allow-unverified-model`:

```powershell
python -m ppe_detection --model models\dfine_s_ppe.onnx --source 0
```

Re-training produces a different file with a different SHA-256; the app will treat it as unapproved
until you repeat steps 6-8.

## Files

| Path | What it is | In git |
| --- | --- | --- |
| `requirements-train.txt` | pinned training environment (PyTorch 2.11.0 + CUDA 12.8) | yes |
| `setup_dfine.py` | fetches D-FINE `956d170` and the audited COCO weights | yes |
| `prepare_dataset.py` | checks a YOLO / COCO dataset and converts it | yes |
| `train.py` | writes the D-FINE config and runs D-FINE's training | yes |
| `evaluate_onnx.py` | accuracy of the exported model + draft licence record | yes |
| `D-FINE\`, `weights\` | downloaded by `setup_dfine.py` | no |
| `work\` | prepared data, config, checkpoints, logs, reports | no |

## What has been verified

On Linux, on the CPU, with a small synthetic dataset: setup, dataset conversion (YOLO and COCO),
a 2-epoch D-FINE-S fine-tune from the audited COCO weights, export with the parity checks, the
evaluation report, and the app loading the result (refused without approval, running with
`--allow-unverified-model`). That proves the steps connect; it says nothing about accuracy.
Not yet verified: training on a real PPE dataset, on a GPU, and on Windows. Report any error
together with the full command output.
