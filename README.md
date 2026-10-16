# PPE Detection

Real-time detection from a webcam, video or image using a **D-FINE** object detector exported to
**ONNX** and run on the **GPU** with **ONNX Runtime (CUDA)**. There is no PyTorch or Ultralytics at
runtime, and every model's licence is recorded in [MODEL_LICENSES.md](MODEL_LICENSES.md).

> **Current status: no PPE model is installed.** The approved models are Apache-2.0 D-FINE COCO
> detectors, so the app shows **people only** and says on screen that PPE compliance is not
> assessed. Once a D-FINE PPE model is fine-tuned, approved and placed in `models/`, the same app
> shows Hardhat / Mask / Safety Vest boxes, `NO-...` violations, and a per-person status.
> See [MODEL_LICENSES.md - Adding a PPE model](MODEL_LICENSES.md#adding-a-ppe-model).
>
> The original `ppe.pt` is kept unchanged for investigation, but it is UNVERIFIED and
> Ultralytics-derived, so the application refuses to load it.

## GPU-first design: what runs where

| Step | Runs on |
| --- | --- |
| BGR to RGB, resize to 640x640, scaling to 0-1 | **GPU**: these are operators inside the ONNX model |
| D-FINE network (backbone, encoder, decoder) | **GPU** (CUDA) |
| Top-300 selection and scaling boxes to frame pixels | **GPU**: inside the model |
| Reading webcam frames | CPU, necessarily: the camera driver delivers frames to system memory |
| Uploading each frame to the GPU | done by ONNX Runtime (one `uint8` copy, about 2.7 MB at 1280x720) |
| Turning the few confident boxes into labels, PPE association, drawing, window, video file | CPU: a few hundred numbers per frame. OpenCV's window and video writer run on the CPU, and the pip OpenCV build has no CUDA support |

CUDA is the default and is **required**: if the CUDA provider cannot start (no NVIDIA driver, wrong
CUDA version, blocked DLL), the app stops with a clear error instead of silently using the CPU.
The CPU is used only when you ask for it with `--device cpu` (e.g. on the team's machines without
a GPU). ONNX Runtime may still keep a handful of tiny shape-calculation operations on the CPU by
design; `scripts/check_gpu.py` shows exactly which operations ran where.

## Requirements

- Windows 10/11 x64 (Linux x64 also works).
- **Python 3.12.** The pinned set needs it: `numpy==2.5.3` requires Python >= 3.12, and
  `onnxruntime-gpu 1.30.0`, `onnxruntime 1.30.0` and `onnxruntime-directml 1.24.4` all publish
  `cp312-win_amd64` wheels. Every pinned package was checked for a Windows wheel.
- **An NVIDIA GPU with a driver that supports CUDA 13.x** (e.g. the RTX 5070). The CUDA 13 runtime
  and cuDNN 9 come from pip (`requirements.txt`); no CUDA Toolkit install is needed.
- A webcam for live mode.
- Alternatives: any DirectX 12 GPU via `requirements-directml.txt` (`--device directml`), or no GPU
  via `requirements-cpu.txt` (`--device cpu`).

## Setup

Run these from the project folder. Use `python -m pip` rather than `pip.exe`: some company
Application Control policies block executables inside a venv.

### PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt            # GPU (CUDA 13): downloads ~1.2 GB of NVIDIA runtime wheels
python -m pip install -r requirements-dev.txt        # test tools
python scripts\check_gpu.py                          # confirm the model really runs on the GPU
python -m pytest                                     # run the tests (GPU tests included)
python -m ppe_detection --list-cameras               # find your camera index
python -m ppe_detection --source 0                   # start the webcam app on the GPU
```

If `Activate.ps1` is blocked by the execution policy, skip activation and call the venv's
interpreter directly, e.g. `.\.venv\Scripts\python.exe -m ppe_detection --source 0`. Alternatively,
allow scripts for your user once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Command Prompt (cmd.exe)

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python scripts\check_gpu.py
python -m pytest
python -m ppe_detection --list-cameras
python -m ppe_detection --source 0
```

Without activating: `.venv\Scripts\python.exe -m ppe_detection --source 0`.

### Machines without an NVIDIA GPU

Use a fresh `.venv` and install `requirements-cpu.txt` (no GPU) or `requirements-directml.txt`
(AMD/Intel/other DirectX 12 GPUs, Windows only) **instead of** `requirements.txt`, then pass
`--device cpu` or `--device directml`. The `onnxruntime`, `onnxruntime-gpu` and
`onnxruntime-directml` packages conflict with each other, so never install two into one venv. Only
one OpenCV package (`opencv-python`) is used; do not add `opencv-contrib-python` or
`opencv-python-headless`.

## Checking that it runs on the GPU

```powershell
python scripts\check_gpu.py                                   # D-FINE-N, 50 frames of Samples\ppe1.mp4
python scripts\check_gpu.py --model models\dfine_s_coco.onnx --frames 100
```

It loads the model exactly as the app does, profiles real frames with ONNX Runtime's profiler, and
prints the session's provider, how many operations ran on CUDA and on the CPU (with their share of
the kernel time), which operation types stayed on the CPU, and the average time per frame. It exits
with `0` when at least 95 % of the kernel time is on the GPU, `1` when it is not, and `3` when the GPU
cannot be used at all (with the reason). The GPU tests run the same check (`python -m pytest -m gpu`).

## Usage

```powershell
python -m ppe_detection --source 0                       # webcam 0, on GPU 0
python -m ppe_detection --source Samples/ppe1.mp4        # video file
python -m ppe_detection --source 0 --conf 0.50           # confidence threshold
python -m ppe_detection --source 1 --width 1920 --height 1080
python -m ppe_detection --source 0 --model models/dfine_s_coco.onnx   # larger model
python -m ppe_detection --source 0 --gpu-id 1                         # second GPU
python -m ppe_detection --source 0 --output output/session.mp4        # also record
python -m ppe_detection --source photo.jpg --output output/photo_annotated.jpg
python -m ppe_detection --source Samples/ppe2.mp4 --no-display --output output/ppe2.mp4
python -m ppe_detection --source 0 --device cpu                       # deliberately on the CPU
python main.py --source 0                                # the old entry point still works
```

Press **q** or **Esc**, or close the window, to quit. Cameras, files, writers and the window are
released even if an error occurs.

| Option | Default | Meaning |
| --- | --- | --- |
| `--source` | `0` | Camera index, or a video/image path. Relative paths are tried from the current folder, then from the project folder |
| `--model` | `models/dfine_n_coco.onnx` | ONNX model, resolved relative to the **project folder** (not the terminal's folder) |
| `--device` | `cuda` | `cuda` (NVIDIA GPU, required: no silent CPU fallback), `directml` (other Windows GPUs), `cpu` (explicit CPU run), `auto` (best available, falls back to CPU with a warning) |
| `--gpu-id` | `0` | Which GPU to use with `cuda`/`directml` |
| `--conf` | `0.5` | Minimum confidence, greater than 0 and at most 1 |
| `--width`, `--height` | `1280`, `720` | Requested camera resolution; a warning shows if the camera delivers another size |
| `--backend` | `auto` | Camera API: `auto` (Windows: Media Foundation, then DirectShow), `msmf`, `dshow`, `v4l2`, `avfoundation`, `any` |
| `--fps` | `30` | Frame rate requested from the camera (it may deliver less; see `scripts\camera_test.py`) |
| `--fourcc` | none | Camera pixel format to request, e.g. `MJPG` |
| `--output` | none | Save the annotated result: `.mp4`/`.avi` for camera/video, an image suffix for image input |
| `--no-display` | off | No window (for batch processing with `--output`) |
| `--classes` | model default | Comma-separated class names to **draw**, or `all`. A PPE model shows everything; a COCO model shows `person`. Hidden classes still count in the person assessment |
| `--required-ppe` | `hardhat,vest` | Items a person must be seen wearing to count as compliant (`hardhat`, `vest`, `mask`). The app refuses items the model cannot detect |
| `--max-frames` | none | Stop after N frames |
| `--allow-unverified-model` | off | Run a model not listed in `models/SHA256SUMS` (marked UNVERIFIED on screen) |
| `--config` | none | TOML file with the same setting names (e.g. `conf = 0.6`, `source = 1`, `gpu_id = 0`, `output = "rec/site.mp4"`). Command-line options override it. Relative `model`, `output` and file `source` paths in it are relative to the TOML file; settings it omits keep their defaults |
| `--list-cameras` | | Probe camera indices 0-5 (`--max-camera-index` to change) with the same backends as `--source`, and exit |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `--version` | | Print the version and exit |

There is no IoU option: D-FINE predicts one box per object query and needs no non-maximum
suppression, so there is no IoU threshold to apply.

Exit codes: `0` success, `1` unexpected error (printed with a traceback), `2` invalid settings
(including an input file that does not exist), `3` model or GPU problem (missing, corrupt,
unapproved or incompatible model; CUDA unavailable or failed to start; ONNX Runtime cannot load),
`4` camera/video/image/output problem, `130` Ctrl+C.

### Troubleshooting

- `--device cuda needs CUDAExecutionProvider, but this ONNX Runtime installation offers only ...`:
  the venv has `requirements-cpu.txt` or `requirements-directml.txt`. Recreate it with `requirements.txt`.
- `CUDAExecutionProvider failed to initialise on GPU 0 ...`: update the NVIDIA driver (CUDA 13.x
  support is needed), check `--gpu-id`, and check for blocked DLLs (next point).
- `WinError 4551` or `DLL load failed`: a Windows Application Control policy blocked a DLL inside
  `.venv` (ONNX Runtime, the NVIDIA wheels, NumPy or OpenCV). Ask IT to allow it; the error names
  the blocked file.
- `Could not open camera N`: close Teams, Zoom or the Camera app, check *Settings > Privacy &
  security > Camera > Let desktop apps access your camera*, then run `--list-cameras`.
- The first few frames on the GPU are slow (CUDA/cuDNN choose their kernels on first use); the
  FPS shown settles after that.
- Low camera frame rate (the FPS shown is much lower than 1000 / inference ms): run
  `python scripts\camera_test.py`. It measures what the camera really delivers with each backend,
  with and without MJPG, at 1280x720 and 640x480, and you then pass the fastest row's options
  (`--backend`, `--fourcc`, `--width`/`--height`). On the tested office webcam, DirectShow gave 5
  FPS at 1280x720 and Media Foundation 30 FPS, which is why `auto` now tries Media Foundation first.
  Dim light also makes webcams lower their own frame rate.
- Video files in folders with non-English characters may fail to open on Windows; use a plain path.
- When recording from a webcam, the video is written at the camera's reported frame rate. If
  processing is slower than that, the recording plays back faster than real time.

## How PPE compliance is decided (when a PPE model is installed)

Boxes are always shown per object: `NO-Hardhat`, `NO-Mask` and `NO-Safety Vest` in red, worn PPE in
green. The app claims something about a **person** only through spatial association
([`ppe_detection/compliance.py`](ppe_detection/compliance.py)):

- a hardhat or mask counts for a person only if at least 50 % of its box lies in the top 40 % of
  that person's box; a vest only if it lies in the 15-80 % band of the box height;
- each item goes to at most one person (the one containing most of it);
- a person is **violation** if a required item is detected as missing on them, **compliant** only
  if every required item is detected on them, and **unknown** otherwise;
- items on nobody (e.g. a hardhat on a table) stay object-level and are counted as unattributed.

This is per-frame geometry, not tracking, and it can be wrong when people overlap heavily.

## Training a PPE model

The shipped COCO models find people but not PPE. [`training/README.md`](training/README.md) gives the
step-by-step commands to fine-tune D-FINE on a PPE dataset on this machine's GPU, export it in the
app's format, measure it, and approve it. Training uses its own environment (`training\.venv`, with
PyTorch); the app's environment stays as it is.

## Project structure

```text
ppe_detection/
  __init__.py       version, PROJECT_ROOT
  __main__.py       CLI, main loop, OpenCV window, exit codes
  config.py         settings, validation, TOML loading, path resolution
  camera.py         camera/video/image sources, camera discovery, output writer
  detector.py       ONNX Runtime GPU session, model approval (SHA-256), D-FINE output decoding
  compliance.py     person-to-PPE association and per-person status
  visualization.py  boxes, labels, status bar
models/             approved ONNX models (preprocessing built in), SHA256SUMS, README
scripts/            export_dfine_onnx.py (training environment), check_gpu.py, camera_test.py
training/           PPE fine-tuning kit: setup, dataset conversion, training, evaluation (own environment)
tests/              unit tests (mocked), real-model tests (CPU) and GPU tests
Samples/            sample videos (origin and licence unknown; internal testing only)
main.py             backward-compatible entry point
ppe.pt              original UNVERIFIED checkpoint: preserved on disk, never loaded, git-ignored
```

Per frame: `camera` reads a BGR frame; `detector` hands that exact `uint8` array to ONNX Runtime,
which uploads it to the GPU, where the model converts, resizes and scales it, runs D-FINE, selects
the top 300 boxes and scales them to frame pixels. Back on the CPU, `detector` keeps the confident
ones and names them from the class list stored in the model, `compliance` assesses people,
`visualization` draws, and the frame is shown and/or written. The model is loaded once at start-up.

## Tests

```powershell
python -m pytest                 # everything: 153 tests (12 need a CUDA GPU, 3 need the training packages)
python -m pytest -m "not model"  # unit tests only: mocked camera/session, no model, GPU or webcam
python -m pytest -m model        # real ONNX inference on frames from Samples/ (CPU, plus GPU if present)
python -m pytest -m gpu          # only the GPU tests
```

- **Mocked unit tests** cover class mapping, output parsing and confidence filtering, box clipping,
  the raw-frame hand-off (no CPU preprocessing), GPU index and provider selection, refusing a
  silent CPU fallback, missing/corrupt/unapproved/incompatible models, configuration and TOML
  validation, camera-open failures and discovery, cleanup after exceptions and Ctrl+C, CLI parsing
  and exit codes, the window's q/Esc/close handling (with OpenCV's GUI calls mocked) and the
  compliance rules.
- **Real-model tests** (`tests/test_real_model.py`) check the shipped models' checksums and
  metadata, run real inference on the sample videos on the CPU provider (people found, boxes in the
  right place, deterministic results), and run the CLI end-to-end on a video and an image.
- **GPU tests** (marker `gpu`, same file) check that the session really uses CUDA, that GPU and CPU
  find the same people (boxes within 3 px, scores within 0.02), that at least 95 % of kernel time
  is on the GPU, and run the CLI end-to-end on the GPU. They are skipped when no CUDA GPU works.

### What has and has not been verified

On the target Windows machine (RTX 5070, Python 3.12.10, `requirements.txt`), run by the user on
2026-10-15:

- `python scripts\check_gpu.py`: session on CUDAExecutionProvider; 743 operations (99.77 % of
  kernel time) on CUDA, 5 small shape operations (Unsqueeze x2, Gather, Tile, Cast; 0.23 %) on the
  CPU; 19.2 ms per 1280x720 frame for D-FINE-N, including upload and preprocessing.
- `python -m pytest`: all 148 tests of that version pass, including the 12 GPU tests for both
  models (the 5 training-kit tests were added afterwards).
- Live webcam (`--source 0`, Media Foundation): 33.6 FPS displayed at 1280x720, inference 12.4 ms
  on CUDA. (Through DirectShow the same camera delivered only 5 FPS at 1280x720, which is why
  `auto` tries Media Foundation first.)

In a Linux x64 / Python 3.12 environment without a GPU: both requirement sets install, the app
refuses to fall back to the CPU when CUDA cannot start (exit code 3), the ONNX models with
built-in preprocessing match PyTorch on real frames, and the OpenCV window displays correctly on a
virtual X display.

Training kit, in the same Linux environment on the CPU with a small synthetic dataset (this
checks the plumbing only, not accuracy): dataset conversion, a 2-epoch D-FINE-S fine-tune from the
audited COCO weights, ONNX export with PyTorch parity checks, evaluation, and the app running the
result. **Not verified yet:** training on a real PPE dataset, on a GPU, or on Windows.

**Not verified yet:** DirectML, `--fourcc` and `--fps` effects on other cameras, and real key
presses (the q/Esc logic is tested with OpenCV's GUI calls mocked).

The built-in resize is plain bilinear, as in OpenCV's `INTER_LINEAR`. D-FINE's own evaluation
pipeline uses PIL's antialiased resize. On three sample frames both found the same people, with
small confidence differences.
