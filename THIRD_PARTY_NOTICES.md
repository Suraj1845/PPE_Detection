# Third-party software notices

Licences below were read from each package's installed metadata (`License` / `License-Expression`
fields) and, for the Windows wheels, from the licence files inside the wheels. "Bundled" lists
binaries a wheel ships inside itself. No package in any runtime set is GPL- or AGPL-licensed; the
GNU-family items are LGPL libraries and a GCC runtime library with its linking exception, listed
explicitly below. **Ultralytics (AGPL-3.0) and cvzone are no longer used.**

Install exactly one runtime set.

## Runtime - common to every set

`requirements.txt` (default, NVIDIA GPU) uses `onnxruntime-gpu` plus the NVIDIA wheels listed in the
next section; `requirements-cpu.txt` uses `onnxruntime`; `requirements-directml.txt` uses
`onnxruntime-directml`. All three share the other packages in this table.

| Package | Version | Licence | Notes |
| --- | --- | --- | --- |
| onnxruntime (`requirements-cpu.txt`) | 1.30.0 | MIT | Ships `ThirdPartyNotices.txt`; its Mbed TLS component is used under Apache-2.0 (Mbed TLS is dual Apache-2.0 / GPL-2.0+) |
| opencv-python | 4.14.0.94 | MIT (packaging scripts), Apache-2.0 (OpenCV) | **Bundled on Windows:** `opencv_videoio_ffmpeg4140_64.dll` = FFmpeg under **LGPL-2.1**. Non-headless **Linux/macOS** wheels also bundle **Qt 5 (LGPL-3.0)** and other libraries listed in `cv2/LICENSE-3RD-PARTY.txt` |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | **Bundled on Windows:** OpenBLAS (BSD-3-Clause), LAPACK (BSD-3-Clause-Open-MPI), GCC runtime library inside the OpenBLAS DLL (**GPL-3.0-or-later WITH GCC-exception-3.1**, which permits use by non-GPL programs), `msvcp140` (Microsoft Visual C++ runtime) |
| protobuf | 7.36.2 | BSD-3-Clause | dependency of onnxruntime |
| flatbuffers | 25.12.19 | Apache-2.0 | dependency of onnxruntime |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | dependency of onnxruntime |

## Runtime - NVIDIA CUDA, the default (`requirements.txt`)

The common packages, with `onnxruntime-gpu` instead of `onnxruntime`, plus NVIDIA's runtime wheels.

| Package | Version | Licence |
| --- | --- | --- |
| onnxruntime-gpu | 1.30.0 | MIT |
| nvidia-cuda-runtime | 13.4.92 | NVIDIA proprietary EULA (`LicenseRef-NVIDIA-Proprietary`; text in the wheel's `License.txt`) |
| nvidia-cublas | 13.8.0.4 | `LicenseRef-NVIDIA-Proprietary` |
| nvidia-cuda-nvrtc | 13.4.92 | `LicenseRef-NVIDIA-Proprietary` |
| nvidia-cufft | 12.4.0.43 | `LicenseRef-NVIDIA-Proprietary` |
| nvidia-curand | 10.4.4.72 | `LicenseRef-NVIDIA-Proprietary` |
| nvidia-nvjitlink | 13.4.92 | `LicenseRef-NVIDIA-Proprietary` |
| nvidia-cudnn-cu13 | 9.27.0.42 | NVIDIA cuDNN licence (proprietary). PyPI metadata for this version states no licence expression; read the licence file in the installed wheel |

The NVIDIA wheels are free to download but are **not open source**. Review NVIDIA's terms before
redistributing them with a product.

## Runtime - DirectML (`requirements-directml.txt`, Windows only)

The common packages, with `onnxruntime-directml` instead of `onnxruntime`, plus:

| Package | Version | Licence | Notes |
| --- | --- | --- | --- |
| onnxruntime-directml | 1.24.4 | MIT | **Bundled:** `DirectML.dll`, Microsoft's DirectML redistributable under the [Microsoft Software License Terms](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4/License) (proprietary; may be distributed inside your own applications) |
| sympy | 1.14.0 | BSD | dependency of onnxruntime-directml |
| mpmath | 1.3.0 | BSD | dependency of sympy |

ONNX Runtime's `Privacy.md` (in the DirectML wheel) notes that official builds may use
platform-supported telemetry on Windows; see Microsoft's ONNX Runtime documentation to disable it.

## Tests only (`requirements-dev.txt`)

| Package | Version | Licence |
| --- | --- | --- |
| pytest | 9.1.1 | MIT |
| pluggy | 1.6.0 | MIT |
| iniconfig | 2.3.1 | MIT |
| pygments | 2.21.0 | BSD-2-Clause |
| colorama (Windows) | 0.4.6 | BSD (PyPI classifier "BSD License") |

## Training and model export only (separate environment, not part of the application)

Installed into `training\.venv` from `training/requirements-train.txt` and used by the training kit
(`training/`) and `scripts/export_dfine_onnx.py`. The application never imports them. The table lists
the packages the training and export code uses directly; the requirements file pins the rest.

| Package / project | Version | Licence |
| --- | --- | --- |
| D-FINE (source code) | commit `956d170` | Apache-2.0 |
| torch | 2.11.0 | BSD-3-Clause (Linux wheels also pull NVIDIA proprietary CUDA wheels) |
| torchvision | 0.26.0 | BSD |
| onnx | 1.23.2 | Apache-2.0 |
| onnxsim | 0.8.1 | MIT AND (Apache-2.0 OR BSD-2-Clause) |
| onnxruntime | 1.30.0 | MIT |
| pillow | 12.3.0 | MIT-CMU |
| PyYAML | 6.0.3 | MIT |
| scipy | 1.18.1 | BSD |
| faster-coco-eval | 1.8.0 | Apache-2.0 (LICENSE file in the wheel) |
| tensorboard | 2.21.0 | Apache-2.0 |
| loguru | 0.7.3 | MIT |
| calflops | 0.3.2 | MIT |
| transformers | 5.19.0 | Apache-2.0 |
| accelerate | 1.15.0 | Apache-2.0 |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based, BSD-compatible) |
| pycocotools | 2.0.11 | BSD-2-Clause ("FreeBSD" in its metadata) |

## Model weights

See [MODEL_LICENSES.md](MODEL_LICENSES.md).
