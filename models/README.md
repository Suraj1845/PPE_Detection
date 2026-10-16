# Models

Only files whose SHA-256 is listed in `SHA256SUMS` are loaded by default. Licences and provenance
are recorded in [`../MODEL_LICENSES.md`](../MODEL_LICENSES.md).

| File | What it detects | Use |
| --- | --- | --- |
| `dfine_n_coco.onnx` | 80 COCO classes (the app shows only `person`) | Default (4M parameters) |
| `dfine_s_coco.onnx` | 80 COCO classes (the app shows only `person`) | More accurate, slower (10M parameters) |

Both are format version 2: they take one raw `uint8` BGR frame (`frame`, any size) and do the
colour conversion, 640x640 resize and scaling inside the model, so all of it runs on the GPU.

Neither model detects PPE. A PPE model must be fine-tuned from the D-FINE COCO checkpoints on a
licence-approved dataset, exported with `scripts/export_dfine_onnx.py` (which records the class
names in the ONNX metadata), reviewed, and then added to `SHA256SUMS` and `MODEL_LICENSES.md`.

`ppe.pt` in the project root is the original, **UNVERIFIED** Ultralytics checkpoint. It is kept for
investigation only and the application refuses to load it.

## Re-creating the ONNX files

The export runs in a separate environment with PyTorch; the application never needs PyTorch.

```powershell
git clone https://github.com/Peterande/D-FINE ..\D-FINE
git -C ..\D-FINE checkout 956d1709314c2c6a4df6f34de232054578a7449f
# download dfine_n_coco.pth / dfine_s_coco.pth from
# https://github.com/Peterande/storage/releases/tag/dfinev1.0
python scripts\export_dfine_onnx.py --dfine-repo ..\D-FINE --variant n `
    --checkpoint ..\weights\dfine_n_coco.pth --output models\dfine_n_coco.onnx `
    --verify-image C:\temp\frame1.jpg --verify-image C:\temp\frame2.jpg   # any 2+ real photos
```

The export environment needs `torch`, `torchvision`, `onnx`, `onnxsim`, `onnxruntime`, `pillow` and
D-FINE's own requirements (`pyyaml`, `scipy`, `faster-coco-eval`, `tensorboard`, `loguru`,
`calflops`, `transformers`). The script refuses checkpoints whose SHA-256 differs from the audited one.
With the same versions, the export here reproduced byte-identical ONNX files.
