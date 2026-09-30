# Simple package tests

From the repository root, use a Python environment with PARSHO's dependencies
and `pytest` installed:

```bash
python -B -m pytest -c test/pytest.ini
```

The suite uses small synthetic images with known measurements. It checks TIFF
loading and time/Z selection, all four detection methods, object size filtering,
nucleus and transfection filtering, nuclear subtraction, physical area units,
cell- and nucleus-centred radial analysis, channel selection, external masks,
multiple signals, control calibration, CSV/figure exports, and invalid inputs.

Segmentation uses a small stand-in model to check how `Analysis` prepares
channels and reuses the model. These tests do not evaluate Cellpose's segmentation
accuracy or require model downloads, a GPU, or example data. Vendor file readers
and interactive notebook widgets are outside this suite's scope.

The configuration keeps temporary files in `tests/.tmp/` (recreated each run) and
plotting caches in `tests/.mplconfig/`. `-B` prevents Python bytecode writes, and
pytest's cache is disabled. No analysis output is written outside `tests/`.
