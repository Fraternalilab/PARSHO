# Simple Python analysis

`parsho.analyze()` runs the common workflow in one call: load channels, detect
cell masks with Cellpose, threshold puncta and optional nuclei, filter cells,
measure morphology/intensity, optionally calculate radial distributions, and
optionally save results. No pandas or notebook widgets are needed.

For the individual loading, thresholding, filtering, radial, plotting and export
functions used in the notebooks, see the [public API guide](api.md).

```python
from parsho import analyze

result = analyze(
    {"cells": "C3-sample.tif", "aggregates": "C2-sample.tif", "nuclei": "C1-sample.tif"},
    segmentation_channels="cells",
    signal_channels="aggregates",
    nucleus_channel="nuclei",
    detection="otsu",
    require_nucleus=True,
    radial=True,
    output_dir="results/sample",       # Must be a new directory.
    save_intermediates=True,
)
print("Cells measured:", len(result["retained_labels"]))
```

Files must be spatially aligned and have the same height and width. Inspect
saved masks and overlays before interpreting measurements. The first segmentation
loads the installed Cellpose-SAM model and may download its weights. GPU use is
automatic; add `gpu=False` to use CPU.

## Reuse settings for several images

`Analysis` is the reusable wrapper object. Its constructor accepts the same
analysis parameters as `analyze()`. One object keeps the Cellpose model loaded
between calls; raw images are not normalized for measurements.

```python
from parsho import Analysis

analysis = Analysis(
    segmentation_channels=["cells", "nuclei"],
    signal_channels="aggregates",
    nucleus_channel="nuclei",
    require_nucleus=True,
    remove_nuclear=True,
    radial=True,
)
for name in ["sample1", "sample2"]:
    result = analysis.run(
        {"cells": f"C3-{name}.tif", "aggregates": f"C2-{name}.tif", "nuclei": f"C1-{name}.tif"},
        image_id=name,
        output_dir=f"results/{name}",
        save_intermediates=True,
    )
```

The settings are explicit, so a nucleus channel can be omitted when unavailable.
Nuclear subtraction (`remove_nuclear`) and cell filtering (`require_nucleus`)
are independent choices. A nucleus-centred radial profile is omitted for a cell
without detected nuclear pixels; its other measurements remain unless filtered.

## Parameters and defaults

| `Analysis(...)` parameter | Default | Meaning |
| --- | --- | --- |
| `segmentation_channels` | `(0,)` | Channel name/index or list used to find cells |
| `signal_channels` | `(1,)` | Channel name/index or list measured as puncta |
| `nucleus_channel` | `None` | Optional nucleus stain |
| `transfection_channel` | `None` | Optional marker used to identify transfected cells |
| `detection` | `"otsu"` | Detection method, `DetectionSettings`, or mapping per measured channel |
| `nucleus_detection` | Otsu, minimum 5 pixels | Nucleus threshold settings |
| `transfection_detection` | Otsu, minimum 2 pixels | Marker threshold settings |
| `segmentation_method` | `"mean"` | Combine normalized segmentation copies: `"mean"`, `"maximum"`, `"stack"` (up to 3 channels) |
| `cellpose_parameters` | `{}` | Overrides for Cellpose's `eval()` |
| `gpu` | `None` | Auto-detect GPU; `False` forces CPU, `True` requests GPU |
| `model` | `None` | Optional existing Cellpose-compatible model |
| `require_nucleus` | `False` | Keep only cells overlapping detected nuclear pixels |
| `require_transfection` | `False` | Keep only cells overlapping the detected marker |
| `require_aggregates` | `False` | Keep only cells with puncta in at least one measured signal |
| `exclude_border` | `False` | Exclude cells touching the image border |
| `remove_nuclear` | `False` | Remove nuclear pixels from puncta and analyzed intensity |
| `radial` | `True` | Calculate radial measurements |
| `radial_center` | `"auto"` | `"nucleus"` if a nucleus channel was supplied, otherwise `"cell"`; either can be explicitly selected |
| `radial_bins` | `10` | Number of shape-adapted centre-to-boundary sections |
| `pixel_size_um` | `None` | Optional square-pixel width, for additional areas in µm² |

Cellpose evaluation defaults are `batch_size=8`, `diameter=None`,
`flow_threshold=0.4`, `cellprob_threshold=0.0`, `min_size=15`, and `normalize=False`.
Segmentation copies have already been normalized using their 1st/99th
percentiles. Cellpose axes are managed by the wrapper, and only 2-D segmentation
is supported. An all-zero cell segmentation raises an actionable error. A field
where all cells are filtered out returns empty measurements and a full filter
audit; exported measurement CSVs still contain column headers.

| `run(...)` / `analyze(...)` argument | Default | Meaning |
| --- | --- | --- |
| `channels` | Required | Named arrays/files, ordered arrays/files, or one multichannel file |
| `cell_masks` | `None` | Existing integer cell-label array or `.npy` path; skips Cellpose |
| `load_options` | `None` | `load_field_channels()` options: axes, scene, time, z_mode, z |
| `image_id` | `"sample"` | Image identifier recorded in settings and radial rows |
| `output_dir` | `None` | New destination directory; no files are written when omitted |
| `save_intermediates` | `False` | Save the notebook inspection figures |
| `save_radial_plots` | `True` | Include per-cell radial figures when saving inspection images |

`save_intermediates=True` requires `output_dir`. `save_radial_plots` has an effect
only when both intermediate export and radial analysis are enabled. Outputs are
never silently overwritten. To repeat an analysis, choose a new directory.

## Results and files

Both `Analysis` and Colab use the notebook output layout. For one measured
signal, an export with `save_intermediates=True` looks like:

```text
results/sample/
├── final_data.csv
├── radial_distribution.csv
└── inspection_images/
    ├── cell_segmentation.png
    ├── nuclear_mask.png
    ├── aggregate_mask.png
    ├── all_cells_overlay.png
    ├── transfected_cells.png
    └── radial_cell_<label>.png
```

- `final_data.csv` contains one row per retained cell and measured signal,
  including morphology, punctum counts, areas and intensities.
- `radial_distribution.csv` is written only when `radial=True`.
- `inspection_images/` is created only when `save_intermediates=True`.
  The nuclear-mask figure needs a nucleus channel; the segmentation figure
  needs segmentation input. `save_radial_plots=False` omits per-cell radial figures.

The filename `transfected_cells.png` follows the examples and shows cells
passing the selected filters, including when no transfection filter is used.
Original cell IDs link the tables and labelled figures. Empty results still
have CSV column headers. Areas are in pixels, with extra µm² columns when
`pixel_size_um` is supplied; intensities remain in the input image's units.

For multiple measured signals, the CSVs contain all signals in the `signal`
column. Their puncta masks, overlays and radial figures go in
`inspection_images/signal_01/`, `signal_02/`, etc., in selected signal order;
segmentation and nuclear figures are shared in `inspection_images/`.
