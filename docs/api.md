# Public API for notebook workflows

Use [the `Analysis` wrapper](analysis.md) for a complete workflow with a few
parameters. This reference covers the individual functions used in the
[local notebooks](../README.md#local-notebook-tutorials), so you can inspect or
replace one step at a time. Each section gives the import location, inputs,
defaults and returned values. Private functions and widget implementation
helpers are omitted.

| Task | Reference |
| --- | --- |
| Read images, choose channels, load existing masks | [Loading](#loading-images-and-masks) |
| Prepare Cellpose input and measure an already segmented field | [Segmentation and field analysis](#segmentation-and-field-analysis) |
| Detect puncta/nuclei or calibrate a threshold | [Thresholding](#thresholding) |
| Exclude nuclear pixels, retain cells, measure morphology | [Filtering and measurements](#filtering-and-cell-measurements) |
| Calculate cell- or nucleus-centred profiles | [Radial analysis](#radial-analysis) |
| Inspect masks, overlays and radial profiles | [Plotting](#plotting-and-overlays) |
| Save tables and intermediate files | [Export](#exporting-results) |
| Try the supplied images | [Example data](#example-data) |

All quantitative examples use raw intensities. Analysis inputs are spatially
aligned two-dimensional arrays with the same `(height, width)`. Cell masks
contain `0` for background and a distinct positive integer for each cell.
Binary signal masks indicate presence; labelled signal masks identify separate
objects. Positions use `(row, column)`, and Python channel/time/Z indices start
at **0**. Distances and areas are in pixels unless explicitly documented otherwise.

## Loading images and masks

Source: [parsho/img_utils.py](../parsho/img_utils.py). The four loading functions
below are also available directly from `parsho`.

### `load_image(input_path, scene=0)`

Read a supported path and return a NumPy array with its raw values and dtype.
`scene` selects a TIFF series or a vendor-container scene. TIFF/raster shapes
follow the stored layout; vendor containers return `TCZYX`. This function does
not select channels, time or depth, or normalize intensities. Use it directly
for the separate 2-D TIFF channels in the titin notebooks:

```python
from parsho import load_image

cell_channel = load_image("C3-sample.tif")
aggregate_channel = load_image("C2-sample.tif")
nuclei_channel = load_image("C1-sample.tif")
```

See [supported image formats](../README.md#supported-image-formats) for formats
and optional microscopy-reader dependencies.

### `load_field_channels(path, *, scene=0, axes="", time=0, z_mode="maximum", z=0)`

Return `(channels, metadata)`, where `channels` is a list of raw 2-D arrays.
The loader selects one time point, then maximum-projects Z (`z_mode="maximum"`)
or selects a plane (`z_mode="plane", z=...`). `axes=""` reads dimension metadata;
provide an explicit order such as `CYX`, `ZYX` or `TCZYX` for ambiguous stacks.
It never averages time points. Metadata records the filename, stored shape and
axes, resolved axes, and scene/time/Z selections.

```python
from parsho import load_field_channels

channels, metadata = load_field_channels(
    "sample.ome.tif", time=1, z_mode="plane", z=2,
)
nuclei_channel, aggregate_channel, cell_channel = channels[:3]
```

The example assumes the file's biological channel order is nucleus, aggregate,
cell; assign roles from your acquisition metadata. This function is also
importable from `parsho.single_image` and `parsho.img_utils`.

### `extract_channels(input_path, normalize=True, scene=0, *, axes="", time=0, z_mode="maximum", z=0)`

A convenience loader returning only the list of 2-D channels. It uses the same
scene/time/Z selection as `load_field_channels()`. Its historical default,
`normalize=True`, contrast-stretches each channel to `uint8` for display.
**Use `normalize=False` for measurements and control calibration.**

```python
from parsho import extract_channels

channels = extract_channels("sample.ome.tif", normalize=False)
```

### `load_mask_npy(input_path, kind="binary", expected_shape=None)`

Load a `.npy` mask with pickle loading disabled. Storage singleton dimensions
are removed, but a 2-D mask is required. `expected_shape=(height, width)` checks
alignment by shape. `kind="binary"` returns a Boolean array, treating every
nonzero value as foreground. `kind="labels"` validates nonnegative,
integer-valued labels and returns `int64` IDs. Use the latter for cell masks;
a binary mask does not distinguish individual cells.

```python
import numpy as np
from parsho import load_mask_npy

# Save a labelled array produced by your segmentation tool:
np.save("cell_masks.npy", cell_masks)
cell_masks = load_mask_npy(
    "cell_masks.npy", kind="labels", expected_shape=aggregate_channel.shape,
)
```

### Display contrast

`imagej_auto(img, saturate_pct=0.35)` returns integer `(low, high)` display
limits, excluding half the saturation percentage from each tail.
`apply_window(img, min_val, max_val)` clips and stretches those limits to
`uint8` values from 0 to 255. Import both from `parsho.img_utils` and use them
on display copies, preserving the original measurement arrays.

## Segmentation and field analysis

Source: [parsho/single_image.py](../parsho/single_image.py). `Analysis` and
`analyze()` are documented separately in [the wrapper guide](analysis.md).

### `combine_segmentation_channels(channels, method="mean")`

Import from `parsho` or `parsho.single_image`. Supply a list of aligned 2-D
arrays. The function copies each to floating point, normalizes using its 1st
and 99th percentiles, then combines them:

| Method | Returned shape | Behaviour |
| --- | --- | --- |
| `"mean"` | `(H, W)` | Mean of normalized channels |
| `"maximum"` | `(H, W)` | Maximum of normalized channels |
| `"stack"` | `(H, W, C)` | Preserve up to three channels in the supplied order |

Constant channels become zero. Original arrays are unchanged. Cellpose itself
provides cell segmentation; its model is not a PARSHO function. A notebook
can prepare and evaluate one field like this:

```python
from cellpose import core, models
from parsho import combine_segmentation_channels

segmentation_input = combine_segmentation_channels(
    [cell_channel, nuclei_channel], method="mean",
)
model = models.CellposeModel(gpu=core.use_gpu())
cell_masks, cell_flows, _ = model.eval(
    segmentation_input,
    channel_axis=-1 if segmentation_input.ndim == 3 else None,
    normalize=False,  # Segmentation copies were already normalized.
    batch_size=8,
    min_size=15,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
)
```

Keep `model` for subsequent images. Inspect its cell masks before proceeding.
Existing labelled masks from another tool can replace this entire step.

### `DetectionSettings(...)` and `detect_signal(image, cells, settings)`

Import `DetectionSettings` from `parsho`; import `detect_signal` from
`parsho.single_image`. The settings object holds validated threshold choices:

| Setting | Default | Applies to |
| --- | --- | --- |
| `method` | `"otsu"` | `"otsu"`, `"percentile"`, `"local"`, `"manual"` |
| `min_size` | `2` | Minimum connected object area, in pixels |
| `scale` | `1.0` | Otsu threshold multiplier |
| `percentile` | `95.0` | Percentile cutoff (0–100) |
| `block_size` | `51` | Local window width: odd integer, at least 3 |
| `manual_threshold` | `1000.0` | Fixed cutoff in raw image units |

`detect_signal()` returns `(binary_mask, labelled_mask, effective_threshold)`.
It clips detections to cells, splits objects at cell boundaries, and removes
objects smaller than `min_size`. The threshold is a scalar for global methods
and an `(H, W)` map for local detection. With scaled Otsu, the returned threshold
includes the multiplier. The settings' `.validate()` method can also be called
before starting an analysis.

```python
from parsho import DetectionSettings
from parsho.single_image import detect_signal

settings = DetectionSettings(method="manual", manual_threshold=1200, min_size=2)
agg_binary, agg_labels, threshold = detect_signal(
    aggregate_channel, cell_masks, settings,
)
```

### `analyze_field(cells, signals, *, ...)`

Import from `parsho` or `parsho.single_image`. This runs detection, cell
filtering, measurements and optional radial analysis on **existing cell labels**.
It neither loads images nor runs Cellpose nor writes files.

`cells` is a 2-D integer label array; `signals` maps unique names to raw 2-D
intensity arrays. `nucleus` and `transfection` accept raw intensity images,
not precomputed masks. The complete keyword options are:

| Argument | Default | Purpose |
| --- | --- | --- |
| `detection` | `None` | Mapping of every signal name to `DetectionSettings`; `None` uses Otsu/minimum 2 pixels |
| `nucleus`, `transfection` | `None` | Optional intensity channels |
| `nucleus_detection` | `None` | Defaults to Otsu/minimum 5 pixels |
| `transfection_detection` | `None` | Defaults to Otsu/minimum 2 pixels |
| `remove_nuclear` | `False` | Exclude nuclear pixels from puncta and analyzed intensity |
| `require_nucleus`, `require_transfection` | `False` | Retain only cells overlapping the specified detected marker |
| `require_aggregates` | `False` | Require puncta in **any** measured signal |
| `exclude_border` | `False` | Remove cells touching the image border |
| `radial` | `True` | Compute radial distributions |
| `radial_center` | `"auto"` | Nucleus when supplied, otherwise cell; also accepts `"cell"`/`"nucleus"` |
| `radial_bins` | `10` | Number of shape-adapted radial sections |
| `pixel_size_um` | `None` | Square-pixel width for optional areas in µm² |

```python
from parsho import DetectionSettings, analyze_field

signals = {"titin": aggregate_channel}
result = analyze_field(
    cell_masks,
    signals,
    detection={"titin": DetectionSettings(method="otsu", min_size=2)},
    nucleus=nuclei_channel,
    require_nucleus=True,
    remove_nuclear=True,
    radial=True,
    radial_bins=10,
)
```

The returned dictionary contains:

| Key | Contents |
| --- | --- |
| `cell_records` | One measurement dictionary per retained cell per signal |
| `object_records` | One dictionary per punctum in a retained cell |
| `filter_records` | Every segmented cell, marker presence, retention decision and reasons |
| `radial_records` | One dictionary per retained cell/signal/radial bin; `image_id` is `"sample"` |
| `retained_labels` | Original cell IDs retained for every signal |
| `cells`, `nucleus_labels`, `transfection_mask` | Cell labels, nuclear object labels and binary marker mask |
| `signal_masks`, `thresholded_signal_masks` | Per-signal labels after and before nuclear exclusion |
| `intensities` | Per-signal raw intensity copies with nuclear pixels zeroed when requested |
| `thresholds` | Effective signal, nucleus and marker thresholds; local thresholds are maps |
| `distributions` | Signal name → cell ID → `AggregateDistribution` |
| `center`, `analysis_options`, `detection_settings` | Actual radial centre and effective configuration |

Cell IDs are preserved throughout. Masks include detections in excluded cells;
measurement records include retained cells only. Missing nuclear measurements
are `None` when no nucleus channel was supplied; zero indicates a supplied
channel without detected nuclear pixels. Nuclear-centred radial analysis skips
cells with no detected nuclei without removing their other measurements unless
a filter does so. No detected cells raises `ValueError`; no cells passing the
filters produces empty measurement lists and a populated filtering audit.

`cell_records` and `object_records` include morphology, areas, counts and
intensity summaries. Aggregate intensities use final detected puncta pixels;
`cell_intensity_sum_raw` uses all raw pixels in the cell, while
`cell_intensity_sum_analyzed` reflects nuclear exclusion. The
`nucleus_intensity_sum_raw` value measures the **signal channel** in the nuclear
region, not the nuclear stain. Coverage is aggregate area divided by whole-cell
area. Aspect ratio is major/minor axis length; circularity is
`4 * pi * area / perimeter**2`. Additional area columns in µm² are filled only
when pixel size is supplied.

## Thresholding

Source: [parsho/segmentation.py](../parsho/segmentation.py). These functions are
used directly in the aggregate, autophagy and RNA-scope notebooks. They all
restrict signal to `cell_masks > 0` and use a strict intensity `>` comparison.

### `extract_masks(aggregate_channel, cell_masks, method="otsu", percentile=95.0, local_block_size=51, min_size_px=10, scale=1.0)`

Return `(binary_mask, labelled_mask, threshold)`. Despite its argument name,
`aggregate_channel` can be a nuclear stain or transfection marker as well as a
puncta channel. `method` accepts `"otsu"`, `"percentile"` or `"local"`.
Otsu and percentile cutoffs use intensities inside all cells; local thresholding
computes a spatial map from the image. Objects are split at cell boundaries
before the minimum-size filter is applied.

The direct function defaults to a **10-pixel** minimum; the wrapper defaults to
2 pixels for puncta and 5 for nuclei. Set the minimum explicitly to reproduce
a notebook workflow. For Otsu, `threshold` is the **unscaled** Otsu cutoff;
the mask uses `threshold * scale`. Use `detect_signal()` when you want the
returned value to be the effective scaled cutoff.

```python
from parsho.segmentation import extract_masks

agg_binary, agg_labels, agg_threshold = extract_masks(
    aggregate_channel, cell_masks, method="otsu", min_size_px=2,
)
nuc_binary, nuc_labels, nuc_threshold = extract_masks(
    nuclei_channel, cell_masks, method="otsu", min_size_px=5,
)
trans_binary, trans_labels, trans_threshold = extract_masks(
    aggregate_channel, cell_masks, method="otsu", min_size_px=2,
)
```

### `re_threshold_masks(aggregate_channel, cell_masks, min_size_px=10, thresh=0.0)`

Return `(binary_mask, labelled_mask)` using a fixed cutoff in original intensity
units. It applies the same cell-boundary splitting and minimum-size filter as
`extract_masks()`. Use it to apply a calibrated threshold or a manually chosen
number; `extract_masks(method="manual")` is not supported.

```python
from parsho.segmentation import re_threshold_masks

agg_binary, agg_labels = re_threshold_masks(
    aggregate_channel, cell_masks, min_size_px=2, thresh=1200,
)
```

### `find_optimal_threshold(pos_img_agregates, pos_masks, neg_img_agregates, neg_masks, pos_nuc_binary, neg_nuc_binary, t_min=None, t_max=None, verbose=False, min_size_px=2)`

Return `(threshold, positive_pixel_count)` for the first tested cutoff with
positive-control signal and no negative-control signal. Each control has its
own cell mask; controls need not have the same dimensions. Nuclear masks must
be Boolean arrays aligned with their respective controls. Supply all-False
nuclear masks to include nuclear signal. The function thresholds and applies
the size filter before removing nuclear pixels, matching the notebook workflow.
No additional size filter follows nuclear exclusion.

Both `t_min` and `t_max` must be supplied. The existing search tries **1,000
linearly spaced cutoffs from `t_min` to `10 * t_max`**; `t_max` is therefore
not the literal upper cutoff. It raises `ValueError` if positive signal is lost
or no tested cutoff separates the controls. Keep the minimum object size,
nuclear exclusion, intensity units and acquisition settings consistent with
the samples. `Analysis.calibrate_thresholds()` handles control preparation and
search bounds for you.

Given raw control images, their cell masks and Boolean nuclear masks:

```python
from parsho.segmentation import extract_masks, find_optimal_threshold

_, _, pos_otsu = extract_masks(pos_signal, pos_cells, min_size_px=2)
_, _, neg_otsu = extract_masks(neg_signal, neg_cells, min_size_px=2)
upper_cutoff = max(
    10 * max(pos_otsu, neg_otsu),
    pos_signal[pos_cells > 0].max(),
    neg_signal[neg_cells > 0].max(),
)
threshold, positive_pixels = find_optimal_threshold(
    pos_signal, pos_cells, neg_signal, neg_cells,
    pos_nuc_binary, neg_nuc_binary,
    t_min=min(pos_otsu, neg_otsu), t_max=upper_cutoff / 10,
    min_size_px=2,
)
agg_binary, agg_labels = re_threshold_masks(
    aggregate_channel, cell_masks, min_size_px=2, thresh=threshold,
)
```

## Filtering and cell measurements

Source: [parsho/maskfilters.py](../parsho/maskfilters.py).

### `subtract_nuclear_from_aggregate(agg_binary, agg_labels, nuc_binary)`

Return copies `(filtered_binary, filtered_labels)` with nuclear pixels removed.
Every nonzero nuclear-mask pixel is excluded. Disconnected fragments receive
separate object labels, while separate original objects remain distinct. There
is no second minimum-size filter. The input masks and intensity arrays are not
modified; zero nuclear pixels in an intensity **copy** separately when needed.

### `filter_cells_by_overlap(cell_masks, *label_masks)`

Return a list of cell IDs overlapping at least one nonzero pixel in **each**
supplied mask. One or more masks are required, all with the cell-mask shape.
This selects cells; it does not crop or relabel the input masks. In contrast,
`analyze_field(require_aggregates=True)` asks for puncta in *any* measured signal.

### `compute_cell_metrics(cell_masks, nuc_masks, agg_labels, transfected_labels)`

Return a list of `CellMetrics`, one per cell ID in `transfected_labels`.
The historical argument name accepts any retained-cell list; transfection is
not required. `nuc_masks` can be binary or labelled; `agg_labels` must label
individual objects for meaningful object shape measurements. If no nuclear
image exists, supply a zero nuclear mask; this low-level function then reports
zero nuclear area. Use `analyze_field()` when missing nuclear data should be
represented explicitly as `None`.

| `CellMetrics` field | Meaning |
| --- | --- |
| `label` | Original cell ID |
| `cell_area`, `nuc_area`, `agg_area` | Pixel counts in the cell, nuclear overlap and aggregate overlap |
| `jaccard` / property `aggregate_coverage_fraction` | Aggregate area / whole cell area; a coverage fraction, not a conventional Jaccard index |
| `cell_aspect_ratio`, `cell_circularity` | Cell shape measurements |
| `agg_aspect_ratio`, `agg_circularity` | Unweighted means over aggregate objects overlapping the cell; zero with no aggregates |

This function measures morphology, not intensity. Use `analyze_field()` for
per-cell/per-object intensity and count tables.

```python
from parsho.maskfilters import (
    subtract_nuclear_from_aggregate, filter_cells_by_overlap, compute_cell_metrics,
)

agg_binary, agg_labels = subtract_nuclear_from_aggregate(
    agg_binary, agg_labels, nuc_binary,
)
retained_labels = filter_cells_by_overlap(cell_masks, trans_labels, nuc_labels)
metrics = compute_cell_metrics(cell_masks, nuc_labels, agg_labels, retained_labels)
for cell in metrics:
    print(cell.label, cell.cell_area, cell.aggregate_coverage_fraction)

radial_intensity = aggregate_channel.copy()
radial_intensity[nuc_binary != 0] = 0
```

### `compute_shape_metrics(labeled)`

Return `{object_id: {"area": ..., "aspect_ratio": ..., "circularity": ...}}`
for each positive labelled region. Area is in pixels; aspect ratio is
major/minor axis length (`inf` when the minor axis is zero); circularity is
`4 * pi * area / perimeter**2` (zero for a zero perimeter). Aspect ratio and
circularity are rounded to three decimal places.

`ChannelMasks` is an optional storage dataclass for the cell labels and
aggregate/nuclear/transfection binary and label arrays. Constructing it does
not perform detection or filtering; the notebooks generally keep these arrays
in separate variables.

## Radial analysis

Source: [parsho/distribution.py](../parsho/distribution.py). The centroid and
distribution functions below, and `AggregateDistribution`, are also exported
from `parsho`.

### `cell_centroids_by_cell(cell_masks, cell_labels=None)`

Return `{cell_id: (row, column)}` geometric centroids. `cell_labels=None` selects
all positive labels; otherwise supply existing cell IDs. A concave cell's
centroid may fall outside its mask and is still valid for cell-centred analysis.

### `nucleus_centroids_by_cell(cell_masks, nucleus_mask, cell_labels=None)`

Return `{cell_id: (row, column)}` using all nuclear pixels overlapping each
selected cell. Any nonzero nuclear-mask value counts, including Boolean,
0/255 and labelled masks. Cells without nuclear pixels are omitted. For
multiple nuclear regions in one cell, their pixels contribute to one combined
centroid. If the centroid rounds outside its cell, it is moved to the nearest
overlapping nuclear pixel.

### `compute_cell_centered_distribution(cell_masks, aggregate_mask, aggregate_channel, radial_bins=10, cell_labels=None)`

Return `{cell_id: AggregateDistribution}` for all selected cells, centred on
their cell centroids. No nucleus is required.

### `compute_nucleus_centered_distribution(cell_masks, nucleus_centroids, aggregate_mask, aggregate_channel, radial_bins=10, cell_labels=None)`

Return the same mapping, centred on supplied nuclei. With `cell_labels=None`,
only cells in `nucleus_centroids` are analysed. Explicitly requested labels must
have supplied centroids, and each centroid must round to a pixel in its cell.
Use `nucleus_centroids_by_cell()` to prepare this mapping.

For both distribution functions, `aggregate_mask` may be binary or labelled;
nonzero pixels define puncta. `aggregate_mask=None` measures **all cell pixels**.
`aggregate_channel` supplies raw intensities, or the intensity copy with nuclear
pixels zeroed. Arrays must match the cell-mask shape. Neither function performs
thresholding, cell filtering or nuclear subtraction.

Radial sections adapt to the cell boundary using
`distance_to_center / (distance_to_center + distance_to_boundary)`.
`radial_bins` equal intervals cover this relative range; bin 0 is nearest the
centre and the last bin reaches the cell edge. These are shape-adapted sections,
not physical-distance annuli.

```python
from parsho import (
    nucleus_centroids_by_cell,
    compute_nucleus_centered_distribution,
    compute_cell_centered_distribution,
)

centroids = nucleus_centroids_by_cell(cell_masks, nuc_binary, retained_labels)
distributions = compute_nucleus_centered_distribution(
    cell_masks, centroids, agg_binary, radial_intensity,
    radial_bins=10, cell_labels=list(centroids),
)
# Alternative origin, using the same masks and measured intensity:
cell_distributions = compute_cell_centered_distribution(
    cell_masks, agg_binary, radial_intensity,
    radial_bins=10, cell_labels=retained_labels,
)
```

Each `AggregateDistribution` stores `label`, `centroid`, and one-dimensional
arrays of length `radial_bins`:

| Array | Per-bin meaning |
| --- | --- |
| `cell_pixels` | Number of cell pixels in the section |
| `aggregate_pixels`, `aggregate_presence` | Number of puncta pixels, and whether any are present |
| `aggregate_fraction` | Puncta pixels / section cell pixels |
| `aggregate_share` | Bin's share of all puncta pixels in this cell |
| `intensity_sum`, `intensity_mean` | Sum and mean over selected puncta pixels |
| `intensity_share` | Bin's share of total selected puncta intensity |

Empty sections have zero fractions/means; zero-total cells have zero shares.
For a cumulative intensity profile, use `np.cumsum(distribution.intensity_share)`.
With `aggregate_mask=None`, the “aggregate” quantities refer to all cell pixels.

### `shape_adapted_radial_bin_map(cell_masks, cell_label, centroid, radial_bins=10)`

Import from `parsho.distribution`. Return an `(H, W)` display array containing
bin IDs for the chosen cell and `NaN` outside it. Reuse a distribution's stored
centroid and bin count to reproduce exactly the bins used in its measurements:

```python
from parsho.distribution import shape_adapted_radial_bin_map

# Select a cell for which a distribution was returned.
cell_id = next(iter(distributions))
distribution = distributions[cell_id]
bin_map = shape_adapted_radial_bin_map(
    cell_masks, cell_id, distribution.centroid,
    radial_bins=len(distribution.cell_pixels),
)
```

## Plotting and overlays

Colab, the wrapper and the example notebooks/scripts share these renderers.
Categorical images use a fixed black/white/grey/blue/red palette with nearest
pixel interpolation, 5 × 5 inch figures and 300 dpi by default. Red labels sit
at the cell bounding-box corners. Raw signals and binary masks have separate
grayscale views; binary masks use a fixed black=False/white=True scale.
Segmentation diagnostics use four panels at 12 × 5 inches/300 dpi; radial
figures use their existing four-panel layout at 12 × 9 inches/150 dpi.
Defaults apply both when displaying figures and when exporting PNG/TIFF.

The `plot_*` functions display their result unless a save path is supplied,
except `plot_radial_distribution()`, which has a separate `show` flag, and
`plot_aggregate_channel(..., ax=...)`, which draws into an existing figure.
Create parent directories before saving individual figures. Display contrast
does not alter the arrays used for measurement.

### `build_overlay(cell_masks, nuc_labels, agg_labels, outlines=None)`

Import from `parsho.maskfilters`. Return a categorical array: background `0`,
cell interior `1`, cell outline `2`, nucleus `3`, aggregate `4`. Later categories
have priority at overlaps. It is neither an RGB image nor an object-label mask.
Omit `outlines` to calculate them automatically, or supply a precomputed mask.

`cell_outlines(cell_masks)`, also in `parsho.maskfilters`, returns a Boolean
external-contour mask matching the notebooks' Cellpose outlines. It preserves
boundaries between touching cells and along the image frame, and does not draw
internal holes. It needs no Cellpose/torch import. Existing notebook calls to
`cellpose.utils.masks_to_outlines(cell_masks)` remain compatible.

### `mask_overlay_to_cells(overlay, cell_masks, cell_labels)`

Import from `parsho.maskfilters`. Return an overlay copy with pixels outside the
selected cells set to zero. The compatibility name
`mask_overlay_to_transfected(overlay, cell_masks, transfected_labels)` performs
the same operation for any selected cell list.

```python
from cellpose.utils import masks_to_outlines
from parsho.maskfilters import build_overlay, mask_overlay_to_cells
from parsho.plotting import plot_aggregate_channel_color_labelled

overlay = build_overlay(cell_masks, nuc_labels, agg_labels, masks_to_outlines(cell_masks))
retained_overlay = mask_overlay_to_cells(overlay, cell_masks, retained_labels)
plot_aggregate_channel_color_labelled(
    retained_overlay, cell_masks,
    text_to_labell={label: str(label) for label in retained_labels},
)
```

### Mask and segmentation figures

Import the following from [parsho.plotting](../parsho/plotting.py). They display
or save figures and return `None`.

| Function | Input and commonly used options |
| --- | --- |
| `plot_segmentation_result(img_cell, masks, flows=None, save_path=None)` | Input image (YX, CYX or YXC), cell labels and optional complete `flows` return value from `model.eval()`; four input/outlines/masks/flow panels, stable colours per cell ID, and an unavailable-flow panel for external masks; no Cellpose import |
| `plot_aggregate_channel(img_agregates, save_path=None, colors=None, cmap="gray", dpi=300, *, ax=None)` | Raw signal or binary mask in grayscale; `ax` embeds a panel without displaying it; `colors` is retained for compatibility |
| `plot_aggregate_channel_color(img_agregates, save_path=None, colors=None, cmap=None, dpi=300)` | Categorical overlay, typically from `build_overlay()`; default colours are black/white/grey/blue/red; `cmap` is a compatibility argument |
| `plot_aggregate_channel_color_labelled(img_aggregates, cell_masks, text_to_labell, save_path=None, colors=None, dpi=300, text_color="red", text_size=7, text_box=False, offset=(2, 2))` | Categorical overlay with cell annotations; `text_to_labell` maps cell IDs to text, and `offset` positions text relative to the bounding box |

The historical spellings `img_agregates` and `text_to_labell` are part of the
current keyword API. Annotation text may differ from the cell ID, but it does
not change the mask or measurement IDs. PNG and TIFF exports use identical
styles; TIFF-only compression options are not passed to PNG or other formats.

### `plot_radial_distribution(cell_masks, aggregate_mask, aggregate_channel, distribution, save_path=None, dpi=150, show=True, cell_name=None, center_label="Nucleus")`

Import from `parsho.plotting`. Return `(figure, axes)` showing the shape-adapted
sections, selected cell's intensity image, puncta coverage and cumulative
normalized puncta intensity. `axes` contains `sections`, `channel`, `coverage`
and `intensity`. Use the same masks/intensity as the distribution calculation.
`cell_name` changes display text; `center_label="Cell"` labels a cell-centred
profile. The stored centroid defines the bins; changing the label does not
change the analysis. Set `show=False` when saving without opening figures.

```python
from pathlib import Path
from parsho.plotting import plot_radial_distribution

output = Path("results/manual_workflow")
output.mkdir(parents=True, exist_ok=True)
for cell_id, distribution in distributions.items():
    plot_radial_distribution(
        cell_masks, agg_binary, radial_intensity, distribution,
        save_path=output / f"radial_cell_{cell_id}.png",
        show=False, center_label="Nucleus", cell_name=f"Cell {cell_id}",
    )
```

`plot_nucleus_centered_distribution(*args, **kwargs)` is the compatibility name
used in the notebooks. It forwards to `plot_radial_distribution()` with the
same arguments and return value; its default centre label is `"Nucleus"`.

### `save_figure(fig, path, dpi=300, compression="tiff_lzw", bbox_inches="tight", pad_inches=0.05)`

Import from `parsho.plotting`. Save and close a Matplotlib figure; returns
`None`. The filename extension selects the output format. TIFF compression is
applied only for `.tif`/`.tiff`; PNG and other Matplotlib-supported formats such
as PDF use their normal writers. For example, after customizing a radial figure
returned without a save path, call `save_figure(figure, "radial.tif", dpi=150)`.

### `save_overlay(path, image, cells, nucleus=None, aggregates=None, retained=None, *, label_cells=True)`

Import from `parsho.colab_results`. Save the same solid categorical mask
overlay as the notebook plotting functions; returns `None`. Supply binary or
labelled nuclear/puncta arrays. `retained=None` displays every cell; a list of
IDs restricts the masks and annotations to those cells. Labels are the original
cell IDs in red at the bounding-box corners. `label_cells=False` produces an
unannotated overview. `image` remains in the signature for compatibility;
intensity is now shown in a separate grayscale figure using
`plot_aggregate_channel(image)`. No Cellpose plotting dependency is required.

```python
from parsho.colab_results import save_overlay

save_overlay(
    output / "retained_cells.png", aggregate_channel, cell_masks,
    nucleus=nuc_labels, aggregates=agg_labels, retained=retained_labels,
)
```

## Exporting results

### `radial_distributions_to_records(distributions_by_image, metadata_by_image=None)`

Import from `parsho` or [parsho.utils](../parsho/utils.py). Supply
`{image_id: {cell_id: AggregateDistribution}}`. Return a list of dictionaries,
one row per image/cell/radial bin, suitable for CSV writing or a pandas
DataFrame. `metadata_by_image` optionally supplies fields repeated for each
image, such as treatment or replicate. Metadata keys must not duplicate
measurement columns or `image_id`.

Rows contain `image_id`, `cell_label`, centroid coordinates, `radial_bin`,
normalized `radius_start`/`radius_end`, `section_size_pixels`,
`aggregate_pixels`, `aggregate_present`, `aggregate_fraction`, `aggregate_share`,
`intensity_sum`, `intensity_mean`, `intensity_share`, and
`cumulative_normalized_intensity`, plus any metadata fields.

### `save_radial_distributions_csv(distributions_by_image, output_path, metadata_by_image=None)`

Import from `parsho` or `parsho.utils`. Write the same long-form rows, create
parent directories, and return the resolved output `Path`. Empty distributions
still produce column headers. This direct CSV writer replaces an existing file;
choose a new path to preserve previous results.

```python
from parsho import radial_distributions_to_records, save_radial_distributions_csv

by_image = {"sample1": distributions}
metadata = {"sample1": {"condition": "treated", "replicate": 1}}
records = radial_distributions_to_records(by_image, metadata_by_image=metadata)
csv_path = save_radial_distributions_csv(
    by_image, output / "radial_distribution.csv", metadata_by_image=metadata,
)
```

### `export_field_result(output, result, signals, settings=None, *, nucleus=None, transfection=None, segmentation_image=None, save_all_radial=True, save_intermediates=True, segmentation_flows=None)`

Import from [parsho.colab_results](../parsho/colab_results.py). This works in
ordinary Python as well as Colab. Pass the result from `analyze_field()` and
the same signal mapping. It creates a **new** directory and returns its `Path`.

The output matches the examples: `final_data.csv`, `radial_distribution.csv`
when radial analysis is enabled, and optional figures in `inspection_images/`.
`save_intermediates=False` saves only CSVs; `save_all_radial=False` omits per-cell
radial figures. The direct exporter defaults to saving inspection figures;
`Analysis.run()` defaults to tables only.

Supply `segmentation_image` and optional `segmentation_flows` for the four-panel
segmentation figure. Nuclear masks come from the result. The legacy `settings`
and `transfection` arguments remain accepted but do not generate extra files.

```python
from parsho.colab_results import export_field_result

saved_dir = export_field_result(
    "results/field_export",  # Must not already exist.
    result, signals,
    segmentation_image=segmentation_input,
    segmentation_flows=cell_flows,
    save_intermediates=True,
    save_all_radial=True,
)
```

No ZIP is created by this function; Colab adds the download step. See
[Results and files](analysis.md#results-and-files) for filenames and grouping
when measuring multiple signals.

## Example data

Import from [parsho.examples](../parsho/examples.py):

- `example_data_dir()` finds the bundled aggregate TIFF directory in common
  repository locations and returns a `Path`. It raises `FileNotFoundError` if
  the data cannot be found locally.
- `demo_files(destination=None)` returns three file-path strings in **cell,
  aggregate, nucleus** order for `E3_115plate4.tif`. It uses local bundled data
  when available; otherwise it downloads the three TIFFs into the chosen
  destination, or a temporary directory when none is supplied.

```python
from parsho.examples import example_data_dir
from parsho import load_image

data_dir = example_data_dir()
aggregate_channel = load_image(data_dir / "C2-E3_115plate4.tif")
```
