"""Export the single-field Colab result with consistent cell identifiers."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

from parsho.plotting import plot_radial_distribution
from parsho.single_image import CELL_COLUMNS, FILTER_COLUMNS, OBJECT_COLUMNS, RADIAL_COLUMNS



OUTPUT_GUIDE = """PARSHO single-field results

cell_measurements.csv: one row per retained cell per measured signal.
aggregate_measurements.csv: one row per connected punctum in a retained cell.
radial_distribution.csv: one row per cell, signal and radial bin (if enabled).
cell_filtering.csv: every segmented cell, whether retained, and exclusion reasons.
summary.csv: totals per signal. Signal identities also appear in analysis_settings.json.

The cell_label is the ORIGINAL positive integer in cell_masks.tif and every
overlay/table/figure. It is never renumbered. Aggregate labels are local to a
signal. Objects crossing a cell boundary are split at that boundary; disconnected
fragments after nuclear exclusion count as separate puncta. The full masks retain
objects in excluded cells, while measurement tables contain retained cells only.

Areas ending in _pixels are pixel counts. _um2 columns are filled only when a
square-pixel width in micrometres was supplied. Centroid coordinates are zero-based
row/column pixel coordinates. Aspect ratio is major/minor axis length (infinity
for degenerate thin objects); circularity is 4*pi*area/perimeter^2. Aggregate shape
and size means are unweighted averages over puncta in that cell.

aggregate_coverage_fraction = aggregate area / whole cell area (the historical
PARSHO jaccard quantity). It is not a conventional overlap Jaccard coefficient.
aggregate_intensity_* use pixels inside the final detected puncta. cell_intensity
*_raw include all cell pixels before nuclear exclusion. cell_intensity_sum_analyzed
zeros nuclear pixels only when exclusion is enabled. nucleus_intensity_sum_raw is
the measured SIGNAL intensity over nuclear pixels, not the nuclear stain intensity.
Nuclear measurements are blank when no nucleus channel was supplied, and zero
only when a supplied nuclear channel contains no detected nuclear pixels.

Radial bin 0 is nearest the selected centre; larger bins approach the cell edge.
radius_start/end are normalized distances (0..1), not physical radii. Bins follow
the cell shape. section_size_pixels counts all cell pixels in the bin;
aggregate_fraction = puncta pixels / section pixels; aggregate_share = fraction
of this cell's puncta pixels in the bin. intensity_sum/mean use detected puncta
pixels only; intensity_share is the bin's fraction of total puncta intensity.
cumulative_normalized_intensity accumulates that share from centre to boundary.
Empty bins/cells have zero normalized shares. With nucleus centring, cells with
no detected nucleus have radial_included=False and no radial rows; they remain
in the cell table unless a filter removes them. The 'center' column records the
actual origin. Disabling radial analysis produces a header-only radial CSV.

signal_XX/ contains the raw selected/projected signal image, final binary and
label masks, analyzed intensity, threshold image (for local thresholds), overlays
and optional per-cell radial plots. Whole-image overlays show detected cells;
retained overlays show cell labels used in the tables. Blue = nucleus, red =
puncta, grey = cell boundary. Nucleus and puncta may overlap if exclusion is off.
The nucleus stain image/labels and transfection image/mask are saved if supplied.
cell_masks.tif and .npy preserve integer labels. Images are 2-D selections or
maximum Z projections at ONE time point. No time averaging or 3-D measurements
are performed. Display contrast does not rescale stored measurement intensities.

analysis_settings.json records file/channel assignments, scene/time/Z selection,
all controls and parameters, effective thresholds, and installed package versions.
Control files use the same raw intensity units and acquisition conditions as the
sample. A calibrated threshold is applied with the same minimum object size and
nuclear-exclusion rule used during calibration.
"""


def write_table(path, rows, columns):
    """Keep column headers even when no cells or objects pass the filters."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_overlay(path, image, cells, nucleus=None, aggregates=None, retained=None):
    """Save one readable, numbered overlay without Cellpose plotting dependencies."""
    fig, axis = plt.subplots(figsize=(9, 9))
    low, high = np.percentile(image, (1, 99))
    axis.imshow(image, cmap="gray", vmin=low, vmax=high if high > low else low + 1)
    selected_cells = cells if retained is None else np.where(np.isin(cells, retained), cells, 0)
    overlay = np.zeros((*cells.shape, 4), dtype=np.float32)
    if nucleus is not None:
        overlay[(nucleus > 0) & (selected_cells > 0)] = (0.1, 0.3, 1.0, 0.65)
    if aggregates is not None:
        overlay[(aggregates > 0) & (selected_cells > 0)] = (1.0, 0.1, 0.1, 0.65)
    overlay[find_boundaries(selected_cells, mode="inner")] = (0.8, 0.8, 0.8, 1.0)
    axis.imshow(overlay)
    for cell_id in np.unique(selected_cells):
        if cell_id:
            rows, columns = np.nonzero(selected_cells == cell_id)
            axis.text(columns.mean(), rows.mean(), str(cell_id), color="yellow", fontsize=8,
                      ha="center", va="center", bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"))
    axis.set_title("Cell labels / blue: nuclei / red: puncta")
    axis.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_field_result(output, result, signals, settings, *, nucleus=None, transfection=None,
                        segmentation_image=None, save_all_radial=True):
    """Write tables, raw arrays, figures and an output guide."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_table(output / "cell_measurements.csv", result["cell_records"], CELL_COLUMNS)
    write_table(output / "aggregate_measurements.csv", result["object_records"], OBJECT_COLUMNS)
    write_table(output / "cell_filtering.csv", result["filter_records"], FILTER_COLUMNS)
    write_table(output / "radial_distribution.csv", result["radial_records"], RADIAL_COLUMNS)
    cells = result["cells"]
    tifffile.imwrite(output / "cell_masks.tif", cells.astype(np.int32))
    np.save(output / "cell_masks.npy", cells.astype(np.int32))
    if segmentation_image is not None:
        tifffile.imwrite(output / "segmentation_input.tif", segmentation_image, photometric="minisblack")
    if nucleus is not None:
        tifffile.imwrite(output / "nucleus_image.tif", nucleus)
        tifffile.imwrite(output / "nucleus_labels.tif", result["nucleus_labels"].astype(np.int32))
    if transfection is not None:
        tifffile.imwrite(output / "transfection_image.tif", transfection)
        tifffile.imwrite(output / "transfection_mask.tif", result["transfection_mask"].astype(np.uint8))
    threshold_metadata = {}
    for kind in ("nucleus", "transfection"):
        value = result["thresholds"][kind]
        if isinstance(value, np.ndarray):
            filename = f"{kind}_threshold.tif"
            tifffile.imwrite(output / filename, value.astype(np.float32))
            threshold_metadata[kind] = filename
        else:
            threshold_metadata[kind] = None if value is None else float(value)
    summary, signal_exports = [], {}
    for index, (name, image) in enumerate(signals.items(), 1):
        folder = output / f"signal_{index:02d}"
        folder.mkdir()
        labels = result["signal_masks"][name]
        intensity = result["intensities"][name]
        tifffile.imwrite(folder / "signal_image.tif", image)
        tifffile.imwrite(folder / "aggregate_labels.tif", labels.astype(np.int32))
        tifffile.imwrite(folder / "aggregate_mask.tif", (labels > 0).astype(np.uint8))
        tifffile.imwrite(folder / "analyzed_intensity.tif", intensity)
        threshold = result["thresholds"]["signals"][name]
        if isinstance(threshold, np.ndarray):
            tifffile.imwrite(folder / "threshold.tif", threshold.astype(np.float32))
            threshold = "threshold.tif"
        else:
            threshold = float(threshold)
        signal_exports[name] = dict(folder=folder.name, effective_threshold=threshold)
        save_overlay(folder / "all_cells.png", image, cells, result["nucleus_labels"], labels)
        save_overlay(folder / "retained_cells.png", image, cells, result["nucleus_labels"], labels,
                     result["retained_labels"])
        if save_all_radial:
            for cell_id, distribution in result["distributions"][name].items():
                plot_radial_distribution(
                    cells, labels, intensity, distribution,
                    save_path=folder / f"radial_cell_{cell_id}.png", show=False,
                    cell_name=f"{name} / cell {cell_id}", center_label=result["center"].title(),
                )
        records = [row for row in result["cell_records"] if row["signal"] == name]
        summary.append(dict(signal=name, detected_cells=len(result["filter_records"]), retained_cells=len(records),
                            radial_cells=len(result["distributions"][name]),
                            aggregate_count=sum(row["aggregate_count"] for row in records),
                            aggregate_area_pixels=sum(row["aggregate_area_pixels"] for row in records),
                            aggregate_intensity_sum=sum(row["aggregate_intensity_sum"] for row in records)))
    write_table(output / "summary.csv", summary, list(summary[0]))

    (output / "READ_ME.txt").write_text(OUTPUT_GUIDE, encoding="utf-8")
    return output
