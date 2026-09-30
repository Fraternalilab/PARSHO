"""Save notebook-style tables and inspection figures for one field."""

import csv
from pathlib import Path

import numpy as np

from parsho.maskfilters import build_overlay, mask_overlay_to_cells
from parsho.plotting import (
    plot_aggregate_channel,
    plot_aggregate_channel_color,
    plot_aggregate_channel_color_labelled,
    plot_radial_distribution,
    plot_segmentation_result,
)
from parsho.single_image import CELL_COLUMNS, RADIAL_COLUMNS


def write_table(path, rows, columns):
    """Keep column headers even when no cells or objects pass the filters."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_overlay(path, image, cells, nucleus=None, aggregates=None, retained=None, *, label_cells=True):
    """Save the same categorical overlay used by the example notebooks.

    Colours are black background, white cells, grey outlines, blue nuclei and
    red puncta. Labels use the original cell IDs in red at the bounding-box
    corner. ``image`` is retained for call compatibility; raw intensities are
    displayed separately. Set label_cells=False for an unannotated overview.
    """
    empty = np.zeros(cells.shape, dtype=np.uint8)
    overlay = build_overlay(cells, empty if nucleus is None else nucleus,
                            empty if aggregates is None else aggregates)
    labels = [int(value) for value in np.unique(cells) if value > 0] if retained is None else retained
    if retained is not None:
        overlay = mask_overlay_to_cells(overlay, cells, labels)
    if label_cells:
        plot_aggregate_channel_color_labelled(
            overlay, cells, {label: str(label) for label in labels}, save_path=path,
        )
    else:
        plot_aggregate_channel_color(overlay, save_path=path)


def export_field_result(output, result, signals, settings=None, *, nucleus=None, transfection=None,
                        segmentation_image=None, save_all_radial=True, save_intermediates=True,
                        segmentation_flows=None):
    """Save final_data.csv, optional radial data, and notebook inspection figures.

    Figures go in inspection_images/. With multiple signals, their masks,
    overlays and radial figures go in signal_01/, signal_02/, etc. in input order.
    Set save_intermediates=False for tables only. Existing directories are
    never overwritten. Legacy settings/transfection arguments remain accepted;
    configuration and raw arrays are no longer exported.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_table(output / "final_data.csv", result["cell_records"], CELL_COLUMNS)
    if result["analysis_options"]["radial"]:
        write_table(output / "radial_distribution.csv", result["radial_records"], RADIAL_COLUMNS)
    if not save_intermediates:
        return output

    inspection = output / "inspection_images"
    inspection.mkdir()
    cells = result["cells"]
    if segmentation_image is not None:
        plot_segmentation_result(segmentation_image, cells, segmentation_flows,
                                 save_path=inspection / "cell_segmentation.png")
    if nucleus is not None or result["analysis_options"]["nucleus_supplied"]:
        plot_aggregate_channel(result["nucleus_labels"] > 0,
                               save_path=inspection / "nuclear_mask.png")
    for index, (name, image) in enumerate(signals.items(), 1):
        folder = inspection if len(signals) == 1 else inspection / f"signal_{index:02d}"
        folder.mkdir(exist_ok=True)
        labels = result["signal_masks"][name]
        plot_aggregate_channel(labels > 0, save_path=folder / "aggregate_mask.png")
        save_overlay(folder / "all_cells_overlay.png", image, cells,
                     result["nucleus_labels"], labels, label_cells=False)
        save_overlay(folder / "transfected_cells.png", image, cells,
                     result["nucleus_labels"], labels, result["retained_labels"])
        if save_all_radial:
            for cell_id, distribution in result["distributions"][name].items():
                plot_radial_distribution(
                    cells, labels, result["intensities"][name], distribution,
                    save_path=folder / f"radial_cell_{cell_id}.png", show=False,
                    cell_name=f"{name} / cell {cell_id}", center_label=result["center"].title(),
                )
    return output
