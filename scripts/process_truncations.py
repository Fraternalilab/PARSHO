"""Measure cells and aggregates in multichannel titin-truncation images.

Each immediate subfolder of ``DATA_DIR`` is one truncation condition. TIFF
channels are ordered as nucleus, membrane marker, and aggregate/transfection.
This workflow does not calculate radial distributions.
"""

from pathlib import Path

import cellpose.utils as cellpose_utils
import numpy as np
import pandas as pd
from cellpose import core, models

from parsho.img_utils import extract_channels
from parsho.maskfilters import (
    build_overlay,
    compute_cell_metrics,
    filter_cells_by_overlap,
    mask_overlay_to_transfected,
    subtract_nuclear_from_aggregate,
)
from parsho.plotting import (
    plot_aggregate_channel,
    plot_aggregate_channel_color,
    plot_aggregate_channel_color_labelled,
    plot_segmentation_result,
)
from parsho.segmentation import extract_masks


# Replace this with the directory containing the condition subdirectories.
DATA_DIR = Path("path/to/your/data")
RESULTS_DIR = DATA_DIR / "results"
INSPECTION_DIR = RESULTS_DIR / "inspection_images"
CELL_CSV = RESULTS_DIR / "final_data.csv"

NUCLEUS_MIN_SIZE = 5
AGGREGATE_MIN_SIZE = 2
REMOVE_NUCLEAR_SIGNAL_FROM_AGGREGATES = True

CELLPOSE_PARAMETERS = {
    "batch_size": 32,
    "diameter": None,
    "flow_threshold": 0.0,
    "cellprob_threshold": -2.0,
    "min_size": 15,
    "normalize": {"tile_norm_blocksize": 0},
}

IMAGE_EXTENSIONS = {".tif", ".tiff"}


def image_files_by_folder(data_dir: Path) -> list[tuple[Path, list[Path]]]:
    """Return TIFF files grouped by their immediate parent folder."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    grouped_files = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir() or folder == RESULTS_DIR:
            continue
        image_files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_files:
            grouped_files.append((folder, image_files))

    if not grouped_files:
        raise FileNotFoundError(f"No TIFF images found below {data_dir}")
    return grouped_files


def save_inspection_figures(
    image_dir: Path,
    cell_channel: np.ndarray,
    cell_masks: np.ndarray,
    cell_flows,
    nuc_binary: np.ndarray,
    nuc_labels: np.ndarray,
    agg_binary: np.ndarray,
    agg_labels: np.ndarray,
    retained_labels: list[int],
    cell_index_by_label: dict[int, int],
) -> None:
    """Save segmentation, mask, and overlay quality-control figures."""
    image_dir.mkdir(parents=True, exist_ok=True)
    outlines = cellpose_utils.masks_to_outlines(cell_masks)
    overlay = build_overlay(cell_masks, nuc_labels, agg_labels, outlines)
    filtered_overlay = mask_overlay_to_transfected(
        overlay, cell_masks, retained_labels
    )

    plot_segmentation_result(
        cell_channel,
        cell_masks,
        cell_flows,
        save_path=image_dir / "cell_segmentation.png",
    )
    plot_aggregate_channel(
        agg_binary, save_path=image_dir / "aggregate_mask.png"
    )
    plot_aggregate_channel(
        nuc_binary, save_path=image_dir / "nuclear_mask.png"
    )
    plot_aggregate_channel_color(
        overlay, save_path=image_dir / "all_cells_overlay.png"
    )
    plot_aggregate_channel_color_labelled(
        filtered_overlay,
        cell_masks,
        cell_index_by_label,
        save_path=image_dir / "retained_cells.png",
    )


def main() -> None:
    """Process each truncation folder and write one combined cell table."""
    grouped_files = image_files_by_folder(DATA_DIR)
    total_images = sum(len(image_files) for _, image_files in grouped_files)

    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    gpu_available = core.use_gpu()
    print(f"Using GPU: {gpu_available}")
    print(
        f"Found {total_images} images in {len(grouped_files)} "
        "truncation folders"
    )
    model = models.CellposeModel(gpu=gpu_available)

    cell_records = []
    processed_image_count = 0

    for folder_index, (data_folder, image_files) in enumerate(
        grouped_files, start=1
    ):
        condition = data_folder.name
        print(
            f"Folder {folder_index} of {len(grouped_files)}: "
            f"{condition} ({len(image_files)} images)"
        )

        for folder_image_number, image_path in enumerate(image_files, start=1):
            processed_image_count += 1
            print(
                f"  Image {folder_image_number} of {len(image_files)} "
                f"({processed_image_count} of {total_images} overall): "
                f"{image_path.name}"
            )

            channels = extract_channels(image_path, normalize=False)
            if len(channels) < 3:
                raise ValueError(
                    f"{image_path} contains {len(channels)} channels; "
                    "three are required"
                )
            nuclei_channel, aggregate_channel, cell_channel = channels[:3]

            cell_masks, cell_flows, _ = model.eval(
                cell_channel, **CELLPOSE_PARAMETERS
            )
            if not np.any(cell_masks):
                print("    No cells detected; skipping image")
                continue

            nuc_binary, nuc_labels, _ = extract_masks(
                aggregate_channel=nuclei_channel,
                cell_masks=cell_masks,
                method="otsu",
                min_size_px=NUCLEUS_MIN_SIZE,
            )
            agg_binary, agg_labels, _ = extract_masks(
                aggregate_channel=aggregate_channel,
                cell_masks=cell_masks,
                method="otsu",
                min_size_px=AGGREGATE_MIN_SIZE,
            )
            transfection_labels = agg_labels
            if REMOVE_NUCLEAR_SIGNAL_FROM_AGGREGATES:
                agg_binary, agg_labels = subtract_nuclear_from_aggregate(
                    agg_binary, agg_labels, nuc_binary
                )

            retained_labels = filter_cells_by_overlap(
                cell_masks, transfection_labels, nuc_labels
            )
            metrics_for_image = compute_cell_metrics(
                cell_masks, nuc_labels, agg_labels, retained_labels
            )
            cell_index_by_label = {
                metrics.label: cell_number
                for cell_number, metrics in enumerate(metrics_for_image)
            }

            for cell_number, metrics in enumerate(metrics_for_image):
                cell_records.append(
                    {
                        "truncation": condition,
                        "File Name": image_path.name,
                        "Img number": processed_image_count,
                        "folder image number": folder_image_number,
                        "cell num": cell_number,
                        "cell label": metrics.label,
                        "cell area": metrics.cell_area,
                        "agg area": metrics.agg_area,
                        "nuc area": metrics.nuc_area,
                        "jaccard": metrics.jaccard,
                        "cell aspect ratio": metrics.cell_aspect_ratio,
                        "cell circularity": metrics.cell_circularity,
                        "agg aspect ratio": metrics.agg_aspect_ratio,
                        "agg circularity": metrics.agg_circularity,
                    }
                )

            save_inspection_figures(
                INSPECTION_DIR / condition / image_path.stem,
                cell_channel,
                cell_masks,
                cell_flows,
                nuc_binary,
                nuc_labels,
                agg_binary,
                agg_labels,
                retained_labels,
                cell_index_by_label,
            )
            print(
                f"    Retained {len(retained_labels)} of "
                f"{int(cell_masks.max())} cells"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cell_data = pd.DataFrame(cell_records)
    cell_data.to_csv(CELL_CSV, index=False)
    print(f"Saved {len(cell_data)} cell rows to {CELL_CSV}")


if __name__ == "__main__":
    main()
