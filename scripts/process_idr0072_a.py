"""Measure cells and aggregates in every IDR0072_A image folder.

``SEGMENTATION_SOURCE`` selects the complete merged RGB image or raw ch01 for
Cellpose segmentation. Raw ch01 defines nuclei, and raw ch02 defines
aggregates and their intensity. This workflow does not calculate radial
distributions.
"""

from pathlib import Path

import cellpose.utils as cellpose_utils
import numpy as np
import pandas as pd
from cellpose import core, models

from parsho.img_utils import load_image
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


# Replace this with the directory containing the image-* subdirectories.
DATA_DIR = Path("path/to/your/data")
RESULTS_DIR = DATA_DIR / "results"
INSPECTION_DIR = RESULTS_DIR / "inspection_images"
CELL_CSV = RESULTS_DIR / "final_data.csv"

NUCLEUS_MIN_SIZE = 5
AGGREGATE_MIN_SIZE = 2
REMOVE_NUCLEAR_SIGNAL_FROM_AGGREGATES = True
SEGMENTATION_SOURCE = "merge"  # Use "merge" to use merged channel.

CELLPOSE_PARAMETERS = {
    "batch_size": 32,
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": -2.0,
    "min_size": 15,
    "normalize": {"tile_norm_blocksize": 0},
}


def image_folders(data_dir: Path) -> list[Path]:
    """Return all complete image folders in the dataset."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    folders = sorted(
        folder
        for folder in data_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("image-")
    )
    if not folders:
        raise FileNotFoundError(f"No image folders found in {data_dir}")

    for folder in folders:
        name = folder.name
        expected_paths = (
            folder / f"{name}_merged_z0_t0.jpg",
            folder / f"{name}_ch01_DRAQ5_z0_t0_raw.tif",
            folder / f"{name}_ch02_EGFP_z0_t0_raw.tif",
        )
        missing = [path.name for path in expected_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{folder} is missing required images: {', '.join(missing)}"
            )
    return folders


def load_image_set(
    image_folder: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the merged image and the two raw channels for one folder."""
    name = image_folder.name
    merged_image = load_image(image_folder / f"{name}_merged_z0_t0.jpg")
    nucleus_channel = load_image(
        image_folder / f"{name}_ch01_DRAQ5_z0_t0_raw.tif"
    )
    aggregate_channel = load_image(
        image_folder / f"{name}_ch02_EGFP_z0_t0_raw.tif"
    )

    if merged_image.ndim != 3 or merged_image.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected an RGB merged image in {image_folder}; "
            f"received shape {merged_image.shape}"
        )
    if merged_image.shape[:2] != nucleus_channel.shape:
        raise ValueError(
            f"Merged image and ch01 have different sizes in {image_folder}"
        )
    if nucleus_channel.shape != aggregate_channel.shape:
        raise ValueError(
            f"Raw ch01 and ch02 have different sizes in {image_folder}"
        )
    return merged_image, nucleus_channel, aggregate_channel


def save_inspection_figures(
    image_dir: Path,
    segmentation_image: np.ndarray,
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
        segmentation_image,
        cell_masks,
        cell_flows,
        save_path=image_dir / "cell_segmentation.png",
    )
    plot_aggregate_channel(
        nuc_binary, save_path=image_dir / "nuclear_mask.png"
    )
    plot_aggregate_channel(
        agg_binary, save_path=image_dir / "aggregate_mask.png"
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
    """Process every image folder and write one combined per-cell table."""
    if SEGMENTATION_SOURCE not in {"merge", "nucleus"}:
        raise ValueError(
            "SEGMENTATION_SOURCE must be either 'merge' or 'nucleus'"
        )

    folders = image_folders(DATA_DIR)
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    gpu_available = core.use_gpu()
    print(f"Using GPU: {gpu_available}")
    print(f"Found {len(folders)} image folders")
    model = models.CellposeModel(gpu=gpu_available)

    cell_records = []
    for image_number, image_folder in enumerate(folders, start=1):
        image_name = image_folder.name
        print(f"Image {image_number} of {len(folders)}: {image_name}")
        merged_image, nucleus_channel, aggregate_channel = load_image_set(
            image_folder
        )

        segmentation_image = (
            nucleus_channel
            if SEGMENTATION_SOURCE == "nucleus"
            else merged_image
        )
        segmentation_parameters = CELLPOSE_PARAMETERS.copy()
        if SEGMENTATION_SOURCE == "merge":
            segmentation_parameters["channel_axis"] = -1

        cell_masks, cell_flows, _ = model.eval(
            segmentation_image,
            **segmentation_parameters,
        )
        if not np.any(cell_masks):
            print("  No cells detected; skipping image")
            continue

        nuc_binary, nuc_labels, nucleus_threshold = extract_masks(
            aggregate_channel=nucleus_channel,
            cell_masks=cell_masks,
            method="otsu",
            min_size_px=NUCLEUS_MIN_SIZE,
        )
        agg_binary, agg_labels, aggregate_threshold = extract_masks(
            aggregate_channel=aggregate_channel,
            cell_masks=cell_masks,
            method="otsu",
            min_size_px=AGGREGATE_MIN_SIZE,
        )
        if REMOVE_NUCLEAR_SIGNAL_FROM_AGGREGATES:
            agg_binary, agg_labels = subtract_nuclear_from_aggregate(
                agg_binary, agg_labels, nuc_binary
            )

        retained_labels = filter_cells_by_overlap(cell_masks, nuc_labels)
        metrics_for_image = compute_cell_metrics(
            cell_masks, nuc_labels, agg_labels, retained_labels
        )
        cell_index_by_label = {
            metric.label: cell_number
            for cell_number, metric in enumerate(metrics_for_image)
        }

        for cell_number, metric in enumerate(metrics_for_image):
            aggregate_pixels = (
                (cell_masks == metric.label) & (agg_labels != 0)
            )
            aggregate_values = np.asarray(
                aggregate_channel[aggregate_pixels], dtype=np.float64
            )
            aggregate_intensity_sum = float(aggregate_values.sum())
            aggregate_intensity_mean = (
                float(aggregate_values.mean())
                if aggregate_values.size
                else 0.0
            )
            cell_records.append(
                {
                    "image folder": image_name,
                    "image number": image_number,
                    "cell number": cell_number,
                    "cell label": metric.label,
                    "cell area": metric.cell_area,
                    "aggregate area": metric.agg_area,
                    "nucleus area": metric.nuc_area,
                    "aggregate coverage fraction": metric.jaccard,
                    "aggregate intensity sum": aggregate_intensity_sum,
                    "aggregate intensity mean": aggregate_intensity_mean,
                    "cell aspect ratio": metric.cell_aspect_ratio,
                    "cell circularity": metric.cell_circularity,
                    "aggregate aspect ratio": metric.agg_aspect_ratio,
                    "aggregate circularity": metric.agg_circularity,
                    "nucleus otsu threshold": float(nucleus_threshold),
                    "aggregate otsu threshold": float(aggregate_threshold),
                }
            )

        save_inspection_figures(
            INSPECTION_DIR / image_name,
            segmentation_image,
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
            f"  Retained {len(retained_labels)} of "
            f"{int(cell_masks.max())} cells"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cell_data = pd.DataFrame(cell_records)
    cell_data.to_csv(CELL_CSV, index=False)
    print(f"Saved {len(cell_data)} cell rows to {CELL_CSV}")


if __name__ == "__main__":
    main()
