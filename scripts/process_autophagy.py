"""Analyze autophagy aggregates in aligned three-channel image sets."""

from pathlib import Path

import cellpose.utils as cellpose_utils
import numpy as np
import pandas as pd
from cellpose import core, models

from parsho.distribution import (
    compute_nucleus_centered_distribution,
    nucleus_centroids_by_cell,
)
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
    plot_nucleus_centered_distribution,
    plot_segmentation_result,
)
from parsho.segmentation import extract_masks
from parsho.utils import radial_distributions_to_records


PROJECT_DIR = Path("/path/to/data")
DATA_DIR = PROJECT_DIR
RESULTS_DIR = PROJECT_DIR / "results"
INSPECTION_DIR = RESULTS_DIR / "inspection_images"
CELL_CSV = RESULTS_DIR / "final_data.csv"
RADIAL_CSV = RESULTS_DIR / "radial_distribution.csv"

RADIAL_BINS = 10
AGGREGATE_MIN_SIZE = 2
NUCLEUS_MIN_SIZE = 5
REMOVE_NUCLEAR_SIGNAL_FROM_AGGREGATES = True

CELLPOSE_PARAMETERS = {
    "batch_size": 32,
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": -2.0,
    "min_size": 15,
    "normalize": {"tile_norm_blocksize": 0},
}

IMAGE_EXTENSIONS = {".tif", ".tiff"}


def load_image_set(data_dir: Path, name: str) -> tuple[np.ndarray, ...]:
    """Load cell, aggregate, and nuclear channels for one image set."""
    cell_channel = load_image(data_dir / f"C3-{name}")
    aggregate_channel = load_image(data_dir / f"C2-{name}")
    nuclei_channel = load_image(data_dir / f"C1-{name}")
    return cell_channel, aggregate_channel, nuclei_channel


def image_set_names(data_dir: Path) -> list[str]:
    """Return complete C1/C2/C3 image-set names in deterministic order."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    names = sorted(
        path.name.removeprefix("C1-")
        for path in data_dir.iterdir()
        if path.is_file()
        and path.name.startswith("C1-")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not names:
        raise FileNotFoundError(f"No C1 TIFF images found in {data_dir}")

    for name in names:
        missing = [
            prefix
            for prefix in ("C1-", "C2-", "C3-")
            if not (data_dir / f"{prefix}{name}").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Image set {name} is missing channels: {', '.join(missing)}"
            )
    return names


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
    """Run cell, aggregate, and radial analysis for every image set."""
    image_sets = image_set_names(DATA_DIR)
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    gpu_available = core.use_gpu()
    print(f"Using GPU: {gpu_available}")
    model = models.CellposeModel(gpu=gpu_available)

    cell_records = []
    radial_distributions = {}
    radial_metadata = {}
    cell_indices_by_image = {}

    for image_number, name in enumerate(image_sets, start=1):
        print(f"Set {image_number} of {len(image_sets)}: {name}")
        cell_channel, aggregate_channel, nuclei_channel = load_image_set(
            DATA_DIR, name
        )

        segmentation_input = np.stack([cell_channel, nuclei_channel], axis=0)
        cell_masks, cell_flows, _ = model.eval(
            segmentation_input, **CELLPOSE_PARAMETERS
        )
        if not np.any(cell_masks):
            print("  No cells detected; skipping image")
            continue

        agg_binary, agg_labels, _ = extract_masks(
            aggregate_channel=aggregate_channel,
            cell_masks=cell_masks,
            method="otsu",
            min_size_px=AGGREGATE_MIN_SIZE,
        )
        nuc_binary, nuc_labels, _ = extract_masks(
            aggregate_channel=nuclei_channel,
            cell_masks=cell_masks,
            method="otsu",
            min_size_px=NUCLEUS_MIN_SIZE,
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
            metrics.label: cell_number
            for cell_number, metrics in enumerate(metrics_for_image)
        }
        cell_indices_by_image[name] = cell_index_by_label

        for cell_number, metrics in enumerate(metrics_for_image):
            cell_records.append(
                {
                    "File Name": name,
                    "Img number": image_number,
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

        image_inspection_dir = INSPECTION_DIR / Path(name).stem
        save_inspection_figures(
            image_inspection_dir,
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

        radial_intensity = aggregate_channel.copy()
        if REMOVE_NUCLEAR_SIGNAL_FROM_AGGREGATES:
            radial_intensity[nuc_binary != 0] = 0

        nucleus_centroids = nucleus_centroids_by_cell(
            cell_masks, nuc_binary, cell_labels=retained_labels
        )
        distributions = compute_nucleus_centered_distribution(
            cell_masks=cell_masks,
            nucleus_centroids=nucleus_centroids,
            aggregate_mask=agg_binary,
            aggregate_channel=radial_intensity,
            radial_bins=RADIAL_BINS,
            cell_labels=list(nucleus_centroids),
        )
        radial_distributions[name] = distributions
        radial_metadata[name] = {
            "sample_id": name,
            "image_number": image_number,
        }

        for internal_label, distribution in distributions.items():
            cell_number = cell_index_by_label[internal_label]
            plot_nucleus_centered_distribution(
                cell_masks,
                agg_binary,
                radial_intensity,
                distribution,
                save_path=image_inspection_dir / f"radial_cell_{cell_number}.png",
                show=False,
                cell_name=f"Cell {cell_number}",
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cell_data = pd.DataFrame(cell_records)
    cell_data.to_csv(CELL_CSV, index=False)

    radial_records = radial_distributions_to_records(
        radial_distributions, metadata_by_image=radial_metadata
    )
    for record in radial_records:
        record["cell_label"] = cell_indices_by_image[record["image_id"]][
            record["cell_label"]
        ]
    radial_data = pd.DataFrame(radial_records)
    radial_data.to_csv(RADIAL_CSV, index=False)

    print(f"Saved {len(cell_data)} cell rows to {CELL_CSV}")
    print(f"Saved {len(radial_data)} radial rows to {RADIAL_CSV}")


if __name__ == "__main__":
    main()
