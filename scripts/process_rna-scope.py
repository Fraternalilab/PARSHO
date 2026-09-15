"""Analyze TRPV1 and TRPA1 puncta in four-channel RNA-scope image sets."""

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


PROJECT_DIR = Path(
    "/media/ograciac/light-roast/ograciac/MorroCells/"
    "DatasetsParsho/RNA-scope"
)
DATA_DIR = PROJECT_DIR
RESULTS_DIR = PROJECT_DIR / "tutorial_results"
INSPECTION_DIR = RESULTS_DIR / "inspection_images"
CELL_CSV = RESULTS_DIR / "final_data.csv"
TRPV1_RADIAL_CSV = RESULTS_DIR / "trpv1_radial_distribution.csv"
TRPA1_RADIAL_CSV = RESULTS_DIR / "trpa1_radial_distribution.csv"

RADIAL_BINS = 10
TRPV1_MIN_SIZE = 1
TRPA1_MIN_SIZE = 2
NUCLEUS_MIN_SIZE = 5
REMOVE_NUCLEAR_SIGNAL_FROM_RNA = False

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
    """Load nucleus, TRPV1, TRPA1, and brightfield channels."""
    nuclei_channel = load_image(data_dir / f"C1-{name}")
    trpv1_channel = load_image(data_dir / f"C2-{name}")
    trpa1_channel = load_image(data_dir / f"C3-{name}")
    brightfield_channel = load_image(data_dir / f"C4-{name}")
    return nuclei_channel, trpv1_channel, trpa1_channel, brightfield_channel


def image_set_names(data_dir: Path) -> list[str]:
    """Return complete C1/C2/C3/C4 image sets."""
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
            for prefix in ("C1-", "C2-", "C3-", "C4-")
            if not (data_dir / f"{prefix}{name}").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Image set {name} is missing channels: {', '.join(missing)}"
            )
    return names


def save_inspection_figures(
    image_dir: Path,
    brightfield_channel: np.ndarray,
    cell_masks: np.ndarray,
    cell_flows,
    nuc_binary: np.ndarray,
    nuc_labels: np.ndarray,
    trpv1_binary: np.ndarray,
    trpv1_labels: np.ndarray,
    trpa1_binary: np.ndarray,
    trpa1_labels: np.ndarray,
    retained_labels: list[int],
    cell_index_by_label: dict[int, int],
) -> None:
    """Save segmentation and channel-specific quality-control figures."""
    image_dir.mkdir(parents=True, exist_ok=True)
    outlines = cellpose_utils.masks_to_outlines(cell_masks)

    trpv1_overlay = build_overlay(
        cell_masks, nuc_labels, trpv1_labels, outlines
    )
    trpv1_filtered_overlay = mask_overlay_to_transfected(
        trpv1_overlay, cell_masks, retained_labels
    )
    trpa1_overlay = build_overlay(
        cell_masks, nuc_labels, trpa1_labels, outlines
    )
    trpa1_filtered_overlay = mask_overlay_to_transfected(
        trpa1_overlay, cell_masks, retained_labels
    )

    plot_segmentation_result(
        brightfield_channel,
        cell_masks,
        cell_flows,
        save_path=image_dir / "cell_segmentation.png",
    )
    plot_aggregate_channel(
        trpv1_binary, save_path=image_dir / "trpv1_mask.png"
    )
    plot_aggregate_channel(
        trpa1_binary, save_path=image_dir / "trpa1_mask.png"
    )
    plot_aggregate_channel(
        nuc_binary, save_path=image_dir / "nuclear_mask.png"
    )
    plot_aggregate_channel_color(
        trpv1_overlay, save_path=image_dir / "trpv1_all_cells_overlay.png"
    )
    plot_aggregate_channel_color(
        trpa1_overlay, save_path=image_dir / "trpa1_all_cells_overlay.png"
    )
    plot_aggregate_channel_color_labelled(
        trpv1_filtered_overlay,
        cell_masks,
        cell_index_by_label,
        save_path=image_dir / "trpv1_retained_cells.png",
    )
    plot_aggregate_channel_color_labelled(
        trpa1_filtered_overlay,
        cell_masks,
        cell_index_by_label,
        save_path=image_dir / "trpa1_retained_cells.png",
    )


def main() -> None:
    """Run cell measurements and radial analysis for both RNA channels."""
    image_sets = image_set_names(DATA_DIR)
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

    gpu_available = core.use_gpu()
    print(f"Using GPU: {gpu_available}")
    model = models.CellposeModel(gpu=gpu_available)

    cell_records = []
    trpv1_radial_distributions = {}
    trpv1_radial_metadata = {}
    trpa1_radial_distributions = {}
    trpa1_radial_metadata = {}
    cell_indices_by_image = {}

    for image_number, name in enumerate(image_sets, start=1):
        print(f"Set {image_number} of {len(image_sets)}: {name}")
        nuclei_channel, trpv1_channel, trpa1_channel, brightfield_channel = (
            load_image_set(DATA_DIR, name)
        )

        cell_masks, cell_flows, _ = model.eval(
            brightfield_channel, **CELLPOSE_PARAMETERS
        )
        if not np.any(cell_masks):
            print("  No cells detected; skipping image")
            continue

        trpv1_binary, trpv1_labels, _ = extract_masks(
            aggregate_channel=trpv1_channel,
            cell_masks=cell_masks,
            method="otsu",
            min_size_px=TRPV1_MIN_SIZE,
        )
        trpa1_binary, trpa1_labels, _ = extract_masks(
            aggregate_channel=trpa1_channel,
            cell_masks=cell_masks,
            method="otsu",
            min_size_px=TRPA1_MIN_SIZE,
        )
        nuc_binary, nuc_labels, _ = extract_masks(
            aggregate_channel=nuclei_channel,
            cell_masks=cell_masks,
            method="otsu",
            min_size_px=NUCLEUS_MIN_SIZE,
        )

        if REMOVE_NUCLEAR_SIGNAL_FROM_RNA:
            trpv1_binary, trpv1_labels = subtract_nuclear_from_aggregate(
                trpv1_binary, trpv1_labels, nuc_binary
            )
            trpa1_binary, trpa1_labels = subtract_nuclear_from_aggregate(
                trpa1_binary, trpa1_labels, nuc_binary
            )

        retained_labels = filter_cells_by_overlap(cell_masks, nuc_labels)
        trpv1_metrics = compute_cell_metrics(
            cell_masks, nuc_labels, trpv1_labels, retained_labels
        )
        trpa1_metrics = compute_cell_metrics(
            cell_masks, nuc_labels, trpa1_labels, retained_labels
        )
        trpa1_metrics_by_label = {
            metrics.label: metrics for metrics in trpa1_metrics
        }
        cell_index_by_label = {
            metrics.label: cell_number
            for cell_number, metrics in enumerate(trpv1_metrics)
        }
        cell_indices_by_image[name] = cell_index_by_label

        for cell_number, trpv1_cell_metrics in enumerate(trpv1_metrics):
            trpa1_cell_metrics = trpa1_metrics_by_label[
                trpv1_cell_metrics.label
            ]
            cell_records.append(
                {
                    "File Name": name,
                    "Img number": image_number,
                    "cell num": cell_number,
                    "cell label": trpv1_cell_metrics.label,
                    "cell area": trpv1_cell_metrics.cell_area,
                    "trpv1 agg area": trpv1_cell_metrics.agg_area,
                    "trpa1 agg area": trpa1_cell_metrics.agg_area,
                    "nuc area": trpv1_cell_metrics.nuc_area,
                    "jaccard": trpv1_cell_metrics.jaccard,
                    "cell aspect ratio": trpv1_cell_metrics.cell_aspect_ratio,
                    "cell circularity": trpv1_cell_metrics.cell_circularity,
                    "trpv1 aspect ratio": trpv1_cell_metrics.agg_aspect_ratio,
                    "trpv1 circularity": trpv1_cell_metrics.agg_circularity,
                    "trpa1 aspect ratio": trpa1_cell_metrics.agg_aspect_ratio,
                    "trpa1 circularity": trpa1_cell_metrics.agg_circularity,
                }
            )

        image_inspection_dir = INSPECTION_DIR / Path(name).stem
        save_inspection_figures(
            image_inspection_dir,
            brightfield_channel,
            cell_masks,
            cell_flows,
            nuc_binary,
            nuc_labels,
            trpv1_binary,
            trpv1_labels,
            trpa1_binary,
            trpa1_labels,
            retained_labels,
            cell_index_by_label,
        )

        trpv1_radial_intensity = trpv1_channel.copy()
        trpa1_radial_intensity = trpa1_channel.copy()
        if REMOVE_NUCLEAR_SIGNAL_FROM_RNA:
            trpv1_radial_intensity[nuc_binary != 0] = 0
            trpa1_radial_intensity[nuc_binary != 0] = 0

        nucleus_centroids = nucleus_centroids_by_cell(
            cell_masks, nuc_binary, cell_labels=retained_labels
        )
        trpv1_distributions = compute_nucleus_centered_distribution(
            cell_masks=cell_masks,
            nucleus_centroids=nucleus_centroids,
            aggregate_mask=trpv1_binary,
            aggregate_channel=trpv1_radial_intensity,
            radial_bins=RADIAL_BINS,
            cell_labels=list(nucleus_centroids),
        )
        trpa1_distributions = compute_nucleus_centered_distribution(
            cell_masks=cell_masks,
            nucleus_centroids=nucleus_centroids,
            aggregate_mask=trpa1_binary,
            aggregate_channel=trpa1_radial_intensity,
            radial_bins=RADIAL_BINS,
            cell_labels=list(nucleus_centroids),
        )

        trpv1_radial_distributions[name] = trpv1_distributions
        trpv1_radial_metadata[name] = {
            "sample_id": name,
            "image_number": image_number,
        }
        trpa1_radial_distributions[name] = trpa1_distributions
        trpa1_radial_metadata[name] = {
            "sample_id": name,
            "image_number": image_number,
        }

        for internal_label, distribution in trpv1_distributions.items():
            cell_number = cell_index_by_label[internal_label]
            plot_nucleus_centered_distribution(
                cell_masks,
                trpv1_binary,
                trpv1_radial_intensity,
                distribution,
                save_path=(
                    image_inspection_dir
                    / f"trpv1_radial_cell_{cell_number}.png"
                ),
                show=False,
                cell_name=f"Cell {cell_number}",
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cell_data = pd.DataFrame(cell_records)
    cell_data.to_csv(CELL_CSV, index=False)

    trpv1_radial_records = radial_distributions_to_records(
        trpv1_radial_distributions,
        metadata_by_image=trpv1_radial_metadata,
    )
    trpa1_radial_records = radial_distributions_to_records(
        trpa1_radial_distributions,
        metadata_by_image=trpa1_radial_metadata,
    )
    for records in (trpv1_radial_records, trpa1_radial_records):
        for record in records:
            record["cell_label"] = cell_indices_by_image[record["image_id"]][
                record["cell_label"]
            ]

    trpv1_radial_data = pd.DataFrame(trpv1_radial_records)
    trpv1_radial_data.to_csv(TRPV1_RADIAL_CSV, index=False)
    trpa1_radial_data = pd.DataFrame(trpa1_radial_records)
    trpa1_radial_data.to_csv(TRPA1_RADIAL_CSV, index=False)

    print(f"Saved {len(cell_data)} cell rows to {CELL_CSV}")
    print(
        f"Saved {len(trpv1_radial_data)} radial rows to "
        f"{TRPV1_RADIAL_CSV}"
    )
    print(
        f"Saved {len(trpa1_radial_data)} radial rows to "
        f"{TRPA1_RADIAL_CSV}"
    )


if __name__ == "__main__":
    main()
