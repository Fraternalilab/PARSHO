"""General serialization helpers for Parsho analysis results."""

import csv
from collections.abc import Hashable, Mapping
from pathlib import Path

import numpy as np

from parsho.distribution import AggregateDistribution


def radial_distributions_to_records(
    distributions_by_image: Mapping[
        Hashable, Mapping[int, AggregateDistribution]
    ],
    metadata_by_image: Mapping[Hashable, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Convert radial distributions into one record per cell and radial bin.

    Args:
        distributions_by_image: Mapping from an image identifier to the result
            returned by one of Parsho's shape-adapted distribution functions.
        metadata_by_image: Optional metadata fields to repeat on every record
            belonging to an image. Metadata keys must not duplicate measurement
            column names.

    Returns:
        Long-form records suitable for CSV export or DataFrame construction.

    Raises:
        ValueError: If metadata contains a reserved measurement column.
    """
    metadata_by_image = metadata_by_image or {}
    reserved_columns = set(_radial_measurement_columns()) | {"image_id"}
    records = []

    for image_id, distributions in distributions_by_image.items():
        metadata = dict(metadata_by_image.get(image_id, {}))
        duplicates = reserved_columns.intersection(metadata)
        if duplicates:
            duplicate_names = ", ".join(sorted(duplicates))
            raise ValueError(f"Metadata uses reserved columns: {duplicate_names}")

        for distribution in distributions.values():
            cumulative_intensity = np.cumsum(distribution.intensity_share)
            bin_count = distribution.cell_pixels.size
            for radial_bin in range(bin_count):
                records.append(
                    {
                        "image_id": image_id,
                        **metadata,
                        "cell_label": distribution.label,
                        "centroid_row": distribution.centroid[0],
                        "centroid_column": distribution.centroid[1],
                        "radial_bin": radial_bin,
                        "radius_start": radial_bin / bin_count,
                        "radius_end": (radial_bin + 1) / bin_count,
                        "section_size_pixels": int(
                            distribution.cell_pixels[radial_bin]
                        ),
                        "aggregate_pixels": int(
                            distribution.aggregate_pixels[radial_bin]
                        ),
                        "aggregate_present": bool(
                            distribution.aggregate_presence[radial_bin]
                        ),
                        "aggregate_fraction": float(
                            distribution.aggregate_fraction[radial_bin]
                        ),
                        "aggregate_share": float(
                            distribution.aggregate_share[radial_bin]
                        ),
                        "intensity_sum": float(
                            distribution.intensity_sum[radial_bin]
                        ),
                        "intensity_mean": float(
                            distribution.intensity_mean[radial_bin]
                        ),
                        "intensity_share": float(
                            distribution.intensity_share[radial_bin]
                        ),
                        "cumulative_normalized_intensity": float(
                            cumulative_intensity[radial_bin]
                        ),
                    }
                )
    return records


def save_radial_distributions_csv(
    distributions_by_image: Mapping[
        Hashable, Mapping[int, AggregateDistribution]
    ],
    output_path: str | Path,
    metadata_by_image: Mapping[Hashable, Mapping[str, object]] | None = None,
) -> Path:
    """Save radial distributions as a long-form CSV file.

    Parent directories are created automatically. The file contains one row
    per image, cell, and radial bin.

    Returns:
        The resolved output path.
    """
    records = radial_distributions_to_records(
        distributions_by_image, metadata_by_image=metadata_by_image
    )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_columns = []
    if metadata_by_image:
        for metadata in metadata_by_image.values():
            for column in metadata:
                if column not in metadata_columns:
                    metadata_columns.append(column)
    fieldnames = [
        "image_id",
        *metadata_columns,
        *_radial_measurement_columns(),
    ]

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def _radial_measurement_columns() -> list[str]:
    """Return stable column order for radial-distribution CSV files."""
    return [
        "cell_label",
        "centroid_row",
        "centroid_column",
        "radial_bin",
        "radius_start",
        "radius_end",
        "section_size_pixels",
        "aggregate_pixels",
        "aggregate_present",
        "aggregate_fraction",
        "aggregate_share",
        "intensity_sum",
        "intensity_mean",
        "intensity_share",
        "cumulative_normalized_intensity",
    ]
