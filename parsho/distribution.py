"""Nucleus- or cell-centered, shape-adapted aggregate distributions.

The radial sections adapt to the segmented cell boundary and can use either
the nucleus centroid or the cell centroid as their origin.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

Centroid = tuple[float, float]


@dataclass(frozen=True)
class AggregateDistribution:
    """Aggregate measurements for the radial bins of one cell.

    All measurement arrays have shape ``(radial_bins,)``. Index zero is nearest
    the selected centroid and the last index is nearest the cell boundary.
    """

    label: int
    centroid: Centroid
    cell_pixels: np.ndarray
    aggregate_pixels: np.ndarray
    aggregate_presence: np.ndarray
    aggregate_fraction: np.ndarray
    aggregate_share: np.ndarray
    intensity_sum: np.ndarray
    intensity_mean: np.ndarray
    intensity_share: np.ndarray


def cell_centroids_by_cell(
    cell_masks: np.ndarray,
    cell_labels: Sequence[int] | None = None,
) -> dict[int, Centroid]:
    """Calculate the geometric centroid of each labeled cell.

    The centroid is the mean row and column coordinate of all pixels belonging
    to the cell. For a strongly concave cell this point can lie outside the
    cell mask; it is still a valid origin for the radial calculation.
    """
    placeholder = np.zeros(np.shape(cell_masks), dtype=np.float64)
    _validate_inputs(cell_masks, None, placeholder, radial_bins=1)
    labels = _resolve_cell_labels(cell_masks, cell_labels)
    centroids = {}
    for cell_label in labels:
        rows, columns = np.nonzero(cell_masks == cell_label)
        centroids[cell_label] = (float(rows.mean()), float(columns.mean()))
    return centroids


def nucleus_centroids_by_cell(
    cell_masks: np.ndarray,
    nucleus_mask: np.ndarray,
    cell_labels: Sequence[int] | None = None,
) -> dict[int, Centroid]:
    """Calculate one nucleus centroid for each nucleated cell.

    Nuclear pixels are assigned to cells by spatial overlap. Boolean, binary,
    0/255, and labeled nucleus masks are all accepted because every nonzero
    pixel is treated as nuclear signal. Cells without overlapping nuclear
    pixels are omitted from the returned mapping.

    Args:
        cell_masks: Two-dimensional label image with zero as background.
        nucleus_mask: Equally shaped Boolean, binary, or labeled nucleus mask.
        cell_labels: Optional subset of positive cell labels to inspect.

    Returns:
        Mapping from each nucleated cell label to its ``(row, column)`` nucleus
        centroid. If a geometric centroid rounds to a pixel outside its cell,
        it is moved to the closest overlapping nuclear pixel.

    Raises:
        ValueError: If the masks are malformed or a requested cell is absent.
    """
    placeholder = np.zeros(np.shape(cell_masks), dtype=np.float64)
    _validate_inputs(cell_masks, nucleus_mask, placeholder, radial_bins=1)

    available = {int(label) for label in np.unique(cell_masks) if label > 0}
    if cell_labels is None:
        labels = sorted(available)
    else:
        labels = [int(label) for label in cell_labels]
        missing = sorted(set(labels) - available)
        if missing:
            raise ValueError(f"Cell labels not present in cell_masks: {missing}")

    nuclear_pixels = nucleus_mask != 0
    centroids = {}
    for cell_label in labels:
        rows, columns = np.nonzero(
            (cell_masks == cell_label) & nuclear_pixels
        )
        if rows.size:
            row = float(rows.mean())
            column = float(columns.mean())
            nearest_row = int(np.rint(row))
            nearest_column = int(np.rint(column))

            if cell_masks[nearest_row, nearest_column] != cell_label:
                squared_distances = (rows - row) ** 2 + (columns - column) ** 2
                closest_pixel = int(np.argmin(squared_distances))
                row = float(rows[closest_pixel])
                column = float(columns[closest_pixel])

            centroids[cell_label] = (row, column)
    return centroids


def compute_cell_centered_distribution(
    cell_masks: np.ndarray,
    aggregate_mask: np.ndarray | None,
    aggregate_channel: np.ndarray,
    radial_bins: int = 10,
    cell_labels: Sequence[int] | None = None,
) -> dict[int, AggregateDistribution]:
    """Measure shape-adapted radial distributions from each cell centroid.

    All selected cells are analyzed, including cells without a nucleus mask.
    """
    centroids = cell_centroids_by_cell(cell_masks, cell_labels)
    return _compute_centered_distribution(
        cell_masks,
        centroids,
        aggregate_mask,
        aggregate_channel,
        radial_bins,
        cell_labels,
        require_centroid_inside_cell=False,
    )


def compute_nucleus_centered_distribution(
    cell_masks: np.ndarray,
    nucleus_centroids: Mapping[int, Sequence[float]],
    aggregate_mask: np.ndarray | None,
    aggregate_channel: np.ndarray,
    radial_bins: int = 10,
    cell_labels: Sequence[int] | None = None,
) -> dict[int, AggregateDistribution]:
    """Measure radial distributions from each cell's nucleus centroid.

    Rather than dividing a cell with concentric circles, this method constructs
    a relative-distance field. For every cell pixel, ``d_center`` is its
    Euclidean distance from the supplied nucleus centroid and ``d_boundary`` is
    its shortest distance from the cell-mask boundary. Its normalized radial
    position is:

    ``d_center / (d_center + d_boundary)``

    The result is zero near the nucleus and one at the cell boundary. Equal
    intervals of this field contract and expand with the cell mask, including
    in elongated or concave regions.

    Args:
        cell_masks: Two-dimensional label image with zero as background.
        nucleus_centroids: Mapping from cell label to ``(row, column)`` nucleus
            centroid. When ``cell_labels`` is omitted, only cells in this
            mapping are analyzed; cells without a nucleus centroid are skipped.
        aggregate_mask: Boolean or labeled aggregate mask. Nonzero pixels are
            treated as aggregates. Pass ``None`` to include every cell pixel.
        aggregate_channel: Two-dimensional aggregate-intensity image. Nuclear
            subtraction should be applied before passing this array if nuclear
            intensity must be excluded.
        radial_bins: Number of nucleus-to-boundary shape-adapted sections.
        cell_labels: Optional explicit subset. Every requested cell must have a
            supplied nucleus centroid.

    Returns:
        Per-cell measurements compatible with ``AggregateDistribution`` and
        Parsho's radial CSV utilities.

    Raises:
        ValueError: If inputs, labels, or nucleus centroids are invalid.
    """
    return _compute_centered_distribution(
        cell_masks,
        nucleus_centroids,
        aggregate_mask,
        aggregate_channel,
        radial_bins,
        cell_labels,
        require_centroid_inside_cell=True,
    )


def _compute_centered_distribution(
    cell_masks: np.ndarray,
    centroids: Mapping[int, Sequence[float]],
    aggregate_mask: np.ndarray | None,
    aggregate_channel: np.ndarray,
    radial_bins: int,
    cell_labels: Sequence[int] | None,
    require_centroid_inside_cell: bool,
) -> dict[int, AggregateDistribution]:
    """Shared implementation for nucleus- and cell-centered distributions."""
    _validate_inputs(cell_masks, aggregate_mask, aggregate_channel, radial_bins)
    labels = _resolve_labels(cell_masks, centroids, cell_labels)
    aggregate_pixels = (
        np.ones(cell_masks.shape, dtype=bool)
        if aggregate_mask is None
        else aggregate_mask != 0
    )
    intensity = np.asarray(aggregate_channel, dtype=np.float64)
    distributions = {}

    for cell_label in labels:
        centroid = _validate_centroid(
            centroids[cell_label],
            cell_masks,
            cell_label,
            require_inside_cell=require_centroid_inside_cell,
        )
        cell_region = cell_masks == cell_label
        rows, columns = np.nonzero(cell_region)
        section_ids = _shape_adapted_radial_bin_ids(
            cell_region, rows, columns, centroid, radial_bins
        )

        cell_counts = np.bincount(section_ids, minlength=radial_bins)
        is_aggregate = aggregate_pixels[rows, columns]
        aggregate_counts = np.bincount(
            section_ids, weights=is_aggregate, minlength=radial_bins
        ).astype(np.int64)
        intensity_sums = np.bincount(
            section_ids,
            weights=intensity[rows, columns] * is_aggregate,
            minlength=radial_bins,
        )
        aggregate_fraction = np.divide(
            aggregate_counts,
            cell_counts,
            out=np.zeros(radial_bins, dtype=np.float64),
            where=cell_counts > 0,
        )
        intensity_mean = np.divide(
            intensity_sums,
            aggregate_counts,
            out=np.zeros(radial_bins, dtype=np.float64),
            where=aggregate_counts > 0,
        )

        distributions[cell_label] = AggregateDistribution(
            label=cell_label,
            centroid=centroid,
            cell_pixels=cell_counts,
            aggregate_pixels=aggregate_counts,
            aggregate_presence=aggregate_counts > 0,
            aggregate_fraction=aggregate_fraction,
            aggregate_share=_normalize(aggregate_counts),
            intensity_sum=intensity_sums,
            intensity_mean=intensity_mean,
            intensity_share=_normalize(intensity_sums),
        )

    return distributions


def shape_adapted_radial_bin_map(
    cell_masks: np.ndarray,
    cell_label: int,
    centroid: Sequence[float],
    radial_bins: int = 10,
) -> np.ndarray:
    """Return a display map of shape-adapted bins for one labeled cell.

    Pixels outside the selected cell contain ``NaN``.
    """
    placeholder = np.zeros(cell_masks.shape, dtype=np.float64)
    _validate_inputs(cell_masks, None, placeholder, radial_bins)
    if cell_label <= 0 or not np.any(cell_masks == cell_label):
        raise ValueError(f"Cell label {cell_label} is not present in cell_masks")
    centroid = _validate_centroid(
        centroid, cell_masks, int(cell_label), require_inside_cell=False
    )
    cell_region = cell_masks == cell_label
    rows, columns = np.nonzero(cell_region)
    section_ids = _shape_adapted_radial_bin_ids(
        cell_region, rows, columns, centroid, radial_bins
    )
    bin_map = np.full(cell_masks.shape, np.nan)
    bin_map[rows, columns] = section_ids
    return bin_map


def _shape_adapted_radial_bin_ids(
    cell_region: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    centroid: Centroid,
    radial_bins: int,
) -> np.ndarray:
    """Assign cell pixels using centroid and boundary relative distances."""
    row_start, row_stop = rows.min(), rows.max() + 1
    column_start, column_stop = columns.min(), columns.max() + 1
    cropped_cell = cell_region[row_start:row_stop, column_start:column_stop]

    eroded_cell = binary_erosion(cropped_cell, border_value=0)
    boundary = cropped_cell & ~eroded_cell
    distance_to_boundary = distance_transform_edt(~boundary)
    local_rows = rows - row_start
    local_columns = columns - column_start
    boundary_distances = distance_to_boundary[local_rows, local_columns]

    center_distances = np.hypot(
        rows - centroid[0], columns - centroid[1]
    )
    denominator = center_distances + boundary_distances
    normalized_radius = np.divide(
        center_distances,
        denominator,
        out=np.zeros_like(center_distances, dtype=np.float64),
        where=denominator > 0,
    )
    normalized_radius[boundary_distances == 0] = 1.0
    radial_ids = np.floor(normalized_radius * radial_bins).astype(int)
    return np.clip(radial_ids, 0, radial_bins - 1)


def _resolve_labels(
    cell_masks: np.ndarray,
    centroids: Mapping[int, Sequence[float]],
    cell_labels: Sequence[int] | None,
) -> list[int]:
    """Resolve cells to analyze and require a centroid for each one."""
    available = {int(label) for label in np.unique(cell_masks) if label > 0}
    provided = {int(label) for label in centroids}
    unknown_centroids = sorted(provided - available)
    if unknown_centroids:
        raise ValueError(
            f"Centroids supplied for absent cells: {unknown_centroids}"
        )

    if cell_labels is None:
        return sorted(available & provided)

    labels = [int(label) for label in cell_labels]
    missing_cells = sorted(set(labels) - available)
    if missing_cells:
        raise ValueError(f"Cell labels not present in cell_masks: {missing_cells}")
    missing_centroids = sorted(set(labels) - provided)
    if missing_centroids:
        raise ValueError(
            f"Centroids not supplied for cells: {missing_centroids}"
        )
    return labels


def _resolve_cell_labels(
    cell_masks: np.ndarray,
    cell_labels: Sequence[int] | None,
) -> list[int]:
    """Resolve an optional subset against the positive cell-mask labels."""
    available = {int(label) for label in np.unique(cell_masks) if label > 0}
    if cell_labels is None:
        return sorted(available)
    labels = [int(label) for label in cell_labels]
    missing = sorted(set(labels) - available)
    if missing:
        raise ValueError(f"Cell labels not present in cell_masks: {missing}")
    return labels


def _validate_nucleus_centroid(
    centroid: Sequence[float],
    cell_masks: np.ndarray,
    cell_label: int,
) -> Centroid:
    """Validate and normalize a nucleus centroid for one cell."""
    return _validate_centroid(centroid, cell_masks, cell_label, True)


def _validate_centroid(
    centroid: Sequence[float],
    cell_masks: np.ndarray,
    cell_label: int,
    require_inside_cell: bool,
) -> Centroid:
    """Validate and normalize a centroid used as a radial origin."""
    if len(centroid) != 2:
        raise ValueError("Each centroid must contain (row, column)")
    row, column = float(centroid[0]), float(centroid[1])
    if not np.isfinite(row) or not np.isfinite(column):
        raise ValueError("Centroid coordinates must be finite")
    nearest_row, nearest_column = int(np.rint(row)), int(np.rint(column))
    height, width = cell_masks.shape
    if not (0 <= nearest_row < height and 0 <= nearest_column < width):
        raise ValueError(f"Centroid for cell {cell_label} is outside the image")
    if (
        require_inside_cell
        and cell_masks[nearest_row, nearest_column] != cell_label
    ):
        raise ValueError(f"Nucleus centroid is outside cell {cell_label}")
    return row, column


def _normalize(values: np.ndarray) -> np.ndarray:
    """Normalize values to sum to one, retaining zeros for empty input."""
    total = values.sum()
    if total == 0:
        return np.zeros(values.shape, dtype=np.float64)
    return values.astype(np.float64) / total


def _validate_inputs(
    cell_masks: np.ndarray,
    aggregate_mask: np.ndarray | None,
    aggregate_channel: np.ndarray,
    radial_bins: int,
) -> None:
    """Validate image geometry, cell labels, and the radial bin count."""
    if np.ndim(cell_masks) != 2 or np.ndim(aggregate_channel) != 2:
        raise ValueError("All distribution inputs must be two-dimensional")
    if cell_masks.shape != aggregate_channel.shape:
        raise ValueError("All distribution inputs must have the same shape")
    if aggregate_mask is not None:
        if np.ndim(aggregate_mask) != 2:
            raise ValueError("All distribution inputs must be two-dimensional")
        if aggregate_mask.shape != cell_masks.shape:
            raise ValueError("All distribution inputs must have the same shape")
    is_real_numeric = any(
        np.issubdtype(cell_masks.dtype, dtype)
        for dtype in (np.bool_, np.integer, np.floating)
    )
    if cell_masks.dtype.hasobject or not is_real_numeric:
        raise ValueError("cell_masks must have a real numeric dtype")
    if not np.all(np.isfinite(cell_masks)):
        raise ValueError("cell_masks must contain only finite values")
    if np.any(cell_masks < 0) or not np.all(cell_masks == np.floor(cell_masks)):
        raise ValueError("cell_masks must contain nonnegative integer labels")
    if isinstance(radial_bins, bool) or not isinstance(
        radial_bins, (int, np.integer)
    ):
        raise ValueError("radial_bins must be a positive integer")
    if radial_bins < 1:
        raise ValueError("radial_bins must be positive")
