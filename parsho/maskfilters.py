"""Combine segmentation masks and calculate per-cell measurements."""

from dataclasses import dataclass

import numpy as np
from skimage.measure import regionprops


@dataclass
class ChannelMasks:
    """Holds all binary/label masks derived from a single image set."""

    cell_masks: np.ndarray  # Cellpose integer labels.
    agg_binary: np.ndarray  # Aggregate binary mask after filtering.
    agg_labels: np.ndarray  # Aggregate label mask after filtering.
    nuc_binary: np.ndarray  # Nuclear binary mask.
    nuc_labels: np.ndarray  # Nuclear label mask.
    trans_binary: np.ndarray  # Transfection binary mask.
    trans_labels: np.ndarray  # Transfection label mask.


@dataclass
class CellMetrics:
    """Per-cell measurement result."""

    label: int
    cell_area: int
    agg_area: int
    nuc_area: int
    jaccard: float
    cell_aspect_ratio: float
    cell_circularity: float
    agg_aspect_ratio: float
    agg_circularity: float

    @property
    def aggregate_coverage_fraction(self) -> float:
        """Fraction of cell pixels occupied by aggregates (legacy: jaccard)."""
        return self.jaccard


def subtract_nuclear_from_aggregate(
    agg_binary: np.ndarray,
    agg_labels: np.ndarray,
    nuc_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of the aggregate masks with nuclear pixels removed."""
    agg_labels = agg_labels.copy()
    agg_binary = agg_binary.copy()
    nuclear_pixels = nuc_binary != 0
    agg_labels[nuclear_pixels] = 0
    agg_binary[nuclear_pixels] = 0
    # Preserve separate original objects, but give disconnected fragments
    # distinct identities, matching the single-field object table.
    from skimage.measure import label

    return agg_binary, label(agg_labels)


def build_overlay(
    cell_masks: np.ndarray,
    nuc_labels: np.ndarray,
    agg_labels: np.ndarray,
    outlines: np.ndarray,
) -> np.ndarray:
    """Compose a categorical overlay image.

    Values identify background (0), cell interior (1), cell outline (2),
    nucleus (3), and aggregate (4). Later assignments take precedence in
    overlapping regions.
    """
    overlay = np.zeros_like(agg_labels)
    overlay[cell_masks > 0] = 1
    overlay[outlines > 0] = 2
    overlay[nuc_labels > 0] = 3
    overlay[agg_labels > 0] = 4
    return overlay


def filter_cells_by_overlap(
    cell_masks: np.ndarray,
    *label_masks: np.ndarray,
) -> list[int]:
    """Return cell labels that overlap every supplied label mask.

    A cell passes the filter when its region contains at least one non-zero
    pixel from each mask in ``label_masks``. All masks must have the same shape
    as ``cell_masks``, and at least one label mask must be supplied.
    """
    if not label_masks:
        raise ValueError("At least one label mask is required")

    cell_masks = np.asarray(cell_masks)
    label_masks = tuple(np.asarray(mask) for mask in label_masks)
    for index, mask in enumerate(label_masks, start=1):
        if mask.shape != cell_masks.shape:
            raise ValueError(
                f"Label mask {index} has shape {mask.shape}; "
                f"expected {cell_masks.shape}"
            )

    retained_labels = []
    for label in np.unique(cell_masks):
        if label == 0:
            continue
        cell_region = cell_masks == label
        if all(np.any(mask[cell_region]) for mask in label_masks):
            retained_labels.append(int(label))
    return retained_labels


def mask_overlay_to_transfected(
    overlay: np.ndarray,
    cell_masks: np.ndarray,
    transfected_labels: list[int],
) -> np.ndarray:
    """Compatibility name for :func:`mask_overlay_to_cells`."""
    return mask_overlay_to_cells(overlay, cell_masks, transfected_labels)


def mask_overlay_to_cells(overlay, cell_masks, cell_labels):
    """Copy an overlay, retaining only the explicitly selected cell IDs.

    Selection can represent any filter; it need not indicate transfection.
    """
    overlay = overlay.copy()
    final_cells = np.zeros_like(cell_masks)
    for label in cell_labels:
        final_cells[cell_masks == label] = 1
    overlay[final_cells == 0] = 0
    return overlay


def compute_shape_metrics(labeled: np.ndarray) -> dict[int, dict[str, float]]:
    """Return area, aspect ratio, and circularity for each labeled object.

    Aspect ratio is ``major_axis / minor_axis`` and circularity is
    ``4π * area / perimeter²``.
    """

    metrics = {}
    for region in regionprops(labeled):
        major = region.axis_major_length
        minor = region.axis_minor_length
        aspect_ratio = (major / minor) if minor > 0 else float("inf")
        circularity = (
            (4 * np.pi * region.area) / (region.perimeter**2)
            if region.perimeter > 0
            else 0
        )
        metrics[region.label] = {
            "area": region.area,
            "aspect_ratio": round(aspect_ratio, 3),
            "circularity": round(circularity, 3),
        }
    return metrics


def compute_cell_metrics(
    cell_masks: np.ndarray,
    nuc_masks: np.ndarray,
    agg_labels: np.ndarray,
    transfected_labels: list[int],
) -> list[CellMetrics]:
    """Compute area and shape metrics for each transfected cell.

    The historical ``jaccard`` field is aggregate area divided by cell area;
    it is an aggregate coverage fraction, not a conventional Jaccard index.
    """
    results = []
    cell_metrics = compute_shape_metrics(cell_masks)
    agg_metrics = compute_shape_metrics(agg_labels)
    for label in transfected_labels:
        cell_region = cell_masks == label
        cell_area = int(np.count_nonzero(cell_masks[cell_region]))
        agg_area = int(np.count_nonzero(agg_labels[cell_region]))
        nuc_area = int(np.count_nonzero(nuc_masks[cell_region]))

        # Average shape measurements across aggregates overlapping this cell.
        agg_of_cell = np.unique(agg_labels[cell_region])
        agg_of_cell = agg_of_cell[agg_of_cell > 0]
        agg_aspect_ratio = 0
        agg_circularity = 0
        for aggregate_label in agg_of_cell:
            if aggregate_label == 0:
                continue
            agg_aspect_ratio += agg_metrics[aggregate_label].get("aspect_ratio", 0.0)
            agg_circularity += agg_metrics[aggregate_label].get("circularity", 0.0)

        if len(agg_of_cell):
            agg_aspect_ratio /= len(agg_of_cell)
            agg_circularity /= len(agg_of_cell)

        results.append(
            CellMetrics(
                label=label,
                cell_area=cell_area,
                agg_area=agg_area,
                nuc_area=nuc_area,
                jaccard=agg_area / cell_area if cell_area > 0 else 0.0,
                cell_aspect_ratio=cell_metrics[label].get("aspect_ratio", 0.0),
                cell_circularity=cell_metrics[label].get("circularity", 0.0),
                agg_aspect_ratio=agg_aspect_ratio,
                agg_circularity=agg_circularity,
            )
        )
    return results
