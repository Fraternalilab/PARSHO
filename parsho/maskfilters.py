import numpy as np 
from skimage.measure import regionprops

from dataclasses import dataclass
 
import numpy as np
 
 
# ------------------------------------------------------------------ #
# Data containers
# ------------------------------------------------------------------ #
 
@dataclass
class ChannelMasks:
    """Holds all binary/label masks derived from a single image set."""
    cell_masks: np.ndarray       # cellpose integer labels
    agg_binary: np.ndarray       # aggregate binary mask (post-filter)
    agg_labels: np.ndarray       # aggregate label mask (post-filter)
    nuc_binary: np.ndarray       # nuclear binary mask
    nuc_labels: np.ndarray       # nuclear label mask
    trans_binary: np.ndarray     # transfection binary mask
    trans_labels: np.ndarray     # transfection label mask
 
 
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
 
 
# ------------------------------------------------------------------ #
# Mask processing
# ------------------------------------------------------------------ #
 
def subtract_nuclear_from_aggregate(
    agg_binary: np.ndarray,
    agg_labels: np.ndarray,
    nuc_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove nuclear regions from aggregate masks.
 
    Returns updated (agg_binary, agg_labels) with nuclear pixels zeroed out.
    """
    agg_labels = agg_labels.copy()
    agg_binary = agg_binary.copy()
    agg_labels[nuc_binary == 1] = 0
    agg_binary[nuc_binary == 1] = 0
    return agg_binary, agg_labels
 
 
def build_overlay(
    cell_masks: np.ndarray,
    nuc_labels: np.ndarray,
    agg_labels: np.ndarray,
    outlines: np.ndarray,
) -> np.ndarray:
    """
    Compose a colour-coded overlay image with the following label scheme:
        0 - background
        1 - cell interior
        2 - cell outline
        3 - nucleus
        4 - aggregate
    """
    overlay = np.zeros_like(agg_labels)
    overlay[cell_masks > 0] = 1
    overlay[outlines > 0] = 2
    overlay[nuc_labels > 0] = 3
    overlay[agg_labels > 0] = 4
    return overlay
 
 
def filter_transfected_cells(
    cell_masks: np.ndarray,
    trans_labels: np.ndarray,
    nuc_labels: np.ndarray,
) -> list[int]:
    """
    Return the cell labels that are both transfected and contain nuclear signal.
 
    A cell passes the filter when its pixel region overlaps with at least one
    non-zero value in *both* trans_labels and nuc_labels.
    """
    transfected = []
    for label in np.unique(cell_masks):
        if label == 0:
            continue
        cell_region = cell_masks == label
        if np.any(trans_labels[cell_region]) and np.any(nuc_labels[cell_region]):
            transfected.append(label)
    return transfected
 
 
def mask_overlay_to_transfected(
    overlay: np.ndarray,
    cell_masks: np.ndarray,
    transfected_labels: list[int],
) -> np.ndarray:
    """
    Zero out overlay regions that do not belong to transfected cells.
    Returns a new overlay array.
    """
    overlay = overlay.copy()
    final_cells = np.zeros_like(cell_masks)
    for label in transfected_labels:
        final_cells[cell_masks == label] = 1
    overlay[final_cells == 0] = 0
    return overlay
 
 
# ------------------------------------------------------------------ #
# Per-cell metrics
# ------------------------------------------------------------------ #


def compute_shape_metrics(labeled: np.ndarray):
    """
    Returns area, aspect ratio, and circularity for each labeled object.
    Aspect ratio = major_axis / minor_axis  (1.0 = circle, higher = elongated)
    Circularity  = (4π x area) / perimeter**2 (1.0 = circle, lower = irregular)
    """
    
    props = regionprops(labeled)
    metrics = {}
    for p in props:
        major = p.major_axis_length
        minor = p.minor_axis_length
        aspect_ratio = (major / minor) if minor > 0 else float('inf')
        circularity = (4 * np.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0
        metrics[p.label] = {
            "area": p.area,
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
    """
    Compute area and Jaccard index for each transfected cell.
 
    Jaccard here is defined as agg_area / cell_area (overlap fraction).
    """
    results = []
    cell_metrics = compute_shape_metrics(cell_masks)
    agg_metrics = compute_shape_metrics(agg_labels)


    for label in transfected_labels:
        cell_region = cell_masks == label
        cell_area = int(np.count_nonzero(cell_masks[cell_region]))
        agg_area = int(np.count_nonzero(agg_labels[cell_region]))
        nuc_area = int(np.count_nonzero(nuc_masks[cell_region]))

        # extract agg label
        agg_of_cell = np.unique(agg_labels[cell_region])
        agg_aspect_ratio = 0
        agg_circularity = 0
        for l in agg_of_cell:
            if l == 0:
                continue
            agg_aspect_ratio += agg_metrics[l].get("aspect_ratio", 0.0)
            agg_circularity += agg_metrics[l].get("circularity", 0.0)
        
        if len(agg_of_cell) > 1:
            agg_aspect_ratio /= len(agg_of_cell) - 1
            agg_circularity /= len(agg_of_cell) - 1

        results.append(CellMetrics(
            label=label,
            cell_area=cell_area,
            agg_area=agg_area,
            nuc_area=nuc_area,
            jaccard=agg_area / cell_area if cell_area > 0 else 0.0,
            cell_aspect_ratio=cell_metrics[label].get("aspect_ratio", 0.0),
            cell_circularity=cell_metrics[label].get("circularity", 0.0),
            agg_aspect_ratio=agg_aspect_ratio,
            agg_circularity=agg_circularity,
        ))
    return results
