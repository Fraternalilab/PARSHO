"""Threshold fluorescence channels and label aggregate regions."""

import numpy as np
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects


Threshold = float | np.ndarray


def extract_masks(
    aggregate_channel: np.ndarray,
    cell_masks: np.ndarray,
    method: str = "otsu",
    percentile: float = 95.0,
    local_block_size: int = 51,
    min_size_px: int = 10,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, Threshold]:
    """Extract masks constrained within cell boundaries.

    Returns:
        binary_mask:      Boolean (H, W) — True wherever an aggregate is detected.
        labelled_mask:    Integer (H, W) — each aggregate gets a unique ID (0 = background).
        thresh:           Scalar or local threshold image used for segmentation.
    """
    _validate_detection(aggregate_channel, cell_masks, min_size_px)
    img = aggregate_channel.astype(np.float32)
    cell_interior = cell_masks > 0

    # ── Threshold ────────────────────────────────────────────────────────
    if method == "otsu":
        thresh = threshold_otsu(img[cell_interior])
        binary = (img > (thresh * scale)) & cell_interior

    elif method == "percentile":
        thresh = np.percentile(img[cell_interior], percentile)
        binary = (img > thresh) & cell_interior

    elif method == "local":
        thresh = threshold_local(img, block_size=local_block_size)
        binary = (img > thresh) & cell_interior

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # ── Morphological cleanup ────────────────────────────────────────────
    binary, labelled = _label_in_cells(binary, cell_masks, min_size_px)

    return binary.astype(bool), labelled, thresh


def re_threshold_masks(
    aggregate_channel: np.ndarray,
    cell_masks: np.ndarray,
    min_size_px: int = 10,
    thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fixed aggregate threshold within cell boundaries.

    Returns:
        binary_mask:      Boolean (H, W) — True wherever an aggregate is detected.
        labelled_mask:    Integer (H, W) — each aggregate gets a unique ID (0 = background).
    """
    _validate_detection(aggregate_channel, cell_masks, min_size_px)
    img = aggregate_channel.astype(np.float32)
    cell_interior = cell_masks > 0

    # ── Threshold ────────────────────────────────────────────────────────
    binary = (img > thresh) & cell_interior

    # ── Morphological cleanup ────────────────────────────────────────────
    binary, labelled = _label_in_cells(binary, cell_masks, min_size_px)

    return binary, labelled


def _validate_detection(image, cells, minimum):
    if np.ndim(image) != 2 or np.shape(image) != np.shape(cells):
        raise ValueError("Signal and cell labels must be aligned 2-D arrays.")
    if not np.isfinite(image).all():
        raise ValueError("Signal intensities must be finite.")
    if not np.issubdtype(cells.dtype, np.integer) or (cells < 0).any():
        raise ValueError("Cell masks must contain nonnegative integer labels.")
    if not np.any(cells > 0):
        raise ValueError("No cells are available for thresholding; inspect segmentation first.")
    if not isinstance(minimum, (int, np.integer)) or minimum < 1:
        raise ValueError("Minimum object area must be a positive integer.")


def _label_in_cells(binary, cells, minimum):
    """Split touching objects at cell boundaries, then filter by object area."""
    labels = label(np.where(binary, cells, 0))
    keep = np.bincount(labels.ravel()) >= minimum
    keep[0] = False
    binary = keep[labels]
    return binary, label(np.where(binary, cells, 0))


def _remove_objects_smaller_than(
    binary: np.ndarray,
    min_size_px: int,
) -> np.ndarray:
    """Remove components smaller than ``min_size_px`` using skimage's new API.

    ``max_size`` removes objects whose size is less than or equal to its value,
    whereas the deprecated ``min_size`` removed objects strictly smaller than
    its value. Subtracting one preserves the existing public API semantics.
    """
    return remove_small_objects(binary, max_size=min_size_px - 1)


def find_optimal_threshold(
    pos_img_agregates,
    pos_masks,
    neg_img_agregates,
    neg_masks,
    pos_nuc_binary,
    neg_nuc_binary,
    t_min=None,
    t_max=None,
    verbose: bool = False,
    min_size_px: int = 2,
):
    """Find the first threshold with positive signal and no negative signal.

    Nuclear pixels are excluded before the positive and negative aggregate
    signals are compared. Supply empty nuclear masks to include nuclear signal.
    ``min_size_px`` should match the minimum aggregate size used for analysis.
    """

    if t_min is None or t_max is None:
        raise ValueError("t_min and t_max must both be provided")

    thresholds = np.linspace(t_min, t_max * 10, 1000)

    for i, threshold in enumerate(thresholds):
        pos_binary, _ = re_threshold_masks(
            pos_img_agregates, pos_masks, min_size_px=min_size_px, thresh=threshold
        )
        neg_binary, _ = re_threshold_masks(
            neg_img_agregates, neg_masks, min_size_px=min_size_px, thresh=threshold
        )

        # Exclude nuclear signal from the aggregate measurements.
        pos_binary[pos_nuc_binary] = 0
        neg_binary[neg_nuc_binary] = 0

        if verbose:
            print(f"Threshold {i + 1} of {len(thresholds)}")

        s1 = pos_binary.sum()  # True positives
        s2 = neg_binary.sum()  # False positives
        if s1 > 0 and s2 == 0:
            return threshold, s1
        elif s1 == 0:
            raise ValueError("No positives found at any threshold")
    raise ValueError("No threshold separated positive and negative signal")
