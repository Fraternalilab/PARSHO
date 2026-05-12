import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects
from skimage.measure import label


def extract_aggregate_masks(
    aggregate_channel: np.ndarray,
    cell_masks: np.ndarray,
    method: str = "otsu",
    percentile: float = 95.0,
    local_block_size: int = 51,
    min_size_px: int = 10,
    scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract aggregate masks constrained within cell boundaries.

    Returns:
        binary_mask:      Boolean (H, W) — True wherever an aggregate is detected.
        labelled_mask:    Integer (H, W) — each aggregate gets a unique ID (0 = background).
        thresh:           Float — the threshold value used for segmentation.
    """
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
        local_thresh = threshold_local(img, block_size=local_block_size)
        binary = (img > local_thresh) & cell_interior

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # ── Morphological cleanup ────────────────────────────────────────────
    binary = remove_small_objects(binary, min_size=min_size_px)

    # ── Label connected components ───────────────────────────────────────
    labelled = label(binary)

    return binary.astype(bool), labelled, thresh


def re_threshold_masks(
    aggregate_channel: np.ndarray,
    cell_masks: np.ndarray,
    min_size_px: int = 10,
    thresh: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract aggregate masks constrained within cell boundaries.

    Returns:
        binary_mask:      Boolean (H, W) — True wherever an aggregate is detected.
        labelled_mask:    Integer (H, W) — each aggregate gets a unique ID (0 = background).
    """
    img = aggregate_channel.astype(np.float32)
    cell_interior = cell_masks > 0

    # ── Threshold ────────────────────────────────────────────────────────
    binary = (img > thresh) & cell_interior


    # ── Morphological cleanup ────────────────────────────────────────────
    binary = remove_small_objects(binary, min_size=min_size_px)

    # ── Label connected components ───────────────────────────────────────
    labelled = label(binary)

    return binary, labelled


def find_optimal_threshold(
    pos_img_agregates,
    pos_masks,
    neg_img_agregates,
    neg_masks,
    pos_nuc_binary,
    neg_nuc_binary,
    t_min=None,
    t_max=None,
):
    """
    Find threshold that maximizes:
        S1(t) - lambda * S2(t)
    """

    # Define search space if not given
    thresholds = np.linspace(t_min, t_max * 10, 1000)

    for i,t in enumerate(thresholds):
        pos_binary, pos_labelled = re_threshold_masks(pos_img_agregates, pos_masks, min_size_px = 2, thresh = t)
        neg_binary, neg_labelled = re_threshold_masks(neg_img_agregates, neg_masks, min_size_px = 2, thresh = t)

        # compute the substraction
        pos_binary[pos_nuc_binary] = 0

        # compute the substraction
        neg_binary[neg_nuc_binary] = 0

        print(i)

        s1 = pos_binary.sum()  # True positives
        s2 = neg_binary.sum()  # False positives
        if s1 > 0 and s2 == 0:
            return t, s1
        elif s1 == 0:
            raise ValueError("No positives found at any threshold")
    raise t

