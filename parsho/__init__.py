"""Tools for cell segmentation and fluorescence puncta analysis."""

import torch


# Cellpose constructs trusted sparse tensors from internally generated,
# bounds-clamped coordinates. Explicitly retain PyTorch's performance-oriented
# default while preventing its implicit-policy warning.
torch.sparse.check_sparse_tensor_invariants.disable()

__version__ = "0.1.0"

from parsho.img_utils import extract_channels, load_image, load_mask_npy
from parsho.distribution import (
    AggregateDistribution,
    cell_centroids_by_cell,
    compute_cell_centered_distribution,
    compute_nucleus_centered_distribution,
    nucleus_centroids_by_cell,
)
from parsho.utils import (
    radial_distributions_to_records,
    save_radial_distributions_csv,
)

__all__ = [
    "__version__",
    "AggregateDistribution",
    "compute_cell_centered_distribution",
    "compute_nucleus_centered_distribution",
    "cell_centroids_by_cell",
    "nucleus_centroids_by_cell",
    "load_mask_npy",
    "load_image",
    "extract_channels",
    "radial_distributions_to_records",
    "save_radial_distributions_csv",
]
