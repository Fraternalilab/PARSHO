import numpy as np
import pytest

from parsho.distribution import (
    compute_nucleus_centered_distribution,
    nucleus_centroids_by_cell,
)


def test_nucleus_centroid_is_unchanged_when_it_rounds_inside_cell():
    cell_masks = np.ones((3, 3), dtype=int)
    nucleus_mask = np.zeros_like(cell_masks)
    nucleus_mask[1, 1:3] = 1

    assert nucleus_centroids_by_cell(cell_masks, nucleus_mask) == {1: (1.0, 1.5)}


def test_nucleus_centroid_snaps_to_nuclear_pixel_when_mean_is_outside_cell():
    cell_masks = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    nucleus_mask = np.array(
        [
            [1, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )

    centroids = nucleus_centroids_by_cell(cell_masks, nucleus_mask)

    assert centroids == {1: (0.0, 0.0)}
    distributions = compute_nucleus_centered_distribution(
        cell_masks=cell_masks,
        nucleus_centroids=centroids,
        aggregate_mask=np.zeros_like(cell_masks),
        aggregate_channel=np.zeros_like(cell_masks, dtype=float),
        radial_bins=3,
    )
    assert set(distributions) == {1}


def test_user_supplied_nucleus_centroid_outside_cell_is_still_rejected():
    cell_masks = np.array([[1, 0], [1, 0]])

    with pytest.raises(ValueError, match="Nucleus centroid is outside cell 1"):
        compute_nucleus_centered_distribution(
            cell_masks=cell_masks,
            nucleus_centroids={1: (0.0, 1.0)},
            aggregate_mask=np.zeros_like(cell_masks),
            aggregate_channel=np.zeros_like(cell_masks, dtype=float),
            radial_bins=2,
        )
