import numpy as np
import pytest

from parsho.maskfilters import filter_cells_by_overlap


@pytest.fixture
def cell_masks():
    return np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0],
        ]
    )


def test_filters_cells_overlapping_every_label_mask(cell_masks):
    first_mask = np.zeros_like(cell_masks)
    first_mask[0, 0] = 4
    first_mask[0, 3] = 5
    second_mask = np.zeros_like(cell_masks)
    second_mask[1, 1] = 7
    second_mask[3, 0] = 8

    assert filter_cells_by_overlap(cell_masks, first_mask, second_mask) == [1]


def test_accepts_any_number_of_label_masks(cell_masks):
    masks = []
    for position in [(0, 0), (0, 1), (1, 0)]:
        mask = np.zeros_like(cell_masks)
        mask[position] = 1
        masks.append(mask)

    assert filter_cells_by_overlap(cell_masks, *masks) == [1]
    assert filter_cells_by_overlap(cell_masks, masks[0]) == [1]


def test_requires_at_least_one_label_mask(cell_masks):
    with pytest.raises(ValueError, match="At least one label mask"):
        filter_cells_by_overlap(cell_masks)


def test_rejects_label_mask_with_different_shape(cell_masks):
    with pytest.raises(ValueError, match="Label mask 1 has shape"):
        filter_cells_by_overlap(cell_masks, np.zeros((2, 2), dtype=int))
