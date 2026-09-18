import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from parsho.distribution import compute_nucleus_centered_distribution
from parsho.plotting import plot_nucleus_centered_distribution


def test_radial_plot_contours_only_aggregates_from_selected_cell(monkeypatch):
    cell_masks = np.zeros((12, 12), dtype=np.int64)
    cell_masks[4:9, 4:9] = 1
    cell_masks[2:4, 2:4] = 2

    aggregate_mask = np.zeros(cell_masks.shape, dtype=bool)
    aggregate_mask[5, 5] = True
    aggregate_mask[2, 2] = True

    aggregate_channel = np.zeros(cell_masks.shape, dtype=np.float64)
    aggregate_channel[5, 5] = 10.0
    aggregate_channel[2, 2] = 100.0

    distribution = compute_nucleus_centered_distribution(
        cell_masks=cell_masks,
        nucleus_centroids={1: (6.0, 6.0)},
        aggregate_mask=aggregate_mask,
        aggregate_channel=aggregate_channel,
        radial_bins=4,
        cell_labels=[1],
    )[1]
    captured_contours = []

    def capture_contour(_axis, mask, *args, **kwargs):
        captured_contours.append(
            (kwargs.get("colors"), np.asarray(mask, dtype=bool).copy())
        )
        return None

    monkeypatch.setattr(Axes, "contour", capture_contour)

    figure, _ = plot_nucleus_centered_distribution(
        cell_masks,
        aggregate_mask,
        aggregate_channel,
        distribution,
        show=False,
    )

    white_contour = [
        mask for color, mask in captured_contours if color == "white"
    ]
    cyan_contour = [
        mask for color, mask in captured_contours if color == "cyan"
    ]
    assert len(white_contour) == 1
    assert len(cyan_contour) == 1
    assert white_contour[0].sum() == 1
    assert cyan_contour[0].sum() == 1
    assert np.array_equal(white_contour[0], cyan_contour[0])
    assert cyan_contour[0][3, 3]
    assert not cyan_contour[0][0, 0]
    assert distribution.aggregate_pixels.sum() == 1
    assert distribution.intensity_sum.sum() == 10.0

    plt.close(figure)
