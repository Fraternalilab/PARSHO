"""Plot and export segmentation and aggregate-analysis results."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from parsho.distribution import (
    AggregateDistribution,
    shape_adapted_radial_bin_map,
)


def save_figure(
    fig,
    path,
    dpi=300,
    compression="tiff_lzw",
    bbox_inches="tight",
    pad_inches=0.05,
):
    """
    Save a matplotlib figure as a high-quality TIFF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str
        Output path (should end with .tiff or .tif).
    dpi : int
        Resolution (300–600 for publications).
    compression : str
        TIFF compression ('tiff_lzw', 'tiff_deflate', or None).
    bbox_inches : str
        Cropping behavior.
    pad_inches : float
        Padding around the figure.
    """
    fig.savefig(
        path,
        dpi=dpi,
        pil_kwargs={"compression": compression},
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    plt.close(fig)


def plot_segmentation_result(img_cell, masks, flows, save_path=None):
    """Display or save Cellpose segmentation output."""
    from cellpose import plot

    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img_cell, masks, flows[0])
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()


def plot_aggregate_channel(
    img_agregates,
    save_path=None,
    colors=None,
    cmap="gray",
    dpi=300,
):
    """Display or save an aggregate intensity image."""
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(img_agregates, cmap=cmap)
    ax.axis("off")  # clean look

    plt.tight_layout(pad=0)

    if save_path:
        fig.savefig(
            save_path,
            dpi=dpi,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
    else:
        plt.show()


def plot_aggregate_channel_color(
    img_agregates,
    save_path=None,
    colors=None,
    cmap=None,  # kept for compatibility
    dpi=300,
):
    """Display or save a categorical aggregate overlay."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # Default discrete colors
    if colors is None:
        colors = ["black", "white", "grey", "blue", "red"]

    cmap = mcolors.ListedColormap(colors)

    # Extend boundaries to include 4
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    ax.imshow(img_agregates, cmap=cmap, norm=norm)
    ax.axis("off")

    plt.tight_layout(pad=0)

    if save_path:
        fig.savefig(
            save_path,
            dpi=dpi,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
    else:
        plt.show()


def plot_aggregate_channel_color_labelled(
    img_aggregates,
    cell_masks,
    text_to_labell,
    save_path=None,
    colors=None,
    dpi=300,
    text_color="red",
    text_size=7,
    text_box=False,
    offset=(2, 2),
):
    """Plot a categorical aggregate image with per-cell annotations.

    Parameters
    ----------
    img_aggregates : ndarray
        Labeled mask image.

    cell_masks : ndarray
        Cell label image used to position annotations.

    text_to_labell : mapping
        Mapping from cell labels to annotation text.

    save_path : str, optional
        Output path.

    colors : list, optional
        List of colors for discrete labels.

    dpi : int
        Figure and saved-image resolution. Using the same resolution for both
        keeps annotations sharp in notebook output as well as exported files.

    text_color : str
        Annotation text color.

    text_size : int
        Font size.

    text_box : bool
        Whether to draw a semi-transparent background box.

    offset : tuple(int, int)
        Pixel offset from top-right corner of bounding box.
        (x_offset, y_offset)
    """
    from skimage.measure import regionprops

    # Set the figure DPI at creation time too. Previously ``dpi`` was only
    # passed to ``savefig``, so interactive/notebook output rendered the text
    # at Matplotlib's (usually much lower) default DPI and then scaled it up.
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    # Default discrete colors
    if colors is None:
        colors = ["black", "white", "grey", "blue", "red"]

    cmap = mcolors.ListedColormap(colors)

    norm = mcolors.BoundaryNorm(
        boundaries=[i - 0.5 for i in range(len(colors) + 1)], ncolors=cmap.N
    )

    # This is a categorical mask, so interpolation only softens its edges.
    ax.imshow(img_aggregates, cmap=cmap, norm=norm, interpolation="nearest")

    # Add one annotation per labeled object
    for region in regionprops(cell_masks):

        label = region.label

        if label not in text_to_labell:
            continue

        min_row, min_col, max_row, max_col = region.bbox

        # Top-right corner with slight inward offset
        x = max_col - offset[0]
        y = min_row + offset[1]

        bbox = dict(facecolor="black", alpha=0.5, pad=1) if text_box else None

        ax.text(
            x,
            y,
            text_to_labell[label],
            color=text_color,
            fontsize=text_size,
            ha="right",
            va="top",
            bbox=bbox,
        )

    ax.axis("off")
    plt.tight_layout(pad=0)

    if save_path:
        fig.savefig(
            save_path,
            dpi=dpi,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
    else:
        plt.show()


def plot_nucleus_centered_distribution(
    cell_masks: np.ndarray,
    aggregate_mask: np.ndarray | None,
    aggregate_channel: np.ndarray,
    distribution: AggregateDistribution,
    save_path=None,
    dpi: int = 150,
    show: bool = True,
    cell_name: str | None = None,
    center_label: str = "Nucleus",
):
    """Plot a nucleus-centered, cell-shape-adapted radial distribution.

    ``distribution`` may come from either PARSHO radial calculation. Its stored centroid
    is used to reconstruct the same shape-adapted bins used for measurement.

    Args:
        cell_masks: Integer cell-label image used for the distribution.
        aggregate_mask: Boolean or labeled aggregate mask, or ``None``.
        aggregate_channel: Aggregate fluorescence image used for intensity.
        distribution: Distribution for one cell.
        save_path: Optional output filename.
        dpi: Resolution used when saving the figure.
        show: Display the figure when ``True``.
        cell_name: Optional display name for the cell. The numeric
            ``distribution.label`` is still used internally to select the
            correct mask. When omitted, the title uses that numeric label.
        center_label: Display label for the origin; use ``"Cell"`` for a
            cell-centered distribution. This does not alter the stored bins.

    Returns:
        The Matplotlib figure and a dictionary containing its four axes.
    """
    if not (
        cell_masks.shape == aggregate_channel.shape and cell_masks.ndim == 2
    ):
        raise ValueError("Plot inputs must be equally shaped 2D arrays")
    if aggregate_mask is not None and aggregate_mask.shape != cell_masks.shape:
        raise ValueError("Plot inputs must be equally shaped 2D arrays")

    section_map = shape_adapted_radial_bin_map(
        cell_masks,
        distribution.label,
        distribution.centroid,
        radial_bins=distribution.cell_pixels.size,
    )
    return _plot_distribution_panels(
        cell_masks,
        aggregate_mask,
        aggregate_channel,
        distribution,
        section_map,
        f"{center_label}-centered shape-adapted bins (white = aggregates)",
        save_path,
        dpi,
        show,
        cell_name,
    )


def _plot_distribution_panels(
    cell_masks,
    aggregate_mask,
    aggregate_channel,
    distribution,
    section_map,
    section_title,
    save_path,
    dpi,
    show,
    cell_name,
):
    """Draw the shared measurement panels for either radial method."""
    cell_region = cell_masks == distribution.label
    rows, columns = np.nonzero(cell_region)
    padding = 2
    row_slice = slice(
        max(rows.min() - padding, 0),
        min(rows.max() + padding + 1, cell_masks.shape[0]),
    )
    column_slice = slice(
        max(columns.min() - padding, 0),
        min(columns.max() + padding + 1, cell_masks.shape[1]),
    )

    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    section_axis = figure.add_subplot(grid[0, 0])
    image_axis = figure.add_subplot(grid[1, 0])
    coverage_axis = figure.add_subplot(grid[0, 1])
    intensity_axis = figure.add_subplot(grid[1, 1])

    section_image = section_axis.imshow(
        section_map[row_slice, column_slice], cmap="twilight", interpolation="nearest"
    )
    effective_mask = (
        cell_region
        if aggregate_mask is None
        else (aggregate_mask != 0) & cell_region
    )
    cropped_aggregates = effective_mask[row_slice, column_slice]
    if np.any(cropped_aggregates):
        section_axis.contour(
            cropped_aggregates.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=1.5,
        )
    section_axis.set_title(section_title)
    section_axis.axis("off")
    figure.colorbar(section_image, ax=section_axis, label="Radial bin index")

    intensity_image = np.ma.masked_where(
        ~cell_region[row_slice, column_slice],
        aggregate_channel[row_slice, column_slice],
    )
    channel_image = image_axis.imshow(intensity_image, cmap="magma")
    if np.any(cropped_aggregates):
        image_axis.contour(
            cropped_aggregates.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=1.5,
        )
    image_axis.set_title("Aggregate channel (cyan = aggregate mask)")
    image_axis.axis("off")
    figure.colorbar(channel_image, ax=image_axis, label="Pixel intensity")

    _plot_radial_values(
        coverage_axis,
        distribution.aggregate_fraction,
        "Aggregate coverage",
        "Aggregate pixels / cell pixels",
        "#26828e",
    )

    _plot_radial_values(
        intensity_axis,
        np.cumsum(distribution.intensity_share),
        "Cumulative normalized intensity",
        "Fraction of total cell intensity",
        "#d1495b",
    )
    intensity_axis.set_ylim(0, 1.05)
    display_name = (
        f"cell {distribution.label}" if cell_name is None else str(cell_name)
    )
    figure.suptitle(f"Aggregate distribution — {display_name}")

    if save_path is not None:
        figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    elif save_path is not None:
        plt.close(figure)

    axes = {
        "sections": section_axis,
        "channel": image_axis,
        "coverage": coverage_axis,
        "intensity": intensity_axis,
    }
    return figure, axes


def _plot_radial_values(axis, values, title, ylabel, color):
    """Draw a center-to-boundary measurement profile."""
    bin_count = values.size
    bin_centers = (np.arange(bin_count) + 0.5) / bin_count
    bin_width = 0.9 / bin_count
    axis.bar(bin_centers, values, width=bin_width, color=color, edgecolor="white")
    axis.plot(bin_centers, values, color="black", marker="o", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Normalized radius (center → boundary)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
