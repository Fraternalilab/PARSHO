import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cellpose import plot
from skimage.measure import regionprops

def save_figure_tiff(
    fig,
    path,
    dpi=300,
    compression="tiff_lzw",
    bbox_inches="tight",
    pad_inches=0.05
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
        format="tiff",
        dpi=dpi,
        pil_kwargs={"compression": compression},
        bbox_inches=bbox_inches,
        pad_inches=pad_inches
    )
    plt.close(fig)


def plot_segmentation_result(img_cell, masks, flows, save_path=None):
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img_cell, masks, flows[0])
    plt.tight_layout()

    if save_path:
        save_figure_tiff(fig, save_path)
    else:
        plt.show()


def plot_aggregate_channel(
    img_agregates,
    save_path=None,
    colors=None,
    cmap="gray",
    dpi=300
):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(img_agregates, cmap=cmap)
    ax.axis("off")  # clean look

    plt.tight_layout(pad=0)

    if save_path:
        fig.savefig(
            save_path,
            format="tiff",
            dpi=dpi,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close(fig)
    else:
        plt.show()


def plot_aggregate_channel_color(
    img_agregates,
    save_path=None,
    colors=None,
    cmap=None,  # kept for compatibility
    dpi=300
):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Default discrete colors
    if colors is None:
        colors = ["black", "white", "grey", "blue", "red"]

    cmap = mcolors.ListedColormap(colors)

    # Extend boundaries to include 4
    norm = mcolors.BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        cmap.N
    )

    ax.imshow(img_agregates, cmap=cmap, norm=norm)
    ax.axis("off")

    plt.tight_layout(pad=0)

    if save_path:
        fig.savefig(
            save_path,
            format="tiff",
            dpi=dpi,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0
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
    text_size=5,
    text_box=False,
    offset=(2, 2),
):
    """
    Plot labeled aggregate image with optional object annotations.

    Parameters
    ----------
    img_aggregates : ndarray
        Labeled mask image.

    save_path : str, optional
        Output path.

    colors : list, optional
        List of colors for discrete labels.

    dpi : int
        Figure dpi.

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

    fig, ax = plt.subplots(figsize=(5, 5))

    # Default discrete colors
    if colors is None:
        colors = ["black", "white", "grey", "blue", "red"]

    cmap = mcolors.ListedColormap(colors)

    norm = mcolors.BoundaryNorm(
        boundaries=[i - 0.5 for i in range(len(colors) + 1)],
        ncolors=cmap.N
    )

    ax.imshow(img_aggregates, cmap=cmap, norm=norm)

    # Add one annotation per labeled object
    for region in regionprops(cell_masks):

        label = region.label

        if label not in text_to_labell:
            continue

        min_row, min_col, max_row, max_col = region.bbox

        # Top-right corner with slight inward offset
        x = max_col - offset[0]
        y = min_row + offset[1]

        bbox = (
            dict(facecolor="black", alpha=0.5, pad=1)
            if text_box
            else None
        )

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
            format="tiff",
            dpi=dpi,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
    else:
        plt.show()