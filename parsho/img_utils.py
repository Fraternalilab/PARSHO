"""Utilities for loading and contrast-normalizing microscopy images."""

from pathlib import Path

import numpy as np


TIFF_EXTENSIONS = {".tif", ".tiff"}
RASTER_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png"}
DICOM_EXTENSIONS = {"", ".dcm", ".dicom", ".dic", ".ima"}
MICROSCOPY_EXTENSIONS = {".nd2", ".lif", ".czi", ".lof"}
SUPPORTED_IMAGE_EXTENSIONS = (
    TIFF_EXTENSIONS | RASTER_EXTENSIONS | DICOM_EXTENSIONS | MICROSCOPY_EXTENSIONS
)


def load_image(input_path: str | Path, scene: int = 0) -> np.ndarray:
    """Read a supported image file into a NumPy array.

    This is the package's equivalent of :func:`cellpose.io.imread`. TIFF and
    common raster images have lightweight readers; proprietary microscopy
    formats and DICOM require the ``microscopy-io`` optional dependencies.

    For ND2, LIF, CZI, and LOF files the returned axes are always ``TCZYX``.
    ``scene`` selects an image/series in containers
    that hold more than one image. Raster images use ``YX`` or ``YXS`` (RGB),
    while multi-frame DICOM follows pydicom's documented array layout.
    """
    path = Path(input_path)
    if not isinstance(scene, int) or isinstance(scene, bool) or scene < 0:
        raise ValueError("scene must be a nonnegative integer (starting at 0)")
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(
            extension or "extensionless DICOM"
            for extension in sorted(SUPPORTED_IMAGE_EXTENSIONS)
        )
        raise ValueError(f"Unsupported image format '{suffix}'. Supported: {supported}")

    if suffix in TIFF_EXTENSIONS:
        try:
            import tifffile
        except ImportError as error:
            raise ImportError("Reading TIFF images requires 'tifffile'") from error
        with tifffile.TiffFile(path) as image:
            if scene >= len(image.series):
                raise ValueError(f"scene must be below {len(image.series)} for {path.name} (TIFF series)")
            return np.asarray(image.series[scene].asarray())

    if scene and suffix not in MICROSCOPY_EXTENSIONS:
        raise ValueError("Scene selection is supported for TIFF and vendor containers only")

    if suffix in RASTER_EXTENSIONS:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError(
                "Reading JPEG, BMP, and PNG images requires 'Pillow'"
            ) from error
        with Image.open(path) as image:
            return np.asarray(image)

    if suffix in DICOM_EXTENSIONS:
        try:
            from pydicom.pixels import pixel_array
        except ImportError as error:
            raise ImportError("Reading DICOM images requires 'pydicom'") from error
        try:
            return np.asarray(pixel_array(path))
        except Exception as error:
            raise ValueError(
                f"Could not decode DICOM pixel data in {path}: {error}"
            ) from error

    return _load_microscopy_image(path, scene)


def _load_microscopy_image(path: Path, scene: int) -> np.ndarray:
    """Read a vendor container and put its axes in canonical TCZYX order."""
    try:
        from bioio import BioImage
    except ImportError as error:
        raise ImportError(
            f"Reading {path.suffix.upper()} images requires the 'microscopy-io' "
            "extra: pip install 'parsho[microscopy-io]'"
        ) from error

    try:
        image = BioImage(path)

        if not isinstance(scene, int) or scene < 0 or scene >= len(image.scenes):
            raise IndexError(
                f"scene must be between 0 and {len(image.scenes) - 1}; received {scene}"
            )
        image.set_scene(scene)
        return np.asarray(image.get_image_data("TCZYX"))
    except (ImportError, IndexError):
        raise
    except Exception as error:
        raise ValueError(f"Could not read microscopy image {path}: {error}") from error


def load_field_channels(input_path, *, scene=0, axes="", time=0, z_mode="maximum", z=0):
    """Return raw 2-D channels and source metadata for one field.

    This is the same metadata-aware loader used by Colab. Select one time
    point before projecting Z. Indices start at zero. Unknown stack axes
    require an explicit order such as CYX, ZYX or TCZYX.
    """
    from parsho.single_image import load_field_channels as load_field

    return load_field(input_path, scene=scene, axes=axes, time=time, z_mode=z_mode, z=z)


def extract_channels(
    input_path: str | Path,
    normalize: bool = True,
    scene: int = 0,
    *,
    axes: str = "",
    time: int = 0,
    z_mode: str = "maximum",
    z: int = 0,
) -> list[np.ndarray]:
    """Return channels from the same 2-D field selection used by Colab.

    For measurements use normalize=False; the historical default True
    contrast-stretches each channel to uint8 for display. Time is selected
    before Z projection and is never flattened into depth. Ambiguous TIFF
    stacks must specify axes. Use load_field_channels for source metadata.
    """
    channels, _ = load_field_channels(
        input_path, scene=scene, axes=axes, time=time, z_mode=z_mode, z=z
    )
    if not normalize:
        return channels
    normalized = []
    for channel in channels:
        channel = channel.astype(np.float32)
        min_val, max_val = imagej_auto(channel, saturate_pct=0.35)
        normalized.append(apply_window(channel, min_val, max_val))
    return normalized


def load_mask_npy(
    input_path: str | Path,
    kind: str = "binary",
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load and validate an external mask stored in NumPy ``.npy`` format.

    A two-dimensional array is expected. Singleton dimensions are removed, so
    shapes such as ``(1, H, W)`` and ``(H, W, 1)`` are accepted. Object arrays
    and arrays with remaining extra dimensions are rejected.

    Args:
        input_path: Path to a ``.npy`` mask file.
        kind: ``"binary"`` to convert every nonzero value to ``True``, or
            ``"labels"`` for a nonnegative integer cell-label image.
        expected_shape: Optional ``(H, W)`` shape used to verify alignment.

    Returns:
        A Boolean binary mask or an ``int64`` label mask.

    Raises:
        ValueError: If the file contents, mask values, kind, or shape are
            invalid.
    """
    if kind not in {"binary", "labels"}:
        raise ValueError("kind must be 'binary' or 'labels'")

    path = Path(input_path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Mask path must point to a .npy file")

    try:
        mask = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load NumPy mask {path}: {error}") from error

    if not isinstance(mask, np.ndarray) or mask.dtype.hasobject:
        raise ValueError("Mask file must contain a numeric NumPy array")
    # Remove storage axes such as (1, H, W) without collapsing a valid (1, W)
    # or (H, 1) two-dimensional mask.
    while mask.ndim > 2:
        singleton_axes = [
            axis for axis, axis_size in enumerate(mask.shape) if axis_size == 1
        ]
        if not singleton_axes:
            break
        mask = np.squeeze(mask, axis=singleton_axes[0])
    if mask.ndim != 2:
        raise ValueError(f"Mask must reduce to shape (H, W); received {mask.shape}")
    if expected_shape is not None and mask.shape != tuple(expected_shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match expected shape "
            f"{tuple(expected_shape)}"
        )
    is_real_numeric = any(
        np.issubdtype(mask.dtype, dtype)
        for dtype in (np.bool_, np.integer, np.floating)
    )
    if not is_real_numeric:
        raise ValueError("Mask must have a Boolean, integer, or floating dtype")
    if not np.all(np.isfinite(mask)):
        raise ValueError("Mask values must all be finite")

    if kind == "binary":
        return mask != 0

    _validate_label_values(mask)
    return mask.astype(np.int64, copy=False)


def _validate_label_values(mask: np.ndarray) -> None:
    """Validate that a mask contains nonnegative integer-valued labels."""
    if np.any(mask < 0):
        raise ValueError("Label masks cannot contain negative values")
    if not np.all(mask == np.floor(mask)):
        raise ValueError("Label masks must contain integer-valued labels")


def imagej_auto(img: np.ndarray, saturate_pct: float = 0.35) -> tuple[int, int]:
    """Replicate ImageJ's automatic brightness and contrast calculation.

    ImageJ saturates 0.35% of pixels by default (0.175% each tail).
    """
    lo = np.percentile(img, saturate_pct / 2)
    hi = np.percentile(img, 100 - saturate_pct / 2)
    return int(lo), int(hi)


def apply_window(img: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    """Linear stretch between min and max, clip outliers."""
    if max_val <= min_val:
        return np.zeros(img.shape, dtype=np.uint8)
    img_clipped = np.clip(img, min_val, max_val)
    img_norm = (img_clipped - min_val) / (max_val - min_val)
    return (img_norm * 255).astype(np.uint8)
