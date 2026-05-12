from pathlib import Path
import tifffile
import numpy as np


def extract_channels(
    input_path: str | Path,
) -> list[np.ndarray]:
    """
    Extract each channel of a multi-channel TIFF and save as separate files.
 
    Parameters
    ----------
    input_path    : Path to the source TIFF file.

    Returns
    -------
    List of NumPy arrays representing the extracted channels.
    """
    src = Path(input_path)
 
    img = tifffile.imread(str(src))
 
    # Determine channel slices based on array shape
    ndim = img.ndim
    if ndim == 2:
        # (H, W) – single channel
        channels = [img]
    elif ndim == 3:
        d0, d1, d2 = img.shape
        if d2 in (1, 2, 3, 4) and d2 <= min(d0, d1):
            # (H, W, C) interleaved
            channels = [img[:, :, c] for c in range(d2)]
        else:
            # (C, H, W) channel-first
            channels = [img[c] for c in range(d0)]
    elif ndim == 4:
        # (C, Z, H, W) – split on axis 0
        channels = [img[c] for c in range(img.shape[0])]
    elif ndim == 5:
        # (T, C, Z, H, W) – split on axis 1, flatten T and Z
        T, C, Z, H, W = img.shape
        channels = [img[:, c].reshape(T * Z, H, W) for c in range(C)]
    else:
        raise ValueError(f"Unsupported array shape: {img.shape}")
 
    return channels
 
