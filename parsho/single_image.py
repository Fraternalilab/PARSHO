"""Single-field loading and analysis shared by the interactive Colab workflow.

This module deliberately has no Cellpose, pandas, or widget imports. Cellpose
provides the cell labels; the same analysis works with or without nuclear data.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops

from parsho.distribution import (
    compute_cell_centered_distribution,
    compute_nucleus_centered_distribution,
    nucleus_centroids_by_cell,
)
from parsho.img_utils import DICOM_EXTENSIONS, MICROSCOPY_EXTENSIONS, load_image
from parsho.maskfilters import compute_cell_metrics
from parsho.segmentation import extract_masks, re_threshold_masks
from parsho.utils import radial_distributions_to_records


def load_field_channels(path, *, scene=0, axes="", time=0, z_mode="maximum", z=0):
    """Load raw 2-D channels, selecting time BEFORE projecting depth.

    TIFF axes come from the selected series. Vendor containers use TCZYX.
    Ambiguous TIFF/DICOM stacks require an explicit axes choice rather than
    guessing whether pages represent channels, depth, or time. Indices are
    zero-based here; the notebook presents them starting at one.
    """
    path = Path(path)
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0
           for value in (scene, time, z)):
        raise ValueError("Scene, time and Z indices must be nonnegative integers.")
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        with tifffile.TiffFile(path) as image:
            if scene >= len(image.series):
                raise ValueError(f"{path.name} has only {len(image.series)} series.")
            series = image.series[scene]
            array = series.asarray()
            stored_axes = series.axes
    else:
        if scene and path.suffix.lower() not in MICROSCOPY_EXTENSIONS:
            raise ValueError("Scene selection applies only to TIFF or vendor containers.")
        array = load_image(path, scene=scene)
        if path.suffix.lower() in MICROSCOPY_EXTENSIONS:
            stored_axes = "TCZYX"
        elif path.suffix.lower() in DICOM_EXTENSIONS:
            import pydicom

            header = pydicom.dcmread(path, stop_before_pixels=True)
            # DICOM frames can encode time or depth: leave F unresolved.
            color = int(getattr(header, "SamplesPerPixel", 1)) > 1
            stored_axes = "F" if array.ndim > (3 if color else 2) else ""
            stored_axes += "YXS" if color else "YX"
        else:
            stored_axes = "YX" if array.ndim == 2 else "YXS"

    original_shape = array.shape
    selected_axes = (axes.strip().upper() or stored_axes).replace("S", "C")
    if len(selected_axes) != array.ndim:
        raise ValueError(f"Axes {selected_axes!r} do not match shape {array.shape} in {path.name}.")
    # Unknown singleton storage axes do not need user intervention.
    for index in range(len(selected_axes) - 1, -1, -1):
        if selected_axes[index] not in "TCZYX" and array.shape[index] == 1:
            array = np.take(array, 0, axis=index)
            selected_axes = selected_axes[:index] + selected_axes[index + 1:]
    if (set(selected_axes) - set("TCZYX") or len(set(selected_axes)) != len(selected_axes)
            or "Y" not in selected_axes or "X" not in selected_axes):
        raise ValueError(
            f"{path.name}: shape {original_shape}, axes {stored_axes!r} are ambiguous. "
            "Choose the file's dimension order: e.g. CYX for channels, ZYX for depth, "
            "TYX for time, or TCZYX. Y and X are image height and width."
        )
    resolved_axes = selected_axes
    if "T" in selected_axes:
        axis = selected_axes.index("T")
        if time >= array.shape[axis]:
            raise ValueError(f"{path.name}: requested time is outside the available range.")
        array = np.take(array, time, axis=axis)
        selected_axes = selected_axes.replace("T", "")
    elif time:
        raise ValueError(f"{path.name} has no time axis; choose time point 1.")
    if z_mode not in {"maximum", "plane"}:
        raise ValueError("Z handling must be 'maximum' or 'plane'.")
    if "Z" in selected_axes:
        axis = selected_axes.index("Z")
        if z_mode == "maximum":
            array = array.max(axis=axis)
        else:
            if z >= array.shape[axis]:
                raise ValueError(f"{path.name}: requested Z plane is outside the available range.")
            array = np.take(array, z, axis=axis)
        selected_axes = selected_axes.replace("Z", "")
    elif z_mode == "plane" and z:
        raise ValueError(f"{path.name} has no Z axis; choose plane 1.")
    if "C" not in selected_axes:
        array = array[np.newaxis]
        selected_axes = "C" + selected_axes
    array = array.transpose(tuple(selected_axes.index(axis) for axis in "CYX"))
    if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array) or not np.isfinite(array).all():
        raise ValueError(f"{path.name} must contain finite real pixel intensities.")
    metadata = dict(file=path.name, stored_shape=list(original_shape), stored_axes=stored_axes,
                    axes=resolved_axes, scene=scene, time=time, z_mode=z_mode, z=z)
    return list(array), metadata


def combine_segmentation_channels(channels, method="mean"):
    """Normalize only the segmentation copies and combine selected channels."""
    if not channels:
        raise ValueError("Tick 'Use for cell segmentation' for at least one channel.")
    if any(image.ndim != 2 or image.shape != channels[0].shape for image in channels):
        raise ValueError("Selected segmentation channels must be aligned 2-D images.")
    normalized = []
    for image in channels:
        image = image.astype(np.float32)
        low, high = np.percentile(image, (1, 99))
        normalized.append(np.zeros_like(image) if high <= low else np.clip((image - low) / (high - low), 0, 1))
    stack = np.stack(normalized)
    if method == "mean":
        return stack.mean(axis=0)
    if method == "maximum":
        return stack.max(axis=0)
    if method == "stack":
        if len(channels) > 3:
            raise ValueError("Multichannel Cellpose input allows at most 3 channels; choose mean or maximum.")
        return np.moveaxis(stack, 0, -1)
    raise ValueError("Choose mean, maximum or stack for segmentation channels.")


@dataclass(frozen=True)
class DetectionSettings:
    method: str = "otsu"
    min_size: int = 2
    scale: float = 1.0
    percentile: float = 95.0
    block_size: int = 51
    manual_threshold: float = 1000.0

    def validate(self):
        if self.method not in {"otsu", "percentile", "local", "manual"}:
            raise ValueError("Choose Otsu, percentile, local or manual detection.")
        if not isinstance(self.min_size, int) or self.min_size < 1:
            raise ValueError("Minimum object area must be a positive integer (pixels).")
        if self.method == "otsu" and (not np.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("Otsu multiplier must be positive and finite.")
        if self.method == "percentile" and (not np.isfinite(self.percentile) or not 0 <= self.percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        if self.method == "manual" and not np.isfinite(self.manual_threshold):
            raise ValueError("Manual threshold must be finite.")
        if self.method == "local" and (not isinstance(self.block_size, int) or self.block_size < 3 or self.block_size % 2 != 1):
            raise ValueError("Local window width must be an odd integer of at least 3 pixels.")


def detect_signal(image, cells, settings):
    """Detect a signal with the same threshold options used throughout Colab."""
    settings.validate()
    if settings.method == "manual":
        binary, labels = re_threshold_masks(image, cells, min_size_px=settings.min_size,
                                            thresh=settings.manual_threshold)
        threshold = settings.manual_threshold
    else:
        binary, labels, threshold = extract_masks(
            image, cells, method=settings.method, min_size_px=settings.min_size,
            scale=settings.scale, percentile=settings.percentile,
            local_block_size=settings.block_size,
        )
        if settings.method == "otsu":
            threshold = float(threshold) * settings.scale
    return binary, labels, threshold


CELL_COLUMNS = [
    "signal", "cell_label", "cell_area_pixels", "nucleus_area_pixels", "nucleus_count",
    "aggregate_count", "aggregate_area_pixels", "aggregate_coverage_fraction",
    "cell_aspect_ratio", "cell_circularity", "aggregate_mean_aspect_ratio",
    "aggregate_mean_circularity", "aggregate_mean_area_pixels", "aggregate_intensity_sum",
    "aggregate_intensity_mean", "cell_intensity_sum_raw", "cell_intensity_mean_raw",
    "cell_intensity_sum_analyzed", "nucleus_intensity_sum_raw", "radial_center",
    "radial_included", "cell_area_um2", "nucleus_area_um2", "aggregate_area_um2",
]
OBJECT_COLUMNS = ["signal", "cell_label", "aggregate_label", "area_pixels", "centroid_row",
                  "centroid_column", "aspect_ratio", "circularity", "intensity_sum",
                  "intensity_mean", "area_um2"]
FILTER_COLUMNS = ["cell_label", "has_nucleus", "has_transfection", "has_aggregate",
                  "touches_border", "retained", "reason"]
RADIAL_COLUMNS = ["signal", "center", "image_id", "cell_label", "centroid_row", "centroid_column",
                  "radial_bin", "radius_start", "radius_end", "section_size_pixels",
                  "aggregate_pixels", "aggregate_present", "aggregate_fraction", "aggregate_share",
                  "intensity_sum", "intensity_mean", "intensity_share", "cumulative_normalized_intensity"]


def analyze_field(
    cells, signals, *, detection=None, nucleus=None, nucleus_detection=None,
    transfection=None, transfection_detection=None, remove_nuclear=False,
    require_nucleus=False, require_transfection=False, require_aggregates=False,
    exclude_border=False, radial=True, radial_center="auto", radial_bins=10,
    pixel_size_um=None,
):
    """Measure one field, with a separate result for every named puncta channel.

    Retention is shared across signals. The aggregate filter requires an object
    in ANY selected signal. Labels are original cell labels in every output.
    Missing nuclear measurements are None, not biological zero measurements.

    Args:
        cells: Aligned (H, W) integer cell labels; 0 is background.
        signals: Mapping of unique signal names to raw (H, W) intensity arrays.
        detection: Mapping of signal names to DetectionSettings; defaults to Otsu.
        nucleus: Optional aligned nuclear intensity image, not a precomputed mask.
        nucleus_detection: Nuclear threshold settings (default Otsu, 5 pixels).
        transfection: Optional aligned marker intensity for cell filtering.
        transfection_detection: Marker threshold settings (default Otsu, 2 pixels).
        remove_nuclear: Exclude nuclear pixels from puncta and analyzed intensities.
        require_nucleus: Retain only cells overlapping detected nuclear pixels.
        require_transfection: Retain only cells overlapping the detected marker.
        require_aggregates: Require puncta in at least one measured signal.
        exclude_border: Remove cells touching the image frame.
        radial: Calculate per-cell radial profiles when True.
        radial_center: 'auto' (nucleus if supplied), 'cell' or 'nucleus'.
        radial_bins: Positive number of shape-adapted centre-to-boundary sections.
        pixel_size_um: Optional square-pixel width; adds area columns in um^2.

    Returns:
        Dictionary of cell/object/radial/filter records, label arrays, intensities,
        thresholds, radial distributions and effective settings. See docs/api.md.
    """
    cells = np.asarray(cells)
    if cells.ndim != 2 or not np.issubdtype(cells.dtype, np.integer) or (cells < 0).any():
        raise ValueError("Cell segmentation must be a 2-D nonnegative integer label image.")
    all_labels = [int(value) for value in np.unique(cells) if value > 0]
    if not all_labels:
        raise ValueError("No cells were found. Check segmentation channels and Cellpose settings, then rerun segmentation.")
    if not signals:
        raise ValueError("Tick 'Measure aggregates / puncta' for at least one channel.")
    if any(not isinstance(name, str) or not name.strip() for name in signals):
        raise ValueError("Measured signals need nonempty names.")
    if detection is not None and set(detection) != set(signals):
        raise ValueError("Supply detection settings for every measured signal, and no others.")
    for image in [*signals.values(), *([] if nucleus is None else [nucleus]),
                  *([] if transfection is None else [transfection])]:
        if np.shape(image) != cells.shape or not np.isrealobj(image) or not np.isfinite(image).all():
            raise ValueError("All analysis channels must contain finite values and match the cell image shape.")
    if nucleus is None and (remove_nuclear or require_nucleus or (radial and radial_center == "nucleus")):
        raise ValueError("Select a nucleus channel or turn off nucleus-only options; cell-centered radial analysis needs no nucleus.")
    if require_transfection and transfection is None:
        raise ValueError("Select a transfection channel or untick the transfection filter.")
    if radial_center not in {"auto", "cell", "nucleus"}:
        raise ValueError("Radial center must be auto, cell or nucleus.")
    if not isinstance(radial_bins, int) or radial_bins < 1:
        raise ValueError("Radial bins must be a positive integer.")
    if pixel_size_um is not None and (not np.isfinite(pixel_size_um) or pixel_size_um <= 0):
        raise ValueError("Pixel size must be positive, or leave it unset for pixel units.")
    center = ("nucleus" if nucleus is not None else "cell") if radial_center == "auto" else radial_center
    detection = detection or {name: DetectionSettings() for name in signals}
    nuc_binary = np.zeros(cells.shape, dtype=bool)
    nuc_threshold = trans_threshold = None
    if nucleus is not None:
        nuc_binary, _, nuc_threshold = detect_signal(nucleus, cells, nucleus_detection or DetectionSettings(min_size=5))
    nuc_labels = label(np.where(nuc_binary, cells, 0))
    trans_binary = np.zeros(cells.shape, dtype=bool)
    if transfection is not None:
        trans_binary, _, trans_threshold = detect_signal(transfection, cells, transfection_detection or DetectionSettings())
    signal_masks, thresholds = {}, {}
    for name, image in signals.items():
        binary, _, thresholds[name] = detect_signal(image, cells, detection[name])
        if remove_nuclear:
            binary &= ~nuc_binary
        # Objects touching across cell boundaries remain separate instances.
        signal_masks[name] = label(np.where(binary, cells, 0))
    any_aggregate = np.logical_or.reduce([mask > 0 for mask in signal_masks.values()])
    border_labels = set(np.concatenate((cells[0], cells[-1], cells[:, 0], cells[:, -1])))
    audit, retained = [], []
    for cell_id in all_labels:
        region = cells == cell_id
        has_nucleus = bool(nuc_binary[region].any()) if nucleus is not None else None
        has_transfection = bool(trans_binary[region].any()) if transfection is not None else None
        has_aggregate = bool(any_aggregate[region].any())
        reasons = []
        if require_nucleus and not has_nucleus:
            reasons.append("no nucleus")
        if require_transfection and not has_transfection:
            reasons.append("no transfection signal")
        if require_aggregates and not has_aggregate:
            reasons.append("no aggregates in any selected signal")
        if exclude_border and cell_id in border_labels:
            reasons.append("touches image border")
        if not reasons:
            retained.append(cell_id)
        audit.append(dict(cell_label=cell_id, has_nucleus=has_nucleus, has_transfection=has_transfection,
                          has_aggregate=has_aggregate, touches_border=cell_id in border_labels,
                          retained=not reasons, reason="; ".join(reasons)))
    cell_records, objects, radial_records, distributions, intensities = [], [], [], {}, {}
    area_factor = None if pixel_size_um is None else pixel_size_um ** 2
    for name, image in signals.items():
        agg_labels = signal_masks[name]
        intensity = image.copy()
        if remove_nuclear:
            intensity[nuc_binary] = 0
        intensities[name] = intensity
        by_cell = {cell_id: [] for cell_id in retained}
        for obj in regionprops(agg_labels, intensity_image=image):
            row, col = obj.coords[0]
            cell_id = int(cells[row, col])
            if cell_id not in by_cell:
                continue
            record = dict(signal=name, cell_label=cell_id, aggregate_label=int(obj.label),
                          area_pixels=int(obj.area), centroid_row=float(obj.centroid[0]),
                          centroid_column=float(obj.centroid[1]),
                          aspect_ratio=float(obj.axis_major_length / obj.axis_minor_length) if obj.axis_minor_length else float("inf"),
                          circularity=float(4 * np.pi * obj.area / obj.perimeter**2) if obj.perimeter else 0.0,
                          intensity_sum=float(image[tuple(obj.coords.T)].astype(np.float64).sum()),
                          intensity_mean=float(obj.intensity_mean),
                          area_um2=None if area_factor is None else float(obj.area * area_factor))
            objects.append(record)
            by_cell[cell_id].append(record)
        distributions[name] = {}
        if radial:
            if center == "cell":
                distributions[name] = compute_cell_centered_distribution(cells, agg_labels, intensity, radial_bins, retained)
            else:
                centroids = nucleus_centroids_by_cell(cells, nuc_binary, retained)
                distributions[name] = compute_nucleus_centered_distribution(
                    cells, centroids, agg_labels, intensity, radial_bins, list(centroids))
        for item in compute_cell_metrics(cells, nuc_labels, agg_labels, retained):
            region = cells == item.label
            selected = (agg_labels > 0) & region
            values = image[selected].astype(np.float64)
            items = by_cell[item.label]
            nucleus_area = item.nuc_area if nucleus is not None else None
            cell_records.append(dict(
                signal=name, cell_label=item.label, cell_area_pixels=item.cell_area,
                nucleus_area_pixels=nucleus_area,
                nucleus_count=len(np.unique(nuc_labels[region & nuc_binary])) if nucleus is not None else None,
                aggregate_count=len(items), aggregate_area_pixels=item.agg_area,
                aggregate_coverage_fraction=item.jaccard,
                cell_aspect_ratio=item.cell_aspect_ratio, cell_circularity=item.cell_circularity,
                aggregate_mean_aspect_ratio=float(np.mean([obj["aspect_ratio"] for obj in items])) if items else 0.0,
                aggregate_mean_circularity=float(np.mean([obj["circularity"] for obj in items])) if items else 0.0,
                aggregate_mean_area_pixels=float(np.mean([obj["area_pixels"] for obj in items])) if items else 0.0,
                aggregate_intensity_sum=float(values.sum()), aggregate_intensity_mean=float(values.mean()) if values.size else 0.0,
                cell_intensity_sum_raw=float(image[region].astype(np.float64).sum()),
                cell_intensity_mean_raw=float(image[region].mean()),
                cell_intensity_sum_analyzed=float(intensity[region].astype(np.float64).sum()),
                nucleus_intensity_sum_raw=float(image[region & nuc_binary].astype(np.float64).sum()) if nucleus is not None else None,
                radial_center=center if radial else None, radial_included=item.label in distributions[name],
                cell_area_um2=None if area_factor is None else item.cell_area * area_factor,
                nucleus_area_um2=None if area_factor is None or nucleus_area is None else nucleus_area * area_factor,
                aggregate_area_um2=None if area_factor is None else item.agg_area * area_factor,
            ))
        for record in radial_distributions_to_records({"sample": distributions[name]}):
            radial_records.append(dict(signal=name, center=center, **record))
    analysis_options = dict(remove_nuclear=remove_nuclear, require_nucleus=require_nucleus,
                            require_transfection=require_transfection, require_aggregates=require_aggregates,
                            exclude_border=exclude_border, radial=radial, radial_center=radial_center,
                            radial_bins=radial_bins, pixel_size_um=pixel_size_um,
                            nucleus_supplied=nucleus is not None, transfection_supplied=transfection is not None,
                            nucleus_detection=asdict(nucleus_detection or DetectionSettings(min_size=5)),
                            transfection_detection=asdict(transfection_detection or DetectionSettings()))
    return dict(cells=cells, nucleus_labels=nuc_labels, transfection_mask=trans_binary,
                analysis_options=analysis_options,
                signal_masks=signal_masks, intensities=intensities, retained_labels=retained,
                cell_records=cell_records, object_records=objects, radial_records=radial_records,
                filter_records=audit, distributions=distributions, center=center,
                thresholds=dict(signals=thresholds, nucleus=nuc_threshold, transfection=trans_threshold),
                detection_settings={name: asdict(value) for name, value in detection.items()})
