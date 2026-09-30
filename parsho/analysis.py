"""Convenient, reusable entry point for segmentation and puncta analysis.

Only segmentation imports Cellpose. Measurements use ``analyze_field`` and
exports use the same tables and images as the guided Colab workflow.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from parsho.img_utils import load_mask_npy
from parsho.segmentation import find_optimal_threshold
from parsho.single_image import (
    DetectionSettings,
    analyze_field,
    combine_segmentation_channels,
    detect_signal,
    load_field_channels,
)

Channel = str | int
Selection = Channel | Sequence[Channel]
Detection = str | DetectionSettings


def _detection(value, *, minimum=2):
    settings = DetectionSettings(method=value, min_size=minimum) if isinstance(value, str) else value
    if not isinstance(settings, DetectionSettings):
        raise ValueError("Detection must be a method name or DetectionSettings object.")
    settings.validate()
    return settings


def _selection(value):
    if isinstance(value, (str, int, np.integer)):
        return [value]
    try:
        return list(value)
    except TypeError as error:
        raise ValueError("Select a channel name/index or a list of channels.") from error


def _channel_name(key, names):
    if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
        if 0 <= key < len(names):
            return names[key]
    elif isinstance(key, str) and key in names:
        return key
    raise ValueError(f"Unknown channel {key!r}. Available channels: {names}; indices start at 0.")


def _load_channels(data, load_options):
    """Resolve a multichannel file or named/ordered single-channel inputs."""
    options = dict(load_options or {})
    sources = {}
    if isinstance(data, (str, Path)):
        images, metadata = load_field_channels(data, **options)
        channels = {f"channel_{index}": image for index, image in enumerate(images)}
        sources = {name: dict(metadata, path=str(data), channel=index)
                   for index, name in enumerate(channels)}
    else:
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("For array inputs, pass a list or dictionary of 2-D channels.")
            data = [data]
        if not isinstance(data, (Mapping, Sequence)):
            raise ValueError("Supply an image path, a dictionary of named channels, or a list of channels.")
        items = data.items() if isinstance(data, Mapping) else (
            (f"channel_{index}", value) for index, value in enumerate(data)
        )
        channels = {}
        for name, value in items:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Channel names must be nonempty strings.")
            if isinstance(value, (str, Path)):
                images, metadata = load_field_channels(value, **options)
                if len(images) != 1:
                    raise ValueError(
                        f"Channel {name!r} has {len(images)} channels. Pass a multichannel file "
                        "directly, or select its channels with load_field_channels() first."
                    )
                channels[name] = images[0]
                sources[name] = dict(metadata, path=str(value), channel=0)
            else:
                channels[name] = np.asarray(value)
                sources[name] = dict(source="array", shape=list(channels[name].shape))
    if not channels:
        raise ValueError("Supply at least one image channel.")
    shape = next(iter(channels.values())).shape
    for name, image in channels.items():
        if (image.ndim != 2 or image.size == 0 or image.shape != shape
                or not np.issubdtype(image.dtype, np.number) or not np.isrealobj(image)
                or not np.isfinite(image).all()):
            raise ValueError(f"Channel {name!r}: all channels must be aligned, finite, real 2-D images.")
    return channels, sources


def _new_output(output_dir):
    if output_dir is None:
        return None
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"Output directory already exists: {path}. Choose a new directory.")
    return path


@dataclass(kw_only=True)
class Analysis:
    """Configure once, then call :meth:`run` for each field of view.

    Channels can be names in an input dictionary or zero-based indices. The
    defaults segment with channel 0 and measure channel 1. A single name/index
    or a list is accepted for ``segmentation_channels`` and ``signal_channels``.

    ``detection`` accepts 'otsu' (default), 'percentile', 'local', or a
    DetectionSettings object (including method='manual'). A dictionary keyed
    by measured channel name/index sets separate detection rules per signal.
    Nuclei default to Otsu with a minimum area of 5 pixels; puncta and the
    transfection marker default to 2 pixels.

    Segmentation copies are normalized and combined by 'mean', 'maximum', or
    'stack' (up to three channels). Raw measurement intensities are preserved.
    ``cellpose_parameters`` overrides the Cellpose eval defaults documented in
    docs/analysis.md. ``gpu=None`` automatically selects available hardware;
    ``gpu=False`` forces CPU. ``model`` optionally supplies an existing model.

    All cell filters and nuclear subtraction default to False. Radial analysis
    defaults to 10 bins, centred on nuclei when supplied, otherwise on cells.
    Set ``radial=False`` to skip it. ``pixel_size_um`` optionally adds area
    measurements in square micrometres. See docs/analysis.md for examples.
    """

    segmentation_channels: Selection = (0,)
    signal_channels: Selection = (1,)
    nucleus_channel: Channel | None = None
    transfection_channel: Channel | None = None
    detection: Detection | Mapping[Channel, Detection] = "otsu"
    nucleus_detection: Detection = field(default_factory=lambda: DetectionSettings(min_size=5))
    transfection_detection: Detection = field(default_factory=DetectionSettings)
    segmentation_method: str = "mean"
    cellpose_parameters: dict = field(default_factory=dict)
    gpu: bool | None = None
    model: object | None = field(default=None, repr=False)
    remove_nuclear: bool = False
    require_nucleus: bool = False
    require_transfection: bool = False
    require_aggregates: bool = False
    exclude_border: bool = False
    radial: bool = True
    radial_center: str = "auto"
    radial_bins: int = 10
    pixel_size_um: float | None = None
    _calibration: dict | None = field(default=None, init=False, repr=False)

    def _options(self):
        return {name: getattr(self, name) for name in (
            "remove_nuclear", "require_nucleus", "require_transfection", "require_aggregates",
            "exclude_border", "radial", "radial_center", "radial_bins", "pixel_size_um",
        )}

    def _prepare(self, data, cell_masks, load_options):
        for name in ("remove_nuclear", "require_nucleus", "require_transfection",
                     "require_aggregates", "exclude_border", "radial"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False.")
        channels, sources = _load_channels(data, load_options)
        names = list(channels)
        segmentation = [_channel_name(key, names) for key in _selection(self.segmentation_channels)]
        measured = [_channel_name(key, names) for key in _selection(self.signal_channels)]
        if not measured or len(set(measured)) != len(measured):
            raise ValueError("Select at least one signal channel, without duplicates.")
        if len(set(segmentation)) != len(segmentation):
            raise ValueError("Select segmentation channels without duplicates.")
        nucleus = None if self.nucleus_channel is None else _channel_name(self.nucleus_channel, names)
        transfection = None if self.transfection_channel is None else _channel_name(self.transfection_channel, names)
        if nucleus is None and (self.remove_nuclear or self.require_nucleus
                                or (self.radial and self.radial_center == "nucleus")):
            raise ValueError("Select a nucleus_channel for nuclear subtraction, filtering or nucleus-centred analysis.")
        if self.require_transfection and transfection is None:
            raise ValueError("Select a transfection_channel to require transfection signal.")
        if self.radial_center not in {"auto", "cell", "nucleus"}:
            raise ValueError("radial_center must be 'auto', 'cell' or 'nucleus'.")
        if isinstance(self.radial_bins, bool) or not isinstance(self.radial_bins, int) or self.radial_bins < 1:
            raise ValueError("radial_bins must be a positive integer.")
        if self.pixel_size_um is not None and (not np.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0):
            raise ValueError("pixel_size_um must be positive and finite, or None.")
        if self.gpu is not None and not isinstance(self.gpu, bool):
            raise ValueError("gpu must be True, False or None (automatic).")
        if self.segmentation_method not in {"mean", "maximum", "stack"}:
            raise ValueError("segmentation_method must be 'mean', 'maximum' or 'stack'.")
        if isinstance(self.detection, Mapping):
            detection = {}
            for key, value in self.detection.items():
                name = _channel_name(key, names)
                if name in detection:
                    raise ValueError(f"Duplicate detection settings for channel {name!r}.")
                detection[name] = _detection(value)
            if set(detection) != set(measured):
                raise ValueError("Supply detection settings for every selected signal channel, and no others.")
        else:
            detection = {name: _detection(self.detection) for name in measured}
        nucleus_detection = _detection(self.nucleus_detection, minimum=5)
        transfection_detection = _detection(self.transfection_detection)
        if not segmentation and cell_masks is None:
            raise ValueError("Select at least one segmentation channel, or supply cell_masks.")
        segmentation_image = (combine_segmentation_channels(
            [channels[name] for name in segmentation], self.segmentation_method
        ) if segmentation else None)
        mask_source = "cellpose"
        if cell_masks is not None:
            mask_source = str(cell_masks) if isinstance(cell_masks, (str, Path)) else "array"
            cell_masks = (load_mask_npy(cell_masks, kind="labels")
                          if isinstance(cell_masks, (str, Path)) else np.asarray(cell_masks))
            self._validate_cells(cell_masks, next(iter(channels.values())).shape)
        return dict(channels=channels, sources=sources, segmentation=segmentation,
                    segmentation_image=segmentation_image, cells=cell_masks, mask_source=mask_source,
                    signals={name: channels[name] for name in measured}, detection=detection,
                    nucleus=None if nucleus is None else channels[nucleus],
                    transfection=None if transfection is None else channels[transfection],
                    nucleus_detection=nucleus_detection, transfection_detection=transfection_detection,
                    roles=dict(segmentation=segmentation, signals=measured,
                               nucleus=nucleus, transfection=transfection))

    @staticmethod
    def _validate_cells(cells, shape):
        if cells.shape != shape or not np.issubdtype(cells.dtype, np.integer) or (cells < 0).any():
            raise ValueError("cell_masks must be aligned 2-D nonnegative integer cell labels.")
        if not np.any(cells):
            raise ValueError("No cells were found. Check segmentation channels and Cellpose settings.")

    def _segment(self, prepared):
        if prepared["cells"] is not None:
            return
        parameters = dict(batch_size=8, diameter=None, flow_threshold=0.4,
                          cellprob_threshold=0.0, min_size=15, normalize=False)
        parameters.update(self.cellpose_parameters)
        if set(parameters) & {"channel_axis", "z_axis", "do_3D", "channels", "x"}:
            raise ValueError("The wrapper manages 2-D/channel axes; set segmentation_channels and segmentation_method instead.")
        for name in ("batch_size", "min_size"):
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"Cellpose {name} must be a positive integer.")
        for name in ("flow_threshold", "cellprob_threshold"):
            if not np.isfinite(parameters[name]):
                raise ValueError(f"Cellpose {name} must be finite.")
        if parameters["flow_threshold"] < 0:
            raise ValueError("Cellpose flow_threshold must be nonnegative.")
        diameter = parameters["diameter"]
        if diameter is not None and (not np.isfinite(diameter) or diameter <= 0):
            raise ValueError("Cellpose diameter must be positive and finite, or None.")
        if parameters.get("compute_masks", True) is not True:
            raise ValueError("Cellpose compute_masks must be True for cell analysis.")
        if self.model is None:
            from cellpose import core, models

            self.model = models.CellposeModel(gpu=core.use_gpu() if self.gpu is None else self.gpu)
        image = prepared["segmentation_image"]
        cells, flows, _ = self.model.eval(image, channel_axis=-1 if image.ndim == 3 else None, **parameters)
        cells = np.asarray(cells)
        self._validate_cells(cells, next(iter(prepared["channels"].values())).shape)
        prepared.update(cells=cells, flows=flows, cellpose_parameters=parameters)

    def _measure(self, prepared, image_id, *, options=None):
        result = analyze_field(
            prepared["cells"], prepared["signals"], detection=prepared["detection"],
            nucleus=prepared["nucleus"], nucleus_detection=prepared["nucleus_detection"],
            transfection=prepared["transfection"], transfection_detection=prepared["transfection_detection"],
            **(self._options() if options is None else options),
        )
        for row in result["radial_records"]:
            row["image_id"] = image_id
        versions = {}
        for package in ("parsho", "cellpose", "numpy", "scikit-image", "tifffile"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not installed"
        result["settings"] = dict(
            image_id=image_id, sources=prepared["sources"], channel_roles=prepared["roles"],
            segmentation_method=self.segmentation_method, cell_mask_source=prepared["mask_source"],
            cellpose_parameters=prepared.get("cellpose_parameters"),
            cellpose_model=str(getattr(self.model, "pretrained_model", "supplied model"))
            if prepared["mask_source"] == "cellpose" else None,
            cellpose_gpu=bool(getattr(self.model, "gpu", False))
            if prepared["mask_source"] == "cellpose" else None,
            detection_settings=result["detection_settings"], calibration=self._calibration,
            versions=versions,
        )
        result["segmentation_image"] = prepared["segmentation_image"]
        result["output_dir"] = None
        return result

    @staticmethod
    def _save(output, result, prepared, save_intermediates, save_radial_plots):
        from parsho.colab_results import export_field_result

        result["output_dir"] = export_field_result(
            output, result, prepared["signals"],
            segmentation_image=prepared["segmentation_image"],
            segmentation_flows=prepared.get("flows"),
            save_intermediates=save_intermediates, save_all_radial=save_radial_plots,
        )

    def run(self, channels, *, cell_masks=None, output_dir=None, save_intermediates=False,
            save_radial_plots=True, image_id="sample", load_options=None):
        """Load, segment, threshold, filter and measure one field.

        ``channels`` is a multichannel file path, a dictionary of named 2-D
        arrays/single-channel paths, or a list of arrays/single-channel paths.
        File scene/time/Z selection can be supplied as ``load_options`` using
        the arguments of load_field_channels(). Array inputs are already 2-D.
        Optional ``cell_masks`` (integer array or .npy path) skips Cellpose.

        No files are written unless ``output_dir`` names a new directory.
        Exports contain final_data.csv, plus radial_distribution.csv when radial
        analysis is enabled. ``save_intermediates`` adds notebook-style figures
        under inspection_images/; ``save_radial_plots=False`` omits per-cell plots.

        Returns the analyze_field() dictionary, with additional ``settings``,
        ``segmentation_image`` and ``output_dir`` entries. Cell/object/filter
        records and masks retain the original cell labels.
        """
        output = _new_output(output_dir)
        if save_intermediates and output is None:
            raise ValueError("Set output_dir when save_intermediates=True.")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("image_id must be a nonempty string.")
        prepared = self._prepare(channels, cell_masks, load_options)
        if self._calibration is not None:
            rules = self._calibration["rules"]
            if (self._calibration["channel_roles"] != prepared["roles"]
                    or rules["remove_nuclear"] != self.remove_nuclear
                    or rules["nucleus_detection"] != asdict(prepared["nucleus_detection"])
                    or rules["detection"] != {name: asdict(value) for name, value in prepared["detection"].items()}):
                raise ValueError("Channel roles, detection or nuclear-exclusion settings changed after calibration. Recalibrate or create a new Analysis.")
        self._segment(prepared)
        result = self._measure(prepared, image_id)
        if output is not None:
            self._save(output, result, prepared, save_intermediates, save_radial_plots)
        return result

    def calibrate_thresholds(self, positive, negative, *, positive_cell_masks=None,
                             negative_cell_masks=None, load_options=None, output_dir=None,
                             save_intermediates=False):
        """Calibrate fixed puncta thresholds and use them on subsequent runs.

        Controls use the same channel roles as samples. The existing threshold
        search finds signal in the positive control and none in the negative,
        using the configured minimum object sizes and nuclear exclusion.
        Otsu thresholds seed a search spanning the control intensity range.
        Failed calibration leaves settings
        unchanged. Returns a dictionary of signal names to threshold values.

        Optional output_dir writes control final_data.csv tables in positive/
        and negative/; save_intermediates also saves their inspection figures.
        """
        output = _new_output(output_dir)
        if save_intermediates and output is None:
            raise ValueError("Set output_dir when save_intermediates=True.")
        controls = [self._prepare(data, masks, load_options) for data, masks in (
            (positive, positive_cell_masks), (negative, negative_cell_masks)
        )]
        pos, neg = controls
        if pos["roles"] != neg["roles"]:
            raise ValueError("Positive and negative controls must use the same channel roles.")
        for prepared in controls:
            self._segment(prepared)
            nuclear = np.zeros(prepared["cells"].shape, dtype=bool)
            if self.remove_nuclear:
                nuclear, _, _ = detect_signal(prepared["nucleus"], prepared["cells"], prepared["nucleus_detection"])
            prepared["nuclear_pixels"] = nuclear
        calibrated, thresholds, positive_pixels = {}, {}, {}
        for name, settings in pos["detection"].items():
            bounds = [float(detect_signal(item["signals"][name], item["cells"],
                                         DetectionSettings(min_size=settings.min_size))[2])
                      for item in controls]
            # Near-zero backgrounds can yield near-zero Otsu cutoffs. The
            # legacy search ends at 10 * t_max, so also cover the observed
            # intensity range instead of stopping below the negative signal.
            upper = max(max(bounds) * 10, *(float(item["signals"][name][item["cells"] > 0].max())
                                            for item in controls))
            try:
                threshold, pixels = find_optimal_threshold(
                    pos["signals"][name], pos["cells"], neg["signals"][name], neg["cells"],
                    pos["nuclear_pixels"], neg["nuclear_pixels"],
                    t_min=min(bounds), t_max=upper / 10, min_size_px=settings.min_size,
                )
            except ValueError as error:
                raise ValueError(f"Could not calibrate {name!r}: {error}. Inspect controls or choose another threshold method.") from error
            thresholds[name], positive_pixels[name] = float(threshold), int(pixels)
            calibrated[name] = DetectionSettings(method="manual", min_size=settings.min_size,
                                                 manual_threshold=float(threshold))
        self.detection = calibrated
        self._calibration = dict(
            positive=pos["sources"], negative=neg["sources"], channel_roles=pos["roles"],
            thresholds=thresholds, positive_pixels=positive_pixels,
            segmentation_method=self.segmentation_method,
            cell_masks={title: item["mask_source"] for title, item in zip(("positive", "negative"), controls)},
            cellpose_parameters={title: item.get("cellpose_parameters")
                                 for title, item in zip(("positive", "negative"), controls)},
            rules=dict(remove_nuclear=self.remove_nuclear,
                       nucleus_detection=asdict(pos["nucleus_detection"]),
                       detection={name: asdict(value) for name, value in calibrated.items()}),
        )
        if output is not None:
            output.mkdir(parents=True, exist_ok=False)
            for title, prepared in zip(("positive", "negative"), controls):
                prepared["detection"] = calibrated
                result = self._measure(prepared, title, options=dict(remove_nuclear=self.remove_nuclear, radial=False))
                self._save(output / title, result, prepared, save_intermediates, False)
        return dict(thresholds)


def analyze(channels, *, cell_masks=None, output_dir=None, save_intermediates=False,
            save_radial_plots=True, image_id="sample", load_options=None, **parameters):
    """Run one field with Analysis defaults; keyword parameters configure Analysis.

    Example::

        result = analyze(
            {"cells": "C3-sample.tif", "titin": "C2-sample.tif"},
            segmentation_channels="cells", signal_channels="titin",
            output_dir="results/sample", save_intermediates=True,
        )

    Reuse an Analysis object for multiple fields to load Cellpose only once.
    """
    return Analysis(**parameters).run(
        channels, cell_masks=cell_masks, output_dir=output_dir,
        save_intermediates=save_intermediates, save_radial_plots=save_radial_plots,
        image_id=image_id, load_options=load_options,
    )
