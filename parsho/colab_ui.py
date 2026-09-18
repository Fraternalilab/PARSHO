"""Optional widgets used by PARSHO_Colab.ipynb (requires ipywidgets).

Google Colab imports are deferred to the upload buttons so the forms can also
be inspected and tested in Jupyter without Google services.
"""

import html
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from parsho.single_image import DetectionSettings, load_field_channels


def control(widget_type, description, **kwargs):
    return widget_type(description=description, style={"description_width": "initial"},
                       layout=widgets.Layout(width="95%", max_width="850px"),
                       tooltip=description, **kwargs)


class FieldPicker:
    """Upload one multichannel image or aligned files; inspect selected planes."""

    def __init__(self, title="Sample"):
        self.title = title
        self.is_demo = False
        self.channels = []
        self.metadata = []
        self.loaded_signature = None
        self.rows = []
        self.output = widgets.Output()
        self.file_box = widgets.VBox()
        upload = widgets.Button(description=f"Upload {title.lower()} files", button_style="info",
                                layout=widgets.Layout(width="240px"))
        upload.on_click(self._upload)
        demo = widgets.Button(description="Try the supplied demo", button_style="",
                              layout=widgets.Layout(width="240px"))
        demo.on_click(self._demo)
        self.paths = widgets.Textarea(placeholder="Optional: one mounted Google Drive file path per line",
                                     layout=widgets.Layout(width="95%", height="65px"))
        use_paths = widgets.Button(description="Use these file paths", layout=widgets.Layout(width="200px"))
        use_paths.on_click(lambda _: self.set_paths(self.paths.value.splitlines()))
        drive = widgets.Accordion(children=[widgets.VBox([self.paths, use_paths])], selected_index=None)
        drive.set_title(0, "Optional: files already in a mounted Google Drive")
        self.load_button = widgets.Button(description="Load and preview channels", button_style="success",
                                          layout=widgets.Layout(width="250px"))
        self.load_button.on_click(self._load)
        self.widget = widgets.VBox([
            widgets.HTML(f"<b>{html.escape(title)} — one field of view</b>"),
            widgets.HBox([upload, demo]) if title == "Sample" else upload, drive,
            self.file_box, self.load_button, self.output,
        ])

    def _upload(self, _):
        from google.colab import files

        with self.output:
            clear_output(wait=True)
            uploaded = files.upload()
        if uploaded:
            self.set_paths([str(Path(name).resolve()) for name in uploaded])
        # files.upload has already saved the bytes; do not retain another copy.

    def _demo(self, _):
        from parsho.examples import demo_files

        with self.output:
            clear_output(wait=True)
            try:
                self.set_paths(demo_files())
                self.load()
                self.is_demo = True
                print("Demo roles: channel 1 = cells, channel 2 = aggregates, channel 3 = nuclei.")
            except Exception as error:
                print(f"Demo could not be loaded: {error}. You can still upload your own files.")

    def set_paths(self, paths):
        self.is_demo = False
        paths = [Path(str(value).strip()).expanduser() for value in paths if str(value).strip()]
        self.rows = []
        self.channels = []
        self.loaded_signature = None
        boxes = []
        for path in paths:
            controls = dict(
                scene=control(widgets.BoundedIntText, "Image / series (starts at 1)", value=1, min=1, max=100000),
                time=control(widgets.BoundedIntText, "Time point (starts at 1)", value=1, min=1, max=100000),
                z_mode=control(widgets.Dropdown, "Depth", options=[("Maximum Z projection", "maximum"), ("One Z plane", "plane")]),
                z=control(widgets.BoundedIntText, "Z plane (used for one plane)", value=1, min=1, max=100000),
                axes=control(widgets.Combobox, "Dimension order", value="", options=["YX", "CYX", "YXC", "ZYX", "TYX", "CZYX", "ZCYX", "TCZYX"],
                             placeholder="Automatic from file metadata"),
            )
            self.rows.append((path, controls))
            advanced = widgets.Accordion(children=[widgets.VBox(list(controls.values()))], selected_index=None)
            advanced.set_title(0, "Image series, time, Z and dimension order (usually leave unchanged)")
            boxes.append(widgets.VBox([widgets.HTML(f"<b>{html.escape(path.name)}</b>"), advanced]))
        self.file_box.children = boxes
        with self.output:
            clear_output(wait=True)
            print(f"{len(paths)} file(s) selected. Click Load and preview channels.")

    def signature(self):
        return [(str(path), {name: widget.value for name, widget in row.items()}) for path, row in self.rows]

    def _load(self, _):
        self.load_button.disabled = True
        with self.output:
            clear_output(wait=True)
            try:
                self.load()
            except Exception as error:
                print(f"Could not load images: {error}")
                print("Check the selected files and dimension settings, then click Load again.")
            finally:
                self.load_button.disabled = False

    def load(self):
        self.channels, self.metadata, self.loaded_signature = [], [], None
        if not self.rows:
            raise ValueError("Upload an image, or provide a mounted Drive path first.")
        channels, metadata = [], []
        for path, row in self.rows:
            arrays, info = load_field_channels(
                path, scene=row["scene"].value - 1, time=row["time"].value - 1,
                z_mode=row["z_mode"].value, z=row["z"].value - 1, axes=row["axes"].value,
            )
            metadata.append(info)
            for index, array in enumerate(arrays):
                channels.append(dict(name=f"{path.name} / channel {index + 1}", image=array))
            print(f"{path.name}: stored {info['stored_shape']} ({info['stored_axes']}); {len(arrays)} channel(s), {arrays[0].shape}")
        if len({item["image"].shape for item in channels}) != 1:
            raise ValueError("These channels have different image sizes. Select aligned files from the same field and matching planes.")
        self.channels, self.metadata = channels, metadata
        self.loaded_signature = self.signature()
        columns = min(3, len(channels))
        fig, axes = plt.subplots((len(channels) + columns - 1) // columns, columns,
                                 figsize=(5 * columns, 4 * ((len(channels) + columns - 1) // columns)), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for index, (axis, item) in enumerate(zip(axes.ravel(), channels), 1):
            image = item["image"]
            low, high = np.percentile(image, (1, 99))
            axis.imshow(image, cmap="gray", vmin=low, vmax=high if high > low else low + 1)
            axis.set_title(f"Channel {index}: {item['name']}\n{image.dtype}", fontsize=9)
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        print("Preview contrast is adjusted for visibility; measurements use the original pixel values.")
        print("Check alignment and channel identity, then continue to the next numbered step.")

    def require_loaded(self):
        if not self.channels or self.loaded_signature != self.signature():
            raise ValueError(f"Load and preview the {self.title.lower()} channels before continuing.")


class ThresholdForm:
    def __init__(self, min_size=2, controls=False):
        options = [("Otsu (automatic)", "otsu"), ("Percentile", "percentile"),
                   ("Local / adaptive", "local"), ("Manual intensity", "manual")]
        if controls:
            options.append(("Positive / negative controls", "controls"))
        self.fields = dict(
            method=control(widgets.Dropdown, "Threshold method", options=options),
            min_size=control(widgets.BoundedIntText, "Minimum object area (pixels)", value=min_size, min=1, max=10000000),
            scale=control(widgets.FloatText, "Otsu threshold multiplier", value=1.0),
            percentile=control(widgets.BoundedFloatText, "Percentile (0–100)", value=95.0, min=0, max=100),
            block_size=control(widgets.IntText, "Local window width (odd pixels)", value=51),
            manual_threshold=control(widgets.FloatText, "Manual threshold (raw intensity)", value=1000.0),
        )
        self.fields["method"].observe(self._refresh, names="value")
        self._refresh()
        self.widget = widgets.VBox(list(self.fields.values()))

    def _refresh(self, *_):
        method = self.fields["method"].value
        for name, relevant in [("scale", "otsu"), ("percentile", "percentile"),
                               ("block_size", "local"), ("manual_threshold", "manual")]:
            self.fields[name].disabled = method != relevant

    def values(self):
        return {name: widget.value for name, widget in self.fields.items()}

    def settings(self, calibrated=None):
        values = self.values()
        if values["method"] == "controls":
            if calibrated is None:
                raise ValueError("Load and assign positive and negative controls in step 4.")
            values.update(method="manual", manual_threshold=calibrated)
        settings = DetectionSettings(**values)
        settings.validate()
        return settings


class AnalysisForm:
    """Channel identities and all editable analysis parameters in labelled tabs."""

    def __init__(self, picker):
        picker.require_loaded()
        self.picker = picker
        self.source_channels = picker.channels
        self.rows, self.thresholds = [], []
        role_rows = [widgets.HTML("<b>For each channel, enter a biological name and tick its uses. A channel may have both uses.</b>")]
        for index, item in enumerate(picker.channels):
            name = control(widgets.Text, f"Channel {index + 1} name", value=f"Signal {index + 1}")
            segmentation = control(widgets.Checkbox, "Use for cell segmentation", value=False)
            signal = control(widgets.Checkbox, "Measure aggregates / puncta", value=False)
            self.rows.append(dict(name=name, segmentation=segmentation, signal=signal))
            role_rows.append(widgets.VBox([widgets.HTML(html.escape(item["name"])), name,
                                          segmentation, signal]))
            self.thresholds.append(ThresholdForm(controls=True))
        options = [("None — no channel available", None)] + [(f"Channel {i + 1}: {item['name']}", i) for i, item in enumerate(picker.channels)]
        self.nucleus = control(widgets.Dropdown, "Nucleus channel (optional)", options=options)
        self.transfection = control(widgets.Dropdown, "Transfection marker (optional)", options=options)
        self.nucleus_threshold = ThresholdForm(min_size=5)
        self.transfection_threshold = ThresholdForm()
        self.segmentation = dict(
            combination=control(widgets.Dropdown, "Combine segmentation channels", options=[("Mean", "mean"), ("Maximum", "maximum"), ("Keep channels separate (up to 3)", "stack")]),
            diameter=control(widgets.FloatText, "Expected cell diameter (pixels; 0 = automatic)", value=0),
            flow_threshold=control(widgets.FloatText, "Flow threshold", value=0.4),
            cellprob_threshold=control(widgets.FloatText, "Cell probability threshold", value=0.0),
            min_size=control(widgets.BoundedIntText, "Minimum cell area (pixels)", value=15, min=1, max=10000000),
            batch_size=control(widgets.BoundedIntText, "Cellpose tile batch size", value=8, min=1, max=256),
        )
        self.options = dict(
            remove_nuclear=control(widgets.Checkbox, "Exclude nuclear pixels from aggregates AND analyzed intensity", value=False),
            require_nucleus=control(widgets.Checkbox, "Keep only cells with a detected nucleus", value=False),
            require_transfection=control(widgets.Checkbox, "Keep only cells with the transfection marker", value=False),
            require_aggregates=control(widgets.Checkbox, "Keep only cells with puncta in at least one measured channel", value=False),
            exclude_border=control(widgets.Checkbox, "Exclude cells touching the image border", value=False),
            radial=control(widgets.Checkbox, "Calculate radial distributions", value=True),
            radial_center=control(widgets.Dropdown, "Radial centre", options=[("Automatic: nucleus if available, otherwise cell", "auto"), ("Cell centre", "cell"), ("Nucleus centre", "nucleus")]),
            radial_bins=control(widgets.BoundedIntText, "Number of radial bins", value=10, min=1, max=1000),
            pixel_size_um=control(widgets.FloatText, "Pixel width in micrometres (0 = unknown)", value=0),
        )
        self.save_all_radial = control(widgets.Checkbox, "Save a radial figure for every retained cell and signal", value=True)
        detection = widgets.Accordion(children=[form.widget for form in self.thresholds] +
                                        [self.nucleus_threshold.widget, self.transfection_threshold.widget], selected_index=None)
        for i in range(len(self.thresholds)):
            detection.set_title(i, f"Channel {i + 1}: puncta detection (only used if ticked for measurement)")
            def rename(change, index=i):
                detection.set_title(index, f"Channel {index + 1}: {change['new']} — puncta detection")
            self.rows[i]["name"].observe(rename, names="value")
        detection.set_title(len(self.thresholds), "Nucleus detection (only with a nucleus channel)")
        detection.set_title(len(self.thresholds) + 1, "Transfection detection (only with a marker channel)")
        self.nucleus.observe(self._refresh_optional, names="value")
        self.transfection.observe(self._refresh_optional, names="value")
        self._refresh_optional()
        tabs = widgets.Tab(children=[
            widgets.VBox(role_rows + [self.nucleus, self.transfection]),
            widgets.VBox([widgets.HTML("Default model: the installed Cellpose-SAM model. Normalization applies only to segmentation copies.<br>Lower cell probability may detect dimmer cells. Lower flow threshold rejects more irregular masks.<br>Tile batch size controls GPU memory, not the number of input images.")] + list(self.segmentation.values())),
            widgets.VBox([widgets.HTML("Start with Otsu. Only the fields used by the chosen method are enabled. Minimum area is in pixels.<br>Local window must be odd. Manual thresholds use original image intensities. Control calibration uses the same minimum area and nuclear exclusion."), detection]),
            widgets.VBox([widgets.HTML("All filters are optional. Without a nucleus, automatic radial analysis uses the cell centre.<br>With nucleus centring, cells without detected nuclear pixels stay in the cell table but have no radial rows.<br>Radial intensity is measured in detected puncta only. Pixel size assumes square pixels; leave 0 for pixel units.")] + list(self.options.values()) + [self.save_all_radial]),
        ])
        for i, title in enumerate(["1. Channel roles", "2. Cell segmentation", "3. Signal detection", "4. Filtering, radial & export"]):
            tabs.set_title(i, title)
        self.widget = tabs

    def _refresh_optional(self, *_):
        for name, enabled in [("remove_nuclear", self.nucleus.value is not None),
                              ("require_nucleus", self.nucleus.value is not None),
                              ("require_transfection", self.transfection.value is not None)]:
            self.options[name].disabled = not enabled
            if not enabled:
                self.options[name].value = False

    def snapshot(self):
        self.picker.require_loaded()
        if self.source_channels is not self.picker.channels:
            raise ValueError("Images were reloaded. Rerun step 3 to assign the new channels.")
        return dict(roles=[{key: widget.value for key, widget in row.items()} for row in self.rows],
                    nucleus=self.nucleus.value, transfection=self.transfection.value,
                    segmentation={key: widget.value for key, widget in self.segmentation.items()},
                    detection=[form.values() for form in self.thresholds],
                    nucleus_detection=self.nucleus_threshold.values(), transfection_detection=self.transfection_threshold.values(),
                    options={key: widget.value for key, widget in self.options.items()},
                    save_all_radial=self.save_all_radial.value, sources=self.picker.metadata)

    def selected(self):
        snapshot = self.snapshot()
        segmentation, signals = [], {}
        for index, row in enumerate(snapshot["roles"]):
            if row["segmentation"]:
                segmentation.append(index)
            if row["signal"]:
                name = row["name"].strip()
                if not name or name in signals:
                    raise ValueError("Give every measured signal a different non-empty name.")
                signals[name] = index
        if not segmentation or not signals:
            raise ValueError("In Channel roles, tick at least one segmentation channel and one measured puncta channel.")
        return segmentation, signals

    def restore(self, snapshot):
        """Restore edits when the same loaded channels' form is redisplayed."""
        if snapshot["sources"] != self.picker.metadata or len(snapshot["roles"]) != len(self.rows):
            raise ValueError("Saved settings belong to different input channels.")
        for row, saved in zip(self.rows, snapshot["roles"]):
            for name, value in saved.items():
                row[name].value = value
        self.nucleus.value = snapshot["nucleus"]
        self.transfection.value = snapshot["transfection"]
        for fields, values in [(self.segmentation, snapshot["segmentation"]), (self.options, snapshot["options"])]:
            for name, value in values.items():
                fields[name].value = value
        for form, values in zip(self.thresholds, snapshot["detection"]):
            for name, value in values.items():
                form.fields[name].value = value
        for form, key in [(self.nucleus_threshold, "nucleus_detection"), (self.transfection_threshold, "transfection_detection")]:
            for name, value in snapshot[key].items():
                form.fields[name].value = value
        self.save_all_radial.value = snapshot["save_all_radial"]
        self._refresh_optional()


class ControlAssignment:
    """Map a control's channels to the named sample signals without filenames."""

    def __init__(self, picker, signal_names, use_nucleus=False):
        picker.require_loaded()
        self.picker = picker
        self.source_channels = picker.channels
        choices = [(item["name"], index) for index, item in enumerate(picker.channels)]
        self.segmentation = [control(widgets.Checkbox, item["name"], value=False) for item in picker.channels]
        self.signals = {name: control(widgets.Dropdown, f"Puncta: {name}", options=[("Choose a channel", None)] + choices) for name in signal_names}
        self.nucleus = control(widgets.Dropdown, "Nucleus channel", options=[("Choose a channel", None)] + choices) if use_nucleus else None
        self.widget = widgets.VBox([widgets.HTML(f"<b>{html.escape(picker.title)}: tick segmentation channels and match the measured signals</b>")] +
                                   self.segmentation + list(self.signals.values()) + ([] if self.nucleus is None else [self.nucleus]))

    def snapshot(self):
        self.picker.require_loaded()
        if self.source_channels is not self.picker.channels:
            raise ValueError("Control images were reloaded; rerun the control assignment cell.")
        indices = [index for index, widget in enumerate(self.segmentation) if widget.value]
        signals = {name: widget.value for name, widget in self.signals.items()}
        if not indices or any(value is None for value in signals.values()) or (self.nucleus is not None and self.nucleus.value is None):
            raise ValueError(f"Complete segmentation, signal and nucleus assignments for the {self.picker.title.lower()}.")
        return dict(segmentation=indices, signals=signals,
                    nucleus=None if self.nucleus is None else self.nucleus.value, sources=self.picker.metadata)
