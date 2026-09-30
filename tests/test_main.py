"""Small, deterministic checks of the public analysis workflow."""

import csv
import os
from pathlib import Path
import sys

# Keep plotting headless and its cache inside this folder.
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent / ".mplconfig")

import numpy as np
from PIL import Image
import pytest
import tifffile

from parsho import (
    Analysis, DetectionSettings, analyze, analyze_field,
    compute_cell_centered_distribution, load_field_channels, load_image,
)
from parsho.segmentation import re_threshold_masks
from parsho.single_image import detect_signal


MANUAL = DetectionSettings(method="manual", manual_threshold=10, min_size=2)


@pytest.fixture
def field():
    """Two touching cells, one shared bright patch, and one small nucleus."""
    cells = np.zeros((20, 24), dtype=np.int32)
    cells[2:18, 2:12] = 3
    cells[2:18, 12:22] = 8
    signal = np.zeros(cells.shape, dtype=np.uint16)
    signal[6:10, 8:16] = 100
    nuclei = np.zeros_like(signal)
    nuclei[6:8, 8:10] = 50
    return cells, {"cells": (cells > 0).astype(np.uint16) * 100,
                   "signal": signal, "nuclei": nuclei}


def test_tiff_loading_selects_time_before_projecting_depth(tmp_path):
    pixels = np.arange(2 * 2 * 3 * 8 * 9, dtype=np.uint16).reshape(2, 2, 3, 8, 9)
    path = tmp_path / "channels.tif"
    tifffile.imwrite(path, pixels, metadata={"axes": "TCZYX"}, photometric="minisblack")

    np.testing.assert_array_equal(load_image(path), pixels)
    planes, _ = load_field_channels(path, time=1, z_mode="plane", z=1)
    projected, _ = load_field_channels(path, time=1)
    np.testing.assert_array_equal(planes, pixels[1, :, 1])
    np.testing.assert_array_equal(projected, pixels[1].max(axis=1))


@pytest.mark.parametrize("method", ["otsu", "percentile", "local", "manual"])
def test_thresholds_find_signal_and_split_objects_at_cell_boundaries(field, method):
    cells, channels = field
    signal = channels["signal"].copy()
    signal[0, 0] = 255  # Bright background must not become a cell aggregate.
    settings = DetectionSettings(method=method, percentile=50, block_size=7,
                                 manual_threshold=10, min_size=2)
    binary, labels, _ = detect_signal(signal, cells, settings)

    np.testing.assert_array_equal(binary, (signal > 0) & (cells > 0))
    assert len(np.unique(labels[labels > 0])) == 2
    assert labels[6, 11] != labels[6, 12]
    assert signal[0, 0] == 255


def test_minimum_object_size_applies_within_each_cell(field):
    cells, channels = field
    # The 32-pixel patch crosses a boundary: each cell contains only 16 pixels.
    binary, labels = re_threshold_masks(channels["signal"], cells, thresh=10, min_size_px=20)
    assert not binary.any()
    assert not labels.any()


@pytest.mark.parametrize("required", ["require_nucleus", "require_transfection"])
def test_filtering_nuclear_subtraction_and_measurements(field, required):
    cells, channels = field
    result = analyze_field(
        cells, {"signal": channels["signal"]}, detection={"signal": MANUAL},
        nucleus=channels["nuclei"], nucleus_detection=MANUAL,
        transfection=channels["nuclei"], transfection_detection=MANUAL,
        remove_nuclear=True, radial_bins=4, pixel_size_um=0.5, **{required: True},
    )

    assert result["retained_labels"] == [3]
    row, = result["cell_records"]
    assert row["cell_area_pixels"] == 160
    assert row["nucleus_count"] == 1
    assert row["nucleus_area_pixels"] == 4
    assert row["aggregate_count"] == 1
    assert row["aggregate_area_pixels"] == 12
    assert row["aggregate_intensity_sum"] == 1200
    assert row["cell_intensity_sum_raw"] == 1600
    assert row["cell_intensity_sum_analyzed"] == 1200
    assert row["cell_area_um2"] == 40
    assert row["aggregate_area_um2"] == 3
    assert row["radial_center"] == "nucleus"
    assert len(result["radial_records"]) == 4
    assert sum(r["intensity_sum"] for r in result["radial_records"]) == 1200
    assert not result["filter_records"][1]["retained"]
    assert channels["signal"].sum() == 3200  # Measurements preserve raw data.


def test_radial_profile_places_centre_and_boundary_in_expected_bins():
    cells = np.full((9, 9), 7, dtype=np.int32)
    signal = np.zeros(cells.shape)
    signal[4, 4] = 20
    signal[0, 4] = 60
    profile = compute_cell_centered_distribution(cells, signal > 0, signal, radial_bins=4)[7]

    np.testing.assert_array_equal(profile.aggregate_pixels, [1, 0, 0, 1])
    np.testing.assert_array_equal(profile.intensity_sum, [20, 0, 0, 60])
    np.testing.assert_allclose(profile.intensity_share, [0.25, 0, 0, 0.75])
    assert profile.cell_pixels.sum() == 81


@pytest.mark.parametrize("method", ["mean", "maximum", "stack"])
def test_analysis_segments_selected_channels_and_reuses_model(field, method):
    cells, channels = field

    class Model:
        """Stand in for Cellpose so tests need no model weights or GPU."""

        def __init__(self):
            self.calls = []

        def eval(self, image, **parameters):
            self.calls.append((image.copy(), parameters))
            return cells.copy(), [], None

    model = Model()
    analysis = Analysis(segmentation_channels=["cells", "signal"], signal_channels="signal",
                        segmentation_method=method, model=model, radial=False)
    originals = {name: image.copy() for name, image in channels.items()}
    for _ in range(2):
        result = analysis.run(channels)
        assert result["retained_labels"] == [3, 8]
        assert [r["aggregate_area_pixels"] for r in result["cell_records"]] == [16, 16]
        assert [r["aggregate_intensity_sum"] for r in result["cell_records"]] == [1600, 1600]
        assert result["radial_records"] == []
        assert result["output_dir"] is None

    assert len(model.calls) == 2
    cell_channel = (channels["cells"] > 0).astype(float)
    signal_channel = (channels["signal"] > 0).astype(float)
    expected = {"mean": (cell_channel + signal_channel) / 2,
                "maximum": np.maximum(cell_channel, signal_channel),
                "stack": np.stack([cell_channel, signal_channel], axis=-1)}[method]
    for image, parameters in model.calls:
        np.testing.assert_allclose(image, expected)
        assert parameters["channel_axis"] == (-1 if method == "stack" else None)
    for name in channels:
        np.testing.assert_array_equal(channels[name], originals[name])


def test_external_masks_and_multiple_signals_need_no_cellpose(field, tmp_path, monkeypatch):
    cells, channels = field
    path = tmp_path / "cells.npy"
    np.save(path, cells)
    monkeypatch.setitem(sys.modules, "cellpose", None)
    result = analyze(
        {"signal": channels["signal"], "double": channels["signal"] * 2},
        cell_masks=path, segmentation_channels=[], signal_channels=["signal", "double"],
        detection=MANUAL, radial_bins=3,
    )

    np.testing.assert_array_equal(result["cells"], cells)
    assert result["retained_labels"] == [3, 8]
    assert len(result["cell_records"]) == 4
    for row in result["cell_records"]:
        assert row["aggregate_intensity_sum"] == (1600 if row["signal"] == "signal" else 3200)
        assert row["radial_center"] == "cell"
    assert len(result["radial_records"]) == 12


def test_control_calibration_keeps_positive_signal_and_rejects_negative(field):
    cells, channels = field
    positive = {"signal": channels["signal"]}
    negative = {"signal": channels["signal"] * 0.3}
    analysis = Analysis(segmentation_channels=[], signal_channels="signal", radial=False)
    thresholds = analysis.calibrate_thresholds(
        positive, negative, positive_cell_masks=cells, negative_cell_masks=cells,
    )

    assert 30 <= thresholds["signal"] < 100
    for data, expected_area in [(positive, 32), (negative, 0)]:
        result = analysis.run(data, cell_masks=cells)
        assert result["thresholds"]["signals"] == thresholds
        assert sum(r["aggregate_area_pixels"] for r in result["cell_records"]) == expected_area


@pytest.mark.parametrize("save_images", [False, True])
def test_exports_contain_tables_and_optional_notebook_figures(field, tmp_path, save_images):
    cells, channels = field
    output = tmp_path / "result"
    result = analyze(
        channels, cell_masks=cells, segmentation_channels="cells", signal_channels="signal",
        nucleus_channel="nuclei", detection=MANUAL, nucleus_detection=MANUAL,
        radial_bins=3, output_dir=output, save_intermediates=save_images,
    )

    expected = {"final_data.csv", "radial_distribution.csv"}
    if save_images:
        expected.add("inspection_images")
    assert {p.name for p in output.iterdir()} == expected
    for name, key in [("final_data.csv", "cell_records"), ("radial_distribution.csv", "radial_records")]:
        with (output / name).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(result[key])
        assert [int(r["cell_label"]) for r in rows] == [r["cell_label"] for r in result[key]]
        assert sum(float(r["aggregate_area_pixels" if key == "cell_records" else "intensity_sum"])
                   for r in rows) == (32 if key == "cell_records" else 1600)
    if save_images:
        images = list((output / "inspection_images").iterdir())
        assert {p.name for p in images} == {
            "cell_segmentation.png", "nuclear_mask.png", "aggregate_mask.png",
            "all_cells_overlay.png", "transfected_cells.png", "radial_cell_3.png",
        }
        for path in images:
            with Image.open(path) as image:
                image.verify()


def test_empty_result_exports_headers_and_existing_output_is_protected(field, tmp_path):
    cells, channels = field
    channels["signal"][:] = 0
    analysis = Analysis(segmentation_channels=[], signal_channels="signal",
                        detection=MANUAL, require_aggregates=True, radial=False)
    output = tmp_path / "empty"
    result = analysis.run(channels, cell_masks=cells, output_dir=output)
    assert result["cell_records"] == []
    assert result["radial_records"] == []
    assert {p.name for p in output.iterdir()} == {"final_data.csv"}
    table = output / "final_data.csv"
    original = table.read_bytes()
    with table.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert "cell_label" in reader.fieldnames
        assert list(reader) == []
    with pytest.raises(FileExistsError):
        analysis.run(channels, cell_masks=cells, output_dir=output)
    assert table.read_bytes() == original


def test_invalid_input_is_rejected(field):
    cells, channels = field
    with pytest.raises(ValueError, match="shape"):
        analyze_field(cells, {"signal": channels["signal"][:-1]})
    nonfinite = channels["signal"].astype(float)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze_field(cells, {"signal": nonfinite})
    with pytest.raises(ValueError, match="nonnegative"):
        analyze_field(-cells, {"signal": channels["signal"]})
    with pytest.raises(ValueError, match="nucleus"):
        analyze_field(cells, {"signal": channels["signal"]}, require_nucleus=True)
