import csv
import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from parsho.colab_results import export_field_result
from parsho.single_image import (
    DetectionSettings, analyze_field, combine_segmentation_channels,
    detect_signal, load_field_channels,
)


def test_tczyx_selects_time_before_depth_projection(tmp_path):
    image = np.arange(2 * 3 * 4 * 8 * 9, dtype=np.uint16).reshape(2, 3, 4, 8, 9)
    path = tmp_path / "sample.ome.tif"
    tifffile.imwrite(path, image, metadata={"axes": "TCZYX"}, photometric="minisblack")
    channels, metadata = load_field_channels(path, time=1)
    np.testing.assert_array_equal(channels, image[1].max(axis=1))
    assert metadata["axes"] == "TCZYX"
    channels, _ = load_field_channels(path, time=0, z_mode="plane", z=2)
    np.testing.assert_array_equal(channels, image[0, :, 2])


def test_tiff_series_and_ambiguous_axes(tmp_path):
    path = tmp_path / "series.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((10, 11), dtype=np.uint16))
        writer.write(np.ones((12, 13), dtype=np.uint16))
    channels, _ = load_field_channels(path, scene=1)
    assert channels[0].shape == (12, 13)
    assert channels[0].min() == 1
    path = tmp_path / "unknown.tif"
    image = np.arange(5 * 12 * 13, dtype=np.uint16).reshape(5, 12, 13)
    tifffile.imwrite(path, image, metadata={"axes": "QYX"}, photometric="minisblack")
    with pytest.raises(ValueError, match="ambiguous"):
        load_field_channels(path)
    channels, _ = load_field_channels(path, axes="CYX")
    np.testing.assert_array_equal(channels, image)


@pytest.mark.parametrize("extension", [".png", ".bmp", ".jpg"])
def test_rgb_and_separated_channels_keep_intensities(tmp_path, extension):
    rgb = np.full((12, 13, 3), 75, dtype=np.uint8)
    path = tmp_path / f"image{extension}"
    Image.fromarray(rgb).save(path)
    channels, _ = load_field_channels(path)
    assert len(channels) == 3
    np.testing.assert_array_equal(channels, np.moveaxis(rgb, -1, 0))
    for i, image in enumerate(channels):
        separate = tmp_path / f"ch{i}.tif"
        tifffile.imwrite(separate, image)
        np.testing.assert_array_equal(load_field_channels(separate)[0][0], image)


@pytest.mark.parametrize("extension", [".nd2", ".lif", ".lof", ".czi"])
def test_vendor_files_use_package_reader_and_canonical_axes(monkeypatch, tmp_path, extension):
    import parsho.single_image as workflow

    image = np.arange(2 * 2 * 3 * 8 * 9, dtype=np.uint16).reshape(2, 2, 3, 8, 9)
    calls = []
    def load(path, scene=0):
        calls.append((path.suffix, scene))
        return image
    monkeypatch.setattr(workflow, "load_image", load)
    channels, _ = load_field_channels(tmp_path / f"image{extension}", scene=2, time=1)
    assert calls == [(extension, 2)]
    np.testing.assert_array_equal(channels, image[1].max(axis=1))


def test_dicom_frames_need_explicit_time_or_depth(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    import parsho.single_image as workflow

    image = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    monkeypatch.setattr(workflow, "load_image", lambda *_args, **_kwargs: image)
    monkeypatch.setitem(sys.modules, "pydicom", SimpleNamespace(dcmread=lambda *_args, **_kwargs: SimpleNamespace(SamplesPerPixel=1)))
    path = tmp_path / "extensionless_dicom"
    with pytest.raises(ValueError, match="ambiguous"):
        load_field_channels(path)
    channels, _ = load_field_channels(path, axes="TYX", time=2)
    np.testing.assert_array_equal(channels[0], image[2])


def test_combining_channels_does_not_modify_raw_values():
    channels = [np.arange(100).reshape(10, 10) * factor for factor in (1, 2, 3, 4)]
    originals = [image.copy() for image in channels]
    assert combine_segmentation_channels(channels, "mean").shape == (10, 10)
    assert combine_segmentation_channels(channels[:3], "stack").shape == (10, 10, 3)
    with pytest.raises(ValueError, match="at most 3"):
        combine_segmentation_channels(channels, "stack")
    for image, original in zip(channels, originals):
        np.testing.assert_array_equal(image, original)


@pytest.fixture
def field():
    cells = np.zeros((20, 20), dtype=np.int32)
    cells[2:18, 2:10] = 3
    cells[2:18, 10:18] = 8
    signal = np.zeros(cells.shape, dtype=np.uint16)
    signal[4:8, 4:14] = 100
    nucleus = np.zeros(cells.shape, dtype=np.uint16)
    nucleus[4:6, 4:6] = 80
    settings = DetectionSettings(method="manual", manual_threshold=10, min_size=1)
    return cells, signal, nucleus, settings


def test_without_nucleus_multiple_signals_and_cross_cell_objects(field):
    cells, signal, _, settings = field
    original = signal.copy()
    result = analyze_field(cells, {"RNA A": signal, "RNA B": signal * 2},
                           detection={"RNA A": settings, "RNA B": settings})
    assert result["center"] == "cell"
    assert len(result["cell_records"]) == 4
    assert len(result["object_records"]) == 4
    assert {row["cell_label"] for row in result["cell_records"]} == {3, 8}
    assert all(row["nucleus_area_pixels"] is None for row in result["cell_records"])
    assert all(row["radial_included"] for row in result["cell_records"])
    assert len(result["radial_records"]) == 40
    assert sum(row["aggregate_intensity_sum"] for row in result["cell_records"]) == 12000
    assert sum(row["intensity_sum"] for row in result["radial_records"] if row["signal"] == "RNA A") == 4000
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("remove", [False, True])
def test_nucleus_exclusion_and_radial_missing_nuclei(field, remove):
    cells, signal, nucleus, settings = field
    result = analyze_field(cells, {"signal": signal}, detection={"signal": settings},
                           nucleus=nucleus, nucleus_detection=settings, remove_nuclear=remove,
                           pixel_size_um=0.5)
    assert result["center"] == "nucleus"
    by_label = {row["cell_label"]: row for row in result["cell_records"]}
    assert by_label[3]["radial_included"]
    assert not by_label[8]["radial_included"]
    assert by_label[8]["nucleus_area_pixels"] == 0
    assert by_label[3]["cell_intensity_sum_raw"] == 2400
    assert by_label[3]["aggregate_intensity_sum"] == (2000 if remove else 2400)
    assert by_label[3]["cell_intensity_sum_analyzed"] == (2000 if remove else 2400)
    assert by_label[3]["nucleus_area_um2"] == 1
    assert set(result["distributions"]["signal"]) == {3}
    assert signal[4, 4] == 100


def test_filter_audit_and_empty_exports(field, tmp_path):
    cells, signal, nucleus, settings = field
    signal = np.zeros_like(signal)
    result = analyze_field(cells, {"signal": signal}, detection={"signal": settings},
                           nucleus=nucleus, nucleus_detection=settings, require_aggregates=True)
    assert result["cell_records"] == result["object_records"] == result["radial_records"] == []
    assert all("no aggregates" in row["reason"] for row in result["filter_records"])
    output = export_field_result(tmp_path / "results", result, {"signal": signal}, {}, nucleus=nucleus)
    for name in ["cell_measurements.csv", "aggregate_measurements.csv", "radial_distribution.csv"]:
        with (output / name).open() as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames
            assert list(reader) == []
    assert json.loads((output / "analysis_settings.json").read_text())["actual_radial_center"] == "nucleus"


def test_transfection_filter_radial_off_and_no_nucleus_export(field, tmp_path):
    cells, signal, nucleus, settings = field
    result = analyze_field(cells, {"signal": signal}, detection={"signal": settings},
                           transfection=nucleus, transfection_detection=settings,
                           require_transfection=True, radial=False)
    assert result["retained_labels"] == [3]
    assert not result["radial_records"]
    assert "no transfection" in result["filter_records"][1]["reason"]
    output = export_field_result(tmp_path / "results", result, {"signal": signal}, {}, transfection=nucleus)
    assert not (output / "nucleus_labels.tif").exists()
    assert (output / "transfection_mask.tif").exists()
    assert (output / "READ_ME.txt").is_file()


@pytest.mark.parametrize("method", ["otsu", "percentile", "local", "manual"])
def test_all_detection_modes(field, method):
    cells, signal, _, _ = field
    settings = DetectionSettings(method=method, min_size=1, block_size=5, percentile=50, manual_threshold=10)
    binary, labels, threshold = detect_signal(signal, cells, settings)
    assert binary.any()
    assert not binary[cells == 0].any()
    assert (labels > 0).sum() == binary.sum()
    assert np.ndim(threshold) == (2 if method == "local" else 0)


def test_nucleus_options_and_empty_segmentation_raise_clear_errors(field):
    cells, signal, _, _ = field
    with pytest.raises(ValueError, match="No cells"):
        analyze_field(np.zeros_like(cells), {"signal": signal})
    with pytest.raises(ValueError, match="nucleus channel"):
        analyze_field(cells, {"signal": signal}, remove_nuclear=True)


def test_control_calibration_honours_minimum_area():
    from parsho.segmentation import find_optimal_threshold

    cells = np.ones((8, 8), dtype=np.int32)
    pos, neg = np.zeros((8, 8)), np.zeros((8, 8))
    pos[2:5, 2:5] = 100
    neg[1, 1:3] = 60
    nuclear = np.zeros_like(pos, dtype=bool)
    threshold, pixels = find_optimal_threshold(pos, cells, neg, cells, nuclear, nuclear,
                                              t_min=1, t_max=50, min_size_px=3)
    assert threshold == 1
    assert pixels == 9
