import hashlib
import subprocess
import sys

import numpy as np
import pytest
import tifffile
from skimage.measure import regionprops

from parsho.img_utils import extract_channels, load_image
from parsho.maskfilters import compute_cell_metrics, subtract_nuclear_from_aggregate
from parsho.provenance import runtime_provenance
from parsho.segmentation import re_threshold_masks
from parsho.single_image import DetectionSettings


def test_fully_covered_cells_average_all_objects():
    cells = np.ones((8, 8), dtype=np.int32)
    objects = np.ones_like(cells)
    objects[:, 4:] = 2
    metric = compute_cell_metrics(cells, np.zeros_like(cells), objects, [1])[0]
    expected = np.mean([round(p.axis_major_length / p.axis_minor_length, 3) for p in regionprops(objects)])
    assert metric.agg_aspect_ratio == pytest.approx(expected)
    assert metric.aggregate_coverage_fraction == metric.jaccard == 1


def test_nuclear_subtraction_relabels_disconnected_fragments():
    labels = np.ones((5, 5), dtype=np.int32)
    nucleus = np.zeros_like(labels, dtype=bool)
    nucleus[:, 2] = True
    binary, objects = subtract_nuclear_from_aggregate(labels > 0, labels, nucleus)
    assert objects.max() == 2
    assert not binary[:, 2].any()
    assert labels.min() == 1  # Inputs are not modified.


def test_object_size_filter_is_applied_after_splitting_at_cell_boundaries():
    cells = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32)
    binary, objects = re_threshold_masks(np.ones_like(cells), cells, min_size_px=3, thresh=0)
    assert binary.sum() == 4
    assert not objects[:, 2].any()


def test_legacy_channel_extraction_uses_explicit_time_and_depth(tmp_path):
    raw = np.arange(2 * 2 * 3 * 8 * 9, dtype=np.uint16).reshape(2, 2, 3, 8, 9)
    path = tmp_path / "image.ome.tif"
    tifffile.imwrite(path, raw, metadata={"axes": "TCZYX"}, photometric="minisblack")
    channels = extract_channels(path, normalize=False, time=1, z_mode="plane", z=2)
    np.testing.assert_array_equal(channels, raw[1, :, 2])
    series = tmp_path / "series.tif"
    with tifffile.TiffWriter(series) as writer:
        writer.write(np.zeros((10, 11), dtype=np.uint16))
        writer.write(np.ones((12, 13), dtype=np.uint16))
    assert load_image(series, scene=1).shape == (12, 13)
    with pytest.raises(ValueError, match="series"):
        load_image(series, scene=2)


def test_disabled_threshold_parameters_do_not_block_analysis():
    DetectionSettings(method="manual", manual_threshold=1, block_size=2, scale=-1).validate()
    with pytest.raises(ValueError):
        DetectionSettings(method="local", block_size=2).validate()


def test_import_does_not_load_torch_or_cellpose():
    subprocess.run([sys.executable, "-c",
                    "import sys, parsho; assert 'torch' not in sys.modules; assert 'cellpose' not in sys.modules"],
                   check=True)


def test_provenance_hashes_the_actual_model_file(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"test weights")
    model = type("Model", (), {"pretrained_model": str(path)})()
    captured = runtime_provenance(model)
    assert captured["model_sha256"] == hashlib.sha256(b"test weights").hexdigest()
    assert captured["versions"]["numpy"] == np.__version__
