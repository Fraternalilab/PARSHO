import sys
from types import ModuleType

import numpy as np
import pytest
import tifffile
from PIL import Image

from parsho.img_utils import extract_channels, load_image


@pytest.mark.parametrize("extension", [".tif", ".ome.tif", ".ome.tiff"])
def test_load_tiff_variants(tmp_path, extension):
    expected = np.arange(20, dtype=np.uint16).reshape(4, 5)
    path = tmp_path / f"image{extension}"
    tifffile.imwrite(path, expected)
    np.testing.assert_array_equal(load_image(path), expected)


@pytest.mark.parametrize(
    "extension,format_name",
    [(".png", "PNG"), (".jpg", "JPEG"), (".bmp", "BMP")],
)
def test_load_common_raster_formats(tmp_path, extension, format_name):
    expected = np.zeros((4, 5, 3), dtype=np.uint8)
    expected[..., 0] = 100
    path = tmp_path / f"image{extension}"
    Image.fromarray(expected).save(path, format=format_name)
    loaded = load_image(path)
    assert loaded.shape == expected.shape
    assert loaded.dtype == expected.dtype


def test_extract_rgb_channels(tmp_path):
    expected = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    path = tmp_path / "image.png"
    Image.fromarray(expected).save(path)
    channels = extract_channels(path, normalize=False)
    assert len(channels) == 3
    np.testing.assert_array_equal(channels[1], expected[..., 1])


def test_microscopy_reader_uses_tczyx_and_scene(monkeypatch, tmp_path):
    expected = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)

    class FakeBioImage:
        scenes = ("one", "two")

        def __init__(self, path, **kwargs):
            self.scene = None

        def set_scene(self, scene):
            self.scene = scene

        def get_image_data(self, order):
            assert order == "TCZYX"
            assert self.scene == 1
            return expected

    module = ModuleType("bioio")
    module.BioImage = FakeBioImage
    monkeypatch.setitem(sys.modules, "bioio", module)
    path = tmp_path / "image.nd2"
    path.touch()
    np.testing.assert_array_equal(load_image(path, scene=1), expected)


def test_dicom_dispatch(monkeypatch, tmp_path):
    expected = np.arange(12).reshape(3, 4)
    pixels = ModuleType("pydicom.pixels")
    pixels.pixel_array = lambda path: expected
    monkeypatch.setitem(sys.modules, "pydicom", ModuleType("pydicom"))
    monkeypatch.setitem(sys.modules, "pydicom.pixels", pixels)
    path = tmp_path / "image.dcm"
    path.touch()
    np.testing.assert_array_equal(load_image(path), expected)


def test_rejects_unknown_extension(tmp_path):
    path = tmp_path / "image.xyz"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(path)
