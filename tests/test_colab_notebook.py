"""Execute the actual Colab cells with real widgets and a deterministic model.

The browser upload/download bridge and expensive Cellpose inference are mocked;
loading, role selection, analysis, export and stale-state guards run unchanged.
"""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest
import tifffile

pytest.importorskip("ipywidgets")
nbformat = pytest.importorskip("nbformat")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "PARSHO_Colab.ipynb"


@pytest.fixture
def notebook(monkeypatch, tmp_path):
    import cellpose.core
    import cellpose.models

    downloads = []
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(download=lambda path: downloads.append(path))
    colab.output = SimpleNamespace(enable_custom_widget_manager=lambda: None)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cellpose.core, "use_gpu", lambda: False)

    class Model:
        pretrained_model = "test_model"

        def __init__(self, **_kwargs):
            pass

        def eval(self, image, **_kwargs):
            cells = np.zeros(image.shape[:2], dtype=np.int32)
            cells[2:15, 2:15] = 3
            cells[17:30, 17:30] = 8
            return cells, [], None

    monkeypatch.setattr(cellpose.models, "CellposeModel", Model)
    book = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(book)
    sources = {cell.id: cell.source for cell in book.cells if cell.cell_type == "code"}
    context = {}

    def run(name):
        exec(compile(sources[name], f"PARSHO_Colab:{name}", "exec"), context)

    run("initialise")
    context["display"] = lambda *_args, **_kwargs: None
    return run, context, downloads


def make_images(tmp_path):
    cells = np.zeros((32, 32), dtype=np.uint16)
    cells[2:15, 2:15] = 100
    cells[17:30, 17:30] = 120
    signal = np.zeros_like(cells)
    signal[5:9, 5:9] = 1000
    signal[21:25, 21:25] = 1200
    nuclei = np.zeros_like(cells)
    nuclei[5:7, 5:7] = 400
    separate = []
    for index, image in enumerate([cells, signal, nuclei]):
        path = tmp_path / f"channel_{index + 1}.tif"
        tifffile.imwrite(path, image)
        separate.append(str(path))
    combined = tmp_path / "multichannel.ome.tif"
    tifffile.imwrite(combined, np.stack([cells, signal, nuclei]), metadata={"axes": "CYX"}, photometric="minisblack")
    return separate, str(combined)


@pytest.mark.parametrize("separate,has_nucleus", [(False, False), (True, True)])
def test_complete_notebook_through_download(notebook, tmp_path, separate, has_nucleus):
    run, context, downloads = notebook
    paths, combined = make_images(tmp_path)
    run("sample-upload")
    context["sample_picker"].set_paths(paths if separate else [combined])
    context["sample_picker"].load()
    run("settings")
    form = context["form"]
    form.rows[0]["segmentation"].value = True
    form.rows[1]["signal"].value = True
    form.rows[1]["name"].value = "Puncta"
    form.save_all_radial.value = False
    if has_nucleus:
        form.nucleus.value = 2
        form.nucleus_threshold.fields["min_size"].value = 1
        form.options["remove_nuclear"].value = True
    run("controls-upload")
    run("controls-roles")
    run("segment")
    run("analyse")
    result = context["result"]
    assert len(result["cell_records"]) == 2
    assert result["center"] == ("nucleus" if has_nucleus else "cell")
    run("explore")
    run("download")
    assert len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "cell_measurements.csv" in archive.namelist()
        assert "radial_distribution.csv" in archive.namelist()
        assert "READ_ME.txt" in archive.namelist()
        saved = json.loads(archive.read("analysis_settings.json"))
        assert saved["actual_radial_center"] == result["center"]
        assert len(saved["sources"]) == (3 if separate else 1)
    form.options["radial_bins"].value = 6
    with pytest.raises(ValueError, match="Settings have changed"):
        run("download")
    run("analyse")
    assert len(context["result"]["radial_records"]) == (6 if has_nucleus else 12)
    form.segmentation["flow_threshold"].value = 0.5
    with pytest.raises(ValueError, match="Rerun step 5"):
        run("analyse")
    assert context["result_state"] is None


def test_controls_separate_and_multichannel_with_nuclear_exclusion(notebook, tmp_path):
    run, context, _ = notebook
    paths, combined = make_images(tmp_path)
    run("sample-upload")
    context["sample_picker"].set_paths([combined])
    context["sample_picker"].load()
    run("settings")
    form = context["form"]
    form.rows[0]["segmentation"].value = True
    form.rows[1]["signal"].value = True
    form.nucleus.value = 2
    form.nucleus_threshold.fields["min_size"].value = 1
    form.options["remove_nuclear"].value = True
    form.thresholds[1].fields["method"].value = "controls"
    form.save_all_radial.value = False
    run("controls-upload")
    context["positive_picker"].set_paths(paths)
    context["positive_picker"].load()
    negative = tmp_path / "negative.tif"
    image = tifffile.imread(combined)
    image[1] = 0
    tifffile.imwrite(negative, image, metadata={"axes": "CYX"}, photometric="minisblack")
    context["negative_picker"].set_paths([str(negative)])
    context["negative_picker"].load()
    run("controls-roles")
    for name in ["positive_assignment", "negative_assignment"]:
        assignment = context[name]
        assignment.segmentation[0].value = True
        assignment.signals["Signal 2"].value = 1
        assignment.nucleus.value = 2
    run("segment")
    calibrate = context["find_optimal_threshold"]
    def fail_calibration(*args, **kwargs):
        raise ValueError("fixture control mismatch")
    context["find_optimal_threshold"] = fail_calibration
    with pytest.raises(ValueError, match="fixture control mismatch"):
        run("analyse")
    assert context["result_state"] is None
    assert (context["WORK_DIR"] / "control_positive.png").is_file()
    assert (context["WORK_DIR"] / "control_negative.png").is_file()
    context["find_optimal_threshold"] = calibrate
    run("analyse")
    assert context["calibrated"]["Signal 2"] >= 0
    assert (context["WORK_DIR"] / "control_positive.png").is_file()
    assert (context["WORK_DIR"] / "control_negative.png").is_file()
    assert len(context["result"]["cell_records"]) == 2
    run("download")
    context["negative_assignment"].signals["Signal 2"].value = 0
    with pytest.raises(ValueError, match="Control settings changed"):
        run("download")


def test_widget_disables_nucleus_only_options_and_requires_reload(notebook, tmp_path):
    run, context, _ = notebook
    paths, _ = make_images(tmp_path)
    run("sample-upload")
    picker = context["sample_picker"]
    picker.set_paths(paths)
    picker.load()
    run("settings")
    form = context["form"]
    assert form.options["remove_nuclear"].disabled
    form.nucleus.value = 2
    form.options["remove_nuclear"].value = True
    form.nucleus.value = None
    assert not form.options["remove_nuclear"].value
    picker.rows[0][1]["time"].value = 2
    with pytest.raises(ValueError, match="Load and preview"):
        form.snapshot()


def test_demo_prefills_roles_and_preserves_edits(notebook, tmp_path, monkeypatch):
    import parsho.examples
    run, context, _ = notebook
    paths, _ = make_images(tmp_path)
    monkeypatch.setattr(parsho.examples, "demo_files", lambda: [Path(path) for path in paths])
    run("sample-upload")
    context["sample_picker"]._demo(None)
    run("settings")
    form = context["form"]
    assert form.selected() == ([0], {"Aggregates": 1})
    assert form.nucleus.value == 2
    form.options["radial_bins"].value = 6
    form.rows[1]["name"].value = "My puncta"
    run("settings")
    assert context["form"] is form
    assert form.options["radial_bins"].value == 6
    assert form.selected()[1] == {"My puncta": 1}
    saved = form.snapshot()
    replacement = context["AnalysisForm"](context["sample_picker"])
    replacement.restore(saved)
    assert replacement.snapshot() == saved
    context["sample_picker"].set_paths(paths)
    context["sample_picker"].load()
    run("settings")
    assert context["form"] is not form
    assert not context["sample_picker"].is_demo
