"""Check teaching examples remain explicit and the external-mask tutorial runs."""

import ast
import json
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np
import pytest
import tifffile

nbformat = pytest.importorskip("nbformat")
ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_clean_and_compile():
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            assert not re.search(r"/(?:media|home)/[A-Za-z]", cell.source)
            assert "parsho.batch" not in cell.source
            if cell.cell_type != "code":
                continue
            assert cell.execution_count is None and not cell.outputs
            source = "\n".join(line for line in cell.source.splitlines() if not line.startswith("%"))
            compile(source, f"{path.name}:{cell.id}", "exec")


@pytest.mark.parametrize("path", sorted((ROOT / "scripts").glob("process_*.py")), ids=lambda p: p.name)
def test_scripts_show_the_workflow_locally(path):
    source = path.read_text()
    tree = ast.parse(source)
    assert "parsho.batch" not in source
    assert "preset_settings" not in source
    assert 'DATA_DIR = Path("path/to/your/data")' in source
    assert "CELLPOSE_PARAMETERS" in source
    assert "model.eval(" in source
    assert "extract_masks(" in source
    assert "compute_cell_metrics(" in source
    assert ".to_csv(" in source
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert len(main.body) > 5


def test_external_mask_tutorial_execution(tmp_path, monkeypatch):
    import parsho.examples
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    data = tmp_path / "data"
    data.mkdir()
    cells = np.zeros((40, 40), dtype=np.uint16)
    cells[2:38, 2:38] = 100
    signal = np.zeros_like(cells)
    signal[5:10, 5:10] = 500
    nucleus = np.zeros_like(cells)
    nucleus[18:22, 18:22] = 250
    for prefix, image in [("C1", nucleus), ("C2", signal), ("C3", cells)]:
        tifffile.imwrite(data / f"{prefix}-E3_115plate4.tif", image)
    monkeypatch.setattr(parsho.examples, "example_data_dir", lambda: data)
    # The external-mask walkthrough must neither import nor instantiate Cellpose.
    monkeypatch.setitem(sys.modules, "cellpose", ModuleType("cellpose"))
    book = nbformat.read(ROOT / "notebooks/parsho_external_cell_masks_tutorial.ipynb", as_version=4)
    context = {"display": lambda *args, **kwargs: None}
    for cell in book.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"external_masks:{cell.id}", "exec"), context)
    output = context["RESULTS_DIR"]
    saved = json.loads((output / "analysis_settings.json").read_text())
    assert saved["analysis_options"]["nucleus_supplied"]
    assert (output / "aggregate_measurements.csv").exists()
    assert (output / "radial_distribution.csv").exists()
    assert context["result"]["cell_records"]
    plt.close("all")
