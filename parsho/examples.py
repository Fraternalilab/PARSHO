"""Locate bundled example data or download the small Colab demo image set."""

from pathlib import Path
import shutil
import tempfile
from urllib.request import urlretrieve


def example_data_dir():
    """Find the bundled aggregate TIFFs from common repository locations."""
    candidates = [
        Path.cwd() / "notebooks/data/aggregates",
        Path.cwd() / "data/aggregates",
        Path(__file__).resolve().parents[1] / "notebooks/data/aggregates",
    ]
    for path in candidates:
        if (path / "C3-E3_115plate4.tif").is_file():
            return path
    raise FileNotFoundError(
        "Bundled examples not found. Clone the repository or use the Colab demo button."
    )


def demo_files(destination=None):
    """Return the cell, aggregate and nucleus demo files, downloading if needed."""
    filenames = [f"C{index}-E3_115plate4.tif" for index in (3, 2, 1)]
    try:
        folder = example_data_dir()
    except FileNotFoundError:
        folder = Path(destination) if destination else Path(tempfile.mkdtemp(prefix="parsho_demo_"))
        folder.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            path = folder / name
            if path.exists():
                continue
            # Validate a temporary download before exposing it as a demo TIFF.
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".download") as stream:
                urlretrieve(
                    "https://raw.githubusercontent.com/Fraternalilab/PARSHO/"
                    f"main/notebooks/data/aggregates/{name}",
                    stream.name,
                )
                import tifffile

                with tifffile.TiffFile(stream.name) as image:
                    _ = image.series[0].shape
                shutil.copyfile(stream.name, path)
    return [str(folder / name) for name in filenames]
