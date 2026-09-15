# PARSHO

PARSHO is a Python package for quantitative fluorescence-puncta analysis in
segmented cells. It provides tools for loading microscopy data, thresholding
aggregate and nuclear channels, filtering transfected cells, calculating cell
and aggregate morphology, and measuring shape-adapted radial distributions.

Radial distributions can be centered on either the nucleus centroid or the
cell centroid. Unlike concentric circular rings, PARSHO combines distance from
the selected centroid with distance from the segmented cell boundary, allowing
the radial sections to adapt to elongated and irregular cell shapes.

PARSHO is available as a google colab at [ref]

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- scikit-image 0.26 or newer
- Matplotlib
- tifffile
- Pillow
- Cellpose

The complete tutorial additionally uses pandas, JupyterLab, and
IPython/Jupyter kernel support. All of these packages are declared in
`environment.yml`; the libraries imported by the installable package are also
declared in `pyproject.toml`.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/Fraternalilab/PARSHO.git
cd PARSHO
```

Create the complete Conda environment from the supplied file:

```bash
conda env create --file environment.yml
conda activate parsho
```

This installs PARSHO in editable mode, so changes made to files under
`parsho/` are immediately available in the environment. Confirm the
installation with:

```bash
python -c "import parsho; print(parsho.__version__)"
```

After `environment.yml` changes, update an existing environment with:

```bash
conda env update --name parsho --file environment.yml --prune
```

[Cellpose's official installation guidance](https://github.com/MouseLand/cellpose#installation)
recommends creating a Conda environment and installing Cellpose with pip
inside it. The supplied environment follows that approach. Cellpose is also a
required dependency in `pyproject.toml`, so installing PARSHO outside this
repository still installs it automatically.

To install a non-editable copy into an already active Conda environment:

```bash
python -m pip install .
```

To read Nikon ND2, Leica LIF/LOF, Zeiss CZI, and DICOM files, install the
optional microscopy readers:

```bash
python -m pip install -e '.[microscopy-io]'
```

`load_image(path, scene=0)` reads TIFF/OME-TIFF, JPEG, BMP, PNG, DICOM, ND2,
LIF, CZI, and LOF files as NumPy arrays. `extract_channels(path, scene=0)`
splits the result into channels and optionally applies display normalization.
LOF is read through the BioIO Bio-Formats plug-in, which installs its Java
runtime support automatically. Proprietary containers can contain several
images; use `scene` to choose one. OME-TIFF names ending in `.ome.tif` or
`.ome.tiff` are handled as TIFF automatically.

## Tutorial

Start JupyterLab from the repository root:

```bash
jupyter lab notebooks/parsho_aggregates_tutorial.ipynb
```

The tutorial demonstrates:

1. loading aligned nuclear, aggregate, and cell channels;
2. calibrating an aggregate threshold with positive and negative controls;
3. segmenting cells with Cellpose;
4. extracting aggregate, nuclear, and transfection masks;
5. retaining cells containing the required signals;
6. calculating per-cell morphology and aggregate measurements;
7. measuring nucleus-centered radial distributions;
8. saving inspection figures and CSV tables for a batch of images.

For a simpler, interactive single-image workflow in Google Colab, open
[the PARSHO Colab notebook](https://colab.research.google.com/github/Fraternalilab/PARSHO/blob/main/notebooks/PARSHO_Colab.ipynb).
Its form controls support either one
multichannel image or separately uploaded channels, optional positive and
negative controls, combined-channel Cellpose segmentation, nucleus/aggregate
detection, radial analysis, and downloading all masks, figures, settings, and
CSV results as one ZIP file.

The example workflow expects filenames of the form `C1-<name>` for nuclei,
`C2-<name>` for aggregates, and `C3-<name>` for cells. Adjust the notebook's
loading helper if your naming convention differs.


### Additional examples

The notebooks and scripts needed to reproduce the paper figures can be found in the notabooks and scripts section respectively. They can be used as examples for analysis of different experimental setups. The whole raw data needed can be dowloaded from [ref]


## Package layout

```text
parsho/
├── distribution.py   # Cell- and nucleus-centered radial measurements
├── img_utils.py      # Image loading, channel extraction, masks, and contrast
├── maskfilters.py    # Cell filtering, overlays, and morphology metrics
├── plotting.py       # Segmentation, mask, and radial plots
├── segmentation.py   # Thresholding and connected-component extraction
└── utils.py          # Radial records and CSV serialization
```

## License

PARSHO is distributed under the MIT License. See [LICENSE](LICENSE).
