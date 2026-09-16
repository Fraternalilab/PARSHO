# PARSHO

PARSHO measures fluorescent aggregates and puncta within segmented cells,
including autophagy markers and RNA signals. It reports cell and punctum
morphology, signal intensity, and where puncta lie between the cell or nucleus
centre and the cell boundary.

## Choose your starting point

| What you want to do | Start here | What you need |
| --- | --- | --- |
| Analyse your images without writing code | [Open PARSHO in Google Colab](https://colab.research.google.com/github/Fraternalilab/PARSHO/blob/main/notebooks/PARSHO_Colab.ipynb) | A browser and your image files; no local Python installation |
| Learn the workflow on supplied data | [Local notebook tutorials](#local-notebook-tutorials) | A local environment and basic notebook familiarity |
| Use existing cell segmentations or integrate PARSHO into Python | [Python API and external masks](#python-api-and-external-masks) | Aligned image arrays and integer cell labels |
| Process many images with an experiment-specific workflow | [Batch scripts](#batch-scripts) | Python familiarity and the expected dataset layout |

New to image analysis? Start with Colab. A *channel* is an image of one stain
or signal. A *cell mask* identifies the pixels belonging to each cell.
*Segmentation* finds those cell boundaries; *thresholding* selects bright
pixels within them as candidate puncta. Inspect both before interpreting results.

## No-code analysis in Google Colab

**[Open the guided Colab notebook](https://colab.research.google.com/github/Fraternalilab/PARSHO/blob/main/notebooks/PARSHO_Colab.ipynb)**

Use the play button beside each numbered step, in order. Complete the forms
and uploads before continuing; **do not use “Run all.”** All analysis settings
are labelled checkboxes, dropdowns or numeric fields.

1. **Install and initialise.** Run both setup cells. For faster segmentation,
   select **Runtime → Change runtime type → T4 GPU** if available. CPU also
   works. Installation and the first Cellpose model download can take time.
2. **Upload and preview.** Select one multichannel image or several aligned
   images containing separate channels from the same field. Filenames are
   unrestricted. All [supported formats](#supported-image-formats) are accepted.
   Large files can also be selected from a mounted Google Drive.
3. **Assign channels and settings.** Name each signal, tick the channels to
   combine for cell segmentation, and tick each channel to measure as puncta.
   Select a nucleus or transfection-marker channel only if you have one.
   Several puncta channels can be analysed independently.
4. **Choose optional controls.** Start with automatic Otsu thresholding, or
   choose percentile, local/adaptive or manual thresholds per signal.
   Positive/negative control calibration is optional; skip step 4 unless
   selected. Control images must have comparable acquisition settings and
   intensity units.
5. **Segment and analyse.** Check the numbered cell outlines, then the nuclear
   and puncta overlays. Adjust settings and rerun the indicated steps as needed.
   Inspect cell/punctum tables and individual radial plots.
6. **Download before closing Colab.** Run the final download cell to create a
   ZIP containing all results. If the browser download does not start, click
   **Download results ZIP** again, or use Colab's **Files** panel to locate the
   printed ZIP path, right-click it and choose **Download**. Session files
   are temporary.

The four settings tabs cover **channel roles**, **cell segmentation**,
**signal detection**, and **filtering, radial analysis and export**. Changes
to segmentation require rerunning segmentation and analysis; changes to
thresholds or filters require rerunning analysis and downloading a new ZIP.
Changing uploaded files requires loading and assigning the channels again.

### Nuclei, subtraction and radial analysis

A nucleus channel is optional in Colab and the single-field Python workflow.
Without it, leave the selection at **None**: nuclear measurements are blank,
nucleus-dependent options are disabled, and automatic radial analysis uses
the cell centre. With it, you can choose the nucleus or cell centre.

Nuclear exclusion is off by default in Colab. When enabled, it removes nuclear
pixels from both the puncta mask and analysed signal intensity. Original raw
cell-intensity totals remain available. The nucleus filter is a separate
choice: it controls which cells are retained, not which pixels are measured.

Radial sections adapt to the cell shape rather than forming circular rings.
Bin 0 is closest to the selected centre; later bins approach the boundary.
Radial intensity uses detected puncta pixels. With nucleus centring, a cell
without detected nuclear pixels has no radial rows and
`radial_included=False`; it remains in the cell table unless filtered out.

This workflow analyses **one 2-D field or Z projection at one time point**.
It does not perform 3-D measurements, time-series analysis, batch processing
or external-mask loading. Those last two workflows have separate entry points
below. The older experiment-specific notebooks/scripts have their own nuclear
requirements and defaults.

### What is in the download?

| Output | Contents |
| --- | --- |
| `summary.csv` | Cell/punctum counts, punctum area and intensity totals per signal |
| `cell_measurements.csv` | One row per retained cell per signal: morphology, areas, counts, intensities and radial inclusion |
| `aggregate_measurements.csv` | One row per punctum: parent cell, size, shape, position and intensity |
| `cell_filtering.csv` | Every segmented cell, retention status and exclusion reasons |
| `radial_distribution.csv` | One row per cell, signal and radial bin |
| `cell_masks.tif`, `cell_masks.npy` | Integer cell labels matching the Colab tables and figures |
| `signal_XX/` | Selected/projected signal, puncta masks, analysed intensity, overlays and optional per-cell radial figures |
| `analysis_settings.json` | Channel/file assignments, settings, effective thresholds and software versions |
| `READ_ME.txt` | Measurement definitions, units and missing-data conventions |

Nucleus and transfection images/masks are included when supplied. Areas are
pixel counts; entering a square-pixel width adds square-micrometre columns.
Intensity remains in the input image's units. Blank nuclear measurements mean
“no nucleus channel supplied”; zero means “supplied but none detected.”

`aggregate_coverage_fraction` is punctum area divided by cell area. Older
outputs call this `jaccard`, but it is **not** a conventional Jaccard overlap
index. Colab preserves original cell labels; older examples may also use
renumbered display indices. Check each workflow's mapping before joining tables.
Full masks include excluded cells, while measurement tables contain retained
cells. A header-only table can therefore be a valid result.

## Local installation

Skip this section if you are using Colab.

Use **Python 3.12** for the local setup below. Core dependencies include NumPy,
SciPy, scikit-image, Matplotlib, tifffile, Pillow and Cellpose. Notebooks and
batch scripts additionally use packages such as pandas; these are included
by the installation below. See [pyproject.toml](pyproject.toml) for dependency
declarations and optional extras.

From a terminal, clone the repository and enter it:

```bash
git clone https://github.com/Fraternalilab/PARSHO.git
cd PARSHO
```

### Complete environment for notebooks and scripts

With Conda installed:

```bash
conda create --name parsho python=3.12 pip
conda activate parsho
python -m pip install -e '.[tutorial,microscopy-io]'
python -c "import parsho; print(parsho.__version__)"
```

This installs PARSHO in editable mode, all optional image readers, JupyterLab,
pandas and widgets. Select a notebook kernel using this same environment.

The current [environment.yml](environment.yml) pins Python 3.10, while the
required scikit-image 0.26 needs Python 3.11 or newer. The package's declared
Python minimum is also still 3.10. Until those declarations are aligned, use
the Python 3.12 commands above rather than creating/updating an environment
from that YAML file.

### Existing Python environment

In an environment using Python 3.11 or newer, from the repository root,
choose the installation matching your use:

```bash
# Core package, including TIFF and common raster readers:
python -m pip install -e .

# Notebooks/scripts, widgets and all optional image readers:
python -m pip install -e '.[tutorial,microscopy-io]'
```

Use `python -m pip install .` for a non-editable core installation.
Cellpose is currently a required package dependency even when you use existing
masks and do not run segmentation. A GPU is useful for Cellpose but is not
required for downstream PARSHO measurements.

## Local notebook tutorials

For the supplied aggregate example, launch JupyterLab from the **notebooks**
directory so its `Path.cwd() / "data"` configuration resolves correctly:

```bash
cd notebooks
jupyter lab parsho_aggregates_tutorial.ipynb
```

Run cells from top to bottom in a fresh kernel. The bundled
[data/aggregates](notebooks/data/aggregates) directory contains C1/C2/C3 TIFF
sets and positive/negative controls. Here C1 = nuclei, C2 = aggregates and
C3 = cell channel. These naming rules belong to these examples, not to the
package or Colab. The full tutorial also runs a batch section; stop before
that section if you only want its single-image walkthrough.

| Notebook | Purpose | Data/setup |
| --- | --- | --- |
| [Aggregate tutorial](notebooks/parsho_aggregates_tutorial.ipynb) | Controls, segmentation, filtering, cell measurements, radial analysis and a batch example | Bundled C1/C2/C3 TIFFs; starts with control calibration |
| [External cell masks](notebooks/parsho_external_cell_masks_tutorial.ipynb) | Save/reload labelled masks and skip Cellpose inference | Bundled TIFFs; creates illustrative masks, not validated biological segmentations |
| [Autophagy example](notebooks/parsho_autophagy_analysis_example.ipynb) | Combined-channel segmentation and autophagy-puncta radial analysis | Supply matching C1/C2/C3 images and set `DATA_DIR` |
| [RNA-scope example](notebooks/parsho_rna_scope_example.ipynb) | Separate TRPV1 and TRPA1 measurements and radial profiles | Supply C1 nuclei, C2 TRPV1, C3 TRPA1, C4 brightfield; set `DATA_DIR` |
| [IDR0072_A example](notebooks/parsho_idr0072_a_single_image_analysis.ipynb) | One image folder, merged-image segmentation, raw-signal intensity and morphology | Supply the matching dataset; set `DATA_DIR` and `IMAGE_FOLDER`; no radial section |
| [Colab notebook](notebooks/PARSHO_Colab.ipynb) | Guided form-based analysis | Run in Google Colab; upload your files |

The experiment-specific notebooks require their corresponding datasets;
complete download instructions for those datasets are not currently supplied.
Some older notebooks still contain local paths, fixed example cell numbers and
saved outputs. Replace their configuration values, select an available cell
label, and rerun before interpreting those outputs. Their defaults illustrate
particular experiments rather than universal analysis settings.

## Supported image formats

| Image type | Accepted extensions | Reader installation |
| --- | --- | --- |
| TIFF / OME-TIFF | `.tif`, `.tiff`, `.ome.tif`, `.ome.tiff` | Included: tifffile |
| Common raster images | `.png`, `.jpg`, `.jpeg`, `.bmp` | Included: Pillow |
| DICOM | `.dcm`, `.dicom`, `.dic`, `.ima`, or no extension | `microscopy-io` extra: pydicom |
| Nikon | `.nd2` | `microscopy-io` extra: BioIO ND2 |
| Leica | `.lif`, `.lof` | `microscopy-io` extra: BioIO LIF / Bio-Formats |
| Zeiss | `.czi` | `microscopy-io` extra: BioIO CZI |

Colab's installation step and the complete local setup above include the
optional readers. In an existing local installation, add them with
`python -m pip install -e '.[microscopy-io]'` from the repository root.
LOF reading uses the BioIO Bio-Formats integration. NumPy `.npy` is a mask
format handled separately by `load_mask_npy()`, not an image-reader format.
Use original microscopy images for quantitative intensity work; JPEG is lossy.

### Shapes, channel order and intensity

All channels and masks passed to the 2-D analysis must have the same height,
width and spatial alignment. Matching dimensions alone do not prove alignment.

- `load_image(path, scene=0)` returns raw pixel values. TIFF/raster layouts
  follow their readers; vendor containers return `TCZYX` (time, channel,
  depth, height, width). Its `scene` selection currently applies to vendor
  containers; use the field loader below for TIFF-series selection.
- `load_field_channels()` in [single_image.py](parsho/single_image.py) reads
  dimension metadata, selects a time point, then selects/projects Z and returns
  raw 2-D channels plus metadata. This is the loader used by Colab. It supports
  TIFF series and vendor scenes and asks for explicit axes when a stack is
  ambiguous. Use it for multidimensional single-field analysis.
- `extract_channels(path, normalize=False)` is a simpler splitter used in
  older examples. Its default `normalize=True` contrast-stretches channels
  to `uint8` for display; use `False` for quantitative measurements. It
  does not provide explicit time/Z selection and flattens time and depth for
  5-D input, so it is not interchangeable with the field loader.

Python channel, scene, time and Z indices start at **0**. Colab's corresponding
form selections start at **1**. A Z maximum projection is a 2-D representation
of the stack, not a 3-D measurement.

## Python API and external masks

Cellpose supplies cell labels, but downstream PARSHO analysis also accepts
labels from another segmentation tool. A label mask is a 2-D nonnegative
integer array: `0` is background and each cell has a distinct positive ID.
Save it with `np.save("cell_masks.npy", labelled_mask)`, then load it with
`load_mask_npy(..., kind="labels", expected_shape=image.shape)`.
`kind="binary"` discards individual IDs and is unsuitable for a multi-cell
label mask.

The example below assumes an existing labelled mask and one aligned 2-D
aggregate image. It runs the reusable single-field analysis without a nucleus
and exports the same types of tables/masks as Colab:

```python
from pathlib import Path

from parsho import load_image, load_mask_npy
from parsho.single_image import DetectionSettings, analyze_field
from parsho.colab_results import export_field_result

data_dir = Path("path/to/your/data")
signal = load_image(data_dir / "aggregate.tif")
cells = load_mask_npy(
    data_dir / "cell_masks.npy", kind="labels", expected_shape=signal.shape
)
signals = {"puncta": signal}
options = dict(radial=True, radial_center="cell", radial_bins=10)
result = analyze_field(
    cells,
    signals,
    detection={"puncta": DetectionSettings(method="otsu", min_size=2)},
    **options,
)
print(f"Measured {len(result['cell_records'])} cell rows")
export_field_result(
    data_dir / "parsho_results",
    result,
    signals,
    settings={"input_image": "aggregate.tif", "cell_mask": "cell_masks.npy", **options},
)
```

Choose a **new** output directory for each export: `export_field_result()`
rejects an existing directory. It writes tables, arrays, figures and an output
guide; ZIP creation and browser downloads are handled by the Colab notebook.

To include nuclei, pass an aligned `nucleus` image to `analyze_field()`.
Set `remove_nuclear=True` to exclude nuclear pixels, and choose
`radial_center="nucleus"`, `"cell"` or `"auto"`. Additional named arrays
in `signals` support multiple measured channels. Other options include
`nucleus_detection`, `transfection_detection`, signal-presence filters,
border exclusion and `pixel_size_um`; see the function definition for defaults.

For a multichannel file or stack, first use:

```python
from parsho.single_image import load_field_channels

channels, input_metadata = load_field_channels(
    "path/to/image.ome.tif", scene=0, time=0, z_mode="maximum"
)
# Inspect the channels, then choose their biological roles explicitly.
```

For custom pipelines, the lower-level modules expose individual steps:

| Task | Functions / module |
| --- | --- |
| Detect nuclear, puncta or transfection pixels | `extract_masks()` in [segmentation.py](parsho/segmentation.py): `otsu`, `percentile` or `local` |
| Apply a fixed threshold or calibrate controls | `re_threshold_masks()`, `find_optimal_threshold()` in [segmentation.py](parsho/segmentation.py) |
| Filter cells, combine masks, measure morphology | `filter_cells_by_overlap()`, `build_overlay()`, `compute_cell_metrics()` in [maskfilters.py](parsho/maskfilters.py) |
| Compute radial profiles | `compute_cell_centered_distribution()`, `nucleus_centroids_by_cell()`, `compute_nucleus_centered_distribution()` in [distribution.py](parsho/distribution.py) |
| Export radial rows | `radial_distributions_to_records()`, `save_radial_distributions_csv()` in [utils.py](parsho/utils.py) |
| Plot masks and radial profiles | [plotting.py](parsho/plotting.py) |

The wrapper `DetectionSettings` also accepts `method="manual"`.
The lower-level `extract_masks()` does **not** accept `"manual"` or
`"adaptive"`: use `re_threshold_masks()` or `method="local"`, respectively.
When composing lower-level steps, nuclear-mask subtraction and zeroing nuclear
intensity are separate operations. Pass `aggregate_mask=None` to a radial
function only when you intend to measure all cell pixels rather than detected
puncta. The historical plotting function `plot_nucleus_centered_distribution()`
can also plot cell-centred results with `center_label="Cell"`.

Known lower-level limitation: `compute_cell_metrics()` can miscalculate mean
aggregate shape when multiple aggregate labels completely cover a cell, because
its averaging denominator assumes a background label is present. The
single-field `analyze_field()` workflow computes its aggregate-shape averages
from individual objects separately. Use that workflow for these measurements
until the lower-level averaging is corrected.

## Batch scripts

The scripts are experiment-specific Python workflows, not a general command-line
interface. From the repository root, edit the chosen script's top configuration:
set `DATA_DIR = Path("path/to/your/data")`, check its channel mapping, thresholds,
Cellpose parameters and nuclear-exclusion settings, then run, for example:

```bash
python scripts/process_autophagy.py
```

Use the complete local environment or install the `tutorial` extra for pandas.
Test a representative image interactively before running a dataset. These
scripts currently expect nuclear data; Colab's optional-nucleus behaviour is
not automatically applied to them.

| Script | Expected inputs | Analysis |
| --- | --- | --- |
| [process_autophagy.py](scripts/process_autophagy.py) | C1 nuclei, C2 puncta, C3 cell-channel TIFFs sharing a filename suffix | Cell measurements and nucleus-centred radial profiles |
| [process_titin_aggregates.py](scripts/process_titin_aggregates.py) | C1/C2/C3 TIFFs plus `positive_control.tif` and `negative_control.tif` sets | Control-calibrated aggregates and radial profiles |
| [process_rna-scope.py](scripts/process_rna-scope.py) | C1 nuclei, C2 TRPV1, C3 TRPA1, C4 brightfield TIFFs | Separate measurements and radial tables for both RNA signals |
| [process_truncations.py](scripts/process_truncations.py) | Condition subdirectories containing multichannel TIFFs | Cell/punctum morphology; no radial analysis |
| [process_idr0072_a.py](scripts/process_idr0072_a.py) | `image-*` folders with matching merged JPEG, raw ch01 and raw ch02 TIFFs | Morphology and raw aggregate intensity; no radial analysis |

Check the truncation script's channel mapping especially carefully: its current
unpacking uses **nucleus, aggregate, cell** order, while its opening docstring
describes a different order. Confirm the actual data order before running it.
For IDR0072_A, `SEGMENTATION_SOURCE` selects merged RGB or raw ch01; raw ch01
defines nuclei and raw ch02 supplies aggregate measurements.

The first four scripts discover only `.tif`/`.tiff` files, including OME-TIFF
suffixes. IDR0072_A uses its explicit dataset filenames. Broader package reader
support does not change these discovery rules.

Outputs are written beneath `DATA_DIR`: normally `results/`, or
`tutorial_results/` for RNA-scope. Scripts write `final_data.csv`,
inspection figures and, where implemented, radial CSVs. They do not produce
the same complete export bundle as Colab. Repeated runs reuse output paths;
set a fresh `RESULTS_DIR` when preserving previous analyses.

## Troubleshooting and reproducibility

| Symptom | What to check |
| --- | --- |
| Colab forms or imports are missing | Run both setup cells in the current session; if prompted, restart the runtime and rerun imports |
| Notebook cannot find example data | Launch the aggregate tutorial from `notebooks/`; inspect its configured paths |
| “Ambiguous axes” or wrong channel count | Check acquisition/export dimension order; use the field loader's axes, series, time and Z controls |
| Channels/masks have different shapes | Select aligned files from the same field and matching planes; inspect registration |
| No cells, no retained cells or empty radial tables | Inspect outlines and signal masks, then filtering decisions; nucleus-centred profiles require detected nuclear pixels |
| Radial intensity differs from total raw cell intensity | Radial analysis uses detected puncta, with any requested nuclear exclusion |
| Output directory already exists | Choose a new directory for the single-field exporter |
| Download does not start | Retry the download button or use the Files panel; download before the Colab session ends |

For reproducibility, keep the raw inputs, channel assignments, masks,
thresholds, filters, software/model versions and any scene/time/Z selections
with your results. Colab exports its settings; older notebooks/scripts need
these recorded separately. The hosted notebook installs PARSHO from GitHub,
so notebook and helper-module changes must be published together. For a fixed
analysis, retain the repository revision and environment versions you used.

For development checks, install the test dependencies from the repository root:

```bash
python -m pip install -e '.[dev,tutorial]'
python -m pytest -q
```

Colab widget tests require the notebook dependencies; their upload/download
bridge and model inference are simulated. They do not replace a browser check
of the hosted notebook. Report reproducible problems through
[GitHub Issues](https://github.com/Fraternalilab/PARSHO/issues), including the
workflow, input shape/format, settings and error message.

## Package layout

```text
parsho/
├── single_image.py   # Field loading and analysis with optional nuclei
├── colab_ui.py       # Optional upload, channel-role and settings widgets
├── colab_results.py  # Single-field tables, arrays, figures and output guide
├── img_utils.py      # General image readers, channel extraction and masks
├── segmentation.py   # Thresholding and connected-component extraction
├── maskfilters.py    # Filtering, overlays and morphology
├── distribution.py   # Cell- and nucleus-centred radial measurements
├── plotting.py       # Inspection and radial figures
└── utils.py          # Radial records and CSV serialization
```

## License

PARSHO is distributed under the MIT License. See [LICENSE](LICENSE).
