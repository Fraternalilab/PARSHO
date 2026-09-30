"""Example: configure Analysis and run it on titin images.

Edit the parameters below, then run from the repository root:
    python scripts/titin_aggregates_simple.py

The defaults use the bundled E3_115plate4 sample and positive/negative controls.
Add sample filenames to SAMPLE_NAMES to reuse the same analysis for more images.
"""

from pathlib import Path

from parsho import Analysis


# User parameters: change these paths, sample names and analysis choices.
DATA_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "data" / "aggregates" # Replace with your path.
RESULTS_DIR = Path("results/titin_simple")  # Choose a new folder for each run.
SAMPLE_NAMES = ["E3_115plate4.tif"]  # Filenames without the C1-/C2-/C3- prefix.
USE_CONTROLS = True                # False uses Otsu separately for each sample.
POSITIVE_CONTROL_NAME = "positive_control.tif"
NEGATIVE_CONTROL_NAME = "negative_control.tif"
SAVE_INTERMEDIATES = True          # False saves only the measurement CSVs.
RADIAL_ANALYSIS = True
RADIAL_BINS = 10
GPU = None                        # None selects automatically; False uses CPU.


def channel_files(data_dir, name):
    """Assign the three channel files to their roles in this experiment."""
    return {
        "cells": data_dir / f"C3-{name}",
        "titin": data_dir / f"C2-{name}",
        "nuclei": data_dir / f"C1-{name}",
    }


def main():

    required_names = list(SAMPLE_NAMES)
    if USE_CONTROLS:
        required_names += [POSITIVE_CONTROL_NAME, NEGATIVE_CONTROL_NAME]

    # Configure once. Edit these parameters to match your experiment.
    # A channel may serve multiple roles: titin also marks transfected cells.
    analysis = Analysis(
        segmentation_channels="cells",  # Or ["cells", "nuclei"] to combine both.
        signal_channels="titin",
        nucleus_channel="nuclei",
        transfection_channel="titin",
        detection="otsu",               # Also "percentile", "local", or DetectionSettings(...).
        require_nucleus=True,
        require_transfection=True,
        remove_nuclear=True,
        radial=RADIAL_ANALYSIS,
        radial_center="nucleus",
        radial_bins=RADIAL_BINS,
        gpu=GPU,
        # Titin experiment settings; general Analysis defaults are more conservative.
        cellpose_parameters={"flow_threshold": 0.0, "cellprob_threshold": -2.0},
    )

    # Optional: calibrate once, then apply the fixed threshold to every sample.
    if USE_CONTROLS:
        thresholds = analysis.calibrate_thresholds(
            positive=channel_files(DATA_DIR, POSITIVE_CONTROL_NAME),
            negative=channel_files(DATA_DIR, NEGATIVE_CONTROL_NAME),
        )
        print(f"Control-calibrated thresholds: {thresholds}")

    # Run the same analysis on each sample, keeping one Cellpose model loaded.
    for name in SAMPLE_NAMES:
        print(f"Analysing {name}...")
        result = analysis.run(
            channel_files(DATA_DIR, name),
            image_id=name,
            output_dir=RESULTS_DIR / name,
            save_intermediates=SAVE_INTERMEDIATES,
        )
        print(f"  Retained {len(result['retained_labels'])} cells; saved to {result['output_dir']}")


if __name__ == "__main__":
    main()
