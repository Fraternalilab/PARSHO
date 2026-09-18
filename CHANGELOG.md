# Changes and migration notes

## Unreleased — usability and workflow coherence

- Colab remains the no-code entry point, with a bundled-data demo button,
  preserved forms for unchanged loaded channels, and control previews before
  calibration. All settings remain editable.
- Local tutorials and all five analysis scripts retain their explicit,
  step-by-step implementations and original parameters. The proposed preset
  runner and CLI have been removed: loading, segmentation, detection, filtering,
  measurements and saving are visible in each example.
- Python metadata now requires 3.11+; the Conda environment uses 3.12.
  Cellpose is constrained to major version 4. Importing PARSHO no longer imports
  torch or changes global torch sparse-tensor validation.
- Raw channel loading selects one time point before Z projection.
  **Behaviour change:** `extract_channels` no longer flattens time/Z into
  channels; ambiguous stacks need explicit axes. Use `normalize=False`
  for measurements. `load_image(scene=...)` also selects TIFF series.
- **Measurement correction:** minimum object area is applied after splitting
  objects at cell boundaries. Nuclear subtraction relabels disconnected
  fragments. These changes can alter object counts relative to older exports.
  No second minimum-size filter is applied after nuclear subtraction.
- Fixed lower-level aggregate shape averaging when a cell is fully covered
  by positive aggregate labels (no background label).
- Colab/single-field tables use original cell labels, not renumbered display IDs.
  `aggregate_coverage_fraction` replaces ambiguous output naming; legacy
  `CellMetrics.jaccard` remains accessible. Generic plotting/overlay names
  have compatibility wrappers for historical names.
- Teaching scripts retain their original output schemas, filenames and display
  mappings. They reuse configured output paths; choose a new results directory
  to preserve earlier analyses. Full automatic exports and overwrite protection
  remain available in Colab/the single-field exporter, not imposed on examples.
- Truncations explicitly defaults to nucleus/aggregate/cell indices 0/1/2,
  preserving the historical implementation rather than its contradictory prose.
  Confirm the biological order from acquisition metadata.
- Colab/API settings exports include analysis flags, actual versions, source revision and
  model hash when available. Missing identities remain null.
- Removed committed notebook outputs and a historical run log; source notebooks
  and sample images remain. Added regression tests and a CPU CI workflow.

To compare old and new analyses, keep both environments, settings and masks.
Do not pool tables across versions without checking the schema, cell IDs,
threshold choices and object definitions.
