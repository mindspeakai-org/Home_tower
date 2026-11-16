# Generated dataset views

This directory is local-only apart from this README and its placeholder.

- `furniture_original_cache/`: one byte-preserving extraction of the source ZIP.
- `furniture_sanitized_shadow/`: symlinked images and new detection-format labels.
- `stage1_workstation_kitchen_bootstrap/`: unreviewed model proposals, not ground truth.
- `furniture_reviewed_evaluation*/`: immutable copies built only from accepted
  or corrected human review decisions.

Keep review decisions and source images private unless their publication has
been explicitly chosen and source licensing permits it. Do not commit caches,
symlinks into local data, or model predictions as approved training labels.
