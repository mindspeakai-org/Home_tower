# Development and validation

Run commands from the repository root with Python 3.10+. Native review/calibration
windows need a desktop display. The current measured environment was Linux/WSL2
on an Intel i5-1335U; the complete workflow has not been validated on every OS.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

CPU PyTorch goes first to avoid the default CUDA distribution. The optional
`requirements-legacy.txt` adds FiftyOne for the retained Open Images script;
the active workflow does not need it or any Open Images download.

## Rebuild local data and model artifacts

1. Put the separately provided Roboflow v20 ZIP in `data/`, with the filename
   and checksum documented in `data/README.md`.
2. Obtain the two existing checkpoints using the weights-only helper:

   ```bash
   python -c 'from src.stage1_calibration.download_stage1_data import download_weights; download_weights()'
   python -m src.utils.export_cpu_models
   ```

3. Create and verify the shadow dataset:

   ```bash
   python -m src.stage1_calibration.dataset_sanitizer
   ```

4. Cached audit JSON/predictions are versioned. Rebuild the local diagnostic
   pictures and comparison artifacts from the same dataset/checkpoint:

   ```bash
   python reports/furniture_audit/inspect_indoor.py
   ```

   This regenerates images and checks source integrity; it does not retrain or
   repeat the full test-split inference audit. The archived checkpoint/ZIP
   identities must match before cached predictions can be reused.

5. Create the manual review pack once:

   ```bash
   python -m src.stage1_calibration.annotation_review --create
   python -m src.stage1_calibration.bootstrap_missing_classes
   ```

   Review packs/candidate exports are immutable: the command refuses to replace
   existing decisions. Use a new version path if you intentionally need another
   pack. All new decisions are pending; none are automatically accepted.

See [the detailed workflow](WORKFLOWS.md) for review controls, approved calibration,
clean evaluation and model-export fallback behavior. Keep personal camera
calibration in a separate ignored local JSON file, for example
`configs/room_zones.local.json`; the tracked `room_zones.json` is a demo template.

## Tests without local weights or source data

The GitHub Actions workflow runs the following profile. It includes native metric
arithmetic and mocked camera startup, but no real inference or GUI windows:

```bash
python -m unittest \
  tests.test_cpu_scheduler \
  tests.test_stage2_alerts \
  tests.test_e2e_pipeline.ZoneValidationTests \
  tests.test_stage1_runtime_calibration.CalibrationUnitTests \
  tests.test_stage1_runtime_calibration.AnnotationReviewTests \
  tests.test_stage1_runtime_calibration.CPUCalibrationIntegrationTests.test_camera_startup_calibrates_once_outside_the_shared_scheduler \
  tests.test_stage1_runtime_calibration.CPUCalibrationIntegrationTests.test_demo_geometry_cannot_enable_live_camera_monitoring \
  -v
```

## Complete local regression suite

After preparing the source cache, weights and CPU exports:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The full suite includes real-model smoke/throughput checks and a real indoor
image. The E2E scene is generated locally, with injected observations to test
deterministic alert phases. CPU timing is hardware-dependent; CUDA is skipped
when unavailable. The E2E test appends synthetic transitions to the local event
log, not to a production camera acceptance record.

## Repository content

The Git repository contains source, tests, the demo configuration, documentation,
and selected audit metadata. It excludes local environments, datasets, weights,
runtime logs, generated imagery, camera calibration, and individual human review
decisions. Keep personal calibration in `configs/*.local.json`.

Archived reports retain measured claims and tables. References to intentionally
omitted local artifacts appear as file paths in the published copy. Recreate
those artifacts locally using the workflow above; do not treat absent generated
files as approved annotations or completed camera validation.
