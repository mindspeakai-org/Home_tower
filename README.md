# Home Tower

CPU-first smart-home computer vision with furniture-zone calibration, shared
person tracking, and temporal monitoring for falls/inactivity, study focus, and
unattended cooking.

This repository contains the implementation and recorded development evidence.
It is a prototype: human annotation review and real-camera calibration are still
pending. The models remain pretrained COCO models, not furniture-fine-tuned
checkpoints. Synthetic test success is not real-world alert accuracy.

## What is implemented

- **Stage 1:** non-destructive dataset sanitization, a deterministic manual-review
  workflow, CPU furniture proposals, editable polygons, and explicit zone approval.
- **Stage 2:** shared pose boxes/track IDs for all monitors, with pose inference
  every three frames and gated object inference every fifteen frames.
- **CPU inference:** YOLO11n/YOLO11n-pose at 480×480, OpenVINO on x86_64 with ONNX
  fallback, bounded asynchronous inference slots, and no CUDA requirement.
- **Monitoring:** timestamp-based fall/inactivity, phone distraction, ergonomic
  posture, and unattended-stove state machines; transition-only JSONL events.
- **Video:** zone/skeleton overlays, alert banners, timing counters, headless MP4
  export, and deterministic integration fixtures.

Dining-table detections are not automatically called desks. Oven detections are
not automatically called stoves. Every operational zone needs human confirmation.
The tracked configuration is demo-only and cannot silently enable a camera.

## Quick setup

Use Python 3.10+ and run commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Get only the two model checkpoints, then prepare CPU exports:

```bash
python -c 'from src.stage1_calibration.download_stage1_data import download_weights; download_weights()'
python -m src.utils.export_cpu_models
```

Weights, datasets and generated media are not included in Git. Place the
separately provided Roboflow Furniture Detection v20 ZIP in `data/`, then run:

```bash
python -m src.stage1_calibration.dataset_sanitizer
```

See [development setup](docs/DEVELOPMENT.md) for the full reproducible workflow,
source checksum, artifact reconstruction and prerequisites. The optional
`requirements-legacy.txt` is only for the retained FiftyOne preparation script;
no Open Images download is required for the current workflow.

## Camera calibration

A desktop display is needed for the interactive editor. Use a separate local
configuration so personal camera geometry stays out of Git:

```bash
python -m src.main_pipeline --source 0 --calibrate \
  --calibration-reviewer YOUR_NAME --zones configs/room_zones.local.json
```

Review each proposal, draw unsupported desk/stove zones manually, and confirm
all polygons and overlaps before saving. Existing approved calibration files are
preserved; use a new version path for revisions. On later starts:

```bash
python -m src.main_pipeline --source 0 --zones configs/room_zones.local.json
```

For headless approval from a representative frame, review controls, annotation
review and clean evaluation, use the [detailed workflow](docs/WORKFLOWS.md).

## Tests and recorded status

GitHub Actions runs a unit profile without datasets, checkpoints or GUI windows.
After preparing local models and the source cache, the complete suite is:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The recorded full run had **39 passed and one CUDA test skipped**. Real Stage 1
CPU proposal checks covered 12 existing indoor dataset images. All 125 real
annotation review decisions remain pending, so `clean_metrics.csv` deliberately
contains no numerical clean-set accuracy results.

The E2E test generates its 360-frame video locally and uses scripted observations
to verify the scheduler, alert state machines, logging, and HUD. See
[project status](docs/STATUS.md) for measured timings and explicit limitations.
Hardware-dependent throughput and real inference are tested locally, not claimed
by the data-free CI profile.

## Project layout

```text
src/
  stage1_calibration/     # sanitization, review, proposals and approval
  stage2_monitors/        # temporal alert algorithms
  utils/                 # CPU exports and event logger
  main_pipeline.py       # startup integration and video loop
configs/                 # tracked demo; *.local.json stays local
tests/                   # unit, CPU integration and generated E2E checks
docs/                    # setup, workflows and current status
reports/                 # selected audit/validation records
```

Local `data/`, `datasets/`, `weights/`, `logs/`, review decisions and generated
images/videos are excluded from Git. Cached audit prediction/annotation metadata
is retained so review packs can be rebuilt without repeating the full audit.

## Evidence and attribution

- [Furniture audit](reports/furniture_audit/AUDIT_REPORT.md)
- [Original strict test metrics](reports/furniture_audit/strict_original_metrics.csv)
- [Stage 1 validation](reports/stage1_validation/VALIDATION_REPORT.md)
- [Current implementation and remaining blockers](docs/STATUS.md)
- [Model and dataset attribution](THIRD_PARTY_NOTICES.md)

Some archived reports reference generated local artifacts that are intentionally
not redistributed. Source licensing has not been assigned automatically; model
and dataset terms are documented separately in the third-party notices.
