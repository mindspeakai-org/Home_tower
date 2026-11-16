# Project status

Recorded implementation/validation: 14 November 2026. Repository preparation:
16 November 2026. This is a CPU monitoring prototype with mandatory calibration
approval, not a validated safety device.

## Implemented

- Modular Stage 1 calibration/data preparation and Stage 2 monitoring packages.
- Stock YOLO11n detection and YOLO11n-pose tracking, with 480×480 OpenVINO/ONNX
  CPU inference and separate, bounded pose/object inference slots.
- Shared person boxes/IDs across fall/inactivity, study focus, and stove monitors.
- Timestamp-based alert state machines, transition-only JSONL logging, HUD,
  source-rate video loop, headless MP4 export, and synthetic E2E verification.
- Non-destructive mixed-box/polygon sanitizer and native loader validation.
- Independent furniture proposals, manual polygon editing, per-zone approval,
  overlap checks, source-size checks, and protection of existing approved zones.
- Deterministic 125-image annotation review pack and clean-evaluation guards.

## Recorded results

| Check | Recorded outcome |
|---|---|
| Full local regression suite | 40 tests: 39 passed, one CUDA test skipped |
| Native sanitized dataset parsing | 6424 train, 891 validation, 739 test; zero corrupt/missing files |
| Stage 1 real model checks | Valid, unapproved proposals on 12 existing indoor dataset images |
| Stage 1 CPU timing | Mean 52.61 ms; p95 77.90 ms; isolated process high-water RSS about 532 MiB |
| Full pipeline CPU benchmark | 29.9 frame FPS, 9.8 pose FPS, 2.0 object FPS; 40.60 ms amortized work/frame |
| Synthetic E2E | 360 rendered frames, nine ordered transitions; scripted track loss/reacquisition |

These are the historical recorded runs linked below. Fresh CI runs a data-free
unit profile; it does not reproduce the dataset or hardware benchmarks.
On 16 November 2026, the publishable checkout passed all 35 tests in that unit
profile with no local dataset cache or model checkpoints present. GitHub-hosted
execution is a separate run after pushing.

## Pending human work

- All 125 real annotation-review decisions remain pending. `clean_metrics.csv`
  is header-only; reviewed precision, recall and AP are not available.
- Desk/stove bootstrap proposals are unreviewed. The stock COCO model has no
  dedicated desk or stove class. Dining tables remain dining tables and ovens
  remain ovens until an operator defines the intended zone geometry.
- Default polygons are demo-only. Real camera zones, interactive editor use,
  camera hardware and indoor event accuracy still need operator validation.
- Models have not been retrained on the Roboflow dataset.

Passing synthetic tests or CPU throughput does not establish real-world alert
accuracy. No extra Open Images dataset was downloaded.

## Evidence

- [Furniture audit](../reports/furniture_audit/AUDIT_REPORT.md)
- [Original strict metrics](../reports/furniture_audit/strict_original_metrics.csv)
- [Stage 1 validation](../reports/stage1_validation/VALIDATION_REPORT.md)
- [Machine-readable validation status](../reports/stage1_validation/VALIDATION_STATUS.json)
- [Clean evaluation status](../reports/stage1_validation/clean_evaluation_status.json)

Generated pictures, videos and human review decisions are local-only. The source
archive, model weights and Stage 2 algorithms were preserved during Stage 1 work.
