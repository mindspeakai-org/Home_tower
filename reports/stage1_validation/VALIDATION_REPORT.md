# Stage 1 calibration and dataset validation

Date: 14 November 2026. CPU target: Intel Core i5-1335U, existing YOLO11n OpenVINO export at 480×480. No retraining or new dataset download was performed. The completed audit and its strict original-test metrics were reused rather than repeated.

Stage 1 now has an independent furniture proposal path, a detection-only shadow dataset that passes the native loader, a deterministic 125-image manual-review pack, guarded clean evaluation, and explicit calibration approval. The scheduler, stock detection/pose model classes, pose tracking, and Stage 2 alert modules are unchanged.

**This is not a completed accuracy acceptance. All 125 annotation decisions are pending and no real camera zones have been manually approved.** `clean_metrics.csv` is intentionally header-only, and no cleaned ground-truth dataset has been fabricated. Desk/stove remain unsupported model classes; operators can create those zone polygons manually. Default test geometry is explicitly marked as demonstration-only and is rejected for camera monitoring.

## Implemented and tested behavior

| Area | Implemented result | Verification and remaining limit |
|---|---|---|
| Runtime furniture filtering | `auto_detect_zones()` passes COCO `[56, 57, 59, 60, 62, 63, 69]`, `device=cpu`, `imgsz=480`, `rect=False`. It bypasses Stage 2's four-object filter. | Unit test asserts the exact filter; fresh CPU predictions produced valid proposals on 12 existing indoor dataset images. |
| Taxonomy | Bed, couch, chair and dining table remain distinct. TV/laptop detections are context; oven remains oven. | No context rule or mode silently maps a dining table to desk or an oven front to stove. This follows the latest task constraints, superseding the earlier requested proxy mapping. |
| Sanitization | Every original bbox or polygon row becomes a five-token detection row in a shadow directory, with exact original IDs and canonical aliases. | All three splits load with zero missing/corrupt files; source hashes and image symlinks checked. Formatting repair does not correct semantic annotation errors. |
| Human annotation review | Deterministic 125-image stratified pack, full overlays, editable per-image decisions, and a native OpenCV accept/reject/correct interface. | Creation and decision guards tested. All real decisions remain pending. Actual interactive GUI use has not been performed in this headless session. |
| Clean evaluation | Only accepted/corrected annotations with reviewer, timezone-aware review time, image hash and valid class/box data can enter a new immutable evaluation copy. | Synthetic tests prove review gating and TP/FP arithmetic. No numerical clean-set metrics are available without real manual review. |
| Calibration proposals | Boxes become editable polygons with source label, confidence, dimensions, frame hash, source path and unapproved status. | JSON round-trip and rendering tested on real proposals; geometry tests use synthetic fixtures. |
| Manual camera approval | Vertex dragging, polygon creation, zone type editing, per-zone confirmation and final save confirmation. Overlaps require acknowledgment. | Approved save/reload/preservation tested using explicitly synthetic temporary fixtures. No real proposal has been marked approved by the agent. |
| Startup integration | One representative camera frame uses the existing CPU detector before the strided scheduler starts. Existing approved boundaries are preserved. | Mocked two-frame camera test asserts exactly one calibration call. Actual camera hardware is unverified. |
| Safe configuration | Resolution mismatch, invalid geometry, missing approval or unresolved overlaps fail with configuration errors. | Numeric cameras, device paths and live stream URLs cannot opt into demo geometry. File E2E tests explicitly opt in. |

## Root cause and runtime separation

The earlier `zone_calibrator.py` only validated JSON; it had no detector-backed calibration function. The production secondary detector intentionally filters to phone, laptop, bowl and cup for Stage 2. Expanding that continuously running filter would have changed the Stage 2 workload and violated the requested separation.

The new Stage 1 function calls the existing COCO model directly and independently. At camera startup the main CLI reuses `CPUModels.objects` for this one call. Later Stage 2 object inference continues to pass its original four-class filter explicitly. `CPUModels` and `FrameScheduler` retain identical AST hashes; every file in `src/stage2_monitors/` retains its original content hash.

Model IDs 56/57/59 map to chair/couch/bed. ID 60 maps to `dining_table`, including when a monitor or laptop is nearby. ID 69 maps to `oven`, whose front-facing rectangle is not a validated cooktop polygon. IDs 62/63 are retained as context rather than approved zone proposals. Desk and stove are available manual zone types, not invented detection results.

Camera calibration is a single startup operation or an explicit `--calibrate` request. It does not run per frame. With existing approval, fresh proposals are exported for inspection beside the event log while approved geometry remains untouched. Monitoring checks source dimensions before processing frames. Dataset-image geometry must not be reused as a camera calibration.

## Non destructive dataset quality repair

Source archive: `data/furniture detection.v20i.yolov11.zip`.

```text
SHA-256 7236062f0af92ec59a9f4cdb62743ebdcf1814b3170f4de95465e51ac175a728
```

The supplied source is a ZIP, not a filesystem image tree. Symlinks cannot point directly into ZIP members, so one byte-preserving extraction exists at `datasets/furniture_original_cache/`. Its 16,113 source files are read-only and their hashes match the ZIP contents. The sanitized view at `datasets/furniture_sanitized_shadow/` symlinks every image to that cache and writes only new detection labels, manifests, YAML and native loader caches. It duplicates no image payloads; extraction, symlink metadata and normalized label text still consume filesystem space.

Each polygon row is converted separately using min/max vertices, then normalized center/width/height. Existing bbox rows retain their own semantics. This repairs the 558 mixed files without discarding their images. Original polygon geometry remains in the cache and archive; detection bounding boxes, particularly floor/surface boxes, are not automatically approved room polygons.

Class IDs are unchanged. `Sofa` becomes canonical `couch`; `Dining Table` is `dining_table` at ID 8, while `Table` remains `table` at ID 20. No Desk/Stove IDs were invented or merged into the 25-class annotation taxonomy. An alias collision would raise an error rather than silently merge IDs.

| Split | Native images loaded | Annotations | Empty labels | Missing | Corrupt |
|---|---:|---:|---:|---:|---:|
| train | 6424 | 8564 | 369 | 0 | 0 |
| valid | 891 | 1023 | 66 | 0 | 0 |
| test | 739 | 846 | 50 | 0 | 0 |

Evidence: native loader output (`../../datasets/furniture_sanitized_shadow/native_validation.json`, generated locally), sanitizer command log (`sanitizer_command.log`, generated locally), and [preservation checks](preservation_and_review_checks.json). These results demonstrate format compatibility, not that original annotations are complete or correct.

## Deterministic review subset

The review pack contains **125 original-test images**, seed `20261114`. Selection uses a stable SHA-256 ordering, class strata, all 36 mixed-format test files, a stratum of empty labels, and known annotation/model problems from the completed audit. It includes examples with likely wrong classes, omitted chairs, table ambiguity, model confusions, and workstation/kitchen scenes. No training or validation image is included. All 24 classes present in test are covered; Door has no original test examples and cannot be created by sampling.

Source-image hashes, original-label hashes, archive membership, exact numeric mappings, reasons for selection and original normalized boxes are recorded in review_subset/manifest.json (`review_subset/manifest.json`, generated locally). Each individual decision starts `pending`. The exported review sheets (`review_subset/review_sheet_1.jpg`, generated locally) and `overlays/` display source annotations, not verified clean truth and not model-derived replacement labels.

| Original ID | Canonical name | Original test objects | Review subset objects | Reviewed decisions |
|---:|---|---:|---:|---:|
| 0 | bed | 59 | 14 | 0 |
| 1 | cabinet | 28 | 8 | 0 |
| 2 | carpet | 14 | 8 | 0 |
| 3 | ceramic_floor | 38 | 18 | 0 |
| 4 | chair | 48 | 9 | 0 |
| 5 | closet | 19 | 4 | 0 |
| 6 | cupboard | 16 | 3 | 0 |
| 7 | curtains | 17 | 9 | 0 |
| 8 | dining_table | 20 | 6 | 0 |
| 9 | door | 0 | 0 | 0 |
| 10 | frame | 2 | 2 | 0 |
| 11 | futec_frame | 15 | 8 | 0 |
| 12 | futech_tiles | 2 | 2 | 0 |
| 13 | gypsum_board | 9 | 3 | 0 |
| 14 | lamp | 28 | 10 | 0 |
| 15 | nightstand | 13 | 6 | 0 |
| 16 | shelf | 24 | 6 | 0 |
| 17 | sideboard | 7 | 4 | 0 |
| 18 | couch | 176 | 9 | 0 |
| 19 | tv_stand | 18 | 5 | 0 |
| 20 | table | 200 | 34 | 0 |
| 21 | transparent_closet | 3 | 3 | 0 |
| 22 | wall_panel | 13 | 5 | 0 |
| 23 | window | 53 | 15 | 0 |
| 24 | wooden_floor | 24 | 18 | 0 |

Counts refer to annotated objects in selected images, not independent images per class. The test set's repeated images and incomplete labels remain review issues; acceptance should include deciding whether each duplicated source should be retained or rejected. Do not bulk-accept pending decisions.

### Operator annotation workflow

```bash
.venv/bin/python -m src.stage1_calibration.annotation_review --reviewer YOUR_NAME
```

Controls: drag adds a box; `[` and `]` select the original class ID; C clears; X removes the last box; A confirms unchanged or corrected annotations; R rejects; N/P skip/return; Q exits. Correct all visible target furniture, not only boxes the detector found. Original source classes remain available, and Dining Table must not be renamed Desk as a shortcut.

For headless review, inspect individual `overlays/*.jpg` alongside the original image path and edit the matching `decisions/*.json`. Accepted/corrected decisions need an operator name, ISO review timestamp with timezone, matching image hash, and valid normalized boxes. Changed labels require status `corrected`. Rejected/pending entries are excluded from evaluation.

After manual review, build and re-evaluate a new immutable copy in one command:

```bash
.venv/bin/python -m src.stage1_calibration.validate_stage1 --skip-rooms \
  --clean-dir datasets/furniture_reviewed_evaluation_v1
```

Use a new version path when review decisions change. The clean builder refuses to overwrite existing snapshots. The evaluator uses fresh CPU inference, the same strict semantic mapping and thresholds as the preserved baseline, and records per-class precision, recall, AP50, TP/FP/misses. It generates reviewed failure/correct-detection overlays only after reviewed ground truth exists. Synthetic decision fixtures in unit tests are isolated temporary data and do not enter this real review pack.

## Clean evaluation status and preserved baseline

[clean_metrics.csv](clean_metrics.csv) currently contains only its column header. [clean_evaluation_status.json](clean_evaluation_status.json) states `unavailable`, `reviewed_images: 0`, with the reason that manually reviewed annotations are absent. There is no `datasets/furniture_reviewed_evaluation/` dataset pretending to be cleaned. Creating zeros, evaluating predictions against themselves, or silently treating the unreviewed original subset as clean would produce misleading results and was not done.

The original strict baseline remains in `reports/furniture_audit/strict_original_metrics.csv`, with an identical separate [baseline copy](baseline_strict_original_metrics.csv). Its SHA-256 is:

```text
8565a2b45e080cfb6e37f77be3529b65aaee3ab087d06ec0bf35656a517ca0a1
```

Those are old, unreviewed original-test metrics, not the new clean-subset results. failure_cases/ (`failure_cases/`, generated locally) contains explicitly named `unreviewed_baseline_*` visualizations of a missed bed, apparent chair false positives, a table confused as bed, a spurious oven, an empty-labelled workstation, and a correct bed detection. Their status (`failure_cases/STATUS.json`, generated locally) identifies the annotation limitations. These are useful review material; they are not validated clean-set FP/miss rates.

## Bootstrap candidates without prediction ground truth

`datasets/stage1_workstation_kitchen_bootstrap/` contains **47 candidate records representing 46 distinct image hashes**, with symlinked source images and pending JSON proposals. Existing stock-model audit predictions were reused after checking checkpoint/archive identity. No audit was repeated, no data downloaded, and no pseudo-ground-truth `data.yaml` or accepted desk/stove label files were generated.

Workstation candidates use table plus chair/electronics proximity; monitor+keyboard context without a table is explicitly marked as needing manual drawing. Kitchen candidates retain source `oven` boxes and report adjacent sink/refrigerator/microwave context where present. Oven candidates without such context remain unconfirmed; the preview includes a known false-positive oven to make this limitation visible. Many table candidates are ordinary dining tables. **Neither candidate ranking nor a model prediction establishes a desk or stove label.**

The bootstrap preview (`../furniture_audit/bootstrapped_desk_stove_preview.png`, generated locally) is an operator verification sheet, not an accuracy result. Missing classes still require actual human annotation and, if detection of those classes is later needed, a separate reviewed Stage 1 taxonomy and training task. No models were retrained.

## Twelve indoor image calibration checks

Fresh real CPU inference ran on 12 existing dataset images spanning bedrooms, living rooms, a gaming workstation, and kitchen ranges. These are source images from the supplied dataset, including catalog imagery whose photographic/rendering provenance is not certified; they are not the drawn figures used by the synthetic E2E video. The sample is diagnostic and selected from existing examples, so it cannot estimate real-camera detection accuracy.

Every image produced a nonempty, valid polygon proposal dictionary. The generated proposal JSON reloaded without changing values, and reloads rendered successfully. Actual types returned were bed, couch, chair, dining_table and oven. Desk and stove were **not** fabricated to satisfy a five-zone assertion or written as approved production calibration.

| Image index | Supported proposal types | Proposal count | Status |
|---:|---|---:|---|
| 00 | bed, chair, couch | 3 | Unconfirmed |
| 01 | bed | 1 | Unconfirmed |
| 02 | couch | 1 | Unconfirmed |
| 03 | couch | 1 | Unconfirmed |
| 04 | chair | 1 | Unconfirmed |
| 05 | chair, dining_table | 7 | Unconfirmed |
| 06 | dining_table | 1 | Unconfirmed |
| 07 | dining_table, oven | 2 | Unconfirmed |
| 08 | chair, dining_table, oven | 3 | Unconfirmed |
| 09 | chair | 1 | Unconfirmed |
| 10 | bed | 1 | Unconfirmed |
| 11 | bed | 1 | Unconfirmed |

See the comparison grid (`../furniture_audit/fixed_calibration_verification.png`, generated locally), per-image proposals (`room_proposals/`, generated locally), and [execution measurements](real_image_execution.json). Orange means unconfirmed. Unsupported/unavailable desk/stove geometry is explicitly identified on the previews. The real workstation image demonstrates the limitation: the stock model finds a chair but no dedicated desk class.

### CPU latency and memory on this machine

- CPU: 13th Gen Intel Core i5-1335U; CPU OpenVINO path at 480×480.
- Model load and blank-frame warmup: **3.153 s**.
- Mean proposal latency on the 12 images: **52.61 ms**; p95 **77.90 ms**.
- Process RSS before loading: **266.07 MiB**; after warmup **499.46 MiB**.
- Sampled peak RSS during proposals: **527.76 MiB**; process high-water RSS **532.18 MiB**.

Latency includes image preprocessing, model inference/postprocessing, proposal mapping and frame hashing; it excludes model load/warmup, disk reads and drawing. Memory includes Python, PyTorch, OpenCV, OpenVINO and the model, not only the weights. This isolated process measurement is not a claim of the incremental memory cost of reusing the detector in the full pipeline. ONNX fallback is available through the unchanged existing loader but was not separately benchmarked in this task.

The unchanged whole-pipeline benchmark also passed in the full suite: 29.9 frame FPS, 9.8 pose FPS, 2.0 object FPS, 8.94 ms foreground processing and 40.60 ms conservative amortized CPU work per source frame. It is the existing throughput scenario, not furniture accuracy or a live-camera acceptance test.

## Manual camera calibration and safe startup

The default `configs/room_zones.json` retains its demonstration polygons and now explicitly records `status: demo`, source 640×480, and manual confirmation required. It is not an approved room calibration. Camera startup fails rather than silently using those invented room coordinates. The file/synthetic test-only `--allow-demo-zones` flag is forbidden for live camera indices, devices and stream URLs.

For a real representative camera frame, export proposals first:

```bash
.venv/bin/python -m src.stage1_calibration.zone_calibrator \
  --image camera_frame.png --output logs/camera_zone_proposals.json
```

Then either edit interactively or manually edit a separate copy of the JSON. The editor allows polygon creation, dragging vertices, changing the zone type, deleting proposals and individually confirming zones. Save requires explicit final confirmation and acknowledgment of reported overlaps. Desk/stove geometry must be created/confirmed by the operator; a stock COCO oven box is not a burner ROI.

```bash
.venv/bin/python -m src.main_pipeline --source 0 --calibrate \
  --calibration-reviewer YOUR_NAME --zones configs/room_zones.json
```

For a headless approval submission after human editing:

```bash
.venv/bin/python -m src.stage1_calibration.zone_calibrator \
  --approve logs/operator_reviewed_zones.json --output configs/room_zones.json \
  --reviewer YOUR_NAME --acknowledge-overlaps
```

Every saved zone must have explicit approval. The save records reviewer/time, label, polygon, original detector label where applicable, frame dimensions/hash/source, overlap findings and absent/uncalibrated types. Invalid or out-of-image polygons are rejected. Existing approved or unknown-provenance calibration files cannot be overwritten; choose a new path for a revision. Only an explicitly marked demo file can be replaced after genuine operator confirmation. Atomic writing with an exclusive lock protects concurrent approval writers.

Approved serialization/reload/display and preservation were tested using synthetic temporary fixtures. The 12 real proposal reloads remain unapproved. The native interactive windows, physical camera setup and actual human approval have **not** been exercised in this headless session. These remain necessary before live use.

## Commands and regression results

Commands executed from the project root:

```bash
YOLO_CONFIG_DIR=/tmp/tower-ultralytics YOLO_AUTOINSTALL=false \
  .venv/bin/python -m src.stage1_calibration.dataset_sanitizer
YOLO_CONFIG_DIR=/tmp/tower-ultralytics YOLO_AUTOINSTALL=false \
  .venv/bin/python -m src.stage1_calibration.annotation_review --create
YOLO_CONFIG_DIR=/tmp/tower-ultralytics YOLO_AUTOINSTALL=false \
  .venv/bin/python -m src.stage1_calibration.bootstrap_missing_classes
YOLO_CONFIG_DIR=/tmp/tower-ultralytics YOLO_AUTOINSTALL=false \
  .venv/bin/python -m src.stage1_calibration.validate_stage1
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

The first three commands completed: zero native dataset rejects, 125 pending review decisions, and pending bootstrap candidates. The validation command completed the 12-image CPU proposal checks and correctly reported unavailable clean metrics. Exact native/validation stdout/stderr is retained in sanitizer_command.log (`sanitizer_command.log`, generated locally), review_pack_command.log (`review_pack_command.log`, generated locally), bootstrap_command.log (`bootstrap_command.log`, generated locally), and real_image_validation_command.log (`real_image_validation_command.log`, generated locally). These outputs establish executed work, not manual approval.

The full regression command ran **40 tests in 23.850 s: 39 passed, one CUDA test skipped because CUDA is unavailable**. It includes all 15 tests in `tests/test_stage1_runtime_calibration.py`: class mappings/filtering, independent mixed-row parsing, review gates, synthetic metric accounting, geometry, overlap acknowledgment, protected serialization, dimensions, one mocked startup call, forbidden live demo use, and one real indoor CPU-model test. The existing 360-frame scripted E2E test passed with nine ordered transitions and its explicit demo flag. The complete result is saved in full_regression_command.log (`full_regression_command.log`, generated locally).

Synthetic/mocked tests prove control flow, coordinate validation and state transitions. The real-image tests prove model execution and proposal serialization on those images. Neither substitutes for reviewed accuracy metrics, manual camera-zone approval or recorded real indoor event validation. No `tools/jev.py` was called.

## Preservation and remaining blockers

[preservation_and_review_checks.json](preservation_and_review_checks.json) verifies the original ZIP and detection/pose PT hashes, all Stage 2 module hashes, unchanged `CPUModels`/`FrameScheduler` ASTs, all cached-source file hashes, and symlink-only shadow images. The original strict baseline is preserved separately. No pose/tracking math, scheduler behavior, Stage 2 alert algorithm, source dataset file or model weights were changed. CLI startup integration and explicit synthetic test opt-in are the intended integration changes.

Remaining blockers before Stage 2 perception acceptance:

1. A person must review the 125-image pack, correcting all target objects and rejecting unsuitable/duplicate cases. Until then, clean precision, recall, AP50, FP and misses are unavailable.
2. Desk/stove candidate locations and semantics need manual annotation. The COCO head still cannot detect dedicated desk/stove classes; only manually created zones can bridge that gap for current monitoring.
3. Every real camera needs approved, source-compatible polygons. The demonstration file must not be treated as calibrated.
4. Run the native review UI with a display, confirm actual camera startup/reload behavior, and evaluate recorded real indoor clips with independently reviewed object/person/event labels. Existing synthetic E2E success is not real-world alert accuracy.

The runnable code, pending review material and safe gates are implemented and tested. Human-reviewed clean accuracy, real camera-zone approval, interactive usability and live domain accuracy remain explicitly unverified rather than being inferred or manufactured.
