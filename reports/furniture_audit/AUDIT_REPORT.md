# Smart Home CV furniture dataset and model audit

Audit date: 14 November 2026. Deployment target: CPU, fixed 480×480 inference.

The supplied Roboflow v20 archive is present locally and readable, but the project had not integrated it into a training or furniture-evaluation workflow. The deployed detector remains an 80-class COCO model. This audit evaluated its existing OpenVINO export on all 739 nominal held-out test images without training, downloading data, or modifying the original archive, checkpoints, scheduler, pose tracking, alert algorithms, configurations, or tests.

The COCO model provides a useful bed/chair/couch baseline but cannot express most of the dataset's 25 classes, including a distinct generic Table class. Neither the dataset nor the model provides a dedicated desk or stove class. Annotation errors, incomplete room annotations, and mixed box/segment label files must be addressed in a derived dataset before using these results to accept furniture perception. Do not download Open Images V7 at this stage: existing images already contain useful desks and kitchen ranges that can first be reviewed and annotated.

## Dataset provenance and actual project use

Source: [Roboflow furniture detection v20](https://universe.roboflow.com/mokhamed-nagy-u69zl/furniture-detection-qiufc/dataset/20), workspace `mokhamed-nagy-u69zl`, project `furniture-detection-qiufc`, version 20. The archive metadata states CC BY 4.0. Its README records an export on 18 October 2024.

| Check | Finding and evidence |
|---|---|
| Downloaded or supplied locally | Yes. `data/furniture detection.v20i.yolov11.zip`, approximately 172 MiB, contains `data.yaml`, images and YOLO annotations. Its embedded project URL/version matches the supplied link. |
| Integrated before this audit | No operational integration found. The original `datasets/` contained only `.gitkeep`; source code did not reference this archive or a derived Roboflow training configuration. The README explicitly described the ZIP as a separate, unextracted dataset. |
| Used for local training | No evidence. No `best.pt`, `last.pt`, training `args.yaml`, or `results.csv` existed. Original `src/` and `tests/` contained no `.train()` calls. The detector's own metadata identifies COCO training. Files and a dataset link are not evidence of training. |
| Used for evaluation before this audit | No Roboflow evaluation artifact or `.val()` workflow was found. Prior tests covered model loading, CPU throughput, synthetic alert sequences, and a scripted MP4 scenario. Those do not measure Roboflow furniture accuracy. |
| Used for evaluation now | Yes, by this audit only: 739/739 test images, 846 GT objects, with saved predictions, per-class metrics, and diagnostic overlays. The ZIP is read directly; labels are converted only in memory. |
| Open Images V7 downloaded | No. The downloader script exists, but its expected `datasets/stage1_zones_yolo` output does not. This audit made no dataset or checkpoint download. |

Archive SHA-256:

```text
7236062f0af92ec59a9f4cdb62743ebdcf1814b3170f4de95465e51ac175a728
```

See [inventory.json](inventory.json), the unmodified [data.yaml copy](original_metadata/data.yaml), and both [Roboflow README](original_metadata/README.roboflow.txt) and [dataset README](original_metadata/README.dataset.txt).

## Dataset splits and class IDs

The original YAML contains `nc: 25` and an ordered `names` list; list positions define numeric class IDs. Its paths are `train: ../train/images`, `val: ../valid/images`, and `test: ../test/images`. The actual ZIP directories are `train/`, `valid/`, `test/`, and a separate `export/` sample. Future training should use a derived, explicit-path YAML rather than assuming the archive path is itself a usable dataset root.

| Split | Images | Label files | Empty label files | Annotated objects |
|---|---:|---:|---:|---:|
| train | 6424 | 6424 | 369 | 8564 |
| valid | 891 | 891 | 66 | 1023 |
| test | 739 | 739 | 50 | 846 |
| export | 1 | 1 | 0 | 1 |

All 8055 images decode successfully. All image files have matching labels. Empty label files are distinguished from missing labels; visible furniture in an empty-labelled room is not necessarily a valid negative. The `export/` image is excluded from evaluation and should not be silently assigned to a split.

| ID | Original class | Proposed canonical name | Train objects | Validation objects | Test objects |
|---:|---|---|---:|---:|---:|
| 0 | Bed | `bed` | 1014 | 71 | 59 |
| 1 | Cabinet | `cabinet` | 251 | 55 | 28 |
| 2 | Carpet | `carpet` | 190 | 18 | 14 |
| 3 | Ceramic floor | `ceramic_floor` | 322 | 12 | 38 |
| 4 | Chair | `chair` | 495 | 76 | 48 |
| 5 | Closet | `closet` | 317 | 31 | 19 |
| 6 | Cupboard | `cupboard` | 250 | 25 | 16 |
| 7 | Curtains | `curtains` | 509 | 23 | 17 |
| 8 | Dining Table | `dining_table` | 192 | 39 | 20 |
| 9 | Door | `door` | 39 | 9 | 0 |
| 10 | Frame | `frame` | 157 | 3 | 2 |
| 11 | Futec frame | `futec_frame` | 196 | 7 | 15 |
| 12 | Futech tiles | `futech_tiles` | 48 | 3 | 2 |
| 13 | Gypsum Board | `gypsum_board` | 108 | 10 | 9 |
| 14 | Lamp | `lamp` | 205 | 32 | 28 |
| 15 | Nightstand | `nightstand` | 227 | 13 | 13 |
| 16 | Shelf | `shelf` | 185 | 38 | 24 |
| 17 | Sideboard | `sideboard` | 137 | 16 | 7 |
| 18 | Sofa | `couch` | 1804 | 238 | 176 |
| 19 | TV stand | `tv_stand` | 126 | 34 | 18 |
| 20 | Table | `table` | 576 | 109 | 200 |
| 21 | Transparent Closet | `transparent_closet` | 70 | 5 | 3 |
| 22 | Wall Panel | `wall_panel` | 287 | 11 | 13 |
| 23 | Window | `window` | 480 | 126 | 53 |
| 24 | Wooden floor | `wooden_floor` | 379 | 19 | 24 |

`Door` has no test examples, and `Frame` and `Futech tiles` have only two each. Small test counts cannot establish reliable per-class field performance.

## Safe class normalization without changing originals

There are **no exact or case-only duplicate class names in this archive**. It contains `Chair`, `Bed`, `Sofa`, and `Table`; it does not simultaneously contain lowercase `chair`, `bed`, `sofa`, or `table`, nor a separate `Couch` class. The requested aliases matter when joining this taxonomy to COCO or another source, rather than describing duplicate IDs already present here.

Use a versioned sidecar mapping: strip surrounding whitespace, case-fold, normalize spaces, and map `Sofa`/`sofa`/`Couch`/`couch` to `couch`. Keep original numeric IDs and original names in provenance. Thus Roboflow ID 18 remains traceable to `Sofa`, while its canonical name is `couch`. A full proposal is saved in [normalization_proposal.json](normalization_proposal.json); it has not been applied to the archive.

Keep `Dining Table` and `Table` separate in a strict taxonomy. They are related but are not case-only duplicates: `Table` may include coffee tables, desks, side tables, and dining tables. A merged `table` supercategory is useful for an explicitly labelled coarse evaluation, provided both GT and predictions use the same mapping. It is not permission to relabel all tables as desks.

Do not automatically merge Cabinet, Closet, Cupboard, Sideboard, and Transparent Closet. Likewise, `Futec frame` and `Futech tiles` require source-owner/annotation review; do not silently correct their spelling into other classes. A TV stand is not a COCO `tv`, and an oven front is not a stove/cooktop polygon.

For a future training view, write a separate destination and manifest with source archive hash, original ID, canonical ID, split membership, label conversion, and any reviewed corrections. Assign contiguous IDs only in that view, remapping the first token of every box or polygon annotation consistently. Preserve the original ZIP, avoid random re-splitting, and keep test labels out of training. No normalized training dataset has been created by this audit.

## Checkpoint evidence and deployed CPU export

`weights/yolo11n.pt` has task `detect`, 80 COCO class names, an upstream checkpoint date `2024-09-25T21:10:26.629566`, and Ultralytics version `8.2.100`. Its recorded training data is:

```text
/usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml
```

The embedded training record has 600 epochs and COCO training metrics. Those are upstream pretraining records, not evidence of a local Roboflow run. `epoch: -1` indicates a stripped inference checkpoint and is not, by itself, evidence that training never occurred.

Detector SHA-256:

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

The OpenVINO `metadata.yaml` explicitly describes a model trained on `coco.yaml`, with input size `[480, 480]`, batch one, and the same 80 names. On 20 deterministic held-out samples, PyTorch and OpenVINO produced the same detection counts at confidence 0.25: 26 boxes each, with minimum same-class box overlap IoU 0.9999976. This supports equivalence on that subset; it is not a full numeric-equivalence proof for every image.

COCO IDs relevant to this audit are chair 56, couch 57, bed 59, dining table 60, and oven 69. Desk and stove have no dedicated COCO IDs. Roboflow IDs are different: Bed 0, Chair 4, Dining Table 8, Sofa 18, Table 20. Evaluating raw IDs against one another would produce invalid results.

See [checkpoint_evidence.json](checkpoint_evidence.json) and [export_checkpoint_comparison.json](export_checkpoint_comparison.json). The original pose checkpoint also remains unchanged and records COCO-pose training; furniture labels do not constitute pose or temporal-event training data.

## CPU evaluation method

The measured model is the existing `weights/yolo11n_openvino_model`, on CPU only, with fixed 480×480 preprocessing, batch one, `rect=False`, class-aware NMS IoU 0.70 and `max_det=300`. All 739 test images are 416×416, as exported by Roboflow. Inference is unrestricted over all 80 classes; no furniture-only filter or runtime scheduling gate is applied to this diagnostic evaluation.

Two evaluations reuse the same predictions:

1. **Strict original classes:** COCO bed 59 → Roboflow Bed 0; chair 56 → Chair 4; couch 57 → Sofa 18; dining table 60 → Dining Table 8. The other dataset classes have no distinct output head. Table 20 deliberately remains unmatched because the COCO model cannot distinguish this dataset's two table IDs.
2. **Coarse table comparison:** the same mappings, but GT Dining Table 8 and Table 20 are merged before matching. All other classes are unchanged. This evaluates a broad table detector, not desk detection.

Box annotations are decoded as normalized center/size boxes. Polygon rows are converted individually to their enclosing boxes, then all GT boxes are scaled to the actual image dimensions. Original labels are never rewritten. This row-by-row conversion matters because the archive mixes boxes and polygons within some label files.

Matching uses the installed Ultralytics confidence-ordered, one-to-one matcher. Precision, recall, FP and missed counts use confidence ≥ 0.25 and IoU ≥ 0.50. AP50 uses a confidence sweep down to 0.001; AP50–95 is also saved. AP is per-class average precision; mAP is its mean across classes. FP means an unmatched model prediction relative to the supplied annotations; a visible but unlabelled chair still counts as an FP under this protocol. Missed GT includes wrong-class and insufficient-overlap predictions, not just cases with no visible model box.

Unsupported classes have zero recall/AP when GT exists, zero attributed predictions/FP, and undefined precision (`N/A`). A class with no test GT has undefined recall/AP. The unsupported zeros measure taxonomy coverage, not the quality of a trained 25-class model. Nonmapped COCO outputs such as person, plant, sink and oven are recorded separately in the summary rather than incorrectly assigned to furniture classes.

## Per class held out test results

| ID | Original class | Test GT | Precision | Recall | AP50 | False positives | Missed GT |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Bed | 59 | 53.03% | 59.32% | 50.55% | 31 | 24 |
| 1 | Cabinet | 28 | N/A | 0.00% | 0.00% | 0 | 28 |
| 2 | Carpet | 14 | N/A | 0.00% | 0.00% | 0 | 14 |
| 3 | Ceramic floor | 38 | N/A | 0.00% | 0.00% | 0 | 38 |
| 4 | Chair | 48 | 14.22% | 62.50% | 24.92% | 181 | 18 |
| 5 | Closet | 19 | N/A | 0.00% | 0.00% | 0 | 19 |
| 6 | Cupboard | 16 | N/A | 0.00% | 0.00% | 0 | 16 |
| 7 | Curtains | 17 | N/A | 0.00% | 0.00% | 0 | 17 |
| 8 | Dining Table | 20 | 20.00% | 55.00% | 29.57% | 44 | 9 |
| 9 | Door | 0 | N/A | N/A | N/A | 0 | 0 |
| 10 | Frame | 2 | N/A | 0.00% | 0.00% | 0 | 2 |
| 11 | Futec frame | 15 | N/A | 0.00% | 0.00% | 0 | 15 |
| 12 | Futech tiles | 2 | N/A | 0.00% | 0.00% | 0 | 2 |
| 13 | Gypsum Board | 9 | N/A | 0.00% | 0.00% | 0 | 9 |
| 14 | Lamp | 28 | N/A | 0.00% | 0.00% | 0 | 28 |
| 15 | Nightstand | 13 | N/A | 0.00% | 0.00% | 0 | 13 |
| 16 | Shelf | 24 | N/A | 0.00% | 0.00% | 0 | 24 |
| 17 | Sideboard | 7 | N/A | 0.00% | 0.00% | 0 | 7 |
| 18 | Sofa | 176 | 66.08% | 64.20% | 64.21% | 58 | 63 |
| 19 | TV stand | 18 | N/A | 0.00% | 0.00% | 0 | 18 |
| 20 | Table | 200 | N/A | 0.00% | 0.00% | 0 | 200 |
| 21 | Transparent Closet | 3 | N/A | 0.00% | 0.00% | 0 | 3 |
| 22 | Wall Panel | 13 | N/A | 0.00% | 0.00% | 0 | 13 |
| 23 | Window | 53 | N/A | 0.00% | 0.00% | 0 | 53 |
| 24 | Wooden floor | 24 | N/A | 0.00% | 0.00% | 0 | 24 |

Strict totals: **189 TP, 314 FP and 657 missed GT**, from 503 mapped predictions and 846 annotated objects. Strict mAP50 across the 24 classes with test GT is **7.05%**; averaged over only the four mapped classes it is **42.31%**. The former is dominated by unsupported classes, and neither number should be presented as fine-tuned Roboflow performance.

When Table and Dining Table are merged, their combined 220 GT objects yield **25 TP, 30 FP and 195 misses**, precision **45.45%**, recall **11.36%**, AP50 **18.18%**. Bed/chair/sofa metrics stay unchanged. Coarse overall totals become 203 TP, 300 FP and 643 misses. The low table recall demonstrates that alias normalization alone does not solve table/desk coverage.

The mean standalone CPU prediction-call latency was **44.38 ms/image** after warmup. It includes preprocessing, inference and postprocessing for this image set and confidence floor. It is not a new whole-pipeline FPS benchmark and does not replace the earlier scheduler/pose throughput results.

Machine-readable results: [strict CSV](strict_original_metrics.csv), [strict JSON](strict_original_metrics.json), [merged-table CSV](grouped_table_metrics.csv), [evaluation settings and summary](evaluation_summary.json), [all raw test predictions](test_predictions.jsonl), and [per-image matching counts](strict_original_per_image.json).

## Indoor room image inspection

The entire test set was inferred, and 28 diagnostic examples were saved with green GT and orange model boxes. Examples intentionally include successes, misses, kitchen/oven candidates, and incomplete annotations; they are not a random sample for estimating rates. These are existing dataset images, including photographic IKEA room scenes and a gaming-workstation photograph, alongside catalog cutouts, collages, and images that may be architectural renderings. No external indoor dataset was downloaded, and no live camera was used.

| Example | Visible result | Interpretation |
|---|---|---|
| 01 bedroom (`indoor_examples/01.jpg`, generated locally) | Bed detected at about 0.77 confidence. | Useful baseline on a contextual bedroom image. |
| 02 bedroom (`indoor_examples/02.jpg`, generated locally) | Annotated bed receives a couch prediction around 0.32 instead of bed. | Bed/couch semantic confusion; fails the intended bed class. |
| 03 IKEA living room (`indoor_examples/03.jpg`, generated locally) | An armchair-like object is labelled Bed; the model predicts chair. | Likely ground-truth class error, requiring review rather than blindly training on it. |
| 04 living room (`indoor_examples/04.jpg`, generated locally) | Sofa/couch detected around 0.88. | Contextual sofa success. |
| 08 chair (`indoor_examples/08.jpg`, generated locally) and 10 occluded chair (`indoor_examples/10.jpg`, generated locally) | A clear chair is detected, while a heavily occluded/noisy chair behind a table is missed. | Visibility and degraded-image failure modes matter. |
| 12 dining room (`indoor_examples/12.jpg`, generated locally) | Table and several visible chairs are detected; GT does not label those chairs. | Some of the 181 nominal chair FPs are real chairs omitted from annotations. Precision is annotation-relative, not a calibrated field false-alarm rate. |
| 14 dining table (`indoor_examples/14.jpg`, generated locally) | Model draws a table box but it does not match the supplied GT extent at IoU 0.50. | Localization/GT-extent mismatch can count as a miss despite visible detection. |
| 16 generic table (`indoor_examples/16.jpg`, generated locally) | COCO dining table around 0.54 over GT Table. | Strict taxonomy mismatch; coarse table evaluation can recover this case. |
| 17 cloth-covered table (`indoor_examples/17.jpg`, generated locally) | Model predicts bed around 0.73. | A genuine visible furniture confusion. |
| 18 kitchen range (`indoor_examples/18.jpg`, generated locally), 19 IKEA kitchen (`indoor_examples/19.jpg`, generated locally) | Existing photos contain ranges; COCO detects oven fronts, about 0.71 and 0.69 respectively. | Useful source images for new stove/cooktop annotations. Oven detection is not stove-polygon or burner-state validation. |
| 21 door scene (`indoor_examples/21.jpg`, generated locally) | Spurious oven prediction around 0.59 in a door/room scene. | An oven proxy would itself have false positives. |
| 22 gaming desk (`indoor_examples/22.jpg`, generated locally) | Real workstation and chair are visible. Model finds the chair, but cannot emit desk. The label file is empty. | Existing material for desk annotation; also an incomplete-negative example. No quantitative desk score is possible here. |
| 23 IKEA daybed (`indoor_examples/23.jpg`, generated locally) | The model emits both bed (about 0.57) and couch (about 0.28) hypotheses; chairs are also detected. | Overlapping hypotheses on a daybed illustrate why bed/couch policy and duplicate-object handling need review. |
| 24 (`indoor_examples/24.jpg`, generated locally), 25 (`indoor_examples/25.jpg`, generated locally), 26 (`indoor_examples/26.jpg`, generated locally), 27 (`indoor_examples/27.jpg`, generated locally) IKEA bedrooms | Bed detections around 0.83, 0.49, 0.39 and 0.87. | Several photographic bedroom successes, but confidence and box extent vary. |

Contact sheets: examples 00 to 11 (`indoor_contact_sheet_1.jpg`, generated locally), examples 12 to 23 (`indoor_contact_sheet_2.jpg`, generated locally), examples 24 to 27 (`indoor_contact_sheet_3.jpg`, generated locally). [indoor_examples.json](indoor_examples.json) records each original archive filename, GT and predictions. The examples support visual observations, not a claim of accuracy on unseen deployment cameras.

## Annotation integrity and coverage gaps

The archive has 7546 five-column box rows and 2888 polygon rows. Of 558 files mixing both formats, 495 are in train, 27 in validation, and **36 in test**. Those test files contain 116 GT objects. In the installed Ultralytics version, `verify_image_label` explicitly rejects files mixing segment and detection rows. A native-loader probe reproduced `ignoring corrupt image/label: labels mix segment and detection rows`, with corrupt flag 1. A naive direct validation/training invocation can therefore drop mixed-label images, disproportionately affecting contextual room scenes. This audit included all of them by converting each row independently. See [native_loader_evidence.json](native_loader_evidence.json) and [mixed_format_impact.json](mixed_format_impact.json).

There are no nonfinite, degenerate, or out-of-bounds annotations under the audit's row-level checks, and no exact duplicate boxes within an annotation file. There are 27 pairs of byte-identical images within validation/test splits: 17 validation pairs and 10 test pairs, 54 files total. Their labels agree. They overweight repeated scenes even though they do not cross splits. No byte-identical cross-split group was found. The original filename stems create 521 cross-split candidates, but names alone do not prove leakage. A conservative pHash/thumbnail check found no further cross-split candidates at its declared threshold; rotation, shear and other transformations can still evade it. See [content_integrity.json](content_integrity.json).

The README documents stretching images to 416×416, rotation up to ±30°, and shear up to ±15°. Some held-out examples also visibly contain heavy noise or composited furniture. Resizing those images to 480 does not restore the original camera aspect ratio or undo degradation. Performance on such exports cannot establish performance on an undistorted 640×480 room camera.

The test split has 614 single-object annotation files, 50 empty files, and only 75 files with multiple annotated objects. The visual inspection confirms that some room furniture is omitted or inconsistently labelled. Missing labels and taxonomy ambiguity must be fixed in a versioned evaluation view before treating precision as an operational false-alert rate.

Coverage is missing for **desk, stove/cooktop, burner indicators and active/off cooking states**. The dataset also has no dedicated person, keypoint, phone, laptop, bowl or cup annotations and no fall, inactivity, posture, tracking-ID or temporal-event labels. It cannot validate Stage 2 perception or temporal event accuracy. It does contain some relevant photographic desks and kitchens that can be relabelled first.

## What works and what must be fixed before perception acceptance

The original archive is usable input material, all images decode, class IDs are recoverable, some bed/chair/sofa detections work, CPU OpenVINO inference works, and the 20-image comparison supports the exported model matching the COCO checkpoint. Prior scheduler and alert tests remain valid tests of their stated computational/state-machine behavior.

Furniture calibration is not operationally learned from this dataset. `src/stage1_calibration/zone_calibrator.py` validates JSON polygons; it does not run a furniture detector to calibrate room zones. Existing room polygons are demonstration coordinates. In addition, the production secondary detector in `src/main_pipeline.py` is deliberately restricted to cell phone, laptop, bowl and cup. The unrestricted furniture predictions audited here are **not** outputs currently delivered by the live secondary detector. That runtime restriction is consistent with the existing Stage 2 design and has not been changed.

Before claiming end-to-end furniture/perception readiness:

1. **Define the task taxonomy.** Specify bed vs daybed/couch, chair vs armchair, dining table vs generic table vs desk, and stove/cooktop vs oven. Keep the original 25-class taxonomy and the new target mapping traceable.
2. **Create a derived detection view.** Convert segment rows individually to boxes, repair mixed files in the copy, review likely wrong labels, and annotate all target objects in evaluation rooms. Preserve source/archive hashes and split membership; group proven duplicate sources for evaluation weighting and future split hygiene.
3. **Close the desk/stove annotation gap using existing photos first.** Review the gaming desk and kitchen examples already provided, then assess whether enough distinct scenes remain for train/validation/test. If not, collect a small, labelled set from the actual deployment rooms before considering a broad external dataset. No Open Images V7 download is justified yet.
4. **Train and evaluate a separate Stage 1 furniture model only after the data view is ready.** A path such as `weights/stage1_furniture_yolo11n.pt` would keep its taxonomy distinct. Do not overwrite the existing COCO `yolo11n.pt` with a 25-class checkpoint: the Stage 2 loader requires phone/laptop/bowl/cup names and would lose those classes. Keep the pose model, scheduler and all alert algorithms intact.
5. **Measure rather than assume improvement.** Use training data only for fitting, validation for thresholds/model choices, and a frozen, fully annotated test view for final class precision/recall/AP and FP/miss inspection. Re-export the approved Stage 1 weights to OpenVINO at 480 and benchmark on CPU. Training and this future export have not been performed by the audit.
6. **Calibrate and test real rooms.** Establish camera-specific desk/chair/bed/couch/stove polygons, then use recorded indoor clips with labelled persons, objects and event timing to evaluate the full perception chain. The existing scripted-video test verifies scheduling, logging and transitions; it does not certify neural furniture detection, stove activation sensing or safety-event accuracy.

No change to the existing CPU scheduler, pose tracking, Stage 2 alert algorithms, checkpoints, or default zones is required to complete this audit. Additional domain annotations, a separate calibrated furniture model, and real indoor acceptance evidence are the outstanding work.

## Reproduction and preservation

Run from the project root with the existing virtual environment:

```bash
YOLO_CONFIG_DIR=/tmp/tower-ultralytics YOLO_AUTOINSTALL=false \
  .venv/bin/python reports/furniture_audit/run_audit.py
YOLO_CONFIG_DIR=/tmp/tower-ultralytics YOLO_AUTOINSTALL=false \
  .venv/bin/python reports/furniture_audit/inspect_indoor.py
```

The first command reads the archive, inventories all labels, performs four metric sanity checks, evaluates every test image on CPU, and saves strict/coarse metrics. The second writes overlays, decodes all images for integrity checks, checks duplicate candidates, and compares PyTorch/OpenVINO on 20 fixed test samples. It uses existing weights and existing data only. The additional native mixed-row probe is recorded in `native_loader_evidence.json`.

All 23 pre-existing source, test, config, requirements and README files captured at audit start retain their original SHA-256 values. The archive and both PT checkpoints also retain their original hashes. [preservation_checks.json](preservation_checks.json) records these checks; [protected_files_before.json](protected_files_before.json) contains the source baselines. Only files under `reports/furniture_audit/` were added. No training, dataset download, scheduler edit, pose-tracking edit, alert-algorithm edit, or original-label rewrite occurred.
