# Smart Home Computer Vision Monitoring

Stage one prepares furniture detections for room-zone calibration. The CPU live
pipeline shares pose tracking across the stage-two alert monitors.

```text
src/
  stage1_calibration/download_stage1_data.py
  stage2_monitors/
  utils/export_cpu_models.py
  main_pipeline.py
configs/
data/          # supplied furniture dataset archive
datasets/      # immutable source cache, sanitized shadow, review candidates
tests/
weights/       # downloaded YOLO11 checkpoints
```

Create a Python 3.10+ environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Installing CPU PyTorch before the requirements avoids downloading CUDA packages.

Use the supplied Roboflow archive to prepare a detection-only shadow dataset:

```bash
python -m src.stage1_calibration.dataset_sanitizer
```

This converts each box/polygon annotation independently while preserving class
IDs. Names use canonical aliases (`Sofa` → `couch`); `Dining Table` and `Table`
stay distinct. It verifies every split with Ultralytics' native loader. A ZIP
requires one extracted source cache; all shadow images are symlinks, not image
copies. Source/cache bytes remain unchanged. Formatting fixes do not constitute
manual annotation review or establish ground-truth accuracy.

The supplied `data/furniture detection.v20i.yolov11.zip` remains available as a
separate 25-class source dataset. Original images/labels are preserved in the
read-only cache; standardized shadow labels are separate. The legacy Open Images
download script is retained but is not part of this workflow. No additional
dataset download is needed for Stage 1 review.

To download only the weights and run the smoke tests:

```bash
python -c 'from src.stage1_calibration.download_stage1_data import download_weights; download_weights()'
python -m unittest tests.test_models_smoke -v
```

The tests load both local models and run a blank-image inference on CPU and,
when available, CUDA. The CUDA test is explicitly skipped without a CUDA GPU.
These pretrained checkpoints use their original class sets; exporting a dataset
does not train them on the five calibration classes.

## CPU live pipeline

Export both checkpoints and start a camera (use `--source video.mp4` for a file):

```bash
python -m src.utils.export_cpu_models
python -m src.main_pipeline --source 0 --zones configs/room_zones.json \
  --calibrate --calibration-reviewer YOUR_NAME
```

The exporter prefers OpenVINO on x86_64/AMD64 and automatically exports ONNX
elsewhere, or if OpenVINO export/CPU validation fails. Both formats use fixed
480×480 inputs and batch size one. Runtime loads and warms exported models on
CPU before capture. It tries an existing ONNX model if OpenVINO fails to load;
it reports an actionable error if no usable export exists. `onnx` is needed for
export, `onnxruntime` for inference, and `lap` for Ultralytics' ByteTrack tracker.

`configs/room_zones.json` includes valid 640×480 demonstration geometry. Replace
it with camera-specific calibration before live deployment. Named entries use
the following format (the original polygon-only format is also supported):

```json
{
  "zones": {
    "desk_0": {
      "type": "desk",
      "polygon": [[250, 220], [430, 220], [430, 400], [250, 400]]
    }
  }
}
```

`validate_room_zones(config_path)` rejects null, short, nonfinite, or invalid
polygons with the offending field name. Omit unused zones rather than supplying
null. Multiple named polygons of one type are combined for monitor geometry.
Live startup additionally requires manually approved zones, reviewer/time/source
metadata, matching source dimensions, and acknowledgment of overlaps. Default
demonstration coordinates are explicitly unapproved and cannot enable a camera.

## Stage 1 review and approval

The deterministic 125-image review pack is in
`reports/stage1_validation/review_subset/`. It covers every class available in
the original test split, all 36 mixed-format test files, empty labels, and known
annotation problems. No decisions have been automatically accepted.

```bash
python -m src.stage1_calibration.annotation_review --reviewer YOUR_NAME
```

In the native review window, drag to add a box, use `[`/`]` to select a class,
**C** to clear, **X** to remove the last box, **A** to accept/correct, **R** to
reject, **N/P** to skip/return, and **Q** to quit. Each accepted decision records
the operator and time. Headless users can inspect `overlays/` and edit individual
`decisions/*.json`: supply reviewer/time and choose `accepted`, `corrected`, or
`rejected`. Changed labels require `corrected`. Pending/rejected entries never
enter the cleaned evaluation copy. Predictions are not ground truth.

After review, build and evaluate an immutable clean snapshot in one command:

```bash
python -m src.stage1_calibration.validate_stage1 --skip-rooms \
  --clean-dir datasets/furniture_reviewed_evaluation_v1
```

Use a new output version for a changed review. With zero reviewed entries, no
clean dataset is created, `clean_metrics.csv` contains only its header, and the
status explicitly reports unavailable metrics. The strict original baseline
remains separate.

Calibration calls the stock CPU detector once, independently of Stage 2's
phone/laptop/bowl/cup filter. Bed, couch, chair and dining-table boxes become
unapproved polygons. COCO ovens stay `oven`; monitor/laptop boxes are context.
Neither context nor room mode automatically converts dining tables to desks or
oven fronts to cooktops. Bootstrap workstation/kitchen candidates are pending
review in `datasets/stage1_workstation_kitchen_bootstrap/` and are not a training
dataset.

For an existing camera frame, export proposals or edit them interactively:

```bash
python -m src.stage1_calibration.zone_calibrator --image camera_frame.png \
  --output logs/camera_zone_proposals.json
python -m src.stage1_calibration.zone_calibrator --image camera_frame.png \
  --output configs/room_zones.json --edit --reviewer YOUR_NAME
```

The zone editor supports **TAB** selection, **T** type selection, **N** new zone,
clicks then **C** to close a polygon, vertex dragging, **E** redraw, **D** delete,
and **A** confirmation of each zone. **S** then **Y** explicitly approves saving
and the displayed overlap count. Draw actual desk and stove polygons manually.
Existing approved calibration files are never overwritten; choose a new path
for a revised calibration. Headless operators can edit proposal JSON, mark each
reviewed zone `approved: true`, then explicitly submit it with `--approve FILE
--output PATH --reviewer NAME --acknowledge-overlaps`.

At camera startup, approved boundaries are preserved and one fresh proposal
preview is exported beside the event log. Calibration is never run at full
camera FPS. Source-resolution mismatch or unapproved geometry stops monitoring
with a configuration error. Camera hardware and interactive review still need
operator validation; dataset-image proposals are not camera calibration.

Mid-hip/ankle points resolve bed/couch occupancy; mid-hip/torso intersections
activate desk/chair monitoring. Boundary points count as inside. Keypoints with
confidence below 0.3 are unavailable. Without calibrated bed/couch polygons,
fall zone resolution defaults to floor; without desk/chair polygons, study
monitoring is inactive. Kitchen counts use person bottom-center points and
default to all visible people without a kitchen polygon.

The configured stove polygon arms the guard and enables its gated object checks,
including initial cookware detection before a burner has been confirmed active.
Stove activity comes from cookware overlap (intersection divided by detection
box area > 0.3), or saturated bright HSV pixels occupying > 1.5% of the actual
stove polygon. The HSV check uses original frames before the HUD is drawn.
`--stove-active` or **s** controls an additional manual override; **q** exits.

At a 25–30 FPS input rate, pose runs every third frame (about 8–10 FPS) and the
object detector is eligible every fifteenth frame (about 2 FPS), only while a
person occupies the desk zone or the stove guard is active. The object detector
is restricted to cell phone, laptop, bowl, and cup; person boxes, keypoints, and
track IDs come exclusively from the pose model and are shared by all monitors.

Two background worker slots handle pose and object inference independently,
with at most one outstanding job per model so tracking is never concurrent.
The main loop continues reading and displaying source-rate frames; if a model
is busy, stale frames are skipped rather than queued. The next eligible job
uses the latest frame as soon as that model is free, with at least the stride's
number of frames between submissions. Cached detections expire one second after
their source frame. The HUD reports source-loop FPS, foreground processing time,
and separate pose/object inference latencies, plus active red alert banners,
torso/neck angles, per-track focus time, and unattended stove time. Zone outlines
and confidence-filtered COCO skeletons are drawn underneath a readable metrics
panel. File-input alert timers use frame timestamps; capture/display FPS and
inference latency counters continue to measure actual wall-clock performance.

To run without a display and save every rendered source frame:

```bash
python -m src.main_pipeline --source tests/assets/e2e_test_scene.mp4 \
  --headless --fast-test-timers --allow-demo-zones --save-video logs/e2e_hud_output.mp4
```

`--headless` avoids all OpenCV window calls. `--save-video` writes MP4 at the
source FPS and resolution and refuses to overwrite its input file.
`--fast-test-timers` sets inactivity, phone, slump, and unattended thresholds to
2 seconds; the fall stillness requirement remains 2.5 seconds. Omit this test
flag for live monitoring.
`--allow-demo-zones` is an explicit file/synthetic test opt-in; it is forbidden
for live camera indices, devices, and streams.

## Stage 2 alert policies

- `FallInactivityMonitor` keeps 30 fresh pose steps per track. Floor falls require
  a wide box, > 60° torso tilt, > 120 px/s downward nose velocity in the history,
  then 2.5 s with temporal keypoint std < 4 px. An alert stays active while the
  person remains horizontal on the floor; standing or entering bed/couch resolves
  it. Bed/couch inactivity requires rolling std < 3 px outside local sleep hours
  (default 22:00–07:00) for 7200 s.
- `StudyFocusCoach` accumulates focus only with fresh, phone-free object evidence
  while at the desk/chair. Phone overlap with the upper body or distance ≤ 50 px
  from either wrist pauses focus immediately and alerts after 15 s continuously.
  The higher-confidence shoulder/ear side provides neck tilt, with EMA alpha 0.2
  applied once per fresh pose. Tilt > 35° continuously for 20 s emits a warning.
- `KitchenStoveGuard` alerts after 300 s of continuous active-stove evidence and
  zero kitchen occupants. Person re-entry resets the timer immediately. Turning
  the stove off or losing evidence also breaks the continuous countdown.

Timer thresholds and sleep hours are configurable in the `settings` section of
`configs/room_zones.json`. Durations use monotonic elapsed seconds, not frame
counts. Missing poses, stale objects, or observation gaps break continuous
evidence; they do not count as stillness, focus, or confirmed absence. Track-loss
resolutions include their reason and do not imply that a person recovered.

Transitions append to `logs/home_events.jsonl` (override with `--event-log`). Each
record has an ISO timestamp with timezone, module, event, track ID (or null for
the stove), and metrics. Trigger events use the alert name; resolutions append
`_RESOLVED`. Metrics include `state` and `transition`. Persistent alerts refresh
their HUD metrics without writing duplicate events.

Verify scheduling and real CPU throughput:

```bash
python -m unittest tests.test_cpu_scheduler -v
python -m unittest tests.test_cpu_throughput -v
python -m unittest tests.test_stage2_alerts -v
python -m unittest tests.test_e2e_pipeline -v
python -m unittest tests.test_stage1_runtime_calibration -v
python -m unittest discover -s tests -p "test_*.py" -v
```

The E2E test generates a 12-second, 360-frame scene and invokes the production
CLI with scripted pose/object observations, real MP4 decoding/encoding, real
HSV sensing and all temporal monitors. This tests state transitions and track
loss/reacquisition independently of neural detection quality. It appends its
test transitions to `logs/home_events.jsonl`, preserves existing log entries,
and saves `logs/e2e_hud_output.mp4` and `logs/e2e_hud_contact_sheet.png`. The final
fall remains active at EOF; ending a clip does not imply recovery. Real CPU
model inference is verified by the throughput and model smoke tests.

The throughput benchmark uses Ultralytics' included person image, live tracking,
HUD drawing, and an active object detector over 150 frames paced at 30 FPS. It
excludes model compilation/warmup and frame pacing from processing time, checks
at least 8 pose inferences per second, and requires both foreground and
conservative amortized CPU processing (including background inference and final
drain) to average below 80 ms per source frame. This is a measured hardware
budget, not a guarantee for every CPU, video resolution, camera, or display.
