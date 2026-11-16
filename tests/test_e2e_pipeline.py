"""Real MP4 I/O and production CLI/HUD with explicit scripted observations.

Inference is injected because painted people cannot reliably drive a neural pose
tracker. The scheduler, media clock, monitors, HSV sensor, logger and renderer
are real. Real-model inference is covered separately by the CPU benchmark.
"""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from src.main_pipeline import InferenceResult, main
from src.stage1_calibration.zone_calibrator import validate_room_zones
from src.stage2_monitors.common import AlertEvent, DetectedObject, Person, PROJECT_ROOT, ROOM_CONFIG, load_room_config
from src.utils.event_logger import EventLogger


SCENE = PROJECT_ROOT / "tests" / "assets" / "e2e_test_scene.mp4"
OUTPUT = PROJECT_ROOT / "logs" / "e2e_hud_output.mp4"
EVENTS = PROJECT_ROOT / "logs" / "home_events.jsonl"
CONTACT_SHEET = PROJECT_ROOT / "logs" / "e2e_hud_contact_sheet.png"


def test_person(identifier, x=340.0, *, slumped=False, lying=False):
    points = np.tile([x, 260.0, 1.0], (17, 1))
    points[0, :2] = (x, 185)
    points[[1, 2], :2] = ((x - 5, 182), (x + 5, 182))
    points[[5, 6], :2] = ((x - 15, 230), (x + 15, 230))
    points[[7, 8], :2] = ((x - 30, 270), (x + 30, 270))
    points[[9, 10], :2] = ((x - 45, 310), (x + 45, 310))
    points[[11, 12], :2] = ((x - 15, 310), (x + 15, 310))
    points[[13, 14], :2] = ((x - 15, 365), (x + 15, 365))
    points[[15, 16], :2] = ((x - 15, 420), (x + 15, 420))
    neck = np.array([26.0, -15.0]) if slumped else np.array([0.0, -30.0])
    points[3, :2], points[4, :2] = points[5, :2] + neck, points[6, :2] + neck
    bbox = (x - 60, 165, x + 65, 435)
    if lying:
        # All extremities clear the protected couch and chair polygons.
        xy = points[:, :2] - (x, 310)
        points[:, :2] = np.column_stack((-xy[:, 1], xy[:, 0])) + (x, 450)
        points[0, 1] = 465
        bbox = (225, 395, 510, 478)
    return Person(bbox, identifier, points)


def observations(index):
    laptop = DetectedObject("laptop", (310, 280, 375, 310), 0.95)
    phone = DetectedObject("cell phone", (285, 300, 300, 318), 0.95)
    if index < 90:
        return (test_person(1), test_person(2, 535), test_person(5, 130)), (laptop,)
    if index < 180:
        return (), ()  # Fresh empty detections prove the kitchen is unoccupied.
    if index < 270:
        # Reacquire the desk track with a new ID after one untracked pose.
        identifier = None if index < 183 else 11
        return (test_person(identifier, slumped=True), test_person(20, 535)), (phone, laptop)
    return (test_person(11, lying=True), test_person(20, 535)), ()


class ScriptedModels:
    def __init__(self):
        self.origin = None
        self.pose_runs = self.object_runs = 0

    def infer(self, frame, captured_at, run_pose, run_objects):
        if self.origin is None:
            self.origin = captured_at
        index = round((captured_at - self.origin) * 30)
        started = time.perf_counter()
        people, objects = observations(index)
        elapsed = (time.perf_counter() - started) * 1000
        self.pose_runs += int(run_pose)
        self.object_runs += int(run_objects)
        return InferenceResult(captured_at, people if run_pose else None,
                               objects if run_objects else None,
                               elapsed if run_pose else None, elapsed if run_objects else None)


def generate_scene():
    SCENE.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(SCENE), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the MP4 integration-test fixture")
    photo = cv2.imread(str(ASSETS / "bus.jpg"))
    if photo is None:
        raise RuntimeError("Ultralytics real-image fixture is missing")
    photo = cv2.resize(photo, (120, 150))
    try:
        for index in range(360):
            frame = np.full((480, 640, 3), 45, np.uint8)
            frame[65:215, 15:135] = photo
            cv2.rectangle(frame, (40, 260), (220, 420), (85, 75, 65), -1)
            cv2.rectangle(frame, (250, 220), (430, 400), (70, 85, 90), -1)
            cv2.rectangle(frame, (470, 240), (600, 360), (60, 60, 60), -1)
            cv2.circle(frame, (535, 295), 14, (0, 0, 255), -1)
            for person in observations(index)[0]:
                xy = person.keypoints[:, :2].astype(np.int32)
                for a, b in ((5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9),
                             (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)):
                    cv2.line(frame, tuple(xy[a]), tuple(xy[b]), (145, 150, 160), 7)
                cv2.circle(frame, tuple(xy[0]), 10, (120, 140, 175), -1)
            cv2.putText(frame, f"INJECTED TEST SCENE | phase {index // 90 + 1}", (12, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            writer.write(frame)
    finally:
        writer.release()


class ZoneValidationTests(unittest.TestCase):
    def test_named_defaults_and_legacy_polygons(self):
        config = validate_room_zones(ROOM_CONFIG)
        zones, _ = load_room_config(ROOM_CONFIG)
        for name, entry in config["zones"].items():
            self.assertGreaterEqual(len(entry["polygon"]), 3)
            self.assertIn(name, zones.outlines)
            self.assertGreater(zones.get(entry["type"]).area, 0)
        self.assertTrue(zones.contains("couch", np.array([100, 300])))
        self.assertEqual(config["zones"]["floor_0"]["polygon"], [[0, 350], [640, 350], [640, 480], [0, 480]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            path.write_text(json.dumps({"study_desk": [[0, 0], [100, 0], [100, 100]]}))
            legacy, _ = load_room_config(path)
            self.assertIsNotNone(legacy.get("desk"))

    def test_invalid_polygons_have_field_specific_errors(self):
        for points in (None, [], [[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]]):
            with self.subTest(points=points), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "zones.json"
                path.write_text(json.dumps({"zones": {"couch_0": {"type": "couch", "polygon": points}}}))
                with self.assertRaisesRegex(ValueError, r"zones.couch_0.polygon"):
                    validate_room_zones(path)

    def test_output_cannot_overwrite_input(self):
        with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            main(["--source", str(SCENE), "--headless", "--save-video", str(SCENE)])
        self.assertIn("must differ from the source", stderr.getvalue())

    def test_delayed_observation_is_logged_at_ordered_publication_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = EventLogger(path, monotonic_origin=0,
                                 wall_origin=datetime(2026, 11, 14, 12, tzinfo=timezone.utc))
            logger.log([AlertEvent(9, "StudyFocusCoach", "PHONE_DISTRACTION", 1, {})])
            logger.log([AlertEvent(8, "FallInactivityMonitor", "FALL_ALERT", 1, {})], timestamp=10)
            times = [json.loads(line)["timestamp"] for line in path.read_text().splitlines()]
            self.assertEqual(times, sorted(times))


class EndToEndTests(unittest.TestCase):
    def test_twelve_second_headless_cli_video_alerts_and_track_reacquisition(self):
        generate_scene()
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        previous_size = EVENTS.stat().st_size if EVENTS.exists() else 0
        models = ScriptedModels()
        with patch("src.main_pipeline.CPUModels", return_value=models), \
             patch("src.main_pipeline.cv2.imshow", side_effect=AssertionError("headless imshow")), \
             patch("src.main_pipeline.cv2.waitKey", side_effect=AssertionError("headless waitKey")), \
             patch("src.main_pipeline.cv2.destroyAllWindows", side_effect=AssertionError("headless GUI cleanup")):
            main(["--source", str(SCENE), "--headless", "--fast-test-timers",
                  "--allow-demo-zones",
                  "--save-video", str(OUTPUT), "--event-log", str(EVENTS)])
        self.assertTrue(OUTPUT.is_file())
        self.assertGreater(OUTPUT.stat().st_size, 10000)
        capture = cv2.VideoCapture(str(OUTPUT))
        count, samples = 0, {}
        try:
            self.assertTrue(capture.isOpened())
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (480, 640, 3))
                if count in (45, 165, 260, 351):
                    samples[count] = frame
                count += 1
        finally:
            capture.release()
        self.assertEqual(count, 360)
        self.assertGreaterEqual(models.pose_runs, 110)
        self.assertEqual(models.object_runs, 24)
        with EVENTS.open("rb") as stream:
            stream.seek(previous_size)
            records = [json.loads(line) for line in stream if line.strip()]
        timestamps = [datetime.fromisoformat(row["timestamp"]) for row in records]
        self.assertEqual(timestamps, sorted(timestamps))
        required = {"UNATTENDED_COOKING_CRITICAL_ALERT", "PHONE_DISTRACTION",
                    "ERGONOMIC_SLUMP_WARNING", "FALL_ALERT"}
        triggers = {row["event"] for row in records if row["metrics"]["state"] == "ALERT"}
        self.assertTrue(required <= triggers, records)
        for event in required - {"FALL_ALERT"}:
            self.assertIn(event + "_RESOLVED", [row["event"] for row in records])
        self.assertEqual(sum(row["event"] == "FALL_ALERT" for row in records), 1)
        active = set()
        for row in records:
            key = (row["module"], row["event"].removesuffix("_RESOLVED"), row["track_id"])
            if row["metrics"]["state"] == "ALERT":
                self.assertNotIn(key, active, "Duplicate alert on consecutive frames")
                active.add(key)
            else:
                self.assertIn(key, active, "Resolution must follow its alert")
                active.remove(key)
        for event in ("PHONE_DISTRACTION", "ERGONOMIC_SLUMP_WARNING", "FALL_ALERT"):
            self.assertTrue(all(row["track_id"] == 11 for row in records if row["event"] == event))

        def red_pixels(frame):
            banner = frame[52:98]
            return np.count_nonzero((banner[:, :, 2] > 120) &
                                    (banner[:, :, 1] < 90) & (banner[:, :, 0] < 90))

        self.assertLess(red_pixels(samples[45]), 100)
        self.assertGreater(red_pixels(samples[165]), 5000)
        self.assertGreater(red_pixels(samples[260]), 10000)
        self.assertGreater(red_pixels(samples[351]), 5000)
        wrist = samples[45][306:315, 291:300]
        self.assertGreater(np.count_nonzero((wrist[:, :, 1] > 150) & (wrist[:, :, 2] > 150)
                                           & (wrist[:, :, 0] < 120)), 10)
        couch_edge = samples[45][300:400, 39:43]
        self.assertGreater(np.count_nonzero((couch_edge[:, :, 0] > 120) &
                                           (couch_edge[:, :, 1] > 80) & (couch_edge[:, :, 2] < 120)), 50)
        sheet = np.vstack((np.hstack((samples[45], samples[165])),
                           np.hstack((samples[260], samples[351]))))
        self.assertTrue(cv2.imwrite(str(CONTACT_SHEET), sheet))
        print(f"\nE2E: {count} rendered frames, {len(records)} transitions, "
              f"pose/object runs={models.pose_runs}/{models.object_runs}; {OUTPUT}")


if __name__ == "__main__":
    unittest.main()
