"""Synthetic 10 Hz poses / 2 Hz objects, temporal alerts, and JSONL/HUD wiring."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.main_pipeline import FrameScheduler
from src.stage2_monitors.common import DetectedObject, Person, RoomZones
from src.stage2_monitors.fall_inactivity_monitor import FallInactivityMonitor
from src.stage2_monitors.study_focus_coach import StudyFocusCoach
from src.stage2_monitors.kitchen_stove_guard import KitchenStoveGuard
from src.utils.event_logger import EventLogger


DAY = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
NIGHT = datetime(2026, 11, 14, 23, tzinfo=timezone.utc)
DESK = [[100, 130], [240, 130], [240, 260], [100, 260]]
COUCH = [[0, 150], [400, 150], [400, 350], [0, 350]]
STOVE = [[0, 0], [100, 0], [100, 100], [0, 100]]
KITCHEN = [[0, 0], [400, 0], [400, 400], [0, 400]]
PHONE = DetectedObject("cell phone", (130, 160, 150, 180), 0.95)
LAPTOP = DetectedObject("laptop", (100, 180, 220, 220), 0.9)
BOWL = DetectedObject("bowl", (20, 20, 60, 60), 0.9)


def person(track_id=1, *, lying=False, neck_angle=0.0, offset=0.0):
    points = np.tile([150.0, 160.0, 1.0], (17, 1))
    points[0, :2] = (150, 80)
    points[[5, 6], :2] = ((140, 140), (160, 140))
    points[[11, 12], :2] = ((140, 220), (160, 220))
    points[[9, 10], :2] = ((110, 180), (190, 180))
    points[[15, 16], :2] = ((140, 300), (160, 300))
    radians = np.deg2rad(neck_angle)
    neck = np.array([30 * np.sin(radians), -30 * np.cos(radians)])
    points[3, :2] = points[5, :2] + neck
    points[4, :2] = points[6, :2] + neck
    bbox = np.array([100, 70, 200, 310], dtype=float)
    if lying:
        xy = points[:, :2] - (150, 220)
        points[:, :2] = np.column_stack((-xy[:, 1], xy[:, 0])) + (150, 220)
        points[0, 1] = 260  # Rapid downward nose motion into the impact pose.
        bbox = np.array([50, 190, 330, 280], dtype=float)
    points[:, 0] += offset
    bbox[[0, 2]] += offset
    return Person(tuple(bbox), track_id, points)


class FallTests(unittest.TestCase):
    def test_floor_fall_alert_and_recovery_but_no_couch_fall(self):
        for zones, expect_fall in ((RoomZones(), True), (RoomZones({"couch": COUCH}), False)):
            with self.subTest(floor=expect_fall):
                monitor = FallInactivityMonitor(zones)
                events = monitor.update((person(),), 0.0, DAY)
                for i in range(1, 29):
                    emitted = monitor.update((person(lying=True),), i / 10, DAY)
                    if i < 26:
                        self.assertFalse(any(e.event == "FALL_ALERT" for e in emitted))
                    events += emitted
                falls = [e for e in events if e.event == "FALL_ALERT"]
                self.assertEqual(len(falls), int(expect_fall))
                self.assertEqual(monitor.tracks[1].history.maxlen, 30)
                if expect_fall:
                    self.assertGreater(falls[0].metrics["peak_head_velocity"], 120)
                    self.assertGreater(falls[0].metrics["torso_angle"], 60)
                    recovery = monitor.update((person(),), 2.9, DAY)
                    self.assertEqual([e.event for e in recovery], ["FALL_ALERT_RESOLVED"])

    def test_static_lying_pose_missing_points_and_pose_gaps_do_not_imply_fall(self):
        monitor = FallInactivityMonitor(RoomZones())
        events = []
        for i in range(60):
            events += monitor.update((person(lying=True),), i / 10, DAY)
        self.assertFalse(events)
        self.assertEqual(len(monitor.tracks[1].history), 30)
        monitor = FallInactivityMonitor(RoomZones())
        monitor.update((person(),), 0, DAY)
        monitor.update((person(lying=True),), 0.1, DAY)
        for i in range(30):
            self.assertFalse(monitor.update((person(lying=True),), 5 + i / 10, DAY))
        invalid = person(lying=True)
        invalid.keypoints[:, 2] = 0.0
        self.assertFalse(monitor.update((invalid,), 8.0, DAY))
        self.assertEqual(len(monitor.tracks[1].history), 0)

    def test_ankle_zone_exclusion_and_duplicate_pose_steps(self):
        zones = RoomZones({"bed": [[60, 205], [80, 205], [80, 235], [60, 235]]})
        monitor = FallInactivityMonitor(zones)
        monitor.update((person(lying=True),), 0, DAY)
        self.assertEqual(monitor.tracks[1].zone_name, "bed")
        for _ in range(50):
            monitor.update((person(lying=True),), 0, DAY)
        self.assertEqual(len(monitor.tracks[1].history), 1)

    def test_inactivity_outside_sleep_hours_and_reset_on_movement(self):
        for wall_time, expected in ((DAY, True), (NIGHT, False)):
            with self.subTest(daytime=expected):
                monitor = FallInactivityMonitor(RoomZones({"couch": COUCH}), max_inactive_seconds=3)
                events = []
                for i in range(33):
                    events += monitor.update((person(lying=True),), i / 10, wall_time)
                self.assertEqual(sum(e.event == "INACTIVITY_ALERT" for e in events), int(expected))
                if expected:
                    resolved = monitor.update((person(lying=True, offset=500),), 3.3, wall_time)
                    self.assertEqual([e.event for e in resolved], ["INACTIVITY_ALERT_RESOLVED"])
                    self.assertIsNone(monitor.tracks[1].inactive_since)
        # A missing observation breaks a multi-hour timer; it cannot count the gap.
        monitor = FallInactivityMonitor(RoomZones({"couch": COUCH}), max_inactive_seconds=3)
        monitor.update((person(lying=True),), 0, DAY)
        self.assertFalse(monitor.update((person(lying=True),), 10000, DAY))


class StudyTests(unittest.TestCase):
    def test_focus_phone_pause_distraction_and_resolution_at_two_hz(self):
        monitor = StudyFocusCoach(zones=RoomZones({"desk": DESK}), phone_distraction_seconds=1.5)
        for i in range(11):
            monitor.update((person(),), (LAPTOP,), i / 10, objects_timestamp=(i // 5) / 2)
        self.assertTrue(monitor.active)
        self.assertAlmostEqual(monitor.tracks[1].focused_seconds, 1.0)
        self.assertEqual(monitor.tracks[1].mode, "DEEP_FOCUS")
        events = []
        for i in range(11, 32):
            object_timestamp = 1.1 + ((i - 11) // 5) / 2
            events += monitor.update((person(),), (PHONE,), i / 10, objects_timestamp=object_timestamp)
        self.assertEqual(sum(e.event == "PHONE_DISTRACTION" for e in events), 1)
        self.assertAlmostEqual(monitor.tracks[1].focused_seconds, 1.0)
        resolved = monitor.update((person(),), (LAPTOP,), 3.2)
        self.assertEqual([e.event for e in resolved], ["PHONE_DISTRACTION_RESOLVED"])
        monitor.update((person(),), (LAPTOP,), 3.3)
        self.assertAlmostEqual(monitor.tracks[1].focused_seconds, 1.1)
        monitor.update((person(),), (), 4.4, objects_timestamp=3.2)
        self.assertEqual(monitor.tracks[1].mode, "UNKNOWN")
        self.assertAlmostEqual(monitor.tracks[1].focused_seconds, 1.1)

    def test_phone_wrist_distance_and_torso_intersection(self):
        chair = [[145, 150], [155, 150], [155, 160], [145, 160]]
        monitor = StudyFocusCoach(zones=RoomZones({"chair": chair}))
        nearby = DetectedObject("cell phone", (60, 180, 70, 190), 0.9)
        monitor.update((person(),), (nearby,), 0)
        self.assertTrue(monitor.active)  # Torso crosses chair; hip is outside.
        self.assertEqual(monitor.tracks[1].mode, "PHONE_PENDING")
        far = DetectedObject("cell phone", (900, 900, 920, 930), 0.9)
        monitor.update((person(),), (far,), 0.1)
        self.assertEqual(monitor.tracks[1].mode, "DEEP_FOCUS")

    def test_confidence_side_selection_ema_and_sustained_slump(self):
        monitor = StudyFocusCoach(zones=RoomZones({"desk": DESK}), slump_seconds=1)
        monitor.update((person(),), (), 0)
        monitor.update((person(neck_angle=60),), (), 0.1)
        self.assertAlmostEqual(monitor.tracks[1].smoothed_neck_angle, 12.0)
        monitor.update((person(neck_angle=60),), (), 0.15, pose_timestamp=0.1)
        self.assertAlmostEqual(monitor.tracks[1].smoothed_neck_angle, 12.0,
                               msg="Cached poses must not apply another EMA step")
        events = []
        for i in range(2, 26):
            events += monitor.update((person(neck_angle=60),), (), i / 10)
        self.assertEqual(sum(e.event == "ERGONOMIC_SLUMP_WARNING" for e in events), 1)
        resolved = []
        for i in range(26, 36):
            resolved += monitor.update((person(),), (), i / 10)
        self.assertEqual(sum(e.event == "ERGONOMIC_SLUMP_WARNING_RESOLVED" for e in resolved), 1)
        asymmetric = person(neck_angle=60)
        asymmetric.keypoints[[5, 3, 11], 2] = 0.4
        asymmetric.keypoints[[6, 4], 2] = 0.95
        asymmetric.keypoints[4, :2] = asymmetric.keypoints[6, :2] + (0, -30)
        monitor = StudyFocusCoach(zones=RoomZones({"desk": DESK}))
        monitor.update((asymmetric,), (), 0)
        self.assertAlmostEqual(monitor.tracks[1].smoothed_neck_angle, 0.0)
        asymmetric.keypoints[[5, 3], 2] = 0.99
        monitor.update((asymmetric,), (), 0.1)
        self.assertAlmostEqual(monitor.tracks[1].smoothed_neck_angle, 12.0)


class StoveTests(unittest.TestCase):
    def test_unattended_timer_alert_and_immediate_person_reentry_reset(self):
        monitor = KitchenStoveGuard(zones=RoomZones({"stove": STOVE, "kitchen": KITCHEN}),
                                    max_unattended_seconds=3)
        events = []
        for i in range(32):
            events += monitor.update((), (BOWL,), i / 10, objects_timestamp=(i // 5) / 2)
        self.assertTrue(monitor.active)
        self.assertEqual(sum(e.event == "UNATTENDED_COOKING_CRITICAL_ALERT" for e in events), 1)
        resolved = monitor.update((person(),), (BOWL,), 3.2)
        self.assertEqual(monitor.kitchen_person_count, 1)
        self.assertEqual(monitor.unattended_seconds, 0)
        self.assertEqual([e.event for e in resolved], ["UNATTENDED_COOKING_CRITICAL_ALERT_RESOLVED"])
        self.assertEqual(resolved[0].metrics["reason"], "person_returned")
        monitor.update((), (BOWL,), 3.3)
        self.assertEqual(monitor.unattended_seconds, 0)

    def test_cookware_ioa_kitchen_boundaries_and_stale_evidence(self):
        monitor = KitchenStoveGuard(zones=RoomZones({"stove": STOVE, "kitchen": KITCHEN}))
        half = DetectedObject("cup", (80, 20, 120, 60), 0.9)
        monitor.update((person(offset=1000),), (half,), 0)
        self.assertTrue(monitor.active)
        self.assertEqual(monitor.kitchen_person_count, 0)
        self.assertAlmostEqual(monitor.cookware_ioa, 0.5)
        monitor.update((), (half,), 1.1, pose_timestamp=0, objects_timestamp=0)
        self.assertFalse(monitor.active)
        self.assertEqual(monitor.unattended_seconds, 0)
        edge = DetectedObject("cup", (95, 20, 135, 60), 0.9)
        monitor.update((), (edge,), 1.2)
        self.assertFalse(monitor.active)

    def test_hsv_indicator_threshold_uses_polygon_not_bounding_rectangle(self):
        triangle = [[0, 0], [100, 0], [0, 100]]
        monitor = KitchenStoveGuard(zones=RoomZones({"stove": triangle}))
        frame = np.zeros((150, 150, 3), np.uint8)
        frame[80:100, 80:100] = (0, 0, 255)  # Outside the actual stove triangle.
        monitor.update((), (), 0, frame=frame)
        self.assertFalse(monitor.active)
        frame[10:30, 10:30] = (255, 255, 255)  # Bright white has no saturation.
        monitor.update((), (), 0.1, frame=frame)
        self.assertFalse(monitor.active)
        frame[10:30, 10:30] = (0, 0, 220)  # V must exceed, not equal, 220.
        monitor.update((), (), 0.2, frame=frame)
        self.assertFalse(monitor.active)
        frame[10:30, 10:30] = (0, 0, 255)
        monitor.update((), (), 0.3, frame=frame)
        self.assertTrue(monitor.active)
        self.assertGreater(monitor.indicator_fraction, 0.015)


class PipelineEventTests(unittest.TestCase):
    def test_all_alert_and_resolution_payloads_append_to_jsonl_and_render_hud(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "home_events.jsonl"
            zones = RoomZones({"desk": DESK, "stove": STOVE, "kitchen": KITCHEN,
                               "bed": [[500, 0], [900, 0], [900, 400], [500, 400]]})
            settings = {"max_inactive_seconds": 3, "phone_distraction_seconds": 1,
                        "slump_seconds": 0.5, "max_unattended_seconds": 3}
            with FrameScheduler(MagicMock(), room_zones=zones, monitor_settings=settings,
                                event_log=path) as pipeline:
                pipeline.logger = EventLogger(path, monotonic_origin=0, wall_origin=DAY)

                def step(timestamp, people=(), objects=()):
                    pipeline.people, pipeline.pose_at = people, timestamp
                    pipeline.objects, pipeline.objects_at = objects, timestamp
                    pipeline._update_monitors(timestamp, np.zeros((400, 1000, 3), np.uint8))

                step(0, (person(),))
                for i in range(1, 28):
                    step(i / 10, (person(lying=True),))
                hud = np.zeros((400, 1000, 3), np.uint8)
                with patch("src.main_pipeline.cv2.putText", wraps=cv2.putText) as draw:
                    pipeline._draw_hud(hud)
                labels = [call.args[1] for call in draw.call_args_list]
                self.assertTrue(any("FALL_ALERT" in label for label in labels))
                for metric in ("Torso Angle", "Neck EMA", "Focus Timer", "Unattended Stove Timer"):
                    self.assertTrue(any(metric in label for label in labels), metric)
                self.assertGreater(np.count_nonzero((hud[:, :, 2] > 150) & (hud[:, :, 1] == 0)), 1000)
                step(2.8, (person(),))
                for i in range(30, 51):
                    step(i / 10, (person(neck_angle=60),), (PHONE,))
                for i in range(51, 60):
                    step(i / 10, (person(),))
                for i in range(60, 92):
                    step(i / 10, (person(track_id=2, lying=True, offset=500),))
                step(9.2)
                for i in range(100, 132):
                    step(i / 10, (), (BOWL,))
                step(13.2, (person(),), (BOWL,))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            expected = {
                "FALL_ALERT": ("FallInactivityMonitor", 1),
                "INACTIVITY_ALERT": ("FallInactivityMonitor", 2),
                "PHONE_DISTRACTION": ("StudyFocusCoach", 1),
                "ERGONOMIC_SLUMP_WARNING": ("StudyFocusCoach", 1),
                "UNATTENDED_COOKING_CRITICAL_ALERT": ("KitchenStoveGuard", None),
            }
            self.assertEqual(len(records), 10)
            for event, (module, identifier) in expected.items():
                for suffix, state in (("", "ALERT"), ("_RESOLVED", "RESOLVED")):
                    matches = [r for r in records if r["event"] == event + suffix]
                    self.assertEqual(len(matches), 1, (event, suffix, records))
                    record = matches[0]
                    self.assertEqual(set(record), {"timestamp", "module", "event", "track_id", "metrics"})
                    self.assertEqual(record["module"], module)
                    self.assertEqual(record["track_id"], identifier)
                    self.assertEqual(record["metrics"]["state"], state)
                    self.assertIsNotNone(datetime.fromisoformat(record["timestamp"]).tzinfo)
            original = path.read_bytes()
            EventLogger(path).log([])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
