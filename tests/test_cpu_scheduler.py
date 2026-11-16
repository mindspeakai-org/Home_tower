"""Focused scheduling, shared tracking, gating, and export fallback checks."""

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.main_pipeline import FrameScheduler, InferenceResult
from src.stage2_monitors.monitors import DetectedObject, Person, parse_zone
from src.utils import export_cpu_models as exporter


KEYPOINTS = np.tile([20.0, 25.0, 1.0], (17, 1))
KEYPOINTS[[5, 6], 1] = 20
KEYPOINTS[[11, 12], 1] = 30
PERSON = Person((10, 10, 30, 40), 7, KEYPOINTS)
ZONE = parse_zone([[0, 0], [100, 0], [100, 100], [0, 100]])


class FakeModels:
    def __init__(self):
        self.calls = []

    def infer(self, frame, captured_at, run_pose, run_objects):
        self.calls.append((run_pose, run_objects))
        return InferenceResult(
            captured_at,
            (PERSON,) if run_pose else None,
            (DetectedObject("cup", (1, 1, 5, 5), 0.9),) if run_objects else None,
            1.0 if run_pose else None,
            1.0 if run_objects else None,
        )


class SchedulerTests(unittest.TestCase):
    def test_strides_and_shared_person_ids(self):
        models = FakeModels()
        with FrameScheduler(models, stove_active=True) as pipeline:
            for _ in range(31):
                pipeline.process_frame(np.zeros((120, 160, 3), np.uint8))
                pipeline.drain()
            self.assertEqual(pipeline.pose_runs, 11)  # frames 0, 3, ..., 30
            self.assertEqual(pipeline.object_runs, 3)  # frames 0, 15, 30
            for monitor in (pipeline.fall, pipeline.study, pipeline.stove):
                self.assertIs(monitor.people, pipeline.people)
                self.assertEqual(monitor.people[0].track_id, 7)
            self.assertEqual(pipeline.stove.person_count, 1)

    def test_secondary_detector_requires_desk_person_or_active_stove(self):
        for desk, expected_runs in ((None, 0), (ZONE, 1)):
            with self.subTest(desk=desk is not None):
                with FrameScheduler(FakeModels(), desk_zone=desk) as pipeline:
                    for _ in range(16):
                        pipeline.process_frame(np.zeros((120, 160, 3), np.uint8))
                        pipeline.drain()
                    self.assertEqual(pipeline.object_runs, expected_runs)
                    self.assertEqual(pipeline.study.active, desk is not None)

    def test_busy_worker_does_not_block_or_queue_frames(self):
        started, release = threading.Event(), threading.Event()
        models = FakeModels()
        original = models.infer

        def blocked(*args):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release inference worker")
            self.assertFalse(args[0].any(), "HUD must not mutate worker's frame")
            return original(*args)

        models.infer = blocked
        with FrameScheduler(models, stove_active=True) as pipeline:
            try:
                pipeline.process_frame(np.zeros((120, 160, 3), np.uint8))
                self.assertTrue(started.wait(timeout=5))
                pending = dict(pipeline._pending)
                for _ in range(30):
                    pipeline.process_frame(np.zeros((120, 160, 3), np.uint8))
                self.assertEqual(pipeline._pending, pending)
                self.assertEqual(pipeline.frame_index, 31)
            finally:
                release.set()
            pipeline.drain()
            self.assertEqual(len(models.calls), 2)  # At most one call per model.

    def test_stale_people_expire_for_all_monitors(self):
        with FrameScheduler(FakeModels(), desk_zone=ZONE) as pipeline:
            pipeline.process_frame(np.zeros((120, 160, 3), np.uint8))
            pipeline.drain()
            self.assertTrue(pipeline.study.active)
            with patch("src.main_pipeline.time.perf_counter", return_value=time.perf_counter() + 2):
                pipeline.process_frame(np.zeros((120, 160, 3), np.uint8))
            for monitor in (pipeline.fall, pipeline.study, pipeline.stove):
                self.assertEqual(monitor.people, ())
            self.assertFalse(pipeline.study.active)
            self.assertEqual(pipeline.stove.person_count, 0)

    def test_worker_errors_are_propagated(self):
        models = FakeModels()
        models.infer = MagicMock(side_effect=RuntimeError("inference failed"))
        with FrameScheduler(models) as pipeline:
            pipeline.process_frame(np.zeros((120, 160, 3), np.uint8))
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                pipeline.drain()


class ExportFallbackTests(unittest.TestCase):
    def test_architecture_and_failed_openvino_fall_back_to_onnx(self):
        for x86 in (True, False):
            with self.subTest(x86=x86), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "yolo11n.pt"
                checkpoint.touch()
                model = MagicMock()
                model.export.side_effect = (
                    [RuntimeError("OpenVINO unavailable"), "model.onnx"] if x86 else ["model.onnx"]
                )
                with patch.object(exporter, "is_x86_64", return_value=x86), \
                     patch.object(exporter, "YOLO", return_value=model), \
                     patch.object(exporter, "validate_model"):
                    self.assertEqual(exporter.export_model(checkpoint), Path("model.onnx"))
                formats = [call.kwargs["format"] for call in model.export.call_args_list]
                self.assertEqual(formats, ["openvino", "onnx"] if x86 else ["onnx"])
                for call in model.export.call_args_list:
                    self.assertEqual(call.kwargs["device"], "cpu")
                    self.assertEqual(call.kwargs["imgsz"], 480)


if __name__ == "__main__":
    unittest.main()
