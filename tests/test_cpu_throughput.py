"""Real CPU inference benchmark, including conservative amortized inference cost.

Warmup/export/compilation are excluded. Run at a paced 30 FPS with the actual
models, tracking and HUD, people in the image, and the secondary detector active.
No mocked inference or timing-only blank frames are used for acceptance.
"""

import time
import unittest

import cv2
import numpy as np
from ultralytics.utils import ASSETS

from src.main_pipeline import CPUModels, FrameScheduler
from src.stage2_monitors.monitors import parse_zone


class CPUThroughputTests(unittest.TestCase):
    def test_average_frame_latency_under_80ms(self):
        models = CPUModels()
        image = cv2.imread(str(ASSETS / "bus.jpg"))
        self.assertIsNotNone(image, "Ultralytics real-person benchmark image is required")
        image = cv2.resize(image, (480, 640))
        desk_zone = parse_zone([[0, 0], [479, 0], [479, 639], [0, 639]])
        # Warm real-person tracking and both models before timing steady state.
        for _ in range(3):
            warm = models.infer(image, time.perf_counter(), True, True)
        self.assertTrue(warm.people, "Benchmark must actually detect people")
        self.assertTrue(any(person.track_id is not None for person in warm.people))
        self.assertTrue(all(obj.label in {"cell phone", "laptop", "bowl", "cup"}
                            for obj in warm.objects))
        latencies = []
        frames = 150
        with FrameScheduler(models, desk_zone=desk_zone, stove_active=True) as pipeline:
            start = time.perf_counter()
            for i in range(frames):
                # Capture pacing is not processing time. Include copies, scheduler,
                # result collection, all monitor updates, and HUD in frame latency.
                time.sleep(max(0.0, start + i / 30 - time.perf_counter()))
                frame_start = time.perf_counter()
                pipeline.process_frame(image.copy())
                latencies.append((time.perf_counter() - frame_start) * 1000)
            drain_start = time.perf_counter()
            pipeline.drain()
            drain_ms = (time.perf_counter() - drain_start) * 1000
            elapsed = time.perf_counter() - start
            frame_ms = float(np.mean(latencies))
            # Counting background inference in addition to foreground work is
            # conservative (overlap is counted twice), preventing an empty fast
            # main loop from passing while inference cannot keep up.
            amortized_ms = (
                sum(latencies) + pipeline.pose_total_ms + pipeline.object_total_ms + drain_ms
            ) / frames
            pose_ms = pipeline.pose_total_ms / max(pipeline.pose_runs, 1)
            obj_ms = pipeline.object_total_ms / max(pipeline.object_runs, 1)
            print(
                f"\nCPU benchmark: {frames / elapsed:.1f} frame FPS, "
                f"{pipeline.pose_runs / elapsed:.1f} pose FPS, "
                f"{pipeline.object_runs / elapsed:.1f} object FPS; "
                f"frame={frame_ms:.2f} ms, amortized={amortized_ms:.2f} ms, "
                f"pose={pose_ms:.2f} ms, objects={obj_ms:.2f} ms; "
                f"runs={pipeline.pose_runs}/{pipeline.object_runs}"
            )
            self.assertGreaterEqual(pipeline.pose_runs / elapsed, 8.0,
                                    "Pose inference must sustain at least 8 FPS")
            self.assertGreaterEqual(frames / elapsed, 28.0,
                                    "Foreground processing must keep up with the 30 FPS source")
            self.assertGreaterEqual(pipeline.object_runs, 5,
                                    "Active object detector must not be starved")
            self.assertLess(frame_ms, 80.0, "Foreground frame processing exceeded 80 ms")
            self.assertLess(amortized_ms, 80.0, "Amortized CPU processing exceeded 80 ms")
            for monitor in (pipeline.fall, pipeline.study, pipeline.stove):
                self.assertIs(monitor.people, pipeline.people)


if __name__ == "__main__":
    unittest.main()
