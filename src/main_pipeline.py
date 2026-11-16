"""CPU-only live capture with bounded, strided background inference."""

import argparse
import json
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from src.stage2_monitors.monitors import (
    DetectedObject, FallInactivityMonitor, KitchenStoveGuard, Person,
    StudyFocusCoach, RoomZones,
)
from src.stage2_monitors.common import ROOM_CONFIG, load_room_config, pose_points
from src.utils.event_logger import EVENT_LOG, EventLogger
from src.utils.export_cpu_models import IMAGE_SIZE, load_cpu_model


@dataclass(frozen=True)
class InferenceResult:
    captured_at: float
    people: tuple[Person, ...] | None = None
    objects: tuple[DetectedObject, ...] | None = None
    pose_ms: float | None = None
    object_ms: float | None = None


class CPUModels:
    """Each model has one inference slot; pose tracking is never concurrent."""

    def __init__(self) -> None:
        self.pose = load_cpu_model("yolo11n-pose.pt")
        self.objects = load_cpu_model("yolo11n.pt")
        target_names = {"cell phone", "laptop", "bowl", "cup"}
        self.object_classes = [
            i for i, name in self.objects.names.items() if name in target_names
        ]
        if len(self.object_classes) != len(target_names):
            raise ValueError("Object model is missing required COCO classes")
        # Initialize tracker imports/state before opening the video stream.
        self.pose.track(
            np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            persist=True, tracker="bytetrack.yaml", classes=[0],
            device="cpu", imgsz=IMAGE_SIZE, rect=False, verbose=False,
        )
        for tracker in self.pose.predictor.trackers:
            tracker.reset()  # Warmup must not advance the live stream's tracking state.

    def infer(self, frame: np.ndarray, captured_at: float,
              run_pose: bool, run_objects: bool) -> InferenceResult:
        people = objects = None
        pose_ms = object_ms = None
        kwargs = dict(device="cpu", imgsz=IMAGE_SIZE, rect=False, verbose=False)
        if run_pose:
            started = time.perf_counter()
            result = self.pose.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=[0], **kwargs,
            )[0]
            boxes = result.boxes
            ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            people = tuple(
                Person(tuple(map(float, box)), int(track_id) if track_id is not None else None,
                       keypoints[i] if keypoints is not None else None)
                for i, (box, track_id) in enumerate(zip(boxes.xyxy.cpu().tolist(), ids))
            )
            pose_ms = (time.perf_counter() - started) * 1000
        if run_objects:
            started = time.perf_counter()
            result = self.objects.predict(frame, classes=self.object_classes, **kwargs)[0]
            objects = tuple(
                DetectedObject(result.names[int(cls)], tuple(map(float, box)), float(conf))
                for box, cls, conf in zip(
                    result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                )
            )
            object_ms = (time.perf_counter() - started) * 1000
        return InferenceResult(captured_at, people, objects, pose_ms, object_ms)


class FrameScheduler:
    """Keep native-rate frames flowing; skip busy inference slots, never queue.

    Cached detections retain their capture timestamp and expire after one second.
    Only drain()/close() wait for the worker; process_frame() never waits for it.
    """

    def __init__(self, models: CPUModels, pose_stride: int = 3, obj_stride: int = 15,
                 desk_zone: np.ndarray | None = None, stove_zone: np.ndarray | None = None,
                 stove_active: bool = False, max_result_age: float = 1.0,
                 room_zones: RoomZones | None = None, monitor_settings: dict | None = None,
                 event_log: Path = EVENT_LOG) -> None:
        if pose_stride < 1 or obj_stride < 1 or max_result_age <= 0:
            raise ValueError("Strides and max_result_age must be positive")
        self.models = models
        self.pose_stride = pose_stride
        self.obj_stride = obj_stride
        self.max_result_age = max_result_age
        zones = room_zones if room_zones is not None else RoomZones({"desk": desk_zone, "stove": stove_zone})
        self.zones = zones
        settings = monitor_settings or {}
        self.fall = FallInactivityMonitor(
            zones, max_inactive_seconds=settings.get("max_inactive_seconds", 7200.0),
            sleep_hours=settings.get("sleep_hours", ["22:00", "07:00"]),
            max_pose_gap=max_result_age,
        )
        self.study = StudyFocusCoach(
            zones=zones, phone_distraction_seconds=settings.get("phone_distraction_seconds", 15.0),
            slump_seconds=settings.get("slump_seconds", 20.0), max_evidence_age=max_result_age,
        )
        self.stove = KitchenStoveGuard(
            zones=zones, active=stove_active,
            max_unattended_seconds=settings.get("max_unattended_seconds", 300.0),
            max_evidence_age=max_result_age,
        )
        self.logger = EventLogger(event_log)
        self._last_fall_pose_at = float("-inf")
        self.people: tuple[Person, ...] = ()
        self.objects: tuple[DetectedObject, ...] = ()
        self.pose_at = self.objects_at = float("-inf")
        self.frame_index = 0
        self._last_pose_frame = -pose_stride
        self._last_object_frame = -obj_stride
        self.pose_runs = self.object_runs = 0
        self.pose_total_ms = self.object_total_ms = 0.0
        self.frame_ms: deque[float] = deque(maxlen=120)
        self.pose_ms: deque[float] = deque(maxlen=120)
        self.object_ms: deque[float] = deque(maxlen=120)
        self.frame_times: deque[float] = deque(maxlen=120)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu-inference")
        self._pending: dict[str, Future] = {}
        self._last_frame_timestamp: float | None = None

    def _collect(self, kind: str) -> None:
        pending = self._pending.pop(kind)
        result = pending.result()  # Called only when done, or explicitly draining.
        if result.people is not None:
            self.people, self.pose_at = result.people, result.captured_at
            self.pose_runs += 1
            self.pose_ms.append(result.pose_ms)
            self.pose_total_ms += result.pose_ms
        if result.objects is not None:
            self.objects, self.objects_at = result.objects, result.captured_at
            self.object_runs += 1
            self.object_ms.append(result.object_ms)
            self.object_total_ms += result.object_ms

    def _update_monitors(self, now: float, frame: np.ndarray | None = None) -> None:
        if now - self.pose_at > self.max_result_age:
            self.people = ()
        if now - self.objects_at > self.max_result_age:
            self.objects = ()
        events = []
        if now - self.pose_at <= self.max_result_age:
            if self.pose_at != self._last_fall_pose_at:
                events.extend(self.fall.update(
                    self.people, self.pose_at, self.logger.wall_time(self.pose_at).astimezone(),
                ))
                self._last_fall_pose_at = self.pose_at
        elif self.fall.tracks:
            events.extend(self.fall.update((), now))
        self.fall.people = self.people
        events.extend(self.study.update(
            self.people, self.objects, now,
            pose_timestamp=self.pose_at, objects_timestamp=self.objects_at,
        ))
        events.extend(self.stove.update(
            self.people, self.objects, now, frame=frame,
            pose_timestamp=self.pose_at, objects_timestamp=self.objects_at,
        ))
        self.logger.log(events, timestamp=now)

    @property
    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        return (len(self.frame_times) - 1) / max(self.frame_times[-1] - self.frame_times[0], 1e-9)

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        started = time.perf_counter()
        observed_at = started if timestamp is None else timestamp
        self._last_frame_timestamp = observed_at
        self.frame_times.append(started)
        for kind, pending in tuple(self._pending.items()):
            if pending.done():
                self._collect(kind)  # Propagate failures instead of freezing detections.
        self._update_monitors(observed_at, frame)
        run_pose = self.frame_index - self._last_pose_frame >= self.pose_stride
        run_objects = self.frame_index - self._last_object_frame >= self.obj_stride and (
            self.study.active or self.stove.guard_active
        )
        for kind, due in (("pose", run_pose), ("objects", run_objects)):
            if due and kind not in self._pending:
                self._pending[kind] = self._executor.submit(
                    self.models.infer, frame.copy(), observed_at,
                    kind == "pose", kind == "objects",
                )
                if kind == "pose":
                    self._last_pose_frame = self.frame_index
                else:
                    self._last_object_frame = self.frame_index
        self.frame_index += 1
        self._draw_hud(frame)
        self.frame_ms.append((time.perf_counter() - started) * 1000)
        return frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        for name, polygon in self.zones.outlines.items():
            coordinates = np.asarray(polygon.exterior.coords, np.int32)
            cv2.polylines(frame, [coordinates], True, (220, 160, 30), 1)
            x, y = coordinates[0]
            cv2.putText(frame, name, (int(x) + 3, max(12, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 160, 30), 1)
        bones = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                 (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                 (11, 13), (13, 15), (12, 14), (14, 16))
        for person in self.people:
            x1, y1, x2, y2 = map(int, person.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 1)
            cv2.putText(frame, f"ID {person.track_id}", (x1, max(16, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
            xy, _ = pose_points(person)
            for a, b in bones:
                if np.isfinite(xy[[a, b]]).all():
                    cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (255, 210, 0), 1)
            for point in xy[np.isfinite(xy).all(axis=1)]:
                cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 255), -1)

        def average(values: deque) -> str:
            return f"{sum(values) / len(values):.1f}" if values else "--"

        lines = [
            f"CPU FPS: {self.fps:.1f} | Frame: {average(self.frame_ms)} ms",
            f"Pose: {average(self.pose_ms)} ms | Objects: {average(self.object_ms)} ms",
            f"People: {len(self.people)} | Desk: {self.study.active} | "
            f"Stove: {self.stove.active} ({self.stove.person_count})",
            f"Unattended Stove Timer: {self.stove.unattended_seconds:.1f} s",
        ]
        for identifier in sorted(self.fall.tracks.keys() | self.study.tracks.keys()):
            fall = self.fall.tracks.get(identifier)
            study = self.study.tracks.get(identifier)
            torso = "--" if fall is None or fall.torso_angle is None else f"{fall.torso_angle:.1f}"
            neck = "--" if study is None or study.smoothed_neck_angle is None else f"{study.smoothed_neck_angle:.1f}"
            focus = 0.0 if study is None else study.focused_seconds
            lines.append(f"ID {identifier} Torso Angle: {torso} | Neck EMA: {neck} | Focus Timer: {focus:.1f}s")
        alerts = [alert for monitor in (self.fall, self.study, self.stove)
                  for alert in monitor.active_alerts.values()]
        panel_width = min(frame.shape[1] - 1, max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in lines
        ) + 20)
        panel_height = 24 * (len(lines) + len(alerts)) + 5
        cv2.rectangle(frame, (0, 0), (panel_width, panel_height), (20, 20, 20), -1)
        for i, line in enumerate(lines):
            # Reserve banner rows immediately after FPS/inference counters.
            position = (10, 24 + i * 24 + (len(alerts) * 24 if i >= 2 else 0))
            cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, alert in enumerate(alerts):
            y = 52 + i * 24
            cv2.rectangle(frame, (0, y), (frame.shape[1] - 1, y + 22), (0, 0, 200), -1)
            label = alert.event + (f" | ID {alert.track_id}" if alert.track_id is not None else "")
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0][0]
            scale = min(0.55, (frame.shape[1] - 20) / max(text_width, 1))
            cv2.putText(frame, label, (10, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 1)

    def drain(self) -> None:
        """Wait for the last jobs at end-of-stream, outside the live frame loop."""
        for kind in tuple(self._pending):
            self._collect(kind)
        now = time.perf_counter() if self._last_frame_timestamp is None else self._last_frame_timestamp
        self._update_monitors(now)

    def close(self) -> None:
        try:
            self.drain()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(argv: list[str] | None = None) -> None:
    from src.stage1_calibration.zone_calibrator import (
        auto_detect_zones, draw_calibration, edit_zones, save_approved_zones,
        validate_room_zones, validate_runtime_calibration,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--zones", type=Path, default=ROOM_CONFIG, help="Room polygons and alert settings JSON")
    parser.add_argument("--event-log", type=Path, default=EVENT_LOG, help="Append-only alert JSONL log")
    parser.add_argument("--stove-active", action="store_true", help="Manual stove activity override; toggle with s")
    parser.add_argument("--save-video", type=Path, help="Write every rendered frame to an MP4")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV window functions")
    parser.add_argument("--fast-test-timers", action="store_true", help="Use 2-second test alert thresholds")
    parser.add_argument("--calibrate", action="store_true", help="Review furniture/manual polygons once at startup")
    parser.add_argument("--calibration-reviewer", help="Operator who explicitly confirms every saved zone")
    parser.add_argument("--acknowledge-zone-overlaps", action="store_true")
    parser.add_argument("--allow-demo-zones", action="store_true", help="Explicit synthetic/file test geometry; forbidden for cameras")
    args = parser.parse_args(argv)
    is_camera = (args.source.isdecimal() or Path(args.source).is_char_device()
                 or args.source.startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")))
    if args.calibrate and (args.headless or not args.calibration_reviewer):
        parser.error("Interactive --calibrate needs a display and --calibration-reviewer; export proposals separately in headless mode")
    if is_camera and args.allow_demo_zones:
        parser.error("--allow-demo-zones is forbidden for live cameras")
    if args.zones.exists():
        zone_config = validate_room_zones(args.zones)
        zones, settings = load_room_config(args.zones)
    elif args.calibrate:
        zone_config, zones, settings = {}, RoomZones(), {}
    else:
        parser.error("Calibration file is missing; run Stage 1 --calibrate or approve reviewed polygons")
    if args.fast_test_timers:
        settings = {**settings, **dict.fromkeys(("max_inactive_seconds", "phone_distraction_seconds",
                                                "slump_seconds", "max_unattended_seconds"), 2.0)}
    if args.save_video is not None and not is_camera and args.save_video.resolve() == Path(args.source).resolve():
        parser.error("--save-video must differ from the source video")
    models = CPUModels()  # Model compilation and warmup happen before capture.
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    writer = None
    output_size = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {args.source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        ok, first_frame = capture.read()
        if not ok:
            raise RuntimeError("Video source contains no readable representative frame")
        if is_camera or args.calibrate:
            proposals = auto_detect_zones(first_frame, models.objects, source=str(args.source))
            proposals["settings"] = settings
            preview_dir = args.event_log.parent
            preview_dir.mkdir(parents=True, exist_ok=True)
            (preview_dir / "startup_zone_proposals.json").write_text(json.dumps(proposals, indent=2))
            cv2.imwrite(str(preview_dir / "startup_zone_proposals.jpg"), draw_calibration(first_frame, proposals))
            if args.calibrate:
                if zone_config.get("calibration", {}).get("status") == "approved":
                    raise FileExistsError("Existing approved zones are preserved; select a new --zones output path")
                edited = edit_zones(first_frame, proposals)
                zone_config = save_approved_zones(edited, args.zones, reviewer=args.calibration_reviewer,
                                                  overlap_reviewed=args.acknowledge_zone_overlaps or
                                                  edited["calibration"].get("overlap_reviewed", False))
                zones = RoomZones(zone_config["zones"])
        validate_runtime_calibration(zone_config, first_frame.shape, allow_demo=args.allow_demo_zones)
        missing = {"bed", "couch", "chair", "desk", "stove"} - set(zones.polygons)
        if missing:
            print("Uncalibrated/absent zone types:", ", ".join(sorted(missing)))
        next_frame_at = time.perf_counter()
        stream_origin = next_frame_at
        frame_number = 0
        with FrameScheduler(models, room_zones=zones, monitor_settings=settings,
                            stove_active=args.stove_active, event_log=args.event_log) as scheduler:
            while True:
                if frame_number == 0:
                    frame = first_frame
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                timestamp = None if is_camera else stream_origin + frame_number / fps
                rendered = scheduler.process_frame(frame, timestamp=timestamp)
                if args.save_video is not None:
                    size = (rendered.shape[1], rendered.shape[0])
                    if writer is None:
                        args.save_video.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot open video output: {args.save_video}")
                        output_size = size
                    if size != output_size:
                        raise RuntimeError("Video frame dimensions changed during export")
                    writer.write(rendered)
                frame_number += 1
                key = -1
                if not args.headless:
                    cv2.imshow("Smart Home CPU", rendered)
                    key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    scheduler.stove.manual_active = not scheduler.stove.manual_active
                next_frame_at += 1 / fps
                now = time.perf_counter()
                if next_frame_at < now - 1 / fps:
                    next_frame_at = now
                time.sleep(max(0.0, next_frame_at - now))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
