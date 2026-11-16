"""Per-track floor-fall and prolonged resting-inactivity state machines."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as WallTime
import time
from typing import NamedTuple

import numpy as np

from .common import (
    AlertMonitor, Person, RoomZones, angle_degrees, load_room_config,
    midpoint, pose_points, position_std, positive,
)


class PoseStep(NamedTuple):
    timestamp: float
    bbox_xyxy: tuple
    keypoints_xy: np.ndarray
    zone_name: str


@dataclass
class FallState:
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    still_since: float | None = None
    inactive_since: float | None = None
    torso_angle: float | None = None
    zone_name: str = "floor"


class FallInactivityMonitor(AlertMonitor):
    def __init__(self, zones: RoomZones | None = None, max_inactive_seconds: float = 7200,
                 sleep_hours: tuple | list = ("22:00", "07:00"),
                 stillness_seconds: float = 2.5, max_pose_gap: float = 1.0):
        super().__init__()
        self.zones = zones if zones is not None else load_room_config()[0]
        self.max_inactive_seconds = positive(max_inactive_seconds, "max_inactive_seconds")
        self.stillness_seconds = positive(stillness_seconds, "stillness_seconds")
        self.max_pose_gap = positive(max_pose_gap, "max_pose_gap")
        if len(sleep_hours) != 2:
            raise ValueError("sleep_hours requires a start and end time")
        self.sleep_hours = tuple(WallTime.fromisoformat(value) for value in sleep_hours)
        self.tracks: dict[int, FallState] = {}

    def sleeping_hours(self, wall_time: datetime) -> bool:
        start, end = self.sleep_hours
        current = wall_time.time().replace(tzinfo=None)
        return start <= current < end if start < end else current >= start or current < end

    def update(self, people: tuple[Person, ...], timestamp: float | None = None,
               wall_time: datetime | None = None) -> list:
        timestamp = time.perf_counter() if timestamp is None else timestamp
        wall_time = datetime.now().astimezone() if wall_time is None else wall_time
        self.events = []
        self.people = people
        current_ids = {p.track_id for p in people if p.track_id is not None}
        for identifier in self.tracks.keys() - current_ids:
            self.clear_track(identifier, timestamp, "track_lost")
            del self.tracks[identifier]
        for person in people:
            if person.track_id is None:
                continue
            state = self.tracks.setdefault(person.track_id, FallState())
            if state.history and timestamp <= state.history[-1].timestamp:
                continue
            if state.history and timestamp - state.history[-1].timestamp > self.max_pose_gap:
                self.clear_track(person.track_id, timestamp, "pose_gap")
                state = self.tracks[person.track_id] = FallState()
            xy, _ = pose_points(person)
            hip, shoulder = midpoint(xy, 11, 12), midpoint(xy, 5, 6)
            valid = hip is not None and shoulder is not None and np.isfinite(xy[0]).all()
            if not valid:
                self.clear_track(person.track_id, timestamp, "pose_unavailable")
                state.history.clear()
                state.still_since = state.inactive_since = state.torso_angle = None
                continue
            zone = self.zones.resting_zone(xy)
            previous = state.history[-1] if state.history else None
            state.history.append(PoseStep(timestamp, person.bbox, xy.copy(), zone))
            state.zone_name = zone
            state.torso_angle = angle_degrees(shoulder - hip, np.array([0.0, -1.0]))
            x1, y1, x2, y2 = person.bbox
            ratio = (x2 - x1) / (y2 - y1) if y2 > y1 else 0.0
            lying_on_floor = zone == "floor" and ratio > 1.25 and (
                state.torso_angle is not None and state.torso_angle > 60.0
            )
            velocities = [
                (b.keypoints_xy[0, 1] - a.keypoints_xy[0, 1]) / (b.timestamp - a.timestamp)
                for a, b in zip(state.history, list(state.history)[1:])
            ]
            peak = max(velocities, default=0.0)
            if velocities and velocities[-1] > 120.0:
                state.still_since = timestamp
            elif state.still_since is not None:
                samples = [s.keypoints_xy for s in state.history if s.timestamp >= state.still_since]
                if position_std(samples) >= 4.0:
                    state.still_since = timestamp
            still_seconds = 0.0 if state.still_since is None else timestamp - state.still_since
            already_fallen = ("FALL_ALERT", person.track_id) in self.active_alerts
            fallen = lying_on_floor and (already_fallen or (
                peak > 120.0 and state.still_since is not None
                and still_seconds >= self.stillness_seconds
            ))
            self.set_alert("FALL_ALERT", person.track_id, fallen, timestamp,
                           {"zone": zone, "torso_angle": state.torso_angle,
                            "aspect_ratio": ratio, "peak_head_velocity": float(peak),
                            "still_seconds": still_seconds})
            if not lying_on_floor:
                state.still_since = None

            resting = zone in {"bed", "couch"} and not self.sleeping_hours(wall_time)
            deviation = position_std([s.keypoints_xy for s in state.history])
            if resting and deviation < 3.0:
                if state.inactive_since is None or (previous and previous.zone_name != zone):
                    state.inactive_since = timestamp
            else:
                state.inactive_since = None
            inactive = 0.0 if state.inactive_since is None else timestamp - state.inactive_since
            self.set_alert("INACTIVITY_ALERT", person.track_id,
                           resting and inactive >= self.max_inactive_seconds, timestamp,
                           {"zone": zone, "inactive_seconds": inactive, "keypoint_std": deviation})
        return self.events
