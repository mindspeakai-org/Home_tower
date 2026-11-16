"""Desk/chair occupancy, phone distraction, and EMA ergonomic posture alerts."""

from dataclasses import dataclass
import time

import numpy as np
from shapely.geometry import Point, box

from .common import (
    AlertMonitor, DetectedObject, Person, RoomZones, angle_degrees,
    midpoint, object_box, pose_points, positive,
)


@dataclass
class StudyState:
    focused_seconds: float = 0.0
    phone_since: float | None = None
    slump_since: float | None = None
    smoothed_neck_angle: float | None = None
    last_timestamp: float | None = None
    last_pose_timestamp: float | None = None
    focused_last_step: bool = False
    mode: str = "AWAY"


class StudyFocusCoach(AlertMonitor):
    def __init__(self, desk_zone=None, *, zones: RoomZones | None = None,
                 phone_distraction_seconds: float = 15.0, slump_seconds: float = 20.0,
                 max_evidence_age: float = 1.0):
        super().__init__()
        self.zones = zones if zones is not None else RoomZones({"desk": desk_zone})
        self.phone_distraction_seconds = positive(phone_distraction_seconds, "phone_distraction_seconds")
        self.slump_seconds = positive(slump_seconds, "slump_seconds")
        self.max_evidence_age = positive(max_evidence_age, "max_evidence_age")
        self.objects: tuple[DetectedObject, ...] = ()
        self.tracks: dict[int, StudyState] = {}
        self.active = False

    @staticmethod
    def neck_angle(xy: np.ndarray, confidence: np.ndarray) -> float | None:
        sides = []
        for shoulder, ear, hip in ((5, 3, 11), (6, 4, 12)):
            if not np.isfinite(xy[[shoulder, ear, hip]]).all():
                continue
            angle = angle_degrees(xy[ear] - xy[shoulder], xy[shoulder] - xy[hip])
            if angle is not None:
                sides.append((float(min(confidence[[shoulder, ear]])), angle))
        return max(sides, key=lambda item: item[0])[1] if sides else None

    @staticmethod
    def phone_near(person: Person, xy: np.ndarray, objects: tuple) -> bool:
        hip = midpoint(xy, 11, 12)
        x1, y1, x2, y2 = person.bbox
        upper = box(x1, y1, x2, max(y1, min(y2, float(hip[1]))))
        for obj in objects:
            if obj.label != "cell phone":
                continue
            phone = object_box(obj)
            if phone is not None and (phone.intersects(upper) or any(
                np.isfinite(xy[wrist]).all() and phone.distance(Point(xy[wrist])) <= 50.0
                for wrist in (9, 10)
            )):
                return True
        return False

    def update(self, people: tuple[Person, ...], objects: tuple[DetectedObject, ...] = (),
               timestamp: float | None = None, *, pose_timestamp: float | None = None,
               objects_timestamp: float | None = None) -> list:
        timestamp = time.perf_counter() if timestamp is None else timestamp
        pose_timestamp = timestamp if pose_timestamp is None else pose_timestamp
        objects_timestamp = timestamp if objects_timestamp is None else objects_timestamp
        self.events = []
        self.people, self.objects = people, objects
        self.active = False
        ids = {p.track_id for p in people if p.track_id is not None}
        for identifier in self.tracks.keys() - ids:
            self.clear_track(identifier, timestamp, "track_lost")
            del self.tracks[identifier]
        for person in people:
            if person.track_id is None:
                continue
            state = self.tracks.setdefault(person.track_id, StudyState())
            if state.last_timestamp is not None and timestamp < state.last_timestamp:
                continue
            dt = 0.0 if state.last_timestamp is None else timestamp - state.last_timestamp
            if dt > self.max_evidence_age:
                state.phone_since = state.slump_since = state.smoothed_neck_angle = None
                state.focused_last_step = False
                dt = 0.0
            state.last_timestamp = timestamp
            xy, confidence = pose_points(person)
            seated = 0 <= timestamp - pose_timestamp <= self.max_evidence_age and self.zones.seated(xy)
            self.active |= seated
            if not seated:
                state.mode = "AWAY"
                state.phone_since = state.slump_since = state.smoothed_neck_angle = None
                state.focused_last_step = False
                self.clear_track(person.track_id, timestamp, "left_desk_or_pose_unavailable")
                continue
            object_fresh = 0 <= timestamp - objects_timestamp <= self.max_evidence_age
            phone = object_fresh and self.phone_near(person, xy, objects)
            focused = object_fresh and not phone
            if focused and state.focused_last_step:
                state.focused_seconds += dt
            state.focused_last_step = focused
            if phone:
                if state.phone_since is None:
                    state.phone_since = timestamp
            else:
                state.phone_since = None
            phone_seconds = 0.0 if state.phone_since is None else timestamp - state.phone_since
            distracted = phone and phone_seconds >= self.phone_distraction_seconds
            state.mode = ("PHONE_DISTRACTION" if distracted else "PHONE_PENDING") if phone else (
                "DEEP_FOCUS" if focused else "UNKNOWN"
            )
            self.set_alert("PHONE_DISTRACTION", person.track_id, distracted, timestamp,
                           {"phone_seconds": phone_seconds, "focused_seconds": state.focused_seconds})

            if pose_timestamp != state.last_pose_timestamp:
                angle = self.neck_angle(xy, confidence)
                state.last_pose_timestamp = pose_timestamp
                if angle is None:
                    state.smoothed_neck_angle = None
                else:
                    previous = state.smoothed_neck_angle
                    state.smoothed_neck_angle = angle if previous is None else 0.2 * angle + 0.8 * previous
            slumped = state.smoothed_neck_angle is not None and state.smoothed_neck_angle > 35.0
            if slumped:
                if state.slump_since is None:
                    state.slump_since = timestamp
            else:
                state.slump_since = None
            slump_time = 0.0 if state.slump_since is None else timestamp - state.slump_since
            self.set_alert("ERGONOMIC_SLUMP_WARNING", person.track_id,
                           slumped and slump_time >= self.slump_seconds, timestamp,
                           {"smoothed_neck_angle": state.smoothed_neck_angle, "slump_seconds": slump_time})
        return self.events
