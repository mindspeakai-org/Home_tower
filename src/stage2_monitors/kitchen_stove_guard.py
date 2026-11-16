"""Cookware/HSV stove activity and continuous unattended-cooking alerts."""

import time

import cv2
import numpy as np
from shapely.geometry import Point

from .common import AlertMonitor, RoomZones, object_box, positive


class KitchenStoveGuard(AlertMonitor):
    def __init__(self, stove_zone=None, active: bool = False, *, zones: RoomZones | None = None,
                 max_unattended_seconds: float = 300.0, max_evidence_age: float = 1.0):
        super().__init__()
        self.zones = zones if zones is not None else RoomZones({"stove": stove_zone})
        self.max_unattended_seconds = positive(max_unattended_seconds, "max_unattended_seconds")
        self.max_evidence_age = positive(max_evidence_age, "max_evidence_age")
        self.manual_active = self.active = active
        self.person_count = self.kitchen_person_count = 0
        self.unattended_seconds = 0.0
        self.indicator_fraction = self.cookware_ioa = 0.0
        self.objects = ()
        self._last_timestamp = None
        self._unattended_last_step = False
        self._indicator_at = float("-inf")

    @property
    def guard_active(self) -> bool:
        # Arming a configured guard allows cookware detection to bootstrap.
        return self.manual_active or self.zones.get("stove") is not None

    def indicator_pixels(self, frame: np.ndarray | None) -> float:
        polygon = self.zones.get("stove")
        if frame is None or polygon is None:
            return 0.0
        height, width = frame.shape[:2]
        minx, miny, maxx, maxy = polygon.bounds
        x1, y1 = max(0, int(np.floor(minx))), max(0, int(np.floor(miny)))
        x2, y2 = min(width, int(np.ceil(maxx)) + 1), min(height, int(np.ceil(maxy)) + 1)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame[y1:y2, x1:x2]
        mask = np.zeros(roi.shape[:2], np.uint8)
        contours = []
        for part in getattr(polygon, "geoms", (polygon,)):
            for ring in (part.exterior, *part.interiors):
                coordinates = np.asarray(ring.coords) - (x1, y1)
                contours.append(coordinates.astype(np.int32))
        cv2.fillPoly(mask, contours, 255)
        area = cv2.countNonZero(mask)
        if area == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        indicators = cv2.inRange(hsv, (0, 151, 221), (179, 255, 255))
        return cv2.countNonZero(cv2.bitwise_and(indicators, mask)) / area

    def update(self, people, objects=(), timestamp: float | None = None, *,
               frame: np.ndarray | None = None, pose_timestamp: float | None = None,
               objects_timestamp: float | None = None) -> list:
        timestamp = time.perf_counter() if timestamp is None else timestamp
        pose_timestamp = timestamp if pose_timestamp is None else pose_timestamp
        objects_timestamp = timestamp if objects_timestamp is None else objects_timestamp
        self.events = []
        self.people, self.objects = people, objects
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            return self.events
        dt = 0.0 if self._last_timestamp is None else timestamp - self._last_timestamp
        self._last_timestamp = timestamp
        pose_fresh = 0 <= timestamp - pose_timestamp <= self.max_evidence_age
        if dt > self.max_evidence_age:
            self._unattended_last_step = False
            self.unattended_seconds = 0.0
            dt = 0.0
        kitchen = self.zones.get("kitchen")
        self.kitchen_person_count = sum(
            kitchen is None or kitchen.covers(Point((p.bbox[0] + p.bbox[2]) / 2, p.bbox[3]))
            for p in people
        ) if pose_fresh else 0
        self.person_count = self.kitchen_person_count
        stove = self.zones.get("stove")
        self.cookware_ioa = 0.0
        if stove is not None and 0 <= timestamp - objects_timestamp <= self.max_evidence_age:
            for obj in objects:
                if obj.label not in {"bowl", "cup", "cookware", "pot", "pan", "kettle"}:
                    continue
                shape = object_box(obj)
                if shape is not None:
                    self.cookware_ioa = max(self.cookware_ioa, stove.intersection(shape).area / shape.area)
        if frame is not None:
            self.indicator_fraction = self.indicator_pixels(frame)
            self._indicator_at = timestamp
        elif timestamp - self._indicator_at > self.max_evidence_age:
            self.indicator_fraction = 0.0
        self.active = self.manual_active or self.cookware_ioa > 0.3 or self.indicator_fraction > 0.015
        unattended = self.active and pose_fresh and self.kitchen_person_count == 0
        if unattended:
            if self._unattended_last_step:
                self.unattended_seconds += dt
        else:
            self.unattended_seconds = 0.0
        self._unattended_last_step = unattended
        self.set_alert(
            "UNATTENDED_COOKING_CRITICAL_ALERT", None,
            unattended and self.unattended_seconds >= self.max_unattended_seconds,
            timestamp, {"unattended_seconds": self.unattended_seconds,
                        "kitchen_person_count": self.kitchen_person_count,
                        "stove_active": self.active, "indicator_fraction": self.indicator_fraction,
                        "cookware_ioa": self.cookware_ioa, "pose_fresh": pose_fresh},
            "person_returned" if self.kitchen_person_count else "stove_off_or_evidence_unavailable",
        )
        return self.events
