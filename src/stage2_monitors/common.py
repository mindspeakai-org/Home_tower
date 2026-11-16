"""Shared observations, zone geometry, and edge-triggered alert transitions."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOM_CONFIG = PROJECT_ROOT / "configs" / "room_zones.json"


@dataclass(frozen=True)
class Person:
    bbox: tuple[float, float, float, float]
    track_id: int | None
    keypoints: np.ndarray | None = None


@dataclass(frozen=True)
class DetectedObject:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class AlertEvent:
    timestamp: float  # Monotonic observation/evaluation time.
    module: str
    event: str
    track_id: int | None
    metrics: dict


def positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def parse_zone(points) -> np.ndarray | None:
    if points is None:
        return None
    if isinstance(points, Polygon):
        points = list(points.exterior.coords)
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("A zone requires at least three [x, y] points")
    shape = Polygon(polygon)
    if not np.isfinite(polygon).all() or not shape.is_valid or shape.area <= 0:
        raise ValueError("Zone coordinates must be finite and form a valid polygon")
    return polygon


class RoomZones:
    def __init__(self, zones: dict | None = None):
        self.outlines: dict[str, Polygon] = {}
        self.zone_types: dict[str, str] = {}
        self.polygons: dict[str, Polygon] = {}
        grouped = {}
        for name, entry in (zones or {}).items():
            kind, points = (entry["type"], entry["polygon"]) if isinstance(entry, dict) else (name, entry)
            coordinates = parse_zone(points)
            if coordinates is not None:
                polygon = Polygon(coordinates)
                self.outlines[name] = polygon
                self.zone_types[name] = kind
                grouped.setdefault(kind, []).append(polygon)
        self.polygons = {kind: unary_union(polygons) for kind, polygons in grouped.items()}

    def get(self, name: str) -> Polygon | None:
        return self.polygons.get(name)

    def contains(self, name: str, point: np.ndarray) -> bool:
        polygon = self.get(name)
        return polygon is not None and np.isfinite(point).all() and polygon.covers(Point(point))

    def seated(self, xy: np.ndarray) -> bool:
        hip, shoulder = midpoint(xy, 11, 12), midpoint(xy, 5, 6)
        for name in ("desk", "chair"):
            polygon = self.get(name)
            if polygon is not None and hip is not None:
                if polygon.covers(Point(hip)):
                    return True
                if shoulder is not None and polygon.intersects(LineString([hip, shoulder])):
                    return True
        return False

    def resting_zone(self, xy: np.ndarray) -> str:
        hip = midpoint(xy, 11, 12)
        points = ([hip] if hip is not None else []) + [xy[15], xy[16]]
        for name in ("bed", "couch"):
            if any(self.contains(name, point) for point in points):
                return name
        return "floor"


def load_room_config(path: Path = ROOM_CONFIG) -> tuple[RoomZones, dict]:
    from src.stage1_calibration.zone_calibrator import validate_room_zones

    config = validate_room_zones(path)
    if "zones" in config:
        zones, settings = config["zones"], config.get("settings", {})
    else:  # Earlier polygon-only configuration format.
        zones, settings = dict(config), {}
        for old, new in (("study_desk", "desk"), ("kitchen_stove", "stove")):
            if old in zones:
                zones[new] = zones.pop(old)
    if not isinstance(zones, dict) or not isinstance(settings, dict):
        raise ValueError("zones and settings must be JSON objects")
    return RoomZones(zones), settings


def pose_points(person: Person) -> tuple[np.ndarray, np.ndarray]:
    """Unreliable/missing points become NaN, never motionless zero observations."""
    xy, confidence = np.full((17, 2), np.nan), np.zeros(17)
    if person.keypoints is None:
        return xy, confidence
    data = np.asarray(person.keypoints, dtype=float)
    if data.shape not in ((17, 2), (17, 3)):
        return xy, confidence
    confidence = data[:, 2].copy() if data.shape[1] == 3 else np.ones(17)
    valid = (np.isfinite(data[:, :2]).all(axis=1) & np.isfinite(confidence)
             & (confidence >= 0.3) & np.any(data[:, :2] != 0, axis=1))
    xy[valid] = data[valid, :2]
    confidence[~valid] = 0
    return xy, confidence


def midpoint(xy: np.ndarray, a: int, b: int) -> np.ndarray | None:
    return (xy[a] + xy[b]) / 2 if np.isfinite(xy[[a, b]]).all() else None


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float | None:
    length = float(np.linalg.norm(a) * np.linalg.norm(b))
    if length < 1e-6:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / length, -1, 1))))


def position_std(samples: list[np.ndarray]) -> float:
    """Largest temporal coordinate std of points visible in every sample."""
    values = np.stack(samples)
    common = np.isfinite(values).all(axis=(0, 2))
    return float(values[:, common].std(axis=0).max()) if common.any() else math.inf


def object_box(obj: DetectedObject):
    x1, y1, x2, y2 = obj.bbox
    if (not np.isfinite(obj.bbox).all() or x2 <= x1 or y2 <= y1
            or not math.isfinite(obj.confidence) or obj.confidence < 0.25):
        return None
    return box(x1, y1, x2, y2)


class AlertMonitor:
    """Emit events on edges; refresh active metrics without duplicate logs."""

    def __init__(self):
        self.people: tuple[Person, ...] = ()
        self.events: list[AlertEvent] = []
        self.active_alerts: dict[tuple[str, int | None], AlertEvent] = {}

    def set_alert(self, event: str, track_id: int | None, active: bool,
                  timestamp: float, metrics: dict, reason: str = "condition_cleared") -> None:
        key = (event, track_id)
        previous = key in self.active_alerts
        if active:
            observation = AlertEvent(timestamp, type(self).__name__, event, track_id,
                                     {**metrics, "state": "ALERT"})
            self.active_alerts[key] = observation
            if not previous:
                self.events.append(AlertEvent(timestamp, observation.module, event, track_id,
                                              {**observation.metrics, "transition": "NORMAL -> ALERT"}))
        elif previous:
            self.active_alerts.pop(key)
            self.events.append(AlertEvent(
                timestamp, type(self).__name__, event + "_RESOLVED", track_id,
                {**metrics, "state": "RESOLVED", "transition": "ALERT -> RESOLVED", "reason": reason},
            ))

    def clear_track(self, track_id: int, timestamp: float, reason: str) -> None:
        for event, identifier in tuple(self.active_alerts):
            if identifier == track_id:
                self.set_alert(event, track_id, False, timestamp, {}, reason)
