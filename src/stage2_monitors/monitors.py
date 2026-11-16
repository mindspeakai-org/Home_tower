"""Compatibility imports for Stage 2 monitors and shared observations."""

from .common import DetectedObject, Person, RoomZones, parse_zone
from .fall_inactivity_monitor import FallInactivityMonitor
from .study_focus_coach import StudyFocusCoach
from .kitchen_stove_guard import KitchenStoveGuard

__all__ = [
    "DetectedObject", "Person", "RoomZones", "parse_zone",
    "FallInactivityMonitor", "StudyFocusCoach", "KitchenStoveGuard",
]
