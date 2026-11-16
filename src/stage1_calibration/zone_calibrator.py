"""CPU furniture proposals and explicitly approved camera calibration."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from shapely.geometry import Polygon


CALIBRATION_CLASSES = [56, 57, 59, 60, 62, 63, 69]
COCO_ZONE_TYPES = {56: "chair", 57: "couch", 59: "bed", 60: "dining_table", 69: "oven"}
ZONE_TYPES = ("bed", "couch", "chair", "dining_table", "desk", "stove", "kitchen", "floor", "oven")
MANUAL_TYPES = {"desk", "stove"}


def validate_room_zones(config_path: str | Path) -> dict:
    """Return validated JSON, with field-specific errors instead of silent zones.

    Accept named {type, polygon} entries and the original polygon-only format.
    No calibration file is overwritten or silently replaced with test geometry.
    """
    from src.stage2_monitors.common import parse_zone

    path = Path(config_path)
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read room zones from {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{path}: room configuration must be a JSON object")
    zones = config.get("zones", config)
    if not isinstance(zones, dict) or not zones:
        raise ValueError(f"{path}: zones must be a nonempty JSON object")
    if not isinstance(config.get("settings", {}), dict):
        raise ValueError(f"{path}: settings must be a JSON object")
    known_types = {"bed", "couch", "desk", "chair", "stove", "floor", "kitchen",
                   "study_desk", "kitchen_stove", "dining_table", "oven"}
    for name, entry in zones.items():
        kind, points = name, entry
        if isinstance(entry, dict):
            kind, points = entry.get("type"), entry.get("polygon")
        if not isinstance(kind, str) or kind not in known_types:
            raise ValueError(f"{path}: zones.{name}.type must name a supported room zone")
        if points is None:
            raise ValueError(f"{path}: zones.{name}.polygon is null; supply at least 3 vertices")
        try:
            parse_zone(points)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: zones.{name}.polygon: {exc}") from exc
    return config


def auto_detect_zones(frame: np.ndarray, detector=None, *, source: str = "unknown") -> dict:
    """Return editable proposals, never approved zones or fabricated desk/stove labels."""
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Calibration requires a nonempty BGR frame")
    if detector is None:
        from src.utils.export_cpu_models import load_cpu_model
        detector = load_cpu_model("yolo11n.pt")
    expected = {56: "chair", 57: "couch", 59: "bed", 60: "dining table", 69: "oven"}
    if any(detector.names.get(i) != label for i, label in expected.items()):
        raise ValueError("Stage 1 requires the stock COCO class mapping, not a furniture-trained head")
    results = detector.predict(frame, classes=CALIBRATION_CLASSES, device="cpu", imgsz=480,
                               rect=False, conf=0.25, iou=0.7, verbose=False)[0]
    height, width = frame.shape[:2]
    zones, context = {}, []
    for row in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, class_id = map(float, row)
        class_id = int(class_id)
        x1, x2 = np.clip([x1, x2], 0, width).tolist()
        y1, y2 = np.clip([y1, y2], 0, height).tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        observation = {"source_class_id": class_id, "source_label": detector.names[class_id],
                       "confidence": score, "bbox": [x1, y1, x2, y2]}
        if class_id not in COCO_ZONE_TYPES:
            context.append(observation)
            continue
        kind = COCO_ZONE_TYPES[class_id]
        index = sum(z["type"] == kind for z in zones.values())
        zones[f"{kind}_{index}"] = {
            "type": kind, "label": kind, "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "approved": False, "origin": "model_proposal", **observation,
        }
    return {"zones": zones, "context_objects": context, "settings": {},
            "calibration": {"status": "proposed", "source": source,
                            "source_frame": {"width": width, "height": height},
                            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                            "model": "stock COCO yolo11n CPU 480", "requires_manual_confirmation": True,
                            "unsupported_zone_types": sorted(MANUAL_TYPES),
                            "notes": ["Dining table remains dining_table, even in a study room.",
                                      "Oven proposals are not stove/cooktop polygons."]}}


def geometry_checks(config: dict, frame_shape: tuple) -> list[dict]:
    """Reject invalid/out-of-frame polygons; report positive-area overlaps."""
    from src.stage2_monitors.common import parse_zone

    height, width = frame_shape[:2]
    polygons = {}
    for name, zone in config.get("zones", {}).items():
        if not isinstance(zone, dict) or zone.get("type") not in ZONE_TYPES:
            raise ValueError(f"Zone {name} needs a supported type and named polygon")
        points = parse_zone(zone.get("polygon"))
        if points is None or not np.isfinite(points).all():
            raise ValueError(f"Zone {name} has no valid polygon")
        if (points < 0).any() or (points[:, 0] > width).any() or (points[:, 1] > height).any():
            raise ValueError(f"Zone {name} lies outside the {width}x{height} source frame")
        polygons[name] = Polygon(points)
    overlaps = []
    for i, (name, polygon) in enumerate(polygons.items()):
        for other, shape in list(polygons.items())[i + 1:]:
            area = polygon.intersection(shape).area
            if area > 0:
                overlaps.append({"zones": [name, other], "intersection_area_px": area})
    return overlaps


def validate_runtime_calibration(config: dict, frame_shape: tuple, *, allow_demo: bool = False) -> list[dict]:
    metadata = config.get("calibration", {})
    dimensions = metadata.get("source_frame", {})
    actual = {"width": frame_shape[1], "height": frame_shape[0]}
    if dimensions != actual:
        raise ValueError(f"Calibration resolution {dimensions} is incompatible with frame {actual}; recalibrate")
    overlaps = geometry_checks(config, frame_shape)
    if allow_demo and metadata.get("status") == "demo":
        return overlaps
    if metadata.get("status") != "approved" or not metadata.get("reviewer") or not metadata.get("approved_at"):
        raise ValueError("Uncalibrated camera: every zone needs manual confirmation; run --calibrate")
    if not config.get("zones") or any(z.get("approved") is not True for z in config["zones"].values()):
        raise ValueError("Unapproved zone proposals cannot be used for monitoring")
    if overlaps and metadata.get("overlap_reviewed") is not True:
        raise ValueError("Overlapping zones need explicit manual acknowledgment")
    return overlaps


def save_approved_zones(config: dict, path: Path, *, reviewer: str, overlap_reviewed: bool = False) -> dict:
    """Save operator-confirmed geometry atomically; never replace approved calibration."""
    if not reviewer.strip():
        raise ValueError("An operator/reviewer name is required")
    value = json.loads(json.dumps(config))
    dimensions = value.get("calibration", {}).get("source_frame", {})
    if any(type(dimensions.get(key)) is not int or dimensions[key] <= 0 for key in ("width", "height")):
        raise ValueError("Source frame width and height must be positive integers")
    overlaps = geometry_checks(value, (dimensions["height"], dimensions["width"]))
    if not value.get("zones") or any(z.get("approved") is not True for z in value["zones"].values()):
        raise ValueError("Explicit manual confirmation is required for every zone")
    if overlaps and not overlap_reviewed:
        raise ValueError("Review overlapping zones and explicitly acknowledge them before saving")
    metadata = value["calibration"]
    metadata.update(status="approved", reviewer=reviewer.strip(),
                    approved_at=datetime.now(timezone.utc).isoformat(), overlap_reviewed=overlap_reviewed,
                    overlaps=overlaps, requires_manual_confirmation=False,
                    uncalibrated_zone_types=sorted({"bed", "couch", "chair", "desk", "stove"}
                                                  - {z["type"] for z in value["zones"].values()}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    # Exclusive lock prevents concurrent approval writers racing the preservation check.
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temporary = None
    try:
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("calibration", {}).get("status") != "demo":
                raise FileExistsError("Existing calibration is preserved; choose a new output path")
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        os.close(descriptor)
        lock.unlink()
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return value


def draw_calibration(frame: np.ndarray, config: dict) -> np.ndarray:
    canvas = frame.copy()
    for name, zone in config.get("zones", {}).items():
        points = np.asarray(zone["polygon"], np.int32)
        if len(points) < 3:
            continue
        approved = zone.get("approved") is True
        color = (0, 220, 0) if approved else (0, 165, 255)
        cv2.polylines(canvas, [points], True, color, 2)
        x, y = points[0]
        text = f"{name}: {zone['type']} {'APPROVED' if approved else 'PROPOSAL'}"
        cv2.putText(canvas, text, (max(0, int(x)), max(14, int(y))), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(canvas, "Desk/stove require manual geometry; orange = unconfirmed", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    cv2.putText(canvas, "Dining table is not desk. Oven is not stove.", (5, 34),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return canvas


def edit_zones(frame: np.ndarray, proposals: dict) -> dict:
    """Local OpenCV editor: drag vertices, draw polygons, type/review each zone."""
    config = json.loads(json.dumps(proposals))
    selected = 0
    drawing = []
    dragged = None
    confirm = False
    warning = ""
    overlap_count = 0
    window = "Stage 1 calibration"

    def mouse(event, x, y, flags, _):
        nonlocal dragged
        keys = list(config["zones"])
        if not keys:
            return
        zone = config["zones"][keys[selected % len(keys)]]
        points = zone["polygon"]
        if event == cv2.EVENT_LBUTTONDOWN:
            if points:
                distances = [np.linalg.norm(np.array(point) - (x, y)) for point in points]
                if min(distances) < 12:
                    dragged = int(np.argmin(distances))
                    return
            drawing.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and dragged is not None:
            points[dragged] = [x, y]
            zone["approved"] = False
        elif event == cv2.EVENT_LBUTTONUP:
            dragged = None

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            canvas = draw_calibration(frame, config)
            if drawing:
                cv2.polylines(canvas, [np.array(drawing, np.int32)], False, (255, 255, 0), 2)
            keys = list(config["zones"])
            current = keys[selected % len(keys)] if keys else "none"
            help_text = (f"Y saves {len(keys)} zones and acknowledges {overlap_count} overlaps; other key cancels" if confirm else
                         f"{current}: TAB next T type N new C close E redraw A approve D delete S save Q quit")
            cv2.rectangle(canvas, (0, frame.shape[0] - 25), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
            cv2.putText(canvas, help_text, (4, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
            if warning:
                cv2.rectangle(canvas, (0, 42), (canvas.shape[1], 65), (0, 0, 120), -1)
                cv2.putText(canvas, warning[:95], (4, 58), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if confirm and key != 255:
                confirm = False
                if key == ord("y"):
                    config["calibration"]["overlap_reviewed"] = True
                    return config
            if key == ord("q"):
                raise RuntimeError("Calibration cancelled; existing configuration was preserved")
            if key == ord("n"):
                name = f"manual_{len(keys)}"
                while name in config["zones"]:
                    name += "_new"
                config["zones"][name] = {"type": "desk", "label": "manual zone", "polygon": [],
                                         "approved": False, "origin": "manual"}
                selected, drawing = len(keys), []
            elif keys:
                zone = config["zones"][current]
                if key == 9:
                    selected = (selected + 1) % len(keys)
                elif key == ord("t"):
                    zone["type"] = ZONE_TYPES[(ZONE_TYPES.index(zone["type"]) + 1) % len(ZONE_TYPES)]
                    zone["label"] = zone["type"]
                    zone["approved"] = False
                elif key == ord("e"):
                    zone["polygon"], drawing, zone["approved"] = [], [], False
                elif key == ord("c") and len(drawing) >= 3:
                    zone["polygon"], drawing, zone["approved"] = drawing, [], False
                elif key == ord("a"):
                    try:
                        geometry_checks({"zones": {current: zone}}, frame.shape)
                        zone["approved"], warning = True, ""
                    except ValueError as exc:
                        warning = str(exc)
                elif key == ord("d"):
                    del config["zones"][current]
                    selected = 0
                elif key == ord("s"):
                    try:
                        overlap_count = len(geometry_checks(config, frame.shape))
                        if config["zones"] and all(z.get("approved") for z in config["zones"].values()):
                            confirm = True
                        else:
                            warning = "Confirm every zone with A first"
                    except ValueError as exc:
                        warning = str(exc)
    finally:
        cv2.destroyWindow(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve", type=Path, help="Explicitly approve operator-edited JSON, not model output")
    parser.add_argument("--reviewer")
    parser.add_argument("--acknowledge-overlaps", action="store_true")
    parser.add_argument("--edit", action="store_true")
    args = parser.parse_args()
    if args.approve:
        if not args.reviewer:
            parser.error("--approve requires --reviewer")
        save_approved_zones(json.loads(args.approve.read_text()), args.output, reviewer=args.reviewer,
                            overlap_reviewed=args.acknowledge_overlaps)
        return
    frame = cv2.imread(str(args.image)) if args.image else None
    if frame is None:
        parser.error("--image must name a readable image")
    config = auto_detect_zones(frame, source=str(args.image))
    if args.edit:
        if not args.reviewer:
            parser.error("--edit requires --reviewer")
        config = edit_zones(frame, config)
        save_approved_zones(config, args.output, reviewer=args.reviewer,
                            overlap_reviewed=args.acknowledge_overlaps or config["calibration"].get("overlap_reviewed", False))
    else:
        if args.output.exists():
            parser.error("Proposal output already exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config, indent=2))
        cv2.imwrite(str(args.output.with_suffix(".png")), draw_calibration(frame, config))


if __name__ == "__main__":
    main()
