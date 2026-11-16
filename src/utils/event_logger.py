"""Append-only JSONL logging for alert and resolution transitions."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

from src.stage2_monitors.common import AlertEvent, PROJECT_ROOT


EVENT_LOG = PROJECT_ROOT / "logs" / "home_events.jsonl"


class EventLogger:
    def __init__(self, path: Path = EVENT_LOG, *, monotonic_origin: float | None = None,
                 wall_origin: datetime | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.monotonic_origin = time.perf_counter() if monotonic_origin is None else monotonic_origin
        self.wall_origin = datetime.now(timezone.utc) if wall_origin is None else wall_origin

    def wall_time(self, timestamp: float) -> datetime:
        return self.wall_origin + timedelta(seconds=timestamp - self.monotonic_origin)

    def log(self, events: list[AlertEvent], *, timestamp: float | None = None) -> None:
        if not events:
            return
        records = []
        for event in events:
            # Publication time preserves ordering when delayed pose results carry
            # older capture timestamps than events already evaluated on the HUD.
            published_at = self.wall_time(event.timestamp if timestamp is None else timestamp)
            records.append(json.dumps({
                "timestamp": published_at.isoformat(), "module": event.module,
                "event": event.event, "track_id": event.track_id, "metrics": event.metrics,
            }, allow_nan=False) + "\n")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.writelines(records)
