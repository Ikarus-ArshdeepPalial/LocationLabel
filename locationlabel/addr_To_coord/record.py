from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from common.record import record_path, update_record

Provider = Literal["google", "census"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GeocodeResult:
    input_address: str
    formatted_address: str | None = None
    lat: float | None = None
    lng: float | None = None
    provider: Provider | None = None
    match_quality: str | None = None
    partial_match: bool | None = None
    street_segment: dict[str, Any] | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    timestamp: str = field(default_factory=now_iso)

    @classmethod
    def failure(cls, address: str, error: str, provider: Provider | None = None) -> "GeocodeResult":
        return cls(input_address=address, provider=provider, status="error", error=error)


def save_record(result: GeocodeResult, output_dir: Path) -> Path:
    path = record_path(output_dir, result.input_address)
    update_record(path, asdict(result))
    return path
