import json
import re
from pathlib import Path
from typing import Any


def normalize_address(address: str) -> str:
    return re.sub(r"[\W_]+", " ", address.lower()).strip()


def slugify(address: str) -> str:
    return normalize_address(address).replace(" ", "_")[:100]


def record_path(output_dir: Path, address: str) -> Path:
    return output_dir / slugify(address) / "result.json"


def load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_record(path: Path, values: dict[str, Any]) -> None:
    """Merge values into the record so keys written by other stages are preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = load_record(path)
    record.update(values)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
