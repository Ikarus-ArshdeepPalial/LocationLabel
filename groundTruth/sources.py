from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import SOURCES_YAML


@dataclass
class FieldMapping:
    source_field: str | None = None
    fallback: list[str] = field(default_factory=list)
    unit: str | None = None
    parse: str | None = None

    @classmethod
    def from_yaml(cls, value: dict[str, Any]) -> "FieldMapping":
        return cls(value.get("field"), value.get("fallback") or [], value.get("unit"), value.get("parse"))


@dataclass
class Source:
    name: str
    enabled: bool
    confirmed: bool
    city: str
    state: str
    object_type: str
    url: str
    where: str
    field_mapping: dict[str, FieldMapping]
    notes: str = ""

    @classmethod
    def from_yaml(cls, value: dict[str, Any]) -> "Source":
        return cls(
            name=value["name"],
            enabled=value.get("enabled", True),
            confirmed=value.get("confirmed", False),
            city=value.get("city", ""),
            state=value.get("state", ""),
            object_type=value["object_type"],
            url=value["url"],
            where=value.get("where", "1=1"),
            field_mapping={k: FieldMapping.from_yaml(v) for k, v in (value.get("field_mapping") or {}).items()},
            notes=value.get("notes", ""),
        )

    def to_yaml(self) -> dict[str, Any]:
        mapping = {}
        for attr, m in self.field_mapping.items():
            entry: dict[str, Any] = {"field": m.source_field}
            if m.fallback:
                entry["fallback"] = m.fallback
            if m.unit:
                entry["unit"] = m.unit
            if m.parse:
                entry["parse"] = m.parse
            mapping[attr] = entry
        return {
            "name": self.name, "enabled": self.enabled, "confirmed": self.confirmed,
            "city": self.city, "state": self.state, "object_type": self.object_type,
            "url": self.url, "where": self.where, "field_mapping": mapping, "notes": self.notes,
        }


def load_sources(path: Path = SOURCES_YAML) -> list[Source]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Source.from_yaml(s) for s in doc.get("sources", [])]


def save_sources(sources: list[Source], path: Path = SOURCES_YAML) -> None:
    header = (
        "# Ground-truth data sources. `confirmed: false` sources are skipped by `download`.\n"
        "# Run `python main.py groundtruth inspect` to check reachability and refresh proposed mappings.\n\n"
    )
    body = yaml.safe_dump({"sources": [s.to_yaml() for s in sources]}, sort_keys=False, width=100, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")


def usable_sources(sources: list[Source], object_type: str | None = None) -> list[Source]:
    return [s for s in sources if s.enabled and s.confirmed and (object_type is None or s.object_type == object_type)]
