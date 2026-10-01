import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Job:
    address: str
    images_dir: Path | None = None


def read_jobs(csv_path: Path) -> list[Job]:
    """Rows with an 'address' and an optional 'images_dir' (relative paths are resolved from the CSV's folder)."""
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "address" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} has no 'address' column")
        jobs = []
        for row in reader:
            address = (row["address"] or "").strip()
            images_dir = (row.get("images_dir") or "").strip()
            if address:
                jobs.append(Job(address, csv_path.parent / images_dir if images_dir else None))
        return jobs
