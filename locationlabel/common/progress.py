"""Progress bars for the pipeline; hidden automatically when output isn't a terminal (logs, CI, tests)."""

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")

BAR_FORMAT = "{desc:<16} {percentage:3.0f}%|{bar:28}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}"


def progress(iterable: Iterable[T] | None = None, desc: str = "", total: int | None = None, unit: str = "img", leave: bool = False) -> tqdm:
    """Inner bar (per image/batch); cleared when done unless leave=True."""
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave, disable=None, dynamic_ncols=True, bar_format=BAR_FORMAT)


def track_addresses(addresses: list[str], desc: str) -> Iterator[str]:
    """Stage-level bar over addresses that shows the one being processed."""
    with progress(addresses, desc, unit="addr", leave=True) as bar:
        for address in bar:
            bar.set_postfix_str(address if len(address) <= 40 else address[:37] + "...")
            yield address
