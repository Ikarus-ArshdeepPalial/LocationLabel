"""Progress bars for the pipeline. The app registers a listener instead of drawing bars in a terminal."""

import os
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")

BAR_FORMAT = "{desc:<16} {percentage:3.0f}%|{bar:28}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}"

Listener = Callable[[str, int, int | None], None]
_listener: Listener | None = None
_sink = open(os.devnull, "w")


def set_progress_listener(listener: Listener | None) -> None:
    global _listener
    _listener = listener


class _ReportingBar(tqdm):
    """Invisible bar that tells the listener (description, done, total) on every step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._report()

    def update(self, n: float | None = 1):
        shown = super().update(n)
        self._report()
        return shown

    def _report(self) -> None:
        if _listener:
            _listener(str(self.desc).strip(), int(self.n), int(self.total) if self.total else None)


def progress(iterable: Iterable[T] | None = None, desc: str = "", total: int | None = None, unit: str = "img", leave: bool = False) -> tqdm:
    """Inner bar (per image/batch); cleared when done unless leave=True."""
    if _listener:
        return _ReportingBar(iterable, desc=desc, total=total, unit=unit, leave=False, file=_sink, disable=False, mininterval=0, miniters=1)
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave, disable=None, dynamic_ncols=True, bar_format=BAR_FORMAT)


def track_addresses(addresses: list[str], desc: str) -> Iterator[str]:
    """Stage-level bar over addresses that shows the one being processed."""
    with progress(addresses, desc, unit="addr", leave=True) as bar:
        for address in bar:
            bar.set_postfix_str(address if len(address) <= 40 else address[:37] + "...")
            yield address
