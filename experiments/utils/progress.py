"""
Shared terminal progress helpers.

Adapted from GibbsQ ``gibbsq.utils.progress``.

If ``tqdm`` is installed the user sees a live bar.  Otherwise a
no-op stand-in is used so experiment code never has to branch.

Usage
-----
::

    from experiments.utils.progress import create_progress, iter_progress

    for item in iter_progress(items, desc="running", total=len(items)):
        process(item)

    with create_progress(total=100, desc="trials") as pbar:
        pbar.update(1)
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

PROGRESS_ENV_VAR = "SYNAPSE_PROGRESS"
VALID_MODES = {"auto", "on", "off"}

__all__ = [
    "create_progress",
    "iter_progress",
    "managed_progress",
]


def _resolve_mode(mode: str | None = None) -> str:
    resolved = (mode or os.environ.get(PROGRESS_ENV_VAR, "auto")).strip().lower()
    if resolved not in VALID_MODES:
        raise ValueError(f"progress mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    return resolved


def _should_show(mode: str | None = None) -> bool:
    resolved = _resolve_mode(mode)
    if resolved == "off":
        return False
    if _tqdm is None:
        return False
    if resolved == "on":
        return True
    # auto: show only when stderr is a tty
    if os.environ.get("CI"):
        return False
    is_tty = getattr(sys.stderr, "isatty", None)
    return bool(is_tty and is_tty())


class _NullProgress:
    """No-op progress with the subset of tqdm API we use."""

    def __init__(self, iterable: Iterable[Any] | None = None):
        self.iterable = iterable
        self.n = 0

    def __iter__(self) -> Iterator[Any]:
        return iter(self.iterable or ())

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, ordered_dict: dict | None = None, refresh: bool = True, **kw: Any) -> None:
        pass

    def set_postfix_str(self, s: str | None = None, refresh: bool = True) -> None:
        pass

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        pass

    def write(self, s: str, file: Any | None = None, end: str = "\n") -> None:
        print(s, file=file or sys.stdout, end=end)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *args: Any) -> bool:
        self.close()
        return False


def create_progress(
    *,
    total: int | None = None,
    desc: str | None = None,
    mode: str | None = None,
    iterable: Iterable[Any] | None = None,
    unit: str = "it",
    **kwargs: Any,
):
    """Create a tqdm progress bar or a no-op stand-in."""
    if not _should_show(mode):
        return _NullProgress(iterable=iterable)
    merged = {"dynamic_ncols": True, "mininterval": 0.5}
    merged.update(kwargs)
    return _tqdm(iterable=iterable, total=total, desc=desc, unit=unit, **merged)


def iter_progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str | None = None,
    mode: str | None = None,
    **kwargs: Any,
):
    """Wrap an iterable with a live progress bar when enabled."""
    return create_progress(iterable=iterable, total=total, desc=desc, mode=mode, **kwargs)


@contextmanager
def managed_progress(
    *,
    total: int | None = None,
    desc: str | None = None,
    mode: str | None = None,
    **kwargs: Any,
):
    """Context manager around :func:`create_progress`."""
    progress = create_progress(total=total, desc=desc, mode=mode, **kwargs)
    try:
        yield progress
    finally:
        progress.close()
