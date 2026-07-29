"""ctypes bridge to the Mojo byte scanner."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "chardet.mojo")
LIB = os.environ.get("MOJO_CHARDET_LIB") or os.path.join(
    ROOT, "dist", "libmojo-chardet.so"
)

I = ctypes.c_int64
STATUS = ctypes.c_int64
PARALLEL_THRESHOLD = 262_144
_loaded: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than its source."""
    if os.environ.get("MOJO_CHARDET_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_CHARDET_LIB does not exist: {LIB}")
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC)
    if force or stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        _loaded.chd_scan.argtypes = [I, I, I, I]
        _loaded.chd_scan.restype = STATUS
    return _loaded


def scan(data: bytes | bytearray) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed scanner statistics and byte histogram."""
    stats = np.zeros(32, dtype=np.int64)
    hist = np.zeros(256, dtype=np.int64)
    if not data:
        return stats, hist
    src = np.frombuffer(data, dtype=np.uint8)
    if src.size > (1 << 63) - 1:
        raise OverflowError("input is too large for the native scanner")
    if (
        src.dtype != np.uint8
        or not src.flags.c_contiguous
        or stats.dtype != np.int64
        or not stats.flags.c_contiguous
        or hist.dtype != np.int64
        or not hist.flags.c_contiguous
    ):
        raise RuntimeError("invalid native scanner buffer layout")
    status = lib().chd_scan(
        int(src.ctypes.data),
        int(src.size),
        int(stats.ctypes.data),
        int(hist.ctypes.data),
    )
    if status != 0:
        raise RuntimeError(f"native scanner failed with status {status}")
    return stats, hist
