"""End-to-end mojo-chardet versus upstream chardet benchmarks."""

from __future__ import annotations

import gc
import os
import platform
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import chardet  # noqa: E402
import mojo_chardet  # noqa: E402


def repeated_bytes(text: str, encoding: str, minimum: int) -> bytes:
    unit = text.encode(encoding)
    return unit * ((minimum + len(unit) - 1) // len(unit))


def best_time(fn, repetitions: int = 4) -> float:
    fn()
    gc.disable()
    try:
        best = float("inf")
        for _ in range(repetitions):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
        return best
    finally:
        gc.enable()


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    mib = 1024 * 1024
    russian = repeated_bytes(
        "Это русский текст для проверки кодировки с обычными словами. ",
        "cp1251",
        2 * mib,
    )
    japanese = repeated_bytes(
        "これは日本語の文章です。文字コードを正しく判定するためのテキストです。",
        "shift_jis",
        2 * mib,
    )
    utf8 = repeated_bytes(
        "Unicode text: Καλημέρα κόσμε, こんにちは世界, héllo. ",
        "utf-8",
        5 * mib,
    )
    ascii_data = repeated_bytes(
        "Plain ASCII text with ordinary words and punctuation. ",
        "ascii",
        5 * mib,
    )

    workloads = [
        (
            "detect ASCII, full 5 MiB",
            lambda: mojo_chardet.detect(ascii_data, max_bytes=len(ascii_data)),
            lambda: chardet.detect(ascii_data, max_bytes=len(ascii_data)),
        ),
        (
            "detect UTF-8, default 200 kB",
            lambda: mojo_chardet.detect(utf8),
            lambda: chardet.detect(utf8),
        ),
        (
            "detect UTF-8, full 5 MiB",
            lambda: mojo_chardet.detect(utf8, max_bytes=len(utf8)),
            lambda: chardet.detect(utf8, max_bytes=len(utf8)),
        ),
        (
            "detect Windows-1251, full 2 MiB",
            lambda: mojo_chardet.detect(russian, max_bytes=len(russian)),
            lambda: chardet.detect(russian, max_bytes=len(russian)),
        ),
        (
            "detect cp932, full 2 MiB",
            lambda: mojo_chardet.detect(japanese, max_bytes=len(japanese)),
            lambda: chardet.detect(japanese, max_bytes=len(japanese)),
        ),
        (
            "detect_all Windows-1251, 2 MiB",
            lambda: mojo_chardet.detect_all(russian, max_bytes=len(russian)),
            lambda: chardet.detect_all(russian, max_bytes=len(russian)),
        ),
    ]

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}")
    print(f"Python {platform.python_version()}, chardet {chardet.__version__}")
    print()
    print("| Workload | mojo-chardet | chardet | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_fn, upstream_fn in workloads:
        mojo_result = mojo_fn()
        upstream_result = upstream_fn()
        if isinstance(mojo_result, list):
            assert mojo_result[0]["encoding"] == upstream_result[0]["encoding"]
        else:
            assert mojo_result["encoding"] == upstream_result["encoding"]
        mojo_seconds = best_time(mojo_fn)
        upstream_seconds = best_time(upstream_fn)
        speedup = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
