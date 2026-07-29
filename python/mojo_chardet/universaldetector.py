"""Streaming facade compatible with chardet's UniversalDetector."""

from __future__ import annotations

from collections.abc import Iterable

from ._detector import LEGACY_MAP
from .enums import EncodingEra, LanguageFilter


class UniversalDetector:
    MINIMUM_THRESHOLD = 0.20
    LEGACY_MAP = LEGACY_MAP

    def __init__(
        self,
        lang_filter: LanguageFilter = LanguageFilter.ALL,
        should_rename_legacy: bool = False,
        encoding_era: EncodingEra = EncodingEra.ALL,
        max_bytes: int = 200_000,
        *,
        prefer_superset: bool = False,
        compat_names: bool = True,
        include_encodings: Iterable[str] | None = None,
        exclude_encodings: Iterable[str] | None = None,
        no_match_encoding: str = "cp1252",
        empty_input_encoding: str = "utf-8",
    ) -> None:
        self.lang_filter = lang_filter
        self._options = {
            "should_rename_legacy": should_rename_legacy,
            "encoding_era": encoding_era,
            "max_bytes": max_bytes,
            "prefer_superset": prefer_superset,
            "compat_names": compat_names,
            "include_encodings": include_encodings,
            "exclude_encodings": exclude_encodings,
            "no_match_encoding": no_match_encoding,
            "empty_input_encoding": empty_input_encoding,
        }
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self._max_bytes = max_bytes
        self.reset()

    def reset(self) -> None:
        self._buffer = bytearray()
        self._done = False
        self._closed = False
        self._result = None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def result(self) -> dict[str, str | float | None]:
        if self._result is not None:
            return self._result
        return {
            "encoding": None,
            "confidence": 0.0,
            "language": None,
            "mime_type": None,
        }

    def feed(self, byte_str: bytes | bytearray) -> None:
        if self._closed:
            raise ValueError("feed() called after close() without reset()")
        if self._done:
            return
        remaining = self._max_bytes - len(self._buffer)
        if remaining > 0:
            self._buffer.extend(byte_str[:remaining])
        if len(self._buffer) >= self._max_bytes:
            self._done = True

    def close(self) -> dict[str, str | float | None]:
        if not self._closed:
            from . import detect

            self._result = detect(bytes(self._buffer), **self._options)
            self._closed = True
            self._done = True
        return self.result
