"""Fast charset detection powered by Mojo."""

from __future__ import annotations

import codecs
from collections.abc import Iterable
from typing import TypedDict

from ._detector import Candidate, candidates
from .enums import EncodingEra, LanguageFilter
from .universaldetector import UniversalDetector

VERSION = __version__ = "0.1.0"
DEFAULT_MAX_BYTES = 200_000
MINIMUM_THRESHOLD = 0.20


class DetectionDict(TypedDict):
    encoding: str | None
    confidence: float
    language: str | None
    mime_type: str | None


DetectionResult = Candidate

__all__ = [
    "DEFAULT_MAX_BYTES",
    "MINIMUM_THRESHOLD",
    "DetectionDict",
    "DetectionResult",
    "EncodingEra",
    "LanguageFilter",
    "UniversalDetector",
    "detect",
    "detect_all",
    "__version__",
]


def _validate(
    byte_str: bytes | bytearray, max_bytes: int
) -> bytes | bytearray:
    if not isinstance(byte_str, (bytes, bytearray)):
        raise TypeError(
            f"Expected object of type bytes or bytearray, got: {type(byte_str)}"
        )
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if len(byte_str) <= max_bytes:
        return byte_str
    return byte_str[:max_bytes]


def _normalized(name: str) -> str:
    try:
        return codecs.lookup(name).name
    except LookupError:
        return name.lower().replace("_", "-")


def _filter_names(names: Iterable[str] | None, option: str) -> set[str]:
    normalized = set()
    for name in names or ():
        try:
            normalized.add(codecs.lookup(name).name)
        except (LookupError, TypeError) as exc:
            raise ValueError(f"Unknown encoding {name!r} in {option}") from exc
    return normalized


def _filtered(
    ranked: list[Candidate],
    include_encodings: Iterable[str] | None,
    exclude_encodings: Iterable[str] | None,
) -> list[Candidate]:
    include = _filter_names(include_encodings, "include_encodings")
    exclude = _filter_names(exclude_encodings, "exclude_encodings")
    return [
        item
        for item in ranked
        if (not include or _normalized(item.encoding) in include)
        and _normalized(item.encoding) not in exclude
    ]


def _format(
    item: Candidate,
    prefer_superset: bool,
    compat_names: bool,
) -> DetectionDict:
    result = item.result(prefer_superset)
    if result["encoding"] and not compat_names:
        result["encoding"] = _normalized(str(result["encoding"]))
    return result  # type: ignore[return-value]


def detect(
    byte_str: bytes | bytearray,
    should_rename_legacy: bool = False,
    encoding_era: EncodingEra = EncodingEra.ALL,
    chunk_size: int = 65536,
    max_bytes: int = DEFAULT_MAX_BYTES,
    *,
    prefer_superset: bool = False,
    compat_names: bool = True,
    include_encodings: Iterable[str] | None = None,
    exclude_encodings: Iterable[str] | None = None,
    no_match_encoding: str = "cp1252",
    empty_input_encoding: str = "utf-8",
) -> DetectionDict:
    del encoding_era, chunk_size
    data = _validate(byte_str, max_bytes)
    if not data:
        return {
            "encoding": empty_input_encoding,
            "confidence": 0.1,
            "language": None,
            "mime_type": "text/plain",
        }
    ranked = _filtered(candidates(data), include_encodings, exclude_encodings)
    if ranked and ranked[0].encoding:
        return _format(
            ranked[0], should_rename_legacy or prefer_superset, compat_names
        )
    if b"\x00" in data or sum(value < 9 or 13 < value < 32 for value in data) > len(data) // 10:
        return {
            "encoding": None,
            "confidence": 0.95,
            "language": None,
            "mime_type": "application/octet-stream",
        }
    return {
        "encoding": no_match_encoding,
        "confidence": 0.1,
        "language": None,
        "mime_type": "text/plain",
    }


def detect_all(
    byte_str: bytes | bytearray,
    ignore_threshold: bool = False,
    should_rename_legacy: bool = False,
    encoding_era: EncodingEra = EncodingEra.ALL,
    chunk_size: int = 65536,
    max_bytes: int = DEFAULT_MAX_BYTES,
    *,
    prefer_superset: bool = False,
    compat_names: bool = True,
    include_encodings: Iterable[str] | None = None,
    exclude_encodings: Iterable[str] | None = None,
    no_match_encoding: str = "cp1252",
    empty_input_encoding: str = "utf-8",
) -> list[DetectionDict]:
    del encoding_era, chunk_size
    data = _validate(byte_str, max_bytes)
    if not data:
        return [detect(data, max_bytes=max_bytes, empty_input_encoding=empty_input_encoding)]
    ranked = _filtered(candidates(data), include_encodings, exclude_encodings)
    if not ignore_threshold:
        above = [item for item in ranked if item.confidence > MINIMUM_THRESHOLD]
        if above:
            ranked = above
    if not ranked or not ranked[0].encoding:
        return [
            detect(
                data,
                should_rename_legacy=should_rename_legacy,
                max_bytes=max_bytes,
                prefer_superset=prefer_superset,
                compat_names=compat_names,
                include_encodings=include_encodings,
                exclude_encodings=exclude_encodings,
                no_match_encoding=no_match_encoding,
                empty_input_encoding=empty_input_encoding,
            )
        ]
    return [
        _format(item, should_rename_legacy or prefer_superset, compat_names)
        for item in ranked
        if item.encoding
    ]
