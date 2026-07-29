"""Public enums compatible with chardet's detector API."""

from enum import Enum, IntFlag


class InputState:
    PURE_ASCII = 0
    ESC_ASCII = 1
    HIGH_BYTE = 2


class EncodingEra(IntFlag):
    MODERN_WEB = 1
    LEGACY_ISO = 2
    LEGACY_MAC = 4
    LEGACY_REGIONAL = 8
    DOS = 16
    MAINFRAME = 32
    ALL = MODERN_WEB | LEGACY_ISO | LEGACY_MAC | LEGACY_REGIONAL | DOS | MAINFRAME


class LanguageFilter(IntFlag):
    CHINESE_SIMPLIFIED = 0x01
    CHINESE_TRADITIONAL = 0x02
    JAPANESE = 0x04
    KOREAN = 0x08
    NON_CJK = 0x10
    ALL = 0x1F
    CHINESE = CHINESE_SIMPLIFIED | CHINESE_TRADITIONAL
    CJK = CHINESE | JAPANESE | KOREAN


class ProbingState(Enum):
    DETECTING = 0
    FOUND_IT = 1
    NOT_ME = 2
