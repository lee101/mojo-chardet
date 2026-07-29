"""Detection policy built on the native byte statistics."""

from __future__ import annotations

import codecs
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._lib import scan
from .enums import LanguageFilter

Result = dict[str, str | float | None]

LEGACY_MAP = {
    "ascii": "Windows-1252",
    "iso-8859-1": "Windows-1252",
    "tis-620": "ISO-8859-11",
    "iso-8859-9": "Windows-1254",
    "gb2312": "GB18030",
    "euc-kr": "CP949",
    "utf-16le": "UTF-16",
}


@dataclass(frozen=True)
class Candidate:
    encoding: str
    confidence: float
    language: str | None

    def result(self, rename: bool = False) -> Result:
        name = LEGACY_MAP.get(self.encoding.lower(), self.encoding) if rename else self.encoding
        return {
            "encoding": name,
            "confidence": self.confidence,
            "language": self.language,
            "mime_type": "text/plain",
        }


_RANKED = {
    "Russian": "оеаинтсрвлкмдпуяызьбгчйхжюшцщэфъё",
    "Greek": "αοειντρκσπμληγυδβφχωθξζψ",
    "Turkish": "aeinrlıkmdtuysoübşzgçhvpöğfcj",
    "Hebrew": "יוהלרבתמאשכעדנחקספגצזטםןףךץ",
    "Arabic": "ايلمنوترعبهفقدسحكجشطصضذثخغظ",
    "Polish": "aieonrzswyctkłdpmuęjgżbąhśóćńźf",
}

_BIGRAMS = {
    "Russian": set("ст но то на ен ов ни ра во ко пр по ро го ос ер ал ли ре ор та от ан ти те ла де ес ит ве ар ет ри не ка ру ск ог од ва".split()),
    "Greek": set("ου αι ει το τη στ ον να τα κα σε με ικ εν αρ πο τε ης των για ρο κο".split()),
    "Turkish": set("ar er in an en ler la ve bir de da le li ri ak ma ya ın".split()),
    "Hebrew": set("ים ות של לא את על זה הוא מה עם כי לי לה נה".split()),
    "Arabic": set("ال من في ان ات ون ين عل لا ما ها وا لي".split()),
    "Polish": set("ie rz sz cz ch ni ow wa ze na po pr st ro".split()),
}

_SINGLE_BYTE = (
    ("Windows-1251", "cp1251", "Russian"),
    ("KOI8-R", "koi8-r", "Russian"),
    ("ISO-8859-5", "iso8859-5", "Russian"),
    ("IBM866", "cp866", "Russian"),
    ("MacCyrillic", "mac_cyrillic", "Russian"),
    ("ISO-8859-7", "iso8859-7", "Greek"),
    ("windows-1253", "cp1253", "Greek"),
    ("ISO-8859-9", "iso8859-9", "Turkish"),
    ("Windows-1254", "cp1254", "Turkish"),
    ("ISO-8859-8", "iso8859-8", "Hebrew"),
    ("windows-1255", "cp1255", "Hebrew"),
    ("ISO-8859-6", "iso8859-6", "Arabic"),
    ("Windows-1256", "cp1256", "Arabic"),
    ("ISO-8859-2", "iso8859-2", "Polish"),
    ("Windows-1250", "cp1250", "Polish"),
)


@dataclass(frozen=True)
class _SingleByteTable:
    valid: npt.NDArray[np.bool_]
    printable: npt.NDArray[np.bool_]
    alphabet: npt.NDArray[np.bool_]
    simple_fold: npt.NDArray[np.bool_]
    weights: npt.NDArray[np.float64]
    bigrams: npt.NDArray[np.bool_]


def _single_byte_table(codec: str, language: str) -> _SingleByteTable:
    ranked = _RANKED[language]
    char_weights = {
        char: (len(ranked) - index) / len(ranked)
        for index, char in enumerate(ranked)
    }
    valid = np.zeros(256, dtype=np.bool_)
    printable = np.zeros(256, dtype=np.bool_)
    alphabet = np.zeros(256, dtype=np.bool_)
    simple_fold = np.zeros(256, dtype=np.bool_)
    weights = np.zeros(256, dtype=np.float64)
    encoded_letters: dict[str, list[int]] = {}
    for value in range(256):
        try:
            char = bytes((value,)).decode(codec)
        except UnicodeDecodeError:
            continue
        valid[value] = True
        printable[value] = char.isprintable()
        folded = char.casefold()
        simple_fold[value] = len(folded) == 1
        if len(folded) == 1 and folded in char_weights:
            alphabet[value] = True
            weights[value] = char_weights[folded]
            encoded_letters.setdefault(folded, []).append(value)
    bigrams = np.zeros((256, 256), dtype=np.bool_)
    for pair in _BIGRAMS[language]:
        for first in encoded_letters.get(pair[0], ()):
            for second in encoded_letters.get(pair[1], ()):
                bigrams[first, second] = True
    for array in (valid, printable, alphabet, simple_fold, weights, bigrams):
        array.setflags(write=False)
    return _SingleByteTable(
        valid, printable, alphabet, simple_fold, weights, bigrams
    )


_SINGLE_BYTE_TABLES = {
    (codec, language): _single_byte_table(codec, language)
    for _, codec, language in _SINGLE_BYTE
}

_SIMPLIFIED = set(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主"
    "发年动同工也能下过子说产种面而方后多定行学法所民得经之进着等部度家电力里"
)
_TRADITIONAL = set(
    "的一是在不了有和人這中大為上個國我以要他時來用們生到作地於出就分對成會可主"
    "發年動同工也能下過子說產種面而方後多定行學法所民得經之進著等部度家電力裡"
)


def _is_cjk(c: str) -> bool:
    value = ord(c)
    return 0x3400 <= value <= 0x9FFF or 0xF900 <= value <= 0xFAFF


def _is_kana(c: str) -> bool:
    value = ord(c)
    return 0x3040 <= value <= 0x30FF or 0xFF66 <= value <= 0xFF9F


def _is_hangul(c: str) -> bool:
    value = ord(c)
    return 0x1100 <= value <= 0x11FF or 0x3130 <= value <= 0x318F or 0xAC00 <= value <= 0xD7AF


def _multibyte_score(data: bytes | bytearray, codec: str, language: str) -> float:
    sample = bytes(data[:131072])
    try:
        text = sample.decode(codec, "strict")
    except UnicodeDecodeError:
        return -1.0
    non_ascii = [c for c in text if ord(c) >= 128 and not c.isspace()]
    if not non_ascii:
        return 0.0
    if language == "Japanese":
        kana = sum(_is_kana(c) for c in non_ascii)
        cjk = sum(_is_cjk(c) for c in non_ascii)
        return min(1.0, (kana * 1.4 + cjk * 0.35) / len(non_ascii))
    if language == "Korean":
        return sum(_is_hangul(c) for c in non_ascii) / len(non_ascii)
    cjk_chars = [c for c in non_ascii if _is_cjk(c)]
    if not cjk_chars:
        return 0.0
    common = _SIMPLIFIED | _TRADITIONAL
    return min(1.0, len(cjk_chars) / len(non_ascii) * 0.75 + sum(c in common for c in cjk_chars) / len(cjk_chars) * 0.5)


def _single_byte_sequence(
    data: bytes | bytearray,
    sample: npt.NDArray[np.uint8],
    codec: str,
    language: str,
    table: _SingleByteTable,
) -> float:
    if sample.size < 2:
        return 0.0
    if np.all(table.simple_fold[sample]):
        adjacent = table.alphabet[sample[:-1]] & table.alphabet[sample[1:]]
        total_pairs = int(np.count_nonzero(adjacent))
        hits = int(np.count_nonzero(table.bigrams[sample[:-1], sample[1:]]))
        return min(1.0, 2.0 * hits / max(1, total_pairs))
    text = bytes(data[:8192]).decode(codec).casefold()
    ranked = _RANKED[language]
    alphabet = set(ranked)
    words = "".join(char if char in alphabet else " " for char in text).split()
    total_pairs = sum(max(0, len(word) - 1) for word in words)
    bigrams = _BIGRAMS[language]
    hits = sum(
        word[index : index + 2] in bigrams
        for word in words
        for index in range(len(word) - 1)
    )
    return min(1.0, 2.0 * hits / max(1, total_pairs))


def _single_byte_score(
    data: bytes | bytearray,
    sample: npt.NDArray[np.uint8],
    hist: npt.NDArray[np.int64],
    codec: str,
    language: str,
) -> float:
    table = _SINGLE_BYTE_TABLES[codec, language]
    high_hist = hist[128:]
    if np.any(high_hist[~table.valid[128:]]):
        return -1.0
    printable_high = int(np.dot(high_hist, table.printable[128:]))
    if printable_high == 0:
        return 0.0
    expected = int(np.dot(high_hist, table.alphabet[128:]))
    weighted = float(np.dot(high_hist, table.weights[128:]))
    ascii_letters = int(hist[65:91].sum() + hist[97:123].sum())
    script_share = expected / max(1, printable_high + ascii_letters)
    frequency = weighted / expected if expected else 0.0
    sequence = _single_byte_sequence(data, sample, codec, language, table)
    return script_share * (0.25 * frequency + 0.75 * sequence)


def _utf_without_bom(data: bytes | bytearray, stats: np.ndarray) -> Candidate | None:
    n = len(data)
    if n < 40 or stats[2] == 0:
        return None
    zeros = stats[6:10].astype(float)
    positions = np.array([(n + 3 - i) // 4 for i in range(4)], dtype=float)
    nonzeros = positions - zeros
    chars32 = max(1.0, n / 4)
    choices = (
        ("utf-32-be", zeros[0] / chars32, zeros[1] / chars32, zeros[2] / chars32, nonzeros[3] / chars32),
        ("utf-32-le", nonzeros[0] / chars32, zeros[1] / chars32, zeros[2] / chars32, zeros[3] / chars32),
    )
    for name, *ratios in choices:
        if chars32 >= 20 and min(ratios) > 0.94:
            try:
                bytes(data).decode(name)
            except UnicodeDecodeError:
                continue
            return Candidate(name, 0.95, None)
    chars16 = max(1.0, n / 2)
    utf16 = (
        ("utf-16-be", (zeros[0] + zeros[2]) / chars16, (nonzeros[1] + nonzeros[3]) / chars16),
        ("utf-16-le", (nonzeros[0] + nonzeros[2]) / chars16, (zeros[1] + zeros[3]) / chars16),
    )
    for name, first, second in utf16:
        if chars16 >= 20 and first > 0.94 and second > 0.94:
            try:
                bytes(data).decode(name)
            except UnicodeDecodeError:
                continue
            return Candidate(name, 0.95, None)
    return None


def _bom(data: bytes | bytearray) -> Candidate | None:
    raw = bytes(data[:4])
    if raw.startswith(codecs.BOM_UTF8):
        return Candidate("UTF-8-SIG", 1.0, "")
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return Candidate("UTF-32", 1.0, "")
    if raw.startswith(b"\xFE\xFF\x00\x00"):
        return Candidate("X-ISO-10646-UCS-4-3412", 1.0, "")
    if raw.startswith(b"\x00\x00\xFF\xFE"):
        return Candidate("X-ISO-10646-UCS-4-2143", 1.0, "")
    if raw.startswith((codecs.BOM_LE, codecs.BOM_BE)):
        return Candidate("UTF-16", 1.0, "")
    return None


def candidates(
    data: bytes | bytearray, lang_filter: LanguageFilter = LanguageFilter.ALL
) -> list[Candidate]:
    bom = _bom(data)
    if bom:
        return [bom]
    if not data:
        return [Candidate("", 0.0, "")]

    stats, hist = scan(data)
    utf = _utf_without_bom(data, stats)
    if utf:
        return [utf]
    if stats[1] == 0:
        raw = bytes(data)
        if b"\x1b$" in raw:
            if b"\x1b$)C" in raw:
                return [Candidate("ISO-2022-KR", 0.99, "Korean")]
            return [Candidate("ISO-2022-JP", 0.99, "Japanese")]
        if b"~{" in raw:
            return [Candidate("HZ-GB-2312", 0.99, "Chinese")]
        if float(stats[3]) / len(data) > 0.10:
            return [Candidate("", 0.0, "")]
        return [Candidate("ascii", 1.0, "en")]

    found: list[Candidate] = []
    if stats[10] and stats[11]:
        return [Candidate("utf-8", 0.99, None)]

    mb_specs: list[tuple[str, str, str, int, int, LanguageFilter]] = [
        ("cp932", "shift_jis", "ja", 13, 14, LanguageFilter.JAPANESE),
        ("EUC-JP", "euc_jp", "ja", 16, 17, LanguageFilter.JAPANESE),
        ("GB18030", "gb18030", "zh", 19, 20, LanguageFilter.CHINESE_SIMPLIFIED),
        ("Big5", "big5", "zh", 22, 23, LanguageFilter.CHINESE_TRADITIONAL),
        ("CP949", "euc_kr", "ko", 25, 26, LanguageFilter.KOREAN),
    ]
    if stats[27] and not stats[25]:
        mb_specs.append(("CP949", "cp949", "ko", 27, 28, LanguageFilter.KOREAN))
    for name, codec, language, valid_slot, count_slot, required in mb_specs:
        if not (lang_filter & required) or not stats[valid_slot] or stats[count_slot] < 2:
            continue
        score_language = {"ja": "Japanese", "zh": "Chinese", "ko": "Korean"}[language]
        score = _multibyte_score(data, codec, score_language)
        if score > 0.12:
            found.append(Candidate(name, min(0.99, 0.55 + score * 0.55), language))

    if lang_filter & LanguageFilter.NON_CJK:
        sample = np.frombuffer(data, dtype=np.uint8, count=min(len(data), 8192))
        for name, codec, language in _SINGLE_BYTE:
            score = _single_byte_score(data, sample, hist, codec, language)
            if name == "MacCyrillic":
                score -= 0.04
            if score > 0.18:
                language_code = {
                    "Russian": "ru",
                    "Greek": "el",
                    "Turkish": "tr",
                    "Hebrew": "he",
                    "Arabic": "ar",
                    "Polish": "pl",
                }[language]
                found.append(Candidate(name, min(0.99, 0.20 + score), language_code))
        control_ratio = float(stats[3]) / len(data)
        if control_ratio < 0.02:
            found.append(Candidate("Windows-1252", 0.50, None))

    found.sort(key=lambda item: item.confidence, reverse=True)
    return found


def best_result(
    data: bytes | bytearray,
    lang_filter: LanguageFilter = LanguageFilter.ALL,
    rename: bool = False,
) -> Result:
    ranked = candidates(data, lang_filter)
    if not ranked or ranked[0].encoding == "" or ranked[0].confidence <= 0.20:
        return {
            "encoding": None,
            "confidence": 0.0,
            "language": None,
            "mime_type": None,
        }
    return ranked[0].result(rename)
