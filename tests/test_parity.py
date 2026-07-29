import codecs
import inspect

import chardet
import numpy as np
import pytest

import mojo_chardet
from mojo_chardet import UniversalDetector
from mojo_chardet._lib import PARALLEL_THRESHOLD, lib, scan
from mojo_chardet.detector import UniversalDetector as CanonicalUniversalDetector


def assert_core_parity(data):
    got = mojo_chardet.detect(data)
    reference = chardet.detect(data)
    assert got["encoding"] == reference["encoding"]
    assert got["confidence"] == pytest.approx(reference["confidence"])
    assert got["mime_type"] == reference["mime_type"]


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"plain ASCII text with enough words",
        b"\x00\x01\x02",
        b"\xef\xbb\xbfhello",
        "Zażółć gęślą jaźń".encode(),
    ],
)
def test_exact_parity_for_deterministic_paths(data):
    assert_core_parity(data)


@pytest.mark.parametrize(
    ("data", "encoding"),
    [
        (b"\x1b$B$3$s$K$A$O", "ISO-2022-JP"),
        (b"\x1b$)Cplain", "ISO-2022-KR"),
        (b"~{abcd~}", "HZ-GB-2312"),
    ],
)
def test_escape_signature_detection(data, encoding):
    assert mojo_chardet.detect(data)["encoding"] == encoding


@pytest.mark.parametrize(
    "data",
    [
        codecs.BOM_UTF16_LE + "hello".encode("utf-16le"),
        codecs.BOM_UTF16_BE + "hello".encode("utf-16be"),
        codecs.BOM_UTF32_LE + "hello".encode("utf-32le"),
        codecs.BOM_UTF32_BE + "hello".encode("utf-32be"),
    ],
)
def test_utf_bom_parity(data):
    assert_core_parity(data)


@pytest.mark.parametrize("codec", ["utf-16le", "utf-16be", "utf-32le", "utf-32be"])
def test_bomless_unicode_parity(codec):
    data = (
        "This is a Unicode document with enough ordinary characters for reliable "
        "detection. " * 5
    ).encode(codec)
    assert_core_parity(data)


@pytest.mark.parametrize(
    ("text", "codec"),
    [
        (
            "Это русский текст для проверки определения кодировки. "
            "Здесь много обычных слов и предложений. " * 20,
            "cp1251",
        ),
        (
            "Это русский текст для проверки определения кодировки. "
            "Здесь много обычных слов и предложений. " * 20,
            "koi8-r",
        ),
        (
            "Это русский текст для проверки определения кодировки. "
            "Здесь много обычных слов и предложений. " * 20,
            "iso8859-5",
        ),
        (
            "Это русский текст для проверки определения кодировки. "
            "Здесь много обычных слов и предложений. " * 20,
            "cp866",
        ),
        (
            "Αυτό είναι ελληνικό κείμενο για τον έλεγχο της κωδικοποίησης "
            "και περιέχει πολλές λέξεις. " * 20,
            "cp1253",
        ),
        (
            "Café déjà vu. Voilà une histoire française avec des caractères "
            "accentués. " * 20,
            "cp1252",
        ),
    ],
)
def test_single_byte_top_choice_matches_upstream(text, codec):
    data = text.encode(codec)
    got = mojo_chardet.detect(data)
    reference = chardet.detect(data)
    assert got["encoding"] == reference["encoding"]
    assert got["confidence"] > mojo_chardet.MINIMUM_THRESHOLD
    assert reference["confidence"] > 0


@pytest.mark.parametrize(
    ("text", "codec"),
    [
        (
            "これは日本語の文章です。文字コードの判定を正しく確認するための"
            "テキストです。" * 30,
            "shift_jis",
        ),
        (
            "これは日本語の文章です。文字コードの判定を正しく確認するための"
            "テキストです。" * 30,
            "euc_jp",
        ),
        (
            "这是一个用于测试字符编码检测的中文文本，其中包含许多常用的汉字"
            "和标点符号。" * 30,
            "gb2312",
        ),
        (
            "這是一段用於測試字元編碼偵測的繁體中文文字，其中包含許多常用漢字"
            "和標點符號。" * 30,
            "big5",
        ),
        (
            "이것은 문자 인코딩 감지를 테스트하기 위한 한국어 문장입니다. "
            "일반적인 단어가 포함되어 있습니다. " * 30,
            "euc_kr",
        ),
    ],
)
def test_multibyte_top_choice_matches_upstream(text, codec):
    data = text.encode(codec)
    got = mojo_chardet.detect(data)
    reference = chardet.detect(data)
    assert got["encoding"] == reference["encoding"]
    assert got["language"] == reference["language"]
    assert got["confidence"] >= 0.5
    assert reference["confidence"] >= mojo_chardet.MINIMUM_THRESHOLD


@pytest.mark.parametrize(
    "bad_utf8",
    [
        b"\xc0\xaf" * 20,
        b"\xe0\x80\xaf" * 20,
        b"\xed\xa0\x80" * 20,
        b"\xf4\x90\x80\x80" * 20,
        b"\xf0\x9f\x92",
    ],
)
def test_rfc3629_invalid_sequences_are_rejected(bad_utf8):
    assert mojo_chardet.detect(bad_utf8)["encoding"] != "utf-8"
    stats, _ = scan(bad_utf8)
    assert stats[10] == 0
    assert stats[12] > 0


def test_native_scanner_counts_utf8_and_histogram():
    data = ("héllö 世界" * 100).encode()
    stats, hist = scan(data)
    assert stats[0] == len(data)
    assert stats[10] == 1
    assert stats[11] == 400
    assert hist.sum() == len(data)
    assert hist[ord("h")] == 100


def test_native_scanner_rejects_invalid_abi_arguments():
    native = lib().chd_scan
    output = np.zeros(256, dtype=np.int64)
    address = int(output.ctypes.data)
    assert native(0, 1, address, address) != 0
    assert native(address, -1, address, address) != 0


@pytest.mark.parametrize("size", [PARALLEL_THRESHOLD - 1, PARALLEL_THRESHOLD + 3])
def test_native_scanner_serial_parallel_threshold_and_simd_tail(size):
    data = b"A" * size
    stats, hist = scan(data)
    assert stats[:10].tolist() == [size, 0, 0, 0, 0, size, 0, 0, 0, 0]
    assert stats[10:13].tolist() == [1, 0, 0]
    assert stats[[13, 16, 19, 22, 25, 27]].tolist() == [1, 1, 1, 1, 1, 1]
    assert hist.sum() == size
    assert hist[ord("A")] == size
    assert np.count_nonzero(hist) == 1


def test_full_bytearray_reaches_scanner_without_copy(monkeypatch):
    data = bytearray(b"plain ASCII text")
    seen = {}
    original = mojo_chardet._detector.scan

    def capture(value):
        seen["same"] = value is data
        return original(value)

    monkeypatch.setattr(mojo_chardet._detector, "scan", capture)
    assert mojo_chardet.detect(data)["encoding"] == "ascii"
    assert seen["same"]


def test_detect_all_is_ranked_and_contains_upstream_winner():
    data = (
        "Это русский текст с несколькими обычными предложениями для проверки. " * 20
    ).encode("koi8-r")
    got = mojo_chardet.detect_all(data)
    reference = chardet.detect(data)
    assert got[0]["encoding"] == reference["encoding"]
    assert [item["confidence"] for item in got] == sorted(
        (item["confidence"] for item in got), reverse=True
    )


def test_streaming_matches_one_shot_and_reset():
    data = ("これはストリーミング検出のテストです。" * 40).encode("euc_jp")
    detector = UniversalDetector()
    for start in range(0, len(data), 17):
        detector.feed(data[start : start + 17])
    assert detector.close()["encoding"] == mojo_chardet.detect(data)["encoding"]
    assert detector.done
    with pytest.raises(ValueError):
        detector.feed(b"more")
    detector.reset()
    detector.feed(b"plain ASCII")
    assert detector.close()["encoding"] == "ascii"


def test_max_bytes_and_empty_override_match_upstream():
    data = b"A" * 64 + "日本語".encode()
    assert mojo_chardet.detect(data, max_bytes=64)["encoding"] == chardet.detect(
        data, max_bytes=64
    )["encoding"]
    assert mojo_chardet.detect(
        b"", empty_input_encoding="ascii"
    ) == chardet.detect(b"", empty_input_encoding="ascii")


def test_include_exclude_and_raw_codec_names():
    data = ("Καλημέρα κόσμε. " * 50).encode("utf-8")
    assert mojo_chardet.detect(data, include_encodings=["utf-8"])["encoding"] == "utf-8"
    assert mojo_chardet.detect(data, exclude_encodings=["utf-8"])["encoding"] != "utf-8"
    euc = ("これは日本語です。" * 30).encode("euc_jp")
    got = mojo_chardet.detect(euc, compat_names=False)
    assert codecs.lookup(got["encoding"]).name == "euc_jp"
    with pytest.raises(ValueError, match="Unknown encoding"):
        mojo_chardet.detect(data, include_encodings=["not-a-codec"])


def test_public_function_signatures_match_upstream():
    assert inspect.signature(mojo_chardet.detect) == inspect.signature(chardet.detect)
    assert inspect.signature(mojo_chardet.detect_all) == inspect.signature(
        chardet.detect_all
    )
    assert inspect.signature(mojo_chardet.UniversalDetector) == inspect.signature(
        chardet.UniversalDetector
    )
    assert CanonicalUniversalDetector is UniversalDetector
    assert mojo_chardet.EncodingEra.ALL
    assert mojo_chardet.LanguageFilter.ALL


def test_type_validation_matches_classic_api():
    with pytest.raises(TypeError):
        mojo_chardet.detect("not bytes")
