"""Byte-level charset validation and statistics for the Python detector."""

from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def utf8_scan(src: BPtr, n: Int, stats: IPtr):
    var i = 0
    var chars = 0
    while i < n:
        var a = Int(src[i])
        if a < 0x80:
            i += 1
            continue
        if 0xC2 <= a and a <= 0xDF:
            if i + 1 >= n:
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            var b = Int(src[i + 1])
            if b < 0x80 or b > 0xBF:
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            chars += 1
            i += 2
        elif 0xE0 <= a and a <= 0xEF:
            if i + 2 >= n:
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            var b = Int(src[i + 1])
            var c = Int(src[i + 2])
            if b < 0x80 or b > 0xBF or c < 0x80 or c > 0xBF:
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            if (a == 0xE0 and b < 0xA0) or (a == 0xED and b > 0x9F):
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            chars += 1
            i += 3
        elif 0xF0 <= a and a <= 0xF4:
            if i + 3 >= n:
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            var b = Int(src[i + 1])
            var c = Int(src[i + 2])
            var d = Int(src[i + 3])
            if b < 0x80 or b > 0xBF or c < 0x80 or c > 0xBF or d < 0x80 or d > 0xBF:
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            if (a == 0xF0 and b < 0x90) or (a == 0xF4 and b > 0x8F):
                stats[11] = Int64(chars)
                stats[12] = 1
                return
            chars += 1
            i += 4
        else:
            stats[11] = Int64(chars)
            stats[12] = 1
            return
    stats[10] = 1
    stats[11] = Int64(chars)


def sjis_scan(src: BPtr, n: Int, stats: IPtr):
    var i = 0
    var pairs = 0
    var kana = 0
    while i < n:
        var a = Int(src[i])
        if a <= 0x7F:
            i += 1
        elif 0xA1 <= a and a <= 0xDF:
            kana += 1
            i += 1
        elif (0x81 <= a and a <= 0x9F) or (0xE0 <= a and a <= 0xFC):
            if i + 1 >= n:
                return
            var b = Int(src[i + 1])
            if not ((0x40 <= b and b <= 0x7E) or (0x80 <= b and b <= 0xFC)):
                return
            pairs += 1
            i += 2
        else:
            return
    stats[13] = 1
    stats[14] = Int64(pairs)
    stats[15] = Int64(kana)


def eucjp_scan(src: BPtr, n: Int, stats: IPtr):
    var i = 0
    var pairs = 0
    var kana = 0
    while i < n:
        var a = Int(src[i])
        if a <= 0x7F:
            i += 1
        elif a == 0x8E:
            if i + 1 >= n or Int(src[i + 1]) < 0xA1 or Int(src[i + 1]) > 0xDF:
                return
            else:
                kana += 1
                i += 2
        elif a == 0x8F:
            if i + 2 >= n:
                return
            var b = Int(src[i + 1])
            var c = Int(src[i + 2])
            if b < 0xA1 or b > 0xFE or c < 0xA1 or c > 0xFE:
                return
            else:
                pairs += 1
                i += 3
        elif 0xA1 <= a and a <= 0xFE:
            if i + 1 >= n:
                return
            var b = Int(src[i + 1])
            if b < 0xA1 or b > 0xFE:
                return
            else:
                pairs += 1
                i += 2
        else:
            return
    stats[16] = 1
    stats[17] = Int64(pairs)
    stats[18] = Int64(kana)


def gb18030_scan(src: BPtr, n: Int, stats: IPtr):
    var i = 0
    var pairs = 0
    var quads = 0
    while i < n:
        var a = Int(src[i])
        if a <= 0x7F:
            i += 1
        elif 0x81 <= a and a <= 0xFE:
            if i + 1 >= n:
                return
            var b = Int(src[i + 1])
            if (0x40 <= b and b <= 0x7E) or (0x80 <= b and b <= 0xFE):
                pairs += 1
                i += 2
            elif 0x30 <= b and b <= 0x39:
                if i + 3 >= n:
                    return
                var c = Int(src[i + 2])
                var d = Int(src[i + 3])
                if 0x81 <= c and c <= 0xFE and 0x30 <= d and d <= 0x39:
                    quads += 1
                    i += 4
                else:
                    return
            else:
                return
        else:
            return
    stats[19] = 1
    stats[20] = Int64(pairs)
    stats[21] = Int64(quads)


def big5_scan(src: BPtr, n: Int, stats: IPtr):
    var i = 0
    var pairs = 0
    var common = 0
    while i < n:
        var a = Int(src[i])
        if a <= 0x7F:
            i += 1
        elif 0x81 <= a and a <= 0xFE:
            if i + 1 >= n:
                return
            var b = Int(src[i + 1])
            if not ((0x40 <= b and b <= 0x7E) or (0xA1 <= b and b <= 0xFE)):
                return
            pairs += 1
            if a >= 0xA4:
                common += 1
            i += 2
        else:
            return
    stats[22] = 1
    stats[23] = Int64(pairs)
    stats[24] = Int64(common)


def korean_scan(src: BPtr, n: Int, stats: IPtr):
    var i = 0
    var euc_errors = 0
    var cp_errors = 0
    var euc_pairs = 0
    var cp_pairs = 0
    while i < n:
        var a = Int(src[i])
        if a <= 0x7F:
            i += 1
            continue
        if i + 1 >= n:
            euc_errors += 1
            cp_errors += 1
            break
        var b = Int(src[i + 1])
        var euc_ok = 0xA1 <= a and a <= 0xFE and 0xA1 <= b and b <= 0xFE
        var cp_trail = (0x41 <= b and b <= 0x5A) or (0x61 <= b and b <= 0x7A) or (0x81 <= b and b <= 0xFE)
        var cp_ok = 0x81 <= a and a <= 0xFE and cp_trail
        if euc_ok:
            euc_pairs += 1
        else:
            euc_errors += 1
        if cp_ok:
            cp_pairs += 1
        else:
            cp_errors += 1
        if euc_ok or cp_ok:
            i += 2
        else:
            i += 1
        if euc_errors != 0 and cp_errors != 0:
            return
    stats[25] = Int64(1 if euc_errors == 0 else 0)
    stats[26] = Int64(euc_pairs)
    stats[27] = Int64(1 if cp_errors == 0 else 0)
    stats[28] = Int64(cp_pairs)


@export("chd_scan")
def chd_scan(
    src_addr: Int, n: Int, stats_addr: Int, hist_addr: Int
) abi("C") -> Int64:
    # Do not construct UnsafePointer values until every address and the length
    # have been checked.  The Python wrapper treats a nonzero result as fatal.
    if src_addr == 0 or stats_addr == 0 or hist_addr == 0 or n < 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var stats = IPtr(unsafe_from_address=stats_addr)
    var hist = IPtr(unsafe_from_address=hist_addr)
    comptime W = simd_width_of[DType.float64]()
    var zeroes = SIMD[DType.int64, W](0)
    var i = 0
    while i + W <= 32:
        stats.store(i, zeroes)
        i += W
    while i < 32:
        stats[i] = 0
        i += 1
    i = 0
    while i + W <= 256:
        hist.store(i, zeroes)
        i += W
    while i < 256:
        hist[i] = 0
        i += 1
    stats[0] = Int64(n)

    i = 0
    while i + W <= n:
        var values = src.load[width=W](i)
        stats[1] += values.ge(
            SIMD[DType.uint8, W](0x80)
        ).cast[DType.int64]().reduce_add()
        stats[2] += values.eq(
            SIMD[DType.uint8, W](0)
        ).cast[DType.int64]().reduce_add()
        var controls = (
            values.lt(SIMD[DType.uint8, W](0x20))
            & values.ne(SIMD[DType.uint8, W](9))
            & values.ne(SIMD[DType.uint8, W](10))
            & values.ne(SIMD[DType.uint8, W](12))
            & values.ne(SIMD[DType.uint8, W](13))
            & values.ne(SIMD[DType.uint8, W](0))
        )
        stats[3] += controls.cast[DType.int64]().reduce_add()
        stats[4] += (
            values.ge(SIMD[DType.uint8, W](0x80))
            & values.le(SIMD[DType.uint8, W](0x9F))
        ).cast[DType.int64]().reduce_add()
        stats[5] += (
            values.ge(SIMD[DType.uint8, W](0x20))
            & values.le(SIMD[DType.uint8, W](0x7E))
        ).cast[DType.int64]().reduce_add()
        comptime for lane in range(Int(W)):
            var b = Int(values[lane])
            hist[b] += 1
            if b == 0:
                stats[6 + (i + lane) % 4] += 1
        i += W
    while i < n:
        var b = Int(src[i])
        hist[b] += 1
        if b >= 0x80:
            stats[1] += 1
        if b == 0:
            stats[2] += 1
            stats[6 + i % 4] += 1
        elif b < 0x20 and b != 9 and b != 10 and b != 12 and b != 13:
            stats[3] += 1
        if 0x80 <= b and b <= 0x9F:
            stats[4] += 1
        if 0x20 <= b and b <= 0x7E:
            stats[5] += 1
        i += 1

    utf8_scan(src, n, stats)
    sjis_scan(src, n, stats)
    eucjp_scan(src, n, stats)
    gb18030_scan(src, n, stats)
    big5_scan(src, n, stats)
    korean_scan(src, n, stats)
    return 0
