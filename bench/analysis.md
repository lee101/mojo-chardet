# Kernel analysis and the GPU gate

This file is the input to the GPU gate in the acceleration rules. The gate
admits kernels above roughly 2 flops per byte; everything below is an explicit
"CPU only: memory-bound" decision.

All numbers below were measured on this box, not estimated:

- Intel(R) Xeon(R) CPU E5-2697 v4 (2 sockets, 18 cores each), Linux
  6.8.0-147-generic, Python 3.13.14, chardet 7.4.3.
- NVIDIA GeForce RTX 5090, 5500 MiB free at the time of this analysis.
- Mojo 1.2.0.dev2026092605, `mojo build` at its default `-O3`.
- The scan is one C ABI export, `chd_scan` in `src/chardet.mojo`.

## What the scan does

`chd_scan` runs two passes over every byte of input:

1. **Statistics pass.** Half-byte control counts, NUL distribution over 4 byte
   positions, plus a byte histogram (`hist[b] += 1` with a loop-carried
   dependency on a 256-entry table that stays in L1).
2. **Six finite-state validators.** UTF-8, Shift-JIS, EUC-JP, GB18030, Big5 and
   a combined EUC-KR/CP949 scan. Each one is serial: the byte value determines
   both the next position and the next state, so a validator cannot be
   parallelised within itself.

## Where the time goes

End-to-end `detect()` is dominated by the native scan; the Python policy layer
is a negligible fixed cost and is not a GPU candidate at all. Figures are the
best of a single run (the README baseline table uses medians).

| workload | bytes | native `chd_scan` | policy layer | total |
|---|---:|---:|---:|---:|
| detect ASCII, full | 5242914 | 37.42 ms | 0.09 ms | 37.51 ms |
| detect UTF-8, full | 5242897 | 29.18 ms | ~0 | 28.70 ms |
| detect cp932, full | 2097168 | 66.65 ms | 3.41 ms | 70.07 ms |
| detect Windows-1251, full | 2097154 | 11.08 ms | 0.00 ms | 11.08 ms |
| detect UTF-8, default cap | 200000 | 1.050 ms | ~0 | 1.024 ms |

Native scan throughput on a 200 kB buffer, isolated from `detect()`:

| workload | scan time | throughput |
|---|---:|---:|
| ASCII | 1161 us | 172 MB/s |
| UTF-8 | 1029 us | 194 MB/s |
| cp932 | 1236 us | 162 MB/s |
| Windows-1251 | 610 us | 328 MB/s |

A stream `copy` (read+write) on 256 MiB of fresh pages in the same environment
sustains 17.6 GB/s, so memory alone would cost 131 us per 2 MiB. The scan takes
610-1236 us for 200 kB, which is 8-9x slower than memory per byte. The kernel is
compute-bound, not memory-bound: the GPU gate is about whether the compute can
be moved, not whether it is cheap on the CPU.

## Instructions per byte

Counting executed instructions exactly was not possible here:
`callgrind_control` toggling from an `os.system` child never took effect, and
`perf` is locked down with `perf_event_paranoid=4`. The figure below is derived
from the shipped object code instead, and is order-of-magnitude:

- `objdump -d dist/libmojo-chardet.so` gives 660 instructions in `chd_scan`
  including the six validators. The SIMD statistics loop is still vectorised,
  but on this compiler the validators are emitted as scalar code.
- Per byte, the steady state is one statistics SIMD block of 42 instructions
  per 8 bytes (`42/8 = 5.25` instructions per byte), plus six independent
  validators each stepping one byte per iteration of an ASCII fast path of
  roughly 8-10 instructions (`6 * 9 = 54`). That is about **59 executed
  instructions per byte**, or 59 "flops" per byte if every instruction counts
  as one op, well above the 2 flops/byte gate (about 30x).

So the kernel is above the gate. It is compute-bound and instruction-heavy,
which is exactly the shape that benefits from a GPU, and the gate alone does not
exclude it.

So the kernel is above the gate. It is compute-bound and instruction-heavy,
which is exactly the shape that benefits from a GPU, and the gate alone does not
exclude it.

## Why it is still CPU only: the callers never pass enough data

The GPU has to beat `chd_scan` end to end, not just the arithmetic. That means a
`DeviceContext`, one upload, one launch, one download of results, and
`ctx.synchronize()` on **every call**, because `detect()` is a single-shot,
stateless entry point. There is no streaming/monitored API in this port to
amortise a device across calls.

Serial-cost reality on this box (30 GB/s PCIe 4 x16 effective, so 0.067 ns/byte
for one upload + one download):

| workload | CPU ns/byte | PCIe round trip alone |
|---|---:|---:|
| ASCII | 5.80 | 0.067 ns/byte (1% of CPU) |
| UTF-8 | 5.15 | 0.067 ns/byte (1%) |
| cp932 | 6.18 | 0.067 ns/byte (1%) |
| Windows-1251 | 3.05 | 0.067 ns/byte (2%) |

Transfer is cheap *relative to the current CPU kernel*: an infinitely fast GPU
kernel would cap out at 45x-93x on paper. What kills it is the serial part that
does not scale with GPU width:

- one device allocation (under the 2 GB limit) and the host-side driver call
  for `enqueue_function`, both serial per call;
- the two-pass dependency: the statistics/validator results from the first
  launch determine the second, so a faithful port is two launches with a
  synchronisation between them, plus a three-counter reduction (which needs its
  own kernel and a second launch). That is three or four serial launches per
  `detect()` call, before any data movement starts.

At the default inspection cap (`DEFAULT_MAX_BYTES = 200_000`) the whole native
scan on the CPU takes **1.05 ms**, with the GPU-side serial work still to be
added. Even one sub-millisecond GPU launch plus a synchronisation plus a
download is a wash at that size; the per-call serial overhead is a permanent
floor that a 200 kB input cannot amortise. At the bench sizes (2 MiB, 5 MiB),
a highly optimised kernel might win, but those sizes are not what the default
path passes: only explicit `max_bytes=` callers reach them. A GPU path wins only
above a size that default callers never pass, and loses on the default path
every time. The gate is not about the peak; it is about the sizes real callers
use.

Separately, the GPU host API is **not importable from this environment as
written**. `max` is not a dependency in `pixi.toml`, and
`.pixi/envs/default/lib/mojo` contains only `std.mojoc`. `from max.gpu.host
import DeviceContext` fails with `unable to locate module 'max'`; adding the
pinned `max ==26.7.0.dev2026092605` package would be required first (the same
trap as `MOJO_NOTES.md` §5b). That is a prerequisite, not a reason to give up,
and it is trivially satisfiable; it is the per-call serial floor and the
real-caller sizes above that decide the answer.

## Conclusion

The kernel is well above the 2 flops/byte gate (roughly 59 instructions per
byte), and it is compute-bound rather than bandwidth-bound, so the usual
"nothing to gain on a GPU" reasoning does not apply. The blocker is the shape of
the call, not the arithmetic:

- a GPU `chd_scan` must start a `DeviceContext` and synchronise at least twice
  (three or four launches) on every `detect()` call;
- the default path is 200 kB / ~1.05 ms, where the serial launch cost is
  comparable to or larger than the entire CPU scan;
- the sizes where a GPU might win (2-5 MiB) are opt-in `max_bytes=` callers,
  not what `detect()` receives by default.

Decision: no GPU path is shipped. The CPU path stays the implementation, and a
GPU candidate is only worth revisiting behind a streaming or batched API where a
device can be kept alive across many buffers, or after a change to the default
inspection cap that puts thousands of bytes per call into the multi-megabyte
range.
