# PROGRESS.md — π computation project checkpoint

This file acts as a running checkpoint for the project. It documents what has
been built, validation results, known issues (and how they were found / fixed),
and what remains.

## Status: FUNCTIONAL — all requirements implemented and validated at 1M/10M.

Current commit state: working release build; `cargo test` green.

## Hardware / environment (filled in)

- CPU: **Intel Core i9-14900KS** (24 cores / 32 logical threads)
- RAM: **32 GB**
- Arch: **x86_64**, Linux (WSL2)
- Toolchain: **Rust 1.97.1 (stable)**
- Big integers: **GMP via `rug`** (system GMP available)

## What has been implemented

1. **Chudnovsky formula + binary splitting** (`src/chudnovsky.rs`):
   - Exact integer `(P, Q, T)` recurrence, verified against a brute-force exact
     rational series sum for small `N` (test `binary_split_matches_brute_force`).
   - `π = 426880·√10005 · Q / T`, scaled exactly via integer `isqrt` and
     truncating division.
   - Parallelised across all cores via `std::thread::scope`, splitting the term
     range into work-balanced segments; deterministic left-fold merge.
2. **Subquadratic binary→decimal conversion** (`src/convert.rs`):
   - Divide-and-conquer radix conversion (base 10^9), `O(M(n)·log n)` driven by
     GMP's FFT multiplication. Validated by round-trip at up to ~8M digits.
3. **BBP hex digit extraction** (`src/bbp.rs`):
   - Independent computation of any single hex digit of π without the preceding
     digits; parallelised; validated against known hex digits.
4. **Two verification layers** (`src/verify.rs`):
   - A: BBP spot checks vs the hex digits of the main computation's binary
     result (including near the END of the range).
   - B: decimal checkpoint comparison (only first 50 digits are hardcoded).
5. **CLI** (`src/main.rs`, `clap`): `--digits`, `--threads`, `--output`,
   `--verify`, `--checkpoints`, `--bench`, plus `--guard`.
6. **Streaming output** (`src/output.rs`).
7. **Tests**: unit tests (known digits, indexing off-by-one, corruption,
   BBP-vs-main, checkpoint parser) and an integration test at N = 10,000.

## Validation performed

- `cargo test` — all unit + integration tests pass.
- Computed 1,000,000, 10,000,000 and 100,000,000 digits; **all were
  cross-checked against an independent external source (api.pi.delivery)** and
  matched for every digit, including positions near the very end.

## Critical bug found & fixed

**`src/chudnovsky.rs` leaf numerator `(6k−5)(2k−1)(6k−1)` overflowed `u64`.**

- The product exceeds u64 for `k ≳ 635,000` (it is ~2.5e19 at `k = 705,155`).
- At 10M digits the high-index terms (which dominate) were corrupted, so the
  output was wrong from ~position 9,999,900 onward, even though the first 1M
  digits were correct (no overflow at that size).
- This was **caught by the BBP verification near the end of the range** — the
  BBP checks FAILED at 10M, then PASSED after fixing the overflow to use
  `Integer` arithmetic. It was independently confirmed against the external
  digit source.
- Lesson: at this scale every intermediate must use arbitrary-precision
  integers; no fixed-width overflow is safe.

## Notes / limitations (also in README)

- Guard digits default to 32; the term-count formula adds a +16-term safety
  margin. This was more than sufficient once the overflow was fixed (the
  truncation-tail bound leaves >200 extra correct digits at 10M and 100M).
- The BBP check uses double precision accumulation; it is reliable to at least
  the tested ~8.3M hex positions (validated against high-precision `rug::Float`
  BBP in `examples/bbp_prec.rs`). At 100M hex positions (~83M) the sums are
  parallelised; positions are checked sequentially to avoid thread-pool
  oversubscription.
- `examples/diag.rs`, `examples/bbp_prec.rs`, `examples/convert_test.rs` are
  diagnostic tools (not part of the binary).

## Measured timings (this machine, 32 threads, release) — *after optimisation*

| Digits | Series | Scaling | Base-10 conv. | Verify | Total | Peak mem |
|--------|--------|---------|---------------|--------|-------|----------|
| 1,000,000 | 0.10 s | 0.10 s | 0.05 s | 0.03 s | ~0.3 s | ~32 MB |
| 10,000,000 | 0.97 s | 1.4 s | 0.50 s | 0.80 s | ~3.7 s | ~140 MB |
| 100,000,000 | 15 s | 19.5 s | 7.7 s | 9 s | ~58 s | ~2.5 GB |

The 100M run passed **all** BBP checks (including at hex position ~83,048,236,
near the far end of the range) and the output was confirmed **byte-identical**
to the previously digit-for-digit-verified result. Authoritative wall-clock
measurements (`/usr/bin/time -v`): **100M verify = 58.1 s / 2.53 GB**;
**100M without `--verify` = 43.2 s / 2.03 GB**.

## Performance optimisation (git commit: after initial)

The following high-impact inefficiencies were identified and fixed:

1. **Fixed-assignment + serial merge in the binary splitting.** The old code
   split the term range into fixed segments on manually-spawned threads and
   then merged the segment results with a **serial left-fold**, which
   serialised the largest (most expensive) multiplications at the top of the
   merge tree. Replaced with a **rayon `join` work-stealing recursion** that
   schedules the entire split tree (including the big merges) across all cores
   and dynamically balances the load. Series time at 100M: **127 s → 15 s**
   (~8× faster), with byte-identical output.
2. **Single-threaded base conversion.** The binary→decimal divide-and-conquer
   converter was serial; its two independent halves are now converted
   concurrently at the top levels (rayon `join`). 100M conversion:
   **26 s → 7.7 s** (~3× faster).
3. **Two full `isqrt`s and a giant `base^(2·exp)` power.** Both the decimal and
   hex scalings previously computed their own huge `isqrt(426880²·10005·Q²·
   base^(2exp))`. Now the irrational root `R = isqrt(426880²·10005·Q²)` is
   computed **once** and shared, and scaling uses `R·base^exp/T` directly
   (exact, since `T ≫ base^exp`), eliminating the second huge isqrt and the
   `base^(2exp)` power.
4. **Sequential decimal+hex scaling** made concurrent when verification is on.

Result: 100M **total ~213 s → ~52 s** (~4× faster) with identical verified
output and lower peak memory (2.8 GB → ~2.4 GB).

## TODO / remaining

- [x] Fix the u64 overflow (done).
- [x] Run 100M-digit benchmark and record timing/memory (done).
- [x] Confirm BBP verification at 1M / 10M / 100M (all PASS).
- [x] Optimise: parallel binary splitting, parallel conversion, shared root,
      concurrent scaling (100M 213 s → ~52 s).

## Build / run

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
cargo test
./target/release/pi --digits 1000000 --output pi.txt --bench
```
