# π — high-performance, verifiable arbitrary-precision π computation

This project computes **π to an arbitrary number of decimal places** using the
**Chudnovsky formula with binary splitting**, parallelised across all available
cores, with a **subquadratic** binary→decimal converter, low memory usage, and
**three independent verification layers**:

1. BBP hexadecimal digit checks (against the main computation's *binary* value),
2. a modular check of the decimal **conversion** against the exact truncated
   integer, and
3. externally-sourced decimal checkpoints.

Verification is **on by default**; use `--no-verify` to skip it.

It was developed and benchmarked on:

| Item | Value |
|------|-------|
| CPU | Intel Core i9-14900KS (24 cores / 32 threads) |
| RAM | 32 GB |
| Arch | x86_64, Linux (WSL2) |
| Toolchain | Rust 1.97.1 (stable) |
| Big integers | GMP via `rug` |

---

## Build

Requires a system GMP development environment (`libgmp-dev`), a C toolchain
(GMP itself may be built by `gmp-mpfr-sys` if not found), and a stable Rust
toolchain.

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

`-C target-cpu=native` enables CPU-specific instruction sets (AVX2/AVX-512),
which substantially speeds up GMP. The release profile is configured with
`lto = "fat"`, `codegen-units = 1`, and `panic = "abort"` in `Cargo.toml`.

```bash
cargo test
```

## Usage

```
pi --digits N [--threads T] [--output PATH] [--no-verify]
   [--checkpoints PATH] [--bench] [--guard G]
```

| Flag | Description |
|------|-------------|
| `--digits N` | Number of decimal places after the `"3."` (required). |
| `--threads T` | Worker threads (default: all logical cores). |
| `--output PATH` | Output file (default: `pi.txt`). `"3."` followed by exactly N digits, no newlines or spaces. |
| `--no-verify` | Skip verification (verification is **on by default**). |
| `--checkpoints PATH` | File of externally-sourced decimal checkpoints, one per line: `<position> <digits>` (e.g. `1000000 1`). |
| `--bench` | Per-phase timing and peak memory (`VmHWM`). |
| `--guard G` | Internal guard digits beyond N (default: 32). |

### Examples

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release

./target/release/pi --digits 1000000     --output pi_1m.txt   --bench
./target/release/pi --digits 10000000    --output pi_10m.txt  --bench
./target/release/pi --digits 100000000   --output pi_100m.txt --bench

# with external decimal checkpoints (verification already on by default)
./target/release/pi --digits 1000000 --checkpoints checkpoints.example.txt --bench

# compute only, no verification
./target/release/pi --digits 100000000 --no-verify --output pi_100m.txt
```

A sample `checkpoints.example.txt`:

```
100001 41260
1000000 1
5000000 5
10000000 725915133
```

Running `pi --digits 1000000 --checkpoints checkpoints.example.txt` reports the
two in-range checkpoints as `PASS` and the two beyond-range checkpoints as
`SKIPPED` (with a warning), and **exits 0**.

### Digit indexing convention

**Position 1 is the first digit after the decimal point** (the `1` in
`3.14159…`). Position 2 is `4`, etc. This is enforced by a unit test
(`chudnovsky::tests::digit_indexing_is_one_based`) that catches an off-by-one.
The first 50 hardcoded digits use this convention.

---

## Algorithm overview

### Chudnovsky formula

```
1/π = 12/√C3 · Σ_k (-1)^k (6k)! (A + B·k) / ((3k)! (k!)^3 C3^k)
```

with `A = 13591409`, `B = 545140134`, `C3 = 640320³ = 2⁶·10005`. Rearranging:

```
π = 426880·√10005 · Q / T
```

where `T/Q = Σ s_k` is the partial sum from binary splitting.

### Binary splitting

For a range `[a,b)` we maintain integer triples `(P,Q,T)` with base case
`p_k = (6k−5)(2k−1)(6k−1)`, `q_k = k³·(C3/24)`, and the merge

```
Q(a,b) = Q(a,m)·Q(m,b)
T(a,b) = T(a,m)·Q(m,b) + P(a,m)·T(m,b)
P(a,b) = P(a,m)·P(m,b)          (only when needed)
```

The numerator `P` of the whole range is never needed: π depends only on `Q,T`,
and a merge only ever needs the `P` of its **left** child. The right spine of
the recursion therefore **skips computing `P` entirely**, saving the single
largest multiplication at the root merge.

> **Note:** the leaf numerator `(6k−5)(2k−1)(6k−1)` exceeds `u64` for
> `k ≳ 635,000` and must be multiplied as big integers (see the overflow bug in
> [`PROGRESS.md`](PROGRESS.md)).

### Scaling to the binary fixed-point value (working precision)

`Q` and `T` are enormous — at 100M decimal digits `Q` has ~250M decimal digits —
far more than the output needs, so scaling them directly does ~2.5× more work
than necessary. Instead we first **truncate** `Q` and `T` and compute π as a
binary fixed-point integer.

Let `D = digits + guard`, choose `W` bits with `W ≥ D·log2(10) + 192` and `W` a
multiple of 4, and let `Wm = W + 128`. Shift both `Q` and `T` right by
`a = max(0, max(bitlen Q, bitlen T) − Wm)`:

```
Qh = Q >> a,   Th = T >> a
```

With `Q = Qh·2^a + ql`, `T = Th·2^a + tl` (`0 ≤ ql,tl < 2^a`),

```
|Qh/Th − Q/T| < 2^(−Wm)
```

(numerator `|Qh·tl − Th·ql| < 2^Wm`, denominator `≥ Th² ≈ 2^(2·Wm)`). So the
computed π differs from the true π by `< 426880·√10005·2^(−Wm)`; scaled by `2^W`
this is `< 2^(25.3 + W − Wm) = 2^(−102.7)`, under one bit. We compute

```
G = isqrt(426880²·10005·Qh²)     ≈ 426880·√10005·Qh
M = floor(G · 2^W / Th)          ≈ π · 2^W      (|M − π·2^W| < 2)
```

`M` is the main computation's **binary** result. Its hex digits are simply its
nibbles, so **no second division is needed** for the BBP checks.

### Decimal digits

Using `10^D = 2^D·5^D`:

```
floor(π·10^D) = floor(M·10^D / 2^W) = (M·5^D) >> (W − D)
```

The error in `M` is `< 2` bits, so the error in `π·10^D` is
`< 2·10^D/2^W ≤ 2^(1−192) < 2^(−190)`, which cannot change the floor. The
result is exactly `floor(π·10^D)`; we then drop the `guard` digits by dividing
by `10^guard` to get the correctly **truncated** digit string (never rounded).

### Guard digits and term count

The number of Chudnovsky terms is `N = ceil(D / 14.181647…) + 16`; each term
contributes ≈14.18 digits and the extra 16 terms give a >200-digit margin, so
the last kept digit is never affected by series truncation. `guard` defaults to
32.

---

## Parallelisation strategy

The binary-splitting merge tree is parallelised with a **rayon `join`
work-stealing recursion** over the whole tree (down to a leaf threshold of
`≈ n/(4·threads)` terms), so that *all* levels — including the large merges near
the top, which a fixed-segment scheme would serialise — are scheduled across all
cores. The merge is exact and associative, so the result is deterministic
regardless of scheduling (verified identical across 1 vs 32 threads).

An earlier design split the range into fixed segments computed on
manually-spawned threads and merged them with a **serial left-fold**; that
serialised the largest multiplications and is the reason the series was ~8×
slower before. The base-conversion recursion and the BBP sums are likewise
parallelised.

### Bottlenecks at 100M

- **Big-integer multiplication / division**: the binary scaling (`isqrt` +
  division) and the base conversion each cost several seconds; the series ~14 s.
- **Base conversion**: kept subquadratic (`O(M(n)·log n)`) but still ~7 s.
- **Memory**: the dominant live objects are the fixed-point value `M`, the
  truncated integer, and the digit buffer (all ~output-size); large temporaries
  are dropped as soon as possible and output is streamed in 1 MiB chunks.

---

## Verification design

All three layers run by default after computation.

### A. BBP hexadecimal run checks

The Bailey–Borwein–Plouffe formula gives hexadecimal digits of π at an
arbitrary position without computing preceding digits. For each selected
position we compare a **run of up to 8 hex digits** (the run length is chosen so
`frac_error_bound(n)·16^k ≤ 0.5`, and clamped so the whole run lies in range)
against the nibbles of the main computation's binary result `M`.

Because the BBP sum is accumulated in `f64`, a run whose value lies within the
error bound of a digit boundary cannot be trusted: such a run is reported
`INCONCLUSIVE`, never PASS/FAIL. The error bound is documented in `bbp.rs`
(`√(4n)·2^-53·16`, floor `1e-12`) and calibrated against a 256-bit computation
(`examples/bbp_prec.rs`): the measured error at `n ≈ 8.3M` was `2.3e-13` versus
a bound of `2.0e-11`. Positions are chosen across the whole range (early,
interior fractions, the last-decimal-digit hex position, and near the end).

### B. Modular conversion check

BBP validates only the *binary* value; a bug in the custom base converter would
produce localised wrong digits that BBP and sparse checkpoints could miss. We
therefore reduce both the exact truncated integer `floor(π·10^digits)` and the
rendered decimal string modulo three large primes (2⁶¹−1, 2⁶⁴−59, 2³¹−1) and
compare. The string evaluation uses **Horner's method, parallelised by chunking
the digits** and combining with `10^chunk mod p`. Each prime is reported
PASS/FAIL. A unit test corrupts a single middle digit and confirms this check
FAILS while the BBP checks still pass.

### C. Decimal checkpoints

The final output is compared against `--checkpoints`, plus the hardcoded first
50 digits as a sanity check. Checkpoints beyond the computed range are
`SKIPPED` (not failures). Any `FAIL` makes the process exit nonzero.

> The BBP layer caught a real bug during development (a `u64` overflow in the
> binary-splitting leaf). It failed at 10M digits, then passed after the fix.

---

## Memory

Peak memory is minimised by:
- truncating `Q`/`T` to the working precision before the expensive scaling;
- freeing P/Q/T intermediates as soon as they are merged;
- rendering the decimal digits directly with the leading `"3"` dropped (no
  second full-size copy of the digit buffer);
- extracting hex nibbles directly from `M` (no full hex string is built);
- streaming output to disk in 1 MiB chunks.

---

## Performance: before vs after this review

All times are wall-clock release timings on the reference machine with
verification enabled. "Scaling" merges the binary scaling and decimal scaling
phases.

| Digits | | Series | Scaling | Conv. | Verify | Total | Peak mem |
|--------|--|--------|---------|-------|--------|-------|----------|
| 1,000,000 | before | 0.060 s | 0.083 s | 0.034 s | 0.139 s | 0.317 s | 38.3 MB |
| 1,000,000 | after  | 0.052 s | 0.047 s | 0.039 s | 0.266 s | 0.406 s | 27.9 MB |
| 10,000,000 | before | 0.94 s | 1.42 s | 0.51 s | 0.81 s | 3.68 s | 356 MB |
| 10,000,000 | after  | 0.88 s | 0.69 s | 0.48 s | 1.44 s | 3.50 s | 204 MB |
| 100,000,000 | before | 15.3 s | 19.5 s | 7.7 s | 9.1 s | ~51.6 s¹ | 2.53 GB |
| 100,000,000 | after  | 13.6 s | 9.4 s | 7.3 s | 14.8 s | 45.2 s | 1.67 GB |

¹ Before, 100M verification-on wall clock was **58.1 s** (`/usr/bin/time -v`);
the in-process phase sum was ~51.6 s. After, it is **45.2 s**.

Observations:

- **Scaling** dropped from 19.5 s to 9.4 s at 100M (the review's ~2.5×
  estimate), and **peak memory** from 2.53 GB to 1.67 GB, because `Q`/`T` are
  truncated to the working precision and the second (hex) division is gone.
- **Series** improved ~10% by skipping `P` on the right spine (15.3 s → 13.6 s).
- **Verification** now costs more (more positions, multi-digit runs, and the
  modular conversion check) — that is the intended trade-off for stronger,
  independent checks. The **conversion check itself costs 0.156 s at 100M**, a
  ~2% fraction of the 7.3 s base conversion.
- At 1M the total is slightly higher than before because the stronger
  verification (≈15 BBP positions plus the modular check) dominates the tiny
  computation.

The 1M / 10M / 100M outputs are **byte-identical** to the previously verified
files:

| Digits | SHA-256 |
|--------|---------|
| 1,000,000 | `dd382ef6a0c1e8d920fb72f482d74826251ab97709520bc24f913cd8eb5fc839` |
| 10,000,000 | `46059c61a4de67d6c916fa958168789da324a03ee8a85c30e9ca292c3712eb25` |
| 100,000,000 | `3dd4bc2392f543a8b4dff7634fb18da86e36c88cf35acb7b3fe7b70d864fa823` |

---

## Known limitations

- **f64 BBP**: runs near a digit boundary are `INCONCLUSIVE`; the run length is
  capped at 8 and the error bound is a calibrated statistical bound (~90× the
  measured error), not a strict worst case. For very large positions, expect
  more `INCONCLUSIVE` results.
- The first 50 digits only are hardcoded; all other confidence comes from BBP,
  the modular conversion check, and user checkpoints.
- The program truncates (never rounds), as required.
- Memory scales with output size.

---

## Diagnostics (examples)

- `examples/diag.rs` — compare main hex digits against BBP across a range.
- `examples/bbp_prec.rs` — f64 vs 256-bit BBP at a position (error calibration).
- `examples/convert_test.rs` — validate the converter by round-trip.

## Project layout

```
src/
  lib.rs        crate root / re-exports
  main.rs       CLI entry point
  chudnovsky.rs Chudnovsky binary splitting + binary/decimal scaling
  bbp.rs        BBP hex digit extraction (+ u128 modpow, error bound)
  convert.rs    subquadratic binary→decimal conversion
  verify.rs     three verification layers + checkpoints
  output.rs     streaming file output
tests/
  integration.rs  N = 10,000 end-to-end test
examples/
  diag.rs, bbp_prec.rs, convert_test.rs
README.md
PROGRESS.md
```
