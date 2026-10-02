# π — high-performance, verifiable arbitrary-precision π computation

This project computes **π to an arbitrary number of decimal places** using the
**Chudnovsky formula with binary splitting**, parallelised across all available
cores, with a **subquadratic** binary→decimal converter, low memory usage, and
**two independent verification layers** (BBP hexadecimal spot-checks and
user-supplied decimal checkpoints).

It is designed to run correctly, verify itself, and be fast. It was developed
and benchmarked on:

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

To test:

```bash
cargo test
```

## Usage

```
pi --digits N [--threads T] [--output PATH] [--verify]
   [--checkpoints PATH] [--bench] [--guard G]
```

| Flag | Description |
|------|-------------|
| `--digits N` | Number of decimal places after the `"3."` (required). |
| `--threads T` | Worker threads (default: all logical cores). |
| `--output PATH` | Output file (default: `pi.txt`). Format is `"3."` followed by exactly N digits, no newlines or spaces. |
| `--verify` | Run verification after computation (BBP spot-checks + decimal checkpoints). **Off by default** — just `pi --digits N` computes and writes the digits. |
| `--checkpoints PATH` | File of externally-sourced decimal checkpoints, one per line: `<position> <digits>` (e.g. `1000000 1`). Only used when `--verify` is given. |
| `--bench` | Print per-phase timing and peak memory (`VmHWM`). |
| `--guard G` | Internal guard digits beyond N (default: 32). |

### Examples

```bash
# 1,000,000 digits
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/pi --digits 1000000 --output pi_1m.txt --bench

# 10,000,000 digits
./target/release/pi --digits 10000000 --output pi_10m.txt --bench

# 100,000,000 digits
./target/release/pi --digits 100000000 --output pi_100m.txt --bench

# With external decimal checkpoints
./target/release/pi --digits 1000000 --checkpoints checkpoints.txt --bench
```

A sample `checkpoints.txt`:

```
1000000 1
1000001 4
2000000 14159
```

### Digit indexing convention

**Position 1 is the first digit after the decimal point** (the `1` in
`3.14159…`). Position 2 is `4`, position 3 is `1`, etc. This convention is used
everywhere in the program and is enforced by a unit test
(`chudnovsky::tests::digit_indexing_is_one_based`) that would catch an
off-by-one. The first 50 hardcoded digits are interpreted under this
convention.

---

## Algorithm overview

### Chudnovsky formula

The computation uses the Chudnovsky series:

```
1/π = 12/√C3 · Σ_k (-1)^k (6k)! (A + B·k) / ((3k)! (k!)^3 C3^k)
```

with `A = 13591409`, `B = 545140134`, `C3 = 640320³`. Rearranged (since
`C3 = 2⁶·10005`), this becomes:

```
π = 426880·√10005 · Q / T
```

where `T/Q = Σ s_k` is the partial sum obtained by binary splitting.

### Binary splitting

Instead of summing terms one-by-one (quadratic), we compute the partial sum via
an exact integer divide-and-conquer recurrence over the term range. For a
range `[a,b)` we maintain integer triples `(P,Q,T)` with the base case
`p_k = (6k−5)(2k−1)(6k−1)`, `q_k = k³·(C3/24)` (after cancelling common
factors), and the merge

```
P(a,b) = P(a,m)·P(m,b)
Q(a,b) = Q(a,m)·Q(m,b)
T(a,b) = T(a,m)·Q(m,b) + P(a,m)·T(m,b)
```

This is exact integer arithmetic; the number of big-integer multiplications is
roughly linear in the number of terms, rather than quadratic.

> **Note:** every product above is a full arbitrary-precision integer. In
> particular the leaf numerator `(6k−5)(2k−1)(6k−1)` **exceeds 64 bits** for
> `k ≳ 635,000` and must be multiplied as big integers. (See the overflow bug
> documented in [`PROGRESS.md`](PROGRESS.md).)

### Scaling to decimal digits

`π = 426880·√10005·Q/T`. To obtain digits we compute, exactly,

```
floor(π · 10^(digits+guard))
  = floor( isqrt(426880²·10005·Q²·10^(2·(digits+guard))) / T )
```

`isqrt` is floor(sqrt) via GMP (exact), and the division is a truncating
integer division. Because `T` is a positive integer,
`floor(floor(√M)/T) = floor(√M/T)` exactly, so this is the exact floor of
`π·10^D`. We then drop the guard digits (divide by `10^guard`) to obtain the
correctly **truncated** digit string — never rounded. A separate
`floor(π·16^hex_len)` is computed (the `hex_scaled` value) solely for the BBP
verification.

### Guard digits

We compute `D = digits + guard` digits internally and truncate to `digits`. The
number of Chudnovsky terms is chosen so that the series truncation tail is far
below `10^-D`: `N = ceil(D / 14.181647…) + 16`. Each term contributes ≈14.18
decimal digits, and the extra 16 terms give a >200-digit safety margin beyond
the requested range, so the last kept digit is never affected by truncation.
`guard` defaults to 32 (configurable via `--guard`).

---

## Parallelisation strategy

The binary splitting is a tree of independent merges and is embarrassingly
parallel in structure, but the work per term grows with the term index (the
big-integer sizes grow linearly in `k`). Therefore:

1. The term range `[0, N)` is partitioned into contiguous **segments sized so
   the total work is balanced**, not the term count. Because work ∝
   `∫k·dk = (b²−a²)/2`, segment boundaries are chosen so each segment gets an
   equal share of `b²−a²` — this makes the high-index (expensive) segments
   shorter in term count. This is implemented in
   `balanced_partition` in `src/chudnovsky.rs`.
2. Each segment is computed **independently on its own thread** via
   `std::thread::scope` (scoped threads, no `'static` requirement), each doing a
   sequential recursive binary split.
3. The segment results are merged in a **deterministic left-fold** so the output
   does not depend on scheduling. (Verified identical across 1 vs 32 threads.)

The BBP digit extraction and the BBP verification checks are also parallelised
with `rayon` across the available cores.

### Expected performance bottlenecks at 100M digits

- **Big-integer multiplication** dominates. GMP's FFT multiplication is
  subquadratic, but the numbers involved are enormous (hundreds of millions of
  bits), so the top levels of the binary-splitting tree and the final
  `floor(π·10^D)` sqrt/division are the most expensive single operations.
- **Binary→decimal conversion** is a major cost; the divide-and-conquer method
  keeps it `O(M(n)·log n)` rather than quadratic, but it is still substantial.
- **memory**: holding the Q/T intermediates, `Q²`, the `10^(2D)` power, the
  scaled integer, and the digit buffer. The large temporary integers are
  dropped as soon as they are merged (intermediates are consumed/freed during
  the merge), and output is streamed to disk in chunks rather than duplicating
  the digit string.
- The **verification** (BBP) is comparatively cheap at these sizes but is not
  free; it is the independent check that gives confidence.

---

## Verification design (critical)

Two **independent** layers, enabled with `--verify`, run after computation.

### A. BBP hexadecimal spot-checks

The Bailey–Borwein–Plouffe formula computes a single hexadecimal digit of π at
an arbitrary position **without computing any preceding digits**. We compute
several such digits (including positions very close to the **end** of the
computed range, since precision/truncation errors show up there) and compare
them against the hexadecimal digits of the main computation's **binary result**
before decimal conversion (`hex_scaled = floor(π·16^hex_len)`).

This is a genuinely independent algorithm from the Chudnovsky computation, so a
match strongly corroborates correctness. The BBP checks are parallelised.

### B. Decimal checkpoints

The final decimal output is compared against checkpoints loaded from
`--checkpoints`. The **only digits hardcoded** are the **first 50 decimal
places**, used purely as a sanity check; every other checkpoint must come from
the user-supplied file. Each checkpoint is reported as PASS/FAIL with its
position, expected, and actual values, and the process **exits with a nonzero
status** if any check fails. A test deliberately corrupts a digit and confirms
verification detects it.

> The BBP layer caught a real bug during development (a `u64` overflow in the
> binary-splitting leaf at large term indices). It failed at 10M digits,
> pinpointing the far end of the range, and passed after the fix.

---

## Memory

Peak memory is minimised by:
- consuming/freeing P/Q/T intermediates as soon as they are merged;
- computing `Q²` once and reusing it for both the decimal and hex scalings;
- writing output to disk in 1 MiB chunks from a single digit buffer (no extra
  full-size copy of the string).

Measured peak resident memory (`VmHWM`): **~32 MB at 1M**, **~140 MB at 10M**,
**~2.4 GB at 100M**. The dominant live objects at 100M are the scaled integers
and the digit buffer (both on the order of the output size in bits). This is far
under the 32 GB RAM of the reference machine.

---

## Expected runtime (this machine, 32 threads, release)

| Digits | Series (BS) | Scaling | Base-10 conv. | Verify | Total | Peak mem |
|--------|-------------|---------|---------------|--------|-------|----------|
| 1,000,000 | 0.10 s | 0.10 s | 0.05 s | 0.03 s | ~0.3 s | ~32 MB |
| 10,000,000 | 0.97 s | 1.4 s | 0.50 s | 0.80 s | ~3.7 s | ~140 MB |
| 100,000,000 | 15 s | 19.5 s | 7.7 s | 9 s | ~58 s | ~2.5 GB |

*(Wall-clock release timings on the reference machine. The 100M row is measured
with `/usr/bin/time -v`: **58.1 s / 2.53 GB** with verification enabled, and
**43.2 s / 2.03 GB** without `--verify` (the "Verify" timing is excluded). All
runs pass every verification check. Figures are indicative and vary with
hardware.)*

### Performance optimisations

Beyond the baseline implementation, the following high-impact inefficiencies
were identified and fixed (see [`PROGRESS.md`](PROGRESS.md) for the
before/after):

1. **Parallel binary splitting via work-stealing.** The previous design split
   the term range into fixed segments computed on manually-spawned threads and
   then merged the segment results with a **serial left-fold**. That serialised
   the largest (most expensive) multiplications near the top of the merge tree
   and used fixed, un-balanced work assignment. Replacing it with a rayon
   `join`-based parallel recursion that schedules the *whole* split tree
   (including the big merges) by work-stealing cut the series time by **~8×**
   at 100M (127 s → 15 s) with identical output.
2. **Parallel base conversion.** The binary→decimal divide-and-conquer
   converter was single-threaded. Its two independent halves are now converted
   concurrently at the top levels (rayon `join`), giving a **~3×** speedup
   (26 s → 7.7 s at 100M).
3. **Single shared root.** Both the decimal (`10^D`) and hex (`16^H`) scalings
   need `isqrt(426880²·10005·Q²)`. This irrational root is now computed **once**
   and shared, and the `base^(2·exp)` giant power is no longer built (we scale
   by `base^exp` directly — exact because the Chudnovsky denominator `T`
   dwarfs `base^exp`). This eliminated the second huge `isqrt` and reduced peak
   memory.
4. **Concurrent decimal/hex scaling** (when verification is on): the two
   independent scale operations now run concurrently.

The 100M → 10M wall-clock improvement is ~4× (with identical, verified
output).

---

## Known limitations

- **Double-precision BBP** accumulation is used; it is reliable to at least the
  ~8.3M hex positions tested and is cross-validated against a 256-bit
  high-precision BBP (`examples/bbp_prec.rs`). Positions beyond a few tens of
  millions of hex digits should be treated with care.
- The first 50 digits only are hardcoded; all other confidence comes from BBP
  and user checkpoints.
- The program truncates (never rounds) to the requested digit count, as
  required.
- Memory scales with output size; the peak figures above are for the reference
  machine.

---

## Diagnostics (examples)

- `examples/diag.rs` — compare main hex digits against BBP across a range.
- `examples/bbp_prec.rs` — high-precision (256-bit) BBP for a single position.
- `examples/convert_test.rs` — validate the base converter by round-trip.

## Project layout

```
src/
  lib.rs        crate root / re-exports
  main.rs       CLI entry point
  chudnovsky.rs Chudnovsky binary splitting + scaling
  bbp.rs        BBP hex digit extraction
  convert.rs    subquadratic binary->decimal conversion
  verify.rs     two verification layers + checkpoints
  output.rs     streaming file output
tests/
  integration.rs  N = 10,000 end-to-end test
examples/
  diag.rs, bbp_prec.rs, convert_test.rs
README.md
PROGRESS.md
```
