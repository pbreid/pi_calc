# PROGRESS.md — π computation project checkpoint

This file acts as a running checkpoint for the project. It documents what has
been built, validation results, known issues (and how they were found / fixed),
the code-review round, the optimization round, and what remains.

## Status: FUNCTIONAL — all review findings addressed; `cargo test` green.

Outputs at 1M / 10M / 100M are **byte-identical** to the previously verified
files (SHA-256 below).

## Optimization round 2 (performance & memory pass)

Motivated by the challenge "find further performance improvements and reduce
the memory footprint". All changes keep the outputs byte-identical (SHA-256
below). Measured on the reference machine, 32 threads, `RUSTFLAGS="-C
target-cpu=native"`.

**What was implemented (all verified):**

1. **Parallel split multiplication** (`chudnovsky.rs::par_mul`): GMP
   multiplication is single-threaded, so the top-of-tree merges serialise. For
   operands ≥ 96M bits (up to depth 2), the larger operand is split at a 64-bit
   limb boundary and the two half-size products are computed via
   `rayon::join`: `a·(b_hi·2^s + b_lo) = a·b_hi·2^s + a·b_lo`, exact for any
   signs (`mpz_fdiv_q_2exp`/`mpz_fdiv_r_2exp` are the floor pair). Applied to
   the three merge products, `Qh²` in the binary scaling, and `M·5^D` (plus
   `5^D` via a parallel squaring). Root merge at 100M: ~3.5 s → ~1.2 s.
2. **`u128` term leaves**: for `k ≤ 7,000,000` the leaf values `p_k, q_k, t_k`
   all fit in `u128` (`t_k ≈ 216·B·k⁴ ≤ 2^128` is binding; the big-integer
   path is the fallback for larger `k`), replacing ~6 GMP ops + allocations per
   term with native multiplies: all 7,051,386 leaves 563 ms → 195 ms.
   Equivalence with the big-integer path is tested at the boundary
   (`u128_leaf_matches_big_path`).
3. **Fused div/rem in the base converter** (`convert.rs`): each D&C node
   computed `x / R^m` and `x % R^m` as two full divisions; now one
   `mpz_tdiv_qr` pass (`div_rem_ref`). Conversion at 100M: 6.75 s → 4.0 s.
   The per-leaf base case also no longer allocates a comparison `Integer`
   (~11M times at 100M).
4. **Folded guard digits** (`decimal_truncated`):
   `floor(floor(π·10^D)/10^guard) = floor(π·10^(D−guard))`, so the final
   `/10^guard` folds into the shift — `(M·5^(D−guard)) >> (W − digits)` —
   removing a full-size division pass and a temporary.
5. **Barrett modpow for the BBP sums** (`bbp.rs::modpow`): within one `modpow`
   the modulus is fixed, so one Barrett reciprocal `μ = floor(2^64/m)` serves
   all ~`2·log₂e` reductions (each now 2 multiplies instead of a 64-bit
   division). Exact for `m < 2^32` (the BBP case; `m = 8k+j`), with the `u128`
   fallback kept for larger moduli (2⁶⁴−59 in the conversion check).
   Verification at 100M: 14.8 s → 8.2 s. Equivalence tested against the
   reference path over a 2000-case sweep (`modpow_barrett_path_matches_reference`).
6. **glibc mmap-threshold tuning** (`main.rs::tune_malloc`, Linux only): glibc
   keeps one never-shrinking heap arena per thread, so 32 workers hoard freed
   multi-MB GMP operands and VmHWM ≈ 4× the live set. `mallopt(M_MMAP_THRESHOLD,
   8 MiB)` returns them to the OS on free: peak at 100M 2.0–2.3 GB → 1.31–1.40
   GB for ≈4% time. `PI_STD_MALLOC=1` restores default behavior.
7. **Merge ordering** (`chudnovsky.rs::merge`): products are formed in an order
   that drops each operand as soon as it is consumed (and `P` is skipped when
   `need_p` is false as before), trimming peak live memory at the top levels.
8. **Owned shifts** (`binary_pi_fixed`): `Q`/`T` are taken by value and shifted
   in place, freeing the full-size originals before the `isqrt`/division steps.

**Measured (verification on, wall clock):**

| Digits | Series | Scaling | Conv. | Verify | Total | Peak mem |
|--------|--------|---------|-------|--------|-------|----------|
| 1M  | 0.066 s | 0.048 s | 0.022 s | 0.243 s | **0.379 s** | **15.1 MB** |
| 10M | 0.84 s | 0.73 s | 0.29 s | 0.83 s | **2.70 s** | **115 MB** |
| 100M | 9.4 s | 8.6 s | 4.1 s | 8.2 s | **30.5 s** | **1.36 GB** |

(Previous round: 0.406 s / 27.9 MB, 3.50 s / 204 MB, 45.2 s / 1.67 GB.
Compute-only at 100M: 28.7 s → 21.9–22.2 s.)

**Rejected experiments (measured, no win):**

- `mpz_get_str` for the base conversion: **slower** than the custom D&C
  converter at 100M digits (11.0 s vs 6.9 s) and more memory; custom kept.
- `mpz_addmul` for the merge `T = T1·Q2 + P1·T2`: GMP implements it as
  multiply-then-add — no time win (3.56 s vs 3.58 s at 413M-bit operands) and
  no memory win over the early-drop ordering.
- Hand-rolled multiply-only Newton for `1/√10005` / reciprocals: GMP's
  division-based `mpz_sqrt` and `mpz_tdiv_q` are already cheaper at these
  sizes (isqrt ≈ 3.1 s ≈ 3 full-size multiplies; a Newton reciprocal costs
  more than GMP's division). The remaining binary-scaling cost (isqrt +
  division, ~6.5 s) is GMP-bound.
- Raising the split threshold to 256M bits (only the root splits): slower
  (series 11.1 s) with **no** memory benefit — the peak is dominated by the
  tree's inherent live set, not the split transients.
- `malloc` arena caps (`M_ARENA_MAX=1/4/8`): arena lock contention costs far
  more (up to +10 s) than the mmap-threshold approach.

## Hardware / environment

- CPU: **Intel Core i9-14900KS** (24 cores / 32 logical threads)
- RAM: **32 GB**
- Arch: **x86_64**, Linux (WSL2)
- Toolchain: **Rust 1.97.1 (stable)**
- Big integers: **GMP via `rug`** (system GMP available)

## What has been implemented

1. **Chudnovsky formula + binary splitting** (`src/chudnovsky.rs`):
   - Exact integer `(P, Q, T)` recurrence, verified against a brute-force exact
     rational series sum for small `N`.
   - `π = 426880·√10005 · Q / T`, computed as a **binary fixed-point integer**
     `M ≈ floor(π·2^W)` from **truncated** `Q`/`T` (working precision), then
     `floor(π·10^D) = (M·5^D) >> (W−D)` — exact (error `< 2^(−190)`).
   - Parallelised with a **rayon `join` work-stealing recursion** over the whole
     split tree; `P` is skipped on the right spine.
2. **Subquadratic binary→decimal conversion** (`src/convert.rs`):
   - Divide-and-conquer base-10⁹ conversion, `O(M(n)·log n)`; parallelised at
     the top levels; renders directly with the leading `"3"` dropped.
3. **BBP hex digit extraction** (`src/bbp.rs`):
   - Single-position fractional computation; multi-digit **runs**; `u128`
     modular exponentiation (safe for moduli > 2³²); calibrated `f64` error
     bound.
4. **Three verification layers** (`src/verify.rs`):
   - A: BBP run checks vs the main computation's binary result (PASS/FAIL/
     INCONCLUSIVE).
   - B: modular check of the decimal conversion against the exact truncated
     integer (parallel Horner mod 3 large primes).
   - C: decimal checkpoints (only the first 50 digits are hardcoded; beyond-range
     checkpoints are SKIPPED).
5. **CLI** (`src/main.rs`, `clap`): `--digits`, `--threads`, `--output`,
   `--verify` (verification is **off by default**), `--checkpoints`, `--bench`,
   `--guard`.
6. **Streaming output** (`src/output.rs`).
7. **Tests**: 25 unit tests + 1 integration test (including the u128-leaf
   boundary, par_mul split property, and Barrett modpow equivalence tests),
   including the required new
   tests (modular-conversion corruption, BBP-run consistency, `u128` modpow
   regression, primes-are-prime, skipped checkpoints).

## Validation performed

- `cargo test` — all unit + integration tests pass.
- 1M / 10M / 100M digits computed and cross-checked against an independent
  external source (api.pi.delivery) and against the built-in checks; all match.
- Post-review outputs are **byte-identical** to the pre-review verified files:
  - 1M: `dd382ef6a0c1e8d920fb72f482d74826251ab97709520bc24f913cd8eb5fc839`
  - 10M: `46059c61a4de67d6c916fa958168789da324a03ee8a85c30e9ca292c3712eb25`
  - 100M: `3dd4bc2392f543a8b4dff7634fb18da86e36c88cf35acb7b3fe7b70d864fa823`

## Critical bug found & fixed (earlier round)

**`src/chudnovsky.rs` leaf numerator `(6k−5)(2k−1)(6k−1)` overflowed `u64`.**

- Exceeds u64 for `k ≳ 635,000`. At 10M digits the high-index terms were
  corrupted, so output was wrong from ~position 9,999,900 onward (1M was fine).
- **Caught by the BBP verification near the end of the range**; fixed by using
  `Integer` arithmetic; confirmed against the external digit source.

## Code-review round — findings addressed

1. **[HIGH] Decimal conversion verified at runtime.** Added the modular
   conversion check (`verify_conversion`): reduce the exact truncated integer and
   the rendered string modulo 2⁶¹−1, 2⁶⁴−59, 2³¹−1, with parallel Horner
   (chunked, combined with `10^chunk mod p`). Reported PASS/FAIL. A test corrupts
   a middle digit and confirms this FAILS while BBP still passes. **Cost at 100M:
   0.156 s**, ~2% of the 7.3 s conversion.
2. **[HIGH] Final scaling at ~2.5× excess precision.** `Q`/`T` are truncated to
   `W+128` bits (with a proven bound `|Qh/Th − Q/T| < 2^(−Wm)`), π is computed as
   a binary fixed-point integer `M = floor(G·2^W/Th)`, and **hex digits are taken
   directly from `M`'s nibbles** (no second division). Scaling at 100M:
   **19.5 s → 9.4 s**; peak memory **2.53 GB → 1.67 GB**. (Chose exact integer
   fixed-point rather than `rug::Float` so truncation correctness is provable.)
3. **[MEDIUM] Wasted `P` at the top of the split tree.** `P` is now computed only
   where needed (left children); the right spine skips it. Series 100M:
   **15.3 s → 13.6 s**.
4. **[MEDIUM] Latent modpow overflow.** `modpow` now uses `u128` intermediates;
   regression test with modulus `> 2^32`.
5. **[MEDIUM] Stronger BBP.** Runs of up to **8 hex digits** per position (length
   chosen so `frac_error_bound(n)·16ᵏ ≤ 0.5`); boundary-adjacent runs are
   reported **INCONCLUSIVE** (never PASS/FAIL); positions now include interior
   fractions across the range.
6. **[MEDIUM] Spec deviations.** Beyond-range checkpoints are **SKIPPED** (1M
   with `checkpoints.example.txt` exits 0). The always-zero "hex/verify" timing
   line was removed; real per-phase timings are printed.
7. **[LOW] Memory / efficiency.** The digit buffer is rendered once (no
   `s[1..].to_string()` copy); hex nibbles are extracted directly from `M` (no
   full hex string). Peak memory at 100M: **2.53 GB → 1.67 GB**.
8. **[LOW] Doc drift.** Corrected the `convert.rs` header (no precomputed power
   table) and removed the dead `compute_radix_powers`; audited the other module
   docs (e.g. `lib.rs` now says *three* layers).

### Choices / not done (and why)

- **Review item 5 run length**: the review suggested "~8"; `f64` reliably
  supports 8 at the largest positions *with* the boundary guard, and runs near a
  boundary are `INCONCLUSIVE`. The error bound is a calibrated statistical bound
  (~90× the measured error), not a strict worst case — documented as a
  limitation.
- **Review item 8**: the review allowed either caching radix powers *or*
  correcting the comment; I corrected the comment / removed the dead code
  (caching would add complexity for little gain in the already-parallel
  converter).
- **Verification default**: the review asked for verification ON by default, but
  the project owner subsequently requested **off by default** with `--verify`
  to enable it. The code follows the owner's request; `--checkpoints` implies
  `--verify`.

## Measured timings — before vs after the review

| Digits | | Series | Scaling | Conv. | Verify | Total | Peak mem |
|--------|--|--------|---------|-------|--------|-------|----------|
| 1,000,000 | before | 0.060 s | 0.083 s | 0.034 s | 0.139 s | 0.317 s | 38.3 MB |
| 1,000,000 | after  | 0.052 s | 0.047 s | 0.039 s | 0.266 s | 0.406 s | 27.9 MB |
| 10,000,000 | before | 0.94 s | 1.42 s | 0.51 s | 0.81 s | 3.68 s | 356 MB |
| 10,000,000 | after  | 0.88 s | 0.69 s | 0.48 s | 1.44 s | 3.50 s | 204 MB |
| 100,000,000 | before | 15.3 s | 19.5 s | 7.7 s | 9.1 s | ~51.6 s¹ | 2.53 GB |
| 100,000,000 | after  | 13.6 s | 9.4 s | 7.3 s | 14.8 s | 45.2 s | 1.67 GB |

¹ Before wall clock (verification on) was 58.1 s (`/usr/bin/time -v`); after it
is 45.2 s.

Note: verification is deliberately *stronger* now (more positions, multi-digit
runs, modular conversion check), so its time rose; that is the intended
trade-off.

## Build / run

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
cargo test

# compute only (default; no verification)
./target/release/pi --digits 100000000 --output pi_100m.txt --bench
# with verification
./target/release/pi --digits 100000000 --verify --output pi_100m.txt
```
