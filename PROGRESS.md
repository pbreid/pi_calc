# PROGRESS.md — π computation project checkpoint

This file acts as a running checkpoint for the project. It documents what has
been built, validation results, known issues (and how they were found / fixed),
the code-review round, and what remains.

## Status: FUNCTIONAL — all review findings addressed; `cargo test` green.

Outputs at 1M / 10M / 100M are **byte-identical** to the previously verified
files (SHA-256 below).

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
7. **Tests**: 22 unit tests + 1 integration test, including the required new
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
