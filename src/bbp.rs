//! Bailey–Borwein–Plouffe (BBP) hexadecimal digit extraction of π.
//!
//! The BBP formula expresses π in base 16 in a way that lets one compute
//! hexadecimal digits at an arbitrary position without computing any preceding
//! digits:
//!
//! ```text
//! π = Σ_{k=0}^∞ (1/16^k) · [ 4/(8k+1) − 2/(8k+4) − 1/(8k+5) − 1/(8k+6) ]
//! ```
//!
//! # Digit indexing
//!
//! We use the same contiguous convention as the rest of the crate: the *first*
//! hexadecimal digit after the point is position 1 (the `2` in
//! `π = 3.243F6A8885A308D3…16`). Position `n` is the `n`-th hex digit after
//! the point.
//!
//! # Algorithm
//!
//! The digits starting at position `n` are `floor(16^k · frac(16^{n-1}·π))` for
//! a run of `k` digits. We compute `frac(16^{n-1}·π)` by splitting the BBP
//! series into an integer part and a fractional part:
//!
//! ```text
//! frac(16^{n-1} π) = frac( Σ_j c_j · S_j )
//! ```
//!
//! where `(j, c_j) ∈ {(1,4),(4,−2),(5,−1),(6,−1)}` and
//!
//! ```text
//! S_j = Σ_{k=0}^{n-1} ((16^{n-1-k} mod (8k+j)) / (8k+j))
//!       + Σ_{k=n}^{∞} 1 / (16^{k-n+1} · (8k+j))
//! ```
//!
//! The first sum is handled by modular exponentiation, keeping only the
//! fractional contribution; the second sum is a rapidly-converging tail
//! approximated by a handful of terms.
//!
//! # Precision
//!
//! The sum is accumulated in `f64`. Each term is `(t/m)` with `t,m` exact
//! `f64` values below `2^32` (the modulus is `8k+j`), so each division
//! contributes at most `0.5 ulp ≈ 2^-53` of absolute error. After reducing mod
//! 1 the partial sums stay in `[0,1)`. The accumulated error behaves like a
//! random walk in the rounding directions, i.e. `≈ √(4n)·2^-53`; we multiply
//! that by a generous safety factor (see [`frac_error_bound`]). The error is
//! **not** allowed to silently decide a digit: any run whose value lies within
//! the error bound of a digit boundary is reported `INCONCLUSIVE` by the
//! verifier rather than PASS/FAIL. This is why we compare a run of digits (the
//! run is only trusted when its value is far enough from a boundary), and why
//! the run length is capped at [`MAX_RUN`] hex digits.
//!
//! # Overflow
//!
//! The modular-exponentiation intermediates `result·base` and `base·base` are
//! computed in `u128`. With the modulus up to `8k+6` this matters once
//! `8k+6 ≳ 2^32` (≈650M decimal digits); a `u64` intermediate would overflow
//! there (the same class of bug as the old leaf overflow).

use rayon::prelude::*;

/// The maximum number of hexadecimal digits compared per BBP position.
pub const MAX_RUN: usize = 8;

/// Modular exponentiation `base^exp mod m`.
///
/// For `m < 2^32` (the BBP case: `m = 8k+j < 8n+6`) the products fit in `u64`
/// and a per-`modpow` Barrett reciprocal replaces the hardware division in
/// every reduction — one `u64` division to compute `μ = floor(2^64/m)`, then
/// each of the ~`2·log₂ exp` reductions costs two multiplies instead of one
/// 64-bit division. For `m ≥ 2^32` the products are formed in `u128` (a `u64`
/// intermediate would overflow once the modulus exceeds ≈`2^32`).
pub fn modpow(mut base: u64, mut exp: usize, m: u64) -> u64 {
    debug_assert!(m > 0);
    if m == 1 {
        return 0;
    }
    let mut result = 1u64 % m;
    base %= m;
    if m < 1 << 32 {
        // Barrett: μ = floor(2^64/m) (fits u64 for m ≥ 2). For x < m² < 2^64,
        // q = floor(x·μ / 2^64) ∈ {floor(x/m) − 1, floor(x/m)}, so
        // r = x − q·m < 2m and one conditional subtraction finishes.
        let mu = ((1u128 << 64) / u128::from(m)) as u64;
        let red = |x: u64| -> u64 {
            let q = (((x as u128) * (mu as u128)) >> 64) as u64;
            let r = x - q * m;
            if r >= m { r - m } else { r }
        };
        while exp > 0 {
            if exp & 1 == 1 {
                result = red(result * base);
            }
            base = red(base * base);
            exp >>= 1;
        }
        return result;
    }
    let m128 = m as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m128) as u64;
        }
        base = ((base as u128 * base as u128) % m128) as u64;
        exp >>= 1;
    }
    result
}

/// A conservative bound on the absolute error of the double-precision BBP
/// fractional sum for position `n`.
///
/// Each of the `≈4n` term divisions contributes at most `2^-53`; the rounding
/// directions are effectively random, so the total grows like
/// `√(4n)·2^-53`. We apply a safety factor of 16. Calibration (see
/// `examples/bbp_prec.rs`, which compares against a 256-bit computation): at
/// `n ≈ 8.3M` the measured error was `2.3e-13`, while this bound gives
/// `2.0e-11` (~90×); at `n ≈ 83M` it gives `6.5e-11` (~90× the expected
/// error). The verifier never decides a digit that lies within this bound
/// (such runs are `INCONCLUSIVE`).
pub fn frac_error_bound(n: usize) -> f64 {
    let rw = (4.0 * n as f64).sqrt() * f64::EPSILON * 16.0;
    rw.max(1e-12)
}

/// The fractional part `frac(16^{n-1}·π)` in `[0,1)`, computed in `f64`.
///
/// `threads` controls the inner k-sum (0 = all cores).
pub fn bbp_fractional(n: usize, threads: usize) -> f64 {
    compute_frac(n, threads)
}

/// Compute the run of `k` (`≤ MAX_RUN`) hexadecimal digits of π beginning at
/// position `n` (1-based).
///
/// Returns `(run, err, dist)` where `run` is the integer formed by those `k`
/// hex digits (most significant first), `err` is a bound on the absolute error
/// of `run` induced by the `f64` accumulation, and `dist` is the distance of
/// the underlying real value from the nearest digit boundary (also in `run`
/// units). The run may be trusted iff `dist > err`.
pub fn hex_run(n: usize, k: usize, threads: usize) -> (u64, f64, f64) {
    assert!(k >= 1 && k <= MAX_RUN, "run length {k} out of range");
    let frac = compute_frac(n, threads);
    let scale = 16f64.powi(k as i32);
    let v = frac * scale;
    let floor = v.floor();
    let run = floor as u64;
    let frac_part = v - floor;
    let dist = frac_part.min(1.0 - frac_part);
    let err = frac_error_bound(n) * scale;
    (run, err, dist)
}

/// Compute the `n`-th hexadecimal digit of π (1-based) and the raw
/// `16·frac(16^{n-1}·π)` value (in `[0,16)`), so callers can inspect ambiguity.
pub fn hex_digit_checked(n: usize, threads: usize) -> (u8, f64) {
    let frac = compute_frac(n, threads);
    let frac16 = 16.0 * frac;
    let digit = (frac16.floor() as i64).clamp(0, 15) as u8;
    (digit, frac16)
}

/// Compute `frac(16^{n-1}·π)` in `f64`.
fn compute_frac(n: usize, threads: usize) -> f64 {
    let pool = build_pool(threads);
    let mut acc = 0.0f64;
    for (j, c) in [(1u64, 4.0f64), (4u64, -2.0f64), (5u64, -1.0f64), (6u64, -1.0f64)] {
        let s = sum_over_k(n, j, &pool);
        acc += c * s;
        acc = acc.rem_euclid(1.0);
    }
    acc.rem_euclid(1.0)
}

/// Sum `Σ_{k=0}^{n-1} ((16^{n-1-k} mod (8k+j)) / (8k+j))` plus the tail.
fn sum_over_k(n: usize, j: u64, pool: &rayon::ThreadPool) -> f64 {
    if n == 0 {
        return tail_sum(0, j);
    }
    let s: f64 = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|k| {
                let e = n - 1 - k;
                let m = 8 * k as u64 + j;
                let t = modpow(16, e, m);
                (t as f64) / (m as f64)
            })
            .reduce(|| 0.0, |a, b| (a + b).rem_euclid(1.0))
    });
    let tail = tail_sum(n, j);
    (s + tail).rem_euclid(1.0)
}

/// The rapidly converging tail `Σ_{k=n}^{∞} 1/(16^{k-n+1}·(8k+j))`.
fn tail_sum(n: usize, j: u64) -> f64 {
    let mut total = 0.0f64;
    let mut p: f64 = 16.0; // 16^{k-n+1} for k=n
    for k in n..n + 24 {
        total += 1.0 / (p * (8 * k as u64 + j) as f64);
        p *= 16.0;
    }
    total
}

/// Build a rayon thread pool for the given thread count (0 => all cores).
fn build_pool(threads: usize) -> rayon::ThreadPool {
    let t = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(t)
        .build()
        .expect("failed to build rayon thread pool")
}

/// Compute the `n`-th hexadecimal digit of π (1-based), using all available
/// cores.
pub fn hex_digit(n: usize) -> u8 {
    hex_digit_checked(n, 0).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    // The known hexadecimal digits of π after the point:
    // π = 3.243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89…
    const KNOWN: &[u8] = &[
        0x2, 0x4, 0x3, 0xF, 0x6, 0xA, 0x8, 0x8, 0x8, 0x5, 0xA, 0x3, 0x0, 0x8, 0xD, 0x3,
        0x1, 0x3, 0x1, 0x9, 0x8, 0xA, 0x2, 0xE, 0x0, 0x3, 0x7, 0x0, 0x7, 0x3, 0x4, 0x4,
    ];

    #[test]
    fn bbp_matches_known_hex_digits() {
        for (i, &expected) in KNOWN.iter().enumerate() {
            let n = i + 1;
            let (d, _) = hex_digit_checked(n, 1);
            assert_eq!(
                d, expected,
                "hex digit at position {n} mismatch (got {d:#x}, expected {expected:#x})"
            );
        }
    }

    #[test]
    fn bbp_digit_is_valid_value() {
        for n in [1usize, 2, 5, 10, 100, 1000] {
            let (d, frac) = hex_digit_checked(n, 2);
            assert!(d <= 15, "digit {d} out of range at n={n}");
            assert!((0.0..16.0).contains(&frac), "frac out of range at n={n}: {frac}");
        }
    }

    /// The Barrett fast path (`m < 2^32`) must agree exactly with the u128
    /// reference path, including even moduli (`8k+4`) and `m = 2`.
    #[test]
    fn modpow_barrett_path_matches_reference() {
        let reference = |base: u64, exp: usize, m: u64| -> u64 {
            let m128 = u128::from(m);
            let mut result = 1u64 % m;
            let mut base = base % m;
            let mut exp = exp;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = ((u128::from(result) * u128::from(base)) % m128) as u64;
                }
                base = ((u128::from(base) * u128::from(base)) % m128) as u64;
                exp >>= 1;
            }
            result
        };
        // A LCG sweep over small/odd/even/large-below-2^32 moduli.
        let mut state: u64 = 0x243F_6A88_85A3_08D3;
        for _ in 0..2000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let m = (state % (1u64 << 32)).max(2);
            let base = state >> 33;
            let exp = ((state >> 20) % 5000) as usize;
            assert_eq!(
                modpow(base, exp, m),
                reference(base, exp, m),
                "modpow({base},{exp},{m})"
            );
        }
        for m in [2u64, 3, 4, 8, 9, (1u64 << 31) + 1, (1u64 << 32) - 1] {
            for exp in [0usize, 1, 2, 3, 17, 1000] {
                assert_eq!(modpow(7, exp, m), reference(7, exp, m), "m={m} exp={exp}");
            }
        }
    }

    /// Regression test for the u64 modpow overflow: the modulus here exceeds
    /// `2^32`, where `result*base` / `base*base` would overflow a u64.
    #[test]
    fn modpow_handles_modulus_above_2_32() {
        let m: u64 = 5_000_000_007; // > 2^32
        assert!(m > (1u64 << 32));
        let cases: [(u64, usize); 4] = [
            (16, 0),
            (16, 1),
            (16, 1_234_567_891),
            (123_456_789, 987_654_321),
        ];
        for (b, e) in cases {
            let got = modpow(b, e, m);
            let expect = Integer::from(b)
                .pow_mod(&Integer::from(e), &Integer::from(m))
                .unwrap()
                .to_u64()
                .unwrap();
            assert_eq!(got, expect, "modpow({b},{e},{m})");
        }
    }

    /// The run length is capped so that the f64 accumulation reliably supports
    /// it; a boundary run must be detectable.
    #[test]
    fn hex_run_is_consistent_with_single_digits() {
        // Positions from the known prefix: the first 8 hex digits are
        // 243F6A88.
        let (run, _err, _dist) = hex_run(1, 8, 2);
        assert_eq!(run, 0x243F6A88, "8-hex-digit run at position 1");
    }
}
