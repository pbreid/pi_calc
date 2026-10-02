//! Bailey–Borwein–Plouffe (BBP) hexadecimal digit extraction of π.
//!
//! The BBP formula expresses π in base 16 in a way that lets one compute a
//! *single* hexadecimal digit at an arbitrary position without computing any
//! preceding digits:
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
//! The `n`-th hex digit is `floor(16 · frac(16^{n-1} · π))`. We compute
//! `frac(16^{n-1}·π)` by splitting the BBP series into an integer part and a
//! fractional part:
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
//! The first sum (with `16^{n-1-k}` an integer) is handled by modular
//! exponentiation, keeping only the fractional contribution; the second sum is
//! a rapidly-converging tail approximated by a handful of terms. We work in
//! double precision, which is sufficient because the BBP digit extraction
//! needs only ~1/(16·8k) accuracy and the errors do not accumulate. (See the
//! note in [`hex_digit_checked`].)
//!
//! The sum over `k` is embarrassingly parallel and is distributed across all
//! available threads with `rayon`.

use rayon::prelude::*;

/// Modular exponentiation `base^exp mod m`, computed with u64 arithmetic
/// (safe because the modulus, `8k+j`, is small compared to 2^64).
fn modpow(mut base: u64, mut exp: usize, m: u64) -> u64 {
    debug_assert!(m > 0);
    let mut result = 1u64 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
    result
}

/// The fractional part of `16^{n-1} · π`, i.e. `frac(16^{n-1} π)`, in `[0,1)`.
///
/// `threads` controls how many threads are used for the inner k-sum (0 = all
/// cores). Computed in double precision.
pub fn bbp_fractional(n: usize, threads: usize) -> f64 {
    let (_digits, fr) = bbp_digit_precise(n, threads);
    fr
}

/// Compute the `n`-th hexadecimal digit of π (1-based) and also return the raw
/// fractional value, so callers can detect numerical ambiguity.
///
/// Returns `(digit, fractional)`. `digit` is the hex digit value 0..=15;
/// `fractional = 16·frac(16^{n-1}·π)` (a value in `[0,16)`).
pub fn hex_digit_checked(n: usize, threads: usize) -> (u8, f64) {
    bbp_digit_precise(n, threads)
}

fn bbp_digit_precise(n: usize, threads: usize) -> (u8, f64) {
    let pool = build_pool(threads);
    let mut acc = 0.0f64;
    for (j, c) in [(1u64, 4.0f64), (4u64, -2.0f64), (5u64, -1.0f64), (6u64, -1.0f64)] {
        let s = sum_over_k(n, j, &pool);
        acc += c * s;
        acc = acc.rem_euclid(1.0);
    }
    acc = acc.rem_euclid(1.0);
    let frac16 = 16.0 * acc;
    let digit = frac16.floor() as i64;
    let digit = digit.clamp(0, 15) as u8;
    (digit, frac16)
}

/// Sum `Σ_{k=0}^{n-1} ((16^{n-1-k} mod (8k+j)) / (8k+j))` plus the tail, in
/// double precision, using the provided thread pool.
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
    // The tail from k = n onward.
    let tail = tail_sum(n, j);
    (s + tail).rem_euclid(1.0)
}

/// The rapidly converging tail `Σ_{k=n}^{∞} 1/(16^{k-n+1}·(8k+j))`.
/// Because the terms shrink by a factor of 16 each step, ~18 terms give far
/// more than double-precision accuracy.
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
}
