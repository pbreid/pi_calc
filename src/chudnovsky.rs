//! Chudnovsky formula for π using binary splitting with `rug` (GMP) integers.
//!
//! # Math
//!
//! The Chudnovsky series is
//!
//! ```text
//! 1/π = 12 / √C3 · Σ_{k=0}^∞  (-1)^k (6k)! (A + B·k) / ( (3k)! (k!)^3 C3^k )
//! ```
//! with `A = 13591409`, `B = 545140134`, `C3 = 640320^3 = 2^6 · 10005`.
//! Rearranging (using `√C3 / 12 = 426880·√10005`) gives the form we use:
//!
//! ```text
//! π = 426880·√10005 · Q / T
//! ```
//!
//! where `T/Q = Σ_{k=0}^{N-1} s_k` is the truncated partial sum returned by the
//! binary splitting.
//!
//! # Binary splitting
//!
//! The partial sum is computed with the classic divide-and-conquer recurrence.
//! For a range `[a,b)` we maintain integer triples `(P, Q, T)` defined so that
//! `T(a,b) / Q(a,b)` equals `Σ_{k=a}^{b-1} s_k`. The base case for a single
//! index `k≥1` uses the reduced factors
//!
//! ```text
//! p_k = (6k−5)·(2k−1)·(6k−1),      q_k = k³ · C3 / 24
//! ```
//!
//! These arise after cancelling the common factors between `(6k)!/((3k)!(k!)^3)`
//! and the denominator. The alternating sign `(−1)^k` is folded into `p_k`
//! (i.e. `p_k` is negated), and the `k=0` term is handled specially (`s_0=A`).
//!
//! The merge step is:
//!
//! ```text
//! P(a,b) = P(a,m)·P(m,b)
//! Q(a,b) = Q(a,m)·Q(m,b)
//! T(a,b) = T(a,m)·Q(m,b) + P(a,m)·T(m,b)
//! ```
//!
//! with `m = (a+b)/2`. This is exact integer arithmetic; the number of
//! big-integer multiplications is roughly linear in the number of terms
//! (times a log factor), rather than quadratic, which is why binary splitting
//! is fast.

use rug::ops::NegAssign;
use rug::Integer;
use std::time::Instant;

/// `A` = 13591409
pub const A: i64 = 13_591_409;
/// `B` = 545140134
pub const B: i64 = 545_140_134;
/// `C` = 640320
pub const C: i64 = 640_320;
/// `C3` = 640320^3 = 262537412640768000
pub const C3: u128 = 262_537_412_640_768_000;
/// `C3/24` = 10939058860032000 (always an integer).
pub const C3_OVER_24: u128 = C3 / 24;
/// The constant `426880` in `π = 426880·√10005·Q/T`.
pub const PI_K: u64 = 426_880;
/// 10005 under the square root.
pub const TEN_THOUSAND_FIVE: u64 = 10_005;
/// Decimal digits gained per Chudnovsky term (≈ log₁₀(640320³) − log₁₀(...)).
pub const DIGITS_PER_TERM: f64 = 14.181_647_462_725_477;

/// Minimum extra terms of safety beyond the asymptotic requirement.
const TERM_SAFETY_MARGIN: usize = 16;

/// A computed value of π plus metadata.
pub struct PiResult {
    /// `floor(π · 10^(digits+guard))`, an exact integer. Dividing by `10^guard`
    /// gives the truncated decimal digit string.
    pub scaled: Integer,
    /// `floor(π · 16^hex_len)`, an exact integer, used only for verification.
    /// Its hexadecimal representation is `"3"` followed by `hex_len` hex digits
    /// of π (positions 1..=hex_len), which the BBP verifier compares against.
    pub hex_scaled: Integer,
    /// Number of requested decimal places (digits after the "3.").
    pub digits: usize,
    /// Number of Chudnovsky terms used.
    pub n_terms: usize,
    /// Number of internal guard digits computed beyond `digits`.
    pub guard: usize,
    /// Number of hexadecimal digits represented by `hex_scaled`.
    pub hex_len: usize,
}

impl PiResult {
    /// Return exactly `digits` decimal digits (after the "3.") as a `String`.
    ///
    /// This is the canonical digit buffer used both for streaming output and
    /// for verification. It uses the crate's subquadratic binary→decimal
    /// conversion.
    pub fn decimal_digits(&self) -> String {
        let div = Integer::from(Integer::u_pow_u(10, self.guard as u32));
        let truncated = Integer::from(&self.scaled / div);
        let s = crate::convert::to_decimal_string(&truncated);
        debug_assert!(s.starts_with('3'), "leading digit must be 3 (got {s})");
        debug_assert!(s.len() == self.digits + 1, "digit count mismatch");
        s[1..].to_string()
    }

    /// Returns the full decimal string `"3." + digits` for `self`.
    pub fn decimal_string(&self) -> String {
        let d = self.decimal_digits();
        format!("3.{d}")
    }

    /// Return the n-th hexadecimal digit (1-based) of π as extracted from the
    /// main computation's binary result. Panics if `n` is out of range.
    ///
    /// The hexadecimal representation of `hex_scaled` is `"3"` + `hex_len`
    /// hex digits (positions 1..=hex_len), so position `n` is character index
    /// `n` in that string.
    pub fn main_hex_digit(&self, n: usize) -> u8 {
        assert!(n >= 1 && n <= self.hex_len, "hex position {n} out of range");
        let hex = format!("{:x}", self.hex_scaled);
        let ch = hex.as_bytes()[n]; // index 0 is the leading '3'
        hex_digit_from_char(ch)
    }
}

fn hex_digit_from_char(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => unreachable!("invalid hex char"),
    }
}

/// Configuration controlling a π computation.
#[derive(Clone, Debug)]
pub struct PiConfig {
    /// Number of decimal places after the "3.".
    pub digits: usize,
    /// Number of worker threads. 0 means all available logical cores.
    pub threads: usize,
    /// Extra guard digits computed beyond `digits`.
    pub guard: usize,
}

impl PiConfig {
    /// Create a config; a zero `guard` defaults to a sane value.
    pub fn new(digits: usize, threads: usize, guard: usize) -> Self {
        let guard = if guard == 0 { default_guard(digits) } else { guard };
        Self { digits, threads, guard }
    }

    /// The effective thread count, resolving 0 to the available parallelism.
    pub fn effective_threads(&self) -> usize {
        if self.threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            self.threads.max(1)
        }
    }

    /// Number of Chudnovsky terms needed so that the tail error is safely
    /// below `10^(digits+guard)`.
    pub fn n_terms(&self) -> usize {
        terms_for_digits(self.digits + self.guard)
    }
}

/// A sensible default guard size. We use a fixed guard; the series tail is
/// computed to far more digits than we keep and the integer sqrt/division are
/// exact, so a fixed guard fully absorbs the tiny floor/rounding uncertainty.
pub fn default_guard(digits: usize) -> usize {
    let _ = digits;
    32
}

/// Number of Chudnovsky terms required to obtain `digits` correct decimal
/// places (with a safety margin).
pub fn terms_for_digits(digits: usize) -> usize {
    let keep = digits as f64 / DIGITS_PER_TERM;
    (keep.ceil() as usize) + TERM_SAFETY_MARGIN
}

/// Recursive sequential binary splitting over the range `[a, b)`.
/// Returns `(P, Q, T)` with `T/Q = Σ_{k=a}^{b-1} s_k`.
fn chudnovsky_bs(a: usize, b: usize) -> (Integer, Integer, Integer) {
    debug_assert!(b > a);
    if b - a == 1 {
        return leaf(a);
    }
    let m = a + (b - a) / 2;
    let (p1, q1, t1) = chudnovsky_bs(a, m);
    let (p2, q2, t2) = chudnovsky_bs(m, b);
    merge((p1, q1, t1), (p2, q2, t2))
}

/// The binary-splitting leaf for a single index `k`.
fn leaf(k: usize) -> (Integer, Integer, Integer) {
    if k == 0 {
        // s_0 = A. The recurrence leaf contributes P=1, Q=1, T=A.
        return (Integer::from(1), Integer::from(1), Integer::from(A));
    }
    // p_k = (6k−5)(2k−1)(6k−1); q_k = k³·(C3/24).
    // The (−1)^k sign is folded into p_k (so Π p_j carries the alternating
    // sign automatically).
    //
    // NOTE: the product (6k−5)(2k−1)(6k−1) exceeds u64 for k ≳ 635,000, so we
    // MUST multiply as big integers (each factor fits comfortably in u64, but
    // their product does not).
    let p = Integer::from((6 * k - 5) as u64)
        * Integer::from((2 * k - 1) as u64)
        * Integer::from((6 * k - 1) as u64);
    let mut p = p;
    p.neg_assign();
    let k_big = Integer::from(k);
    let k2 = Integer::from(&k_big * &k_big);
    let q = k2 * &k_big * Integer::from(C3_OVER_24);
    let t = &p * Integer::from(A + B * (k as i64));
    (p, q, t)
}

/// Merge two binary-splitting triples without cloning the big operands.
fn merge(
    (p1, q1, t1): (Integer, Integer, Integer),
    (p2, q2, t2): (Integer, Integer, Integer),
) -> (Integer, Integer, Integer) {
    let p = Integer::from(&p1 * &p2);
    let q = Integer::from(&q1 * &q2);
    let t = Integer::from(&t1 * &q2) + Integer::from(&p1 * &t2);
    (p, q, t)
}

/// `isqrt(x)` = floor(sqrt(x)) for a non-negative integer.
fn isqrt(x: &Integer) -> Integer {
    x.clone().sqrt()
}

/// Compute the decimal digits of π to `cfg.digits` places (truncated).
///
/// Returns a [`PiResult`]; `scaled` holds `floor(π·10^(digits+guard))` and
/// `hex_scaled` holds `floor(π·16^hex_len)` for verification.
pub fn compute_pi(cfg: &PiConfig) -> PiResult {
    let digits = cfg.digits;
    let guard = cfg.guard;
    let (q, t, n_terms) = chudnovsky_split(cfg);
    let q2 = Integer::from(&q * &q);

    // The irrational root `R = isqrt(426880²·10005·Q²)` is shared by both the
    // decimal and hexadecimal scalings. Because `T ≫ base^exp` (see
    // [`scaled_integer`]), this is exact for the requested number of digits.
    let r = chudnovsky_root(&q2);

    let d = digits + guard;
    let scaled = scaled_integer(&r, &t, 10, d);

    // The hex-scaled integer for verification, with a number of hex digits
    // that covers the whole computed range plus margin.
    let hex_len = hex_digit_capacity(d);
    let hex_scaled = scaled_integer(&r, &t, 16, hex_len);

    PiResult { scaled, hex_scaled, digits, guard, n_terms, hex_len }
}

/// Run the parallel Chudnovsky binary splitting and return `(Q, T, n_terms)`.
///
/// The numerator/dummy `P` is not needed to obtain the final result (π uses
/// only `Q` and `T`), so it is dropped as soon as the merge completes to save
/// memory.
pub fn chudnovsky_split(cfg: &PiConfig) -> (Integer, Integer, usize) {
    let n_terms = cfg.n_terms();
    let threads = cfg.effective_threads();
    let (_p, q, t) = chudnovsky_parallel(n_terms, threads);
    (q, t, n_terms)
}

/// `base^integer` as a rug `Integer`, materialising the (huge) power.
fn integer_pow(base: u64, exp: usize) -> Integer {
    Integer::from(Integer::u_pow_u(base as u32, exp as u32))
}

/// `R = isqrt(426880²·10005·Q²)`.
///
/// This is the integer part of `√10005 · 426880 · Q`, i.e. the irrational
/// mantissa of π before the rational denominator `T`. It is shared between the
/// decimal and hexadecimal scalings.
pub fn chudnovsky_root(q2: &Integer) -> Integer {
    let x = Integer::from(PI_K).square() * Integer::from(TEN_THOUSAND_FIVE) * q2;
    isqrt(&x)
}

/// `floor(π · base^exp)` computed exactly, given `R = isqrt(426880²·10005·Q²)`
/// and `t = T`.
///
/// We use `π = (R+δ)/T` with `0 ≤ δ < 1` (since `R = floor(√(426880²·10005·Q²))`
/// and `T` is the Chudnovsky denominator). Then
/// `floor(π·base^exp) = floor(R·base^exp/T)` provided `δ·base^exp/T < 1`. Since
/// `T ≫ base^exp` (the Chudnovsky denominator is vastly larger than the output
/// in magnitude), this correction is astronomically small, so the result is
/// exact for all the digits we keep. This avoids computing the huge
/// `base^(2·exp)` power and the very large `isqrt(base^(2·exp)·…)` that a
/// naive scaling would require.
pub fn scaled_integer(r: &Integer, t: &Integer, base: u64, exp: usize) -> Integer {
    let t0 = Instant::now();
    let base_pow = integer_pow(base, exp);
    let t1 = Instant::now();
    let scaled = r * base_pow;
    let t2 = Instant::now();
    let res = Integer::from(scaled / t);
    let t3 = Instant::now();
    if std::env::var("PI_SCALE_TIMING").is_ok() {
        eprintln!(
            "scale base={base} exp={exp}: pow={:?} mul={:?} div={:?}",
            t1 - t0,
            t2 - t1,
            t3 - t2
        );
    }
    res
}

/// `log10(16)` inverse = 1/log10(16) ≈ 0.83048 (converts decimal digits to hex
/// digit count).
const LOG16_INV: f64 = 1.0 / 1.2041199826559248;

/// The number of hexadecimal digits of π needed to cover `d` decimal places
/// (plus a small margin).
pub fn hex_digit_capacity(d: usize) -> usize {
    ((d as f64) * LOG16_INV).ceil() as usize + 8
}

/// Parallel top-level driver. Splits `[0, n)` into `threads` contiguous
/// ranges, computes each independently on its own thread, and merges the
/// Parallel top-level driver for the Chudnovsky binary splitting.
///
/// We recurse with `rayon::join` so that the *entire* binary-splitting tree —
/// including the large merges near the top, which were previously serialised
/// — is scheduled across all cores by work-stealing. This both removes the
/// serial left-fold merge that dominated the top levels and dynamically
/// balances the load (no fixed per-thread assignment that can be imbalanced).
///
/// `this_threads` is the desired worker count.
pub fn chudnovsky_parallel(
    n: usize,
    this_threads: usize,
) -> (Integer, Integer, Integer) {
    let threads = this_threads.max(1).min(n.max(1));
    if threads == 1 || n <= 1 {
        return chudnovsky_bs(0, n);
    }
    // Split into enough independent subtrees to give the scheduler plenty of
    // work to steal without excessive task overhead. ~threads*4 leaves is far
    // more than we can run concurrently, which lets work-stealing balance.
    let leaf = ((n / (threads * 4)).max(64)) as usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    pool.install(|| bs_par(0, n, leaf))
}

/// Recursively binary-split `[a,b)` in parallel using `rayon::join` until the
/// range is small enough to compute serially. The merge is exact and
/// associative, so the result is deterministic regardless of scheduling.
fn bs_par(a: usize, b: usize, leaf: usize) -> (Integer, Integer, Integer) {
    if b - a <= leaf {
        return chudnovsky_bs(a, b);
    }
    let m = a + (b - a) / 2;
    let (left, right) = rayon::join(|| bs_par(a, m, leaf), || bs_par(m, b, leaf));
    merge(left, right)
}

/// Brute-force the Chudnovsky partial sum in exact rational arithmetic for
/// validation only.
#[cfg(test)]
fn brute_force_sum(n: usize) -> rug::Rational {
    use rug::Rational;
    let mut sum = Rational::from(0);
    for k in 0..n {
        let k = Integer::from(k);
        let sixk = Integer::from(6) * &k;
        let threek = Integer::from(3) * &k;
        let f6 = factorial(&sixk);
        let f3 = factorial(&threek);
        let fk = factorial(&k);
        let fk3 = Integer::from(&fk * &fk) * &fk;
        let numer_part = Integer::from(&f6 / Integer::from(&f3 * &fk3));
        let sgn = if k.to_u32().unwrap() % 2 == 0 { 1 } else { -1 };
        let lin = A as i128 + B as i128 * k.to_i128().unwrap();
        // C3^k computed as an Integer (C3 does not fit in a u32 base).
        let c3 = Integer::from(C3 as u128);
        let mut den = Integer::from(1);
        for _ in 0..k.to_u32().unwrap() {
            den *= &c3;
        }
        let term = Rational::from((
            Integer::from(sgn as i64) * Integer::from(lin) * numer_part,
            den,
        ));
        sum += term;
    }
    sum
}

#[cfg(test)]
fn factorial(x: &Integer) -> Integer {
    let mut r = Integer::from(1);
    let mut i = Integer::from(1);
    while &i <= x {
        r *= &i;
        i += 1;
    }
    r
}

/// The exact rational `T/Q` from the binary splitting for `[a,b)`.
#[cfg(test)]
fn split_sum(a: usize, b: usize) -> rug::Rational {
    let (_, q, t) = chudnovsky_bs(a, b);
    rug::Rational::from((t, q))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_split_matches_brute_force() {
        for n in [5usize, 20usize, 50usize, 100usize] {
            let direct = brute_force_sum(n);
            let split = split_sum(0, n);
            let diff = (direct - split).abs();
            let eps = rug::Rational::from((
                Integer::from(1),
                Integer::from(Integer::u_pow_u(10, 40)),
            ));
            assert!(diff < eps, "n={n}: diff={diff}");
        }
    }

    #[test]
    fn binary_split_is_thread_invariant() {
        for n in [1usize, 7, 40, 100] {
            let seq = split_sum(0, n);
            let (_p, q, t) = chudnovsky_parallel(n, 4);
            let par = rug::Rational::from((t, q));
            assert_eq!(seq, par, "parallel mismatch at n={n}");
        }
    }

    #[test]
    fn first_digits_match_known() {
        let cfg = PiConfig::new(50, 4, 32);
        let res = compute_pi(&cfg);
        let s = res.decimal_string();
        let known = format!("3.{}", crate::verify::first_50_string());
        assert_eq!(s, known, "first 50 digits mismatch");
    }

    /// Regression test for a real bug: the leaf numerator
    /// `(6k−5)(2k−1)(6k−1)` overflows `u64` for `k ≳ 635,000`. This test
    /// asserts the leaf for a large `k` is computed with arbitrary-precision
    /// integers (not fixed-width), exactly matching the true product.
    #[test]
    fn leaf_multiplies_as_bigint_no_overflow() {
        let k = 705_155usize;
        let (p, _q, _t) = leaf(k);
        let expected_p = -(Integer::from((6 * k - 5) as u64)
            * Integer::from((2 * k - 1) as u64)
            * Integer::from((6 * k - 1) as u64));
        assert_eq!(p, expected_p, "leaf p_k overflow regression");
        // Also sanity-check q_k = k^3 · C3/24 for a large k (also must not
        // overflow).
        let (_, q, _) = leaf(k);
        let kk = Integer::from(k as u64);
        let expected_q = Integer::from(&kk * &kk) * &kk * Integer::from(C3_OVER_24);
        assert_eq!(q, expected_q, "leaf q_k overflow regression");
    }

    /// First 1,000 digits must be consistent with the hardcoded first 50 AND
    /// with the independent BBP extraction (this is the required 1,000-digit
    /// validation: hardcoded-50 + BBP, since we do not hardcode beyond 50).
    #[test]
    fn first_1000_digits_pass_verification() {
        let cfg = PiConfig::new(1_000, 4, 32);
        let res = compute_pi(&cfg);
        let digits = res.decimal_digits();
        assert_eq!(digits.len(), 1_000);
        // Hardcoded first 50.
        assert_eq!(&digits[..50], crate::verify::first_50_string());
        // Independent BBP checks across the range (must all agree).
        let report = crate::verify::run_verification(&res, &digits, &[], 4);
        assert!(report.all_pass(), "1,000-digit verification failed:\n{}", report.render());
    }

    /// This test would catch an off-by-one in the digit-indexing convention.
    ///
    /// Convention (documented in the README): position 1 is the FIRST digit
    /// after the decimal point, i.e. the `1` in `3.1415926535897…`.
    /// 0-based array index `i` corresponds to position `i+1`.
    #[test]
    fn digit_indexing_is_one_based() {
        let cfg = PiConfig::new(60, 2, 32);
        let res = compute_pi(&cfg);
        let d = res.decimal_digits();
        let known = crate::verify::first_50_string();
        // Position 1 must be the '1' of 3.14159… and must NOT be the integer
        // part '3' (a 0-based/left-shifted indexing bug would give the wrong
        // first digit).
        assert_eq!(d.as_bytes()[0], b'1', "position 1 must be the first digit after the point");
        // For every allowed position p (1-based), d[p-1] must equal the known
        // digit at that position.
        for p in 1..=known.len() {
            let expected = known.as_bytes()[p - 1];
            let actual = d.as_bytes()[p - 1];
            assert_eq!(actual, expected, "digit at position {p} is wrong");
        }
        // Also verify the length matches exactly the requested digit count.
        assert_eq!(d.len(), cfg.digits);
    }
}
