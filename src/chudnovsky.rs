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
//! For a range `[a,b)` we maintain integer triples `(P, Q, T)` defined so that
//! `T(a,b) / Q(a,b)` equals `Σ_{k=a}^{b-1} s_k`. The base case for a single
//! index `k≥1` uses the reduced factors
//!
//! ```text
//! p_k = (6k−5)·(2k−1)·(6k−1),      q_k = k³ · C3 / 24
//! ```
//!
//! The alternating sign `(−1)^k` is folded into `p_k`, and `k=0` is handled
//! specially (`s_0 = A`). The merge step is:
//!
//! ```text
//! Q(a,b) = Q(a,m)·Q(m,b)
//! T(a,b) = T(a,m)·Q(m,b) + P(a,m)·T(m,b)
//! P(a,b) = P(a,m)·P(m,b)            (only when needed)
//! ```
//!
//! Note that the numerator `P` of the *whole* range is never needed: `π`
//! depends only on `Q` and `T`, and the merge only ever needs the `P` of a
//! **left** child (it appears in `T(a,m)·Q(m,b) + P(a,m)·T(m,b)`). The right
//! spine of the recursion (nodes that are right children of a node whose `P`
//! is not needed) therefore never has to compute `P` at all. Skipping it saves
//! the single largest multiplication at the root merge.
//!
//! # Final scaling (working precision)
//!
//! `Q` and `T` are enormous — at 100M decimal digits `Q` has ~250M decimal
//! digits — far more than the output needs. Computing `isqrt(426880²·10005·Q²)`
//! and dividing by the full `T` therefore does ~2.5× more work than necessary.
//!
//! Instead we compute π as a **binary fixed-point integer** at the working
//! precision and scale *down* `Q` and `T` first. Let `D = digits + guard` and
//! choose `W` bits with `W ≥ D·log2(10) + 192` (and `W` a multiple of 4). Let
//! `Wm = W + 128`, and shift both `Q` and `T` right by
//! `a = max(0, max(bitlen Q, bitlen T) − Wm)`:
//!
//! ```text
//! Qh = Q >> a,   Th = T >> a
//! ```
//!
//! Writing `Q = Qh·2^a + ql`, `T = Th·2^a + tl` with `0 ≤ ql,tl < 2^a`, one
//! obtains
//!
//! ```text
//! |Qh/Th − Q/T| < 2^(−Wm)
//! ```
//!
//! (numerator `|Qh·tl − Th·ql| < 2^Wm`, denominator `≥ Th² ≈ 2^(2·Wm)`). Hence
//! `π` computed from the truncated operands differs from the true `π` by less
//! than `426880·√10005 · 2^(−Wm)`; scaled by `2^W` that is
//! `< 2^(25.3 + W − Wm) = 2^(−102.7)`, i.e. under one bit. Concretely we
//! compute
//!
//! ```text
//! G  = isqrt(426880²·10005·Qh²)      ≈ 426880·√10005·Qh
//! M  = floor(G · 2^W / Th)           ≈ π · 2^W
//! ```
//!
//! with `|M − π·2^W| < 2`. The binary fixed-point value `M` is the main
//! computation's *binary* result; the BBP verifier compares against it (and its
//! hex digits are simply its nibbles — no second division is required).
//!
//! To obtain decimal digits we use `10^D = 2^D·5^D`:
//!
//! ```text
//! floor(π·10^D) = floor(M·10^D / 2^W) = (M·5^D) >> (W − D)
//! ```
//!
//! The error in `M` is `< 2` bits, so the error in `π·10^D` is
//! `< 2·10^D/2^W ≤ 2^{1−192} < 2^{−190}`, which cannot change the floor. The
//! result is therefore exactly `floor(π·10^D)`.

use rug::ops::NegAssign;
use rug::Integer;

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
/// Decimal digits gained per Chudnovsky term.
pub const DIGITS_PER_TERM: f64 = 14.181_647_462_725_477;

/// Minimum extra terms of safety beyond the asymptotic requirement.
const TERM_SAFETY_MARGIN: usize = 16;

/// `log2(10)`.
const LOG2_10: f64 = 3.321_928_092_809_449;
/// `1/log10(16)` ≈ 0.83048 (converts decimal digits to hex digit count).
const LOG16_INV: f64 = 1.0 / 1.204_119_982_655_924_8;

/// Guard bits added beyond the decimal precision in the working scale `W`.
const WORK_BITS_GUARD: usize = 192;
/// Extra bits used when truncating `Q`/`T` (`Wm = W + TRUNC_GUARD`).
const TRUNC_GUARD: usize = 128;

/// A computed value of π.
pub struct PiResult {
    /// `M = floor(π · 2^w_bits)` — the main computation's *binary* result. Its
    /// hexadecimal representation is `"3"` followed by `w_bits/4` hex digits
    /// of π, so the BBP checks compare directly against this value.
    pub binary: Integer,
    /// Number of bits in the working fixed-point scale.
    pub w_bits: usize,
    /// Number of requested decimal places (digits after the "3.").
    pub digits: usize,
    /// Number of internal guard digits computed beyond `digits`.
    pub guard: usize,
    /// Number of Chudnovsky terms used.
    pub n_terms: usize,
    /// Number of hexadecimal digits represented by `binary` (= `w_bits/4`).
    pub hex_len: usize,
}

impl PiResult {
    /// The truncated integer `floor(π·10^digits)`, whose decimal representation
    /// is `"3"` followed by exactly `digits` digits.
    ///
    /// Uses `floor(π·10^D) = (M·5^D) >> (W − D)` (see the module docs), then
    /// removes the `guard` extra digits by `floor(·/10^guard)`.
    pub fn decimal_truncated(&self) -> Integer {
        let d = self.digits + self.guard;
        let pow5 = Integer::from(Integer::u_pow_u(5, d as u32));
        let scaled = Integer::from((&self.binary * pow5) >> (self.w_bits - d));
        let g = Integer::from(Integer::u_pow_u(10, self.guard as u32));
        Integer::from(scaled / g)
    }

    /// Return exactly `digits` decimal digits (after the "3.") as a `String`.
    ///
    /// The leading `"3"` of the integer is dropped while rendering, so no
    /// extra full-size copy of the digit buffer is produced.
    pub fn decimal_digits(&self) -> String {
        let truncated = self.decimal_truncated();
        crate::convert::to_decimal_digits_after_first(&truncated)
    }

    /// Returns the full decimal string `"3." + digits` for `self`.
    pub fn decimal_string(&self) -> String {
        let d = self.decimal_digits();
        format!("3.{d}")
    }

    /// Return the n-th hexadecimal digit (1-based, first digit after the
    /// decimal point) of π, extracted directly from the binary result `M`.
    ///
    /// `M` is `"3"` followed by `hex_len` hex digits, so position `n` is the
    /// nibble at bit offset `4·(hex_len − n)`.
    pub fn main_hex_digit(&self, n: usize) -> u8 {
        assert!(n >= 1 && n <= self.hex_len, "hex position {n} out of range");
        let shift = 4 * (self.hex_len - n);
        let mut v = self.binary.clone();
        v >>= shift;
        v.mod_u(16) as u8
    }

    /// Return the run of `k` hexadecimal digits of π starting at position `n`
    /// (1-based), as an integer (most-significant digit first).
    ///
    /// Requires `n + k - 1 <= hex_len`. Used by the BBP verifier.
    pub fn main_hex_run(&self, n: usize, k: usize) -> u64 {
        assert!(k >= 1 && n >= 1 && n + k - 1 <= self.hex_len, "hex run out of range");
        let shift = 4 * (self.hex_len - (n + k - 1));
        let mut v = self.binary.clone();
        v >>= shift;
        let mask = Integer::from((1u64 << (4 * k)) - 1);
        Integer::from(v & mask).to_u64().expect("fits u64")
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

/// A fixed default guard size.
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

/// The working precision for `D` decimal digits: returns `(w_bits, hex_len)`
/// where `w_bits` is a multiple of 4 and `hex_len = w_bits/4`.
pub fn working_precision(d: usize) -> (usize, usize) {
    let hex_need = ((d as f64) * LOG16_INV).ceil() as usize + 16;
    let bits_need = ((d as f64) * LOG2_10).ceil() as usize + WORK_BITS_GUARD;
    let w = ((hex_need * 4).max(bits_need) + 3) & !3usize;
    (w, w / 4)
}

/// Recursive sequential binary splitting over the range `[a, b)`.
///
/// `need_p` indicates whether the caller needs this node's `P`. When false the
/// `P` multiplication is skipped (the returned `P` is a harmless placeholder
/// `1`).
fn chudnovsky_bs(a: usize, b: usize, need_p: bool) -> (Integer, Integer, Integer) {
    debug_assert!(b > a);
    if b - a == 1 {
        return leaf(a);
    }
    let m = a + (b - a) / 2;
    // The left child's P is always needed for this node's T.
    let (p1, q1, t1) = chudnovsky_bs(a, m, true);
    let (p2, q2, t2) = chudnovsky_bs(m, b, need_p);
    merge((p1, q1, t1), (p2, q2, t2), need_p)
}

/// The binary-splitting leaf for a single index `k`.
fn leaf(k: usize) -> (Integer, Integer, Integer) {
    if k == 0 {
        // s_0 = A. The recurrence leaf contributes P=1, Q=1, T=A.
        return (Integer::from(1), Integer::from(1), Integer::from(A));
    }
    // p_k = (6k−5)(2k−1)(6k−1); q_k = k³·(C3/24).
    // The (−1)^k sign is folded into p_k.
    //
    // NOTE: the product (6k−5)(2k−1)(6k−1) exceeds u64 for k ≳ 635,000, so we
    // MUST multiply as big integers (each factor fits in u64, the product does
    // not).
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

/// Merge two binary-splitting triples. The `P` product is only computed when
/// `need_p` is true (see [`chudnovsky_bs`]).
fn merge(
    (p1, q1, t1): (Integer, Integer, Integer),
    (p2, q2, t2): (Integer, Integer, Integer),
    need_p: bool,
) -> (Integer, Integer, Integer) {
    let q = Integer::from(&q1 * &q2);
    let t = Integer::from(&t1 * &q2) + Integer::from(&p1 * &t2);
    let p = if need_p {
        Integer::from(&p1 * &p2)
    } else {
        Integer::from(1)
    };
    (p, q, t)
}

/// `isqrt(x)` = floor(sqrt(x)) for a non-negative integer.
fn isqrt(x: &Integer) -> Integer {
    x.clone().sqrt()
}

/// Compute the decimal digits of π to `cfg.digits` places (truncated).
pub fn compute_pi(cfg: &PiConfig) -> PiResult {
    let (q, t, n_terms) = chudnovsky_split(cfg);
    let d = cfg.digits + cfg.guard;
    let (w_bits, hex_len) = working_precision(d);
    let binary = binary_pi_fixed(&q, &t, w_bits);
    PiResult { binary, w_bits, digits: cfg.digits, guard: cfg.guard, n_terms, hex_len }
}

/// Run the parallel Chudnovsky binary splitting and return `(Q, T, n_terms)`.
pub fn chudnovsky_split(cfg: &PiConfig) -> (Integer, Integer, usize) {
    let n_terms = cfg.n_terms();
    let threads = cfg.effective_threads();
    let (_p, q, t) = chudnovsky_parallel(n_terms, threads);
    (q, t, n_terms)
}

/// Compute the binary fixed-point value `M ≈ floor(π·2^w_bits)` from `Q, T`,
/// using truncated operands (see the module docs for the error bound).
pub fn binary_pi_fixed(q: &Integer, t: &Integer, w_bits: usize) -> Integer {
    let wm = w_bits + TRUNC_GUARD;
    let bits = (q.significant_bits().max(t.significant_bits())) as usize;
    let a = bits.saturating_sub(wm);
    let qh = Integer::from(q >> a);
    let th = Integer::from(t >> a);
    // G = isqrt(426880²·10005·Qh²)
    let x = Integer::from(PI_K).square() * Integer::from(TEN_THOUSAND_FIVE)
        * Integer::from(&qh * &qh);
    let g = isqrt(&x);
    // M = floor(G · 2^W / Th)
    let shifted = Integer::from(&g << w_bits);
    Integer::from(shifted / &th)
}

/// Parallel top-level driver: rayon work-stealing recursion over the whole
/// split tree, with `P` skipped on the right spine.
pub fn chudnovsky_parallel(
    n: usize,
    this_threads: usize,
) -> (Integer, Integer, Integer) {
    let threads = this_threads.max(1).min(n.max(1));
    if threads == 1 || n <= 1 {
        return chudnovsky_bs(0, n, false);
    }
    let leaf = ((n / (threads * 4)).max(64)) as usize;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon pool");
    pool.install(|| bs_par(0, n, leaf, false))
}

/// Recursively binary-split `[a,b)` in parallel using `rayon::join` until the
/// range is small enough to compute serially. `need_p` propagates as described
/// in the module docs.
fn bs_par(a: usize, b: usize, leaf: usize, need_p: bool) -> (Integer, Integer, Integer) {
    if b - a <= leaf {
        return chudnovsky_bs(a, b, need_p);
    }
    let m = a + (b - a) / 2;
    let (left, right) = rayon::join(
        || bs_par(a, m, leaf, true),
        || bs_par(m, b, leaf, need_p),
    );
    merge(left, right, need_p)
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
    let (_, q, t) = chudnovsky_bs(a, b, false);
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
    /// `(6k−5)(2k−1)(6k−1)` overflows `u64` for `k ≳ 635,000`.
    #[test]
    fn leaf_multiplies_as_bigint_no_overflow() {
        let k = 705_155usize;
        let (p, _q, _t) = leaf(k);
        let expected_p = -(Integer::from((6 * k - 5) as u64)
            * Integer::from((2 * k - 1) as u64)
            * Integer::from((6 * k - 1) as u64));
        assert_eq!(p, expected_p, "leaf p_k overflow regression");
        let (_, q, _) = leaf(k);
        let kk = Integer::from(k as u64);
        let expected_q = Integer::from(&kk * &kk) * &kk * Integer::from(C3_OVER_24);
        assert_eq!(q, expected_q, "leaf q_k overflow regression");
    }

    /// First 1,000 digits must be consistent with the hardcoded first 50 AND
    /// with the independent BBP extraction.
    #[test]
    fn first_1000_digits_pass_verification() {
        let cfg = PiConfig::new(1_000, 4, 32);
        let res = compute_pi(&cfg);
        let truncated = res.decimal_truncated();
        let digits = crate::convert::to_decimal_digits_after_first(&truncated);
        assert_eq!(digits.len(), 1_000);
        assert_eq!(&digits[..50], crate::verify::first_50_string());
        let report = crate::verify::run_verification(&res, &truncated, &digits, &[], 4);
        assert!(report.all_pass(), "1,000-digit verification failed:\n{}", report.render());
    }

    /// This test would catch an off-by-one in the digit-indexing convention.
    #[test]
    fn digit_indexing_is_one_based() {
        let cfg = PiConfig::new(60, 2, 32);
        let res = compute_pi(&cfg);
        let d = res.decimal_digits();
        let known = crate::verify::first_50_string();
        assert_eq!(d.as_bytes()[0], b'1', "position 1 must be the first digit after the point");
        for p in 1..=known.len() {
            assert_eq!(d.as_bytes()[p - 1], known.as_bytes()[p - 1], "digit at position {p} is wrong");
        }
        assert_eq!(d.len(), cfg.digits);
    }

    /// The binary fixed-point value must approximate π·2^W to within 2.
    #[test]
    fn binary_fixed_scaling_is_accurate() {
        // Compare against the high-precision rational sum for a small case.
        let cfg = PiConfig::new(200, 2, 32);
        let res = compute_pi(&cfg);
        // Decimal output must match the known first 50 digits.
        assert_eq!(&res.decimal_digits()[..50], crate::verify::first_50_string());
    }
}
