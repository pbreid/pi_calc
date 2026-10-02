//! Verification layers for a computed π.
//!
//! Verification is **opt-in** via `--verify` (off by default). It consists of
//! three independent checks:
//!
//! A. **BBP hexadecimal spot checks**: independently recompute runs of
//!    hexadecimal digits of π with the Bailey–Borwein–Plouffe formula and
//!    compare against the hexadecimal digits of the main computation's binary
//!    result. This validates the *binary* value (series + scaling). A run whose
//!    value lies within the `f64` accumulation error bound of a digit boundary
//!    is reported `INCONCLUSIVE` rather than PASS/FAIL.
//!
//! B. **Modular conversion check**: the converted decimal string is validated
//!    against the exact truncated integer by reducing both modulo several large
//!    primes and comparing. BBP cannot see the base conversion, so this closes
//!    that gap (a bug in `convert.rs` would produce locally-wrong digits that
//!    BBP and sparse checkpoints would miss).
//!
//! C. **Decimal checkpoints**: compare the final decimal output against
//!    externally-supplied checkpoints, plus the hardcoded first 50 digits as a
//!    sanity check. Checkpoints beyond the computed range are `SKIPPED`.
//!
//! Each check is reported `PASS` / `FAIL` / `INCONCLUSIVE` / `SKIPPED`. A
//! failing check makes the process exit with a nonzero status (enforced by the
//! caller).

use crate::bbp;
use crate::chudnovsky::PiResult;
use rayon::prelude::*;
use rug::Integer;

/// The first 50 decimal places of π after the "3." (the only digits we are
/// allowed to hardcode). Position 1 is the first digit after the decimal
/// point.
pub const FIRST_50: &[u8] = b"14159265358979323846264338327950288419716939937510";

/// Large primes used for the modular conversion check: two Mersenne primes and
/// the largest prime below `2^64`.
const PRIMES: [u64; 3] = [
    2_305_843_009_213_693_951,  // 2^61 − 1
    18_446_744_073_709_551_557, // 2^64 − 59
    2_147_483_647,              // 2^31 − 1
];

/// Returns the first 50 decimal places as a `String` (for tests/display).
pub fn first_50_string() -> String {
    String::from_utf8_lossy(FIRST_50).into_owned()
}

/// Outcome of a single check.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Pass,
    Fail,
    /// The value is too close to a digit boundary for the method's precision to
    /// decide; neither a pass nor a failure.
    Inconclusive,
    /// The check could not be performed (e.g. checkpoint beyond range).
    Skipped,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Inconclusive => "INCONCLUSIVE",
            Status::Skipped => "SKIPPED",
        }
    }
}

/// A single decimal checkpoint: a 1-based `position` and the expected digit
/// string beginning at that position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub position: usize,
    pub digits: String,
}

/// Result of one check.
#[derive(Clone, Debug)]
pub struct CheckResult {
    pub kind: &'static str,
    pub position: usize,
    pub expected: String,
    pub actual: String,
    pub status: Status,
}

/// Aggregate verification report.
#[derive(Clone, Debug, Default)]
pub struct VerifyReport {
    pub checks: Vec<CheckResult>,
}

impl VerifyReport {
    /// True if any check failed (used to decide the exit code).
    pub fn has_failure(&self) -> bool {
        self.checks.iter().any(|c| c.status == Status::Fail)
    }

    /// True if every check passed (no FAIL, INCONCLUSIVE, or SKIPPED).
    pub fn all_pass(&self) -> bool {
        self.checks.iter().all(|c| c.status == Status::Pass)
    }

    pub fn push(&mut self, c: CheckResult) {
        self.checks.push(c);
    }

    /// Render a human-readable summary.
    pub fn render(&self) -> String {
        let mut s = String::new();
        for c in &self.checks {
            s.push_str(&format!(
                "[{}] {} position {}: expected {} got {}\n",
                c.status.label(),
                c.kind,
                c.position,
                c.expected,
                c.actual
            ));
        }
        s
    }
}

/// Parse a checkpoints file. Each line must be `<position> <digits>`, e.g.
/// `1000000 1`. Blank lines and `#` comments are ignored.
pub fn parse_checkpoints(text: &str) -> Result<Vec<Checkpoint>, String> {
    let mut out = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let pos = it
            .next()
            .ok_or_else(|| format!("line {}: missing position", lineno + 1))?
            .parse::<usize>()
            .map_err(|e| format!("line {}: invalid position: {e}", lineno + 1))?;
        let digits = it
            .next()
            .ok_or_else(|| format!("line {}: missing digits", lineno + 1))?;
        if pos == 0 {
            return Err(format!("line {}: position must be >= 1", lineno + 1));
        }
        if !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!(
                "line {}: digits `{digits}` are not all decimal digits",
                lineno + 1
            ));
        }
        if it.next().is_some() {
            return Err(format!("line {}: too many fields", lineno + 1));
        }
        out.push(Checkpoint { position: pos, digits: digits.to_string() });
    }
    if out.is_empty() {
        return Err("checkpoints file contains no entries".to_string());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// A. BBP hexadecimal run checks
// ---------------------------------------------------------------------------

/// Choose the number of hex digits to compare at position `n`.
///
/// We take the largest `k ≤ MAX_RUN` such that the f64 accumulation error,
/// scaled by `16^k`, is at most half a unit (`frac_error_bound(n)·16^k ≤ 0.5`).
/// This guarantees the run is unambiguous *for that value*; the per-position
/// boundary test below additionally downgrades to `INCONCLUSIVE` when the value
/// happens to sit close to a boundary.
fn choose_run_len(n: usize) -> usize {
    let e = bbp::frac_error_bound(n);
    let mut k = 0usize;
    let mut scale = 1.0f64;
    while k < bbp::MAX_RUN {
        let ns = scale * 16.0;
        if e * ns <= 0.5 {
            k += 1;
            scale = ns;
        } else {
            break;
        }
    }
    k.max(1)
}

/// Run the BBP hexadecimal verification against the main computation's binary
/// result. `threads` = 0 uses all available cores.
pub fn verify_bbp(result: &PiResult, threads: usize) -> Vec<CheckResult> {
    let h = result.hex_len.max(1);
    let core = if threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        threads
    };
    select_bbp_positions(result)
        .into_iter()
        .map(|n| {
            let max_k = (h - n + 1).max(1);
            let k = choose_run_len(n).min(max_k);
            let (bbp_run, err, dist) = bbp::hex_run(n, k, core);
            let main_run = result.main_hex_run(n, k);
            let status = if dist <= err {
                Status::Inconclusive
            } else if bbp_run == main_run {
                Status::Pass
            } else {
                Status::Fail
            };
            CheckResult {
                kind: "BBP",
                position: n,
                expected: format!("{:0width$x}", bbp_run, width = k),
                actual: format!("{:0width$x}", main_run, width = k),
                status,
            }
        })
        .collect()
}

/// Choose the hex positions to BBP-check: early positions, positions spread
/// across the range, and positions near the end (where precision errors would
/// surface).
fn select_bbp_positions(result: &PiResult) -> Vec<usize> {
    let h = result.hex_len.max(1);
    let mut v: Vec<usize> = Vec::new();
    for p in [1usize, 2, 3, 16, 64, 512] {
        if p <= h {
            v.push(p);
        }
    }
    // Interior positions spread across the whole range.
    for (num, den) in [(1usize, 8usize), (1, 4), (1, 2), (3, 4), (7, 8)] {
        let p = (h * num) / den;
        if p >= 1 && p <= h {
            v.push(p);
        }
    }
    // The hex position corresponding to the last requested decimal digit.
    let dec_end = ((result.digits as f64) / 1.204_119_982_655_924_8).round() as usize;
    if dec_end >= 1 && dec_end <= h {
        v.push(dec_end);
    }
    // Near the absolute end, choosing positions that still allow a MAX_RUN run.
    let end_start = h.saturating_sub(bbp::MAX_RUN).max(1);
    for d in 0..3usize {
        let p = end_start + d;
        if p <= h {
            v.push(p);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

// ---------------------------------------------------------------------------
// B. Modular conversion check
// ---------------------------------------------------------------------------

/// Verify that the decimal string is a faithful rendering of the truncated
/// integer `truncated`.
///
/// Both the integer and the string are reduced modulo each prime in [`PRIMES`]
/// and compared. A conversion bug that changes a digit will (with overwhelming
/// probability) change the string's residue.
pub fn verify_conversion(
    truncated: &Integer,
    digits_after_point: &str,
    threads: usize,
) -> Vec<CheckResult> {
    // The truncated integer's decimal form is "3" followed by the digits.
    let mut full = String::with_capacity(digits_after_point.len() + 1);
    full.push('3');
    full.push_str(digits_after_point);
    let bytes = full.as_bytes();

    PRIMES
        .iter()
        .map(|&p| {
            let expected =
                Integer::from(truncated % Integer::from(p)).to_u64().expect("fits u64");
            let actual = horner_mod(bytes, p, threads);
            let status = if expected == actual { Status::Pass } else { Status::Fail };
            CheckResult {
                kind: "convert-mod",
                position: 0,
                expected: expected.to_string(),
                actual: actual.to_string(),
                status,
            }
        })
        .collect()
}

/// Evaluate the decimal digit string modulo `p` with Horner's method,
/// parallelised by chunking the digits and combining the per-chunk values with
/// a precomputed power `10^chunk mod p`.
fn horner_mod(bytes: &[u8], p: u64, threads: usize) -> u64 {
    let len = bytes.len();
    if len == 0 {
        return 0;
    }
    let chunk = 1usize << 20; // 1M digits per chunk
    let first_len = if len % chunk == 0 { chunk } else { len % chunk };
    let nchunks = 1 + (len - first_len) / chunk;
    let p128 = p as u128;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(if threads == 0 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            threads.max(1)
        })
        .build()
        .expect("failed to build rayon pool");

    // Per-chunk value mod p (most-significant chunk first).
    let vals: Vec<u64> = pool.install(|| {
        (0..nchunks)
            .into_par_iter()
            .map(|i| {
                let start = if i == 0 { 0 } else { first_len + (i - 1) * chunk };
                let width = if i == 0 { first_len } else { chunk };
                let end = (start + width).min(len);
                let mut v: u64 = 0;
                for &b in &bytes[start..end] {
                    v = ((v as u128 * 10 + (b - b'0') as u128) % p128) as u64;
                }
                v
            })
            .collect()
    });

    // Combine: acc = Σ chunk_i · (10^chunk)^(nchunks-1-i), most-significant
    // first. Every chunk after the first has exactly `chunk` digits.
    let pow = bbp::modpow(10, chunk, p);
    let mut acc = vals[0];
    for &v in &vals[1..] {
        acc = ((acc as u128 * pow as u128 + v as u128) % p128) as u64;
    }
    acc
}

// ---------------------------------------------------------------------------
// C. Decimal checkpoints
// ---------------------------------------------------------------------------

/// Verify the computed decimal digit string against the hardcoded first-50
/// sanity check and the supplied checkpoints.
///
/// `digits` is the exact digit sequence after the "3.". Checkpoints that lie
/// beyond the computed range are reported `SKIPPED` (not a failure).
pub fn verify_decimal(digits: &str, checkpoints: &[Checkpoint]) -> Vec<CheckResult> {
    let digits: &[u8] = digits.as_bytes();
    let mut out = Vec::new();

    // Sanity check: first 50 decimal places (hardcoded, the only allowed one).
    if digits.len() >= FIRST_50.len() {
        let actual: String = String::from_utf8_lossy(&digits[..FIRST_50.len()]).into_owned();
        let status = if actual == first_50_string() { Status::Pass } else { Status::Fail };
        out.push(CheckResult {
            kind: "decimal",
            position: 1,
            expected: first_50_string(),
            actual,
            status,
        });
    }

    for cp in checkpoints {
        let expected = &cp.digits;
        let start = cp.position.saturating_sub(1);
        let end = start + expected.len();
        if end > digits.len() {
            out.push(CheckResult {
                kind: "decimal",
                position: cp.position,
                expected: expected.clone(),
                actual: "(beyond computed range)".to_string(),
                status: Status::Skipped,
            });
            continue;
        }
        let actual = String::from_utf8_lossy(&digits[start..end]).into_owned();
        let status = if actual == *expected { Status::Pass } else { Status::Fail };
        out.push(CheckResult {
            kind: "decimal",
            position: cp.position,
            expected: expected.clone(),
            actual,
            status,
        });
    }

    out
}

/// Run all verification layers.
///
/// `truncated` is `result.decimal_truncated()`; `digits` is the digit string
/// after the "3.".
pub fn run_verification(
    result: &PiResult,
    truncated: &Integer,
    digits: &str,
    checkpoints: &[Checkpoint],
    threads: usize,
) -> VerifyReport {
    let mut report = VerifyReport::default();
    for c in verify_bbp(result, threads) {
        report.push(c);
    }
    for c in verify_conversion(truncated, digits, threads) {
        report.push(c);
    }
    for c in verify_decimal(digits, checkpoints) {
        report.push(c);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chudnovsky::{compute_pi, PiConfig};
    use crate::convert::to_decimal_digits_after_first;

    fn digits_of(res: &PiResult) -> (Integer, String) {
        let truncated = res.decimal_truncated();
        let digits = to_decimal_digits_after_first(&truncated);
        (truncated, digits)
    }

    #[test]
    fn main_hex_matches_bbp() {
        let cfg = PiConfig::new(2_000, 4, 32);
        let res = compute_pi(&cfg);
        let checks = verify_bbp(&res, 2);
        assert!(!checks.is_empty());
        assert!(
            !checks.iter().any(|c| c.status == Status::Fail),
            "BBP checks failed:\n{}",
            VerifyReport { checks }.render()
        );
    }

    #[test]
    fn primes_are_prime() {
        // Deterministic Miller–Rabin for 64-bit n.
        fn is_prime(n: u64) -> bool {
            fn mulmod(a: u64, b: u64, m: u64) -> u64 {
                (a as u128 * b as u128 % m as u128) as u64
            }
            fn powmod(mut b: u64, mut e: u64, m: u64) -> u64 {
                let mut r = 1u64 % m;
                b %= m;
                while e > 0 {
                    if e & 1 == 1 {
                        r = mulmod(r, b, m);
                    }
                    b = mulmod(b, b, m);
                    e >>= 1;
                }
                r
            }
            if n < 2 {
                return false;
            }
            for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
                if n % p == 0 {
                    return n == p;
                }
            }
            let mut d = n - 1;
            let mut s = 0u32;
            while d % 2 == 0 {
                d /= 2;
                s += 1;
            }
            for &a in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
                let mut x = powmod(a % n, d, n);
                if x == 1 || x == n - 1 {
                    continue;
                }
                let mut composite = true;
                for _ in 0..s - 1 {
                    x = mulmod(x, x, n);
                    if x == n - 1 {
                        composite = false;
                        break;
                    }
                }
                if composite {
                    return false;
                }
            }
            true
        }
        for &p in &PRIMES {
            assert!(is_prime(p), "{p} is not prime");
        }
    }

    #[test]
    fn checkpoint_parser_rejects_bad_input() {
        assert!(parse_checkpoints("1000000 1\n2000000 14159\n").unwrap().len() == 2);
        assert!(parse_checkpoints("").is_err());
        assert!(parse_checkpoints("0 1\n").is_err());
        assert!(parse_checkpoints("10 xyz\n").is_err());
        assert!(parse_checkpoints("10\n").is_err());
        assert!(parse_checkpoints("# comment\n\n10 5\n").unwrap().len() == 1);
    }

    /// The modular conversion check must catch a single corrupted digit in the
    /// middle of the string, while the BBP checks (which inspect only the
    /// binary value) still pass. This proves the conversion check covers a gap
    /// BBP cannot.
    #[test]
    fn conversion_check_catches_corruption_bbp_cannot() {
        let cfg = PiConfig::new(5_000, 4, 32);
        let res = compute_pi(&cfg);
        let (truncated, good) = digits_of(&res);

        // BBP checks pass (or are inconclusive) on the binary result.
        let bbp_checks = verify_bbp(&res, 2);
        assert!(
            !bbp_checks.iter().any(|c| c.status == Status::Fail),
            "BBP unexpectedly failed"
        );

        // Corrupt a digit in the middle of the decimal string.
        let mut bad = good.clone().into_bytes();
        let mid = bad.len() / 2;
        bad[mid] = if bad[mid] == b'9' { b'8' } else { b'9' };
        let bad = String::from_utf8(bad).unwrap();

        let conv_bad = verify_conversion(&truncated, &bad, 2);
        assert!(
            conv_bad.iter().any(|c| c.status == Status::Fail),
            "conversion check failed to detect corruption"
        );

        let conv_good = verify_conversion(&truncated, &good, 2);
        assert!(
            conv_good.iter().all(|c| c.status == Status::Pass),
            "conversion check should pass on correct digits"
        );
    }

    #[test]
    fn verify_decimal_detects_corruption() {
        let cfg = PiConfig::new(200, 2, 32);
        let res = compute_pi(&cfg);
        let (_t, good) = digits_of(&res);
        let orig = good.as_bytes()[59] as char;
        let mut corrupt = good.clone().into_bytes();
        corrupt[59] = if corrupt[59] == b'0' { b'1' } else { b'0' };
        let corrupt = String::from_utf8(corrupt).unwrap();
        let cp = Checkpoint { position: 60, digits: orig.to_string() };
        let bad = verify_decimal(&corrupt, std::slice::from_ref(&cp));
        assert!(bad.iter().any(|c| c.status == Status::Fail));
        let good_check = verify_decimal(&good, std::slice::from_ref(&cp));
        assert!(good_check.iter().all(|c| c.status == Status::Pass));
    }

    #[test]
    fn checkpoints_beyond_range_are_skipped_not_failed() {
        let cfg = PiConfig::new(1_000, 2, 32);
        let res = compute_pi(&cfg);
        let (_t, good) = digits_of(&res);
        let cp = Checkpoint { position: 5_000_000, digits: "1".to_string() };
        let checks = verify_decimal(&good, std::slice::from_ref(&cp));
        assert!(checks.iter().any(|c| c.status == Status::Skipped));
        assert!(!checks.iter().any(|c| c.status == Status::Fail));
    }

    #[test]
    fn full_verification_passes_and_has_no_failure() {
        let cfg = PiConfig::new(3_000, 4, 32);
        let res = compute_pi(&cfg);
        let (truncated, good) = digits_of(&res);
        let report = run_verification(&res, &truncated, &good, &[], 4);
        assert!(!report.has_failure(), "verification reported a failure:\n{}", report.render());
    }
}
