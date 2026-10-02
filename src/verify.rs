//! Two independent verification layers for a computed π.
//!
//! A. **BBP-based spot checks**: independently recompute selected hexadecimal
//!    digits of π with the Bailey–Borwein–Plouffe formula and compare against
//!    the hexadecimal digits of the main computation's binary result *before*
//!    decimal conversion. Because precision/truncation errors are most likely
//!    to appear near the *end* of the computed range, we deliberately include
//!    checks close to the final hex digit. These are an independent algorithm,
//!    so a match strongly corroborates correctness.
//!
//! B. **Decimal checkpoints**: compare the final decimal output against
//!    externally-supplied checkpoints. The only digits we hardcode are the
//!    first 50 decimal places (a sanity check); every other decimal checkpoint
//!    must come from the user-supplied file.
//!
//! Each check is reported PASS/FAIL; a failing check makes the process exit
//! with a nonzero status (enforced by the caller).

use crate::bbp;
use crate::chudnovsky::PiResult;

/// The first 50 decimal places of π after the "3." (the only digits we are
/// allowed to hardcode). Position 1 is the first digit after the decimal
/// point.
pub const FIRST_50: &[u8] = b"14159265358979323846264338327950288419716939937510";

/// Returns the first 50 decimal places as a `String` (for tests/display).
pub fn first_50_string() -> String {
    String::from_utf8_lossy(FIRST_50).into_owned()
}

/// A single decimal checkpoint: a 1-based `position` and the expected digit
/// string beginning at that position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub position: usize,
    pub digits: String,
}

/// Result of one check (BBP or decimal), reported to the user.
#[derive(Clone, Debug)]
pub struct CheckResult {
    pub kind: &'static str,
    pub position: usize,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
}

/// Aggregate verification report.
#[derive(Clone, Debug, Default)]
pub struct VerifyReport {
    pub checks: Vec<CheckResult>,
}

impl VerifyReport {
    pub fn all_pass(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    pub fn push(&mut self, c: CheckResult) {
        self.checks.push(c);
    }

    /// Render a human-readable PASS/FAIL summary.
    pub fn render(&self) -> String {
        let mut s = String::new();
        for c in &self.checks {
            let status = if c.passed { "PASS" } else { "FAIL" };
            s.push_str(&format!(
                "[{status}] {} position {}: expected {} got {}\n",
                c.kind, c.position, c.expected, c.actual
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

/// Run the BBP hexadecimal verification against the main computation's binary
/// (hex) result. `threads` = 0 uses all available cores.
///
/// We select several hex positions, including positions very close to the end
/// of the computed range, because insufficient guard/precision would manifest
/// there. The BBP digits are computed in parallel (the inner k-sum is spread
/// across all cores) and compared. We iterate the positions *sequentially* so
/// that only one thread pool is active at a time, avoiding oversubscription
/// from nested pools.
pub fn verify_bbp(result: &PiResult, threads: usize) -> Vec<CheckResult> {
    let positions = select_bbp_positions(result);
    let core_count = if threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        threads
    };
    // Each BBP digit uses the full thread pool for its inner sum; positions
    // are handled one at a time to avoid nested pool oversubscription.
    positions
        .into_iter()
        .map(|n| {
            let main = result.main_hex_digit(n);
            let (bbp_digit, _) = bbp::hex_digit_checked(n, core_count);
            CheckResult {
                kind: "BBP",
                position: n,
                expected: format!("{bbp_digit:x}"),
                actual: format!("{main:x}"),
                passed: bbp_digit == main,
            }
        })
        .collect()
}

/// Choose the hex positions to BBP-check. We include early positions, some
/// interior positions, and (critically) positions **near the end** of the
/// computed hex range where precision errors would surface.
fn select_bbp_positions(result: &PiResult) -> Vec<usize> {
    let hex_len = result.hex_len.max(1);
    let end = hex_len.saturating_sub(1); // last usable position
    let mut v: Vec<usize> = Vec::new();
    for p in [1usize, 2, 3, 16, 64, 512] {
        if p <= end {
            v.push(p);
        }
    }
    // Interior: near the hex position corresponding to the decimal end.
    let decimal_end_hex = ((result.digits as f64) * (1.0 / 1.2041199826559248)).round() as usize;
    if decimal_end_hex >= 1 && decimal_end_hex <= end {
        v.push(decimal_end_hex);
    }
    // Near the absolute end (guard-digit territory).
    for delta in [0usize, 1, 2].iter() {
        let p = end.saturating_sub(*delta);
        if p >= 1 && !v.contains(&p) {
            v.push(p);
        }
    }
    // Deduplicate and sort.
    v.sort_unstable();
    v.dedup();
    v
}

/// Verify the computed decimal digit string against the given checkpoints and
/// the hardcoded first-50 sanity check.
///
/// `digits` must be the exact digit sequence after the "3." (length = number
/// of requested places). Returns a vector of [`CheckResult`] (one per
/// checkpoint plus the first-50 check). A checkpoint requires the substring of
/// `digits` from `position` (1-based) to equal the expected digit string.
pub fn verify_decimal(digits: &str, checkpoints: &[Checkpoint]) -> Vec<CheckResult> {
    let digits: &[u8] = digits.as_bytes();
    let mut out = Vec::new();

    // Sanity check: first 50 decimal places (hardcoded, the only allowed one).
    if digits.len() >= FIRST_50.len() {
        let actual: String = String::from_utf8_lossy(&digits[..FIRST_50.len()]).into_owned();
        let passed = actual == first_50_string();
        out.push(CheckResult {
            kind: "decimal",
            position: 1,
            expected: first_50_string(),
            actual,
            passed,
        });
    }

    for cp in checkpoints {
        let expected = &cp.digits;
        let start = cp.position.saturating_sub(1);
        let end = start + expected.len();
        if end > digits.len() {
            // Not enough computed digits to check this far out -> fail.
            out.push(CheckResult {
                kind: "decimal",
                position: cp.position,
                expected: expected.clone(),
                actual: "(beyond computed range)".to_string(),
                passed: false,
            });
            continue;
        }
        let actual = String::from_utf8_lossy(&digits[start..end]).into_owned();
        let passed = actual == *expected;
        out.push(CheckResult {
            kind: "decimal",
            position: cp.position,
            expected: expected.clone(),
            actual,
            passed,
        });
    }

    out
}

/// Convenience wrapper that runs both verification layers.
///
/// `digits` is the computed decimal digit string after the "3.".
pub fn run_verification(
    result: &PiResult,
    digits: &str,
    checkpoints: &[Checkpoint],
    threads: usize,
) -> VerifyReport {
    let mut report = VerifyReport::default();
    for c in verify_bbp(result, threads) {
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

    #[test]
    fn main_hex_matches_bbp() {
        // The main computation's binary (hex) result must agree with the
        // independent BBP extraction at several positions.
        let cfg = PiConfig::new(500, 4, 32);
        let res = compute_pi(&cfg);
        for n in [1usize, 2, 3, 16, 200] {
            if n > res.hex_len {
                continue;
            }
            let main = res.main_hex_digit(n);
            let (bbp_digit, _) = bbp::hex_digit_checked(n, 2);
            assert_eq!(bbp_digit, main, "hex position {n} mismatch");
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

    #[test]
    fn verify_decimal_detects_corruption() {
        // Build a valid decimal digit string, then corrupt a digit and confirm
        // the verifier fails.
        let cfg = PiConfig::new(200, 2, 32);
        let res = compute_pi(&cfg);
        let good = res.decimal_digits();
        // Remember the original digit at position 60 (0-based index 59), then
        // corrupt it, and check against the original value -> must FAIL.
        let orig = good.as_bytes()[59] as char;
        let mut corrupt = good.clone().into_bytes();
        corrupt[59] = if corrupt[59] == b'0' { b'1' } else { b'0' };
        let corrupt = String::from_utf8(corrupt).unwrap();
        let cp = Checkpoint { position: 60, digits: orig.to_string() };
        let bad_check = verify_decimal(&corrupt, std::slice::from_ref(&cp));
        assert!(
            bad_check.iter().any(|c| !c.passed),
            "corrupted string should fail verification"
        );
        // And the uncorrupted string must pass the same check.
        let cp2 = Checkpoint { position: 60, digits: orig.to_string() };
        let good_check = verify_decimal(&good, std::slice::from_ref(&cp2));
        assert!(good_check.iter().all(|c| c.passed));
    }

    #[test]
    fn verify_decimal_passes_on_correct_digits() {
        let cfg = PiConfig::new(200, 2, 32);
        let res = compute_pi(&cfg);
        let good = res.decimal_digits();
        // Checkpoints at position 4 (digit '5' in 3.14159...) and beyond.
        let cps = vec![
            Checkpoint { position: 4, digits: "5".to_string() },
            Checkpoint { position: 1, digits: "1415".to_string() },
        ];
        let checks = verify_decimal(&good, &cps);
        assert!(checks.iter().all(|c| c.passed));
    }
}
