//! End-to-end integration test: compute π to 10,000 decimal places, write the
//! result, and run all verification layers. This exercises the full pipeline
//! (binary splitting → binary scaling → decimal scaling → base conversion →
//! file write → verification).

use pi::chudnovsky::compute_pi;
use pi::convert::to_decimal_digits_after_first;
use pi::verify::{self, Checkpoint, Status, VerifyReport};
use pi::PiConfig;
use std::io::Write;

#[test]
fn end_to_end_10000_digits() {
    let digits = 10_000usize;
    let cfg = PiConfig::new(digits, 4, 32);
    let result = compute_pi(&cfg);

    let truncated = result.decimal_truncated();
    let ds = to_decimal_digits_after_first(&truncated);

    // The decimal digit string has exactly `digits` digits after "3.".
    assert_eq!(ds.len(), digits, "digit count must equal requested digits");

    // First 50 digits must match the hardcoded known value.
    assert_eq!(&ds[..50], verify::first_50_string(), "first 50 digits mismatch");

    // Position 1 (the first digit after the point) is '1', not the integer '3'.
    assert_eq!(ds.as_bytes()[0], b'1', "indexing off-by-one");

    // Write to a temp file and check the exact byte layout: "3." + N digits,
    // no newlines or spaces.
    let mut path = std::env::temp_dir();
    path.push(format!("pi_it_{}.txt", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp file");
    f.write_all(b"3.").unwrap();
    f.write_all(ds.as_bytes()).unwrap();
    f.flush().unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(content.len(), 2 + digits);
    assert!(content.starts_with("3."));
    assert_eq!(content.len(), content.trim_end().len(), "no trailing whitespace");
    assert!(!content.contains(' '), "no spaces allowed");

    // Run all verification layers. No check may FAIL.
    let report: VerifyReport = verify::run_verification(&result, &truncated, &ds, &[], 4);
    assert!(
        !report.has_failure(),
        "verification should not fail:\n{}",
        report.render()
    );

    // A deliberately corrupted digit must be caught by the decimal checkpoint
    // and by the modular conversion check.
    let mut corrupt = ds.clone().into_bytes();
    corrupt[5000] = if corrupt[5000] == b'9' { b'8' } else { b'9' };
    let corrupt = String::from_utf8(corrupt).unwrap();
    let cp = vec![Checkpoint { position: 5001, digits: (ds.as_bytes()[5000] as char).to_string() }];
    let bad = verify::verify_decimal(&corrupt, &cp);
    assert!(
        bad.iter().any(|c| c.status == Status::Fail),
        "checkpoint corruption must be detected"
    );
    let conv = verify::verify_conversion(&truncated, &corrupt, 4);
    assert!(
        conv.iter().any(|c| c.status == Status::Fail),
        "conversion corruption must be detected"
    );
}
