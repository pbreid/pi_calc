//! Diagnostic: compare the main computation's exact hex digits against the BBP
//! extraction across a range of positions, to identify where (if anywhere) the
//! two diverge.
//!
//! Run: `cargo run --release --example diag -- <digits>`

use pi::bbp;
use pi::chudnovsky::compute_pi;
use pi::PiConfig;

fn main() {
    let digits: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let cfg = PiConfig::new(digits, 8, 32);
    let res = compute_pi(&cfg);
    println!("computed {digits} digits, hex_len={}", res.hex_len);

    // Sweep positions log-spaced across the range.
    let mut positions = vec![1usize, 2, 3, 16, 64, 512, 1_000, 10_000, 100_000];
    let end = res.hex_len.saturating_sub(1);
    for f in [4usize, 2, 1, 1] {
        let p = (end as f64 / f as f64).round() as usize;
        if p >= 1 && p <= end && !positions.contains(&p) {
            positions.push(p);
        }
    }
    positions.push(end);
    // Also include the hex position that maps to the last requested digit and
    // the final guard-edge positions.
    if digits > 0 {
        let dec_end = ((digits as f64) * (1.0 / 1.2041199826559248)).round() as usize;
        if dec_end >= 1 && dec_end <= end && !positions.contains(&dec_end) {
            positions.push(dec_end);
        }
        for d in 0..20usize {
            let p = end.saturating_sub(d);
            if !positions.contains(&p) {
                positions.push(p);
            }
        }
    }
    positions.sort_unstable();
    positions.dedup();

    let mut mismatches = 0;
    for &n in &positions {
        if n > end {
            continue;
        }
        let main = res.main_hex_digit(n);
        let (bbp_digit, frac) = bbp::hex_digit_checked(n, 0);
        let ok = main == bbp_digit;
        if !ok {
            mismatches += 1;
            println!(
                "n={n:>9}:  main={main:x}  bbp={bbp_digit:x}  (16*frac={frac:.6})  MISMATCH"
            );
        } else {
            println!("n={n:>9}:  main={main:x}  bbp={bbp_digit:x}  ok");
        }
    }
    println!("total mismatches: {mismatches} / {}", positions.len());
}
