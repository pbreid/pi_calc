//! Verify the subquadratic base converter by round-tripping (decimal string ->
//! parse back -> compare to the original integer), which avoids GMP's very slow
//! native base-10 `to_string` at large sizes.
//!
//! Run: `cargo run --release --example convert_test -- <digits>`

use pi::convert::to_decimal_string;
use rug::Integer;

fn main() {
    let digits: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    // Build a pseudo-random large integer with roughly `digits` decimal digits
    // by repeatedly multiplying by a large constant (deterministic). Estimate
    // the digit count from the bit length to avoid a slow to_string in the loop.
    let mut v = Integer::from(31415926u64);
    let step = Integer::from(2_718_281_828_459_045u64);
    // Each multiplication by ~1e18 adds ~18 digits; stop when bit length is
    // enough (digits ≈ bits * log10(2)).
    let target_bits = ((digits as f64) / 0.301_029_995_663_98) as usize;
    let mut guard = 0;
    while (v.significant_bits() as usize) < target_bits {
        v *= &step;
        guard += 1;
        if guard > 500_000 {
            break;
        }
    }
    let mine = to_decimal_string(&v);
    let parsed = Integer::from_str_radix(&mine, 10).unwrap();
    if parsed == v {
        println!("digits≈{}: CONVERT OK (len={})", digits, mine.len());
    } else {
        println!("digits≈{}: CONVERT MISMATCH (len={})", digits, mine.len());
    }
}
