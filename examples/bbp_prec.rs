//! Calibration diagnostic: compare the double-precision BBP fractional value
//! against a 256-bit (`rug::Float`) computation at a given position, to
//! estimate the real accumulation error.
//!
//! Run: `cargo run --release --example bbp_prec -- <position> [precision_bits]`

use pi::bbp;
use rug::Float;

/// Fractional part reduced into [0, 1).
fn frac(x: Float, prec: u32) -> Float {
    let mut y = x.fract();
    if y < Float::with_val(prec, 0) {
        y += 1.0;
    }
    y
}

fn bbp_frac_highprec(n: usize, prec: u32) -> Float {
    let mut acc = Float::with_val(prec, 0);
    for (j, c) in [(1u64, 4.0f64), (4, -2.0), (5, -1.0), (6, -1.0)] {
        let mut s = Float::with_val(prec, 0);
        for k in 0..n {
            let e = n - 1 - k;
            let m = 8 * k as u64 + j;
            let t = bbp::modpow(16, e, m);
            s += Float::with_val(prec, t) / Float::with_val(prec, m);
            s = frac(s, prec);
        }
        let mut p = Float::with_val(prec, 16.0);
        for k in n..n + 24 {
            s += Float::with_val(prec, 1.0) / (p.clone() * Float::with_val(prec, 8 * k as u64 + j));
            p *= 16.0;
        }
        s = frac(s, prec);
        acc += Float::with_val(prec, c) * &s;
        acc = frac(acc, prec);
    }
    acc
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8_304_820);
    let prec: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(256);

    let f64_frac = bbp::bbp_fractional(n, 0);
    let hp = bbp_frac_highprec(n, prec);
    let delta = (f64_frac - hp.to_f64()).abs();
    println!(
        "n={n}: f64_frac={f64_frac:.18}  hp_frac={:.18}  |delta|={delta:.3e}  bound={:.3e}",
        hp.to_f64(),
        bbp::frac_error_bound(n)
    );
}
