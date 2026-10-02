//! Diagnostic: compute the BBP hex digit at a given position using both
//! double precision and high-precision (rug::Float) accumulation, and compare
//! against the main computation's exact value, to determine which is wrong.
//!
//! Run: `cargo run --release --example bbp_prec -- <digits> <position>`

use pi::bbp;
use pi::chudnovsky::compute_pi;
use pi::PiConfig;
use rug::Float;

fn modpow(mut base: u64, mut exp: usize, m: u64) -> u64 {
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
            let t = modpow(16, e, m);
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
    let digits: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    let pos: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8_304_820);
    let cfg = PiConfig::new(digits, 8, 32);
    let res = compute_pi(&cfg);
    let main = res.main_hex_digit(pos);
    let (f64d, frac16) = bbp::hex_digit_checked(pos, 0);
    let hp = bbp_frac_highprec(pos, 256);
    let hp16: Float = Float::with_val(256, &hp * 16.0);
    let hp_digit = hp16.clone().floor().to_integer().unwrap().to_u32().unwrap().min(15);
    println!(
        "position {pos}: main={main:x}  bbp_f64={f64d:x} ({frac16:.6})  bbp_highprec={hp_digit:x} (16*frac={:.8})",
        Float::with_val(256, &hp16)
    );
}
