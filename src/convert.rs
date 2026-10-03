//! Subquadratic binary→decimal conversion.
//!
//! Converting a huge binary integer (the π mantissa) to decimal digits is one
//! of the major costs at 100M+ digits if done naively (repeated division by
//! 10). We instead use a divide-and-conquer radix conversion driven by GMP's
//! subquadratic (FFT-based) multiplication and division.
//!
//! The idea: to convert a non-negative integer `x` into base-`10^9` blocks,
//! recursively split `x` at a power of the radix:
//!
//! ```text
//! x = hi · R^m + lo      (R = 10^9)
//! ```
//!
//! where `m` is chosen with `R^m ≈ √x` so that `hi` and `lo` are roughly the
//! same size. We recurse on `hi` and `lo` separately and concatenate their
//! (fixed-width) blocks. Because GMP's operations are subquadratic (FFT), the
//! total cost is `O(M(n)·log n)`, subquadratic in the bit-length `n`. This is
//! the standard divide-and-conquer ("split radix") base-conversion method.
//!
//! Note: `R^m` is recomputed at each recursion node (one `u_pow_u` per node);
//! the recursion is only ~`log n` deep, so this is not a bottleneck. There is
//! **no** precomputed power table — an earlier doc comment claiming one was
//! incorrect and has been removed.
//!
//! The two halves of each split are independent, so the top `PAR_DEPTH` levels
//! are converted concurrently with `rayon::join`.

use rug::{Complete, Integer};

/// The radix used for the conversion is 10^9 so each "block" fits in a `u32`
/// and corresponds directly to exactly 9 decimal characters.
pub const RADIX: u64 = 1_000_000_000;
const RADIX_DIGITS: usize = 9;
/// Recursion depth below which the two halves are converted in parallel.
const PAR_DEPTH: usize = 6;

/// Convert a non-negative integer into its decimal digit string (no leading
/// zeros unless the value is zero).
pub fn to_decimal_string(x: &Integer) -> String {
    render_decimal(x, false)
}

/// Convert a non-negative integer into its decimal digits with the **first**
/// digit removed. Used to render `"3" + digits` directly into the `digits`
/// string without a second full-size copy of the buffer.
pub fn to_decimal_digits_after_first(x: &Integer) -> String {
    render_decimal(x, true)
}

/// Render `x` in base 10 into a single preallocated `String`, optionally
/// dropping the leading digit.
fn render_decimal(x: &Integer, skip_first: bool) -> String {
    if x == &Integer::from(0) {
        return if skip_first { String::new() } else { "0".to_string() };
    }
    let neg = x.is_negative();
    let abs = if neg { -x.clone() } else { x.clone() };
    let blocks = convert_blocks(&abs);
    debug_assert!(!blocks.is_empty());

    // `blocks` is little-endian. Leading zero blocks (produced by internal
    // zero-padding) are skipped; render most-significant first.
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    let mut started = false;
    for &block in blocks.iter().rev() {
        if !started {
            if block == 0 {
                continue;
            }
            let s = block.to_string();
            if skip_first {
                out.push_str(&s[1..]);
            } else {
                out.push_str(&s);
            }
            started = true;
        } else {
            push_padded_block(&mut out, block);
        }
    }
    if !started && !skip_first {
        out.push('0');
    }
    out
}

/// Append a base-`RADIX` block zero-padded to exactly 9 decimal digits,
/// without allocating a temporary `String`.
fn push_padded_block(out: &mut String, block: u32) {
    let mut buf = [b'0'; RADIX_DIGITS];
    let mut v = block;
    let mut i = RADIX_DIGITS;
    while i > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    // Safe: the buffer contains only ASCII digits.
    out.push_str(std::str::from_utf8(&buf).unwrap());
}

/// Recursively convert `x` into a little-endian vector of base-`RADIX` blocks
/// (least-significant block first, with leading zero blocks removed).
fn convert_blocks(x: &Integer) -> Vec<u32> {
    convert_blocks_depth(x, 0)
}

fn convert_blocks_depth(x: &Integer, depth: usize) -> Vec<u32> {
    // Fast small-value exit without allocating a comparison Integer (this is
    // the base case for every ~9-digit block, ~11M times at 100M digits).
    if let Some(v) = x.to_u64() {
        if v < RADIX {
            return vec![v as u32];
        }
    }

    let m = half_power(x);
    let rpow = Integer::from(Integer::u_pow_u(RADIX as u32, m as u32));
    // One fused division pass (mpz_tdiv_qr) instead of separate `/` and `%`,
    // which would each walk the full dividend.
    let (hi, lo) = x.div_rem_ref(&rpow).complete();

    // The low part is converted first and padded to exactly `m` blocks so the
    // high part lands at the correct block position. Internal zero blocks are
    // genuine digits and are preserved.
    let (mut lo_blocks, hi_blocks) = if depth < PAR_DEPTH {
        rayon::join(
            || convert_blocks_depth(&lo, depth + 1),
            || convert_blocks_depth(&hi, depth + 1),
        )
    } else {
        (
            convert_blocks_depth(&lo, depth + 1),
            convert_blocks_depth(&hi, depth + 1),
        )
    };
    while lo_blocks.len() < m {
        lo_blocks.push(0);
    }
    lo_blocks.extend_from_slice(&hi_blocks);
    lo_blocks
}

/// Estimate a power `m` such that `RADIX^m ≈ √x`.
///
/// This balances the two halves of the split and guarantees strict reduction
/// (`RADIX^m < x`), so recursion terminates: for `bits ≥ 30` (we only reach
/// here when `x ≥ RADIX`), `m ≤ bits/(2·log2(RADIX))` so `RADIX^m` has at most
/// `bits/2` bits.
fn half_power(x: &Integer) -> usize {
    let bits = x.significant_bits() as usize;
    const BITS_PER_BLOCK: f64 = 29.897_352_853_36; // log2(10^9)
    let m = (bits as f64 / (2.0 * BITS_PER_BLOCK)).floor() as usize;
    m.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_small_and_zero() {
        assert_eq!(to_decimal_string(&Integer::from(0)), "0");
        assert_eq!(to_decimal_string(&Integer::from(7)), "7");
        assert_eq!(to_decimal_string(&Integer::from(123456789)), "123456789");
        assert_eq!(to_decimal_string(&Integer::from(1_000_000_000u64)), "1000000000");
    }

    #[test]
    fn converts_large_matches_to_string() {
        let mut v = Integer::from(1);
        for _ in 0..200 {
            v *= 1234567;
        }
        assert_eq!(to_decimal_string(&v), v.to_string());
    }

    #[test]
    fn digits_after_first_drops_leading_digit() {
        assert_eq!(to_decimal_digits_after_first(&Integer::from(3_141_592_653u64)), "141592653");
        assert_eq!(to_decimal_digits_after_first(&Integer::from(3)), "");
        let mut v = Integer::from(1);
        for _ in 0..500 {
            v *= 999_983;
        }
        let full = to_decimal_string(&v);
        let tail = to_decimal_digits_after_first(&v);
        assert_eq!(tail, full[1..]);
    }

    #[test]
    fn conversion_is_bijective_on_radix_blocks() {
        let v = Integer::from(987_654_321u64) * Integer::from(RADIX) + 999_999_999u64;
        assert_eq!(to_decimal_string(&v), "987654321999999999");
    }
}
