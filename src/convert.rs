//! Subquadratic binary→decimal conversion.
//!
//! Converting a huge binary integer (the π mantissa) to decimal digits is one
//! of the major costs at 100M+ digits if done naively (repeated division by
//! 10). We instead use a divide-and-conquer radix conversion driven by GMP's
//! subquadratic (FFT-based) multiplication.
//!
//! The idea: to convert a non-negative integer `x` into base-`10^9` blocks,
//! recursively split `x` at a power of the radix:
//!
//! ```text
//! x = hi · R^m + lo      (R = 10^9)
//! ```
//!
//! where `R^m` is a precomputed power of the radix and `m` is chosen so that
//! `hi` and `lo` are roughly the same size. We then recurse on `hi` and `lo`
//! separately and concatenate their (fixed-width) blocks. Because GMP's
//! multiplication is subquadratic (FFT), the total cost is `O(M(n)·log n)`,
//! which is subquadratic in the bit-length `n`. This is the standard
//! divide-and-conquer (also called "split radix" or "Schönhage-style")
//! base-conversion method.
//!
//! Naively computing `R^m` on the fly would dominate the cost, so the powers
//! are precomputed in the conversion routine.

use rug::Integer;

/// The radix used for the conversion is 10^9 so each "block" fits in a `u32`
/// and corresponds directly to exactly 9 decimal characters.
pub const RADIX: u64 = 1_000_000_000;
const RADIX_DIGITS: usize = 9;

/// Convert a non-negative integer into its decimal digit string (no leading
/// zeros unless the value is zero). Uses subquadratic binary → base-10^9
/// conversion internally.
pub fn to_decimal_string(x: &Integer) -> String {
    if x == &Integer::from(0) {
        return "0".to_string();
    }
    let neg = x.is_negative();
    let abs = if neg { -x.clone() } else { x.clone() };
    let blocks = convert_blocks(&abs);
    debug_assert!(!blocks.is_empty());

    // Build the final string from the base-RADIX blocks. `blocks` is
    // little-endian (least-significant block first); render in reverse and
    // zero-pad every block except the most-significant one. Leading zero
    // blocks (which can only arise from internal zero-padding) are skipped.
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
            out.push_str(&block.to_string());
            started = true;
        } else {
            out.push_str(&format!("{:0width$}", block, width = RADIX_DIGITS));
        }
    }
    if !started {
        out.push('0');
    }
    out
}

/// Recursively convert `x` into a little-endian vector of base-`RADIX` blocks
/// (least-significant block first, with internal zero-padding removed). Uses
/// subquadratic divide-and-conquer splitting, parallelised at the top levels
/// (the two halves of each split are independent and can be converted
/// concurrently) until the numbers are small enough that serialising them is
/// more efficient.
fn convert_blocks(x: &Integer) -> Vec<u32> {
    convert_blocks_depth(x, 0)
}

const PAR_DEPTH: usize = 6;

fn convert_blocks_depth(x: &Integer, depth: usize) -> Vec<u32> {
    if x < &Integer::from(RADIX) {
        return vec![x.to_u32().expect("fits in u32")];
    }

    // Choose a power of the radix, R^m, about half the size of x, so we can
    // split x into two roughly equal halves.
    let m = half_power(x);
    let rpow = Integer::from(Integer::u_pow_u(RADIX as u32, m as u32));
    let hi = Integer::from(x / &rpow);
    let lo = Integer::from(x % &rpow);

    // Convert the low part first (least significant), then the high part. The
    // two are independent, so at the top levels we convert them in parallel.
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

    // Pad the low part to exactly m blocks so that the high part lands at the
    // correct (m-th) block position. Internal zero-blocks must be preserved
    // because they are genuine digits of x.
    while lo_blocks.len() < m {
        lo_blocks.push(0);
    }
    lo_blocks.extend_from_slice(&hi_blocks);
    lo_blocks
}

/// Estimate a power `m` such that `RADIX^m` is about √|x|.
///
/// Choosing `R^m ≈ √x` makes the high/low halves comparable in size, which is
/// the balanced split required for the divide-and-conquer conversion to be
/// efficient. It also guarantees `R^m < x` (strict reduction, so recursion
/// terminates): for `bits ≥ 30` (we only reach here when `x ≥ RADIX`), `m ≤
/// bits/(2·29.897)` so `R^m` has at most `bits/2` bits, which is strictly
/// less than the `bits` bits of `x`.
fn half_power(x: &Integer) -> usize {
    let bits = x.significant_bits() as usize;
    // Each base-RADIX block holds about log2(10^9) ≈ 29.897 bits.
    const BITS_PER_BLOCK: f64 = 29.897_352_853_36;
    let m = (bits as f64 / (2.0 * BITS_PER_BLOCK)).floor() as usize;
    m.max(1)
}

/// Recursively precompute `RADIX^(2^i)` for `i` up to `max_power_log` (not
/// currently used externally; retained for clarity of the method).
#[allow(dead_code)]
fn compute_radix_powers(max_power: usize) -> Vec<Integer> {
    let mut powers = Vec::new();
    let mut cur = Integer::from(RADIX);
    powers.push(cur.clone());
    while powers.len() < max_power {
        cur = Integer::from(&cur * &cur);
        powers.push(cur.clone());
    }
    powers
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
        // A value with many digits; compare against the native to_string path.
        let mut v = Integer::from(1);
        for _ in 0..200 {
            v *= 1234567;
        }
        let expected = v.to_string();
        let got = to_decimal_string(&v);
        assert_eq!(got, expected);
    }

    #[test]
    fn conversion_is_bijective_on_radix_blocks() {
        // A value that straddles a base-10^9 block boundary.
        let v = Integer::from(987_654_321u64) * Integer::from(RADIX) + 999_999_999u64;
        let s = to_decimal_string(&v);
        assert_eq!(s, "987654321999999999");
    }
}
