//! Command-line entry point for computing π with the Chudnovsky formula.
//!
//! This binary wires together the library modules: it parses the CLI, runs the
//! (parallel) binary splitting, computes the binary fixed-point value of π,
//! converts it to decimal digits, writes them to a file in a streaming fashion,
//! and (when `--verify` is given) runs the verification layers.

use clap::Parser;
use pi::chudnovsky::{self, PiConfig, PiResult};
use pi::convert;
use pi::output::DigitWriter;
use pi::verify;
use std::path::PathBuf;
use std::time::Instant;

/// Peak resident memory in KB read from /proc/self/status (`VmHWM`).
fn peak_mem_kb() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb = rest.trim().trim_end_matches("kB").trim();
            if let Ok(v) = kb.parse::<u64>() {
                return v;
            }
        }
    }
    0
}

fn fmt_mb(kb: u64) -> String {
    format!("{:.1} MB", kb as f64 / 1024.0)
}

#[derive(Parser, Debug)]
#[command(
    name = "pi",
    version,
    about = "Compute π to an arbitrary number of decimal places (Chudnovsky + binary splitting)."
)]
struct Args {
    /// Number of decimal places after the "3." (required).
    #[arg(long, value_name = "N")]
    digits: usize,

    /// Number of worker threads (default: all logical cores).
    #[arg(long, value_name = "T")]
    threads: Option<usize>,

    /// Number of guard digits computed internally beyond `--digits`
    /// (default: a small fixed guard).
    #[arg(long, value_name = "G")]
    guard: Option<usize>,

    /// Output file path (default: pi.txt).
    #[arg(long, default_value = "pi.txt", value_name = "PATH")]
    output: PathBuf,

    /// Run verification (BBP hex checks + modular conversion check + decimal
    /// checkpoints). Off by default: `--digits N` just computes and writes the
    /// digits.
    #[arg(long)]
    verify: bool,

    /// File of externally-sourced decimal checkpoints, one per line:
    /// `<position> <digits>` (e.g. `1000000 1`).
    #[arg(long, value_name = "PATH")]
    checkpoints: Option<PathBuf>,

    /// Print a per-phase timing breakdown and peak memory usage.
    #[arg(long)]
    bench: bool,
}

struct Timings {
    series: std::time::Duration,
    scale_binary: std::time::Duration,
    scale_decimal: std::time::Duration,
    conversion: std::time::Duration,
    write: std::time::Duration,
    verify: std::time::Duration,
    verify_conversion_check: std::time::Duration,
}

fn main() {
    let args = Args::parse();
    match run(&args) {
        Ok(code) => std::process::exit(code),
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(2);
        }
    }
}

fn run(args: &Args) -> Result<i32, String> {
    if args.digits == 0 {
        return Err("--digits must be >= 1".to_string());
    }
    let threads = args.threads.unwrap_or(0);
    let guard = args.guard.unwrap_or(0);
    let cfg = PiConfig::new(args.digits, threads, guard);
    let d = cfg.digits + cfg.guard;
    // Verification is opt-in via `--verify`. Supplying `--checkpoints` also
    // implies verification (that is the only reason to provide them).
    let verify_on = args.verify || args.checkpoints.is_some();

    // --- Phase 1: series evaluation (parallel binary splitting) ---
    let t0 = Instant::now();
    let (q, t, n_terms) = chudnovsky::chudnovsky_split(&cfg);
    let t1 = Instant::now();

    // --- Phase 2: final scaling to a binary fixed-point value M ≈ π·2^W ---
    let (w_bits, hex_len) = chudnovsky::working_precision(d);
    let binary = chudnovsky::binary_pi_fixed(&q, &t, w_bits);
    let t2 = Instant::now();

    let result = PiResult {
        binary,
        w_bits,
        digits: cfg.digits,
        guard: cfg.guard,
        n_terms,
        hex_len,
    };

    // --- Phase 3: decimal scaling: floor(π·10^digits) as an integer ---
    let truncated = result.decimal_truncated();
    let t3 = Instant::now();

    // --- Phase 4: binary -> decimal base conversion ---
    let digits = convert::to_decimal_digits_after_first(&truncated);
    let t4 = Instant::now();

    // --- Phase 5: streaming file write ---
    let mut writer = DigitWriter::create(&args.output)
        .map_err(|e| format!("cannot open output file {}: {e}", args.output.display()))?;
    writer
        .write_prefix()
        .and_then(|_| writer.write_digits(digits.as_bytes()))
        .map_err(|e| format!("error writing output file {}: {e}", args.output.display()))?;
    let written = writer
        .finish()
        .map_err(|e| format!("error flushing output file {}: {e}", args.output.display()))?;
    let t5 = Instant::now();
    if written != 2 + cfg.digits {
        return Err(format!("wrote {} bytes, expected {}", written, 2 + cfg.digits));
    }

    // --- Phase 6: verification ---
    let mut report = verify::VerifyReport::default();
    let mut conv_check = std::time::Duration::ZERO;
    if verify_on {
        let checkpoints = match &args.checkpoints {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read checkpoints {}: {e}", path.display()))?;
                verify::parse_checkpoints(&text)?
            }
            None => Vec::new(),
        };
        for c in verify::verify_bbp(&result, threads) {
            report.push(c);
        }
        let tc = Instant::now();
        for c in verify::verify_conversion(&truncated, &digits, threads) {
            report.push(c);
        }
        conv_check = tc.elapsed();
        for c in verify::verify_decimal(&digits, &checkpoints) {
            report.push(c);
        }
    }
    let t6 = Instant::now();

    // --- Reporting ---
    if args.bench {
        let timings = Timings {
            series: t1 - t0,
            scale_binary: t2 - t1,
            scale_decimal: t3 - t2,
            conversion: t4 - t3,
            write: t5 - t4,
            verify: t6 - t5,
            verify_conversion_check: conv_check,
        };
        print_bench(&cfg, &timings, verify_on);
    }

    eprintln!(
        "computed {} digits ({} terms, {} threads) -> {}",
        cfg.digits,
        n_terms,
        cfg.effective_threads(),
        args.output.display()
    );

    if verify_on {
        eprint!("{}", report.render());
        if report.has_failure() {
            eprintln!("verification: FAILURES DETECTED");
            Ok(1)
        } else {
            eprintln!("verification: NO FAILURES");
            Ok(0)
        }
    } else {
        Ok(0)
    }
}

fn print_bench(cfg: &PiConfig, t: &Timings, verify_on: bool) {
    let total: std::time::Duration =
        [t.series, t.scale_binary, t.scale_decimal, t.conversion, t.write]
            .iter()
            .copied()
            .sum::<std::time::Duration>()
            + t.verify;
    eprintln!("\n--- timing breakdown ---");
    eprintln!("digits:                    {}", cfg.digits);
    eprintln!("threads:                   {}", cfg.effective_threads());
    eprintln!("Chudnovsky series (BS):    {:>10.3?}", t.series);
    eprintln!("scaling (binary sqrt/div): {:>10.3?}", t.scale_binary);
    eprintln!("decimal scaling (10^D):    {:>10.3?}", t.scale_decimal);
    eprintln!("base conversion (10):      {:>10.3?}", t.conversion);
    eprintln!("file write:                {:>10.3?}", t.write);
    if verify_on {
        eprintln!("verification:              {:>10.3?}", t.verify);
        eprintln!("  └ conversion check:      {:>10.3?}", t.verify_conversion_check);
    }
    eprintln!("total (compute+write):     {:>10.3?}", total);
    eprintln!("peak memory (VmHWM):       {}", fmt_mb(peak_mem_kb()));
}
