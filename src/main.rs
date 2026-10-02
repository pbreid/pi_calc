//! Command-line entry point for computing π with the Chudnovsky formula.
//!
//! This binary wires together the library modules: it parses the CLI, runs the
//! (parallel) binary splitting, converts to decimal digits, writes them to a
//! file in a streaming fashion, and runs the two verification layers.

use clap::Parser;
use pi::chudnovsky::{self, PiConfig, PiResult};
use pi::output::DigitWriter;
use pi::verify;
use rug::Integer;
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
    /// (default: a small fixed guard). More guard digits increase confidence
    /// but cost work.
    #[arg(long, value_name = "G")]
    guard: Option<usize>,

    /// Output file path (default: pi.txt).
    #[arg(long, default_value = "pi.txt", value_name = "PATH")]
    output: PathBuf,

    /// Skip verification (verification is on by default).
    #[arg(long)]
    no_verify: bool,

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
    final_pow: std::time::Duration,
    hex_pow: std::time::Duration,
    conversion: std::time::Duration,
    write: std::time::Duration,
    verify: std::time::Duration,
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
    let hex_len = chudnovsky::hex_digit_capacity(d);
    let verify_on = !args.no_verify;

    // --- Phase 1: series evaluation (parallel binary splitting) ---
    let t0 = Instant::now();
    let (q, t, n_terms) = chudnovsky::chudnovsky_split(&cfg);
    let t1 = Instant::now();

    // --- Phase 2: final scaling (integer sqrt + division) ---
    let q2 = Integer::from(&q * &q);
    // The irrational root R = isqrt(426880²·10005·Q²) is shared by the decimal
    // and hex scalings (computed once, not twice).
    let r = chudnovsky::chudnovsky_root(&q2);
    let scaled = chudnovsky::scaled_integer(&r, &t, 10, d);
    let t2 = Instant::now();
    // The hex-scaled integer is only needed for the BBP verification; skip it
    // entirely when verification is disabled to save a large multiply+division.
    let (hex_scaled, used_hex_len) = if verify_on {
        (chudnovsky::scaled_integer(&r, &t, 16, hex_len), hex_len)
    } else {
        (Integer::from(1), 0usize)
    };
    let t3 = Instant::now();

    let result = PiResult {
        scaled,
        hex_scaled,
        digits: cfg.digits,
        guard: cfg.guard,
        n_terms,
        hex_len: used_hex_len,
    };

    // --- Phase 3: binary -> decimal base conversion ---
    let digits = result.decimal_digits();
    let t4 = Instant::now();

    // --- Phase 4: streaming file write ---
    let mut writer = DigitWriter::create(&args.output).map_err(|e| {
        format!("cannot open output file {}: {e}", args.output.display())
    })?;
    writer
        .write_prefix()
        .and_then(|_| writer.write_digits(digits.as_bytes()))
        .map_err(|e| format!("error writing output file {}: {e}", args.output.display()))?;
    let written = writer
        .finish()
        .map_err(|e| format!("error flushing output file {}: {e}", args.output.display()))?;
    let t5 = Instant::now();
    if written != 2 + cfg.digits {
        return Err(format!(
            "wrote {} bytes, expected {}",
            written,
            2 + cfg.digits
        ));
    }

    // --- Phase 5: verification ---
    let mut report = verify::VerifyReport::default();
    if verify_on {
        let checkpoints = match &args.checkpoints {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read checkpoints {}: {e}", path.display()))?;
                verify::parse_checkpoints(&text)?
            }
            None => Vec::new(),
        };
        report = verify::run_verification(&result, &digits, &checkpoints, threads);
    }
    let t6 = Instant::now();

    // --- Reporting ---
    if args.bench {
        let timings = Timings {
            series: t1 - t0,
            final_pow: t2 - t1,
            hex_pow: t3 - t2,
            conversion: t4 - t3,
            write: t5 - t4,
            verify: t6 - t5,
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
        let text = report.render();
        eprint!("{text}");
        if report.all_pass() {
            eprintln!("verification: ALL CHECKS PASSED");
            Ok(0)
        } else {
            eprintln!("verification: FAILURES DETECTED");
            Ok(1)
        }
    } else {
        Ok(0)
    }
}

fn print_bench(cfg: &PiConfig, t: &Timings, verify_on: bool) {
    let total: std::time::Duration =
        [t.series, t.final_pow, t.hex_pow, t.conversion, t.write]
            .iter()
            .copied()
            .sum::<std::time::Duration>()
            + t.verify;
    eprintln!("\n--- timing breakdown ---");
    eprintln!("digits:                    {}", cfg.digits);
    eprintln!("threads:                   {}", cfg.effective_threads());
    eprintln!("Chudnovsky series (BS):    {:>10.3?}", t.series);
    eprintln!("final scaling (decimal):   {:>10.3?}", t.final_pow);
    eprintln!("final scaling (hex/verify):{:>10.3?}", t.hex_pow);
    eprintln!("base conversion (10):      {:>10.3?}", t.conversion);
    eprintln!("file write:                {:>10.3?}", t.write);
    if verify_on {
        eprintln!("verification:              {:>10.3?}", t.verify);
    }
    eprintln!("total (compute+write):     {:>10.3?}", total);
    eprintln!("peak memory (VmHWM):       {}", {
        let kb = peak_mem_kb();
        fmt_mb(kb)
    });
}
