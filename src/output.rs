//! Streaming output of π digits to a file.
//!
//! The digit buffer is written incrementally in chunks so that we do not need
//! to build an extra full-size copy of the output string in memory. We write
//! the `"3."` prefix, then the digit bytes in successive chunks to a buffered
//! writer, and flush at the end.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

const CHUNK: usize = 1 << 20; // write in 1 MiB chunks

/// A streaming writer for the π digit output file.
pub struct DigitWriter {
    writer: BufWriter<File>,
    written: usize,
}

impl DigitWriter {
    /// Open `path` for writing digits. Creates the file (truncating it).
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::with_capacity(CHUNK, file),
            written: 0,
        })
    }

    /// Write the `"3."` prefix.
    pub fn write_prefix(&mut self) -> io::Result<()> {
        self.writer.write_all(b"3.")?;
        self.written += 2;
        Ok(())
    }

    /// Stream a slice of digit bytes in chunks.
    pub fn write_digits(&mut self, bytes: &[u8]) -> io::Result<()> {
        for chunk in bytes.chunks(CHUNK) {
            self.writer.write_all(chunk)?;
            self.written += chunk.len();
        }
        Ok(())
    }

    /// Flush and return the total number of bytes written (including prefix).
    pub fn finish(mut self) -> io::Result<usize> {
        self.writer.flush()?;
        Ok(self.written)
    }
}
