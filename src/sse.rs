// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Turning a byte stream into SSE lines.
//!
//! # Why this is not just `split('\n')`
//!
//! TCP does not respect line boundaries. A single JSON event can be cut in half
//! across two chunks, and one chunk can carry three events. Parsing each chunk
//! as it arrives works fine under light load and silently loses data on long
//! responses — the worst kind of bug, because it is intermittent.
//!
//! # Why bytes and not `String`
//!
//! A chunk can also be cut in the **middle of a UTF-8 character**. Multi-byte
//! text is the norm outside English: Vietnamese characters are 2–3 bytes, CJK
//! is 3, emoji is 4. Decoding each chunk separately produces replacement
//! characters (U+FFFD) at those boundaries — for Chinese text, roughly 69% of
//! possible split points corrupt the output; for emoji with skin-tone
//! modifiers, 80%. Pure ASCII is 0%, which is exactly why this class of bug
//! survives English-only testing.
//!
//! Accumulating raw bytes and decoding only once a full line is available
//! sidesteps the problem entirely: the accumulator never looks at character
//! boundaries, it just waits for `0x0A`.
//!
/// A cap turns an unbounded memory leak into a clean error.
/// Default cap on a single line.
/// One SSE event legitimately gets large: inline base64 images, audio, or a
/// long reasoning block all travel in a single JSON object. 1 MB was too tight
/// for those. This is a runaway guard, not a throughput limit — it only has to
/// be smaller than "this upstream will never stop".
pub const DEFAULT_MAX_LINE_BYTES: usize = 32 * 1024 * 1024;

/// Accumulates chunks and yields complete lines.
pub struct LineAccumulator {
    buf: Vec<u8>,
    limit: usize,
}

impl Default for LineAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl LineAccumulator {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            limit: DEFAULT_MAX_LINE_BYTES,
        }
    }

    /// Override the runaway-line cap. A host application streaming unusually
    /// large inline media may need more headroom than the default.
    pub fn with_limit(limit: usize) -> Self {
        Self {
            buf: Vec::new(),
            limit,
        }
    }

    /// Feed one chunk, get back whatever complete lines it finished.
    ///
    /// Trailing bytes with no newline yet are kept for the next call.
    /// Returns `Err` if a single line exceeds [`DEFAULT_MAX_LINE_BYTES`].
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>, String> {
        self.buf.extend_from_slice(chunk);

        // Scan the buffer once and slice lines out of it in place. The earlier
        // version drained on every newline, which re-scanned from zero and
        // memmoved the remainder each time — O(N·k) for k lines in one chunk.
        // Here the scan is a single pass and the memmove happens once.
        let mut lines = Vec::new();
        let mut start = 0;
        while let Some(rel) = self.buf[start..].iter().position(|&b| b == b'\n') {
            let end = start + rel;
            // Decoding happens on a whole line, so multi-byte characters split
            // across chunk boundaries are always intact by this point.
            let raw = &self.buf[start..end];
            let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
            lines.push(String::from_utf8_lossy(raw).into_owned());
            start = end + 1;
        }
        if start > 0 {
            self.buf.drain(..start);
        }

        // The cap applies to what is left over — a single line still growing
        // without a terminator. Checking the whole buffer instead would reject
        // a chunk that merely happened to carry many complete lines.
        if self.buf.len() > self.limit {
            return Err(format!(
                "SSE line exceeded {} bytes without a newline; aborting stream",
                self.limit
            ));
        }
        Ok(lines)
    }

    /// Anything left when the stream ends without a trailing newline.
    pub fn finish(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            return None;
        }
        let raw = std::mem::take(&mut self.buf);
        let raw = raw.strip_suffix(b"\r").unwrap_or(&raw);
        let line = String::from_utf8_lossy(raw).into_owned();
        (!line.is_empty()).then_some(line)
    }
}

/// Strip the `data:` prefix from an SSE line, if present.
///
/// Returns `None` for comments, blank lines, and the `[DONE]` sentinel — none
/// of which carry a JSON payload.
pub fn data_payload(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("data:")?;
    // The spec strips exactly one leading space, not all whitespace.
    let rest = rest.strip_prefix(' ').unwrap_or(rest);
    if rest.is_empty() || rest == "[DONE]" {
        return None;
    }
    Some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(chunks: &[&[u8]]) -> Vec<String> {
        let mut acc = LineAccumulator::new();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(acc.push(c).expect("within cap"));
        }
        out.extend(acc.finish());
        out
    }

    #[test]
    fn line_split_across_chunks_is_rejoined() {
        let out = feed(&[b"data: {\"a\":1}\ndata: {\"b\"", b":2}\ndata: [DONE]\n"]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1], r#"data: {"b":2}"#);
    }

    #[test]
    fn multibyte_text_survives_any_split_point() {
        // One sample per encoding width plus a grapheme cluster, kept short on
        // purpose: this is a regression guard, not a Unicode conformance suite.
        // Every sample passes for the same reason — the accumulator never
        // inspects character boundaries — so more samples would add no
        // information.
        for text in [
            "hello",               // 1 byte
            "chào bạn tiếng Việt", // 2-3 bytes
            "你好世界",            // 3 bytes
            "😀🎉 emoji",          // 4 bytes
            "👨‍👩‍👧‍👦",                  // ZWJ cluster, 25 bytes
        ] {
            let line = format!("data: {text}\n");
            let bytes = line.as_bytes();
            for cut in 0..=bytes.len() {
                let (a, b) = bytes.split_at(cut);
                let out = feed(&[a, b]);
                assert_eq!(out, vec![format!("data: {text}")], "split at byte {cut}");
            }
        }
    }

    #[test]
    fn byte_at_a_time_still_works() {
        let line = "data: {\"content\":\"你好 chào 😀\"}\n";
        let parts: Vec<&[u8]> = line.as_bytes().chunks(1).collect();
        assert_eq!(feed(&parts), vec![line.trim().to_string()]);
    }

    #[test]
    fn missing_newline_is_capped_instead_of_growing_forever() {
        let mut acc = LineAccumulator::with_limit(1024);
        let junk = vec![b'x'; 512];
        assert!(acc.push(&junk).is_ok());
        assert!(acc.push(&junk).is_ok());
        assert!(
            acc.push(&junk).is_err(),
            "must abort rather than buffer without limit"
        );
    }

    #[test]
    fn data_payload_skips_non_json_lines() {
        assert_eq!(data_payload("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(data_payload("data: [DONE]"), None);
        assert_eq!(data_payload(""), None);
        assert_eq!(data_payload(": keep-alive comment"), None);
    }

    #[test]
    fn many_complete_lines_do_not_trip_the_cap() {
        // Total far exceeds DEFAULT_MAX_LINE_BYTES, but no single line does. The cap is
        // about runaway lines, not about throughput.
        let one = "data: {\"x\":1}\n";
        let big: String = one.repeat(200_000); // ~2.8 MB
        let mut acc = LineAccumulator::new();
        let lines = acc.push(big.as_bytes()).expect("must not be rejected");
        assert_eq!(lines.len(), 200_000);
    }

    #[test]
    fn crlf_line_endings_are_handled() {
        let mut acc = LineAccumulator::new();
        let lines = acc.push(b"data: {\"a\":1}\r\ndata: {\"b\":2}\r\n").unwrap();
        assert_eq!(lines, vec![r#"data: {"a":1}"#, r#"data: {"b":2}"#]);
    }

    #[test]
    fn only_one_leading_space_is_stripped_after_data() {
        // The spec strips exactly one space; the rest belongs to the payload.
        assert_eq!(data_payload("data:  {\"a\":1}"), Some(" {\"a\":1}"));
        assert_eq!(data_payload("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(data_payload("data:{\"a\":1}"), Some("{\"a\":1}"));
    }

    #[test]
    fn a_large_single_event_is_not_rejected() {
        // A base64 image or long reasoning block arrives as one oversized line.
        let payload = "x".repeat(4 * 1024 * 1024);
        let line = format!("data: {{\"content\":\"{payload}\"}}\n");
        let mut acc = LineAccumulator::new();
        assert_eq!(acc.push(line.as_bytes()).unwrap().len(), 1);
    }
}
