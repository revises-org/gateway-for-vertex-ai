// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Preserving Gemini thought signatures across turns.
//!
//! # The problem
//!
//! Gemini 3 thinks before it emits a tool call. Google signs that reasoning and
//! attaches the signature to the tool call in a **non-standard** field:
//!
//! ```json
//! "tool_calls": [{
//!   "id": "call_abc",
//!   "extra_content": { "google": { "thought_signature": "AY89..." } }
//! }]
//! ```
//!
//! Gemini 3 then *requires* that signature to come back with the conversation
//! history on the next turn. Any standard OpenAI client drops `extra_content`
//! because it does not know the field, so the second turn of every tool-calling
//! conversation fails with:
//!
//! ```text
//! 400 INVALID_ARGUMENT: Function call is missing a thought_signature
//! ```
//!
//! This is not a niche bug. VS Code Copilot, Codex CLI, Continue, LiteLLM and
//! the OpenAI Python SDK have all hit it. The client cannot fix it without
//! learning a Google-specific field, which defeats the point of using an
//! OpenAI-compatible endpoint.
//!
//! # The fix
//!
//! A gateway sees both directions, so it can carry the signature itself. The
//! trick — borrowed from LiteLLM — is to smuggle it inside the one field the
//! client is guaranteed to preserve verbatim: the tool call **id**. Clients
//! must echo ids back to match tool results to calls, so an id survives where
//! `extra_content` does not.
//!
//! ```text
//! Vertex  -> gateway:  id = "call_abc", signature = "AY89..."
//! gateway -> client:   id = "call_abc__thought__AY89..."
//! client  -> gateway:  id = "call_abc__thought__AY89..."   (echoed unchanged)
//! gateway -> Vertex:   id = "call_abc", signature restored
//! ```
//!
//! The gateway stays **stateless**: nothing is cached between requests, because
//! the signature travels with the conversation rather than in server memory.

use serde_json::{json, Map, Value};

/// Separator between the real id and the smuggled signature.
///
/// Deliberately verbose and unlikely to occur in a provider-generated id. Note
/// that signatures are base64 and may contain `+`, `/` and `=`, so splitting
/// must happen on this marker and nothing else.
const MARKER: &str = "__thought__";

/// Escape character used to keep packed ids inside a conservative charset.
const ESCAPE: char = '-';

/// Encode a signature so the packed id matches `^[A-Za-z0-9_-]+$`.
///
/// # Why this is needed
///
/// Signatures are base64 and routinely contain `+`, `/` and `=`. Appending one
/// verbatim produces something no OpenAI client has ever seen: real tool call
/// ids look like `call_` plus 24 alphanumerics. A client, proxy or database
/// that validates ids — a regex, a charset check, a URL path segment — will
/// reject or mangle it, and the failure will look like a gateway bug.
///
/// Every byte outside `[A-Za-z0-9_]` becomes `-` plus two hex digits. This
/// assumes nothing about the input being valid base64, so it keeps working if
/// Google changes the encoding. A literal `-` is itself escaped, which is what
/// makes the transform unambiguous.
fn escape_signature(sig: &str) -> String {
    let mut out = String::with_capacity(sig.len() + 8);
    for b in sig.bytes() {
        let c = b as char;
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push(ESCAPE);
            out.push_str(&format!("{b:02x}"));
        }
    }
    out
}

/// Reverse [`escape_signature`]. Returns `None` on malformed input rather than
/// guessing — a corrupted signature would be rejected by Vertex anyway.
fn unescape_signature(escaped: &str) -> Option<String> {
    let bytes = escaped.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == ESCAPE as u8 {
            let hex = escaped.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

pub fn pack_id(id: &str, signature: Option<&str>) -> String {
    match signature {
        Some(sig) => format!("{id}{MARKER}{}", escape_signature(sig)),
        None => id.to_string(),
    }
}

pub fn unpack_id(packed: &str) -> (String, Option<String>) {
    match packed.split_once(MARKER) {
        Some((id, escaped)) => (id.to_string(), unescape_signature(escaped)),
        None => (packed.to_string(), None),
    }
}

// ----------------------------------------------------------- delta merging

/// Merge a streaming delta into an accumulator, keeping fields we do not know.
///
/// This is the function that prevents the bug the whole module exists for.
///
/// The obvious way to accumulate tool call deltas is to copy the fields you
/// care about — id, name, arguments. That silently discards `extra_content`,
/// and with it the signature. It is exactly how the same bug was introduced in
/// other projects: the merge code predates the field, and nothing fails loudly
/// when a new one appears.
///
/// Merging recursively and copying *everything* means unknown fields ride along
/// for free, which is the same principle the rest of this gateway follows: what
/// we do not interpret, we do not damage.
///
/// String values are **appended**, not replaced, because that is how OpenAI
/// streams tool call arguments — one fragment of JSON text per chunk.
pub fn merge_delta(acc: &mut Value, patch: &Value) {
    match (acc, patch) {
        (Value::Object(a), Value::Object(p)) => {
            for (key, value) in p {
                match a.get_mut(key) {
                    Some(existing) => merge_delta(existing, value),
                    None => {
                        a.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (Value::String(a), Value::String(p)) => a.push_str(p),
        (Value::Array(a), Value::Array(p)) => {
            // Arrays inside a tool call delta are not fragmented by index the
            // way the top-level tool_calls array is, so replacing is correct.
            *a = p.clone();
        }
        (a, p) => *a = p.clone(),
    }
}

// ------------------------------------------------- response: pack signatures

/// Read the signature out of a tool call object, if Vertex attached one.
fn signature_of(tool_call: &Value) -> Option<&str> {
    tool_call
        .pointer("/extra_content/google/thought_signature")
        .and_then(Value::as_str)
}

/// Rewrite one tool call in place: fold its signature into the id and drop the
/// now-redundant `extra_content`.
///
/// Dropping it matters. Leaving both would mean the signature travels twice,
/// and a client that *does* understand `extra_content` would send back a
/// mismatched pair.
pub fn pack_tool_call(tool_call: &mut Value) {
    let Some(sig) = signature_of(tool_call).map(str::to_string) else {
        return;
    };
    let Some(obj) = tool_call.as_object_mut() else {
        return;
    };

    if let Some(id) = obj.get("id").and_then(Value::as_str) {
        let packed = pack_id(id, Some(&sig));
        obj.insert("id".into(), json!(packed));
        obj.remove("extra_content");
    }
    // A tool call with a signature but no id yet only happens mid-stream; the
    // streaming path accumulates deltas before calling this, so by the time we
    // get here the id is present.
}

/// Pack every tool call in a non-streaming chat completion response.
pub fn pack_response(body: &mut Value) {
    let Some(choices) = body.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        if let Some(calls) = choice
            .pointer_mut("/message/tool_calls")
            .and_then(Value::as_array_mut)
        {
            for call in calls {
                pack_tool_call(call);
            }
        }
    }
}

// ------------------------------------------ request: restore signatures

/// Undo the packing on the way back to Vertex.
///
/// Two places carry a packed id, and missing either one breaks the request:
/// - assistant messages, in `tool_calls[].id`
/// - tool result messages, in `tool_call_id`
///
/// Vertex matches results to calls by id, so both sides have to be unpacked
/// consistently.
pub fn restore_request(body: &mut Value) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };

    for message in messages {
        // Tool result: `tool_call_id` echoes the packed id.
        if let Some(packed) = message.get("tool_call_id").and_then(Value::as_str) {
            let (id, _) = unpack_id(packed);
            if let Some(obj) = message.as_object_mut() {
                obj.insert("tool_call_id".into(), json!(id));
            }
        }

        // Assistant message: unpack each call and put the signature back where
        // Vertex expects to find it.
        if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for call in calls {
                restore_tool_call(call);
            }
        }
    }
}

fn restore_tool_call(call: &mut Value) {
    let Some(packed) = call.get("id").and_then(Value::as_str) else {
        return;
    };
    let (id, signature) = unpack_id(packed);

    let Some(obj) = call.as_object_mut() else {
        return;
    };
    obj.insert("id".into(), json!(id));

    let Some(sig) = signature else {
        return;
    };

    // Rebuild extra_content.google.thought_signature without clobbering any
    // other google-specific content the client may have sent.
    let google = obj
        .entry("extra_content")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .map(|ec| {
            ec.entry("google")
                .or_insert_with(|| json!({}))
                .as_object_mut()
        });

    if let Some(Some(g)) = google {
        g.insert("thought_signature".into(), json!(sig));
    }
}

// ------------------------------------------------ streaming accumulation

/// Collects tool call deltas across chunks so signatures can be folded into ids.
///
/// # Why tool calls are buffered but text is not
///
/// Text deltas pass straight through, so typing still appears live. Tool calls
/// cannot: the signature may arrive in a different chunk than the id, and once
/// an id has been sent to the client it cannot be rewritten. Holding tool call
/// deltas until the stream completes makes correctness independent of chunk
/// ordering.
///
/// The cost is that tool call arguments no longer appear character by character.
/// That is invisible in practice — a client cannot execute a tool call until
/// its arguments are complete anyway.
#[derive(Default)]
pub struct ToolCallAccumulator {
    /// Keyed by the `index` field, which is how OpenAI correlates fragments of
    /// the same call. Parallel tool calls arrive interleaved under different
    /// indices.
    calls: Vec<(u64, Value)>,
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Absorb the `tool_calls` array from one streamed delta.
    pub fn absorb(&mut self, deltas: &[Value]) {
        for delta in deltas {
            let index = delta.get("index").and_then(Value::as_u64).unwrap_or(0);
            match self.calls.iter_mut().find(|(i, _)| *i == index) {
                Some((_, acc)) => merge_delta(acc, delta),
                None => self.calls.push((index, delta.clone())),
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Produce the consolidated `tool_calls` array, ids packed, in index order.
    pub fn finish(mut self) -> Vec<Value> {
        self.calls.sort_by_key(|(i, _)| *i);
        self.calls
            .into_iter()
            .map(|(_, mut call)| {
                pack_tool_call(&mut call);
                call
            })
            .collect()
    }
}

/// Pull `usage` out of a chunk, if this is the one carrying it.
pub fn usage_of(chunk: &Value) -> Option<&Map<String, Value>> {
    chunk.get("usage").and_then(Value::as_object)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- id packing ----------

    #[test]
    fn id_round_trips() {
        let packed = pack_id("call_abc", Some("AY89xyz"));
        assert_eq!(packed, "call_abc__thought__AY89xyz");
        assert_eq!(
            unpack_id(&packed),
            ("call_abc".into(), Some("AY89xyz".into()))
        );
    }

    #[test]
    fn id_without_signature_passes_through() {
        assert_eq!(unpack_id("call_abc"), ("call_abc".into(), None));
        assert_eq!(pack_id("call_abc", None), "call_abc");
    }

    #[test]
    fn base64_signature_is_not_truncated() {
        // Real signatures are long base64 and contain + / = characters.
        let sig = "AY89a18oR7U68B4xDq8Rjld7IAxkMavumzPIzKdmoo+YRwfL/78SWawC10UIoq27NQZyahmy4fGf7XYJSIIpDbe4twDkhK1o0BZEY3S03=";
        let (id, got) = unpack_id(&pack_id("call_1", Some(sig)));
        assert_eq!(id, "call_1");
        assert_eq!(got.as_deref(), Some(sig));
    }

    // ---------- delta merging ----------

    fn tool_call_deltas() -> Vec<Value> {
        vec![
            json!({"index":0,"id":"call_abc","type":"function",
                   "function":{"name":"read_file","arguments":""},
                   "extra_content":{"google":{"thought_signature":"SIG1"}}}),
            json!({"index":0,"function":{"arguments":"{\"path\":"}}),
            json!({"index":0,"function":{"arguments":"\"a.rs\"}"}}),
        ]
    }

    #[test]
    fn merging_keeps_unknown_fields() {
        let mut acc = json!({});
        for d in tool_call_deltas() {
            merge_delta(&mut acc, &d);
        }
        assert_eq!(acc["function"]["arguments"], r#"{"path":"a.rs"}"#);
        assert_eq!(
            acc["extra_content"]["google"]["thought_signature"], "SIG1",
            "this assertion is the whole point of the module"
        );
    }

    #[test]
    fn merging_appends_strings_rather_than_replacing() {
        let mut acc = json!({"function":{"arguments":"ab"}});
        merge_delta(&mut acc, &json!({"function":{"arguments":"cd"}}));
        assert_eq!(acc["function"]["arguments"], "abcd");
    }

    // ---------- streaming accumulation ----------

    #[test]
    fn accumulator_packs_signature_into_id() {
        let mut acc = ToolCallAccumulator::new();
        for d in tool_call_deltas() {
            acc.absorb(std::slice::from_ref(&d));
        }
        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_abc__thought__SIG1");
        assert!(
            calls[0].get("extra_content").is_none(),
            "extra_content must be removed so the signature is not sent twice"
        );
    }

    #[test]
    fn parallel_tool_calls_stay_separate() {
        // Two calls interleaved across chunks, each with its own signature.
        let deltas = vec![
            json!({"index":0,"id":"call_a","function":{"name":"read","arguments":""},
                   "extra_content":{"google":{"thought_signature":"SIG_A"}}}),
            json!({"index":1,"id":"call_b","function":{"name":"grep","arguments":""},
                   "extra_content":{"google":{"thought_signature":"SIG_B"}}}),
            json!({"index":1,"function":{"arguments":"{\"q\":1}"}}),
            json!({"index":0,"function":{"arguments":"{\"p\":2}"}}),
        ];
        let mut acc = ToolCallAccumulator::new();
        for d in &deltas {
            acc.absorb(std::slice::from_ref(d));
        }
        let calls = acc.finish();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_a__thought__SIG_A");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"p":2}"#);
        assert_eq!(calls[1]["id"], "call_b__thought__SIG_B");
        assert_eq!(calls[1]["function"]["arguments"], r#"{"q":1}"#);
    }

    #[test]
    fn tool_call_without_signature_is_untouched() {
        let mut acc = ToolCallAccumulator::new();
        acc.absorb(&[json!({"index":0,"id":"call_x","function":{"name":"f","arguments":"{}"}})]);
        let calls = acc.finish();
        assert_eq!(calls[0]["id"], "call_x");
    }

    // ---------- full round trip ----------

    #[test]
    fn signature_survives_a_full_turn() {
        // 1. Vertex responds with a signed tool call.
        let mut response = json!({
            "choices":[{"message":{"role":"assistant","tool_calls":[{
                "id":"call_abc","type":"function",
                "function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"},
                "extra_content":{"google":{"thought_signature":"SIG_ROUNDTRIP"}}
            }]}}]
        });
        pack_response(&mut response);

        let packed_id = response
            .pointer("/choices/0/message/tool_calls/0/id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(packed_id, "call_abc__thought__SIG_ROUNDTRIP");

        // 2. The client echoes the id back verbatim and drops everything it
        //    does not understand — which is what breaks other clients.
        let mut next_request = json!({
            "model":"gemini-3.7-flash",
            "messages":[
                {"role":"user","content":"read a.rs"},
                {"role":"assistant","tool_calls":[{
                    "id": packed_id,
                    "type":"function",
                    "function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}
                }]},
                {"role":"tool","tool_call_id": packed_id,"content":"fn main() {}"}
            ]
        });

        // 3. The gateway restores it before forwarding.
        restore_request(&mut next_request);

        assert_eq!(
            next_request.pointer("/messages/1/tool_calls/0/id").unwrap(),
            "call_abc",
            "id must be the one Vertex issued"
        );
        assert_eq!(
            next_request
                .pointer("/messages/1/tool_calls/0/extra_content/google/thought_signature")
                .unwrap(),
            "SIG_ROUNDTRIP",
            "signature must be back where Vertex expects it"
        );
        assert_eq!(
            next_request.pointer("/messages/2/tool_call_id").unwrap(),
            "call_abc",
            "tool result id must match the call id"
        );
    }

    #[test]
    fn requests_without_tool_calls_are_left_alone() {
        let original = json!({"model":"gemini-pro","messages":[{"role":"user","content":"hi"}]});
        let mut body = original.clone();
        restore_request(&mut body);
        assert_eq!(body, original);
    }

    #[test]
    fn restoring_preserves_other_google_extra_content() {
        let mut body = json!({"messages":[{"role":"assistant","tool_calls":[{
            "id":"call_a__thought__SIG",
            "extra_content":{"google":{"something_else":true}}
        }]}]});
        restore_request(&mut body);
        let ec = body
            .pointer("/messages/0/tool_calls/0/extra_content/google")
            .unwrap();
        assert_eq!(ec["thought_signature"], "SIG");
        assert_eq!(
            ec["something_else"], true,
            "must not clobber sibling fields"
        );
    }

    #[test]
    fn packed_id_stays_in_a_conservative_charset() {
        let sig = "AY89+abc/def=ghi-jkl_mno";
        let packed = pack_id("call_1", Some(sig));
        assert!(
            packed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "packed id must satisfy ^[A-Za-z0-9_-]+$, got {packed}"
        );
    }

    #[test]
    fn signature_round_trips_through_escaping() {
        for sig in [
            "AY89a18oR7U68B4x",
            "with+plus/slash=pad",
            "with-hyphen_and_underscore", // must not collide with the escape char
            "==",
        ] {
            let (id, got) = unpack_id(&pack_id("call_1", Some(sig)));
            assert_eq!(id, "call_1");
            assert_eq!(got.as_deref(), Some(sig), "round trip failed for {sig}");
        }
    }
}
