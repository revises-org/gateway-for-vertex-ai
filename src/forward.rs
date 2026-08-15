// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! The forwarding core.
//!
//! # Adding another protocol
//!
//! Every Vertex protocol shares the same send procedure: attach a token, make
//! the call, decide between an error response and an open stream, then relay
//! the body. That is [`forward`].
//!
//! Only two things differ between protocols — the URL and how the body is
//! adjusted — so a new protocol module needs three steps:
//!
//! 1. Build the target URL (add a `Config::*_base()` helper if the path shape
//!    differs).
//! 2. Adjust the body to suit that protocol.
//! 3. Call [`forward`].
//!
//! Claude on Vertex is the concrete example: the request body is almost exactly
//! the Anthropic Messages API, differing in only two ways — `model` moves out
//! of the body and into the URL, and `"anthropic_version": "vertex-2023-10-16"`
//! has to be inserted. The endpoint is
//! `publishers/anthropic/models/{model}:rawPredict`, switching to
//! `:streamRawPredict` when streaming.
//!
//! # Scope
//!
//! The HTTP call, the error-versus-stream decision, backpressure and usage
//! observation are protocol-independent. The line handling in [`process_line`]
//! is **not**: it assumes the OpenAI chunk shape
//! (`choices[0].delta.tool_calls`), and it only tracks the first choice.
//!
//! A second protocol with a different chunk shape — Anthropic's
//! `content_block_delta`, say — will need its own line handler. Whether that
//! becomes a trait or simply a second function depends on how much the two turn
//! out to share, which is not knowable from one implementation.
//!
//! # Why the stream is parsed at all
//!
//! Earlier versions piped bytes through untouched, which is the safest thing a
//! proxy can do: data you never read is data you cannot corrupt. That stopped
//! being possible with Gemini 3, which requires thought signatures to be echoed
//! back on the next turn (see [`crate::thought`]). Carrying them means reading
//! the response.
//!
//! Since the stream has to be parsed anyway, the same pass extracts `usage` for
//! [`crate::observe`]. One pipeline applied uniformly — no branching on whether
//! the request happened to include tools, which would mean two code paths to
//! test and inconsistent logging between them.

use std::{sync::Arc, time::Instant};

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    observe::{parse_usage, CompletionEvent, Usage},
    sse::{data_payload, LineAccumulator},
    state::AppState,
    thought::{pack_response, usage_of, ToolCallAccumulator},
};

/// How many processed chunks may sit in flight before the reader pauses.
///
/// A bounded channel is what propagates backpressure from a slow client back to
/// the upstream read, so a client that stops consuming cannot make the gateway
/// buffer a whole response in memory.
const CHANNEL_DEPTH: usize = 16;

/// Send `body` to `url` on Vertex and relay the response back to the client.
pub async fn forward(st: Arc<AppState>, url: String, body: Value, streaming: bool) -> Response {
    let started = Instant::now();
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let token = match st.token().await {
        Ok(t) => t,
        Err(e) => return upstream_error(e),
    };

    let resp = match st
        .http()
        .post(&url)
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return upstream_error(format!("request to Vertex failed: {e}")),
    };

    // reqwest and axum depend on different major versions of the `http` crate,
    // so their StatusCode types are not interchangeable — round-trip via u16.
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    // On failure, read the whole body and return JSON — even when the client
    // asked for a stream.
    //
    // This decision has to happen HERE: once `200 OK` has been sent and bytes
    // start flowing, the headers are gone and the status code can no longer be
    // changed. Stuffing an error into an SSE frame makes clients report an
    // "empty response" instead of showing the user what Vertex complained about.
    if !streaming || !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let mut payload: Value =
            serde_json::from_str(&text).unwrap_or_else(|_| json!({"error": {"message": text}}));

        let usage = usage_of(&payload).map(parse_usage);
        if status.is_success() {
            pack_response(&mut payload);
        }

        st.notify(CompletionEvent {
            model,
            status: status.as_u16(),
            streamed: false,
            duration: started.elapsed(),
            usage,
        });

        return (status, Json(payload)).into_response();
    }

    stream_response(st, resp, status, model, started)
}

/// Relay an SSE stream, rewriting tool calls and observing usage as it passes.
fn stream_response(
    st: Arc<AppState>,
    resp: reqwest::Response,
    status: StatusCode,
    model: String,
    started: Instant,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(CHANNEL_DEPTH);

    // A detached task owns the rewriting state. Writing it as a plain async
    // block rather than a stream-generator macro keeps the control flow
    // ordinary Rust, which both humans and tooling read more reliably.
    tokio::spawn(async move {
        let mut upstream = resp.bytes_stream();
        let mut lines = LineAccumulator::new();
        let mut tools = ToolCallAccumulator::new();
        let mut usage: Option<Usage> = None;
        // Held back so the consolidated tool_calls delta can be emitted just
        // before it; clients treat this chunk as "the turn is over".
        let mut finish_chunk: Option<Value> = None;

        // Set when the stream ends abnormally, so the terminator can be
        // withheld — see the end of this function.
        let mut failed = false;

        'outer: while let Some(part) = upstream.next().await {
            let part = match part {
                Ok(b) => b,
                Err(e) => {
                    let msg = format!("upstream stream failed: {e}");
                    let _ = tx.send(Ok(sse_error(&msg))).await;
                    failed = true;
                    break;
                }
            };

            let complete = match lines.push(&part) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Ok(sse_error(&e))).await;
                    failed = true;
                    break;
                }
            };

            for line in complete {
                if let Some(out) = process_line(&line, &mut tools, &mut usage, &mut finish_chunk) {
                    if tx.send(Ok(out)).await.is_err() {
                        // Client hung up. Stop reading upstream; usage stays
                        // unknown, which is reported honestly rather than as 0.
                        break 'outer;
                    }
                }
            }
        }

        // A leftover fragment means the upstream stopped mid-event. It is not a
        // valid SSE message, so anything accumulated may be incomplete too.
        let mut truncated = false;
        if let Some(partial) = lines.finish() {
            tracing::warn!(
                bytes = partial.len(),
                "upstream ended mid-event; discarding incomplete SSE data"
            );
            truncated = true;
        }

        // Three ways a stream can end, and they need different handling.
        if truncated {
            // Cut mid-event: accumulated tool calls may have incomplete
            // arguments, so emitting them would make the client invoke a
            // function with malformed JSON. Report instead.
            let _ = tx
                .send(Ok(sse_error(
                    "upstream ended before the response completed",
                )))
                .await;
        } else {
            // Capture before the accumulator is consumed just below.
            let had_tool_calls = !tools.is_empty();

            // Tool calls are emitted whether or not a finish chunk arrived.
            // Holding them hostage to finish_reason would silently drop the
            // entire turn when an upstream closes cleanly without one.
            if had_tool_calls {
                let reference = finish_chunk.clone().unwrap_or_else(|| json!({}));
                let calls = std::mem::take(&mut tools).finish();
                let _ = tx.send(Ok(tool_calls_chunk(&reference, calls))).await;
            }

            // Synthesise a terminal chunk if the upstream never sent one, so the
            // client is not left waiting for an end-of-turn signal it will never
            // get.
            //
            // The reason is inferred from what actually happened rather than
            // defaulted: telling a client "stop" right after emitting tool calls
            // contradicts what we just sent it, and a client that believes the
            // turn simply ended will never invoke the function. "length" is not
            // inferable — only the upstream knows it hit a token ceiling, and
            // this branch runs precisely when the upstream said nothing.
            let mut chunk = finish_chunk.take().unwrap_or_else(|| {
                let reason = if had_tool_calls { "tool_calls" } else { "stop" };
                json!({
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                })
            });

            // Re-attach usage so clients reading it from the final chunk are
            // unaffected by the reordering above.
            if let Some(u) = &usage {
                if chunk.get("usage").is_none() {
                    chunk["usage"] = json!({
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
                    });
                }
            }
            let _ = tx.send(Ok(sse_line(&chunk))).await;
        }

        // `[DONE]` means "this turn completed". Emitting it after an error lets
        // SDKs that stop reading at the terminator swallow the error frame and
        // report success. When something went wrong, end the body abnormally
        // instead: the client sees a broken stream, which is what happened.
        if failed || truncated {
            let _ = tx
                .send(Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "stream ended before completion",
                )))
                .await;
        } else {
            let _ = tx
                .send(Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")))
                .await;
        }

        st.notify(CompletionEvent {
            model: model.clone(),
            status: status.as_u16(),
            streamed: true,
            duration: started.elapsed(),
            usage,
        });
    });

    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

/// Decide what to do with one SSE line.
///
/// Returns the bytes to forward, or `None` when the line was consumed — either
/// absorbed as a tool call fragment, or held back as the finish chunk.
pub(crate) fn process_line(
    line: &str,
    tools: &mut ToolCallAccumulator,
    usage: &mut Option<Usage>,
    finish_chunk: &mut Option<Value>,
) -> Option<bytes::Bytes> {
    // Comments, blanks and `[DONE]` carry no payload. `[DONE]` is dropped here
    // and re-emitted at the very end, after the held-back chunks.
    let payload = data_payload(line)?;

    // Anything unparseable is forwarded untouched rather than dropped: an
    // unrecognised line is more useful to the client than silence.
    let Ok(mut chunk) = serde_json::from_str::<Value>(payload) else {
        return Some(bytes::Bytes::from(format!("{line}\n\n")));
    };

    if let Some(u) = usage_of(&chunk) {
        *usage = Some(parse_usage(u));
    }

    // Absorb tool call fragments; see ToolCallAccumulator for why these cannot
    // stream through the way text does.
    let has_tool_deltas = chunk
        .pointer("/choices/0/delta/tool_calls")
        .and_then(Value::as_array)
        .map(|deltas| {
            tools.absorb(deltas);
            true
        })
        .unwrap_or(false);

    if has_tool_deltas {
        // Remove only the tool_calls field; the same chunk may also carry text.
        if let Some(delta) = chunk
            .pointer_mut("/choices/0/delta")
            .and_then(Value::as_object_mut)
        {
            delta.remove("tool_calls");
            let empty = delta.is_empty();
            let terminal = chunk
                .pointer("/choices/0/finish_reason")
                .is_some_and(|v| !v.is_null());
            if empty && !terminal {
                return None;
            }
        }
    }

    // Hold the terminal chunk so consolidated tool calls can precede it.
    if chunk
        .pointer("/choices/0/finish_reason")
        .is_some_and(|v| !v.is_null())
    {
        *finish_chunk = Some(chunk);
        return None;
    }

    Some(sse_line(&chunk))
}

/// Build a chunk carrying the consolidated tool calls, reusing the id and
/// metadata of the chunk it will precede.
fn tool_calls_chunk(reference: &Value, calls: Vec<Value>) -> bytes::Bytes {
    let mut chunk = reference.clone();
    chunk["choices"] = json!([{
        "index": 0,
        "delta": {"role": "assistant", "tool_calls": calls},
        "finish_reason": Value::Null,
    }]);
    if let Some(obj) = chunk.as_object_mut() {
        obj.remove("usage");
    }
    sse_line(&chunk)
}

fn sse_line(chunk: &Value) -> bytes::Bytes {
    bytes::Bytes::from(format!("data: {chunk}\n\n"))
}

fn sse_error(message: &str) -> bytes::Bytes {
    sse_line(&json!({"error": {"message": message}}))
}

/// An error raised while calling Vertex, shaped like an OpenAI error object so
/// clients can display it instead of reporting an invalid response.
pub fn upstream_error(msg: String) -> Response {
    (
        StatusCode::BAD_GATEWAY,
        Json(json!({"error": {"message": msg}})),
    )
        .into_response()
}
