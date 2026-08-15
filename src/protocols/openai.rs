// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! The OpenAI Chat Completions protocol.
//!
//! Serves Gemini, third-party MaaS models (`meta/llama-...`), and self-deployed
//! Model Garden models (addressed by numeric endpoint ID).

use std::{collections::HashMap, sync::Arc};

use axum::{extract::State, http::HeaderMap, response::Response, Json};
use serde_json::{json, Value};

use crate::{
    forward::forward,
    routes::{check_auth, unauthorized},
    state::AppState,
    thought::restore_request,
    upstream_error,
};

/// Fields an OpenAI client may send that Vertex rejects with a 400. We drop
/// them silently rather than letting the request fail — they are all metadata
/// that has no effect on the generated output.
const DROP_FIELDS: &[&str] = &["prompt_cache_key", "safety_identifier", "store", "metadata"];

/// Forward an OpenAI-shaped request to Vertex.
///
/// This is the embedding entry point: no authentication, no axum extractors,
/// just the protocol logic. A host application that does its own auth and
/// routing calls this directly.
///
/// The request body is passed through almost untouched. Only two things change:
/// the model name is resolved through the alias table, and a handful of fields
/// Vertex rejects are dropped. Everything else — messages, tools,
/// reasoning_effort — reaches Vertex exactly as the caller wrote it, so new
/// Vertex features work without changing this crate.
///
/// The returned [`Response`] carries a live SSE stream when the request asked
/// for one, so the caller can hand it straight back to its own client.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use serde_json::json;
/// # async fn demo(state: Arc<gateway_for_vertex_ai::AppState>) {
/// let response = gateway_for_vertex_ai::chat(
///     state,
///     json!({
///         "model": "gemini-pro",
///         "messages": [{"role": "user", "content": "hello"}],
///     }),
/// )
/// .await;
/// # }
/// ```
pub async fn chat(st: Arc<AppState>, mut body: Value) -> Response {
    let raw = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let resolved = resolve_model(st.config().aliases(), raw);

    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".into(), json!(resolved));
        for f in DROP_FIELDS {
            obj.remove(*f);
        }
    }

    // Put thought signatures back where Vertex expects them. Standard OpenAI
    // clients drop the field they arrived in, so the gateway smuggled them
    // inside tool call ids on the way out; see crate::thought.
    restore_request(&mut body);

    // Tool call accumulation only tracks choices[0]; see forward::process_line.
    // Rather than silently mishandling extra choices, say so. No editor sends
    // n > 1, and supporting it would mean one accumulator per choice for a case
    // nobody has hit.
    if body.get("n").and_then(Value::as_u64).is_some_and(|n| n > 1) {
        return upstream_error(
            "n > 1 is not supported: tool call signatures are tracked for one choice only".into(),
        );
    }

    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let url = format!("{}/chat/completions", st.config().openai_base());

    forward(st, url, body, streaming).await
}

/// HTTP handler for `POST /v1/chat/completions`.
///
/// Checks the gateway key, then defers to [`chat`]. Kept separate so embedders
/// can skip the auth layer and supply their own.
pub async fn chat_completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if !check_auth(&st, &headers) {
        return unauthorized();
    }
    chat(st, body).await
}

/// Resolve an alias, then add a publisher prefix if the user wrote a bare name.
///
/// Vertex expects `google/gemini-2.5-pro`; typing plain `gemini-2.5-pro` still
/// works thanks to this step. Two cases are left alone:
/// - the name already contains `/` — e.g. `meta/llama-4-scout-maas`
/// - the name is all digits — a self-deployed endpoint ID, e.g. `5464397967697903616`
pub fn resolve_model(aliases: &HashMap<String, String>, raw: String) -> String {
    let resolved = aliases.get(&raw).cloned().unwrap_or(raw);
    if resolved.contains('/') || resolved.chars().all(|c| c.is_ascii_digit()) {
        resolved
    } else {
        format!("google/{resolved}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::default_aliases;

    #[test]
    fn alias_is_resolved() {
        assert_eq!(
            resolve_model(&default_aliases(), "gemini-pro".into()),
            "google/gemini-2.5-pro"
        );
    }

    #[test]
    fn bare_name_gets_a_prefix() {
        assert_eq!(
            resolve_model(&default_aliases(), "gemini-3-pro".into()),
            "google/gemini-3-pro"
        );
    }

    #[test]
    fn name_with_slash_is_left_alone() {
        assert_eq!(
            resolve_model(&default_aliases(), "meta/llama-4-scout-maas".into()),
            "meta/llama-4-scout-maas"
        );
        assert_eq!(
            resolve_model(&default_aliases(), "google/gemini-2.5-flash".into()),
            "google/gemini-2.5-flash"
        );
    }

    #[test]
    fn numeric_endpoint_id_is_left_alone() {
        // Self-deployed Model Garden models are addressed by a bare numeric
        // endpoint ID. Prefixing it with "google/" makes Vertex return a 404.
        assert_eq!(
            resolve_model(&default_aliases(), "5464397967697903616".into()),
            "5464397967697903616"
        );
    }
}
