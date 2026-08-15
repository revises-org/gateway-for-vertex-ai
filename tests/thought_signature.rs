// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for thought signature handling.
//!
//! These run against a stub upstream on loopback rather than mocking internals,
//! so they exercise the real HTTP path: request rewriting, SSE parsing, tool
//! call consolidation, and the observer hook.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use axum::{extract::State, routing::post, Json, Router};
use gateway_for_vertex_ai::{
    chat, AppState, CompletionEvent, Config, Observer, TokenSource, Usage,
};
use serde_json::{json, Value};

// ------------------------------------------------------------ test doubles

struct FakeTokens;

#[async_trait]
impl TokenSource for FakeTokens {
    async fn token(&self) -> Result<String, String> {
        Ok("fake-token".into())
    }
}

/// Records what the gateway forwarded, so tests can assert on it.
#[derive(Clone, Default)]
struct Seen(Arc<Mutex<Vec<Value>>>);

impl Seen {
    fn last(&self) -> Value {
        self.0.lock().unwrap().last().cloned().expect("a request")
    }
}

#[derive(Default)]
struct Events(Arc<Mutex<Vec<CompletionEvent>>>);

#[async_trait]
impl Observer for Events {
    async fn on_completion(&self, event: CompletionEvent) {
        self.0.lock().unwrap().push(event);
    }
}

/// Spawn a stub that answers like Vertex and returns its address.
async fn spawn_upstream(seen: Seen, reply: Reply) -> SocketAddr {
    async fn handler(
        State((seen, reply)): State<(Seen, Reply)>,
        Json(body): Json<Value>,
    ) -> axum::response::Response {
        seen.0.lock().unwrap().push(body);
        match &reply {
            Reply::Json(v) => Json(v.clone()).into_response(),
            Reply::Sse(chunks) => {
                let payload = chunks.join("");
                axum::response::Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(axum::body::Body::from(payload))
                    .unwrap()
            }
        }
    }
    use axum::response::IntoResponse;

    let app = Router::new()
        .route(
            "/v1/projects/p/locations/global/endpoints/openapi/chat/completions",
            post(handler),
        )
        .with_state((seen, reply));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

#[derive(Clone)]
enum Reply {
    Json(Value),
    Sse(Vec<String>),
}

fn state_for(addr: SocketAddr) -> Arc<AppState> {
    let cfg = Config::new("p").with_base_url_override(format!("http://{addr}"));
    AppState::with_token_source(cfg, Arc::new(FakeTokens)).unwrap()
}

async fn body_of(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn sse(chunk: Value) -> String {
    format!("data: {chunk}\n\n")
}

// ---------------------------------------------------------------- tests

#[tokio::test]
async fn non_streaming_signature_is_packed_into_the_id() {
    let seen = Seen::default();
    let addr = spawn_upstream(
        seen.clone(),
        Reply::Json(json!({
            "choices":[{"message":{"role":"assistant","tool_calls":[{
                "id":"call_abc","type":"function",
                "function":{"name":"read_file","arguments":"{}"},
                "extra_content":{"google":{"thought_signature":"SIG1"}}
            }]}}],
            "usage":{"prompt_tokens":8,"completion_tokens":6,
                     "completion_tokens_details":{"reasoning_tokens":917}}
        })),
    )
    .await;

    let resp = chat(
        state_for(addr),
        json!({"model":"gemini-pro","messages":[{"role":"user","content":"hi"}]}),
    )
    .await;

    let out: Value = serde_json::from_str(&body_of(resp).await).unwrap();
    assert_eq!(
        out.pointer("/choices/0/message/tool_calls/0/id").unwrap(),
        "call_abc__thought__SIG1"
    );
    assert!(
        out.pointer("/choices/0/message/tool_calls/0/extra_content")
            .is_none(),
        "the signature must travel once, not twice"
    );
}

#[tokio::test]
async fn client_echo_is_restored_before_reaching_vertex() {
    let seen = Seen::default();
    let addr = spawn_upstream(seen.clone(), Reply::Json(json!({"choices":[]}))).await;

    // Exactly what a standard OpenAI client sends back: the id echoed verbatim,
    // and no trace of the field the signature originally arrived in.
    chat(
        state_for(addr),
        json!({"model":"gemini-pro","messages":[
            {"role":"user","content":"read a.rs"},
            {"role":"assistant","tool_calls":[{
                "id":"call_abc__thought__SIG1","type":"function",
                "function":{"name":"read_file","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call_abc__thought__SIG1","content":"fn main(){}"}
        ]}),
    )
    .await;

    let forwarded = seen.last();
    assert_eq!(
        forwarded.pointer("/messages/1/tool_calls/0/id").unwrap(),
        "call_abc"
    );
    assert_eq!(
        forwarded
            .pointer("/messages/1/tool_calls/0/extra_content/google/thought_signature")
            .unwrap(),
        "SIG1",
        "this is the field whose absence returns 400 from Gemini 3"
    );
    assert_eq!(
        forwarded.pointer("/messages/2/tool_call_id").unwrap(),
        "call_abc",
        "tool results are matched by id, so both sides must be unpacked"
    );
}

#[tokio::test]
async fn streaming_tool_calls_are_consolidated_with_the_signature() {
    let seen = Seen::default();
    // The signature arrives with the first fragment; arguments arrive in
    // pieces afterwards. This is the shape that loses signatures elsewhere.
    let chunks = vec![
        sse(
            json!({"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{
            "index":0,"id":"call_abc","type":"function",
            "function":{"name":"read_file","arguments":""},
            "extra_content":{"google":{"thought_signature":"SIG_STREAM"}}}]}}]}),
        ),
        sse(
            json!({"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{
            "index":0,"function":{"arguments":"{\"path\":"}}]}}]}),
        ),
        sse(
            json!({"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{
            "index":0,"function":{"arguments":"\"a.rs\"}"}}]}}]}),
        ),
        sse(
            json!({"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],
                   "usage":{"prompt_tokens":5,"completion_tokens":12,
                            "completion_tokens_details":{"reasoning_tokens":40}}}),
        ),
        "data: [DONE]\n\n".to_string(),
    ];
    let addr = spawn_upstream(seen.clone(), Reply::Sse(chunks)).await;

    let resp = chat(
        state_for(addr),
        json!({"model":"gemini-pro","stream":true,"messages":[]}),
    )
    .await;

    let out = body_of(resp).await;

    // One consolidated tool call, with the signature folded into the id.
    assert!(
        out.contains("call_abc__thought__SIG_STREAM"),
        "signature missing from stream output:\n{out}"
    );
    assert!(
        out.contains(r#"{\"path\":\"a.rs\"}"#),
        "arguments must still be joined correctly:\n{out}"
    );
    assert!(out.trim_end().ends_with("data: [DONE]"), "terminator last");

    // Ordering matters: the consolidated call must precede the finish chunk,
    // or a client will consider the turn over before it sees the call.
    let call_at = out.find("call_abc__thought__").unwrap();
    let finish_at = out
        .find("tool_calls\"")
        .map(|_| out.find("finish_reason").unwrap());
    assert!(
        call_at < finish_at.unwrap(),
        "tool call must come first:\n{out}"
    );
}

#[tokio::test]
async fn parallel_streaming_tool_calls_keep_their_own_signatures() {
    let seen = Seen::default();
    let chunks = vec![
        sse(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"call_a","function":{"name":"read","arguments":""},
             "extra_content":{"google":{"thought_signature":"SIG_A"}}},
            {"index":1,"id":"call_b","function":{"name":"grep","arguments":""},
             "extra_content":{"google":{"thought_signature":"SIG_B"}}}]}}]})),
        // Fragments interleave across indices.
        sse(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":1,"function":{"arguments":"{\"q\":1}"}}]}}]})),
        sse(json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":"{\"p\":2}"}}]}}]})),
        sse(json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]})),
        "data: [DONE]\n\n".to_string(),
    ];
    let addr = spawn_upstream(seen.clone(), Reply::Sse(chunks)).await;

    let resp = chat(
        state_for(addr),
        json!({"model":"gemini-pro","stream":true,"messages":[]}),
    )
    .await;
    let out = body_of(resp).await;

    assert!(out.contains("call_a__thought__SIG_A"), "{out}");
    assert!(out.contains("call_b__thought__SIG_B"), "{out}");
    assert!(out.contains(r#"{\"p\":2}"#), "call_a arguments:\n{out}");
    assert!(out.contains(r#"{\"q\":1}"#), "call_b arguments:\n{out}");
}

#[tokio::test]
async fn plain_text_streaming_is_unaffected() {
    let seen = Seen::default();
    let chunks = vec![
        sse(json!({"choices":[{"index":0,"delta":{"content":"Xin"}}]})),
        sse(json!({"choices":[{"index":0,"delta":{"content":" chào 😀"}}]})),
        sse(json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})),
        "data: [DONE]\n\n".to_string(),
    ];
    let addr = spawn_upstream(seen.clone(), Reply::Sse(chunks)).await;

    let resp = chat(
        state_for(addr),
        json!({"model":"gemini-pro","stream":true,"messages":[]}),
    )
    .await;
    let out = body_of(resp).await;

    assert!(out.contains("Xin"), "{out}");
    assert!(
        out.contains("chào 😀"),
        "multi-byte text must survive:\n{out}"
    );
    assert!(out.contains("finish_reason"), "{out}");
}

#[tokio::test]
async fn observer_receives_token_counts() {
    let seen = Seen::default();
    let addr = spawn_upstream(
        seen.clone(),
        Reply::Json(json!({
            "choices":[{"message":{"content":"hi"}}],
            "usage":{"prompt_tokens":8,"completion_tokens":6,
                     "completion_tokens_details":{"reasoning_tokens":917}}
        })),
    )
    .await;

    let events = Arc::new(Events::default());
    let state = state_for(addr).with_observer(events.clone());

    chat(state, json!({"model":"gemini-pro","messages":[]})).await;

    // The observer runs on a detached task, so give it a moment.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let recorded = events.0.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].model, "google/gemini-2.5-pro");
    assert_eq!(recorded[0].status, 200);
    assert_eq!(
        recorded[0].usage,
        Some(Usage {
            prompt_tokens: 8,
            completion_tokens: 6,
            reasoning_tokens: 917,
        }),
        "reasoning tokens are billed but invisible; they must be reported"
    );
}

#[tokio::test]
async fn truncated_stream_reports_an_error_instead_of_partial_json() {
    let seen = Seen::default();
    let chunks = vec![
        sse(json!({"choices":[{"index":0,"delta":{"content":"Xin"}}]})),
        r#"data: {"choices":[{"index":0,"delta":{"content":"ch"#.to_string(),
    ];
    let addr = spawn_upstream(seen.clone(), Reply::Sse(chunks)).await;

    let resp = chat(
        state_for(addr),
        json!({"model":"gemini-pro","stream":true,"messages":[]}),
    )
    .await;

    // A truncated turn must not read cleanly to the end: an SDK that stops at
    // `[DONE]` would otherwise report success for an incomplete response.
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await;
    assert!(bytes.is_err(), "the body must end abnormally:\n{bytes:?}");
}

#[tokio::test]
async fn tool_calls_survive_a_stream_that_ends_without_finish_reason() {
    let seen = Seen::default();
    // A complete tool call, then the stream simply stops — no finish_reason,
    // no [DONE]. Everything is well-formed; the upstream just closed early.
    let chunks = vec![sse(json!({"choices":[{"index":0,"delta":{"tool_calls":[{
            "index":0,"id":"call_abc","type":"function",
            "function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"},
            "extra_content":{"google":{"thought_signature":"SIG_NOFINISH"}}}]}}]}))];
    let addr = spawn_upstream(seen.clone(), Reply::Sse(chunks)).await;

    let resp = chat(
        state_for(addr),
        json!({"model":"gemini-pro","stream":true,"messages":[]}),
    )
    .await;
    let out = body_of(resp).await;

    assert!(
        out.contains("call_abc__thought__SIG_NOFINISH"),
        "tool calls must not be dropped when finish_reason never arrives:\n{out}"
    );
    assert!(
        out.contains("finish_reason"),
        "a terminal chunk must be synthesised:\n{out}"
    );
    assert!(out.trim_end().ends_with("data: [DONE]"), "{out}");
}

#[tokio::test]
async fn multiple_choices_are_rejected_rather_than_mishandled() {
    let seen = Seen::default();
    let addr = spawn_upstream(seen.clone(), Reply::Json(json!({"choices":[]}))).await;

    let resp = chat(
        state_for(addr),
        json!({"model":"gemini-pro","n":2,"messages":[]}),
    )
    .await;

    assert_eq!(resp.status(), axum::http::StatusCode::BAD_GATEWAY);
    let out = body_of(resp).await;
    assert!(out.contains("n > 1 is not supported"), "{out}");
}
