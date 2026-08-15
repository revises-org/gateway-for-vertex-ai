// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Proves the crate is usable as a library: everything here goes through the
//! public API only, with no network access and no environment variables.

use std::sync::Arc;

use async_trait::async_trait;
use gateway_for_vertex_ai::{router, AppState, Config, TokenSource};

/// A token source that never touches the network. This is the seam that used
/// to require patching the source to test.
struct FakeTokens;

#[async_trait]
impl TokenSource for FakeTokens {
    async fn token(&self) -> Result<String, String> {
        Ok("fake-token".to_string())
    }
}

#[test]
fn config_is_buildable_without_environment_variables() {
    let cfg = Config::new("my-project")
        .with_location("asia-southeast1")
        .with_gateway_key("secret");

    assert_eq!(cfg.project(), "my-project");
    assert_eq!(
        cfg.host(),
        "https://asia-southeast1-aiplatform.googleapis.com"
    );
    assert!(cfg.openai_base().ends_with("/endpoints/openapi"));
}

#[tokio::test]
async fn state_and_router_build_without_google_credentials() {
    let cfg = Config::new("my-project").with_base_url_override("http://127.0.0.1:1");
    let state = AppState::with_token_source(cfg, Arc::new(FakeTokens)).expect("state builds");

    assert_eq!(state.token().await.unwrap(), "fake-token");

    // The router is a plain axum Router, so a host application can nest it.
    let _app: axum::Router = axum::Router::new()
        .nest("/ai", router(state))
        .route("/", axum::routing::get(|| async { "portal" }));
}

#[tokio::test]
async fn chat_is_callable_without_the_router() {
    // Point at a port nothing is listening on: the call must fail cleanly with
    // an OpenAI-shaped error rather than panicking or hanging.
    let cfg = Config::new("my-project").with_base_url_override("http://127.0.0.1:1");
    let state = AppState::with_token_source(cfg, Arc::new(FakeTokens)).expect("state builds");

    let response = gateway_for_vertex_ai::chat(
        state,
        serde_json::json!({
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    )
    .await;

    // 502 is what a host application would relay onward.
    assert_eq!(response.status(), axum::http::StatusCode::BAD_GATEWAY);
}
