// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Router assembly and the protocol-independent endpoints.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

use crate::{protocols::openai, state::AppState};

/// Build the gateway router.
///
/// Returned as a plain [`axum::Router`] so it can be served on its own or
/// nested inside a larger application:
///
/// ```no_run
/// # use std::sync::Arc;
/// # async fn demo(state: Arc<gateway_for_vertex_ai::AppState>) {
/// let app = axum::Router::new().nest("/ai", gateway_for_vertex_ai::router(state));
/// # }
/// ```
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .with_state(state)
}

pub fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": {"message": "Invalid gateway API key"}})),
    )
        .into_response()
}

/// Check the caller's bearer token against the configured gateway key.
///
/// An empty configured key means no check at all, which is only safe when bound
/// to loopback.
pub fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    let expected = state.config().gateway_key();
    if expected.is_empty() {
        return true;
    }
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|k| k == expected)
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "project": st.config().project(),
        "location": st.config().location(),
    }))
}

/// List the configured aliases. Zed does not read this endpoint (it requires
/// `available_models` to be declared in settings.json), but other clients do,
/// and it is a quick way to check which names the gateway understands.
async fn list_models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&st, &headers) {
        return unauthorized();
    }
    let data: Vec<Value> = st
        .config()
        .aliases()
        .keys()
        .map(|id| json!({"id": id, "object": "model", "created": 0, "owned_by": "google"}))
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}
