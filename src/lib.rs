// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! An OpenAI-compatible gateway for Vertex AI.
//!
//! # Why this exists
//!
//! Vertex AI already speaks the OpenAI Chat Completions protocol, so in theory
//! any client could talk to it directly. Authentication is the blocker: Vertex
//! wants a Google Cloud access token, and those expire after about an hour.
//! Editors (Zed, Cursor, ...) can only store a **static** API key.
//!
//! This crate sits in the middle and resolves exactly that mismatch:
//!
//! ```text
//!   editor  --(static key)-->  gateway  --(auto-refreshed token)-->  Vertex AI
//! ```
//!
//! # Using it as a library
//!
//! The binary is a thin wrapper. Everything is reusable — including mounting
//! the gateway inside a larger service:
//!
//! ```no_run
//! use std::sync::Arc;
//! use gateway_for_vertex_ai::{AppState, Config, router};
//!
//! # async fn demo() -> anyhow::Result<()> {
//! let cfg = Config::new("my-project")
//!     .with_location("asia-southeast1")
//!     .with_gateway_key("shared-secret");
//!
//! let state = AppState::discover(cfg).await?;
//!
//! let app = axum::Router::new()
//!     .nest("/ai", router(state))
//!     .route("/", axum::routing::get(|| async { "portal" }));
//! # Ok(())
//! # }
//! ```
//!
//! For finer control, skip the router and call [`chat`] directly — that is the
//! path a portal takes when it does its own authentication and routes between
//! several providers:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use serde_json::json;
//! # async fn demo(state: Arc<gateway_for_vertex_ai::AppState>) {
//! let response = gateway_for_vertex_ai::chat(
//!     state,
//!     json!({"model": "gemini-pro", "messages": []}),
//! )
//! .await;
//! # }
//! ```
//!
//! Bring your own credentials with [`AppState::with_token_source`] if the host
//! application already manages them.

pub mod config;
pub mod forward;
pub mod observe;
pub mod protocols;
pub mod routes;
pub mod sse;
pub mod state;
pub mod thought;

pub use config::{default_aliases, Config};
pub use forward::{forward, upstream_error};
pub use observe::{CompletionEvent, Observer, Usage};
pub use protocols::openai::chat;
pub use routes::router;
pub use state::{AppState, GcpTokenSource, TokenSource};
