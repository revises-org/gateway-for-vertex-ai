// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Mounting the gateway inside a larger application.
//!
//! Run against a real Vertex project:
//!     GCP_PROJECT_ID=my-project cargo run --example embedded
//!
//! Or against a stub upstream, with no Google credentials at all:
//!     GATEWAY_UPSTREAM=http://127.0.0.1:9911 cargo run --example embedded

use std::sync::Arc;

use async_trait::async_trait;
use gateway_for_vertex_ai::{router, AppState, Config, TokenSource};

/// A token source that never touches the network. A real portal would return a
/// token it already holds.
struct StaticToken(String);

#[async_trait]
impl TokenSource for StaticToken {
    async fn token(&self) -> Result<String, String> {
        Ok(self.0.clone())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cfg = Config::new("demo").with_gateway_key("secret123");
    if let Ok(base) = std::env::var("GATEWAY_UPSTREAM") {
        cfg = cfg.with_base_url_override(base);
    }

    let state = AppState::with_token_source(cfg, Arc::new(StaticToken("fake-token".into())))?;

    // The host application owns the root; the gateway is nested underneath.
    let app = axum::Router::new()
        .route("/", axum::routing::get(|| async { "portal home" }))
        .nest("/ai", router(state));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8787").await?;
    println!("listening on http://127.0.0.1:8787 (gateway under /ai)");
    axum::serve(listener, app).await?;
    Ok(())
}
