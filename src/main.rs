// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Thin binary wrapper: read the environment, bind a port, serve.
//!
//! All the logic lives in the library so it can be reused without starting a
//! server. See the crate documentation for embedding.

use gateway_for_vertex_ai::{router, AppState, Config};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gateway_for_vertex_ai=info".into()),
        )
        .init();

    let cfg = Config::from_env()?;
    if cfg.gateway_key().is_empty() {
        tracing::warn!("GATEWAY_API_KEY is empty — the gateway will not check authentication");
    }

    // Fails fast if credentials are missing or unusable.
    let state = AppState::discover(cfg).await?;

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8787".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("gateway-for-vertex-ai listening on http://{addr}");

    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received, exiting");
}
