// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Shared state: the HTTP client and whatever supplies access tokens.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;

use crate::{
    config::Config,
    observe::{CompletionEvent, Observer},
};

/// Give up if the TCP/TLS handshake does not complete. A healthy connection to
/// Google is set up in well under a second, so this only fires when something
/// is genuinely wrong.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Give up if the upstream goes silent for this long **between reads**.
///
/// This is deliberately not a total-request timeout. A long streamed answer can
/// legitimately run for many minutes, and a total timeout would cut it off
/// mid-sentence. A read timeout only fires when nothing arrives at all, which
/// is the case we actually want to catch: without it, a hung upstream hangs the
/// gateway, and the client waits forever.
pub const READ_TIMEOUT: Duration = Duration::from_secs(120);

const SCOPES: &[&str] = &["https://www.googleapis.com/auth/cloud-platform"];

/// Anything that can produce a bearer token for Vertex AI.
///
/// This trait exists so the gateway is not welded to one credential mechanism.
/// [`GcpTokenSource`] is the real implementation; tests and embedders can
/// supply their own — a portal that already holds a token, say, or a fake in a
/// test suite that must not touch the network.
#[async_trait]
pub trait TokenSource: Send + Sync + 'static {
    async fn token(&self) -> Result<String, String>;
}

/// Tokens from Application Default Credentials.
///
/// Resolution order is `GOOGLE_APPLICATION_CREDENTIALS` -> the ADC file
/// (`~/.config/gcloud/application_default_credentials.json`) -> the GCE
/// metadata server. That last branch is why this runs unmodified on Cloud Run,
/// where credentials come from the instance's service account.
pub struct GcpTokenSource {
    provider: Arc<dyn gcp_auth::TokenProvider>,
}

impl GcpTokenSource {
    /// Discover credentials and verify they work.
    ///
    /// A token is fetched eagerly so that broken credentials surface here,
    /// rather than halfway through someone's coding session.
    pub async fn discover() -> anyhow::Result<Self> {
        let provider = gcp_auth::provider()
            .await
            .map_err(|e| anyhow::anyhow!("could not initialise GCP credentials: {e}"))?;

        provider
            .token(SCOPES)
            .await
            .map_err(|e| anyhow::anyhow!("GCP credentials are not usable: {e}"))?;

        Ok(Self { provider })
    }
}

#[async_trait]
impl TokenSource for GcpTokenSource {
    /// Looks like a network call every time, but it isn't: `gcp_auth` caches
    /// the token in memory and only contacts Google when it is close to
    /// expiring. All expiry tracking lives in the library — the single easiest
    /// part of this problem to get wrong is the part we don't write.
    async fn token(&self) -> Result<String, String> {
        self.provider
            .token(SCOPES)
            .await
            .map(|t| t.as_str().to_string())
            .map_err(|e| format!("could not obtain a GCP access token: {e}"))
    }
}

/// State shared across every request, handed around behind an `Arc`.
pub struct AppState {
    http: reqwest::Client,
    tokens: Arc<dyn TokenSource>,
    observer: Option<Arc<dyn Observer>>,
    cfg: Config,
}

impl AppState {
    /// Build state using Application Default Credentials.
    pub async fn discover(cfg: Config) -> anyhow::Result<Arc<Self>> {
        let tokens = GcpTokenSource::discover().await?;
        Self::with_token_source(cfg, Arc::new(tokens))
    }

    /// Build state with a caller-supplied token source.
    ///
    /// This is the seam for embedding: a portal that already manages Google
    /// credentials can hand them in instead of having this crate rediscover
    /// them.
    pub fn with_token_source(
        cfg: Config,
        tokens: Arc<dyn TokenSource>,
    ) -> anyhow::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            // One client for the whole process, so the connection pool is
            // reused. Negotiating fresh TLS per request would add hundreds of
            // milliseconds to time-to-first-token.
            http: reqwest::Client::builder()
                .connect_timeout(CONNECT_TIMEOUT)
                .read_timeout(READ_TIMEOUT)
                .pool_idle_timeout(Duration::from_secs(90))
                .build()?,
            tokens,
            observer: None,
            cfg,
        }))
    }

    /// Attach an observer for logging and cost accounting.
    ///
    /// Takes `Arc<Self>` and returns a new one because state is shared
    /// immutably across requests; there is no interior mutability to hide.
    pub fn with_observer(self: Arc<Self>, observer: Arc<dyn Observer>) -> Arc<Self> {
        Arc::new(Self {
            http: self.http.clone(),
            tokens: self.tokens.clone(),
            observer: Some(observer),
            cfg: self.cfg.clone(),
        })
    }

    /// Hand an event to the observer, if one is attached.
    ///
    /// Fire-and-forget on a detached task: a slow or broken observer must never
    /// delay a response, and must never be able to fail a request.
    pub fn notify(&self, event: CompletionEvent) {
        if let Some(observer) = &self.observer {
            let observer = observer.clone();
            tokio::spawn(async move { observer.on_completion(event).await });
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub async fn token(&self) -> Result<String, String> {
        self.tokens.token().await
    }
}
