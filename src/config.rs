// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! Configuration: which project, which region, which model names.
//!
//! [`Config::from_env`] is the convenience path used by the binary. Library
//! consumers can build a `Config` programmatically instead, without any
//! environment variables being involved.

use std::collections::HashMap;

/// All configuration needed to talk to Vertex AI.
///
/// ```
/// use gateway_for_vertex_ai::Config;
///
/// let cfg = Config::new("my-project").with_location("asia-southeast1");
/// assert_eq!(cfg.host(), "https://asia-southeast1-aiplatform.googleapis.com");
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    project: String,
    location: String,
    gateway_key: String,
    aliases: HashMap<String, String>,
    base_url_override: Option<String>,
}

impl Config {
    /// A config with sensible defaults: the `global` endpoint, no auth check,
    /// and the built-in model aliases.
    pub fn new(project: impl Into<String>) -> Self {
        Self {
            project: project.into(),
            location: "global".into(),
            gateway_key: String::new(),
            aliases: default_aliases(),
            base_url_override: None,
        }
    }

    /// Build from environment variables. Used by the binary; see the README for
    /// the full list.
    pub fn from_env() -> anyhow::Result<Self> {
        let project = std::env::var("GCP_PROJECT_ID")
            .map_err(|_| anyhow::anyhow!("missing environment variable GCP_PROJECT_ID"))?;

        Ok(Self {
            project,
            location: std::env::var("GCP_LOCATION").unwrap_or_else(|_| "global".into()),
            gateway_key: std::env::var("GATEWAY_API_KEY").unwrap_or_default(),
            aliases: parse_aliases()?.unwrap_or_else(default_aliases),
            base_url_override: None,
        })
    }

    /// Vertex region, or `global`. Defaults to `global`.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// Shared secret clients must present as a bearer token. An empty key
    /// disables the check entirely.
    pub fn with_gateway_key(mut self, key: impl Into<String>) -> Self {
        self.gateway_key = key.into();
        self
    }

    /// Replace the alias table wholesale.
    pub fn with_aliases(mut self, aliases: HashMap<String, String>) -> Self {
        self.aliases = aliases;
        self
    }

    /// Send requests somewhere other than `googleapis.com`.
    ///
    /// Intended for tests and for routing through a corporate proxy. The path
    /// suffix is still appended, so the override replaces only the scheme and
    /// host.
    pub fn with_base_url_override(mut self, base: impl Into<String>) -> Self {
        self.base_url_override = Some(base.into());
        self
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn location(&self) -> &str {
        &self.location
    }

    pub fn gateway_key(&self) -> &str {
        &self.gateway_key
    }

    pub fn aliases(&self) -> &HashMap<String, String> {
        &self.aliases
    }

    /// Vertex has two hostname shapes, and this is an easy place to get it
    /// wrong:
    /// - a specific region -> `https://asia-southeast1-aiplatform.googleapis.com`
    /// - `global`          -> `https://aiplatform.googleapis.com` (NO prefix)
    ///
    /// Building `global-aiplatform...` by mistake produces a DNS failure rather
    /// than a 404, which makes for a confusing error message.
    pub fn host(&self) -> String {
        if let Some(base) = &self.base_url_override {
            return base.clone();
        }
        if self.location == "global" {
            "https://aiplatform.googleapis.com".into()
        } else {
            format!("https://{}-aiplatform.googleapis.com", self.location)
        }
    }

    /// Base URL of the OpenAI compatibility layer. Note that `location` appears
    /// **twice**: once in the hostname and once in the path. Missing either one
    /// breaks the request.
    pub fn openai_base(&self) -> String {
        format!(
            "{}/v1/projects/{}/locations/{}/endpoints/openapi",
            self.host(),
            self.project,
            self.location
        )
    }
}

/// Read `MODEL_ALIASES` (JSON shaped like `{"alias":"google/model-id"}`).
///
/// Three cases are kept distinct, because collapsing them would hide user
/// mistakes:
/// - unset or blank -> `Ok(None)`, caller falls back to the defaults
/// - malformed JSON -> `Err`, reported at startup
/// - valid JSON     -> `Ok(Some(map))`
fn parse_aliases() -> anyhow::Result<Option<HashMap<String, String>>> {
    match std::env::var("MODEL_ALIASES") {
        Err(_) => Ok(None),
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => serde_json::from_str(&raw)
            .map(Some)
            .map_err(|e| anyhow::anyhow!("MODEL_ALIASES is not valid JSON: {e}")),
    }
}

/// Defaults so the gateway is usable with no extra configuration. Setting
/// `MODEL_ALIASES` replaces this list entirely rather than merging into it.
pub fn default_aliases() -> HashMap<String, String> {
    [
        ("gemini-pro", "google/gemini-2.5-pro"),
        ("gemini-flash", "google/gemini-2.5-flash"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_location_has_no_region_prefix() {
        let cfg = Config::new("p");
        assert_eq!(cfg.host(), "https://aiplatform.googleapis.com");
        assert!(cfg.openai_base().contains("/locations/global/"));
    }

    #[test]
    fn regional_location_has_a_prefix() {
        let cfg = Config::new("p").with_location("asia-southeast1");
        assert_eq!(
            cfg.host(),
            "https://asia-southeast1-aiplatform.googleapis.com"
        );
    }

    #[test]
    fn base_url_override_wins() {
        let cfg = Config::new("p").with_base_url_override("http://127.0.0.1:9911");
        assert!(cfg.openai_base().starts_with("http://127.0.0.1:9911/v1/"));
    }
}
