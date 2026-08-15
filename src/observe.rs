// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! An optional hook for logging and cost accounting.
//!
//! # Why this exists in the gateway rather than in Google Cloud
//!
//! Cloud Billing reports lag by about a day and Cloud Monitoring aggregates
//! across everything. Neither can answer "which request cost what". The gateway
//! already sees the `usage` block on every response, so it is the natural place
//! to emit per-request numbers.
//!
//! # Why observation never blocks the response
//!
//! Observers are invoked after the response has been delivered, on a detached
//! task. A slow database write or a panicking logger must not stall someone's
//! editor. This also means an observer cannot veto a request — it watches, it
//! does not control.

use std::time::Duration;

use async_trait::async_trait;

/// Token counts as reported by Vertex.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Tokens spent on internal reasoning. Billed, but never shown to the user.
    /// A one-line greeting to a thinking model can spend hundreds of these, so
    /// request counts are a poor proxy for cost.
    pub reasoning_tokens: u64,
}

/// What happened during one completion.
#[derive(Debug, Clone)]
pub struct CompletionEvent {
    /// The resolved model id actually sent to Vertex, e.g. `google/gemini-2.5-pro`.
    pub model: String,
    pub status: u16,
    pub streamed: bool,
    pub duration: Duration,
    /// `None` when the token counts are unknown — most often because the client
    /// disconnected before the final chunk arrived.
    ///
    /// This is deliberately not defaulted to zero. Those tokens were spent and
    /// will appear on the bill; recording them as zero would make local totals
    /// quietly drift below reality.
    pub usage: Option<Usage>,
}

/// Receives one event per completion.
#[async_trait]
pub trait Observer: Send + Sync + 'static {
    async fn on_completion(&self, event: CompletionEvent);
}

/// Parse the `usage` object from a Vertex response.
///
/// Reasoning tokens live in a nested `completion_tokens_details` object rather
/// than at the top level.
pub fn parse_usage(usage: &serde_json::Map<String, serde_json::Value>) -> Usage {
    let num = |v: Option<&serde_json::Value>| v.and_then(serde_json::Value::as_u64).unwrap_or(0);
    Usage {
        prompt_tokens: num(usage.get("prompt_tokens")),
        completion_tokens: num(usage.get("completion_tokens")),
        reasoning_tokens: num(usage
            .get("completion_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_nested_reasoning_tokens() {
        let raw = json!({
            "prompt_tokens": 8,
            "completion_tokens": 6,
            "total_tokens": 931,
            "completion_tokens_details": {"reasoning_tokens": 917}
        });
        let u = parse_usage(raw.as_object().unwrap());
        assert_eq!(u.prompt_tokens, 8);
        assert_eq!(u.completion_tokens, 6);
        assert_eq!(u.reasoning_tokens, 917);
    }

    #[test]
    fn missing_fields_default_to_zero() {
        let raw = json!({"prompt_tokens": 1});
        let u = parse_usage(raw.as_object().unwrap());
        assert_eq!(u.prompt_tokens, 1);
        assert_eq!(u.reasoning_tokens, 0);
    }
}
