# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-15

### Added
- OpenAI-compatible `/v1/chat/completions` proxy to Vertex AI, with SSE
  streaming passed through untouched.
- Automatic Google Cloud token refresh via Application Default Credentials,
  service account keys, or the GCE metadata server.
- Model aliases through `MODEL_ALIASES`, plus automatic `google/` prefixing.
- Static bearer token auth (`GATEWAY_API_KEY`).
- `/health` and `/v1/models` endpoints.
- launchd and systemd unit files under `dist/`.

[Unreleased]: https://github.com/revises-org/gateway-for-vertex-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/revises-org/gateway-for-vertex-ai/releases/tag/v0.1.0
