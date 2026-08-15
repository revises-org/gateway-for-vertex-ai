# gateway-for-vertex-ai

An OpenAI-compatible gateway for Google Vertex AI. It holds your Google Cloud
credentials, refreshes access tokens automatically, and exposes a **static** API
key that editors can store.

[![CI](https://github.com/revises-org/gateway-for-vertex-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/revises-org/gateway-for-vertex-ai/actions/workflows/ci.yml)

## Why this exists

Vertex AI already speaks the OpenAI Chat Completions protocol, so in principle
any OpenAI-compatible client can talk to it directly. The blocker is auth:
Vertex expects a Google Cloud access token, and those **expire after about an
hour**. Editors like Zed and Cursor only store a static API key — they have no
way to run `gcloud` and refresh anything.

```
  editor  --(static key)-->  gateway  --(auto-refreshed token)-->  Vertex AI
```

That is the entire job. The gateway does not wrap, reinterpret, or "improve"
your requests — `messages`, `tools`, `reasoning_effort` and everything else are
forwarded untouched, so new Vertex features work without changing this code.

## Build

Requires Rust 1.75 or newer.

```bash
cargo build --release
```

The binary lands at `target/release/gateway-for-vertex-ai`.
It is a single self-contained executable — no runtime, no dependency directory.
Put it wherever you like.

## Authenticate

Create a service account with the `roles/aiplatform.user` role, download its
JSON key, and point the gateway at it:

```bash
chmod 600 ~/.config/gateway/sa.json
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gateway/sa.json
```

The key file is a credential that can spend money on your Google Cloud account.
Treat it like a password: never commit it, never share it, keep it at mode 600.

<details>
<summary>Alternative: user credentials, if you'd rather not keep a key file on disk</summary>

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project MY_PROJECT
```

This writes a refresh token to
`~/.config/gcloud/application_default_credentials.json`. Unlike a service
account key it is tied to your personal account, so it can be revoked and it
expires when you leave the organisation — but it is also less stable for
long-running processes, and it does not work on a server.

Note that `gcloud auth login` and `gcloud auth application-default login` are
different commands that write different files. Only the second one is picked up
here.

</details>

Credentials are resolved in this order, stopping at the first hit:

1. `GOOGLE_APPLICATION_CREDENTIALS` → service account key file
2. `~/.config/gcloud/application_default_credentials.json`
3. GCE metadata server — this is what makes the gateway work unmodified on
   Cloud Run, where no key file is needed at all

## Run

```bash
export GCP_PROJECT_ID=my-project
export GATEWAY_API_KEY=$(openssl rand -hex 16)   # generate once, then keep it
gateway-for-vertex-ai
```

`GATEWAY_API_KEY` is your own shared secret between editor and gateway; it has
nothing to do with Google. Generate it once and store it — regenerating gives
you a new key and every client configured with the old one starts getting 401s.

Check it came up:

```bash
curl -s localhost:8787/health
curl -s -H "Authorization: Bearer $GATEWAY_API_KEY" localhost:8787/v1/models
```

Startup is fail-fast: it reads the config, resolves credentials, and fetches one
token before binding the port. If your Google Cloud setup is wrong you find out
immediately, with a clear message, instead of halfway through a coding session.

Running it in a terminal tab is fine. If you want it to start with the machine,
`dist/` has a launchd plist (macOS) and a systemd user unit (Linux).

## Using it as a library

The binary is a thin wrapper; all the logic lives in the library, so a host
application can mount the gateway without starting a server of its own:

```rust
use gateway_for_vertex_ai::{router, AppState, Config};

let cfg = Config::new("my-project")
    .with_location("asia-southeast1")
    .with_gateway_key("shared-secret");

let state = AppState::discover(cfg).await?;

let app = axum::Router::new()
    .route("/", get(portal_home))
    .nest("/ai", router(state));
```

`router()` returns a plain `axum::Router`, so it nests anywhere. `Config` is
built programmatically — no environment variables are read unless you call
`Config::from_env()`.

If the host application already manages Google credentials, implement
`TokenSource` and pass it to `AppState::with_token_source` instead of letting
this crate rediscover them. That is also the seam used by the test suite, which
runs without network access.

For finer control, skip the router and call `chat()` directly — the path a
portal takes when it authenticates its own users and routes between several
providers:

```rust
let response = gateway_for_vertex_ai::chat(
    state,
    json!({"model": "gemini-pro", "messages": [...]}),
).await;
```

`chat()` performs no authentication and uses no axum extractors; it returns a
`Response` that already carries a live SSE stream when one was requested.

See `examples/embedded.rs` for a runnable version.

## Development

```bash
cargo check      # type-check only — fastest feedback loop
cargo run        # build (debug) and run
cargo test
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

`cargo run` builds in debug mode, which compiles much faster and runs slower —
the right trade-off while iterating. Add `--release` only when measuring
performance or producing a real binary.

There is no `.env` support, so environment variables have to be present in the
shell:

```bash
GCP_PROJECT_ID=my-project GATEWAY_API_KEY=devkey cargo run
```

To rebuild on save, `cargo install cargo-watch` then `cargo watch -x run`.

## Configure your editor

Point any OpenAI-compatible client at `http://127.0.0.1:8787/v1` and use
`GATEWAY_API_KEY` as the API key.

<details>
<summary>Zed</summary>

```json
{
  "language_models": {
    "openai_compatible": {
      "vertex": {
        "api_url": "http://127.0.0.1:8787/v1",
        "available_models": [
          {
            "name": "gemini-pro",
            "display_name": "Gemini 2.5 Pro (Vertex)",
            "max_tokens": 1048576,
            "max_output_tokens": 65536,
            "reasoning_effort": "medium",
            "capabilities": { "tools": true, "images": true }
          }
        ]
      }
    }
  }
}
```

Enter the API key through the UI so Zed stores it in your OS keychain, rather
than putting it in `settings.json`.

Note that Zed does not read `/v1/models` — you have to declare
`available_models` by hand. Adding a model means editing two places:
`MODEL_ALIASES` here and `settings.json` there.

</details>

## Gemini 3 thought signatures

Gemini 3 thinks before it emits a tool call, signs that reasoning, and attaches
the signature to the tool call in a non-standard field. It then **requires** the
signature back with the conversation history on the next turn. Standard OpenAI
clients drop the field, so the second turn of every tool-calling conversation
fails with `400 INVALID_ARGUMENT: Function call is missing a thought_signature`.

This is not obscure — VS Code Copilot, Codex CLI, Continue, LiteLLM and the
OpenAI Python SDK have all hit it.

A gateway sees both directions, so it can carry the signature itself. This one
folds it into the tool call id, which clients preserve verbatim because they
need it to match tool results to calls:

```
Vertex  -> gateway:  id = "call_abc", signature = "AY89..."
gateway -> client:   id = "call_abc__thought__AY89..."
client  -> gateway:  id = "call_abc__thought__AY89..."   (echoed unchanged)
gateway -> Vertex:   id = "call_abc", signature restored
```

No state is kept between requests: the signature travels with the conversation
rather than in server memory. Tool calling with Gemini 3 works through this
gateway with no client changes.

### Known limitation: id length

Signatures are encrypted data, so they cannot be shortened — truncating one
makes Vertex reject the request. Packed tool call ids therefore run roughly
150–800 characters, against the ~29 of a normal OpenAI id.

This is invisible to editors, which treat ids as opaque strings, and no
limit on id length exists in the OpenAI specification. It only matters if
something downstream stores ids under a fixed-width schema — if you persist
conversations, use `TEXT` rather than `VARCHAR(64)`.

Removing the length cost would mean holding signatures in server memory keyed
by id, which trades away statelessness: a cache needs eviction, and it breaks
as soon as more than one instance sits behind a load balancer. Not worth it
until someone hits the problem for real.

## Observing usage and cost

Cloud Billing lags by about a day and Cloud Monitoring aggregates across
everything, so neither answers "what did this request cost". The gateway already
parses every response, so it can report per-request numbers:

```rust
use gateway_for_vertex_ai::{AppState, CompletionEvent, Observer};

struct MyLogger;

#[async_trait::async_trait]
impl Observer for MyLogger {
    async fn on_completion(&self, event: CompletionEvent) {
        // event.model, event.status, event.duration, event.usage
    }
}

let state = AppState::discover(cfg).await?.with_observer(Arc::new(MyLogger));
```

Observers run on a detached task after the response is delivered, so a slow
logger cannot stall an editor, and an observer cannot fail a request.

`usage` is `Option`, not a zero default. When a client disconnects before the
final chunk, the token counts are genuinely unknown — those tokens were still
spent and will appear on the bill, so recording them as zero would make local
totals quietly drift below reality.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `GCP_PROJECT_ID` | *(required)* | the project ID, not its name or number |
| `GCP_LOCATION` | `global` | `global` has no region prefix in the hostname |
| `BIND_ADDR` | `127.0.0.1:8787` | port choice is arbitrary; change freely |
| `GATEWAY_API_KEY` | *(empty)* | empty disables the auth check |
| `MODEL_ALIASES` | two built-ins | JSON `{"alias":"google/model-id"}`, replaces the defaults entirely |
| `RUST_LOG` | `gateway=info` | logs go to stderr |

Model names are resolved through `MODEL_ALIASES`, then given a `google/` prefix
unless they already contain a `/` or are all digits. So `gemini-pro`,
`gemini-2.5-pro` and `google/gemini-2.5-pro` all reach the same model.

## Supported models

The `/v1/chat/completions` route covers everything Vertex exposes through its
OpenAI-compatible layer:

| Kind | How to name it | Example |
|---|---|---|
| Gemini | alias or bare name | `gemini-pro` |
| Third-party MaaS | publisher-prefixed | `meta/llama-4-scout-maas` |
| Self-deployed (Model Garden) | numeric endpoint ID | `5464397967697903616` |

**Claude on Vertex is not reachable through this route.** It uses a different
protocol — Anthropic Messages against `publishers/anthropic/models/...`, with
the model in the URL instead of the body. Adding it means a second route rather
than a translation layer; see the notes above `forward()` in `src/main.rs`.

## Security

This process holds credentials that can spend money on your Google Cloud
account.

- **Bind to loopback.** The default is `127.0.0.1`. Exposing it on `0.0.0.0`
  puts an unmetered Vertex endpoint on your network.
- **Leaving `GATEWAY_API_KEY` empty disables authentication entirely.** That is
  survivable on a personal machine bound to loopback, and a bad idea anywhere
  else.
- **Vertex bills per token and has no default hard cap.** If you deploy this
  somewhere public, set a budget alert and put real authentication in front of
  it.
- Never commit service account keys. See `.gitignore`.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2026 Huy Nguyen Nhu.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this work by you, as defined in the Apache-2.0 license, shall
be licensed as above, without any additional terms or conditions.

---

*This is an unofficial, community-maintained tool. It is not affiliated with,
endorsed by, or sponsored by Google LLC. "Vertex AI", "Google Cloud", and
"Gemini" are trademarks of Google LLC.*
