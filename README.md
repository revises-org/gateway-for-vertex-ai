# gateway-for-vertex-ai

> **This project is now part of [cram](https://github.com/revises-org/cram).**
>
> cram is the same gateway with a dashboard, config file, and room for more
> platforms. The Vertex provider lives on as the `cram-vertex` crate.
>
> This repository is kept for reference. It is not maintained.

---

## What this was

An OpenAI-compatible gateway for Google Vertex AI. It held Google Cloud
credentials, refreshed access tokens automatically, and exposed a **static** API
key that editors could store.

Vertex AI already speaks the OpenAI Chat Completions protocol. The blocker was
auth: Vertex expects a Google Cloud access token, and those expire after about
an hour. Editors like Zed and Cursor only store a static API key — they have no
way to run `gcloud` and refresh anything.

```
  editor  --(static key)-->  gateway  --(auto-refreshed token)-->  Vertex AI
```

## The Gemini 3 thought signature problem

If you landed here searching for this error:

```
400 INVALID_ARGUMENT: Function call is missing a thought_signature in
functionCall parts. This is required for tools to work correctly.
```

Here is what causes it, and how this gateway solved it.

Gemini 3 thinks before it emits a tool call, signs that reasoning, and attaches
the signature to the tool call in a **non-standard** field:

```json
"tool_calls": [{
  "id": "call_abc",
  "extra_content": { "google": { "thought_signature": "AY89..." } }
}]
```

Gemini 3 then *requires* that signature back with the conversation history on
the next turn. Any standard OpenAI client drops `extra_content` because it does
not know the field, so the second turn of every tool-calling conversation fails.

This is not obscure — VS Code Copilot, Codex CLI, Continue, LiteLLM and the
OpenAI Python SDK have all hit it. The client cannot fix it without learning a
Google-specific field, which defeats the point of using an OpenAI-compatible
endpoint.

### The fix

A gateway sees both directions, so it can carry the signature itself. Fold it
into the tool call **id** — the one field a client is guaranteed to preserve
verbatim, because it needs it to match tool results to calls.

```
Vertex  -> gateway:  id = "call_abc", signature = "AY89..."
gateway -> client:   id = "call_abc__thought__AY89..."
client  -> gateway:  id = "call_abc__thought__AY89..."   (echoed unchanged)
gateway -> Vertex:   id = "call_abc", signature restored
```

No state is kept between requests: the signature travels with the conversation
rather than in server memory.

Two details that matter in practice:

**Escape the signature before packing it.** Signatures are base64 and contain
`+`, `/` and `=`, so appending one raw produces an id outside the charset some
clients validate.

**Merge streamed deltas recursively.** Tool calls arrive fragmented across
chunks. The obvious accumulator copies the fields it knows about — id, name,
arguments — and silently discards `extra_content` along with the signature.
That is exactly how this bug gets reintroduced.

The working implementation is in
[`crates/cram-vertex/src/thought.rs`](https://github.com/revises-org/cram/blob/main/crates/cram-vertex/src/thought.rs).

## Migrating

```bash
cargo install cram
cram auth vertex --key-file /path/to/sa.json
cram
```

`gateway-for-vertex-ai` on crates.io is yanked. Existing installs keep working;
new work should use [cram](https://github.com/revises-org/cram).

Beyond what was here, cram adds a request dashboard with time-to-first-token and
cached-token accounting, a config file so credentials are not passed on every
run, and a workspace laid out for Bedrock and Azure AI Foundry.

## License

[Apache 2.0](LICENSE). Copyright 2026 Huy Nguyen Nhu.

---

*Unofficial and community-maintained. Not affiliated with, endorsed by, or
sponsored by Google LLC. "Vertex AI" and "Gemini" are trademarks of Google LLC.*
