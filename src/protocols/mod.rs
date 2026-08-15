// Copyright 2026 Huy Nguyen Nhu
// SPDX-License-Identifier: Apache-2.0

//! One module per wire protocol Vertex speaks.
//!
//! Each module owns exactly two concerns: building the target URL, and shaping
//! the request body. Everything after that is [`crate::forward::forward`].
//!
//! A future `anthropic` module would live beside `openai` and serve
//! `/v1/messages`.

pub mod openai;
