//! # ewm-app — the Noether-loop driver
//!
//! An LLM (deterministic stub first, then a real local model) plus
//! `ewm-cortex-fpga-v2` as the side-car context/memory back-end.
//!
//! ```text
//!         ┌────────────────────────────┐
//!         │  LLM (produces tokens)     │  fire-and-forget, never blocking
//!         └────────────┬───────────────┘
//!                      │ token turns
//!         ┌────────────▼───────────────┐
//!         │  ewm-app (the driver)      │  stateless; runs the Noether loop
//!         │  ingest → S(t) → evolve    │
//!         │  → persist → advance head  │
//!         └────────────┬───────────────┘
//!                      │ CIDs, wire commands
//!         ┌────────────▼───────────────┐
//!         │  ewm-cortex-fpga-v2        │  side-car: context + memory
//!         └────────────────────────────┘
//! ```
//!
//! The side-car stays *behind* the LLM: the LLM never waits on it, and the
//! cortex never sits in the inference path. The app is stateless between
//! steps — recovery is reading the head of the commit tree, never a replay.

pub mod app;
pub mod llm;
pub mod token;

pub use app::{CortexApp, AppError, TurnOutcome, TurnRecord, APP_ENCODING_NAME};
pub use llm::{OllamaLlm, StubLlm, TurnSource};
pub use token::TokenEncoding;
