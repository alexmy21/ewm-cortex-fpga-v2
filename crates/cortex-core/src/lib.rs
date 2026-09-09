//! # cortex-core — enhanced hllset-cortex pipeline (Rust port)
//!
//! The black-box interface between the DeepSeek-OCR encoder and decoder,
//! ported from the reference implementation and built on the tested
//! ewm-cortex primitives:
//!
//! ```text
//! tokens → hash → tokenLUT → HLLSet → materialize → gate_TF → restored → decoder
//! ```
//!
//! Modules:
//!
//! - [`encoding`] — simulated `tid{n}` encoder/decoder (opaque IDs only)
//! - [`gate`] — `gate_TF` output TokenGate (decoder-vocabulary limit) + exact membership
//! - [`lut`] — monotonic TF reverse index (collection intersection per bit; TF only on collision ties)
//! - [`pipeline`] — the [`CortexPipeline`] black box
//! - [`context`] — the grow-only [`ContextMatrix`] (moved from the bridge;
//!   the authoritative memory of the cortex, per the separation contract)
//! - [`grounding`] — the τ/ρ grounding verdict over the matrix
//! - [`role`] — the expert/leader role flag and the leader's registry+merge

/// The LLM token id — the cortex-side vocabulary type.
pub type TokenId = u32;

pub mod context;
pub mod context_unified;
pub mod encoding;
pub mod gate;
pub mod grounding;
pub mod lut;
pub mod pipeline;
pub mod role;

pub use context::ContextMatrix;
pub use context_unified::Context;
pub use encoding::{tid, SimCodec};
pub use gate::Gate;
pub use grounding::{recommend, recommend_with, GroundingConfig, GroundingReport};
pub use lut::TfLut;
pub use pipeline::{CortexPipeline, PipelineResult};
pub use role::{CortexConfig, Leader, Role};
