//! `cortex-fpga` — the ewm-cortex pipeline re-expressed over
//! `ewm-fpga-bridge` (milestone M4-fpga).
//!
//! The original `ewm-cortex` keeps its self-contained first-party pipeline;
//! this crate is the **bridge test application**: it declares the cortex
//! black-box as an `ewm-dsl` pipeline, submits wire commands to the bridge's
//! `SimModuleDriver` backend, and crosschecks bit-exactly against the
//! `hllset-next-v2`/`cortex-core` reference.

#![forbid(unsafe_code)]

pub mod context;
pub mod golden;

#[cfg(test)]
mod crosscheck;

pub use context::{
    context_tree_for, full_image, h_key, leaf_for, unified_context_for, view_record,
};
pub use golden::{cortex_pipeline, run_cortex_pipeline, view_from_tokens, FpgaPipelineResult};
