//! # ewm-git — content-addressed evolution store for HLLSets
//!
//! A minimal, 2005-style Git as the replacement for the temporal pyramid:
//!
//! - **Commit is the timer.** The SHA1 of a commit is the only time stamp
//!   the evolution needs; wall-clock never enters object identity.
//! - **The state is a union HLLSet.** The commit tree is a serialized
//!   HLLSet — the current state of the system.
//! - **Diffs are lattice differences.** `D = H(t-1) \ S(t)`,
//!   `R = H(t-1) ∩ S(t)`, `N = S(t) \ H(t-1)`.
//! - **Merge is the lattice join**, recorded as a two-parent commit.
//! - **Pruning is GC** — objects unreachable from `HEAD`/branches are
//!   dropped (policy-driven forgetting, unlike an append-only LUT).

pub mod hllset_lut;
pub mod ingest;
pub mod object;
pub mod repo;
pub mod store;
pub mod view;

pub use hllset_lut::{BitTf, HllsetLut};
pub use ingest::{IngestOutput, IngestSink, IngestStats, Ingestor};
pub use object::{Commit, Gx, Object, ObjectId};
pub use repo::{ContextWarning, GcReport, LatticeState, Repository};
pub use store::{LooseStore, MemoryStore, ObjectStore, StoreError};
pub use view::{view, view_channel, views, CommitView};

/// The canonical HLLSet key for an object: `h:<sha1>`.
///
/// This is the address convention shared with `hllset-storage` (and the
/// IPFS archive backends), so any archived object can be retrieved by its
/// content address from any realm.
pub fn hllset_key(id: &ObjectId) -> String {
    format!("h:{}", id.as_str())
}
