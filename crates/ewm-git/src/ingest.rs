//! The streaming ingestor — de-vendored onto the `hllset-next-v2` foundation.
//!
//! Replaces the legacy `hllset-materialize::Ingestor` with the same public
//! shape, built on `hllset-morphisms::Ingest` (complete single-touch ingest)
//! plus the two things `ewm-git` still needs locally: the cumulative bit-TF
//! snapshot (`TFVec`) and the per-pass original-emission callback.

use hllset_core::core::hashing::sha1_hex;
use hllset_core::{HLLSet, TFVec};
use hllset_morphisms::Ingest;
use std::collections::{HashMap, HashSet};

/// Receives each produced original (one per seed channel per pass).
pub trait IngestSink {
    fn on_original(&mut self, hllset: &HLLSet, sha1: String);
}

/// Pass statistics (kept for API compatibility with the legacy ingestor).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IngestStats {
    pub tokens: u64,
    pub new_lut_entries: u64,
    pub channels: usize,
    pub active_bits: u64,
}

/// One streaming pass: the per-pass channel HLLSets plus stats.
#[derive(Clone, Debug)]
pub struct IngestOutput {
    pub stats: IngestStats,
    pub channels: Vec<HLLSet>,
}

/// The streaming ingestor: cumulative across passes.
#[derive(Clone, Debug)]
pub struct Ingestor {
    seeds: Vec<u64>,
    seen: HashSet<Vec<u8>>,
    token_tf: HashMap<Vec<u8>, u64>,
    bit_tf: TFVec,
    total_tokens: u64,
    total_new_entries: u64,
}

impl Ingestor {
    /// Create an ingestor with the given seeds (default `[0, 1, 2]`).
    pub fn new(seeds: &[u64]) -> Self {
        assert!(seeds.len() >= 2, "need at least 2 seeds for consensus");
        Self {
            seeds: seeds.to_vec(),
            seen: HashSet::new(),
            token_tf: HashMap::new(),
            bit_tf: TFVec::new(),
            total_tokens: 0,
            total_new_entries: 0,
        }
    }

    /// Per-token TF (0 if never seen).
    pub fn token_tf(&self, token: &[u8]) -> u64 {
        self.token_tf.get(token).copied().unwrap_or(0)
    }

    /// The cumulative bit-TF vector over all ingested HLLSets.
    pub fn bit_tf(&self) -> &TFVec {
        &self.bit_tf
    }

    /// Total tokens ingested across all passes.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Total new LUT entries across all passes.
    pub fn total_new_entries(&self) -> u64 {
        self.total_new_entries
    }

    /// One streaming pass over the tokens (complete, single-touch).
    ///
    /// Each token is fully processed before the next is read — the foundation
    /// `Ingest` sets all three seeded atoms, updates its per-seed LUTs, and
    /// increments TF in one pass. The per-pass channel HLLSets leave only at
    /// the end (as `IngestOutput::channels` and via `sink.on_original`).
    pub fn ingest_stream<I, B>(&mut self, tokens: I, sink: &mut impl IngestSink) -> IngestOutput
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut pass_tokens = 0u64;
        let mut new_entries = 0u64;

        let mut foundation: Ingest = Ingest::new();
        for token in tokens {
            let t = token.as_ref();
            if self.seen.insert(t.to_vec()) {
                new_entries += 1;
            }
            foundation.ingest_token(t);
            *self.token_tf.entry(t.to_vec()).or_insert(0) += 1;
            pass_tokens += 1;
        }

        let working: Vec<HLLSet> = foundation.hllsets.iter().cloned().collect();

        // Finalize: bit-TF accumulates one touch per channel HLLSet, then
        // each original is emitted to the sink (HLLSet-LUT provenance).
        for hll in &working {
            self.bit_tf.increment_from_hllset(hll, 1.0);
        }
        for hll in &working {
            let cid = sha1_hex(&hll.to_bytes());
            sink.on_original(hll, cid);
        }

        self.total_tokens += pass_tokens;
        self.total_new_entries += new_entries;

        let stats = IngestStats {
            tokens: pass_tokens,
            new_lut_entries: new_entries,
            channels: self.seeds.len(),
            active_bits: working.iter().map(|h| h.popcount()).max().unwrap_or(0),
        };

        IngestOutput {
            stats,
            channels: working,
        }
    }
}
