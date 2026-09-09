//! K-storage: the LUTs as the K side of attention (de-vendored).
//!
//! A [`KStorage`] resolves tokens to [`KeyRef`]s (the K direction) and
//! HLLSets to candidate tokens (the V direction), with a coverage gauge
//! [`KStorage::confidence`]. The two provided storages are built on the
//! `hllset-next-v2` foundation:
//!
//! - [`TokenLutStorage`] — tier 1, single-seed, over `hllset-lut::LutIndex`.
//! - [`CatalogLutStorage`] — tier 2, multi-seed quorum, over a local
//!   per-seed catalog (consensus implemented here; the foundation keeps the
//!   single morphism, the quorum is an application-level decision).

use hllset_core::core::hashing::{
    hash_to_position, murmur3_hash, murmur3_hash_seeded, token_to_position,
    token_to_position_seeded,
};
use hllset_core::{BITS_PER_REG, HLLSet};
use hllset_lut::LutIndex;

/// Default catalog seeds (G1 convention).
pub const DEFAULT_CATALOG_SEEDS: [u64; 3] = [0, 1, 2];

/// A key reference into the HLLSet realm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyRef {
    /// One bit cell (tier 1): `bit = reg * 32 + tz`.
    Cell(u32),
    /// Multi-seed cell set (tier 2): one cell per seed, plus the quorum
    /// (minimum seeds that must match for consensus).
    Cells(Vec<u32>, usize),
}

impl KeyRef {
    /// All cell positions carried by this key.
    pub fn cells(&self) -> Vec<u32> {
        match self {
            KeyRef::Cell(bit) => vec![*bit],
            KeyRef::Cells(cells, _) => cells.clone(),
        }
    }

    /// Consensus quorum, if this is a multi-seed key.
    pub fn quorum(&self) -> Option<usize> {
        match self {
            KeyRef::Cell(_) => None,
            KeyRef::Cells(_, q) => Some(*q),
        }
    }
}

/// K-storage: a reverse index that resolves tokens to keys and keys to tokens.
///
/// The coverage invariant (`confidence == 1.0`) is the single condition
/// under which sub-lattice extraction is precise.
pub trait KStorage {
    /// The key(s) a token maps to under this storage.
    fn key_of(&self, token: &[u8]) -> KeyRef;

    /// Materialized candidate tokens for a sub-lattice HLLSet (the V side).
    fn candidates(&self, hllset: &HLLSet) -> Vec<Vec<u8>>;

    /// Coverage gauge: fraction of `hllset` bits resolvable through the LUT.
    /// `1.0` iff every set bit has at least one registered candidate.
    fn confidence(&self, hllset: &HLLSet) -> f64;

    /// Register a token in the LUT (append-only, idempotent).
    ///
    /// Departure is a lattice operation, never a LUT deletion: once a token
    /// is registered it stays registered so that `M(D(t))` remains
    /// computable for departed bits.
    fn register(&mut self, token: Vec<u8>);

    /// Resolve a content address (murmur3 seed-0 hash) back to token bytes,
    /// if the token is registered in the LUT.
    fn resolve(&self, hash: u64) -> Option<Vec<u8>>;
}

// ── Tier 1: TokenLutStorage ────────────────────────────────────────────────

/// Tier-1 K-storage over the foundation `LutIndex` (single-seed, ordered
/// streams).
#[derive(Clone, Debug, Default)]
pub struct TokenLutStorage {
    lut: LutIndex,
}

impl TokenLutStorage {
    /// Create an empty storage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap an existing LUT.
    pub fn from_lut(lut: LutIndex) -> Self {
        Self { lut }
    }

    /// Build storage from tokens.
    pub fn from_tokens<I, B>(tokens: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut lut: LutIndex = LutIndex::default();
        for token in tokens {
            lut.insert_token(token.as_ref().to_vec());
        }
        Self { lut }
    }

    /// Register a token in the LUT (idempotent).
    pub fn insert(&mut self, token: Vec<u8>) {
        self.lut.insert_token(token);
    }

    /// Access the underlying LUT.
    pub fn lut(&self) -> &LutIndex {
        &self.lut
    }

    /// Consume and return the underlying LUT.
    pub fn into_lut(self) -> LutIndex {
        self.lut
    }

    /// Whether the token is registered in the LUT.
    pub fn registered(&self, token: &[u8]) -> bool {
        let (reg, zeros) = token_to_position(token);
        !self.lut.fiber(reg * BITS_PER_REG + zeros).is_empty()
    }
}

impl KStorage for TokenLutStorage {
    fn key_of(&self, token: &[u8]) -> KeyRef {
        let (reg, zeros) = token_to_position(token);
        KeyRef::Cell(reg * BITS_PER_REG + zeros)
    }

    fn candidates(&self, hllset: &HLLSet) -> Vec<Vec<u8>> {
        self.lut.materialize(hllset).into_iter().collect()
    }

    fn confidence(&self, hllset: &HLLSet) -> f64 {
        confidence_by_bit(hllset, |reg, zeros| {
            !self.lut.fiber(reg * BITS_PER_REG + zeros).is_empty()
        })
    }

    fn register(&mut self, token: Vec<u8>) {
        self.insert(token);
    }

    fn resolve(&self, hash: u64) -> Option<Vec<u8>> {
        let (reg, zeros) = hash_to_position(hash);
        self.lut
            .fiber(reg * BITS_PER_REG + zeros)
            .into_iter()
            .find(|t| murmur3_hash(t) == hash)
    }
}

// ── Tier 2: CatalogLutStorage ──────────────────────────────────────────────

/// Tier-2 K-storage: a local multi-seed catalog with quorum consensus
/// (unordered streams).
#[derive(Clone, Debug)]
pub struct CatalogLutStorage {
    values: Vec<Vec<u8>>,
    seeds: Vec<u64>,
}

impl Default for CatalogLutStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalogLutStorage {
    /// Create an empty storage with the default seeds `[0, 1, 2]`.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            seeds: DEFAULT_CATALOG_SEEDS.to_vec(),
        }
    }

    /// Create storage with custom seeds (at least 2 for consensus).
    pub fn with_seeds(seeds: &[u64]) -> Self {
        assert!(seeds.len() >= 2, "need at least 2 seeds for consensus");
        Self {
            values: Vec::new(),
            seeds: seeds.to_vec(),
        }
    }

    /// Build storage from values.
    pub fn from_values<I, B>(values: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut storage = Self::new();
        for value in values {
            storage.insert(value.as_ref().to_vec());
        }
        storage
    }

    /// Register a value in the LUT (idempotent).
    pub fn insert(&mut self, value: Vec<u8>) {
        if !self.values.contains(&value) {
            self.values.push(value);
        }
    }

    /// The seeds used by this storage.
    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    /// The registered values.
    pub fn values(&self) -> &[Vec<u8>] {
        &self.values
    }

    fn quorum(&self) -> usize {
        std::cmp::max(1, self.seeds.len() - 1)
    }

    fn seed0_candidates(&self, reg: u32, zeros: u32) -> Vec<Vec<u8>> {
        self.candidates_at(reg, zeros)
    }

    /// Candidates registered at `(reg, zeros)` under **any** seed.
    fn candidates_at(&self, reg: u32, zeros: u32) -> Vec<Vec<u8>> {
        self.values
            .iter()
            .filter(|v| {
                self.seeds.iter().any(|&seed| {
                    let (r, z) = hash_to_position(murmur3_hash_seeded(v, seed));
                    r == reg && z == zeros
                })
            })
            .cloned()
            .collect()
    }
}

impl KStorage for CatalogLutStorage {
    fn key_of(&self, token: &[u8]) -> KeyRef {
        let cells: Vec<u32> = self
            .seeds
            .iter()
            .map(|&seed| {
                let (reg, zeros) = token_to_position_seeded(token, seed);
                reg * BITS_PER_REG + zeros
            })
            .collect();
        KeyRef::Cells(cells, self.quorum())
    }

    fn candidates(&self, hllset: &HLLSet) -> Vec<Vec<u8>> {
        let quorum = self.quorum();
        let mut out = Vec::new();
        for value in &self.values {
            let hits = self
                .seeds
                .iter()
                .filter(|&&seed| {
                    let (reg, zeros) = token_to_position_seeded(value, seed);
                    hllset.bitmap().contains(reg * BITS_PER_REG + zeros)
                })
                .count();
            if hits >= quorum {
                out.push(value.clone());
            }
        }
        out
    }

    fn confidence(&self, hllset: &HLLSet) -> f64 {
        confidence_by_bit(hllset, |reg, zeros| {
            !self.seed0_candidates(reg, zeros).is_empty()
        })
    }

    fn register(&mut self, value: Vec<u8>) {
        self.insert(value);
    }

    fn resolve(&self, hash: u64) -> Option<Vec<u8>> {
        self.values
            .iter()
            .find(|v| murmur3_hash_seeded(v, 0) == hash)
            .cloned()
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Fraction of set bits in `hllset` for which `resolve(reg, zeros)` is true.
fn confidence_by_bit(hllset: &HLLSet, resolve: impl Fn(u32, u32) -> bool) -> f64 {
    let bits = hllset.bitmap();
    if bits.is_empty() {
        return 1.0;
    }
    let total = bits.len() as f64;
    let resolved = bits
        .iter()
        .filter(|pos| {
            let reg = pos / BITS_PER_REG;
            let zeros = pos % BITS_PER_REG;
            resolve(reg, zeros)
        })
        .count() as f64;
    resolved / total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_of_matches_hash_decomposition() {
        let storage = TokenLutStorage::from_tokens(&["alpha", "beta"]);
        let (reg, zeros) = token_to_position(b"alpha");
        assert_eq!(
            storage.key_of(b"alpha"),
            KeyRef::Cell(reg * BITS_PER_REG + zeros)
        );
    }

    #[test]
    fn key_ref_cells_and_quorum() {
        assert_eq!(KeyRef::Cell(7).cells(), vec![7]);
        assert_eq!(KeyRef::Cell(7).quorum(), None);
        let cells = KeyRef::Cells(vec![1, 2, 3], 2);
        assert_eq!(cells.cells(), vec![1, 2, 3]);
        assert_eq!(cells.quorum(), Some(2));
    }

    #[test]
    fn confidence_is_one_on_covered_set() {
        let storage = TokenLutStorage::from_tokens(&["alpha", "beta", "gamma"]);
        let h = HLLSet::from_tokens(&["alpha", "beta", "gamma"]);
        assert_eq!(storage.confidence(&h), 1.0);
    }

    #[test]
    fn confidence_drops_on_unknown_bits() {
        let storage = TokenLutStorage::from_tokens(&["alpha"]);
        let h = HLLSet::from_tokens(&["alpha", "definitely-not-registered"]);
        assert!(storage.confidence(&h) < 1.0);
    }

    #[test]
    fn catalog_key_is_multi_seed_with_quorum() {
        let storage = CatalogLutStorage::from_values(&["alice@example.com"]);
        match storage.key_of(b"alice@example.com") {
            KeyRef::Cells(cells, quorum) => {
                assert_eq!(cells.len(), 3);
                assert_eq!(quorum, 2);
            }
            other => panic!("expected Cells, got {other:?}"),
        }
    }
}
