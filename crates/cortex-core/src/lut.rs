//! TF-LUT — the monotonic term-frequency reverse index.
//!
//! The reference `TokenLUT` stores `encoding_id → hash_position (+ TF)`;
//! materialization disambiguates collision groups by term frequency. Here:
//!
//! - `register` interns an encoding ID in the append-only LUT;
//! - `observe` accumulates per-ID term frequency from ingested inputs
//!   (monotonic, **ungated** — the LUT is never filtered);
//! - `materialize` resolves an HLLSet through the LUT and ranks the
//!   candidates by TF (ties broken by byte order).

use std::collections::HashMap;

use hllset_attn::{KStorage, TokenLutStorage};
use hllset_core::HLLSet;

/// Monotonic TF reverse index over encoding IDs.
#[derive(Clone, Debug, Default)]
pub struct TfLut {
    storage: TokenLutStorage,
    tf: HashMap<Vec<u8>, u64>,
}

impl TfLut {
    /// Create an empty TF-LUT.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an encoding ID (append-only, idempotent).
    pub fn register(&mut self, id: Vec<u8>) {
        self.storage.insert(id.clone());
        self.tf.entry(id).or_insert(0);
    }

    /// Accumulate term frequency for a batch of ingested IDs.
    pub fn observe(&mut self, ids: &[Vec<u8>]) {
        for id in ids {
            self.register(id.clone());
            *self.tf.get_mut(id).expect("registered above") += 1;
        }
    }

    /// Term frequency of an encoding ID (0 if never observed).
    pub fn tf(&self, id: &[u8]) -> u64 {
        self.tf.get(id).copied().unwrap_or(0)
    }

    /// Whether the ID is registered in the forward map (exact LUT gate).
    pub fn known(&self, id: &[u8]) -> bool {
        self.tf.contains_key(id)
    }

    /// Materialize an HLLSet through the LUT.
    ///
    /// The primary filter is the token collection per bit: for each set
    /// bit, the LUT yields its candidate collection. If exactly one token
    /// is a candidate, it is chosen. **Only when a hash collision leaves
    /// more than one token for a bit** does TF select the winner (ties:
    /// byte order). The output is in bit order, not TF order.
    pub fn materialize(&self, hllset: &HLLSet) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for addr in hllset.bit_addresses() {
            let candidates: Vec<Vec<u8>> = self.storage.lut().fiber(addr.bit()).into_iter().collect();
            match candidates.as_slice() {
                [] => {}
                [single] => out.push(single.clone()),
                many => {
                    // Collision group: TF disambiguates.
                    let best = many.iter().min_by(|a, b| {
                        self.tf(b)
                            .cmp(&self.tf(a))
                            .then_with(|| a.cmp(b))
                    });
                    if let Some(best) = best {
                        out.push(best.clone());
                    }
                }
            }
        }
        out
    }

    /// Coverage gauge over `hllset` (1.0 iff every bit resolves).
    pub fn confidence(&self, hllset: &HLLSet) -> f64 {
        self.storage.confidence(hllset)
    }

    /// Number of registered encoding IDs.
    pub fn len(&self) -> usize {
        self.tf.len()
    }

    /// Whether the LUT is empty.
    pub fn is_empty(&self) -> bool {
        self.tf.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hllset_core::core::hashing::token_to_position;
    use std::collections::HashMap;

    /// Brute-force a pair of distinct tokens that hash to the same bit.
    fn collision_pair() -> (Vec<u8>, Vec<u8>) {
        let mut seen: HashMap<(u32, u32), Vec<u8>> = HashMap::new();
        for i in 0..200_000u32 {
            let token = format!("tok{i}").into_bytes();
            let pos = token_to_position(&token);
            if let Some(first) = seen.get(&pos) {
                return (first.clone(), token);
            }
            seen.insert(pos, token);
        }
        panic!("no collision found");
    }

    #[test]
    fn single_candidate_bits_resolve_without_tf() {
        let mut lut = TfLut::new();
        lut.observe(&[b"alpha".to_vec(), b"beta".to_vec(), b"gamma".to_vec()]);
        let doc = HLLSet::from_tokens(&[b"alpha".as_slice(), b"beta".as_slice()]);
        let restored = lut.materialize(&doc);
        // Both resolve (barring an astronomically unlikely collision here).
        assert!(restored.contains(&b"alpha".to_vec()));
        assert!(restored.contains(&b"beta".to_vec()));
        assert!(!restored.contains(&b"gamma".to_vec()));
    }

    #[test]
    fn tf_only_breaks_collision_ties() {
        let (a, b) = collision_pair();
        let mut lut = TfLut::new();
        // a is observed three times, b once — both collide on one bit.
        lut.observe(&[a.clone(), a.clone(), a.clone(), b.clone()]);
        let doc = HLLSet::from_tokens([a.as_slice()]);
        let restored = lut.materialize(&doc);
        // The bit resolves to exactly one token: the higher-TF one.
        assert_eq!(restored, vec![a.clone()]);
        assert!(!restored.contains(&b));
        assert_eq!(lut.tf(&a), 3);
        assert_eq!(lut.tf(&b), 1);
    }

    #[test]
    fn register_is_idempotent() {
        let mut lut = TfLut::new();
        lut.register(b"x".to_vec());
        lut.register(b"x".to_vec());
        assert_eq!(lut.len(), 1);
        assert!(lut.known(b"x"));
    }
}
