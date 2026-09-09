//! Crosscheck: the bridge's soldered core vs the vendored first-party crates
//! of `ewm-cortex` (Session 5.2, risk #1 — bit-exactness before M4).
//!
//! These tests pin, at the crate boundary:
//!
//! 1. hashing — MurmurHash3 + `hash_to_position` are bit-identical;
//! 2. InLUT recovery — bridge `InLut` matches vendored `materialize_inlut`;
//! 3. gate semantics — bridge `GateModule` matches `cortex_core::Gate`;
//! 4. the known collision divergence — the bridge returns the full candidate
//!    group, the vendored `TfLut` picks the TF winner (review 2026-09-04
//!    §1.4: the composed TF-ranked materializer is upstream work).

use ewm_core::{
    parse_token_id, slice_positions_with, token_in_bytes, token_to_position, InLut, TokenId,
    BITS_PER_REG, M, P,
};
use ewm_hostif::Packet;
use ewm_modules::{GateModule, Module, Stream};

/// Collision-free fixture (excludes the pinned 262/48300 pair).
const FIXTURE_IDS: &[TokenId] = &[1169, 412, 22117, 995, 2746, 9384, 27140, 44, 16326];

fn tid(id: TokenId) -> Vec<u8> {
    token_in_bytes(id).to_vec()
}

fn parse_tid(token: &[u8]) -> TokenId {
    let s = String::from_utf8_lossy(token);
    s.trim_start_matches("tid").parse().expect("tid{n}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_matches_vendored_core_bit_exactly() {
        use hllset_core::core::hashing as vendored;

        // Soldered geometry is identical (the vendored crate re-exports the
        // constants at its root, not from the hashing module; their integer
        // widths differ, so compare as u64).
        assert_eq!(P as u64, hllset_core::P as u64);
        assert_eq!(M as u64, hllset_core::M as u64);
        assert_eq!(BITS_PER_REG as u64, hllset_core::BITS_PER_REG as u64);

        let samples: Vec<Vec<u8>> = ["tid0", "tid671", "tid18308", "hello", "tid262", "tid48300"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        for sample in &samples {
            assert_eq!(
                ewm_core::murmur3_hash(sample),
                vendored::murmur3_hash(sample),
                "murmur3 diverges for {sample:?}"
            );
            assert_eq!(
                ewm_core::murmur3_hash_seeded(sample, 1),
                vendored::murmur3_hash_seeded(sample, 1),
                "seeded murmur3 diverges for {sample:?}"
            );
            assert_eq!(
                token_to_position(sample),
                vendored::token_to_position(sample),
                "position diverges for {sample:?}"
            );
        }
    }

    #[test]
    fn inlut_recovery_matches_foundation_materialize() {
        use hllset_morphisms::{materialize, Ingest};

        let tokens: Vec<Vec<u8>> = FIXTURE_IDS.iter().map(|&id| tid(id)).collect();

        // Foundation: complete ingest (seed 0 path) + LUT-first materialization.
        let mut ingest: Ingest = Ingest::new();
        ingest.ingest_tokens(tokens.iter().map(|t| t.as_slice()));
        let sketch = ingest.hllset(0).clone();
        let foundation_out = materialize(&[(&sketch, ingest.lut(0))], ingest.tf());

        // Bridge golden model: InLUT keyed with the same tid bytes.
        let mut bridge_lut = InLut::new();
        for &id in FIXTURE_IDS {
            bridge_lut.insert(tid(id));
        }
        let positions: Vec<(u32, u32)> = FIXTURE_IDS
            .iter()
            .map(|&id| token_to_position(&tid(id)))
            .collect();
        let bridge_out = slice_positions_with(&positions, &bridge_lut, parse_token_id).ids;

        let mut foundation_ids: Vec<TokenId> =
            foundation_out.iter().map(|t| parse_tid(t)).collect();
        foundation_ids.sort_unstable();
        let mut bridge_ids = bridge_out.clone();
        bridge_ids.sort_unstable();

        assert_eq!(bridge_ids, foundation_ids, "InLUT recovery diverges");
        assert_eq!(bridge_ids.len(), FIXTURE_IDS.len(), "collision-free fixture");
    }

    #[test]
    fn gate_module_matches_cortex_core_gate() {
        use cortex_core::Gate;

        let vocab_ids: Vec<TokenId> = (0..8).collect();
        let vocab: Vec<Vec<u8>> = vocab_ids.iter().map(|&n| tid(n)).collect();
        let reference = Gate::from_vocab(vocab.iter());

        let mut module = GateModule::new(0);
        module
            .load_config(&[(
                "lut_ids".to_string(),
                "0,1,2,3,4,5,6,7".to_string(),
            )])
            .expect("valid config");

        let input_ids = [0u32, 2, 9, 4];
        let input_tokens: Vec<Vec<u8>> = input_ids.iter().map(|&n| tid(n)).collect();

        let reference_kept = reference.filter_tokens(&input_tokens);
        let mut reference_kept: Vec<TokenId> =
            reference_kept.iter().map(|t| parse_tid(t)).collect();
        reference_kept.sort_unstable();

        let mut inputs = vec![Stream {
            data: Packet {
                ids: input_ids.to_vec(),
                values: Vec::new(),
            },
            valid: true,
            ready: false,
        }];
        let out = module.step(&mut inputs).expect("no trap");
        let mut bridge_kept = out[0].data.ids.clone();
        bridge_kept.sort_unstable();

        assert_eq!(bridge_kept, reference_kept, "gate semantics diverges");
        assert!(bridge_kept.contains(&0) && !bridge_kept.contains(&9));
    }

    #[test]
    fn collision_divergence_is_pinned_and_one_sided() {
        use cortex_core::lut::TfLut;

        // The pinned collision pair: 262 and 48300 share bit (759, 0) under
        // the LE inscription; under tid they may differ, so re-derive a
        // collision pair by brute force to stay encoding-agnostic.
        let (a, b) = collision_pair_tid();
        assert_ne!(a, b);

        // Vendored TfLut: a observed 3x, b 1x → a wins the tie.
        let mut lut = TfLut::new();
        lut.observe(&[tid(a), tid(a), tid(a), tid(b)]);
        let doc = hllset_core::HLLSet::from_tokens([tid(a)]);
        let vendored_winner = lut.materialize(&doc);
        assert_eq!(vendored_winner.len(), 1);
        let winner = parse_tid(&vendored_winner[0]);

        // Bridge InLUT: the full candidate group for the shared bit.
        let mut bridge_lut = InLut::new();
        bridge_lut.insert(tid(a));
        bridge_lut.insert(tid(b));
        let positions: Vec<(u32, u32)> = vec![token_to_position(&tid(a))];
        let bridge_out = slice_positions_with(&positions, &bridge_lut, parse_token_id).ids;

        // Known divergence: the bridge returns both candidates (review §1.4);
        // the vendored TfLut picks the TF winner. One-sided: the winner is
        // always present in the bridge's candidate group.
        assert!(
            bridge_out.len() >= 2,
            "expected a real collision group, got {:?}",
            bridge_out
        );
        assert!(bridge_out.contains(&a));
        assert!(bridge_out.contains(&b));
        assert!(bridge_out.contains(&winner));
    }

    /// Brute-force a tid-encoded collision pair.
    fn collision_pair_tid() -> (TokenId, TokenId) {
        use std::collections::HashMap;
        let mut seen: HashMap<(u32, u32), TokenId> = HashMap::new();
        for i in 0..200_000u32 {
            let pos = token_to_position(&tid(i));
            if let Some(first) = seen.get(&pos) {
                return (*first, i);
            }
            seen.insert(pos, i);
        }
        panic!("no collision found in 200k ids");
    }
}
