//! Relational views + context tree, wired to the bridge's golden InLUT model
//! (LUT_VIEW.md §7; notebook 17).
//!
//! - [`view_record`] — `V_j(H_i) = M(H_i, L_j)` with `(h, l)` provenance;
//! - [`full_image`] — `V(H_i) = ∪_j V_j(H_i)` over a LUT family;
//! - [`context_tree_for`] — the working set of HLLSets as a Merkle
//!   [`ContextTree`] with per-leaf views.
//!
//! The tree is the algebraic `S(t)` in the Noether equation
//! `H(t) = (S(t), H(t-1), D, R, N)`; the bit-level D/R/N are computed from
//! the unions of the tree's leaves, and the tests verify they agree with
//! `ewm-git`'s `CommitView`.

use context_tree::{ContextTree, Leaf};
use ewm_core::{parse_token_id, slice_positions_with, token_in_bytes, InLut, TokenId};
use hllset_core::HLLSet;
use hllset_core::core::content_addr::content_key_from_tokens;
use lut_view::ViewRecord;

/// The content key `h:<sha1>` of an HLLSet, from its source token ids.
pub fn h_key(ids: &[TokenId]) -> String {
    let tokens: Vec<Vec<u8>> = ids.iter().map(|&n| token_in_bytes(n)).collect();
    content_key_from_tokens(tokens.iter())
}

/// `V_j(H_i) = M(H_i, L_j)`: materialize an HLLSet through one LUT.
pub fn view_record(hset: &HLLSet, ids: &[TokenId], lut: &InLut, l: &str) -> ViewRecord {
    let positions = hset.active_positions();
    let recovered = slice_positions_with(&positions, lut, parse_token_id).ids;
    let tokens: Vec<Vec<u8>> = recovered.iter().map(|&n| token_in_bytes(n)).collect();
    ViewRecord::new(h_key(ids), l, tokens)
}

/// The full image `V(H_i) = ∪_j V_j(H_i)`: the union of per-LUT views.
pub fn full_image(hset: &HLLSet, ids: &[TokenId], luts: &[(&str, &InLut)]) -> ViewRecord {
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    for (l, lut) in luts {
        tokens.extend(view_record(hset, ids, lut, l).tokens);
    }
    ViewRecord::new(h_key(ids), "all", tokens)
}

/// A context-tree leaf for one HLLSet: its key plus all per-LUT view keys.
pub fn leaf_for(hset: &HLLSet, ids: &[TokenId], luts: &[(&str, &InLut)]) -> Leaf {
    let views: Vec<(String, String)> = luts
        .iter()
        .map(|(l, lut)| {
            let record = view_record(hset, ids, lut, l);
            (l.to_string(), record.digest)
        })
        .collect();
    Leaf {
        h: h_key(ids),
        views,
    }
}

/// Build the context tree for a working set of `(HLLSet, source ids, LUTs)`.
pub fn context_tree_for(
    items: &[(&HLLSet, &[TokenId], &[(&str, &InLut)])],
) -> ContextTree {
    ContextTree::build(
        items
            .iter()
            .map(|(hset, ids, luts)| leaf_for(hset, ids, luts))
            .collect(),
    )
}

/// Build the unified [`cortex_core::Context`] for a working set: the tree,
/// the grow-only follow matrix over each item's bigrams, and the full-image
/// views. This is the one structure the cortex owns (stage 4).
pub fn unified_context_for(
    items: &[(&HLLSet, &[TokenId], &[(&str, &InLut)])],
) -> cortex_core::Context {
    let tree = context_tree_for(items);

    let mut matrix = cortex_core::ContextMatrix::new();
    for (_, ids, _) in items {
        for w in ids.windows(2) {
            matrix.observe(w[0], w[1], 1);
        }
    }

    let views: Vec<lut_view::ViewRecord> = items
        .iter()
        .map(|(hset, ids, luts)| full_image(hset, ids, luts))
        .collect();

    cortex_core::Context::build(tree, matrix, views)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ewm_git::{view, LatticeState, MemoryStore, ObjectId, Repository};

    fn lut(range: std::ops::Range<u32>) -> InLut {
        let mut l = InLut::new();
        for n in range {
            l.insert(token_in_bytes(n));
        }
        l
    }

    fn hll(ids: &[TokenId]) -> HLLSet {
        HLLSet::from_tokens(ids.iter().map(|&n| token_in_bytes(n)))
    }

    const HA_IDS: &[TokenId] = &[0, 2, 5, 8];
    const HB_IDS: &[TokenId] = &[1, 4];
    const HC_IDS: &[TokenId] = &[3, 8, 12];

    #[test]
    fn relational_view_carries_provenance_and_full_image_is_a_union() {
        let lut_main = lut(0..8);
        let lut_extra = lut(3..10);
        let ha = hll(HA_IDS);

        let v_main = view_record(&ha, HA_IDS, &lut_main, "main");
        let v_extra = view_record(&ha, HA_IDS, &lut_extra, "extra");

        assert_eq!(v_main.tokens, vec![tid(0), tid(2), tid(5)]);
        assert_eq!(v_extra.tokens, vec![tid(5), tid(8)]);
        assert_eq!(v_main.cache_key(), (h_key(HA_IDS), "main".to_string()));

        let image = full_image(&ha, HA_IDS, &[("main", &lut_main), ("extra", &lut_extra)]);
        assert_eq!(image.tokens, vec![tid(0), tid(2), tid(5), tid(8)]);
        assert_eq!(image.cache_key(), (h_key(HA_IDS), "all".to_string()));
    }

    #[test]
    fn context_tree_is_reversible_and_diff_is_exact() {
        let lut_main = lut(0..8);
        let lut_extra = lut(3..10);
        let luts: &[(&str, &InLut)] = &[("main", &lut_main), ("extra", &lut_extra)];

        let ha = hll(HA_IDS);
        let hb = hll(HB_IDS);
        let hc = hll(HC_IDS);

        let prev = context_tree_for(&[(&ha, HA_IDS, luts), (&hb, HB_IDS, luts)]);
        let now = context_tree_for(&[(&ha, HA_IDS, luts), (&hc, HC_IDS, luts)]);

        // Reversibility: insert∘remove = id and remove∘insert = id.
        assert_eq!(
            prev.insert(leaf_for(&hc, HC_IDS, luts)).remove(&h_key(HC_IDS)),
            prev,
            "insert then remove = id"
        );
        assert_eq!(
            now.remove(&h_key(HC_IDS)).insert(leaf_for(&hc, HC_IDS, luts)),
            now,
            "remove then insert = id"
        );

        let diff = prev.diff(&now);
        assert_eq!(diff.added, vec![h_key(HC_IDS)]);
        assert_eq!(diff.removed, vec![h_key(HB_IDS)]);
        assert_eq!(diff.retained, vec![h_key(HA_IDS)]);
    }

    #[test]
    fn noether_equation_agrees_between_tree_math_and_ewm_git() {
        let ha = hll(HA_IDS);
        let hb = hll(HB_IDS);
        let hc = hll(HC_IDS);

        // S(t-1) = {H_a, H_b}; S(t) = {H_a, H_c}.
        let h_prev = HLLSet::union_all(vec![ha.clone(), hb.clone()]);
        let s_now = HLLSet::union_all(vec![ha.clone(), hc.clone()]);

        let departed = h_prev.difference(&s_now);
        let retained = h_prev.intersection(&s_now);
        let novel = s_now.difference(&h_prev);

        // Noether invariants.
        assert_eq!(departed.union(&retained).popcount(), h_prev.popcount());
        assert_eq!(retained.union(&novel).popcount(), s_now.popcount());
        assert_eq!(departed.intersection(&novel).popcount(), 0);

        // ewm-git commits the same states; its CommitView must agree.
        let mut repo: Repository<MemoryStore> = Repository::new(MemoryStore::default());
        let root: ObjectId = repo
            .commit(&LatticeState::single(&h_prev), &[], "S(t-1)")
            .unwrap();
        let tip: ObjectId = repo
            .commit(&LatticeState::single(&s_now), &[root.clone()], "S(t)")
            .unwrap();
        let cv = view(&repo, &tip).unwrap();

        assert_eq!(cv.departed.popcount(), departed.popcount());
        assert_eq!(cv.retained.popcount(), retained.popcount());
        assert_eq!(cv.new.popcount(), novel.popcount());
    }

    #[test]
    fn unified_context_combines_tree_matrix_and_views() {
        let lut_main = lut(0..8);
        let lut_extra = lut(3..10);
        let luts: &[(&str, &InLut)] = &[("main", &lut_main), ("extra", &lut_extra)];

        let ha = hll(HA_IDS);
        let hb = hll(HB_IDS);

        let ctx = unified_context_for(&[(&ha, HA_IDS, luts), (&hb, HB_IDS, luts)]);

        // Tree: one leaf per working-set item.
        assert_eq!(ctx.tree().leaves().len(), 2);
        // Matrix: the grow-only follow relation over each item's bigrams.
        assert!(ctx.matrix().grounded(HA_IDS[1]));
        assert!(ctx.matrix().grounded(HB_IDS[1]));
        // Views: the full image of each item, deduplicated on join.
        let tokens = ctx.full_image_tokens();
        for id in HA_IDS.iter().chain(HB_IDS.iter()) {
            assert!(tokens.contains(&tid(*id)), "missing token tid{id}");
        }
        // Joining the context with itself is the lattice idempotence.
        assert_eq!(ctx.join(&ctx).tree().root(), ctx.tree().root());
    }
}

fn tid(n: TokenId) -> Vec<u8> {
    token_in_bytes(n)
}
