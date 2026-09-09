//! The unified cortex context — the one structure the cortex owns.
//!
//! Per `ARCHITECTURE_V2` stage 4 and the separation contract: the cortex
//! holds exactly three structures, together:
//!
//! - **tree** — the working set `S(t)` as a sparse Merkle tree over the
//!   active atoms (leaves = content-addressed HLLSets);
//! - **matrix** — the grow-only tropical follow relation over grounded
//!   tokens ([`ContextMatrix`]);
//! - **views** — the full image `V(H) = ∪_j V_j(H)` with `(h, l)` provenance
//!   ([`lut_view::ViewRecord`]).
//!
//! `Context` is the stack item of the machine (separation contract §11):
//! the Noether loop produces a new `Context` per step, and `join` is the
//! lattice join of configurations. Everything is immutable; the combinators
//! return new values.

use crate::context::ContextMatrix;
use crate::TokenId;
use context_tree::{ContextTree, TreeDiff};
use lut_view::ViewRecord;

/// The unified context: tree + matrix + views.
#[derive(Clone, Debug)]
pub struct Context {
    tree: ContextTree,
    matrix: ContextMatrix,
    views: Vec<ViewRecord>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            tree: ContextTree::empty(),
            matrix: ContextMatrix::new(),
            views: Vec::new(),
        }
    }

    pub fn build(tree: ContextTree, matrix: ContextMatrix, views: Vec<ViewRecord>) -> Self {
        Self {
            tree,
            matrix,
            views,
        }
    }

    pub fn tree(&self) -> &ContextTree {
        &self.tree
    }

    pub fn matrix(&self) -> &ContextMatrix {
        &self.matrix
    }

    pub fn views(&self) -> &[ViewRecord] {
        &self.views
    }

    // ── Immutable combinators (the stack discipline) ─────────────────────

    pub fn with_tree(&self, tree: ContextTree) -> Self {
        Self {
            tree,
            matrix: self.matrix.clone(),
            views: self.views.clone(),
        }
    }

    pub fn with_matrix(&self, matrix: ContextMatrix) -> Self {
        Self {
            tree: self.tree.clone(),
            matrix,
            views: self.views.clone(),
        }
    }

    pub fn with_views(&self, views: Vec<ViewRecord>) -> Self {
        Self {
            tree: self.tree.clone(),
            matrix: self.matrix.clone(),
            views,
        }
    }

    /// Learn one sequence into the matrix (grow-only bigram observations).
    pub fn observe_sequence(&self, tokens: &[TokenId]) -> Self {
        let mut matrix = self.matrix.clone();
        for w in tokens.windows(2) {
            matrix.observe(w[0], w[1], 1);
        }
        self.with_matrix(matrix)
    }

    /// The lattice join of two configurations: tree merge, pointwise-max
    /// matrix merge, and the deduplicated union of views.
    pub fn join(&self, other: &Self) -> Self {
        let mut views = self.views.clone();
        for view in &other.views {
            if !views.contains(view) {
                views.push(view.clone());
            }
        }
        Self {
            tree: self.tree.merge(&other.tree),
            matrix: self.matrix.merge(&other.matrix),
            views,
        }
    }

    /// Tree-level Noether diff: the atoms added / removed / retained when
    /// moving from `self` to `next`.
    pub fn diff(&self, next: &Self) -> TreeDiff {
        self.tree.diff(&next.tree)
    }

    /// The full image tokens: the union of every view's tokens, sorted and
    /// deduplicated.
    pub fn full_image_tokens(&self) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = self.views.iter().flat_map(|v| v.tokens.clone()).collect();
        out.sort();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_tree::Leaf;
    use lut_view::ViewRecord;

    fn leaf(h: &str) -> Leaf {
        Leaf {
            h: h.to_string(),
            views: Vec::new(),
        }
    }

    fn view(h: &str, l: &str, tokens: &[&str]) -> ViewRecord {
        ViewRecord::new(
            h,
            l,
            tokens.iter().map(|t| t.as_bytes().to_vec()).collect(),
        )
    }

    fn context(hs: &[&str], seq: &[TokenId], views: Vec<ViewRecord>) -> Context {
        Context::build(
            ContextTree::build(hs.iter().map(|h| leaf(h)).collect()),
            ContextMatrix::from_sequence(seq),
            views,
        )
    }

    #[test]
    fn build_and_accessors() {
        let ctx = context(
            &["h:aa", "h:bb"],
            &[1, 2, 3],
            vec![view("h:aa", "main", &["tid1", "tid2"])],
        );
        assert_eq!(ctx.tree().leaves().len(), 2);
        assert_eq!(ctx.matrix().len(), 3);
        assert_eq!(ctx.views().len(), 1);
        assert_eq!(
            ctx.full_image_tokens(),
            vec![b"tid1".to_vec(), b"tid2".to_vec()]
        );
    }

    #[test]
    fn observe_sequence_grows_the_matrix_only() {
        let ctx = context(&["h:aa"], &[1, 2], Vec::new());
        let grown = ctx.observe_sequence(&[2, 3, 4]);
        assert!(grown.matrix().grounded(3));
        assert!(!ctx.matrix().grounded(3), "original is unchanged");
        assert_eq!(grown.tree().root(), ctx.tree().root());
    }

    #[test]
    fn join_is_the_lattice_join_of_configurations() {
        let a = context(
            &["h:aa", "h:bb"],
            &[1, 2],
            vec![view("h:aa", "main", &["tid1"])],
        );
        let b = context(
            &["h:bb", "h:cc"],
            &[2, 3],
            vec![view("h:cc", "main", &["tid3"])],
        );

        let joined = a.join(&b);
        assert_eq!(joined.tree().leaves().len(), 3, "tree merge");
        assert!(joined.matrix().grounded(1) && joined.matrix().grounded(3));
        assert_eq!(joined.views().len(), 2, "views union, deduplicated");
        assert_eq!(
            joined.full_image_tokens(),
            vec![b"tid1".to_vec(), b"tid3".to_vec()]
        );

        // Join is commutative and idempotent (lattice laws). Matrix equality
        // is checked semantically: same node set, same cell weights.
        let m1 = joined.matrix();
        let other = b.join(&a);
        let m2 = other.matrix();
        let mut n1 = m1.nodes().to_vec();
        n1.sort_unstable();
        let mut n2 = m2.nodes().to_vec();
        n2.sort_unstable();
        assert_eq!(n1, n2, "same grounded nodes");
        for &x in &n1 {
            for &y in &n1 {
                assert_eq!(m1.cell(x, y), m2.cell(x, y), "cell ({x},{y})");
            }
        }
        assert_eq!(joined.tree().root(), b.join(&a).tree().root());
        assert_eq!(a.join(&a).tree().root(), a.tree().root());
    }

    #[test]
    fn diff_is_the_tree_level_noether_decomposition() {
        let prev = context(&["h:aa", "h:bb"], &[1, 2], Vec::new());
        let next = context(&["h:aa", "h:cc"], &[2, 3], Vec::new());

        let diff = prev.diff(&next);
        assert_eq!(diff.added, vec!["h:cc".to_string()]);
        assert_eq!(diff.removed, vec!["h:bb".to_string()]);
        assert_eq!(diff.retained, vec!["h:aa".to_string()]);
    }
}
