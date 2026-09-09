//! The redesigned LLM context: a grow-only relation matrix over grounded
//! tokens (nanoLM ARCHITECTURE §4 — ported to the bridge, wire-native).
//!
//! `M[a][b]` = weight of the measured transition `a -> b`.
//!
//! - **nodes** — grounded tokens (`TokenId`), append-only;
//! - **sparsity pattern** — the measured `a -> b` transition graph
//!   (De Bruijn bigrams);
//! - **cells** — grow-only registers (`cell = max(cell, w)`), `i64` so the
//!   matrix maps 1:1 onto wire `Packet.values` (Q6: integer streams).
//!
//! `merge` is pointwise max ⇒ idempotent / commutative / associative ⇒ a
//! monotonic CRDT that never regresses.

use crate::TokenId;
use std::collections::{HashMap, VecDeque};

/// The redesigned LLM context (nanoLM §4).
#[derive(Debug, PartialEq, Eq)]
pub struct ContextMatrix {
    nodes: Vec<TokenId>,
    index: HashMap<TokenId, usize>,
    /// `n*n` row-major; `cells[i * n + j]` = weight of `nodes[i] -> nodes[j]`.
    cells: Vec<i64>,
}

impl ContextMatrix {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            index: HashMap::new(),
            cells: Vec::new(),
        }
    }

    /// Build `M` from a token sequence: every adjacent pair `a -> b` is
    /// observed with its bigram count as the cell weight (grow-only `max`).
    pub fn from_sequence(tokens: &[TokenId]) -> Self {
        let mut counts: HashMap<(TokenId, TokenId), i64> = HashMap::new();
        for w in tokens.windows(2) {
            *counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
        let mut m = Self::new();
        for ((a, b), w) in counts {
            m.observe(a, b, w);
        }
        m
    }

    /// The grounded token set, in insertion order.
    pub fn nodes(&self) -> &[TokenId] {
        &self.nodes
    }

    /// Number of grounded nodes (the context is `len × len`).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Append-only: record a measured edge `a -> b` with evidence `w`.
    ///
    /// Nodes are added on first sight and never removed; the cell is a
    /// grow-only register (`cell = max(cell, w)`).
    pub fn observe(&mut self, a: TokenId, b: TokenId, w: i64) {
        let ia = self.ensure(a);
        let ib = self.ensure(b);
        let n = self.nodes.len();
        let cell = self.cells[ia * n + ib].max(w);
        self.cells[ia * n + ib] = cell;
    }

    /// The weight of the measured transition `a -> b`, or `None` if either
    /// node is absent from this context.
    pub fn cell(&self, a: TokenId, b: TokenId) -> Option<i64> {
        let ia = *self.index.get(&a)?;
        let ib = *self.index.get(&b)?;
        Some(self.cells[ia * self.nodes.len() + ib])
    }

    /// CRDT merge: union nodes + pointwise-max cells.
    pub fn merge(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for &t in &other.nodes {
            out.ensure(t);
        }
        let n = out.nodes.len();
        for (ia, &a) in out.nodes.iter().enumerate() {
            for (ib, &b) in out.nodes.iter().enumerate() {
                let va = self.cell(a, b).unwrap_or(0);
                let vb = other.cell(a, b).unwrap_or(0);
                out.cells[ia * n + ib] = va.max(vb);
            }
        }
        out
    }

    /// The measured transition graph (sparsity pattern): all `a -> b` with a
    /// non-zero cell weight.
    pub fn edges(&self) -> Vec<(TokenId, TokenId)> {
        let n = self.nodes.len();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if self.cells[i * n + j] > 0 {
                    edges.push((self.nodes[i], self.nodes[j]));
                }
            }
        }
        edges
    }

    /// Is `t` grounded *in this context*?
    pub fn grounded(&self, t: TokenId) -> bool {
        self.index.contains_key(&t)
    }

    /// A sequence as a path through `M`: the shortest measured
    /// `start -> end` path, or empty if unreachable / ungrounded.
    pub fn path(&self, start: TokenId, end: TokenId) -> Vec<TokenId> {
        if !self.grounded(start) {
            return Vec::new();
        }
        if start == end {
            return vec![start];
        }
        let n = self.nodes.len();
        let si = self.index[&start];
        let ei = self.index[&end];

        let mut prev = vec![usize::MAX; n];
        let mut seen = vec![false; n];
        let mut queue = VecDeque::new();
        queue.push_back(si);
        seen[si] = true;
        while let Some(u) = queue.pop_front() {
            if u == ei {
                break;
            }
            for v in 0..n {
                if !seen[v] && self.cells[u * n + v] > 0 {
                    seen[v] = true;
                    prev[v] = u;
                    queue.push_back(v);
                }
            }
        }
        if !seen[ei] {
            return Vec::new();
        }
        let mut path = Vec::new();
        let mut cur = ei;
        loop {
            path.push(self.nodes[cur]);
            if cur == si {
                break;
            }
            cur = prev[cur];
        }
        path.reverse();
        path
    }

    /// The token-level prior projected onto the LLM's attention: grounded
    /// nodes (sorted for wire determinism) + per-node bias = total outgoing
    /// evidence (sum of that node's row).
    pub fn prior(&self) -> (Vec<TokenId>, Vec<i64>) {
        let mut nodes = self.nodes.clone();
        nodes.sort_unstable();
        let n = self.nodes.len();
        let bias = nodes
            .iter()
            .map(|&t| {
                let i = self.index[&t];
                (0..n).map(|j| self.cells[i * n + j]).sum()
            })
            .collect();
        (nodes, bias)
    }

    /// Ensure `t` is a node, returning its dense index; grows the matrix so
    /// existing `(row, col) -> weight` associations stay in place.
    fn ensure(&mut self, t: TokenId) -> usize {
        if let Some(&i) = self.index.get(&t) {
            return i;
        }
        let old_n = self.nodes.len();
        let i = old_n;
        self.nodes.push(t);
        self.index.insert(t, i);

        let new_n = old_n + 1;
        let mut new_cells = vec![0i64; new_n * new_n];
        for (idx, &v) in self.cells.iter().enumerate() {
            let (r, c) = (idx / old_n, idx % old_n);
            new_cells[r * new_n + c] = v;
        }
        self.cells = new_cells;
        i
    }
}

impl Default for ContextMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ContextMatrix {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            index: self.index.clone(),
            cells: self.cells.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m1() -> ContextMatrix {
        let mut m = ContextMatrix::new();
        m.observe(1, 2, 1);
        m.observe(2, 3, 1);
        m
    }

    fn m2() -> ContextMatrix {
        let mut m = ContextMatrix::new();
        m.observe(2, 3, 1);
        m.observe(3, 4, 1);
        m
    }

    #[test]
    fn observe_appends_nodes_and_sets_cells() {
        let m = m1();
        assert_eq!(m.nodes(), &[1, 2, 3]);
        assert!(m.grounded(2));
        assert!(!m.grounded(99));
        assert_eq!(m.cell(1, 2), Some(1));
        assert_eq!(m.cell(1, 3), Some(0), "both nodes, transition unobserved");
        assert_eq!(m.cell(99, 1), None);
    }

    #[test]
    fn observe_is_monotone() {
        let mut m = ContextMatrix::new();
        m.observe(1, 2, 5);
        m.observe(1, 2, 20);
        assert_eq!(m.cell(1, 2), Some(20));
        m.observe(1, 2, 10);
        assert_eq!(m.cell(1, 2), Some(20), "never regresses");
    }

    #[test]
    fn merge_is_a_crdt() {
        // The CRDT laws hold up to order-independent canonical form (merge
        // unions nodes; node insertion order differs between the two sides).
        fn canonical(m: &ContextMatrix) -> (Vec<TokenId>, Vec<(TokenId, TokenId, i64)>) {
            let mut nodes = m.nodes().to_vec();
            nodes.sort_unstable();
            let mut edges: Vec<(TokenId, TokenId, i64)> = m
                .edges()
                .into_iter()
                .map(|(a, b)| (a, b, m.cell(a, b).unwrap()))
                .collect();
            edges.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
            (nodes, edges)
        }

        let a = m1();
        let b = m2();
        let c = ContextMatrix::from_sequence(&[5, 6, 7]);

        assert_eq!(canonical(&a.merge(&a)), canonical(&a), "idempotent");
        assert_eq!(
            canonical(&a.merge(&b)),
            canonical(&b.merge(&a)),
            "commutative"
        );
        let left = a.merge(&b).merge(&c);
        let right = a.merge(&b.merge(&c));
        assert_eq!(canonical(&left), canonical(&right), "associative");
    }

    #[test]
    fn merge_never_regresses_a_cell() {
        let (a, b) = (m1(), m2());
        let merged = a.merge(&b);
        for x in [1, 2, 3] {
            for y in [1, 2, 3] {
                let v = merged.cell(x, y).unwrap_or(0);
                assert!(v >= a.cell(x, y).unwrap_or(0));
                assert!(v >= b.cell(x, y).unwrap_or(0));
            }
        }
    }

    #[test]
    fn from_sequence_records_bigram_tf() {
        let m = ContextMatrix::from_sequence(&[1, 2, 1, 2, 3]);
        assert_eq!(m.cell(1, 2), Some(2));
        assert_eq!(m.cell(2, 1), Some(1));
        assert_eq!(m.cell(2, 3), Some(1));
    }

    #[test]
    fn path_finds_measured_transitions() {
        let m = ContextMatrix::from_sequence(&[1, 2, 3, 4]);
        assert_eq!(m.path(1, 4), vec![1, 2, 3, 4]);
        assert_eq!(m.path(2, 4), vec![2, 3, 4]);
        assert!(m.path(4, 1).is_empty(), "no reverse edge");
        assert!(m.path(99, 1).is_empty(), "ungrounded start");
    }

    #[test]
    fn prior_is_sorted_and_biased_by_outgoing_evidence() {
        let mut m = ContextMatrix::new();
        m.observe(9, 1, 2);
        m.observe(9, 2, 3);
        m.observe(1, 2, 4);
        let (nodes, bias) = m.prior();
        assert_eq!(nodes, vec![1, 2, 9]);
        // node 9: 2+3 = 5; node 1: 4; node 2: 0
        assert_eq!(bias, vec![4, 0, 5]);
    }
}
