# LUT-view — design discussion and v1 contract

> Status: **recorded design discussion** (2026-09-04). The reasoning matters as
> much as the outcome; do not re-derive it from scratch.

## 1. Origin

While reviewing where `ewm-cortex-fpga` should go after M4-fpga step 1/2, we
discussed the **LUT-view**: a structure derived from the active context that
lets us keep the *working vocabulary* in memory instead of touching the whole
LUT.

The five motivations, in the proposer's own words:

1. The LUT would grow (append-only, monotone TF).
2. The tokens (hashes) in the LUT are almost never all used at the same time.
3. Smaller memory → faster access.
4. Materialization already works well and brings all working/active tokens.
5. We can keep them in memory (cache).

The Merkle tree was mentioned as **an instinct, not a conclusion** — and was
explicitly dropped for v1. This document records both why it was dropped and
when it should return.

## 2. What the discussion settled

- **The LUT-view is a first-class, SHA-1-addressable structure that is used
  as a cache today** — not merely a cache, and not yet a Merkle tree.
- **v1 shape:** a *vector* of token hashes (`sha1(token_bytes)`),
  identified by `digest = sha1(leaves concatenated)`, keyed `v:<40-hex>`.
- **Update rule:** recompute on context change; replace the view **iff the
  digest changed** (`refresh` returns `Option`).
- **Ephemeral and transitional:** the view is derived and disposable; dropping
  it loses only a recomputation, never data. The LUT it mirrors stays whole
  and append-only — we make **no changes** to how LUTs are used.
- **The view belongs in the app layer** (`ewm-cortex-fpga`), not in
  `ewm-fpga-bridge`: the bridge emits `tokens_out` (`Slice` output / `Ground`
  prior); the view is the managed, hash-addressable version of that stream.

## 3. The design space (important distinctions, kept for later)

### 3.1 Three legitimate views, three leaf identities

| View | Leaves | Meaning | Purpose |
| --- | --- | --- | --- |
| **Vocabulary view** | `sha1(token)` | the active vocabulary slice | membership, versioning, sync — v1 is this |
| **Coverage view** | `murmur3(token)` (LUT position hash) | active positions the LUT resolves | coverage/confidence reporting |
| **Sequence view** | `sha1(token)` as an ordered vector | materialized token order | order recovery, De Bruijn partner |

Consequence: murmur3 leaves are **not wrong** — they are the coverage view,
not the vocabulary view. v1 ships the vocabulary view; the other two are
future types, not flags.

### 3.2 Set vs vector

- A **set** (sorted, deduplicated) ⇒ `root = f(content) only` ⇒ identity is
  order-independent. This is the future canonical form.
- A **vector** ⇒ order is part of the identity. v1 pins vector semantics
  (materialization order); the canonical-set form is the first upgrade.

### 3.3 The view is a function of the *resolved* stream

`tokens_out = materialize(H(cache))` is strategy-dependent: today's InLUT
returns all candidates for a collided bit; the future composed materializer
(upstream-first, see the bridge review 2026-09-04 §1.4) returns the TF winner.
So:

> The view is defined over the **resolved token stream**, not over raw HLLSet
> bits. On collided bits the v1 view is a superset and will narrow when the
> composed materializer lands.

### 3.4 The 7 LUTs map to view kinds, not seven identical views

| LUT | View kind | Feed to |
| --- | --- | --- |
| main token LUT ("original HLLSets") | vocabulary view | membership, versioning, gate |
| 1-gram LUT | vocabulary view (unigram provenance) | `tokens_out`, prior |
| 2-gram / 3-gram LUTs | transition views (NUL-joined n-gram hashes) | `ContextMatrix` edges |
| seed-0/1/2 catalog LUTs | catalog views (3 views of the same values) | consensus materialization |

Per-LUT views are separate `LutView` instances with a provenance tag — later.

## 4. Why not Merkle now — and when it returns

Reasons it was dropped for v1:

- The v1 view is **ephemeral and transitional**: it lives in one process's
  memory and is rebuilt on context change. No proofs, no persistence, no
  cross-tier sync, no version history are needed.
- A Merkle tree buys O(log n) membership proofs, subtree diffs, and syncable
  snapshots. None of those are the cache scenario.

Triggers that bring the Merkle tree back (as a deliberate feature, not an
instinct):

1. Views must cross process/tier boundaries (e.g. FPGA ↔ host ↔ disk).
2. Subtree-level diff/merge or membership proofs are needed.
3. View roots are pinned in `ewm-git` commit objects (vocabulary version per
   commit — the strongest long-term payoff).
4. Incremental refresh: reuse unchanged subtrees instead of full recompute
   (the Session 4.3 ladder: golden recompute → coalescing → incremental).

When it returns, the v1 flat digest remains a valid root of the flat tree —
no breaking change.

## 5. v1 contract (as implemented in `crates/lut-view`)

```rust
pub type Hash = [u8; 20];        // sha1(token bytes)
pub type ViewDigest = [u8; 20];  // sha1(leaves concatenated, in order)

pub struct LutView { /* hashes: Vec<Hash>, digest: ViewDigest */ }

impl LutView {
    pub fn from_hashes(iter) -> Self;
    pub fn from_tokens(iter_of_bytes) -> Self;
    pub fn refresh(&self, hashes: &[Hash]) -> Option<Self>; // Some only if changed
    pub fn hashes(&self) -> &[Hash];
    pub fn digest(&self) -> &ViewDigest;
    pub fn key(&self) -> String;          // "v:<40-hex>"
    pub fn len / is_empty / contains;
}
```

Wired into `cortex-fpga`: `view_from_tokens(&[TokenId])` builds the view from
a pass's materialized ids; the golden test asserts a stable corpus produces
zero view replacements and a new token produces exactly one.

## 6. Points to never forget

1. **LUTs stay whole and append-only.** The view never mutates the LUT.
2. **The view is the active vocabulary slice in hash form** — a memory cache,
   not a new persistent store.
3. **Replace iff digest changed** — content-addressing gives idempotence for
   free; no churn when the context is stable.
4. **v1 = vector semantics.** Order is part of the identity; the set form is
   an upgrade, and it changes what the digest means.
5. **The view is disposable.** Evict it freely; one materialization rebuilds
   it.
6. **The bridge stays view-agnostic.** It emits `tokens_out`; LUT-view is an
   app-layer structure.
7. **Merkle is a future feature with explicit triggers** (§4), not a default.

## 7. Design v2 — the relational view and the HLLSet tree (recorded after review)

Review of the v1 implementation produced two correct critiques and one new
idea. This section records the resulting design; **v1 code stays as-is until
the prototype is approved.**

### 7.1 The two critiques of v1

1. **v1 mirrors HLLSet.** It is a content-addressed container
   (`digest + payload`) — but a view is not a primary object; it is a
   *derived function value*.
2. **v1 carries no LUT reference.** A view without `(H_i, L_j)` is an
   anonymous result: it cannot be recomputed on a miss, verified, or
   meaningfully compared ("same digest" = same output bytes, not same function
   at the same point).

### 7.2 The relational model

```text
V_j(H_i) = M(H_i, L_j)      # per-LUT materialization — the morphism at a point
V(H_i)   = ∪_j V_j(H_i)     # full image of H_i over the LUT family
```

Consequences:

- The **cache key is the pair `(h_i, l_j)`**, not the output digest. `M` is
  deterministic, so the pair *is* the identity; the output digest is a
  recomputable checksum. A view record is
  `{ h: ContentKey, l: LutKey, tokens, digest }` — recompute on miss, drop
  freely, replace iff digest changed.
- The **full image is a CRDT union**: `V(H_i) = ∪_j V_j(H_i)` over the 7 LUTs
  (main, unigram, bigram, trigram, seed-0/1/2) is token-set union —
  idempotent/commutative/associative via `TokenSet`.
- The bridge already emits `V_j` for one configured LUT (`Slice`); the full
  image is the app-layer union of slice outputs across LUTs. **Bridge
  unchanged.**
- **Materialization is not invertible**, so views are keyed by `(H, L)` only,
  never reverse-indexed by output alone (two-space rule preserved).

### 7.3 The operational idea: views hang off their HLLSets, HLLSets form a tree

```text
context root (sha1)
 ├── internal node = sha1(left ‖ right) + aggregate view digests
 ├── leaf H_a (h:<sha1>)
 │      ├── (main,  v:<sha1>)   ← V_main(H_a)
 │      ├── (uni,   v:<sha1>)
 │      └── (seed0, v:<sha1>) …
 └── leaf H_b …
```

Why this is the right place for Merkle (unlike the dropped per-view tree):

- **Exact collection identity.** `H(cache) = ∪ HLLSets` is a cheap structural
  summary but cannot be un-ORed (Session 4's subtraction problem). A Merkle
  tree over the HLLSets is an exact set of digests: add/remove a leaf = tree
  edit → new root. **This supports eviction correctly**, which the union never
  could.
- **Subtree aggregates.** Internal nodes carry the digest of their subtree's
  full image, so `V(subtree)` is provable and syncable without holding the
  whole thing.
- **Context diff = tree diff.** Two roots differ in exactly the leaves that
  entered/left — structural D/R/N at the HLLSet level, for free.
- **`ewm-git` is the natural host.** A Merkle tree of HLLSets is Git's tree
  object. Decision: the operational implementation **reuses `ewm-git`**
  (HLLSet = blob, view = derived blob keyed `(h, l)`, context = tree, commit
  references the root) rather than building a parallel tree. Commits then pin
  both the structural state **and** the vocabulary views.
- The **replace-iff-changed rule moves up one level**: the context root
  changes iff the leaf set changes; views change iff leaves or LUTs change.
  One digest comparison at the top.

### 7.4 What the code became (implemented 2026-09-05)

- `lut-view` gained the relational **`ViewRecord`** (`{ h, l, tokens, digest }`):
  canonical token-set semantics, `(h, l)` as the cache identity, `v:<sha1>` as
  the output checksum.
- New **`context-tree`** crate: a persistent Merkle tree over HLLSet leaves
  (sorted by `h:<sha1>`, deduplicated), each leaf carrying its per-LUT view
  keys. Operations are **reversible** (insert∘remove = id) and form a
  **lattice** (`merge`/`intersection`/`difference`, `diff` = leaf-level D/R/N).
- `cortex-fpga::context` wires the bridge's golden InLUT to the model:
  `view_record` (`V_j(H_i)`), `full_image` (`∪_j V_j(H_i)`),
  `context_tree_for` (the algebraic `S(t)`), with tests asserting the Noether
  equation agrees with `ewm-git`'s `CommitView`.

### 7.5 Settled / remaining open points

Settled by the implementation:

1. Internal nodes store **digests only** (full levels kept; union aggregates
   deferred).
2. Canonical leaf order: **sorted by `h:<sha1>`**, deduplicated by `h`.
3. Per-LUT views live as **leaf payload** (`(lut, v:<sha1>)` pairs, sorted),
   hashed into the leaf — not a side map.

Still open:

4. How `ContextTree` roots are referenced from `ewm-git` commits (the root is
   recomputable from the committed states today; storing it in the commit
   object is the next step).
5. Subtree aggregates (the full union set at chosen levels) — digests only for
   now.

**Prototype recorded** in the now-discharged notebook 17 (the legacy
`ewm-cortex-fpga/docs/notebooks/17_context_tree_prototype.ipynb`; gen2
discharged all legacy notebooks on 2026-09-07 and will author a new set
aligned with the gen2 architecture): relational views with `(h, l)`
provenance, full image as CRDT union, a Merkle `ContextTree` over HLLSets
with leaf-level views, the algebraic `H(t) = (S(t), H(t-1), D, R, N)`
(invariants asserted), and the `ewm-git` commit link (D/R/N agree between the
tree math and `CommitView`).

## 8. Related documents

- [`ALGEBRAIC_FOUNDATION.md`](ALGEBRAIC_FOUNDATION.md) — the algebraic
  structure this design instantiates (the paramount synthesis).
- Bridge `docs/DECISIONS.md` Session 5.2 — two-structure rule, store-agnostic
  bridge, `ewm-cortex-fpga` boundary.
- Bridge `docs/DECISIONS.md` Session 6.1 — dual-encoding `SliceModule` +
  `ewm-cortex-fpga` creation.
- Bridge `docs/REVIEW_2026-09-04.md` §1.4 — the composed materializer gap
  (the view's resolved-stream semantics depend on it).
- This repo: `crates/lut-view/src/lib.rs` — v1 implementation + upgrade path.
