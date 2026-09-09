# The Algebraic Foundation of EWM AI Systems

> Status: **synthesis of the discoveries** (2026-09-05). The algebra is the
> contract; implementations — Merkle trees, git stores, LUT views — are its
> concrete forms and can be re-derived from it.

## 1. The discipline: two spaces, two morphisms

```text
Token space                                HLLSet space
encoding ids (tid{n} / LE)                 32,768-bit sketches
        │                                        │
        │  ingest: tokens → HLLSet               │
        └────────────────────────────────────────┘
        │                                        │
        └──────── materialize: HLLSet → tokens ──┘
```

A token becomes structure only through **ingest**; structure becomes a token
only through **materialize**. Everything else is either pure token-space work
or pure HLLSet-space work. This single rule keeps every later layer sane.

### 1.1 The hinge: `BitAddress`, and where the QKV legs live

Both morphisms pass through one object — the **bit address**
`(reg, tz)` = `reg * 32 + tz`, a structural position inside the sketch:

```text
ingest:      token ──hash──▶ BitAddress ──set bit──▶ HLLSet
materialize: HLLSet ──bits──▶ BitAddress ──LUT lookup──▶ tokens
```

The QKV decomposition of attention maps onto the two spaces; **only K is
structural**, so only K descends:

| QKV leg | Our object | Space | Home |
| --- | --- | --- | --- |
| Q (query) | the HLLSet itself | structural | `hllset-core` |
| K (key) | `BitAddress` | structural hinge | **`hllset-contracts`** |
| V (value) | the TokenLUT reverse index | token (Structure B) | `hllset-materialize` / `hllset-attn` |

Rules that follow:

- `BitAddress` lives in the contracts leaf (flat `u32` storage, `reg()`/`tz()`
  accessors, bitmap `Ord`, serde as `u32`); it is re-exported by `hllset-core`
  and `ewm-core`, so the reference, the bridge wire, and the apps share one
  definition. `HLLSet::bit_addresses()` exposes it.
- The word **Key** is reserved for the QKV interpretation:
  `hllset-attn::KeyRef::Cell(BitAddress)` wraps the hinge; `E_bit[bit]`,
  `KStorage`, and the multi-seed `KeyRef::Cells` quorum stay app-layer.
- The seed is a parameter of the hash (`of_token_seeded(token, seed)`), never
  part of the address.

The attention mechanism is then, literally: `K = address(H)`,
`V = lut(bit)`, `Q = H` — the whole K-space sits on the hinge.

### 1.2 The LUT lattice

The LUT side has one algebra too, and it forbids ad-hoc partitioning.

```text
T    token universe            B = {0 … TOTAL_BITS−1}  bit addresses
pos: T → B                      the soldered position map
K_i = pos⁻¹(i)                  the fiber of address i — atoms of the
                                partition of T by pos
```

- The **general LUT is the subset lattice `2^T`**: every subset of tokens is
  a LUT. `TokenLUT`, `CatalogLUT`, and the 1-gram/2-gram/3-gram token
  subsets (`G1/G2/G3`) are not types — they are **named nodes** of this
  lattice; `K_i` are the fiber nodes. "7 LUT kinds" was a fixed, arbitrary
  sample of the lattice.
- **Materialization is a join in `2^T`**:

  ```text
  M(H, L) = ⋃ { L ∩ K_i : i ∈ A(H) }      A(H) = active addresses of H
  ```

  The InLUT table is literally the fiber decomposition `{L ∩ K_i}`.
- **`M(H, ·)` is a join-homomorphism**:

  ```text
  M(H, L₁ ∪ L₂) = M(H, L₁) ∪ M(H, L₂)
  ```

  Consequences: the full image is one materialization against the join —
  `V(H) = M(H, ⋃_j L_j)` — and **Slice and Gate are the same morphism**
  (`slice = M(H, L_lut)`, `gate = M(H, L_vocab)`); their separation is
  operational (where the LUT lives, fixed-point), not algebraic.

The same discipline applies to the two remaining structures:

- **ContextMatrix** is an element of the **tropical matrix lattice** over
  `T × T` (`⊕ = max`, `⊗ = +`): `C = ⋁ observe(a→b, w)` (a join of event
  observations), and the mirror is a lattice projection `C|_V = π_V(C)`.
  "Id buckets" are a storage trick, not structure.
- **Merkle leaves are the atoms of `S(t)`.** The HLLSet space is the Boolean
  lattice `2^B` (`B` = bit addresses), so every sketch decomposes *uniquely*
  into join-irreducibles — its active atoms `{i}`, `i ∈ A(H)`, equivalently
  the fibers `K_i` (each `K_i` is itself a node of the LUT lattice `2^T`).
  The context tree is therefore a **sparse Merkle tree over `B`**: leaves =
  active atoms, ordered by bit address; internal nodes = joins; root =
  `S(t)`; and tree diff = atom-level D/R/N, which coincides with the
  bit-level Noether decomposition. **The tree is the restoration rule of the
  `K_i` HLLSet**: each leaf pairs the atom with its fiber,
  `leaf_i = ({i}, K_i)` — the declared inverse of `ingest` at the finest
  granularity. Source attribution (which working-set HLLSet contributed a
  bit) is a **label on atoms**; the source-grouped tree is a rollup *view* —
  operational, not the definition.

### 1.3 The correspondence rule (definition ⇒ inscription, sketch ⇒ restoration)

The `{tokens} ↔ {HLLSets}` relation is closed by one bidirectional rule.

```text
C = the defined token collections: { L ⊆ T : L has an extraction rule }
H = the HLLSet lattice

Rule A (forward):   every L ∈ C has its HLLSet  H_L = ⋁_{t∈L} {pos(t)} = ingest(L)
Rule B (backward):  every H ∈ H carries a restoration rule — a declared
                    token-collection type (LUT node) L such that V(H) = M(H, L),
                    or a declared family of LUT nodes for the full image
```

Consequences:

- **Anonymous sketches are illegal.** Every HLLSet in the system is presented
  with its restoration pointer — the `(h, l)` provenance that `ViewRecord`
  already has becomes mandatory for every sketch, including persisted ones.
- **The two lattices mirror each other at the level of definitions.** Each
  named LUT node (TokenLUT, CatalogLUT, G1/G2/G3, gate vocab, doc vocab)
  has its own content-addressed HLLSet; `K_i` is the canonical finest case:
  its definition (`pos(t) = i`) inscribes to the atom `{i}`.
- **`ingest` and `materialize` become a declared pair.** Nothing on one side
  exists without its counterpart (or pointer to it) on the other.
- **Persistence implication.** The store never saves a bare sketch: commits
  persist each HLLSet together with its restoration label, so any stored
  sketch can be materialized later (G1/G2/G3 channels are exactly
  "sketch + its token-collection type").

### 1.4 Naming: prefix types and LUT isolation

- **Named LUT nodes are physically isolated collections.** `tokenLUT` is not
  one merged store but a family `{ tokenLUT₁ (1-gram), tokenLUT₂ (2-gram),
  tokenLUT₃ (3-gram), … }`; `catalogLUT`, `gateLUT`, and every other defined
  node is its own store. A node is identified by its extraction rule (its
  name), and the join of all physically present nodes is "the LUT" as a
  whole.
- **HLLSet prefixes are types.** `o:<sha1>` = original token collections
  (immutable source); `v:<sha1>` = views (materialization outputs,
  ephemeral); `r:`/`d:`/`n:` = the Noether derivations. Each prefix names
  what the sketch *is* — the existing prefix table
  (`o, h, r, d, n, t, v, l, c, u`) is the type system.
- **`h:<sha1>` is the only untyped prefix.** A heterogeneous sketch has no
  intrinsic token collection; its restoration rule is the materialization
  morphism against a **specific** LUT node.
- **Default restoration = all known nodes.** When no specific LUT is
  declared, `V(H) = M(H, ⋁ {named nodes})` — materialization runs against the
  join of every known node in the LUT lattice (the full image). Per-node
  materialization is the explicit, typed case.

Consequences:

- Every stored `h:` sketch records its restoration choice: a specific node
  label, or the default "all known".
- Named LUT nodes are themselves original collections, so they carry typed
  sketches (per the prefix table) and are content-addressed.
- The physical isolation of LUT nodes is what makes "all known nodes" a
  well-defined finite join — the LUT lattice is exactly the set of physically
  present stores.

### 1.5 The finest symmetry

```text
HLLSet(LUT) = HLLSet(MerkleTree)
```

Precisely: for any LUT node (or the join of all known nodes) `L`,

```text
ingest(L) = ⋁ { {i} : K_i ∩ L ≠ ∅ } = root of the atom tree of L
```

The two presentations — the token collection `L` and the join of its tree
leaves — are the **same lattice element**:

```text
        ingest
    L ────────────► HLLSet(L)
    │                  ▲
    │ fiber restrict.  │ join of atoms
    ▼                  │
{ K_i ∩ L }  ──────► { {i} }         leaf_i = ({i}, K_i ∩ L)
```

So the LUT and the Merkle tree are not two different things: they are the
token-space and sketch-space presentations of one object, and the morphisms
coincide — ingesting the LUT is joining the tree. The full cycle closes:
`LUT → ingest → sketch → tree → fibers → join → same sketch`.

## 2. HLLSet algebra: the IICA lattice

- An HLLSet is a fixed bitmap: `M = 1024` registers × `32` bits. Every token
  hashes (MurmurHash3) to exactly one bit.
- Operations: `A ∪ B`, `A ∩ B`, `A \ B`, `|A|`, `key(A) = sha1(A)`.
- **IICA** — Idempotent, Immutable, Content-Addressed — is preserved by
  composition. That is why pages → chapters → books → memory need no new
  theory: nested structures are just unions, and their keys are derived.
- **Sub-lattice collapse.** Any collection of HLLSets collapses to its join:
  one HLLSet = one `tokens_out` = one tensor slice.
- **Operational corollary — procedure scope.** Mutability is confined to the
  scope of a procedure (a processing module/step). *Inside* it, an HLLSet may
  be built up (`add_bit`, `add_token`, `merge` — the single-touch ingest pass
  is the canonical mutable scope). *Outside* it, an HLLSet is an immutable
  value, delivered with its SHA-1 content address. Processing takes HLLSets
  and returns **new** HLLSets with new keys; inputs are never mutated
  (`union`/`intersection`/`difference` already obey this).

## 3. Time as algebra: the Noether equation

```text
H(t) = (S(t), H(t-1), D, R, N)

N = S(t) \ H(t-1)      new
D = H(t-1) \ S(t)      departed
R = S(t) ∩ H(t-1)      retained

invariants:  D ∪ R = H(t-1),   R ∪ N = S(t),   D ∩ N = ∅
```

Time is a **derivation**, never stored. The temporal pyramid (L0..L6) is just
unions of these derivations; the commit content-address is the only timer.

## 4. Measurement and rank: TF stored, rank derived

- The TF vector is a monotonic CRDT (only increments).
- Rank is a **projection** of TF, never stored: five levels
  `F(TF) → G(bit) → H(register) → K(HLLSet) → L(compound)`.
- Materialization uses TF only as the **final tie-break** among candidates at
  a collided bit — the composed materializer (upstream-first work).

## 5. Grounding algebra (the nanoLM layer)

- Verdict: `τ = |A ∩ B| / |B|` coverage, `ρ = |B \ A| / |B|` novelty,
  `flagged = B \ A`. Grounding is a **recommendation**, never a mutation.
- The **exact-LUT gate** is the only real gate (0 leak, 0 FN); sketch
  consensus is diagnostics.
- `ContextMatrix` is a grow-only CRDT:
  `cell(a,b) = max(cell(a,b), w)`, `merge` = union nodes + pointwise max
  ⇒ idempotent / commutative / associative; it never regresses.
- The **prior** is the matrix restricted to `tokens_out` — a view over a
  monotone substrate.
- Experts compose by **union of gates + concatenation of logits**, and the
  softmax is applied **once** downstream (`renormalize_once`).

## 6. Context algebra (the cache → mirror loop)

```text
H(cache)   = ⋃ resident HLLSets            (sub-lattice collapse)
tokens_out = materialize(H(cache))         (the only crossing)
mirror     = ContextMatrix | tokens_out    (restricted view)
prior      = renormalize_once(mirror)      (partition over the FULL new set)
```

Ladder: golden full recompute → coalescing → D/N incremental materialization
→ refcounts (deferred). Even incremental, the softmax is **once over the whole
new `tokens_out`**, never over the diff — the partition function changed.

## 7. View algebra (the newest layer)

The LUT-view is the materialization morphism made **relational**:

```text
V_j(H_i) = M(H_i, L_j)          per-LUT materialization
V(H_i)    = ∪_j V_j(H_i)        full image over the LUT family (7 LUTs)
```

- The **cache identity is the pair `(h, l)`**; the `v:<sha1>` digest is a
  recomputable checksum.
- The **ContextTree** organizes the working set of HLLSets as a Merkle tree:
  - **exact collection identity**: the root changes iff the leaf set (or its
    views) changes — add/remove = tree edit → new root, which the union
    `H(cache)` alone can never do (a union cannot be un-ORed);
  - **reversible and persistent**: `insert∘remove = id`, `remove∘insert = id`;
    old roots stay valid forever;
  - **a lattice itself**: `merge`/`intersection`/`difference` over leaves,
    with `diff` giving leaf-level D/R/N.
- **S(t) is now algebraic.** The tree *is* `S(t)`; its union feeds the
  Noether equation, and `ewm-git` commits pin it:
  `H(t) = (S(t), H(t-1), D, R, N)` with `S(t) = ContextTree(t)`.

## 8. The unifying laws (the paramount structure)

1. **Content is identity.** Every object — HLLSet, view, tree, commit — is
   content-addressed; equality is digest equality; time never enters identity.
2. **Lattice closure.** Every layer is a lattice: HLLSets (∪/∩/\), token
   sets (∪/∩/\), ContextMatrix (pointwise max), ContextTree (merge/meet/
   difference). Composition stays inside the structure.
3. **Reversibility.** Every transformation is persistent: it produces a new
   object and leaves the old one valid. Nothing is mutated in place; nothing
   is lost.
4. **Morphism discipline.** Spaces cross only at `ingest` and `materialize`;
   views are the relational form of the latter.
5. **Time is a derivation.** D/R/N are computed from two states; they are
   never stored as primary objects.
6. **Views are function values.** `V_j(H_i) = M(H_i, L_j)`; they are keyed by
   their arguments, disposable, and recomputable on a miss.
7. **The correspondence rule.** Every defined token collection has its
   HLLSet (`H_L = ingest(L)`); every HLLSet declares its restoration rule
   (a LUT node `L`, or a family, such that `V(H) = M(H, L)`). Anonymous
   sketches are illegal — `(h, l)` provenance is mandatory, including in the
   persistent store.
8. **The life cycle is a closed algebra:**

```text
ingest → HLLSet lattice → views V_j → ContextTree S(t)
       → grounding (τ, ρ, flagged) → prior (restricted ContextMatrix)
       → Noether (D, R, N) → commit (ewm-git) → next cycle
```

## 9. Where each law lives (code map)

| Law | Crate / module | Proof |
| --- | --- | --- |
| IICA + hashing | `hllset-core` (vendored), `ewm-core::hashing` | bit-exact crosschecks |
| BitAddress hinge | `hllset-contracts` (re-exported by `hllset-core`, `ewm-core`) | range invariants, bitmap order, collision pin |
| LUT lattice + finest symmetry | `hllset-lut` (new, reference) | fiber partition, `M(H,·)` join-homomorphism, `HLLSet(LUT) = HLLSet(MerkleTree)` |
| Sub-lattice collapse | `ewm-core::slice`, `SliceModule` | notebook-08 fixtures |
| Noether equation | `ewm-git::view`, `cortex-fpga::context` | invariants asserted; agrees with `CommitView` |
| TF stored / rank derived | `hllset-core::tfvec`, `cortex-core::TfLut` | monotonicity tests |
| Grounding CRDT | `ewm-core::context`, `GroundModule` | CRDT laws, τ/ρ fixtures |
| Renormalize once | `ewm-core::slice::renormalized_join`, `ExpertsModule`+`RenormModule` | end-to-end law test |
| Relational view | `lut-view::ViewRecord` | `(h,l)` identity, canonical set semantics |
| Context tree / S(t) | `context-tree`, `cortex-fpga::context` | reversibility, lattice laws, leaf D/R/N |
| Commit pins S(t) | `ewm-git` (link) | D/R/N agreement test |
