# Architecture v2 — operational vs structural/persistence

> Status: **target architecture, agreed 2026-09-07** (code moves follow this
> document). Companion to `GOVERNANCE.md` §12 and
> `ALGEBRAIC_FOUNDATION.md`.

## 1. The split

> Operational duties belong to `hllset-next`, `hllset-fpga-simulator`, and
> `ewm-fpga-bridge` — the trio governed by algebra. Structural and persistence
> duties belong to `ewm-cortex-fpga`: presenting and persisting algebraic
> structures in memory and in a persistent store.

```text
OPERATIONAL (governed by algebra)            STRUCTURAL + PERSISTENCE (app)
─────────────────────────────────────        ─────────────────────────────────
hllset-next-v2   reference algebra           ewm-cortex-fpga-v2
hllset-fpga-     golden FPGA model              ├─ cortex-context  Context = S(t)
  simulator-v2                                  │                  + matrix + views
ewm-fpga-bridge- SPI / modules / DSL / wire     ├─ context-tree    S(t) Merkle tree
  v2                                            ├─ lut-view        V_j(H_i) records
                                                ├─ ewm-git         persistence
                                                └─ mirror          tensor presentation
```

The trio *executes*; the app *presents and persists*. The trio owns no
persistent state beyond what a module step needs; the app owns no operational
execution beyond invoking the trio.

## 2. What leaves the bridge

- **`ContextMatrix`** (`ewm-core::context`) and **grounding verdicts**
  (`ewm-core::grounding`) move to a new app crate **`cortex-context`**. They
  are nanoLM structural concepts, not wire vocabulary.
- `ewm-core` keeps the operational algebra: contracts re-exports, `TokenSet`/
  slice/InLUT, `iica`, `ContentKey`.

### GroundModule — decision: option A (executor stays, state moves)

- The bridge's `GroundModule` remains a **wire-level executor**: recent ids in
  → prior packet + τ/ρ report packet out. It keeps only operational memory
  (what a step needs), never the authoritative matrix.
- The app's `cortex-context` holds the authoritative grow-only CRDT matrix,
  feeds the module, persists its state, and owns policy (when to observe,
  when to commit, what to do with a report).

### Algebraic notes (ALGEBRAIC_FOUNDATION §1.2)

- **LUTs are nodes of `2^T`, not types.** `TokenLUT`, `CatalogLUT`, the
  1-gram/2-gram/3-gram token subsets (`G1/G2/G3`), and `K_i` are labeled
  subsets of the token universe. The bridge's LUT "kinds" become **labels on
  nodes**; `K_i` (the fiber of bit address `i`) is first-class — the InLUT is
  its table.
- **Slice and Gate are the same morphism** `M(H, L)` at different LUT nodes;
  the module split is operational (LUT residence, fixed-point), not
  algebraic. The module vocabulary should document this.
- **Merkle leaves are the atoms of `S(t)`** — the Boolean lattice `2^B`
  decomposes every sketch uniquely into active atoms `{i}` (equivalently the
  fibers `K_i`, each a node of `2^T`). `context-tree` becomes a sparse Merkle
  tree over `B`: leaves = active atoms in bit order, root = `S(t)`, tree diff
  = bit-level D/R/N. Each leaf pairs the atom with its fiber
  (`leaf_i = ({i}, K_i)`) — **the tree is the restoration rule of the `K_i`
  HLLSet**, the declared inverse of `ingest` at the finest granularity.
  Source attribution is a **label on atoms**; the source-grouped
  (working-set HLLSet) tree is a rollup view, not the definition.
- **The correspondence rule is mandatory** (ALGEBRAIC_FOUNDATION §1.3): every
  defined token collection has its HLLSet (`H_L = ingest(L)`); every HLLSet
  declares its restoration rule (`V(H) = M(H, L)`). Consequences for the app:
  every named LUT node carries its own sketch; every stored sketch persists
  its restoration label (G1/G2/G3 channels = "sketch + token-collection
  type"); anonymous sketches are rejected by the store.
- **Prefix typing + LUT isolation** (ALGEBRAIC_FOUNDATION §1.4): named LUT
  nodes are physically separate stores (`tokenLUT₁/₂/₃`, `catalogLUT`,
  `gateLUT`); HLLSet prefixes are types (`o:` original, `v:` views, `r/d/n:`
  Noether derivations); `h:` is the only untyped prefix, restored by
  materialization against a declared LUT node — **default = the join of all
  known nodes** (the full image). The store records the restoration choice
  with every `h:` sketch.
- **The finest symmetry** (ALGEBRAIC_FOUNDATION §1.5):
  `HLLSet(LUT) = HLLSet(MerkleTree)` — ingesting a LUT node is joining its
  atom tree; the LUT and the tree are the token-space and sketch-space
  presentations of the same lattice element.

## 3. The app's target shape

```text
ewm-cortex-fpga-v2/
├── crates/
│   ├── cortex-context   (NEW)   Context: S(t) tree + ContextMatrix + view set
│   ├── context-tree     (kept)  Merkle tree over HLLSets (leaf-level S(t))
│   ├── lut-view         (kept)  ViewRecord v2 (relational views)
│   ├── ewm-git          (kept)  persistence; commits gain the ctx-root reference
│   ├── cortex-core      (kept)  app pipeline semantics (first-class app code)
│   ├── hllset-attn      (kept)  K-storage; KeyRef over contracts BitAddress
│   ├── hllset-repro     (kept)  training harness (first-class app code)
│   └── cortex-fpga      (kept)  bridge-facing pipeline + crosscheck tests
└── deps (direct path, replacing vendored copies):
        hllset-next-v2/crates/hllset-core
        hllset-next-v2/crates/hllset-materialize
```

- **Vendored `hllset-core` and `hllset-materialize` are removed**; the app
  depends on the `hllset-next-v2` crates by path. Presenting an HLLSet
  requires the HLLSet type; owning a copy does not.
- `hllset-attn`, `cortex-core`, `hllset-repro` are **declared first-class app
  crates** (they are app code that travelled with the copy), no longer
  "vendored" in spirit.
- **One `Context` object** unifies the four islands: tree root + matrix +
  view set. `ewm-git` commits gain the context-tree root reference (closing
  LUT_VIEW §7.5 point 4); `LatticeState` remains the low-level commit payload,
  the `Context` is the app's view over it.

## 4. Reach-table refinement

Governance §5 gains one clause:

> **Apps may depend on the reference algebra library** (`hllset-core`,
> `hllset-materialize`) **for structural types and crosscheck tests, but
> never on operational backends** (bridge module executors, simulator
> drivers).

Rationale: using an HLLSet *type* to present or persist a sketch is not
executing an operation; the operational trio remains the only executor.

## 5. Migration stages

1. Create `cortex-context`; move `ContextMatrix` + grounding from `ewm-core`
   to the app (bridge loses the structural state).
2. Rework `GroundModule` to option A (wire executor with operational memory;
   app feeds/reads the authoritative matrix).
3. Replace vendored `hllset-core` / `hllset-materialize` with `hllset-next-v2`
   path deps.
4. Unify `Context` = tree + matrix + views; `ewm-git` commit pins the ctx
   root.
5. Group/rename app crates per §3; update README + docs.

Each stage lands green (workspace tests + crosscheck) before the next begins.

## 6. Open points

- Exact wire contract for feeding/reading the Ground matrix (config fields or
  a dedicated command pair).
- Whether `lut-view` v1 (`LutView` vector) survives or is folded into
  `ViewRecord` only.
- Final naming of the app crates (`cortex-context` vs extending `context-tree`).
