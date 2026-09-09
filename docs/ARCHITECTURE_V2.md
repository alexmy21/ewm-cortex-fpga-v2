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

## Session status (2026-09-09) — stages executed

- Stage 1 (cortex-context): `ContextMatrix` + grounding moved from the bridge
  into `cortex-core`; the bridge's `GroundModule` is the config-driven wire
  executor (Option A).
- Stage 2: `SimModuleDriver` landed in `ewm-sim`; `cortex-fpga` submits
  `ModuleCommand`s and drains `ModuleResponse`s — the cortex never steps a
  module directly.
- Stage 3 (de-vendor): `crates/hllset-core` + `crates/hllset-materialize`
  deleted; all consumers point at `hllset-next-v2` (core/contracts/lut/
  morphisms/ranks/cid). `ewm-git` gained a local `ingest` module on
  `hllset-morphisms`; `hllset-attn::kstorage` rewritten over `hllset-lut`.
- Stage 4 (unified Context): `cortex_core::Context` = context-tree +
  `ContextMatrix` + `lut-view` views; immutable combinators (`with_*`,
  `observe_sequence`, `join` = lattice join, `diff` = tree-level D/R/N,
  `full_image_tokens`). `cortex-fpga::unified_context_for` builds it from a
  working set. Roles landed (`Role`, `CortexConfig`, `Leader`).
- Stage 5 (app crate grouping/naming): next.

## Session status (2026-09-09, ewm-app integration session)

NEXT_SESSION.md §5 decisions — **accepted as recommended**:

- **Q1 — LLM order:** deterministic scripted token stream first, then the
  real local DeepSeek coder (`ollama run deepseek-coder:6.7b`). Both run
  green through `ewm-app`.
- **Q2 — where ewm-app lives:** a new binary crate inside
  `ewm-cortex-fpga-v2` (`crates/ewm-app`), with a lib (driver + token
  sources) and a CLI. A separate project can be split out at the app-level
  migration stage.
- **Q3 — side-car protocol:** async fire-and-forget from the LLM's point of
  view; the app polls/advances the commit head. A synchronous barrier exists
  only in the test harness (proven by
  `side_car_async_llm_never_blocks`).
- **Q4 — persist frequency:** every turn; a pass that brings no new bits is
  skipped (same content = same key ⇒ idempotent), so there is no duplicate
  commit.
- **Q5 — leader routing:** route-to-all-experts + lattice-join merge, as
  implemented by `cortex_core::Leader`. TF-weighted routing stays a derived,
  later increment.

What landed:

- `cortex-fpga` gained `BridgeExecutor` (one `SimModuleDriver` serving many
  submissions; tenant isolation = a fresh `ModuleGraphSpec` per submission),
  `run_ground_passthrough` (the cortex-declared prior + τ/ρ report through
  the bridge's config-driven `GroundModule`, verbatim), and `LutFamily` (the
  known-LUT family wrapper that keeps `InLut` out of app crates).
- `crates/ewm-app`: the Noether-loop driver (`CortexApp`). Each turn runs
  `ingest → S(t) → evolve → persist → advance head`; `S(t)` is the
  cumulative working set, committed so that `view(repo, head)` and the
  context tree agree (the `cortex-fpga` Noether-agreement pattern). Recovery
  is `CortexApp::open` — read the head, dereference the snapshot, rebuild
  the presentation from commit messages; no replay.
- Integration tests (all green, in NEXT_SESSION §4 order): smoke loop,
  recovery (no replay, idempotent), side-car async, multi-instance
  (1 leader + 2 experts + 1 shared bridge), grounding pass-through,
  zero-copy (two threads share one store; only CIDs cross), plus explicit
  dual-encoding and stub-replay tests.
- Baselines at handoff: `hllset-next-v2` 245, `hllset-fpga-simulator-v2`
  101, `ewm-fpga-bridge-v2` 59, `ewm-cortex-fpga-v2` **158** (149 + 9 new
  ewm-app tests).

Open points carried forward:

- The wire `ModuleCommand` has no explicit tenant tag yet; multi-tenant
  isolation is by per-submission `Configure` (the bridge is stateless
  between commands). Adding the tag is a wire change
  (`PROTOCOL_VERSION` bump).
- Skipped (no-change) turns are not part of the durable pointer; after
  recovery their token collections are not rebuilt (they are fully covered
  by committed sketches — a presentation-only limitation).
- `ewm-git` commits do not yet pin the context-tree root (stage 4's
  remaining point); the app rebuilds the tree from commit messages.
