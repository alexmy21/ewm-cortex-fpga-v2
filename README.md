# ewm-cortex-fpga

The **bridge test copy** of `ewm-cortex`: the DeepSeek-OCR black-box pipeline
re-expressed over `ewm-fpga-bridge` (milestone **M4-fpga**). The original
`ewm-cortex` stays self-contained and unchanged; this workspace consumes the
bridge by path and crosschecks bit-exactly against the first-party reference
crates it inherits.

> **v0.3.0 — bridge consumer (intentional break of self-containment).**
> Unlike `ewm-cortex` v0.2.0, this workspace **does** depend on
> `ewm-fpga-bridge` by path (`ewm-core`, `ewm-dsl`, `ewm-hostif`,
> `ewm-modules`). The vendored `hllset-core`/`hllset-materialize` and
> `cortex-core` are kept as the **golden reference** for bit-exact
> crosschecks; `ewm-git` stays at the app layer (bridge is store-agnostic).

See [`docs/CORTEX_ARCHITECTURE.md`](docs/CORTEX_ARCHITECTURE.md) for the
architecture and milestone plan (M4-fpga below).

## M4-fpga — bridge pipeline (this workspace's first milestone)

```text
doc ids (tid{n}) ──► Slice (materialize, tid InLUT) ──► Gate (gate_TF) ──► restored
```

- [`crates/cortex-fpga`](crates/cortex-fpga) declares the pipeline with
  `ewm-dsl`, lowers it to `ewm-hostif` wire config, and runs it over the
  bridge's **golden module registry** (the `SimModuleDriver` substitutes in
  when bridge Phase E lands — the declaration does not change).
- Crosschecks (all bit-exact, run with `cargo test -p cortex-fpga`):
  1. hashing vs vendored `hllset-core`;
  2. InLUT recovery vs vendored `materialize_inlut`;
  3. gate semantics vs `cortex_core::Gate`;
  4. the pinned collision divergence (bridge returns the candidate group; the
     vendored `TfLut` picks the TF winner — upstream-first work, see the
     bridge review 2026-09-04 §1.4).

## Pipeline (reference port + enhancement)

```text
DeepSeek-OCR Encoder                     DeepSeek-OCR Decoder
      │ encoding IDs (tid{n})                 ▲ restored IDs
      ▼                                       │
╔════════════════════════════════════════════════════════════════╗
║                  ewm-cortex-fpga (Rust workspace)              ║
║  cortex-core   : tokens → hash → tokenLUT → HLLSet →           ║
║                  : materialize → gate_TF → restored → decoder  ║
║  cortex-fpga   : the same black box as an ewm-fpga-bridge DSL  ║
║  cortex-context: MoE/ETT → EL → Resolution A+B → F(t)          ║
╚════════════════════════════════════════════════════════════════╝
```

## Layout

```text
ewm-cortex-fpga/
├── Cargo.toml                # workspace root (version 0.3.0)
├── corpus/
│   └── conversation.txt      # corpus for e2e training/testing
├── crates/
│   ├── hllset-core/          # vendored algebra: HLLSet, hashing, ops, TFVec
│   ├── hllset-materialize/   # TokenLUT / CatalogLUT / consensus + engine trait
│   ├── hllset-attn/          # K-storage + MoE/ETT: KStorage, TokenLutStorage,
│   │                         # BitKeyTable, KBridge, setkey, hybrid, moe,
│   │                         # ContextVocabulary, TokenMask
│   ├── ewm-git/              # 2005-style Git evolution store (replaces temporal pyramid)
│   ├── cortex-core/          # black-box pipeline (encoding, gate, TF-LUT, pipeline)
│   ├── cortex-fpga/          # the same black box as an ewm-fpga-bridge DSL
│   ├── lut-view/             # LUT-view: v1 vector cache + v2 relational
│   │                         # ViewRecord (h, l) with SHA-1 identity
│   ├── context-tree/         # Merkle tree over HLLSets + per-leaf views —
│   │                         # the algebraic S(t) of the Noether equation
│   └── hllset-repro/         # token realm: hand-rolled autograd, char-level
│                             # transformer, Phase 0-3 harnesses
└── docs/
    ├── PROJECT_STRUCTURE.md                       # crate graph + ownership rules
    ├── CORTEX_ARCHITECTURE.md                     # enhanced hllset-cortex architecture
    ├── ALGEBRAIC_FOUNDATION.md                    # the algebraic structure (paramount)
    ├── LUT_VIEW.md                                # LUT-view design discussion + v1/v2 contract
    ├── HLLSET_K_SPACE_MATH.md                     # theory (partition, adjunction, MoE/ETT §8)
    ├── HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md     # implementation roadmap (Phases 0-4)
    └── notebooks/                                 # removed from this project: the gen2
                                                   # notebook set lives in
                                                   # hllset-next-v2/_DOCS/notebooks/ (01-05,
                                                   # executed green). This project is synced
                                                   # with the updated foundation in a later
                                                   # phase of the collection roadmap.
```

## Quick start

```bash
# M1 — cortex pipeline (tokens → hash → tokenLUT → HLLSet → materialize → gate_TF → restored)
cargo run -p cortex-core                     # uses corpus/conversation.txt
cargo run -p cortex-core -- path/to/text.txt

# M4-fpga — the cortex pipeline over ewm-fpga-bridge (golden modules)
cargo test -p cortex-fpga                    # bit-exact crosscheck + golden run

# G1 — content-addressed evolution store (commit DAG, H(t) view, merge, gc)
cargo run -p ewm-git

# POC harnesses
cargo run -p hllset-repro --bin phase1       # K-storage attach stats
cargo run -p hllset-repro --bin phase2       # address-key vs baseline
cargo run -p hllset-repro --bin phase3       # MoE/ETT + hybrid gate

# Tests
cargo test --workspace
```

## Status

### **Cortex milestones**

- [x] M1 — reference pipeline port (`cortex-core`: hash → tokenLUT → HLLSet → materialize → gate_TF)
- [x] E1 — evolution store (`ewm-git`: commit DAG, H(t) view, merge, GC)
- [x] E2 — archive-before-prune (`gc_to`: pruned branches stay addressable)
- [ ] E3 — IPFS archive adapter (`hllset-storage::IpfrsNativeStorage`)
- [ ] M2 — MoE/ETT integration (`cortex-context`, candidates from the commit DAG)
- [ ] M3 — grounding/search port
- [x] M4-fpga (step 1/2) — `ewm-fpga-bridge` module DSL: bit-exact crosschecks
      + golden `Slice → Gate` pipeline in `cortex-fpga` (sim backend swaps in
      at bridge Phase E)
- [x] LUT-view v1 (`lut-view` crate) — content-addressed vector of token
      hashes, SHA-1 identity, refresh-iff-changed; upgrade path documented
      (canonical set → Merkle proofs → per-LUT provenance)
- [x] LUT-view v2 (Design v2, notebook 17 promoted to crates) — relational
      `ViewRecord` with `(h, l)` provenance; `context-tree` Merkle tree over
      HLLSets (persistent, reversible, lattice ops); `cortex-fpga::context`
      wiring with Noether/`ewm-git` agreement tests
- [ ] M5 — PyO3 bindings for DeepSeek-OCR integration

### **Attention/K-space POC**

- [x] Phase 0 — reproduction baseline (hand-rolled f32 autograd + transformer)
- [x] Phase 1 — K-storage attach (coverage invariant, collision stats vs 1/3072)
- [x] Phase 2 — address-key attention (`k_t = E_bit[bit(t)]` via `KBridge`; gap +1.2%)
- [x] Phase 3 — MoE/ETT resolution + hybrid gate (`hllset-attn::moe`, `bin/phase3`)
- [ ] Phase 4 — multi-seed exactness, backends, persistence
