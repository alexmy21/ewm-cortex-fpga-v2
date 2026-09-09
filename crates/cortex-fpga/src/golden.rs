//! Golden executor: the cortex black-box as an `ewm-dsl` pipeline over the
//! bridge's golden module registry.
//!
//! ```text
//! doc ids ──► Slice (materialize, tid-encoded InLUT) ──► Gate (gate_TF) ──► restored
//! ```
//!
//! Cortex rule preserved: **the LUT is never gated**. Novel ids in a document
//! are registered into the slice LUT (a host-side `Configure` before `Step`,
//! exactly as the reference `TfLut.observe` does); only the output passes
//! through `gate_TF`. Out-of-vocab ids are reported as leaks, never hidden.

use ewm_core::{
    token_in_bytes, token_to_position, ModuleKind, NodeId, PortId, TokenId, BITS_PER_REG,
};
use ewm_dsl::{NodeKind, Pipeline, PipelineBuilder};
use ewm_hostif::{
    ModuleCommand, ModuleDriver, ModuleEdge, ModuleGraphSpec, ModuleNodeSpec, ModuleResponse,
};
use ewm_modules::{validate_graph, Packet};

/// Comma-separated id list (wire config encoding).
pub fn csv(ids: &[TokenId]) -> String {
    ids.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The cortex pipeline declaration (the only place the app touches the DSL).
///
/// - node 0 `Slice`  — `materialize`: positions → token ids (tid-encoded LUT);
/// - node 1 `Gate`   — `gate_TF`: output-only vocabulary gate.
pub fn cortex_pipeline() -> Pipeline {
    PipelineBuilder::new()
        .node(0, NodeKind::Slice)
        .label(0, "materialize")
        .node(1, NodeKind::Gate)
        .label(1, "gate_TF")
        .connect((0, 0), (1, 0))
        .build()
        .expect("cortex pipeline declaration is valid")
}

/// Result of one golden pipeline pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FpgaPipelineResult {
    /// Ids materialized from the full (ungated) HLLSet.
    pub materialized_ids: Vec<TokenId>,
    /// Materialized ids that survive `gate_TF`.
    pub restored_ids: Vec<TokenId>,
    /// Materialized ids outside the decoder vocabulary (reported, never hidden).
    pub leaks: Vec<TokenId>,
}

impl FpgaPipelineResult {
    pub fn ok(&self) -> bool {
        self.leaks.is_empty()
    }
}

/// One bridge serving many submissions (multi-instance topology, separation
/// contract §2). The executor holds a single [`ewm_sim::SimModuleDriver`];
/// every submission carries its own `ModuleGraphSpec`, so tenants (experts)
/// are isolated by configuration, not by per-tenant state. The bridge keeps
/// no expert state between commands.
pub struct BridgeExecutor {
    driver: ewm_sim::SimModuleDriver,
}

impl Default for BridgeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeExecutor {
    pub fn new() -> Self {
        Self {
            driver: ewm_sim::SimModuleDriver::new(),
        }
    }

    /// Run the cortex pipeline over the golden modules.
    ///
    /// - `doc_ids`: one document of encoding ids (the `tid{n}` stream);
    /// - `lut_ids`: the measured LUT ids (registered before this pass);
    /// - `vocab_ids`: the decoder vocabulary (`gate_TF`).
    ///
    /// The slice LUT is built from `lut_ids ∪ doc_ids` — the ungated-LUT rule.
    pub fn run_cortex_pipeline(
        &mut self,
        doc_ids: &[TokenId],
        lut_ids: &[TokenId],
        vocab_ids: &[TokenId],
    ) -> Result<FpgaPipelineResult, String> {
        // The DSL declaration is the single source of truth for the graph shape.
        let pipeline = cortex_pipeline();
        assert_eq!(pipeline.nodes().len(), 2);
        assert_eq!(pipeline.edges().len(), 1);

        // Ungated LUT: register every ingested id before slicing.
        let mut effective_lut: Vec<TokenId> = lut_ids.to_vec();
        effective_lut.extend_from_slice(doc_ids);
        effective_lut.sort_unstable();
        effective_lut.dedup();

        // Lower the declaration to a wire graph spec and validate it both ways
        // (defense in depth: the DSL already validated the same invariants).
        let spec = ModuleGraphSpec {
            nodes: vec![
                ModuleNodeSpec {
                    node: 0,
                    kind: ModuleKind::Slice,
                    config: vec![
                        ("encoding".to_string(), "tid".to_string()),
                        ("lut_ids".to_string(), csv(&effective_lut)),
                    ],
                },
                ModuleNodeSpec {
                    node: 1,
                    kind: ModuleKind::Gate,
                    config: vec![("lut_ids".to_string(), csv(vocab_ids))],
                },
            ],
            edges: vec![ModuleEdge {
                from: (0, 0),
                to: (1, 0),
            }],
        };
        validate_graph(&spec).map_err(|e| e.to_string())?;

        // The bridge is the execution backend: the cortex submits wire commands
        // to a `ModuleDriver` and drains responses. No module is ever stepped
        // directly from here (separation contract §5).
        self.driver
            .submit(ModuleCommand::Configure { spec })
            .map_err(|e| e.to_string())?;

        // Document HLLSet: active bit positions under the tid inscription.
        let mut positions: Vec<u32> = doc_ids
            .iter()
            .map(|&id| {
                let (reg, tz) = token_to_position(&token_in_bytes(id));
                reg * BITS_PER_REG + tz
            })
            .collect();
        positions.sort_unstable();
        positions.dedup();

        self.driver
            .submit(ModuleCommand::Feed {
                node: 0,
                port: 0,
                packet: Packet {
                    ids: positions,
                    values: Vec::new(),
                },
            })
            .map_err(|e| e.to_string())?;
        self.driver
            .submit(ModuleCommand::Step { node: 0 })
            .map_err(|e| e.to_string())?;
        let materialized_ids = take_output(&mut self.driver, 0, 0)?;

        self.driver
            .submit(ModuleCommand::Feed {
                node: 1,
                port: 0,
                packet: Packet {
                    ids: materialized_ids.clone(),
                    values: Vec::new(),
                },
            })
            .map_err(|e| e.to_string())?;
        self.driver
            .submit(ModuleCommand::Step { node: 1 })
            .map_err(|e| e.to_string())?;
        let restored_ids = take_output(&mut self.driver, 1, 0)?;

        let leaks: Vec<TokenId> = materialized_ids
            .iter()
            .copied()
            .filter(|id| !restored_ids.contains(id))
            .collect();

        Ok(FpgaPipelineResult {
            materialized_ids,
            restored_ids,
            leaks,
        })
    }

    /// Declare the cortex-computed grounding verdict into a one-node
    /// `GroundModule` graph and run it through the wire executor.
    ///
    /// The bridge's `GroundModule` is config-driven (Option A): the cortex
    /// decides (matrix + report), the bridge passes the declared prior and
    /// report through verbatim, quantized by `scale`. No matrix lives in the
    /// bridge.
    pub fn run_ground_passthrough(
        &mut self,
        prior: &[(TokenId, i64)],
        report: &[GroundReportEntry],
        scale: i64,
    ) -> Result<GroundWireResult, String> {
        if scale <= 0 {
            return Err(format!("scale must be positive, got {scale}"));
        }

        let prior_cfg = prior
            .iter()
            .map(|(id, weight)| format!("{id}:{weight}"))
            .collect::<Vec<_>>()
            .join(",");
        let report_cfg = report
            .iter()
            .map(|entry| {
                format!(
                    "{}:{}:{}:{}:{}",
                    entry.id, entry.tau, entry.rho, entry.srho, entry.r_link
                )
            })
            .collect::<Vec<_>>()
            .join(",");

        let spec = ModuleGraphSpec {
            nodes: vec![ModuleNodeSpec {
                node: 0,
                kind: ModuleKind::Ground,
                config: vec![
                    ("prior".to_string(), prior_cfg),
                    ("report".to_string(), report_cfg),
                    ("scale".to_string(), scale.to_string()),
                ],
            }],
            edges: vec![],
        };
        validate_graph(&spec).map_err(|e| e.to_string())?;

        self.driver
            .submit(ModuleCommand::Configure { spec })
            .map_err(|e| e.to_string())?;
        // One valid beat: the module emits prior (port 0) + report (port 1).
        self.driver
            .submit(ModuleCommand::Feed {
                node: 0,
                port: 0,
                packet: Packet {
                    ids: Vec::new(),
                    values: Vec::new(),
                },
            })
            .map_err(|e| e.to_string())?;
        self.driver
            .submit(ModuleCommand::Step { node: 0 })
            .map_err(|e| e.to_string())?;

        let responses = self.driver.drain().map_err(|e| e.to_string())?;
        let mut prior_packet: Option<Packet> = None;
        let mut report_packet: Option<Packet> = None;
        for response in responses {
            if let ModuleResponse::Output { node: 0, port, packet } = response {
                match port {
                    0 => prior_packet = Some(packet),
                    1 => report_packet = Some(packet),
                    _ => {}
                }
            }
        }

        let prior = prior_packet.ok_or_else(|| "no prior packet on ground port 0".to_string())?;
        let report =
            report_packet.ok_or_else(|| "no report packet on ground port 1".to_string())?;

        Ok(GroundWireResult {
            prior_ids: prior.ids,
            prior_values: prior.values,
            report_ids: report.ids,
            report_values: report.values,
        })
    }
}

/// One entry of the cortex-declared grounding report (wire config form).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundReportEntry {
    pub id: TokenId,
    pub tau: f64,
    pub rho: f64,
    pub srho: f64,
    pub r_link: u64,
}

/// The verbatim wire output of the bridge's `GroundModule`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroundWireResult {
    /// Prior ids/weights emitted on output port 0 (verbatim).
    pub prior_ids: Vec<TokenId>,
    pub prior_values: Vec<i64>,
    /// Report ids/values emitted on output port 1 (verbatim, quantized).
    pub report_ids: Vec<TokenId>,
    pub report_values: Vec<i64>,
}

/// Run the cortex pipeline over a fresh bridge executor.
pub fn run_cortex_pipeline(
    doc_ids: &[TokenId],
    lut_ids: &[TokenId],
    vocab_ids: &[TokenId],
) -> Result<FpgaPipelineResult, String> {
    BridgeExecutor::new().run_cortex_pipeline(doc_ids, lut_ids, vocab_ids)
}

/// Drain the bridge driver and take the first output on `(node, port)`.
fn take_output(
    driver: &mut ewm_sim::SimModuleDriver,
    node: NodeId,
    port: PortId,
) -> Result<Vec<TokenId>, String> {
    let responses = driver.drain().map_err(|e| e.to_string())?;
    for response in responses {
        if let ModuleResponse::Output {
            node: n,
            port: p,
            packet,
        } = response
        {
            if n == node && p == port {
                return Ok(packet.ids);
            }
        }
    }
    Err(format!("no output on node {node} port {port}"))
}

/// The LUT-view of a pass: the materialized (active) vocabulary, as a
/// content-addressed vector of token hashes. Ephemeral cache today —
/// upgradeable to a canonical set / Merkle tree later (`lut-view` crate).
pub fn view_from_tokens(ids: &[TokenId]) -> lut_view::LutView {
    lut_view::LutView::from_tokens(ids.iter().map(|&n| token_in_bytes(n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex_core::CortexPipeline;

    /// Collision-free fixture (excludes the pinned 262/48300 collision pair).
    const LUT_IDS: &[TokenId] = &[0, 1, 44, 464, 671, 1169, 16326, 18308, 27140];

    fn bytes(ids: &[TokenId]) -> Vec<Vec<u8>> {
        ids.iter().map(|&n| token_in_bytes(n).to_vec()).collect()
    }

    fn parse(ids: &[Vec<u8>]) -> Vec<TokenId> {
        let mut out: Vec<TokenId> = ids
            .iter()
            .map(|t| {
                let s = String::from_utf8_lossy(t);
                s.trim_start_matches("tid").parse().expect("tid{n}")
            })
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn dsl_declaration_is_valid_and_lowers() {
        let pipeline = cortex_pipeline();
        assert_eq!(pipeline.nodes().len(), 2);
        assert_eq!(pipeline.edges().len(), 1);
        assert_eq!(pipeline.edges()[0], ewm_dsl::Edge {
            from: (0, 0),
            to: (1, 0),
        });
    }

    #[test]
    fn golden_run_matches_vendored_cortex_pipeline() {
        let doc = [1169u32, 44, 18308];
        let fpga = run_cortex_pipeline(&doc, LUT_IDS, LUT_IDS).expect("golden run");

        let mut reference = CortexPipeline::new();
        let vocab = bytes(LUT_IDS);
        reference.set_gate(vocab.iter());
        let result = reference.process(&bytes(&doc));

        assert!(result.ok());
        assert_eq!(fpga.ok(), result.ok());
        assert_eq!(parse(&fpga.restored_ids_bytes()), parse(&result.restored_ids));
        assert_eq!(fpga.leaks, parse(&result.leaks));
    }

    #[test]
    fn out_of_vocab_ids_are_reported_not_hidden() {
        let doc = [0u32, 9];
        let vocab = [0u32, 1, 2, 3];
        let fpga = run_cortex_pipeline(&doc, &vocab, &vocab).expect("golden run");

        // LUT is ungated: tid9 is registered and materialized...
        assert!(fpga.materialized_ids.contains(&9));
        // ...but the output gate filters it and reports it as a leak.
        assert_eq!(fpga.restored_ids, vec![0]);
        assert_eq!(fpga.leaks, vec![9]);
        assert!(!fpga.ok());
    }

    #[test]
    fn lut_view_tracks_the_active_vocabulary_and_refreshes_only_on_change() {
        let vocab = [0u32, 1, 2, 3, 4];

        let first = run_cortex_pipeline(&[0, 1, 2], &vocab, &vocab).expect("golden run");
        let view = view_from_tokens(&first.materialized_ids);
        assert_eq!(view.len(), 3);
        assert!(view.key().starts_with("v:"));

        // Same document → same active vocabulary → no view replacement.
        let again = run_cortex_pipeline(&[0, 1, 2], &vocab, &vocab).expect("golden run");
        let again_view = view_from_tokens(&again.materialized_ids);
        assert_eq!(view.refresh(again_view.hashes()), None, "no churn on stable corpus");

        // New token → the view is replaced with a larger instance.
        let bigger = run_cortex_pipeline(&[0, 1, 2, 3], &vocab, &vocab).expect("golden run");
        let bigger_view = view_from_tokens(&bigger.materialized_ids);
        let next = view
            .refresh(bigger_view.hashes())
            .expect("vocabulary grew");
        assert_eq!(next.len(), 4);
        assert_ne!(view.digest(), next.digest());
    }

    impl FpgaPipelineResult {
        fn restored_ids_bytes(&self) -> Vec<Vec<u8>> {
            self.restored_ids
                .iter()
                .map(|&n| token_in_bytes(n).to_vec())
                .collect()
        }
    }
}
