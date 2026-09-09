//! ewm-app integration tests — the NEXT_SESSION §4 plan, in order.
//!
//! Encoding discipline (NEXT_SESSION §3.7): every test states which token
//! encoding it uses. The app path is `tid{n}`; the dual-encoding test
//! exercises both `tid{n}` and 4-byte LE explicitly.

use cortex_core::{ContextMatrix, CortexConfig, GroundingConfig, Leader, Role, recommend};
use cortex_fpga::{BridgeExecutor, GroundReportEntry, h_key};
use ewm_app::{CortexApp, StubLlm, TokenEncoding, TurnSource};
use ewm_git::{LatticeState, LooseStore, MemoryStore, ObjectId, ObjectStore, Repository};
use hllset_contracts::token::{token_in_bytes, token_in_bytes_le};
use hllset_core::HLLSet;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ewm-app-test-{}-{}-{}",
        std::process::id(),
        name,
        std::thread::current().name().unwrap_or("main")
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

// ── 1. Smoke loop ───────────────────────────────────────────────────────────

#[test]
fn smoke_loop_ingest_commits_and_builds_context() {
    // Encoding: tid{n} — the app path.
    let mut app = CortexApp::new(MemoryStore::default());
    let turn1 = [10u32, 20, 30];
    let turn2 = [20u32, 30, 40];

    let out1 = app.run_turn(&turn1).expect("turn 1");
    let c1 = out1.commit.clone().expect("first turn commits");
    assert_eq!(out1.head.as_ref(), Some(&c1));
    assert!(app.repo().store().contains(&c1), "commit exists in the store");
    assert!(!out1.full_image.is_empty(), "full image is non-empty");
    assert_eq!(out1.context.tree().leaves().len(), 1);

    // Root commit: H(t-1) = ∅, so everything is N.
    let v1 = out1.commit_view.as_ref().expect("root view");
    assert_eq!(v1.parent_state.popcount(), 0);
    assert_eq!(v1.departed.popcount(), 0);
    assert_eq!(v1.retained.popcount(), 0);
    assert_eq!(v1.new.popcount(), v1.state.popcount());

    let out2 = app.run_turn(&turn2).expect("turn 2");
    let c2 = out2.commit.clone().expect("second turn commits");
    assert_eq!(out2.head.as_ref(), Some(&c2));

    // Tree-level D/R/N: turn2's leaf entered, turn1's leaf stayed.
    assert_eq!(out2.context.tree().leaves().len(), 2);
    assert_eq!(out2.diff.added, vec![h_key(&turn2)]);
    assert_eq!(out2.diff.retained, vec![h_key(&turn1)]);
    assert!(out2.diff.removed.is_empty());

    // Bit-level agreement: the tree math (cumulative working set) and
    // ewm-git's view of the head describe the same snapshot.
    let h_prev = app.turns()[0].g1.clone();
    let s_now = out2.commit_view.as_ref().expect("head view").state.clone();
    let departed = h_prev.difference(&s_now);
    let retained = h_prev.intersection(&s_now);
    let novel = s_now.difference(&h_prev);

    let v2 = out2.commit_view.as_ref().expect("head view");
    assert_eq!(v2.departed.popcount(), departed.popcount());
    assert_eq!(v2.retained.popcount(), retained.popcount());
    assert_eq!(v2.new.popcount(), novel.popcount());

    // Noether invariants (ALGEBRAIC_FOUNDATION §3).
    assert_eq!(
        v2.departed.union(&v2.retained).popcount(),
        v2.parent_state.popcount(),
        "D ∪ R = H(t-1)"
    );
    assert_eq!(
        v2.retained.union(&v2.new).popcount(),
        v2.state.popcount(),
        "R ∪ N = S(t)"
    );
    assert_eq!(v2.departed.intersection(&v2.new).popcount(), 0, "D ∩ N = ∅");
}

// ── 2. Recovery ─────────────────────────────────────────────────────────────

#[test]
fn recovery_reads_head_and_resumes_without_replay() {
    // Encoding: tid{n}.
    let dir = temp_dir("recovery");
    let turn1 = [10u32, 20, 30];
    let turn2 = [40u32, 50];

    let head1 = {
        let mut app = CortexApp::new(LooseStore::new(&dir));
        app.run_turn(&turn1).expect("turn 1").commit.expect("commit")
    };

    // Crash (drop the side-car). Restart: read the head, dereference, resume.
    let mut app = CortexApp::open(LooseStore::new(&dir));
    assert_eq!(app.head(), Some(&head1), "the head is the pointer");
    assert_eq!(app.repo().log().unwrap().len(), 1);
    assert_eq!(app.turns().len(), 1, "presentation rebuilt from commit messages");
    assert_eq!(app.turns()[0].ids, turn1);
    assert_eq!(app.context().tree().leaves().len(), 1);

    // Replay the same content: same key ⇒ idempotent — no duplicated commit.
    let replay = app.run_turn(&turn1).expect("replay turn");
    assert_eq!(replay.commit, None, "no new bits → no commit");
    assert_eq!(app.head(), Some(&head1), "HEAD unchanged");
    assert_eq!(app.repo().log().unwrap().len(), 1, "no replay");

    // Resume with new content: exactly one new commit, parented to the
    // recovered head.
    let resumed = app.run_turn(&turn2).expect("resume turn");
    let c2 = resumed.commit.expect("new content commits");
    assert_eq!(app.repo().log().unwrap().len(), 2);
    let commit = app.repo().read_commit(&c2).unwrap();
    assert_eq!(commit.parents, vec![head1], "resumed from the recovered head");

    // The presentation now covers both collections (the replay turn is a
    // duplicate leaf, deduplicated by the tree).
    assert_eq!(app.context().tree().leaves().len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── 3. Side-car async: the LLM never blocks ─────────────────────────────────

#[test]
fn side_car_async_llm_never_blocks() {
    // Encoding: tid{n}.
    use std::sync::{mpsc, Arc, Barrier};

    let script: Vec<Vec<u32>> = vec![
        vec![1, 2],
        vec![2, 3],
        vec![3, 4],
        vec![4, 5],
        vec![5, 6],
    ];

    let script_len = script.len();
    let barrier = Arc::new(Barrier::new(2));
    let producer_barrier = Arc::clone(&barrier);
    let (tx, rx) = mpsc::channel::<Vec<u32>>();

    // The LLM thread: produces turns fire-and-forget. It holds NO handle to
    // the app/cortex, so it cannot block on the side-car.
    let producer = std::thread::spawn(move || {
        let mut llm = StubLlm::new(script.clone());
        producer_barrier.wait(); // harness barrier only
        let mut sent = 0;
        while let Some(turn) = llm.next_turn() {
            tx.send(turn).expect("unbounded channel never blocks");
            sent += 1;
        }
        sent
    });

    barrier.wait(); // harness barrier only
    let mut app = CortexApp::new(MemoryStore::default());
    let mut received = 0;
    while let Ok(turn) = rx.recv() {
        let outcome = app.run_turn(&turn).expect("turn");
        assert!(outcome.commit.is_some(), "every scripted turn brings new bits");
        received += 1;
    }

    let sent = producer.join().expect("producer joins");
    assert_eq!(sent, script_len, "the LLM finished all turns on its own");
    assert_eq!(received, script_len, "the driver committed every turn");
    assert_eq!(app.repo().log().unwrap().len(), script_len);
}

// ── 4. Multi-instance: 1 leader + 2 experts + 1 bridge ──────────────────────

#[test]
fn multi_instance_leader_two_experts_one_bridge() {
    // Encoding: tid{n} — the cortex pipeline's inscription.
    let leader_config = CortexConfig { role: Role::Leader };
    let expert_config = CortexConfig::default();
    assert_eq!(leader_config.role, Role::Leader);
    assert_eq!(expert_config.role, Role::Expert);

    let mut leader = Leader::new();
    leader.register("expert-a");
    leader.register("expert-b");

    // ONE bridge serves both experts (tenant isolation = a fresh graph spec
    // per submission; the bridge keeps no expert state between commands).
    let mut bridge = BridgeExecutor::new();

    let doc_a = [1u32, 2, 3];
    let result_a = bridge
        .run_cortex_pipeline(&doc_a, &doc_a, &doc_a)
        .expect("expert-a submission");
    let doc_b = [4u32, 5, 6];
    let result_b = bridge
        .run_cortex_pipeline(&doc_b, &doc_b, &doc_b)
        .expect("expert-b submission");

    assert!(result_a.ok() && result_b.ok(), "both submissions pass the gate");
    assert_eq!(result_a.restored_ids, vec![1, 2, 3]);
    assert_eq!(result_b.restored_ids, vec![4, 5, 6]);

    // The leader routes to all experts and merges by the lattice join
    // (sorted union).
    let doc = [2u32, 99];
    assert_eq!(
        leader.route(&doc),
        vec!["expert-a".to_string(), "expert-b".to_string()]
    );
    let merged = leader.merge(&[result_a.restored_ids.clone(), result_b.restored_ids.clone()]);
    assert_eq!(merged, vec![1, 2, 3, 4, 5, 6]);
}

// ── 5. Grounding through the bridge's GroundModule ──────────────────────────

#[test]
fn grounding_report_passes_through_bridge_ground_module() {
    // Encoding: tid{n} (the cortex-side vocabulary).
    let matrix = ContextMatrix::from_sequence(&[1, 2, 3, 4]);
    let outputs = [2u32, 3, 999];

    // The cortex decides: τ/ρ over the measured matrix.
    let report = recommend(&matrix, &outputs, &GroundingConfig::default());
    assert_eq!(report.flagged, vec![999], "novel token is flagged");
    assert!((report.tau - 2.0 / 3.0).abs() < 1e-9);
    assert!((report.rho - 1.0 / 3.0).abs() < 1e-9);
    assert!(!report.grounded);

    // The cortex declares the prior + report into a bridge GroundModule
    // config; the wire executor passes them through verbatim (quantized).
    let (prior_ids, prior_values) = matrix.prior();
    let prior: Vec<(u32, i64)> = prior_ids
        .iter()
        .copied()
        .zip(prior_values.iter().copied())
        .collect();

    let scale = 1_000_000i64;
    let entries: Vec<GroundReportEntry> = outputs
        .iter()
        .map(|&id| GroundReportEntry {
            id,
            tau: report.tau,
            rho: report.rho,
            srho: report.srho,
            r_link: report.r_link,
        })
        .collect();

    let mut bridge = BridgeExecutor::new();
    let wire = bridge
        .run_ground_passthrough(&prior, &entries, scale)
        .expect("ground pass-through");

    assert_eq!(wire.prior_ids, prior_ids, "prior ids pass through verbatim");
    assert_eq!(wire.prior_values, prior_values, "prior weights pass through verbatim");

    let quantize = |x: f64| (x * scale as f64).round() as i64;
    let mut expected_ids = Vec::new();
    let mut expected_values = Vec::new();
    for entry in &entries {
        expected_ids.push(entry.id);
        expected_values.extend_from_slice(&[
            quantize(entry.tau),
            quantize(entry.rho),
            quantize(entry.srho),
            quantize(entry.r_link as f64),
        ]);
    }
    assert_eq!(wire.report_ids, expected_ids);
    assert_eq!(wire.report_values, expected_values);
}

// ── 6. Zero-copy: two threads share one store, CIDs only ────────────────────

#[test]
fn zero_copy_two_threads_share_one_store_by_cid() {
    // Encoding: tid{n} (the committed state is built from tid tokens).
    let dir = temp_dir("zero-copy");
    let shared = HLLSet::from_tokens([token_in_bytes(7), token_in_bytes(9)]);
    let expected_popcount = shared.popcount();

    let writer_dir = dir.clone();
    // The channel carries ONLY the CID — by type, payload bytes cannot cross.
    let (tx, rx) = std::sync::mpsc::channel::<ObjectId>();

    let writer = std::thread::spawn(move || {
        let mut repo: Repository<LooseStore> = Repository::new(LooseStore::new(&writer_dir));
        let cid = repo
            .commit(&LatticeState::single(&shared), &[], "shared object")
            .expect("commit");
        tx.send(cid.clone()).expect("send CID");
        cid
    });

    let cid = rx.recv().expect("receive CID");
    let reader_cid = cid.clone();
    let reader_dir = dir.clone();
    let reader = std::thread::spawn(move || {
        let repo: Repository<LooseStore> = Repository::open(LooseStore::new(&reader_dir));
        // The object is already resident in the shared store; the reader
        // dereferences it in place from the CID.
        repo.state(&reader_cid).expect("dereference").popcount()
    });

    let writer_cid = writer.join().expect("writer joins");
    assert_eq!(writer_cid, cid, "both threads agree on the content address");
    assert_eq!(
        reader.join().expect("reader joins"),
        expected_popcount,
        "the reader resolved the same object from the shared store"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── 7. Dual token encodings are both first-class ────────────────────────────

#[test]
fn dual_token_encodings_are_both_first_class() {
    // Explicit: this test exercises BOTH inscriptions, one assertion each.
    let tid = TokenEncoding::Tid;
    let le = TokenEncoding::Le;

    assert_eq!(tid.encode(44), b"tid44");
    assert_eq!(tid.parse(b"tid44"), Some(44));

    assert_eq!(le.encode(44), [44, 0, 0, 0]);
    assert_eq!(le.parse(&[44, 0, 0, 0]), Some(44));

    // Cross-parse fails by construction: neither inscription is the other.
    assert_eq!(tid.parse(&[44, 0, 0, 0]), None);
    assert_eq!(le.parse(b"tid44"), None);

    // Both inscriptions are equally valid tokens for the HLLSet lattice.
    let from_tid = HLLSet::from_tokens([tid.encode(44)]);
    let from_le = HLLSet::from_tokens([le.encode(44)]);
    assert_eq!(from_tid.popcount(), 1);
    assert_eq!(from_le.popcount(), 1);

    // The contracts expose the raw inscriptions too (soldered leaf).
    assert_eq!(token_in_bytes(44), b"tid44");
    assert_eq!(token_in_bytes_le(44), [44, 0, 0, 0]);
}

// The harness uses the stub explicitly: scripted turns make every assertion
// reproducible (NEXT_SESSION §4.1).
#[test]
fn stub_llm_replays_scripted_turns_deterministically() {
    // Encoding: tid{n}.
    let mut llm = StubLlm::with_script(&[&[1, 2, 3], &[4, 5]]);
    assert_eq!(llm.next_turn(), Some(vec![1, 2, 3]));
    assert_eq!(llm.next_turn(), Some(vec![4, 5]));
    assert_eq!(llm.next_turn(), None, "script exhausted");
}
