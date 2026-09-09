//! The Noether-loop driver (separation contract §10).
//!
//! ```text
//! H(t) = (S(t), H(t-1), D, R, N)
//! ```
//!
//! The app is stateless between steps: the only state is the **store + the
//! current pointer** (the head of the commit tree). Each turn runs
//!
//! ```text
//! ingest → S(t) → evolve → persist → advance head
//! ```
//!
//! - **ingest** — one streaming pass through `ewm-git::Ingestor` (complete,
//!   single-touch). The ingestor accumulates per-token and per-bit TF;
//! - **S(t)** — the cumulative working set: the lattice join of every turn's
//!   channel sketches. `S(t)` is what gets committed (the same pattern as
//!   `cortex-fpga`'s Noether-agreement test), so `view(repo, head)` and the
//!   context tree describe the same snapshot at two granularities;
//! - **evolve** — the tree-level Noether diff between the previous and the
//!   current context, and the bit-level `CommitView` from `ewm-git`;
//! - **persist / advance head** — one commit per turn; a pass that brings no
//!   new bits is skipped (idempotent, no-change).
//!
//! Recovery is a read, not a replay (§10.1): `CortexApp::open` reads the
//! head, dereferences the snapshots, and rebuilds the presentation from the
//! commit messages — the durable part of the tape.

use context_tree::TreeDiff;
use cortex_core::{Context, TokenId};
use cortex_fpga::LutFamily;
use ewm_git::{
    view, BitTf, CommitView, IngestSink, Ingestor, LatticeState, ObjectId, ObjectStore, Repository,
    StoreError,
};
use hllset_contracts::token::token_in_bytes;
use hllset_core::{HLLSet, TFVec};

/// The app's inscription choice, explicit per NEXT_SESSION §3.7.
///
/// This app path uses the `tid{n}` encoding (nanoLM/cortex). The 4-byte LE
/// encoding is exercised in `token.rs` and available to pipelines that pick
/// it; every test states which one it uses.
pub const APP_ENCODING_NAME: &str = "tid";

/// One recorded turn: the token collection and its seed-0 sketch.
#[derive(Clone, Debug)]
pub struct TurnRecord {
    /// The turn's token ids (`tid{n}` encoding).
    pub ids: Vec<TokenId>,
    /// `ingest(L)` at seed 0 — the projection of the token collection
    /// (the correspondence rule's forward direction).
    pub g1: HLLSet,
    /// The commit this turn produced (`None` when the pass brought no new
    /// bits and was skipped as idempotent).
    pub commit: Option<ObjectId>,
}

/// The result of one loop step.
#[derive(Clone, Debug)]
pub struct TurnOutcome {
    /// The commit produced by this turn (`None` = no-change pass).
    pub commit: Option<ObjectId>,
    /// The current head (the pointer) after the step.
    pub head: Option<ObjectId>,
    /// The unified context `S(t)` after the step.
    pub context: Context,
    /// Tree-level D/R/N from the previous context to this one.
    pub diff: TreeDiff,
    /// Bit-level Noether view of the head commit (G1). Its `S(t)` is the
    /// cumulative working set — the same snapshot as `context` at bit level.
    pub commit_view: Option<CommitView>,
    /// The full image tokens `V(S(t))` over the known LUT family.
    pub full_image: Vec<Vec<u8>>,
}

/// App errors.
#[derive(Debug)]
pub enum AppError {
    Store(StoreError),
    Other(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Store(e) => write!(f, "store error: {e}"),
            AppError::Other(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AppError {}

impl From<StoreError> for AppError {
    fn from(e: StoreError) -> Self {
        AppError::Store(e)
    }
}

/// Sink that ignores per-pass originals (the app registers the committed
/// cumulative state instead; per-pass provenance is a later increment).
struct NoopSink;

impl IngestSink for NoopSink {
    fn on_original(&mut self, _hllset: &HLLSet, _sha1: String) {}
}

/// The driver: LLM turns in, commits out. Stateless between steps — it can be
/// dropped and reopened from the store (`CortexApp::open` reads the head).
pub struct CortexApp<S: ObjectStore> {
    repo: Repository<S>,
    ingestor: Ingestor,
    luts: LutFamily,
    turns: Vec<TurnRecord>,
    context: Context,
    /// The cumulative working set `S(t)`, per channel (the commit payload).
    working: [HLLSet; 3],
    /// TF baseline restored from the head commit on open (the ingestor's
    /// own accumulator counts only since process start; the sum is the full
    /// monotone TF).
    tf_base: TFVec,
    turn: u64,
}

impl<S: ObjectStore> CortexApp<S> {
    /// A fresh app over `store` (no head yet).
    pub fn new(store: S) -> Self {
        Self {
            repo: Repository::new(store),
            ingestor: Ingestor::new(&[0, 1, 2]),
            luts: LutFamily::new(),
            turns: Vec::new(),
            context: Context::new(),
            working: std::array::from_fn(|_| HLLSet::new()),
            tf_base: TFVec::new(),
            turn: 0,
        }
    }

    /// Recovery: open the store, read the head, dereference, resume — a
    /// read, not a replay (§10.1).
    ///
    /// - the pointer = `repo.head()`;
    /// - the snapshot = the head commit's three channel states;
    /// - the presentation (turns/context/LUT) is rebuilt from the commit
    ///   messages, which carry each turn's token ids.
    pub fn open(store: S) -> Self {
        let repo = Repository::open(store);

        let mut luts = LutFamily::new();
        let mut turns: Vec<TurnRecord> = Vec::new();
        if let Ok(log) = repo.log() {
            for cid in &log {
                if let Ok(commit) = repo.read_commit(cid) {
                    if let Some(ids) = parse_ids_message(&commit.message) {
                        luts.observe(&ids);
                        let g1 = HLLSet::from_tokens(ids.iter().map(|&n| token_in_bytes(n)));
                        turns.push(TurnRecord {
                            ids,
                            g1,
                            commit: Some(cid.clone()),
                        });
                    }
                }
            }
        }

        let working = match repo.head() {
            Some(head) => repo
                .states(head)
                .unwrap_or_else(|_| std::array::from_fn(|_| HLLSet::new())),
            None => std::array::from_fn(|_| HLLSet::new()),
        };

        let tf_base = repo
            .head()
            .and_then(|head| repo.state_tf(head).ok())
            .and_then(|bit_tf| TFVec::from_bytes(&bit_tf.to_bytes()))
            .unwrap_or_default();

        let context = {
            let items: Vec<(&HLLSet, &[TokenId])> = turns
                .iter()
                .map(|turn| (&turn.g1, turn.ids.as_slice()))
                .collect();
            luts.unified_context_for(&items)
        };
        let turn = turns.len() as u64;

        Self {
            repo,
            ingestor: Ingestor::new(&[0, 1, 2]),
            luts,
            turns,
            context,
            working,
            tf_base,
            turn,
        }
    }

    pub fn repo(&self) -> &Repository<S> {
        &self.repo
    }

    pub fn repo_mut(&mut self) -> &mut Repository<S> {
        &mut self.repo
    }

    pub fn ingestor(&self) -> &Ingestor {
        &self.ingestor
    }

    pub fn luts(&self) -> &LutFamily {
        &self.luts
    }

    pub fn turns(&self) -> &[TurnRecord] {
        &self.turns
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn turn_count(&self) -> u64 {
        self.turn
    }

    pub fn head(&self) -> Option<&ObjectId> {
        self.repo.head()
    }

    /// The cumulative working set `S(t)` (G1), derived from the turn leaves.
    pub fn working_g1(&self) -> &HLLSet {
        &self.working[0]
    }

    /// Run one loop step over a turn of `tid{n}`-encoded token ids.
    pub fn run_turn(&mut self, ids: &[TokenId]) -> Result<TurnOutcome, AppError> {
        let bytes: Vec<Vec<u8>> = ids.iter().map(|&n| token_in_bytes(n)).collect();

        // 1. ingest — complete, single-touch (mutates the ingestor's TF only).
        let pass = self.ingestor.ingest_stream(bytes.iter(), &mut NoopSink);
        let turn_g1 = pass.channels[0].clone();

        // 2. S(t) — the cumulative working set (lattice join of all turns).
        for (i, channel) in pass.channels.iter().enumerate() {
            self.working[i] = self.working[i].union(channel);
        }

        // 3. persist — one commit per turn; skipped when no new bits arrive.
        let state = LatticeState {
            g1: self.working[0].clone(),
            g2: self.working[1].clone(),
            g3: self.working[2].clone(),
            tf: self.current_tf(),
        };
        let parents: Vec<ObjectId> = self.repo.head().cloned().into_iter().collect();
        let message = ids_message(ids);
        let commit = if self.repo.new_bits(&state) == 0 {
            None
        } else {
            Some(self.repo.commit(&state, &parents, &message)?)
        };

        // The turn is a defined token collection: its HLLSet is `turn_g1`,
        // and its restoration rule is the known LUT family.
        self.turns.push(TurnRecord {
            ids: ids.to_vec(),
            g1: turn_g1,
            commit: commit.clone(),
        });
        self.luts.observe(ids);
        self.turn += 1;

        // 4. evolve — the unified context over the full turn working set.
        let prev = self.context.clone();
        let items: Vec<(&HLLSet, &[TokenId])> = self
            .turns
            .iter()
            .map(|turn| (&turn.g1, turn.ids.as_slice()))
            .collect();
        let next = self.luts.unified_context_for(&items);
        let diff = prev.diff(&next);
        self.context = next.clone();

        // The head *is* the pointer; the CommitView is the bit-level H(t).
        let head = self.repo.head().cloned();
        let commit_view = match &head {
            Some(cid) => Some(view(&self.repo, cid)?),
            None => None,
        };

        Ok(TurnOutcome {
            commit,
            head,
            context: next,
            diff,
            commit_view,
            full_image: self.context.full_image_tokens(),
        })
    }

    /// The full monotone TF: the restored baseline plus the ingestor's
    /// since-start accumulation (TF stored, never reset).
    fn current_tf(&self) -> BitTf {
        let mut values = self.tf_base.values.clone();
        for (i, v) in values.iter_mut().enumerate() {
            *v += self.ingestor.bit_tf().values[i];
        }
        BitTf::from_tfvec(TFVec::from_values(values).expect("32768 entries"))
    }
}

/// The commit message carries the turn's token ids — the durable record the
/// recovery path reads to rebuild the token-level presentation.
fn ids_message(ids: &[TokenId]) -> String {
    let csv = ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("ids={csv}")
}

/// Parse the turn ids from a commit message (`ids=1,2,3`).
fn parse_ids_message(message: &str) -> Option<Vec<TokenId>> {
    let csv = message.strip_prefix("ids=")?;
    let mut ids = Vec::new();
    for part in csv.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let id: TokenId = part.parse().ok()?;
        ids.push(id);
    }
    if ids.is_empty() {
        return None;
    }
    Some(ids)
}
