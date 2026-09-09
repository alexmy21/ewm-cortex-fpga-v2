//! `Repository` — commit DAG, refs, merge, log, GC, time-travel projection.
//!
//! The minimal 2005-Git surface:
//!
//! - `commit` — store the three channel blobs (G1/G2/G3) and a commit
//!   object, advance `HEAD`;
//! - `state` / `state_channel` / `states` — resolve a commit to its HLLSet
//!   state(s);
//! - `project` — **time travel**: intersect any HLLSet with the global
//!   state of a given channel at any previous commit;
//! - `merge` — lattice join of two commits, per channel, recorded as a
//!   two-parent commit;
//! - `log` — ancestors reachable from `HEAD`;
//! - `gc` / `gc_to` — prune objects unreachable from `HEAD`/branches/keep,
//!   optionally archiving them first.
//!
//! **Commit is the timer.** No wall-clock enters object identity; the SHA1
//! of the commit is the only time stamp the evolution needs.
//!
//! **G1/G2/G3 are commit-linked.** One commit carries all three channel
//! snapshots, so any HLLSet can be projected onto any previous commit state.

use std::collections::{HashMap, HashSet, VecDeque};

use hllset_core::HLLSet;
use crate::ingest::{IngestSink, Ingestor};

use crate::hllset_lut::{BitTf, HllsetLut};
use crate::object::{Commit, Gx, Object, ObjectId};
use crate::store::{ObjectStore, Result, StoreError};

/// The lattice state of one commit: the three channel HLLSets plus the
/// bit-TF vector over the lattice top.
///
/// `G1` = 1-gram / seed-0 bits, `G2` = 2-gram / seed-1 bits,
/// `G3` = 3-gram / seed-2 bits. The channels are just bits — any HLLSet
/// projection can be stored in any channel. `tf` holds the 32K TF vector
/// for each bit of the lattice top (the union of all original HLLSets).
#[derive(Clone, Debug, Default)]
pub struct LatticeState {
    pub g1: HLLSet,
    pub g2: HLLSet,
    pub g3: HLLSet,
    pub tf: BitTf,
}

impl LatticeState {
    /// A single HLLSet replicated into all three channels (single-seed
    /// convenience); the TF vector is touched once by that HLLSet.
    pub fn single(hllset: &HLLSet) -> Self {
        let mut tf = BitTf::new();
        tf.touch(hllset);
        Self {
            g1: hllset.clone(),
            g2: hllset.clone(),
            g3: hllset.clone(),
            tf,
        }
    }

    /// The three channels as an array.
    pub fn channels(&self) -> [HLLSet; 3] {
        [self.g1.clone(), self.g2.clone(), self.g3.clone()]
    }

    /// One channel by selector.
    pub fn get(&self, gx: Gx) -> &HLLSet {
        match gx {
            Gx::G1 => &self.g1,
            Gx::G2 => &self.g2,
            Gx::G3 => &self.g3,
        }
    }
}

/// A repository over a content-addressed object store.
#[derive(Clone, Debug, Default)]
pub struct Repository<S: ObjectStore> {
    store: S,
    head: Option<ObjectId>,
    branches: HashMap<String, ObjectId>,
    /// `<SHA1, TH>` — touch counts of original HLLSets.
    hllset_lut: HllsetLut,
    /// Lattice top per channel (union of all committed channel states) —
    /// an O(1) runtime cache; derivable from the store.
    tops: [HLLSet; 3],
}

impl<S: ObjectStore> Repository<S> {
    pub fn new(store: S) -> Self {
        Self {
            store,
            head: None,
            branches: HashMap::new(),
            hllset_lut: HllsetLut::new(),
            tops: [HLLSet::new(), HLLSet::new(), HLLSet::new()],
        }
    }

    /// Open a repository, restoring `HEAD` if the store persists it.
    pub fn open(store: S) -> Self {
        let head = store.read_head();
        let mut repo = Self {
            store,
            head,
            branches: HashMap::new(),
            hllset_lut: HllsetLut::new(),
            tops: [HLLSet::new(), HLLSet::new(), HLLSet::new()],
        };
        repo.rebuild_tops();
        repo
    }

    /// Rebuild the lattice-top cache from the commit history (open-time only).
    fn rebuild_tops(&mut self) {
        if let Ok(log) = self.log() {
            for id in log {
                if let Ok(states) = self.states(&id) {
                    for (i, hll) in states.iter().enumerate() {
                        let top = &self.tops[i];
                        self.tops[i] = top.union(hll);
                    }
                }
            }
        }
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    pub fn head(&self) -> Option<&ObjectId> {
        self.head.as_ref()
    }

    /// The `<SHA1, TH>` touch-count registry of original HLLSets.
    pub fn hllset_lut(&self) -> &HllsetLut {
        &self.hllset_lut
    }

    /// Create a commit from a lattice state (three channels + bit-TF).
    /// `parents` may be empty (root).
    ///
    /// Updates `HEAD` to the new commit (and persists it, if the store
    /// supports ref persistence). The three channel blobs are registered
    /// and touched in the HLLSet-LUT — each touch counts.
    pub fn commit(
        &mut self,
        state: &LatticeState,
        parents: &[ObjectId],
        message: &str,
    ) -> Result<ObjectId> {
        let mut trees = Vec::with_capacity(3);
        for channel in state.channels() {
            let blob = Object::Blob(channel.to_bytes());
            let blob_id = blob.id();
            self.store.put(&blob_id, &blob)?;
            // HLLSet-LUT is keyed by the HLLSet content address, not the
            // blob-object id (blob ids include the "blob" header).
            let content_id = ObjectId::of_bytes(&channel.to_bytes());
            self.hllset_lut.register(&content_id);
            self.hllset_lut.touch(&content_id);
            trees.push(blob_id);
        }

        let tf_blob = Object::Blob(state.tf.to_bytes());
        let tf_id = tf_blob.id();
        self.store.put(&tf_id, &tf_blob)?;

        let commit = Commit {
            trees: trees.try_into().expect("three channels"),
            tf: tf_id,
            parents: parents.to_vec(),
            message: message.to_string(),
        };
        let commit_id = commit.id();
        self.store.put(&commit_id, &Object::Commit(Box::new(commit)))?;

        // Maintain the lattice-top cache: union the new channel states.
        let channels = state.channels();
        for (i, top) in self.tops.iter_mut().enumerate() {
            *top = top.union(&channels[i]);
        }

        self.head = Some(commit_id.clone());
        self.store.write_head(&commit_id)?;
        Ok(commit_id)
    }

    /// Lattice top of channel G1 (union of every committed channel state).
    pub fn lattice_top(&self, gx: Gx) -> &HLLSet {
        &self.tops[gx.index()]
    }

    /// Context size = bits in the G1 lattice top (the full working set).
    pub fn context_size(&self) -> u64 {
        self.tops[0].popcount()
    }

    /// Bits of `state` that are NOT yet covered by the lattice top — i.e.
    /// the new information a commit of `state` would bring.
    pub fn new_bits(&self, state: &LatticeState) -> u64 {
        let channels = state.channels();
        channels
            .iter()
            .enumerate()
            .map(|(i, hll)| hll.difference(&self.tops[i]).popcount())
            .sum()
    }

    /// Warn when the context outgrows `warn_above` bits. This is the
    /// non-committing guidance path: compound HLLSets do not commit, so the
    /// caller checks the context and compresses when it gets too big.
    pub fn context_warning(&self, warn_above: u64) -> Option<ContextWarning> {
        let bits = self.context_size();
        (bits > warn_above).then(|| ContextWarning {
            bits,
            warn_above,
        })
    }

    /// One streaming ingest pass and automatic commit.
    ///
    /// Runs the ingestor's single pass over `tokens` (which updates all
    /// n-gram/seed-n LUTs, builds the working HLLSets internally, and
    /// updates TF), registers + touches every produced original HLLSet in
    /// the HLLSet-LUT, and commits the resulting three-channel lattice
    /// state (with the ingestor's cumulative bit-TF snapshot).
    ///
    /// Ingestion is the **only automatic commit**: a commit is created
    /// only if the pass brings new bits to the lattice top. If every bit
    /// of the pass is already covered by previously committed states, the
    /// pass is a no-change and returns `Ok(None)` — no unnecessary commit.
    pub fn ingest<I, B>(
        &mut self,
        ingestor: &mut Ingestor,
        tokens: I,
        message: &str,
    ) -> Result<Option<ObjectId>>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        struct LutSink<'a, S: ObjectStore> {
            repo: &'a mut Repository<S>,
        }
        impl<S: ObjectStore> IngestSink for LutSink<'_, S> {
            fn on_original(&mut self, _hllset: &HLLSet, sha1: String) {
                let id = ObjectId(sha1);
                self.repo.hllset_lut.register(&id);
                self.repo.hllset_lut.touch(&id);
            }
        }

        let parents: Vec<ObjectId> = self.head.clone().into_iter().collect();
        let out = ingestor.ingest_stream(tokens, &mut LutSink { repo: self });
        assert_eq!(out.channels.len(), 3, "ewm-git commits require three channels");

        let state = LatticeState {
            g1: out.channels[0].clone(),
            g2: out.channels[1].clone(),
            g3: out.channels[2].clone(),
            tf: BitTf::from_tfvec(ingestor.bit_tf().clone()),
        };

        if self.new_bits(&state) == 0 {
            // No new information: the pass is fully derivable from the
            // lattice top. Skip the commit.
            return Ok(None);
        }
        self.commit(&state, &parents, message).map(Some)
    }

    /// Resolve one channel of a commit to its HLLSet state.
    pub fn state_channel(&self, commit_id: &ObjectId, gx: Gx) -> Result<HLLSet> {
        let commit = self.read_commit(commit_id)?;
        let blob = self.store.get(&commit.trees[gx.index()])?;
        match blob {
            Object::Blob(bytes) => HLLSet::from_bytes(&bytes)
                .ok_or_else(|| StoreError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "corrupt state blob"))),
            _ => Err(StoreError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a blob"))),
        }
    }

    /// All three channels of a commit.
    pub fn states(&self, commit_id: &ObjectId) -> Result<[HLLSet; 3]> {
        let mut out = Vec::with_capacity(3);
        for gx in Gx::ALL {
            out.push(self.state_channel(commit_id, gx)?);
        }
        Ok(out.try_into().expect("three channels"))
    }

    /// The `G1` state (default channel, single-seed convenience).
    pub fn state(&self, commit_id: &ObjectId) -> Result<HLLSet> {
        self.state_channel(commit_id, Gx::G1)
    }

    /// The bit-TF vector snapshot of a commit.
    pub fn state_tf(&self, commit_id: &ObjectId) -> Result<BitTf> {
        let commit = self.read_commit(commit_id)?;
        let blob = self.store.get(&commit.tf)?;
        match blob {
            Object::Blob(bytes) => BitTf::from_bytes(&bytes)
                .ok_or_else(|| StoreError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "corrupt TF blob"))),
            _ => Err(StoreError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a blob"))),
        }
    }

    /// **Time travel**: project `hllset` onto the global state of `gx` at
    /// `commit_id` — the bits of `hllset` that already existed at that
    /// commit. The queried original HLLSet is touched in the HLLSet-LUT.
    ///
    /// ```text
    /// project(H, t, GX) = H ∩ GX(t)
    /// ```
    pub fn project(
        &mut self,
        hllset: &HLLSet,
        commit_id: &ObjectId,
        gx: Gx,
    ) -> Result<HLLSet> {
        let channel = self.state_channel(commit_id, gx)?;
        let content_id = ObjectId::of_bytes(&channel.to_bytes());
        self.hllset_lut.touch(&content_id);
        Ok(hllset.intersection(&channel))
    }

    /// Time-travel projection onto all three channels.
    pub fn project_all(&mut self, hllset: &HLLSet, commit_id: &ObjectId) -> Result<[HLLSet; 3]> {
        let mut out = Vec::with_capacity(3);
        for gx in Gx::ALL {
            out.push(self.project(hllset, commit_id, gx)?);
        }
        Ok(out.try_into().expect("three channels"))
    }

    /// Read a commit object.
    pub fn read_commit(&self, commit_id: &ObjectId) -> Result<Commit> {
        match self.store.get(commit_id)? {
            Object::Commit(c) => Ok(*c),
            _ => Err(StoreError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a commit"))),
        }
    }

    /// Lattice join of two commits, **per channel**: `state = state(a) ∪
    /// state(b)` for G1/G2/G3, TF merged. This is a **derivation, not a
    /// commit**: the joined compound HLLSet is fully determined by the two
    /// parents, so it brings no new information to the lattice. It does
    /// NOT update `HEAD`.
    ///
    /// The parents' original HLLSets are touched in the HLLSet-LUT (they
    /// participated in a compound HLLSet), and the caller may check
    /// [`Repository::context_warning`] afterwards — compound formation is
    /// the place where the context grows, and the remedy is compression,
    /// not a commit.
    pub fn merge(&mut self, a: &ObjectId, b: &ObjectId) -> Result<LatticeState> {
        let states_a = self.states(a)?;
        let states_b = self.states(b)?;
        let mut tf = self.state_tf(a)?;
        tf.merge(&self.state_tf(b)?);
        let joined = LatticeState {
            g1: states_a[0].union(&states_b[0]),
            g2: states_a[1].union(&states_b[1]),
            g3: states_a[2].union(&states_b[2]),
            tf,
        };
        // The parents' original HLLSets participated in a compound HLLSet.
        let content_ids = states_a
            .iter()
            .chain(states_b.iter())
            .map(|h| ObjectId::of_bytes(&h.to_bytes()))
            .collect::<Vec<_>>();
        self.hllset_lut.touch_many(content_ids.iter());
        Ok(joined)
    }

    /// Explicit git-style merge commit: join two commits AND record the
    /// result. Only for callers that deliberately want the topology — the
    /// lattice information itself is already derivable from the parents.
    pub fn merge_commit(&mut self, a: &ObjectId, b: &ObjectId, message: &str) -> Result<ObjectId> {
        let joined = self.merge(a, b)?;
        self.commit(&joined, &[a.clone(), b.clone()], message)
    }

    /// Ancestors reachable from `HEAD`, oldest first.
    ///
    /// NOTE: this is a **diagnostic** convenience only. System navigation
    /// never walks the DAG — content addressing picks the correct next
    /// HLLSet directly (`state`, `project`, `merge` are O(1) lookups).
    pub fn log(&self) -> Result<Vec<ObjectId>> {
        let Some(head) = self.head.clone() else {
            return Ok(Vec::new());
        };
        let mut order = Vec::new();
        let mut queue = VecDeque::from([head]);
        let mut seen: HashSet<ObjectId> = HashSet::new();
        while let Some(id) = queue.pop_front() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let commit = self.read_commit(&id)?;
            for parent in commit.parents {
                queue.push_back(parent);
            }
            order.push(id);
        }
        order.reverse(); // oldest first
        Ok(order)
    }

    /// Create or move a branch ref to a commit.
    pub fn set_branch(&mut self, name: &str, commit: &ObjectId) {
        self.branches.insert(name.to_string(), commit.clone());
    }

    /// Current branch refs.
    pub fn branches(&self) -> &HashMap<String, ObjectId> {
        &self.branches
    }

    /// Mark every object reachable from `HEAD`, branches, and `keep`.
    fn reachable_mark(&self, keep: &[ObjectId]) -> Vec<ObjectId> {
        let mut reachable: Vec<ObjectId> = Vec::new();
        if let Some(head) = self.head.clone() {
            reachable.push(head);
        }
        reachable.extend(self.branches.values().cloned());
        reachable.extend(keep.iter().cloned());

        let mut queue: VecDeque<ObjectId> = reachable.iter().cloned().collect();
        let mut mark: HashSet<ObjectId> = HashSet::new();
        while let Some(id) = queue.pop_front() {
            if !mark.insert(id.clone()) {
                continue;
            }
            if let Ok(Object::Commit(commit)) = self.store.get(&id) {
                for tree in &commit.trees {
                    queue.push_back(tree.clone());
                }
                queue.push_back(commit.tf.clone());
                for parent in &commit.parents {
                    queue.push_back(parent.clone());
                }
            }
        }
        mark.into_iter().collect()
    }

    /// Prune objects unreachable from `HEAD`, branches, and `keep`.
    pub fn gc(&mut self, keep: &[ObjectId]) -> Result<GcReport> {
        let mark = self.reachable_mark(keep);
        let doomed: Vec<ObjectId> = self
            .store
            .ids()
            .into_iter()
            .filter(|id| !mark.contains(id))
            .collect();
        let pruned = doomed.len();
        let mut delete_failed = 0usize;
        for id in doomed {
            if self.store.delete(&id).is_err() {
                delete_failed += 1;
            }
        }
        Ok(GcReport {
            pruned,
            archived: 0,
            delete_failed,
        })
    }

    /// Archive-then-prune: every doomed object is written to `archive`
    /// (a content-addressed store such as an IPFS backend) before it is
    /// deleted from the working store.
    ///
    /// **Prune from the working store, never forget from the archive.**
    /// The archived object keeps its `h:<sha1>` key, so pruned branches
    /// stay addressable as long as the archive persists.
    pub fn gc_to<A: ObjectStore>(
        &mut self,
        archive: &mut A,
        keep: &[ObjectId],
    ) -> Result<GcReport> {
        let mark = self.reachable_mark(keep);
        let doomed: Vec<ObjectId> = self
            .store
            .ids()
            .into_iter()
            .filter(|id| !mark.contains(id))
            .collect();

        let mut archived = 0usize;
        for id in &doomed {
            if let Ok(object) = self.store.get(id) {
                archive.put(id, &object)?;
                archived += 1;
            }
        }
        let pruned = doomed.len();
        let mut delete_failed = 0usize;
        for id in doomed {
            if self.store.delete(&id).is_err() {
                delete_failed += 1;
            }
        }
        Ok(GcReport {
            pruned,
            archived,
            delete_failed,
        })
    }
}

/// Result of an archive-then-prune GC pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Objects removed from the working store.
    pub pruned: usize,
    /// Objects written to the archive before pruning.
    pub archived: usize,
    /// Objects that could not be deleted from the working store.
    pub delete_failed: usize,
}

/// Context growth warning: the working set outgrew `warn_above` bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextWarning {
    /// Current context size (G1 lattice-top bits).
    pub bits: u64,
    /// The threshold that was crossed.
    pub warn_above: u64,
}

impl std::fmt::Display for ContextWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "context at {} bits (warn_above {}): consider compression",
            self.bits, self.warn_above
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hllset_lut::BitTf;
    use crate::store::{MemoryStore, ObjectStore};

    fn hll(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens.iter())
    }

    #[test]
    fn commit_chain_and_state_roundtrip() {
        let mut repo = Repository::new(MemoryStore::default());
        let a = repo.commit(&LatticeState::single(&hll(&["a", "b"])), &[], "first").unwrap();
        let b = repo
            .commit(&LatticeState::single(&hll(&["b", "c"])), &[a.clone()], "second")
            .unwrap();
        assert_eq!(repo.head(), Some(&b));
        assert_eq!(repo.state(&a).unwrap().popcount(), hll(&["a", "b"]).popcount());
        let log = repo.log().unwrap();
        assert_eq!(log, vec![a, b]);
    }

    #[test]
    fn commit_links_all_three_channels() {
        let mut repo = Repository::new(MemoryStore::default());
        let state = LatticeState {
            g1: hll(&["one"]),
            g2: hll(&["two"]),
            g3: hll(&["three"]),
            tf: BitTf::new(),
        };
        let c = repo.commit(&state, &[], "channels").unwrap();
        assert_eq!(repo.state_channel(&c, Gx::G1).unwrap().popcount(), state.g1.popcount());
        assert_eq!(repo.state_channel(&c, Gx::G2).unwrap().popcount(), state.g2.popcount());
        assert_eq!(repo.state_channel(&c, Gx::G3).unwrap().popcount(), state.g3.popcount());
    }

    #[test]
    fn time_travel_projects_onto_previous_commit() {
        let mut repo = Repository::new(MemoryStore::default());
        let early = repo
            .commit(&LatticeState::single(&hll(&["a", "b"])), &[], "early")
            .unwrap();
        let later = repo
            .commit(&LatticeState::single(&hll(&["a", "b", "c"])), &[early.clone()], "later")
            .unwrap();

        // The query "a, c, d" projected onto the EARLY commit: only "a" existed.
        let query = hll(&["a", "c", "d"]);
        let projected = repo.project(&query, &early, Gx::G1).unwrap();
        assert_eq!(projected.popcount(), hll(&["a"]).popcount());

        // Onto the LATER commit: "a" and "c" existed.
        let projected = repo.project(&query, &later, Gx::G1).unwrap();
        assert_eq!(projected.popcount(), hll(&["a", "c"]).popcount());
    }

    #[test]
    fn merge_is_lattice_join_per_channel() {
        let mut repo = Repository::new(MemoryStore::default());
        let a = repo.commit(&LatticeState::single(&hll(&["x", "y"])), &[], "a").unwrap();
        let b = repo.commit(&LatticeState::single(&hll(&["y", "z"])), &[], "b").unwrap();
        // merge is a derivation — no commit, no HEAD update.
        let joined = repo.merge(&a, &b).unwrap();
        let expected = hll(&["x", "y"]).union(&hll(&["y", "z"]));
        assert_eq!(joined.g1.popcount(), expected.popcount());
        assert_eq!(repo.head(), Some(&b), "merge must not update HEAD");

        // Explicit git-style merge commit records the two-parent topology.
        let m = repo.merge_commit(&a, &b, "join").unwrap();
        let commit = repo.read_commit(&m).unwrap();
        assert_eq!(commit.parents, vec![a, b]);
    }

    #[test]
    fn gc_prunes_unreachable() {
        let mut repo = Repository::new(MemoryStore::default());
        let a = repo.commit(&LatticeState::single(&hll(&["keep"])), &[], "root").unwrap();
        let orphan = repo
            .commit(&LatticeState::single(&hll(&["gone"])), &[], "orphan")
            .unwrap();
        let b = repo
            .commit(&LatticeState::single(&hll(&["keep", "more"])), &[a.clone()], "tip")
            .unwrap();
        let pruned = repo.gc(&[]).unwrap().pruned;
        assert!(pruned >= 1);
        assert!(repo.store().contains(&b));
        assert!(repo.store().contains(&a));
        assert!(!repo.store().contains(&orphan));
    }

    #[test]
    fn content_addressing_is_deterministic() {
        let mut r1 = Repository::new(MemoryStore::default());
        let mut r2 = Repository::new(MemoryStore::default());
        let c1 = r1
            .commit(&LatticeState::single(&hll(&["same", "state"])), &[], "same message")
            .unwrap();
        let c2 = r2
            .commit(&LatticeState::single(&hll(&["same", "state"])), &[], "same message")
            .unwrap();
        assert_eq!(c1, c2, "identical content must yield identical addresses");
    }

    #[test]
    fn hllset_lut_counts_touches_across_operations() {
        let mut repo = Repository::new(MemoryStore::default());
        let state_a = LatticeState {
            g1: hll(&["a1"]),
            g2: hll(&["a2"]),
            g3: hll(&["a3"]),
            tf: BitTf::new(),
        };
        let state_b = LatticeState {
            g1: hll(&["b1"]),
            g2: hll(&["b2"]),
            g3: hll(&["b3"]),
            tf: BitTf::new(),
        };
        let a = repo.commit(&state_a, &[], "a").unwrap();
        let b = repo.commit(&state_b, &[a.clone()], "b").unwrap();

        // Each commit registers its three channel blobs (distinct → 6 total).
        assert_eq!(repo.hllset_lut().len(), 6, "two commits × three channels");

        // Merge touches the parents' six channel HLLSets (content ids).
        let joined = repo.merge(&a, &b).unwrap();
        let cid_a = ObjectId::of_bytes(&repo.state_channel(&a, Gx::G1).unwrap().to_bytes());
        let cid_b = ObjectId::of_bytes(&repo.state_channel(&b, Gx::G1).unwrap().to_bytes());
        assert!(repo.hllset_lut().th(&cid_a) >= 2, "commit + merge touch");
        assert!(repo.hllset_lut().th(&cid_b) >= 2, "commit + merge touch");

        // Explicit merge commit (for the project-touch check).
        let m = repo.merge_commit(&a, &b, "join").unwrap();
        let cid_m = ObjectId::of_bytes(&repo.state_channel(&m, Gx::G1).unwrap().to_bytes());
        assert_eq!(repo.hllset_lut().th(&cid_m), 1, "commit touch only so far");
        let _ = repo.project(&hll(&["a1"]), &m, Gx::G1).unwrap();
        assert_eq!(repo.hllset_lut().th(&cid_m), 2, "project touch");
        assert_eq!(joined.g1.popcount(), repo.state(&m).unwrap().popcount());
    }

    #[test]
    fn commit_carries_bit_tf_snapshot() {
        let mut repo = Repository::new(MemoryStore::default());
        let a = repo
            .commit(&LatticeState::single(&hll(&["a", "b", "c"])), &[], "a")
            .unwrap();
        let tf = repo.state_tf(&a).unwrap();
        assert_eq!(tf.active_bits(), hll(&["a", "b", "c"]).popcount() as usize);
        assert_eq!(tf.total(), hll(&["a", "b", "c"]).popcount() as f64);
    }

    #[test]
    fn ingest_commits_and_updates_hllset_lut() {
        let mut repo = Repository::new(MemoryStore::default());
        let mut ingestor = Ingestor::new(&[0, 1, 2]);

        let commit = repo
            .ingest(&mut ingestor, ["alpha", "beta", "alpha"], "ingest pass")
            .unwrap()
            .expect("first ingest brings new bits");

        // Three channels resolved from the commit.
        let states = repo.states(&commit).unwrap();
        assert_eq!(states.len(), 3);
        assert!(states[0].popcount() > 0);

        // HLLSet-LUT: three originals registered; each touched twice —
        // once by the ingest sink, once by the commit.
        assert_eq!(repo.hllset_lut().len(), 3, "one original per channel");
        for state in &states {
            let cid = ObjectId::of_bytes(&state.to_bytes());
            assert_eq!(repo.hllset_lut().th(&cid), 2, "ingest + commit touch");
        }

        // Per-token TF accumulated in the ingestor.
        assert_eq!(ingestor.token_tf(b"alpha"), 2);
        assert_eq!(ingestor.token_tf(b"beta"), 1);
    }

    #[test]
    fn ingest_without_changes_skips_commit() {
        let mut repo = Repository::new(MemoryStore::default());
        let mut ingestor = Ingestor::new(&[0, 1, 2]);

        let first = repo
            .ingest(&mut ingestor, ["a", "b"], "first")
            .unwrap()
            .expect("new bits");
        assert_eq!(repo.head(), Some(&first));

        // Same tokens again: every bit already covered → no commit.
        let second = repo.ingest(&mut ingestor, ["a", "b"], "repeat").unwrap();
        assert_eq!(second, None, "no new bits → no commit");
        assert_eq!(repo.head(), Some(&first), "HEAD unchanged");
        assert_eq!(repo.log().unwrap().len(), 1, "exactly one commit");

        // A token that brings a new bit still commits.
        let third = repo
            .ingest(&mut ingestor, ["c"], "third")
            .unwrap()
            .expect("new bit");
        assert_ne!(&third, &first);
        assert_eq!(repo.log().unwrap().len(), 2);
    }

    #[test]
    fn context_warning_after_threshold() {
        let mut repo = Repository::new(MemoryStore::default());
        let mut ingestor = Ingestor::new(&[0, 1, 2]);
        repo.ingest(&mut ingestor, ["a", "b", "c"], "seed").unwrap();

        let bits = repo.context_size();
        assert!(bits > 0);

        // No warning below/at threshold.
        assert_eq!(repo.context_warning(bits), None);
        // Warning once the threshold is crossed.
        let warning = repo.context_warning(bits - 1).unwrap();
        assert_eq!(warning.bits, bits);
        assert!(warning.to_string().contains("consider compression"));
    }
}
