//! Roles and the leader's minimal duties.
//!
//! Per the separation contract (`BRIDGE_CORTEX_SEPARATION.md`):
//!
//! - **Expert** — one `ewm-cortex` instance holding one domain's context and
//!   knowledge; the unit of specialization in MoE.
//! - **Leader** — an `ewm-cortex` instance configured with `role = leader`;
//!   owns the expert registry, routing, gating, merging, and lifecycle.
//!   It is a *configuration*, not a new codebase.
//!
//! The first increment implements the role flag and the leader's two
//! algebraic duties — registry and merge. Routing defaults to "all experts"
//! (the leader declares the actual gate to the bridge as a `GateModule`
//! config); TF-weighted routing and expert lifecycle are later increments.

use crate::TokenId;

/// The role an instance plays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Role {
    /// A domain expert: owns its context, LUTs/views, grounding, persistence.
    #[default]
    Expert,
    /// The leader: registry, routing, gate, merge, lifecycle.
    Leader,
}

impl Role {
    pub const fn as_str(self) -> &'static str {
        match self {
            Role::Expert => "expert",
            Role::Leader => "leader",
        }
    }

    /// Parse a role from a config string (unknown → `None`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "expert" => Some(Role::Expert),
            "leader" => Some(Role::Leader),
            _ => None,
        }
    }
}

/// Instance configuration. The role flag is the only mandatory field today.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CortexConfig {
    pub role: Role,
}

impl Default for CortexConfig {
    fn default() -> Self {
        Self { role: Role::Expert }
    }
}

impl CortexConfig {
    /// Build a config from `role` key-value pairs (unknown keys are ignored).
    pub fn from_pairs(pairs: &[(String, String)]) -> Self {
        let mut config = Self::default();
        for (key, value) in pairs {
            if key == "role" {
                if let Some(role) = Role::parse(value) {
                    config.role = role;
                }
            }
        }
        config
    }
}

/// The leader: an expert registry plus the two algebraic duties that can be
/// executed without further specification.
#[derive(Clone, Debug, Default)]
pub struct Leader {
    experts: Vec<String>,
}

impl Leader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an expert id (idempotent, order-preserving).
    pub fn register(&mut self, id: impl Into<String>) {
        let id = id.into();
        if !self.experts.contains(&id) {
            self.experts.push(id);
        }
    }

    /// The registered expert ids, in registration order.
    pub fn experts(&self) -> &[String] {
        &self.experts
    }

    /// Route a document to experts.
    ///
    /// First increment: every registered expert receives the document. The
    /// leader's real gating is declared to the bridge (a `GateModule` config
    /// computed from grounding); routing by TF/grounding is a later increment.
    pub fn route(&self, _doc: &[TokenId]) -> Vec<String> {
        self.experts.clone()
    }

    /// Merge expert outputs: the lattice join — sorted, deduplicated union.
    ///
    /// This is the token-space analogue of the HLLSet join; TF-weighted
    /// ranking over the merged set is derived later, never stored here.
    pub fn merge(&self, outputs: &[Vec<TokenId>]) -> Vec<TokenId> {
        let mut out: Vec<TokenId> = outputs.iter().flatten().copied().collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_parses_and_defaults_to_expert() {
        assert_eq!(Role::default(), Role::Expert);
        assert_eq!(Role::parse("expert"), Some(Role::Expert));
        assert_eq!(Role::parse("LEADER"), Some(Role::Leader));
        assert_eq!(Role::parse("guru"), None);
        assert_eq!(CortexConfig::default().role, Role::Expert);
    }

    #[test]
    fn config_reads_role_pair() {
        let config = CortexConfig::from_pairs(&[
            ("role".to_string(), "leader".to_string()),
            ("unknown".to_string(), "ignored".to_string()),
        ]);
        assert_eq!(config.role, Role::Leader);
    }

    #[test]
    fn leader_registry_is_idempotent_and_routes_to_all() {
        let mut leader = Leader::new();
        leader.register("expert-a");
        leader.register("expert-b");
        leader.register("expert-a");
        assert_eq!(leader.experts(), &["expert-a", "expert-b"]);
        assert_eq!(leader.route(&[1, 2, 3]), vec!["expert-a", "expert-b"]);
    }

    #[test]
    fn leader_merge_is_the_lattice_join() {
        let leader = Leader::new();
        let merged = leader.merge(&[vec![1, 2, 3], vec![2, 3, 4], vec![]]);
        assert_eq!(merged, vec![1, 2, 3, 4]);
    }
}
