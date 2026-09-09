//! The grounding verdict (nanoLM §3.6/§7.2 — ported to the bridge).
//!
//! Between the measured context (`A`, the [`ContextMatrix`] nodes) and the LLM
//! response (`B`, the output ids):
//!
//! - `tau` — coverage, `|A ∩ B| / |B|`;
//! - `rho` — novelty/departure, `|B \ A| / |B|` (the ungrounded fraction);
//! - `r_link` — retained links: response bigrams present in the measured
//!   transition graph (the token-space R-link);
//! - `srho` — structural ρ in HLLSet space (BSS). **Backend-supplied**; the
//!   token-space bridge reserves the field and reports `0.0` until the
//!   simulator/physical backend fills it.
//! - `grounded` — `tau >= tau_min && rho <= rho_max`;
//! - `flagged` — the ungrounded ids (one-sided evidence).
//!
//! Grounding is a **recommendation**, never a mutation: `recommend` is
//! read-only over the matrix.

use crate::context::ContextMatrix;
use crate::TokenId;

/// Grounding thresholds.
#[derive(Debug, Clone, Copy)]
pub struct GroundingConfig {
    pub tau_min: f64,
    pub rho_max: f64,
}

impl Default for GroundingConfig {
    fn default() -> Self {
        Self {
            tau_min: 0.8,
            rho_max: 0.2,
        }
    }
}

/// The grounding verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundingReport {
    pub tau: f64,
    pub rho: f64,
    pub srho: f64,
    pub r_link: u64,
    pub grounded: bool,
    pub flagged: Vec<TokenId>,
}

impl Default for GroundingReport {
    fn default() -> Self {
        Self {
            tau: 0.0,
            rho: 0.0,
            srho: 0.0,
            r_link: 0,
            grounded: false,
            flagged: Vec::new(),
        }
    }
}

/// Ground `outputs` against a measured context: `membership` decides which
/// ids are measured (e.g. the lattice's vocabulary, or the matrix nodes),
/// `matrix` supplies the measured transition graph for the R-link. Read-only.
pub fn recommend_with(
    membership: impl Fn(TokenId) -> bool,
    matrix: &ContextMatrix,
    outputs: &[TokenId],
    config: &GroundingConfig,
) -> GroundingReport {
    let n = outputs.len();
    let in_context = outputs.iter().filter(|&&t| membership(t)).count();

    let tau = if n > 0 {
        in_context as f64 / n as f64
    } else {
        1.0
    };
    let rho = if n > 0 {
        (n - in_context) as f64 / n as f64
    } else {
        0.0
    };

    let flagged: Vec<TokenId> = outputs
        .iter()
        .copied()
        .filter(|&t| !membership(t))
        .collect();

    // Token-space R-link: response bigrams whose transition was measured
    // *before* this response (retention against the prior context).
    let r_link = outputs
        .windows(2)
        .filter(|w| matrix.cell(w[0], w[1]).unwrap_or(0) > 0)
        .count() as u64;

    // Structural ρ (BSS over HLLSet space) is backend-supplied; the bridge
    // reserves the wire field and reports 0.0.
    let srho = 0.0;

    let grounded = tau >= config.tau_min && rho <= config.rho_max;

    GroundingReport {
        tau,
        rho,
        srho,
        r_link,
        grounded,
        flagged,
    }
}

/// Ground `outputs` against the matrix nodes as the measured context — the
/// nanoLM §3.6/§7.2 semantics. Read-only.
pub fn recommend(
    matrix: &ContextMatrix,
    outputs: &[TokenId],
    config: &GroundingConfig,
) -> GroundingReport {
    recommend_with(|t| matrix.grounded(t), matrix, outputs, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> ContextMatrix {
        ContextMatrix::from_sequence(&[100, 101, 102, 103, 104])
    }

    #[test]
    fn fully_grounded_response_scores_clean() {
        let m = matrix();
        let r = recommend(&m, &[100, 101, 102], &GroundingConfig::default());
        assert!(r.grounded);
        assert!((r.tau - 1.0).abs() < 1e-9);
        assert_eq!(r.rho, 0.0);
        assert!(r.flagged.is_empty());
        assert_eq!(r.r_link, 2, "both response bigrams were measured");
    }

    #[test]
    fn novel_id_is_flagged_and_fails_grounding() {
        let m = matrix();
        let r = recommend(&m, &[100, 999], &GroundingConfig::default());
        assert!(!r.grounded);
        assert!((r.tau - 0.5).abs() < 1e-9);
        assert!(r.flagged.contains(&999));
        assert_eq!(r.r_link, 0, "unmeasured transition is not a retained link");
    }

    #[test]
    fn empty_output_is_vacuously_grounded() {
        let m = matrix();
        let r = recommend(&m, &[], &GroundingConfig::default());
        assert_eq!(r.tau, 1.0);
        assert_eq!(r.rho, 0.0);
        assert!(r.grounded);
    }
}
