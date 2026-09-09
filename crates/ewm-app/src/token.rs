//! Token encodings — both first-class (NEXT_SESSION §3.7).
//!
//! Every integration test must say which encoding it uses. The app path uses
//! [`TokenEncoding::Tid`] (`tid{n}`, the nanoLM/cortex inscription); the
//! notebook-08 4-byte LE encoding is exercised by the explicit round-trip
//! test and is available here for pipelines that choose it.

use hllset_contracts::token::{
    parse_token_id, parse_token_id_le, token_in_bytes, token_in_bytes_le, TokenId,
};

/// The two soldered token inscriptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenEncoding {
    /// `tid{n}` — the nanoLM/cortex-side inscription (app default).
    Tid,
    /// 4-byte little-endian — the notebook-08 inscription.
    Le,
}

impl TokenEncoding {
    pub const fn name(self) -> &'static str {
        match self {
            TokenEncoding::Tid => "tid",
            TokenEncoding::Le => "le",
        }
    }

    pub fn encode(self, id: TokenId) -> Vec<u8> {
        match self {
            TokenEncoding::Tid => token_in_bytes(id),
            TokenEncoding::Le => token_in_bytes_le(id).to_vec(),
        }
    }

    pub fn parse(self, bytes: &[u8]) -> Option<TokenId> {
        match self {
            TokenEncoding::Tid => parse_token_id(bytes),
            TokenEncoding::Le => parse_token_id_le(bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tid_and_le_roundtrip_explicitly() {
        // Explicit: this test uses BOTH encodings, one assertion each.
        let tid = TokenEncoding::Tid;
        assert_eq!(tid.encode(44), b"tid44");
        assert_eq!(tid.parse(b"tid44"), Some(44));

        let le = TokenEncoding::Le;
        assert_eq!(le.encode(44), [44, 0, 0, 0]);
        assert_eq!(le.parse(&[44, 0, 0, 0]), Some(44));

        // Cross-parse fails by construction: neither inscription is the other.
        assert_eq!(tid.parse(&[44, 0, 0, 0]), None);
        assert_eq!(le.parse(b"tid44"), None);
    }
}
