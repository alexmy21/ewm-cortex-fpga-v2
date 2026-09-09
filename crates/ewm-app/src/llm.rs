//! Token sources: the deterministic stub first, then the real local LLM.
//!
//! NEXT_SESSION §4.1: do not start with a real model. [`StubLlm`] replays a
//! scripted token stream (`Vec<Vec<TokenId>>` turns), which makes every
//! assertion reproducible. [`OllamaLlm`] is the first *real* candidate
//! (local DeepSeek coder), wired the same way: a fire-and-forget producer of
//! token turns that never blocks on the cortex.

use cortex_core::TokenId;
use std::collections::HashMap;
use std::process::Command;

/// A producer of token turns. The LLM never waits on the cortex: this trait
/// has no knowledge of the side-car, the store, or the loop.
pub trait TurnSource {
    /// The next turn's token ids, or `None` when the source is exhausted.
    fn next_turn(&mut self) -> Option<Vec<TokenId>>;
}

/// A deterministic, scripted LLM stub (NEXT_SESSION §4.1).
#[derive(Clone, Debug, Default)]
pub struct StubLlm {
    turns: Vec<Vec<TokenId>>,
    pos: usize,
}

impl StubLlm {
    pub fn new(turns: Vec<Vec<TokenId>>) -> Self {
        Self { turns, pos: 0 }
    }

    /// Build a stub from a compact script: `&[&[a, b], &[c]]` = two turns.
    pub fn with_script(script: &[&[TokenId]]) -> Self {
        Self::new(script.iter().map(|turn| turn.to_vec()).collect())
    }

    /// Number of scripted turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

impl TurnSource for StubLlm {
    fn next_turn(&mut self) -> Option<Vec<TokenId>> {
        let turn = self.turns.get(self.pos).cloned();
        self.pos += 1;
        turn
    }
}

/// A real local LLM through the `ollama` CLI.
///
/// The response text is split into whitespace-delimited tokens; each distinct
/// token string is assigned the next free `TokenId` (session-local,
/// deterministic in the order of first appearance). The app path then uses
/// the `tid{n}` inscription over those ids — the same vocabulary contract as
/// the stub.
#[derive(Clone, Debug)]
pub struct OllamaLlm {
    model: String,
    prompt: String,
    max_turns: usize,
    turn: usize,
    vocab: HashMap<String, TokenId>,
    next_id: TokenId,
}

impl OllamaLlm {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>, max_turns: usize) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            max_turns,
            turn: 0,
            vocab: HashMap::new(),
            next_id: 0,
        }
    }

    /// Map a response text to token ids, growing the session vocabulary.
    pub fn tokenize(&mut self, text: &str) -> Vec<TokenId> {
        text.split_whitespace()
            .map(|word| {
                let next = &mut self.next_id;
                *self.vocab.entry(word.to_string()).or_insert_with(|| {
                    let id = *next;
                    *next += 1;
                    id
                })
            })
            .collect()
    }

    /// Number of distinct token strings seen so far (the session vocab size).
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// One completion through the `ollama` CLI (fire-and-forget: this never
    /// touches the cortex).
    pub fn complete(&self) -> Option<String> {
        let output = Command::new("ollama")
            .args(["run", &self.model, &self.prompt])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

impl TurnSource for OllamaLlm {
    fn next_turn(&mut self) -> Option<Vec<TokenId>> {
        if self.turn >= self.max_turns {
            return None;
        }
        self.turn += 1;
        let text = self.complete()?;
        if text.is_empty() {
            return None;
        }
        Some(self.tokenize(&text))
    }
}
