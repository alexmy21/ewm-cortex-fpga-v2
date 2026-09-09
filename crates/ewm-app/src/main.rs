//! `ewm-app` CLI — run the Noether loop against a token source.
//!
//! ```bash
//! # Deterministic stub (default): three scripted turns, in-memory store
//! cargo run -p ewm-app
//!
//! # Stub with an explicit script and a persistent store
//! cargo run -p ewm-app -- --stub "1,2,3;2,3,4" --repo /tmp/ewm-app-repo
//!
//! # Real local LLM (ollama deepseek-coder:6.7b), two turns, persistent store
//! cargo run -p ewm-app -- --ollama deepseek-coder:6.7b --turns 2 --repo /tmp/ewm-app-repo
//! ```
//!
//! Encoding: `tid{n}` (nanoLM/cortex inscription), explicit per contract.

use ewm_app::{CortexApp, OllamaLlm, StubLlm, TurnSource, APP_ENCODING_NAME};
use ewm_git::LooseStore;
use hllset_contracts::token::token_in_bytes;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut mode = "stub".to_string();
    let mut script = "1,2,3;2,3,4;3,4,5".to_string();
    let mut model = "deepseek-coder:6.7b".to_string();
    let mut prompt = "Reply with exactly four short technical words about content-addressed memory."
        .to_string();
    let mut turns = 3usize;
    let mut repo_path: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ollama" => {
                mode = "ollama".to_string();
                i += 1;
                if let Some(m) = args.get(i) {
                    model = m.clone();
                    i += 1;
                }
            }
            "--stub" => {
                mode = "stub".to_string();
                i += 1;
                if let Some(s) = args.get(i) {
                    script = s.clone();
                    i += 1;
                }
            }
            "--turns" => {
                i += 1;
                turns = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(turns);
                i += 1;
            }
            "--prompt" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    prompt = p.clone();
                    i += 1;
                }
            }
            "--repo" => {
                i += 1;
                repo_path = args.get(i).cloned();
                i += 1;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let mut source: Box<dyn TurnSource> = if mode == "ollama" {
        println!("token source : ollama run {model}");
        println!("prompt       : {prompt}");
        println!("turns        : {turns}");
        Box::new(OllamaLlm::new(model, prompt, turns))
    } else {
        let scripted = parse_script(&script);
        println!("token source : deterministic stub");
        println!("script       : {script}");
        println!("turns        : {}", scripted.len());
        Box::new(StubLlm::new(scripted))
    };
    println!("encoding     : {APP_ENCODING_NAME}");
    println!();

    // The store is the stack memory; the head is the pointer (§10/§11).
    if let Some(path) = repo_path {
        let store = LooseStore::new(&path);
        let mut app = CortexApp::open(store);
        println!("store        : {path} (recovered head: {})", head_or_none(&app));
        run_loop(&mut app, &mut *source);
    } else {
        let mut app = CortexApp::new(ewm_git::MemoryStore::default());
        println!("store        : memory");
        run_loop(&mut app, &mut *source);
    }
}

fn head_or_none<S: ewm_git::ObjectStore>(app: &CortexApp<S>) -> String {
    app.head()
        .map(|h| h.to_string())
        .unwrap_or_else(|| "<none>".to_string())
}

fn run_loop<S: ewm_git::ObjectStore>(app: &mut CortexApp<S>, source: &mut dyn TurnSource) {
    while let Some(turn) = source.next_turn() {
        if turn.is_empty() {
            println!("turn {:>3}: empty turn — skipped", app.turn_count());
            continue;
        }
        match app.run_turn(&turn) {
            Ok(outcome) => {
                let commit = outcome
                    .commit
                    .as_ref()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "no-change (idempotent skip)".to_string());
                let head = outcome
                    .head
                    .as_ref()
                    .map(|h| h.to_string())
                    .unwrap_or_else(|| "<none>".to_string());
                let view = outcome
                    .commit_view
                    .as_ref()
                    .map(|v| {
                        format!(
                            "D={} R={} N={}",
                            v.departed.popcount(),
                            v.retained.popcount(),
                            v.new.popcount()
                        )
                    })
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "turn {:>3}: ids={:?}",
                    app.turn_count() - 1,
                    turn.iter()
                        .map(|&n| String::from_utf8_lossy(&token_in_bytes(n)).into_owned())
                        .collect::<Vec<_>>()
                );
                println!(
                    "           commit={commit}\n           head={head}\n           tree leaves={} added={} removed={} retained={}\n           full_image_tokens={} bits({view})",
                    outcome.context.tree().leaves().len(),
                    outcome.diff.added.len(),
                    outcome.diff.removed.len(),
                    outcome.diff.retained.len(),
                    outcome.full_image.len(),
                );
            }
            Err(e) => {
                eprintln!("turn {}: {e}", app.turn_count());
                std::process::exit(1);
            }
        }
    }
    println!();
    println!("final head   : {}", head_or_none(app));
    println!("context size : {} bits (G1 lattice top)", app.repo().context_size());
    if let Some(cid) = app.head() {
        if let Ok(view) = ewm_git::view(app.repo(), cid) {
            let ok = view.departed.union(&view.retained).popcount() == view.parent_state.popcount()
                && view.retained.union(&view.new).popcount() == view.state.popcount()
                && view.departed.intersection(&view.new).popcount() == 0;
            println!(
                "Noether invariants (head, {}): {}",
                view.channel.name(),
                if ok { "hold" } else { "VIOLATED" }
            );
        }
    }
}

fn parse_script(script: &str) -> Vec<Vec<u32>> {
    script
        .split(';')
        .filter(|s| !s.trim().is_empty())
        .map(|turn| {
            turn.split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|id| id.trim().parse().expect("script ids must be u32"))
                .collect()
        })
        .collect()
}
