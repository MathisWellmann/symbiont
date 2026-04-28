// SPDX-License-Identifier: MPL-2.0
//! Game AI evolution: the LLM learns to play Tic-Tac-Toe.
//!
//! Declares an evolvable `choose_move` function that picks a move given
//! a board state. The harness plays the evolved AI against opponents of
//! varying strength, feeds win/loss/draw statistics back into the prompt,
//! and lets the constrained-generation loop converge on a strong player.
//!
//! This showcases symbiont's ability to evolve **strategic reasoning**
//! through code: the LLM must discover game-theoretic concepts like
//! center control, fork creation, and blocking — expressed as compiled
//! Rust.

use std::fmt::Write;

use romu::Rng;
use symbiont::Runtime;
use tracing::{
    info,
    warn,
};
use tracing_subscriber::EnvFilter;

/// Number of games played per opponent per evaluation round.
const GAMES_PER_OPPONENT: usize = 100;

// Board encoding:
//   0 = empty, 1 = X, 2 = O
//
// Board layout (index positions):
//   0 | 1 | 2
//   ---------
//   3 | 4 | 5
//   ---------
//   6 | 7 | 8
//
// The function receives the board state, board length (always 9),
// and which player it is (1 or 2). Returns an index 0–8 of an empty
// cell to place its mark.
//
// Default: picks the first empty cell — a terrible strategy.
symbiont::evolvable! {
    fn choose_move(board: &[u8], len: usize, player: u8) -> usize {
        let _ = player;
        for i in 0..len {
            if board[i] == 0 {
                return i;
            }
        }
        0
    }
}

// -- Win detection -----------------------------------------------------------

const WIN_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8], // rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8], // cols
    [0, 4, 8],
    [2, 4, 6], // diagonals
];

fn has_won(board: &[u8; 9], player: u8) -> bool {
    WIN_LINES
        .iter()
        .any(|line| line.iter().all(|&i| board[i] == player))
}

fn is_full(board: &[u8; 9]) -> bool {
    board.iter().all(|&c| c != 0)
}

fn is_valid_move(board: &[u8; 9], pos: usize) -> bool {
    pos < 9 && board[pos] == 0
}

fn empty_cells(board: &[u8; 9]) -> Vec<usize> {
    (0..9).filter(|&i| board[i] == 0).collect()
}

// -- Opponents ---------------------------------------------------------------

/// Random opponent: picks a uniformly random empty cell.
fn random_move(board: &[u8; 9], rng: &Rng) -> usize {
    let empty = empty_cells(board);
    if empty.is_empty() {
        return 0;
    }
    empty[rng.mod_usize(empty.len())]
}

/// Smart opponent: takes immediate wins, blocks opponent wins, prefers
/// center, then corners, then edges. Falls back to random.
fn smart_move(board: &[u8; 9], player: u8, rng: &Rng) -> usize {
    let opponent = if player == 1 { 2 } else { 1 };

    // Take a win if available.
    for &pos in &empty_cells(board) {
        let mut b = *board;
        b[pos] = player;
        if has_won(&b, player) {
            return pos;
        }
    }

    // Block opponent's win.
    for &pos in &empty_cells(board) {
        let mut b = *board;
        b[pos] = opponent;
        if has_won(&b, opponent) {
            return pos;
        }
    }

    // Prefer center.
    if board[4] == 0 {
        return 4;
    }

    // Prefer corners.
    let corners = [0, 2, 6, 8];
    let free_corners: Vec<usize> = corners.iter().copied().filter(|&c| board[c] == 0).collect();
    if !free_corners.is_empty() {
        return free_corners[rng.mod_usize(free_corners.len())];
    }

    // Otherwise random.
    random_move(board, rng)
}

/// Minimax opponent: perfect play. Always draws or wins.
fn minimax_move(board: &[u8; 9], player: u8) -> usize {
    let mut best_score = i32::MIN;
    let mut best_pos = 0;

    for pos in empty_cells(board) {
        let mut b = *board;
        b[pos] = player;
        let opponent = if player == 1 { 2 } else { 1 };
        let score = minimax(&b, opponent, player, false);
        if score > best_score {
            best_score = score;
            best_pos = pos;
        }
    }
    best_pos
}

fn minimax(board: &[u8; 9], current: u8, maximizer: u8, is_maximizing: bool) -> i32 {
    let minimizer = if maximizer == 1 { 2 } else { 1 };

    if has_won(board, maximizer) {
        return 10;
    }
    if has_won(board, minimizer) {
        return -10;
    }
    if is_full(board) {
        return 0;
    }

    let next = if current == 1 { 2 } else { 1 };

    if is_maximizing {
        let mut best = i32::MIN;
        for pos in empty_cells(board) {
            let mut b = *board;
            b[pos] = current;
            best = best.max(minimax(&b, next, maximizer, false));
        }
        best
    } else {
        let mut best = i32::MAX;
        for pos in empty_cells(board) {
            let mut b = *board;
            b[pos] = current;
            best = best.min(minimax(&b, next, maximizer, true));
        }
        best
    }
}

// -- Game play ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum GameResult {
    Win,
    Loss,
    Draw,
    Forfeit,
}

/// Play one game. The AI uses the evolvable `choose_move`; the opponent
/// uses the provided closure. `ai_player` is 1 (X) or 2 (O).
fn play_game(
    runtime: &Runtime,
    ai_player: u8,
    opponent: &dyn Fn(&[u8; 9], u8) -> usize,
) -> GameResult {
    let mut board = [0u8; 9];
    let mut current = 1u8; // X always moves first.

    for _ in 0..9 {
        let pos = if current == ai_player {
            // AI's turn — catch panics so a bad evolution doesn't crash us.
            let m = choose_move(&board, 9, current);
            if runtime.take_panic().is_some() {
                return GameResult::Forfeit;
            };
            if !is_valid_move(&board, m) {
                return GameResult::Forfeit;
            }
            m
        } else {
            opponent(&board, current)
        };

        board[pos] = current;

        if has_won(&board, current) {
            return if current == ai_player {
                GameResult::Win
            } else {
                GameResult::Loss
            };
        }
        if is_full(&board) {
            return GameResult::Draw;
        }

        current = if current == 1 { 2 } else { 1 };
    }

    GameResult::Draw
}

// -- Evaluation --------------------------------------------------------------

struct MatchResult {
    opponent_name: &'static str,
    wins: usize,
    losses: usize,
    draws: usize,
    forfeits: usize,
    games: usize,
}

impl MatchResult {
    /// 1.0 = win, 0.5 = draw, 0.0 = loss/forfeit.
    fn score(&self) -> f64 {
        (self.wins as f64 + 0.5 * self.draws as f64) / self.games as f64
    }
}

fn evaluate(runtime: &Runtime) -> Vec<MatchResult> {
    let rng = Rng::from_seed_with_64bit(42);

    type OpponentFn = dyn Fn(&[u8; 9], u8, &Rng) -> usize;
    let opponents: &[(&str, &OpponentFn)] = &[
        ("random", &|b, _p, r| random_move(b, r)),
        ("smart", &|b, p, r| smart_move(b, p, r)),
        ("minimax", &|b, p, _r| minimax_move(b, p)),
    ];

    let mut results = Vec::new();
    for &(name, opp_fn) in opponents {
        let (mut wins, mut losses, mut draws, mut forfeits) = (0, 0, 0, 0);

        for game_idx in 0..GAMES_PER_OPPONENT {
            // Alternate playing as X (first mover) and O.
            let ai_player = if game_idx % 2 == 0 { 1 } else { 2 };
            let opponent = |board: &[u8; 9], player: u8| opp_fn(board, player, &rng);
            match play_game(runtime, ai_player, &opponent) {
                GameResult::Win => wins += 1,
                GameResult::Loss => losses += 1,
                GameResult::Draw => draws += 1,
                GameResult::Forfeit => forfeits += 1,
            }
        }

        results.push(MatchResult {
            opponent_name: name,
            wins,
            losses,
            draws,
            forfeits,
            games: GAMES_PER_OPPONENT,
        });
    }
    results
}

// -- Reporting ---------------------------------------------------------------

fn format_report(results: &[MatchResult]) -> String {
    let mut report = String::from(
        "| Opponent | Wins | Losses | Draws | Forfeits | Score |\n\
         |----------|------|--------|-------|----------|-------|\n",
    );

    for r in results {
        let _ = writeln!(
            report,
            "| {:<8} | {:>4} | {:>6} | {:>5} | {:>8} | {:>4.0}% |",
            r.opponent_name,
            r.wins,
            r.losses,
            r.draws,
            r.forfeits,
            r.score() * 100.0,
        );
    }

    let overall = results.iter().map(|r| r.score()).sum::<f64>() / results.len() as f64;
    let _ = write!(report, "\nOverall score: {:.0}%\n", overall * 100.0);

    if results.iter().any(|r| r.forfeits > 0) {
        report.push_str(
            "WARNING: Some games were forfeited (invalid move or panic). Fix correctness first.\n",
        );
    }

    report
}

fn overall_score(results: &[MatchResult]) -> f64 {
    results.iter().map(|r| r.score()).sum::<f64>() / results.len() as f64
}

// -- Main --------------------------------------------------------------------

#[tokio::main]
async fn main() -> symbiont::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_line_number(true)
        .init();

    let runtime = Runtime::init(SYMBIONT_DECLS, symbiont::Profile::Debug).await?;
    let fn_sigs = runtime.fn_sigs();
    info!("fn_sigs: {fn_sigs:?}");

    let agent = symbiont::inference::init_agent()?;

    // -- Round 0: evaluate the default (first-empty-cell) strategy -----------
    println!("\n=== Round 0: default implementation (first empty cell) ===");
    let mut results = evaluate(runtime);
    let mut report = format_report(&results);
    println!("{report}");

    // Track the best generated code across all rounds.
    let mut best_score = overall_score(&results);
    let mut best_code: Option<String> = None;

    // -- Evolution loop -------------------------------------------------------
    let max_rounds = 5;

    for round in 1..=max_rounds {
        println!("\n=== Round {round}: evolving via LLM ===");

        let prompt = format!(
            "Implement this Tic-Tac-Toe AI:\n\
             ```\n{sig}\n```\n\
             board: 9-element slice, 0=empty 1=X 2=O. Layout: 0|1|2 / 3|4|5 / 6|7|8.\n\
             player: your mark (1 or 2). Return index 0-8 of an empty cell.\n\
             Invalid move or panic = forfeit.\n\n\
             Results ({GAMES_PER_OPPONENT} games/opponent):\n{report}\n\
             Take wins, block losses, control center, create forks. Code only.",
            sig = fn_sigs[0],
        );

        info!("Evolution prompt:\n{prompt}");

        if let Err(e) = runtime.evolve(&agent, &prompt).await {
            warn!("Evolution failed: {e} — retrying next round.");
            continue;
        }

        results = evaluate(runtime);
        report = format_report(&results);
        println!("{report}");

        let score = overall_score(&results);

        // Update best code if this round improved.
        if score > best_score {
            best_score = score;
            best_code = runtime.read_clean_code().ok();
            info!("New best score: {:.0}%", best_score * 100.0);
        }

        if score >= 0.90 {
            println!(
                "Achieved {:.0}% overall score after {round} evolution round(s)!",
                score * 100.0
            );
            break;
        }

        warn!("Score: {:.0}% — continuing evolution.", score * 100.0);
    }

    // -- Print best generated code --------------------------------------------
    println!(
        "\n=== Best generated code (score: {:.0}%) ===",
        best_score * 100.0
    );
    match best_code {
        Some(code) => println!("{code}"),
        None => println!("(no evolution improved on the default implementation)"),
    }

    let final_score = overall_score(&results);
    println!("\nFinal score: {:.0}%", final_score * 100.0);
    Ok(())
}
