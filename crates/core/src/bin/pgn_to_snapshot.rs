use anyhow::{Context, Result, anyhow, bail};
use chess::{ALL_SQUARES, Board, ChessMove, Color, Piece, Square};
use clap::Parser;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use w1z4rdv1510n::schema::{
    DynamicState, EnvironmentSnapshot, Position, Symbol, SymbolState, SymbolType, Timestamp,
};

/// Convert a PGN file into an EnvironmentSnapshot with stack_history frames.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input PGN file (single or multiple games; only the first game is used unless --game-index set)
    #[arg(long)]
    pgn: PathBuf,
    /// Output snapshot json
    #[arg(long)]
    out: PathBuf,
    /// Which game (0-based) to extract from the PGN if multiple games exist
    #[arg(long, default_value_t = 0)]
    game_index: usize,
    /// Limit number of plies to emit (0 = all)
    #[arg(long, default_value_t = 0)]
    max_plies: usize,
}

fn square_to_xy(sq: Square) -> (f64, f64) {
    // a1 = (0,0), h8 = (7,7)
    let file = (sq.get_file().to_index()) as f64;
    let rank = (sq.get_rank().to_index()) as f64;
    (file, rank)
}

fn piece_id(color: Color, piece: Piece, sq: Square) -> String {
    let c = match color {
        Color::White => "white",
        Color::Black => "black",
    };
    let p = match piece {
        Piece::Pawn => "P",
        Piece::Knight => "N",
        Piece::Bishop => "B",
        Piece::Rook => "R",
        Piece::Queen => "Q",
        Piece::King => "K",
    };
    format!("{}_{}_{}", c, p, sq)
}

fn board_to_dynamic_state(board: &Board, ply: usize) -> DynamicState {
    let mut symbol_states = HashMap::new();
    for &sq in ALL_SQUARES.iter() {
        if let Some(piece) = board.piece_on(sq) {
            if let Some(color) = board.color_on(sq) {
                let id = piece_id(color, piece, sq);
                let (x, y) = square_to_xy(sq);
                let mut internal = HashMap::new();
                internal.insert("piece".to_string(), json!(format!("{:?}", piece)));
                internal.insert("color".to_string(), json!(format!("{:?}", color)));
                internal.insert("square".to_string(), json!(sq.to_string()));
                symbol_states.insert(
                    id,
                    SymbolState {
                        position: Position { x, y, z: 0.0 },
                        velocity: None,
                        internal_state: internal,
                    },
                );
            }
        }
    }
    DynamicState {
        timestamp: Timestamp { unix: ply as i64 },
        symbol_states,
    }
}

fn board_to_symbols(board: &Board) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    for &sq in ALL_SQUARES.iter() {
        if let Some(piece) = board.piece_on(sq) {
            if let Some(color) = board.color_on(sq) {
                let id = piece_id(color, piece, sq);
                let (x, y) = square_to_xy(sq);
                let mut props = HashMap::new();
                props.insert("piece".to_string(), json!(format!("{:?}", piece)));
                props.insert("color".to_string(), json!(format!("{:?}", color)));
                props.insert("square".to_string(), json!(sq.to_string()));
                props.insert("radius".to_string(), json!(0.45));
                symbols.push(Symbol {
                    id,
                    symbol_type: SymbolType::Custom,
                    position: Position { x, y, z: 0.0 },
                    properties: props,
                });
            }
        }
    }
    symbols
}

fn parse_game(pgn_text: &str, game_idx: usize) -> Result<Vec<String>> {
    // Lightweight: split games when a new [Event starts].
    let mut games: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in pgn_text.lines() {
        if line.starts_with("[Event") && !current.is_empty() {
            games.push(current.clone());
            current.clear();
        }
        if !line.trim().is_empty() {
            current.push_str(line);
            current.push('\n');
        }
    }
    if !current.is_empty() {
        games.push(current);
    }
    let game_str = games
        .get(game_idx)
        .ok_or_else(|| anyhow!("game index {} not found in PGN", game_idx))?;

    // Strip tag pairs and join remaining lines
    let mut moves_section = String::new();
    for line in game_str.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            continue;
        }
        if !trimmed.is_empty() {
            moves_section.push_str(trimmed);
            moves_section.push(' ');
        }
    }

    let mut moves = Vec::new();
    for tok in moves_section.split_whitespace() {
        if tok.starts_with('{') || tok.starts_with('(') {
            continue;
        }
        let clean = tok
            .trim_end_matches(|c: char| c == '!' || c == '?')
            .trim_end_matches('+')
            .trim_end_matches('#');

        if ["1-0", "0-1", "1/2-1/2", "*"].contains(&clean) {
            break;
        }
        // tokens like "12...Qd7" or "1.e4"
        let san = if clean.contains("...") {
            clean.split("...").last().unwrap_or("")
        } else if clean.contains('.') {
            clean.rsplit('.').next().unwrap_or("")
        } else {
            clean
        };
        if san.is_empty() {
            continue;
        }
        // skip pure move numbers
        if san.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        moves.push(san.to_string());
    }
    Ok(moves)
}

fn apply_moves(moves: &[String], max_plies: usize) -> Result<Vec<DynamicState>> {
    let mut board = Board::default();
    let mut frames = Vec::new();
    frames.push(board_to_dynamic_state(&board, 0));

    let limit = if max_plies == 0 {
        moves.len()
    } else {
        max_plies.min(moves.len())
    };
    for (i, san) in moves.iter().take(limit).enumerate() {
        let mv = match ChessMove::from_san(&board, san) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Skipping invalid SAN '{}' at ply {}: {:?}", san, i + 1, e);
                break;
            }
        };
        board = board.make_move_new(mv);
        frames.push(board_to_dynamic_state(&board, i + 1));
    }
    Ok(frames)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes =
        fs::read(&args.pgn).with_context(|| format!("failed to read PGN {:?}", args.pgn))?;
    let pgn_text = String::from_utf8_lossy(&bytes).to_string();
    let moves = parse_game(&pgn_text, args.game_index)?;
    if moves.is_empty() {
        bail!("no moves parsed from PGN");
    }
    let frames = apply_moves(&moves, args.max_plies)?;

    // symbols from the starting position
    let start_board = Board::default();
    let symbols = board_to_symbols(&start_board);

    let mut bounds = HashMap::new();
    bounds.insert("width".to_string(), 8.0);
    bounds.insert("height".to_string(), 8.0);
    bounds.insert("depth".to_string(), 1.0);

    let mut metadata = HashMap::new();
    metadata.insert("domain".to_string(), json!("chess"));
    metadata.insert("source_pgn".to_string(), json!(args.pgn.to_string_lossy()));

    let snapshot = EnvironmentSnapshot {
        timestamp: Timestamp { unix: 0 },
        bounds,
        symbols,
        metadata,
        stack_history: frames,
    };

    let data = serde_json::to_string_pretty(&snapshot)?;
    fs::write(&args.out, data).with_context(|| format!("failed to write {:?}", args.out))?;
    println!(
        "Wrote snapshot with {} frames to {:?}",
        snapshot.stack_history.len(),
        args.out
    );
    Ok(())
}
