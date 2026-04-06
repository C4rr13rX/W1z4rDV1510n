#!/usr/bin/env python3
"""
Live Chess Prediction Visualizer.

Serves two modes depending on which JSON files are present:
  - CHESS MODE  (default): reads logs/chess_live_board.json written by
    chess_prediction_runner.py.  Shows an animated chess board with probability
    heat-map overlay, move prediction highlights, and accuracy metrics.
  - ANNEALER MODE: reads logs/live_frame.json + logs/live_neuro.json written
    by the Rust annealer (fallback when chess_live_board.json is absent).

Usage:
  python scripts/live_viz_server.py --open
  python scripts/live_viz_server.py --board-file logs/chess_live_board.json --port 8765 --open
"""
from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

# ── Colour palette constants used by the browser-side plasma shader ──────────
# (passed as JS config; not rendered server-side)
PLASMA_STOPS = ["#0d0221", "#190a4a", "#3d0c7a", "#7b1fa2", "#c2185b",
                "#ff5722", "#ff9800", "#ffeb3b", "#ffffff"]

HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>W1z4rd — Chess Prediction Visualizer</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0;
      width: 100%; height: 100%;
      background: #080012;
      color: #c8d6ff;
      font-family: 'Consolas', 'Courier New', monospace;
      overflow: hidden;
    }

    /* ── Layout ──────────────────────────────────────────────────────── */
    #root {
      display: grid;
      grid-template-columns: 1fr 280px;
      grid-template-rows: 48px 1fr 100px;
      height: 100vh;
      gap: 0;
    }

    /* ── Top bar ─────────────────────────────────────────────────────── */
    #topbar {
      grid-column: 1 / -1;
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(0,0,0,0.6);
      border-bottom: 1px solid #2a1060;
      padding: 0 16px;
    }
    #title { color: #c084fc; font-size: 15px; letter-spacing: 2px; }
    #hud   { font-size: 13px; color: #7dd3fc; }
    #accuracy-bar { font-size: 12px; color: #4ade80; }

    /* ── Board area ──────────────────────────────────────────────────── */
    #board-wrap {
      position: relative;
      display: flex; align-items: center; justify-content: center;
      overflow: hidden;
    }
    #board-canvas { position: absolute; top: 0; left: 0; }

    /* ── Right panel ─────────────────────────────────────────────────── */
    #side-panel {
      background: rgba(0,0,0,0.55);
      border-left: 1px solid #2a1060;
      padding: 12px 10px;
      display: flex; flex-direction: column; gap: 10px;
      overflow-y: auto;
    }
    .panel-section { border-bottom: 1px solid #1a0840; padding-bottom: 8px; }
    .panel-label { font-size: 10px; color: #a78bfa; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }

    /* Outcome bars */
    .outcome-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; font-size: 12px; }
    .outcome-bar-bg { flex: 1; height: 10px; background: #1a0a30; border-radius: 2px; overflow: hidden; }
    .outcome-bar-fill { height: 100%; border-radius: 2px; transition: width 0.4s ease; }
    .bar-white  { background: linear-gradient(90deg, #7dd3fc, #e0f2fe); }
    .bar-draw   { background: linear-gradient(90deg, #a78bfa, #c4b5fd); }
    .bar-black  { background: linear-gradient(90deg, #f97316, #fbbf24); }

    /* Prediction list */
    .pred-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 3px 6px; border-radius: 3px; font-size: 12px;
      transition: background 0.3s;
    }
    .pred-row.correct  { background: rgba(74,222,128,0.18); color: #4ade80; }
    .pred-row.wrong    { background: rgba(248,113,113,0.10); color: #aaa; }
    .pred-san  { font-weight: bold; min-width: 48px; }
    .pred-prob { color: #7dd3fc; font-size: 11px; }

    /* Model breakdown */
    .model-row {
      display: grid; grid-template-columns: 52px 42px 1fr 46px;
      align-items: center; gap: 4px;
      padding: 3px 4px; border-radius: 3px; font-size: 11px;
      margin: 2px 0; transition: background 0.3s;
    }
    .model-row.correct { background: rgba(74,222,128,0.14); color: #4ade80; }
    .model-row.wrong   { background: rgba(248,113,113,0.08); color: #aaa; }
    .model-name  { font-size: 10px; text-transform: uppercase; color: #a78bfa; }
    .model-move  { font-weight: bold; }
    .energy-bar-bg  { height: 6px; background: #1a0a30; border-radius: 3px; overflow: hidden; }
    .energy-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease;
                       background: linear-gradient(90deg, #7c3aed, #e879f9); }
    .model-pct   { text-align: right; color: #e879f9; font-size: 10px; }

    /* Accuracy metrics */
    .metric-row { display: flex; justify-content: space-between; font-size: 12px; margin: 2px 0; }
    .metric-val { color: #4ade80; }
    .metric-val.dim { color: #facc15; }

    /* Flash for correct/wrong */
    #flash-overlay {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
      pointer-events: none;
      border-radius: 0;
      opacity: 0;
      transition: opacity 0.15s;
    }
    #flash-overlay.green { background: rgba(74,222,128,0.12); }
    #flash-overlay.red   { background: rgba(248,113,113,0.12); }

    /* ── Bottom console ──────────────────────────────────────────────── */
    #console {
      grid-column: 1 / -1;
      background: rgba(0,0,0,0.7);
      border-top: 1px solid #2a1060;
      padding: 6px 14px;
      font-size: 11px;
      color: #7dd3fc;
      overflow-y: auto;
      display: flex; flex-direction: column-reverse;
    }
    .log-line { line-height: 1.5; white-space: pre; }

    /* Tick counter */
    #tick { color: #6b21a8; font-size: 11px; }
  </style>
</head>
<body>
<div id="root">

  <!-- Top bar -->
  <div id="topbar">
    <div id="title">W1Z4RD :: CHESS PREDICTION FABRIC</div>
    <div id="hud">Connecting…</div>
    <div id="accuracy-bar">─</div>
    <div id="tick">tick 0</div>
  </div>

  <!-- Board canvas -->
  <div id="board-wrap">
    <canvas id="board-canvas"></canvas>
    <div id="flash-overlay"></div>
  </div>

  <!-- Right panel -->
  <div id="side-panel">

    <div class="panel-section">
      <div class="panel-label">Outcome Probability</div>
      <div class="outcome-row">
        <span style="width:46px;font-size:11px">White</span>
        <div class="outcome-bar-bg"><div id="bar-white" class="outcome-bar-fill bar-white" style="width:33%"></div></div>
        <span id="pct-white" style="font-size:11px;width:34px;text-align:right">33%</span>
      </div>
      <div class="outcome-row">
        <span style="width:46px;font-size:11px">Draw</span>
        <div class="outcome-bar-bg"><div id="bar-draw" class="outcome-bar-fill bar-draw" style="width:33%"></div></div>
        <span id="pct-draw" style="font-size:11px;width:34px;text-align:right">33%</span>
      </div>
      <div class="outcome-row">
        <span style="width:46px;font-size:11px">Black</span>
        <div class="outcome-bar-bg"><div id="bar-black" class="outcome-bar-fill bar-black" style="width:33%"></div></div>
        <span id="pct-black" style="font-size:11px;width:34px;text-align:right">33%</span>
      </div>
    </div>

    <div class="panel-section">
      <div class="panel-label">Top Predicted Moves</div>
      <div id="pred-list"></div>
    </div>

    <div class="panel-section">
      <div class="panel-label">Model Breakdown</div>
      <div id="model-breakdown"></div>
      <div id="neuro-stats" style="font-size:10px;color:#6b21a8;margin-top:4px"></div>
    </div>

    <div class="panel-section">
      <div class="panel-label">Running Accuracy</div>
      <div class="metric-row"><span>Move top-1</span><span id="m-top1" class="metric-val">—</span></div>
      <div class="metric-row"><span>Move top-3</span><span id="m-top3" class="metric-val dim">—</span></div>
      <div class="metric-row"><span>Outcome</span><span id="m-out" class="metric-val dim">—</span></div>
      <div class="metric-row"><span>Plies seen</span><span id="m-plies" class="metric-val dim">—</span></div>
    </div>

    <div class="panel-section">
      <div class="panel-label">Game Info</div>
      <div style="font-size:11px;line-height:1.8">
        <div id="g-id" style="color:#a78bfa">—</div>
        <div id="g-ply">ply — / —</div>
        <div id="g-side">to move: —</div>
        <div id="g-count">game #—</div>
      </div>
    </div>

    <div>
      <div class="panel-label">Hourly Checkpoints</div>
      <div id="hourly-list" style="font-size:11px;color:#888;line-height:1.6">—</div>
    </div>

  </div>

  <!-- Console -->
  <div id="console" id="console"></div>

</div>

<script>
// ── Configuration ────────────────────────────────────────────────────────────
const BOARD_URL   = '/board';
const FRAME_URL   = '/frame';
const NEURO_URL   = '/neuro';
const POLL_MS     = 300;

// Plasma colour ramp (cool → hot)
const PLASMA = [
  [13, 2, 33],
  [25, 10, 74],
  [61, 12, 122],
  [123, 31, 162],
  [194, 24, 91],
  [255, 87, 34],
  [255, 152, 0],
  [255, 235, 59],
  [255, 255, 255],
];

function plasmaColor(t) {
  // t in [0..1]
  const n = PLASMA.length - 1;
  const idx = Math.min(n - 1, Math.floor(t * n));
  const frac = t * n - idx;
  const a = PLASMA[idx], b = PLASMA[idx + 1];
  const r = Math.round(a[0] + frac * (b[0] - a[0]));
  const g = Math.round(a[1] + frac * (b[1] - a[1]));
  const bl = Math.round(a[2] + frac * (b[2] - a[2]));
  return `rgb(${r},${g},${bl})`;
}

// ── DOM refs ─────────────────────────────────────────────────────────────────
const canvas     = document.getElementById('board-canvas');
const ctx        = canvas.getContext('2d');
const wrap       = document.getElementById('board-wrap');
const consoleEl  = document.getElementById('console');
const hudEl      = document.getElementById('hud');
const accBarEl   = document.getElementById('accuracy-bar');
const tickEl     = document.getElementById('tick');
const flashEl    = document.getElementById('flash-overlay');

// ── Board geometry ────────────────────────────────────────────────────────────
let CELL = 64, ORIG_X = 0, ORIG_Y = 0, BOARD_PX = 512;

function resizeCanvas() {
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;
  BOARD_PX = Math.floor(Math.min(W * 0.96, H * 0.96) / 8) * 8;
  CELL = BOARD_PX / 8;
  canvas.width  = BOARD_PX + 32;
  canvas.height = BOARD_PX + 32;
  ORIG_X = 24;
  ORIG_Y = 4;
  canvas.style.left = Math.round((W - canvas.width)  / 2) + 'px';
  canvas.style.top  = Math.round((H - canvas.height) / 2) + 'px';
}
window.addEventListener('resize', () => { resizeCanvas(); renderAll(); });
resizeCanvas();

// ── State ─────────────────────────────────────────────────────────────────────
let boardData  = null;   // chess_live_board.json
let frameData  = null;   // live_frame.json (fallback)
let neuroData  = null;   // live_neuro.json

// Animated piece positions: { id -> {x, y, tx, ty, color, piece} }
// x/y = current screen pixel centroid, tx/ty = target
const pieceState = {};
const TWEEN_SPEED = 0.22;  // fraction of distance per frame (exponential ease-out)

let tick = 0;
let animHandle = null;

// ── Coordinate helpers ────────────────────────────────────────────────────────
function squareCX(file) { return ORIG_X + file * CELL + CELL / 2; }
function squareCY(rank) { return ORIG_Y + (7 - rank) * CELL + CELL / 2; }

// ── Drawing ───────────────────────────────────────────────────────────────────
const PIECE_SYMBOLS = {
  K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙',
};
const PIECE_SYMBOLS_BLACK = {
  K: '♚', Q: '♛', R: '♜', B: '♝', N: '♞', P: '♟',
};

function drawBoard(heat, lastMove, predictions, revealPhase) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // ── Board squares with heat overlay ──────────────────────────────────────
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const x = ORIG_X + file * CELL;
      const y = ORIG_Y + (7 - rank) * CELL;
      const base = (file + rank) % 2 === 0 ? '#0e0924' : '#1a1040';
      ctx.fillStyle = base;
      ctx.fillRect(x, y, CELL, CELL);

      // Heat overlay
      const hv = heat ? heat[rank][file] : 0;
      if (hv > 0.03) {
        ctx.globalAlpha = hv * 0.72;
        ctx.fillStyle = plasmaColor(hv);
        ctx.fillRect(x, y, CELL, CELL);
        ctx.globalAlpha = 1.0;
      }
    }
  }

  // ── Last-move highlight ───────────────────────────────────────────────────
  if (lastMove && revealPhase) {
    const pairs = [
      [lastMove.from_file, lastMove.from_rank],
      [lastMove.to_file,   lastMove.to_rank],
    ];
    for (const [f, r] of pairs) {
      if (f == null || r == null || f < 0) continue;
      const x = ORIG_X + f * CELL;
      const y = ORIG_Y + (7 - r) * CELL;
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = '#facc15';
      ctx.fillRect(x, y, CELL, CELL);
      ctx.globalAlpha = 1.0;
    }
  }

  // ── Prediction target highlights ──────────────────────────────────────────
  if (predictions && !revealPhase) {
    for (let i = 0; i < Math.min(3, predictions.length); i++) {
      const p = predictions[i];
      if (p.to_file < 0 || p.to_file == null) continue;
      const x = ORIG_X + p.to_file * CELL;
      const y = ORIG_Y + (7 - p.to_rank) * CELL;
      const alpha = 0.12 + p.probability * 0.45;
      ctx.globalAlpha = Math.min(0.65, alpha);
      ctx.fillStyle = i === 0 ? '#7dd3fc' : '#a78bfa';
      ctx.fillRect(x, y, CELL, CELL);
      // Dot marker
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(x + CELL / 2, y + CELL / 2, 5 + i * (-1.5), 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }
  }

  // ── Grid lines ────────────────────────────────────────────────────────────
  ctx.strokeStyle = 'rgba(80,40,160,0.25)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 8; i++) {
    ctx.beginPath();
    ctx.moveTo(ORIG_X + i * CELL, ORIG_Y);
    ctx.lineTo(ORIG_X + i * CELL, ORIG_Y + BOARD_PX);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ORIG_X, ORIG_Y + i * CELL);
    ctx.lineTo(ORIG_X + BOARD_PX, ORIG_Y + i * CELL);
    ctx.stroke();
  }

  // ── Rank / file labels ────────────────────────────────────────────────────
  ctx.fillStyle = '#5b4080';
  ctx.font = '10px Consolas';
  for (let i = 0; i < 8; i++) {
    ctx.fillText('abcdefgh'[i], ORIG_X + i * CELL + CELL / 2 - 4, ORIG_Y + BOARD_PX + 14);
    ctx.fillText(String(i + 1), ORIG_X - 16, ORIG_Y + (7 - i) * CELL + CELL / 2 + 4);
  }

  // ── Neuro centroids overlay ───────────────────────────────────────────────
  if (neuroData) {
    const centroids = neuroData.neuro?.centroids || {};
    Object.values(centroids).forEach(c => {
      if (typeof c.x !== 'number' || typeof c.y !== 'number') return;
      const f = Math.max(0, Math.min(7, Math.round(c.x)));
      const r = Math.max(0, Math.min(7, Math.round(c.y)));
      const cx2 = ORIG_X + f * CELL + CELL / 2;
      const cy2 = ORIG_Y + (7 - r) * CELL + CELL / 2;
      const pulse = 0.12 + 0.06 * Math.sin(tick / 10);
      ctx.globalAlpha = pulse;
      ctx.fillStyle = '#e879f9';
      ctx.beginPath();
      ctx.arc(cx2, cy2, CELL * 0.38, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;
    });
  }
}

function drawPieces() {
  const fontSize = Math.max(14, Math.floor(CELL * 0.72));
  ctx.font = `${fontSize}px serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (const [id, ps] of Object.entries(pieceState)) {
    if (!ps.alive) continue;
    const sym = ps.color === 'white' ? PIECE_SYMBOLS[ps.piece] : PIECE_SYMBOLS_BLACK[ps.piece];

    // Glow
    if (ps.glow) {
      ctx.shadowBlur  = 18;
      ctx.shadowColor = ps.glow;
    }

    // Drop shadow for readability
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillText(sym, ps.x + 2, ps.y + 2);

    ctx.fillStyle = ps.color === 'white' ? '#e0f2fe' : '#fed7aa';
    ctx.fillText(sym, ps.x, ps.y);

    ctx.shadowBlur  = 0;
    ctx.shadowColor = 'transparent';
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}

function renderAll() {
  if (!boardData && !frameData) return;
  const heat     = boardData?.square_heat || null;
  const lastMove = boardData?.last_move   || null;
  const preds    = boardData?.predictions  || null;
  const reveal   = boardData?.reveal_phase ?? true;
  drawBoard(heat, lastMove, preds, reveal);
  drawPieces();
}

// ── Piece animation loop ──────────────────────────────────────────────────────
function stepAnimation() {
  let needsRedraw = false;
  for (const ps of Object.values(pieceState)) {
    if (!ps.alive) continue;
    const dx = ps.tx - ps.x;
    const dy = ps.ty - ps.y;
    if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
      ps.x += dx * TWEEN_SPEED;
      ps.y += dy * TWEEN_SPEED;
      needsRedraw = true;
    } else {
      ps.x = ps.tx;
      ps.y = ps.ty;
    }
    // Fade glow
    if (ps.glowTime > 0) {
      ps.glowTime--;
      if (ps.glowTime === 0) ps.glow = null;
    }
  }
  if (needsRedraw) renderAll();
  animHandle = requestAnimationFrame(stepAnimation);
}
stepAnimation();

// ── Sync pieces to board data ─────────────────────────────────────────────────
function syncPieces(pieces, lastMove, reveal) {
  // Mark all dead, then revive from current pieces list
  for (const ps of Object.values(pieceState)) ps.alive = false;

  for (const p of (pieces || [])) {
    const tx = squareCX(p.file);
    const ty = squareCY(p.rank);
    if (pieceState[p.id]) {
      const ps = pieceState[p.id];
      ps.alive = true;
      ps.tx = tx;
      ps.ty = ty;
      ps.piece = p.piece;
      ps.color = p.color;
    } else {
      // New piece (or replaced): snap into position
      pieceState[p.id] = { alive: true, x: tx, y: ty, tx, ty, piece: p.piece, color: p.color, glow: null, glowTime: 0 };
    }
  }

  // Apply glow to moved piece when reveal_phase
  if (reveal && lastMove && lastMove.to_file != null && lastMove.to_file >= 0) {
    const tf = lastMove.to_file;
    const tr = lastMove.to_rank;
    // Find piece that ended up there
    for (const ps of Object.values(pieceState)) {
      if (!ps.alive) continue;
      if (Math.round(ps.tx) === Math.round(squareCX(tf)) && Math.round(ps.ty) === Math.round(squareCY(tr))) {
        const correct = boardData?.prediction_correct_top1;
        ps.glow = correct ? '#4ade80' : '#f87171';
        ps.glowTime = 22;
        break;
      }
    }
  }
}

// ── Flash overlay ─────────────────────────────────────────────────────────────
let flashTimer = null;
function flash(correct) {
  flashEl.className = correct ? 'green' : 'red';
  flashEl.style.opacity = '1';
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { flashEl.style.opacity = '0'; flashTimer = null; }, 320);
}

// ── UI updates ────────────────────────────────────────────────────────────────
function fmtPct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '—'; }

function updateSidePanel(data) {
  if (!data) return;

  // Outcome bars
  const op = data.outcome_probs || {};
  const wPct = Math.round((op['1-0'] || 0.33) * 100);
  const dPct = Math.round((op['1/2-1/2'] || 0.33) * 100);
  const bPct = Math.round((op['0-1'] || 0.33) * 100);
  document.getElementById('bar-white').style.width = wPct + '%';
  document.getElementById('bar-draw').style.width  = dPct + '%';
  document.getElementById('bar-black').style.width = bPct + '%';
  document.getElementById('pct-white').textContent = wPct + '%';
  document.getElementById('pct-draw').textContent  = dPct + '%';
  document.getElementById('pct-black').textContent = bPct + '%';

  // Prediction list
  const preds = data.predictions || [];
  const listEl = document.getElementById('pred-list');
  listEl.innerHTML = preds.slice(0, 5).map((p, i) => {
    const cls = p.correct ? 'correct' : 'wrong';
    const probStr = p.energy_prob != null
      ? (p.energy_prob * 100).toFixed(1) + '%'
      : (p.probability * 100).toFixed(1) + '%';
    return `<div class="pred-row ${cls}">
      <span class="pred-san">${p.san}</span>
      <span class="pred-prob">${probStr}</span>
    </div>`;
  }).join('');

  // Model breakdown
  const breakdown = data.model_breakdown || [];
  const breakdownEl = document.getElementById('model-breakdown');
  if (breakdown.length > 0) {
    breakdownEl.innerHTML = breakdown.map(m => {
      const cls    = m.correct ? 'correct' : 'wrong';
      const pct    = m.energy_prob != null ? (m.energy_prob * 100).toFixed(1) + '%' : '—';
      const eStr   = m.energy != null ? m.energy.toFixed(2) : '—';
      const barW   = m.energy_prob != null ? Math.round(m.energy_prob * 100) : 0;
      const moveStr = m.move || '—';
      const shortName = m.model === 'classical' ? 'CLASS' : m.model === 'quantum' ? 'QUANT' : 'NEURO';
      return `<div class="model-row ${cls}">
        <span class="model-name">${shortName}</span>
        <span class="model-move">${moveStr}</span>
        <div class="energy-bar-bg"><div class="energy-bar-fill" style="width:${barW}%"></div></div>
        <span class="model-pct">${pct}<br><span style="color:#555;font-size:9px">E:${eStr}</span></span>
      </div>`;
    }).join('');
  }

  // Neuro network stats from model_ledgers
  const ledgers = data.model_ledgers || {};
  const neuroEl = document.getElementById('neuro-stats');
  if (ledgers.neuro) {
    const n = ledgers.neuro;
    neuroEl.textContent = `neuro: acc=${(n.recent_acc * 100).toFixed(1)}% | vote=${(n.weight * 100).toFixed(1)}% | n=${n.total}`;
  }

  // Running metrics
  const run = data.running || {};
  document.getElementById('m-top1').textContent  = fmtPct(run.move_top1);
  document.getElementById('m-top3').textContent  = fmtPct(run.move_top3);
  document.getElementById('m-out').textContent   = fmtPct(run.outcome_acc);
  document.getElementById('m-plies').textContent = (run.total_plies || 0).toLocaleString();

  // Game info
  document.getElementById('g-id').textContent    = data.game_id || '—';
  document.getElementById('g-ply').textContent   = `ply ${(data.ply ?? '—') + 1} / ${data.total_plies || '—'}`;
  document.getElementById('g-side').textContent  = `to move: ${data.side_to_move || '—'}`;
  document.getElementById('g-count').textContent = `game #${data.game_count || '—'}`;

  // HUD
  const top1 = run.move_top1 != null ? (run.move_top1 * 100).toFixed(1) + '%' : '—';
  const op2   = data.outcome_probs || {};
  const wPct2 = op2['1-0']     != null ? (op2['1-0']     * 100).toFixed(0) : '?';
  const bPct2 = op2['0-1']     != null ? (op2['0-1']     * 100).toFixed(0) : '?';
  hudEl.textContent = `top1: ${top1}  |  W:${wPct2}% B:${bPct2}%  |  ply ${(data.ply ?? '?') + 1}/${data.total_plies || '?'}`;
  accBarEl.textContent = `top-3: ${fmtPct(run.move_top3)}  outcome: ${fmtPct(run.outcome_acc)}`;

  // Hourly checkpoints (last 5)
  const chk = (data.hourly_checkpoints || []).slice(-5).reverse();
  document.getElementById('hourly-list').innerHTML = chk.length === 0 ? '—'
    : chk.map(c => `hr ${c.hour}: top1=${fmtPct(c.move_top1)} (Δ${c.delta_top1 >= 0 ? '+' : ''}${fmtPct(c.delta_top1)})`).join('<br>');
}

// ── Console logger ────────────────────────────────────────────────────────────
const MAX_LINES = 40;
const logLines = [];
function logLine(msg) {
  const time = new Date().toLocaleTimeString();
  logLines.push(`[${time}] ${msg}`);
  if (logLines.length > MAX_LINES) logLines.shift();
  consoleEl.innerHTML = logLines.slice().reverse().map(l => `<div class="log-line">${l}</div>`).join('');
}

// ── Polling ───────────────────────────────────────────────────────────────────
let prevReveal = null;
let prevPly    = null;

async function fetchJson(url) {
  try {
    const r = await fetch(url + '?t=' + Date.now(), { cache: 'no-store' });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

async function poll() {
  tick++;
  tickEl.textContent = `tick ${tick}`;

  // Chess board mode (guard: must have real board data, not just empty {})
  const bd = await fetchJson(BOARD_URL);
  if (bd && bd.pieces && bd.pieces.length > 0) {
    const isNewPly    = bd.ply !== prevPly;
    const isReveal    = bd.reveal_phase;
    const wasPredict  = prevReveal === false;

    boardData = bd;

    if (isNewPly || isReveal !== prevReveal) {
      syncPieces(bd.pieces, bd.last_move, isReveal);
      renderAll();

      if (isReveal && wasPredict && isNewPly !== false) {
        flash(bd.prediction_correct_top1);
        const correctStr = bd.prediction_correct_top1 ? '✓ correct' : '✗ miss';
        logLine(`ply ${bd.ply + 1}  predicted=${(bd.predictions || [])[0]?.san || '?'}  actual=${bd.actual_move?.san || '?'}  ${correctStr}`);
      }

      prevReveal = isReveal;
      prevPly    = bd.ply;
    }

    updateSidePanel(bd);
  } else {
    // Fallback to annealer frame mode
    const frame = await fetchJson(FRAME_URL);
    if (frame) {
      frameData = frame;
      syncAnnealerFrame(frame);
      renderAll();
      hudEl.textContent = `iter ${frame.iteration}  E=${(frame.min_energy || 0).toFixed(3)}  pieces=${(frame.symbols || []).length}`;
      logLine(`annealer iter ${frame.iteration} energy ${(frame.min_energy || 0).toFixed(3)}`);
    }
    const nr = await fetchJson(NEURO_URL);
    if (nr) {
      neuroData = nr;
      renderAll();
    }
  }

  setTimeout(poll, POLL_MS);
}

// ── Annealer fallback ─────────────────────────────────────────────────────────
function syncAnnealerFrame(frame) {
  for (const ps of Object.values(pieceState)) ps.alive = false;
  for (const sym of (frame.symbols || [])) {
    const m = sym.id && sym.id.match(/(white|black)_([KQRBNP])_/i);
    if (!m) continue;
    const color = m[1].toLowerCase();
    const piece  = m[2].toUpperCase();
    const file  = Math.max(0, Math.min(7, Math.round(sym.x ?? 0)));
    const rank  = Math.max(0, Math.min(7, Math.round(sym.y ?? 0)));
    const tx = squareCX(file);
    const ty = squareCY(rank);
    if (pieceState[sym.id]) {
      const ps = pieceState[sym.id];
      ps.alive = true; ps.tx = tx; ps.ty = ty; ps.piece = piece; ps.color = color;
    } else {
      pieceState[sym.id] = { alive: true, x: tx, y: ty, tx, ty, piece, color, glow: null, glowTime: 0 };
    }
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
logLine('Visualizer ready. Polling for chess_live_board.json …');
poll();
</script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args,
                 board_file: Path,
                 frame_file: Optional[Path] = None,
                 neuro_file: Optional[Path] = None,
                 **kwargs):
        self.board_file = board_file
        self.frame_file = frame_file
        self.neuro_file = neuro_file
        super().__init__(*args, **kwargs)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self, path: Optional[Path]) -> dict:
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def do_GET(self):  # noqa: N802
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        elif p == "/board":
            self._send_json(self._read_json(self.board_file))
        elif p == "/frame":
            self._send_json(self._read_json(self.frame_file))
        elif p == "/neuro":
            self._send_json(self._read_json(self.neuro_file))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_):  # suppress default access logging
        pass


def serve(
    board_file: Path,
    frame_file: Optional[Path],
    neuro_file: Optional[Path],
    port: int,
) -> Tuple[socketserver.TCPServer, threading.Thread]:
    def MakeHandler(*args, **kwargs):
        return Handler(
            *args,
            board_file=board_file.resolve(),
            frame_file=frame_file.resolve() if frame_file else None,
            neuro_file=neuro_file.resolve() if neuro_file else None,
            **kwargs,
        )
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", port), MakeHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board-file",  type=Path, default=Path("logs/chess_live_board.json"))
    ap.add_argument("--frame-file",  type=Path, default=Path("logs/live_frame.json"))
    ap.add_argument("--neuro-file",  type=Path, default=Path("logs/live_neuro.json"))
    ap.add_argument("--port",        type=int,  default=8765)
    ap.add_argument("--open",        action="store_true")
    args = ap.parse_args()

    args.board_file.parent.mkdir(parents=True, exist_ok=True)

    httpd, thread = serve(args.board_file, args.frame_file, args.neuro_file, args.port)
    url = f"http://localhost:{args.port}/"
    print(f"Live viz at {url}")
    print(f"  Board file: {args.board_file}")
    print(f"  Frame file: {args.frame_file}")
    if args.open:
        webbrowser.open(url)
    try:
        thread.join()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
