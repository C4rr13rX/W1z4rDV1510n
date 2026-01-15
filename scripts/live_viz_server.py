#!/usr/bin/env python3
"""
Serve a live visualization that polls JSON streams written by the Rust runtime:
- frame stream (logging.live_frame_path)
- neuro stream (logging.live_neuro_path)

Usage:
  python scripts/live_viz_server.py --frame-file logs/live_frame.json --neuro-file logs/live_neuro.json --port 8765 --open
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


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Live Viz</title>
  <script src="https://cdn.jsdelivr.net/npm/phaser@3.80.0/dist/phaser.min.js"></script>
  <style>
    html, body { margin:0; padding:0; width:100%; height:100%; background:black; color:#8ef; font-family: 'Consolas', monospace; overflow:hidden;}
    #status { position:fixed; top:8px; left:12px; color:#f0f; font-size:14px; z-index:10; text-shadow:0 0 6px #f0f; }
    #console { position:fixed; bottom:8px; left:12px; right:12px; height:120px; background:rgba(0,0,0,0.65); color:#8ef; font-size:12px; overflow:auto; border:1px solid #222; padding:6px; }
    #loader { position:fixed; top:50%; left:50%; transform:translate(-50%,-50%); color:#fff; font-size:16px; }
    canvas { display:block; }
  </style>
</head>
<body>
  <div id="status">Booting...</div>
  <div id="loader">Loading interface...</div>
  <div id="console"></div>
  <script>
    const statusEl = document.getElementById('status');
    const consoleEl = document.getElementById('console');
    const loaderEl = document.getElementById('loader');
    const frameUrl = `${window.location.origin}/frame`;
    const neuroUrl = `${window.location.origin}/neuro`;
    const fileMap = {a:0,b:1,c:2,d:3,e:4,f:5,g:6,h:7};

    function logLine(msg) {
      const time = new Date().toLocaleTimeString();
      consoleEl.textContent += `[${time}] ${msg}\\n`;
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    async function fetchJson(url) {
      try {
        const r = await fetch(url, { cache:'no-store' });
        if (!r.ok) throw new Error(`status ${r.status}`);
        return await r.json();
      } catch (e) {
        return null;
      }
    }

    function decodeChess(symbol) {
      const m = symbol.id && symbol.id.match(/(white|black)_([KQRBNP])_/i);
      if (!m) return null;
      const color = m[1].toLowerCase();
      const piece = m[2].toUpperCase();
      // prefer live position; clamp to board
      let fx = Math.round(symbol.x ?? symbol.position?.x ?? 0);
      let fy = Math.round(symbol.y ?? symbol.position?.y ?? 0);
      fx = Math.max(0, Math.min(7, fx));
      fy = Math.max(0, Math.min(7, fy));
      return {x:fx, y:fy, color, piece};
    }

    const config = {
      type: Phaser.AUTO,
      width: window.innerWidth,
      height: window.innerHeight,
      backgroundColor: '#000000',
      scene: {
        preload: preload,
        create: create,
        update: update
      }
    };

    let chessLayer, hudText, heatLayer;
    let latestFrame = null;
    let latestNeuro = null;
    let tick = 0;

    function preload() {}

    function create() {
      const scene = this;
      chessLayer = scene.add.layer();
      heatLayer = scene.add.layer();
      hudText = scene.add.text(20, 28, 'Waiting...', { fontFamily:'Consolas', fontSize:16, color:'#8ef' });
      drawBoard(scene);
      loaderEl.style.display = 'none';
      logLine('Visualizer ready. Polling streams...');
    }

    function drawBoard(scene) {
      const size = Math.min(window.innerHeight * 0.7, window.innerWidth * 0.5);
      const cell = size / 8;
      const originX = window.innerWidth/2 - size/2;
      const originY = window.innerHeight/2 - size/2;
      chessLayer.removeAll(true);
      heatLayer.removeAll(true);
      // squares
      for (let y=0;y<8;y++){
        for (let x=0;x<8;x++){
          const color = (x+y)%2===0 ? 0x111122 : 0x222244;
          const rect = scene.add.rectangle(originX + x*cell + cell/2, originY + (7-y)*cell + cell/2, cell, cell, color, 0.9);
          chessLayer.add(rect);
        }
      }
      // labels
      for (let x=0;x<8;x++){
        scene.add.text(originX + x*cell + cell/2 - 6, originY + size + 6, 'abcdefgh'[x], {fontSize:12, color:'#777'}).setDepth(5);
      }
      for (let y=0;y<8;y++){
        scene.add.text(originX - 16, originY + (7-y)*cell + cell/2 - 8, String(y+1), {fontSize:12, color:'#777'}).setDepth(5);
      }
      chessLayer.setData('origin', {x:originX, y:originY, cell});
    }

    function renderFrame(scene) {
      if (!latestFrame || !chessLayer) return;
      const meta = chessLayer.getData('origin');
      // clear pieces
      chessLayer.list = chessLayer.list.filter(obj => obj.getData('piece') !== true);
      chessLayer.list.filter(obj => obj.getData('piece') === true).forEach(obj => obj.destroy());
      // pieces
      for (const s of latestFrame.symbols || []) {
        const d = decodeChess(s);
        if (!d) continue;
        const cx = meta.x + d.x * meta.cell + meta.cell/2;
        const cy = meta.y + (7 - d.y) * meta.cell + meta.cell/2;
        const txt = scene.add.text(cx - 8, cy - 10, d.piece, {fontSize:24, color: d.color === 'white' ? '#66f4ff' : '#ffb347' });
        txt.setData('piece', true);
        chessLayer.add(txt);
      }
      hudText.setText(`Iter ${latestFrame.iteration} | E=${(latestFrame.min_energy||0).toFixed(3)} | pieces=${(latestFrame.symbols||[]).length}`);
    }

    function renderNeuro(scene) {
      if (!latestNeuro || !heatLayer) return;
      heatLayer.list.forEach(o => o.destroy());
      const meta = chessLayer.getData('origin');
      const size = meta.cell * 8;
      const centerX = meta.x + size/2;
      const centerY = meta.y + size/2;

      const centroids = latestNeuro.neuro?.centroids || {};
      Object.values(centroids).forEach((c) => {
        if (c && typeof c.x === 'number' && typeof c.y === 'number') {
          const cx = meta.x + Math.max(0, Math.min(7, c.x)) * meta.cell + meta.cell/2;
          const cy = meta.y + (7 - Math.max(0, Math.min(7, c.y))) * meta.cell + meta.cell/2;
          const pulse = 0.25 + 0.15 * Math.sin(tick/8);
          const square = heatLayer.scene.add.rectangle(cx, cy, meta.cell, meta.cell, 0xff5cf0, pulse);
          heatLayer.add(square);
        }
      });

      const actCount = (latestNeuro.neuro?.active_labels?.length || 0) + (latestNeuro.neuro?.active_networks?.length || 0);
      const intensity = Math.min(1, actCount / 80);
      const ringRadius = size * (0.2 + 0.3 * intensity);
      const color = Phaser.Display.Color.GetColor(255, 120 + Math.floor(80*intensity), 255);
      const ring = heatLayer.scene.add.circle(centerX, centerY, ringRadius, color, 0.18);
      heatLayer.add(ring);
    }

    async function poll() {
      const ts = Date.now();
      const frame = await fetchJson(frameUrl + '?t=' + ts);
      if (frame) {
        latestFrame = frame;
        renderFrame(window.__scene__);
        logLine(`iter ${frame.iteration} energy ${(frame.min_energy||0).toFixed(3)} pieces ${(frame.symbols||[]).length}`);
      }
      const neuro = await fetchJson(neuroUrl + '?t=' + ts);
      if (neuro) {
        latestNeuro = neuro;
        renderNeuro(window.__scene__);
      }
      tick++;
      statusEl.textContent = `tick ${tick}`;
      setTimeout(poll, 150);
    }

    function update() {}

    const game = new Phaser.Game(config);
    game.events.on('ready', () => {
      window.__scene__ = game.scene.keys.default;
      poll();
    });
  </script>
</body>
</html>
"""


class LiveVizHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, frame_file: Path, **kwargs):
        self.frame_file = frame_file
        self.neuro_file: Optional[Path] = kwargs.pop("neuro_file", None)
        super().__init__(*args, **kwargs)

    def do_GET(self):  # noqa
        if self.path == "/" or self.path == "/index.html":
            data = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/frame":
            try:
                payload = json.loads(self.frame_file.read_text())
            except Exception:
                payload = {}
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/neuro":
            try:
                if self.neuro_file and self.neuro_file.exists():
                    payload = json.loads(self.neuro_file.read_text())
                else:
                    payload = {}
            except Exception:
                payload = {}
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()


def serve(frame_file: Path, neuro_file: Optional[Path], port: int) -> Tuple[socketserver.TCPServer, threading.Thread]:
    frame_file = frame_file.resolve()
    neuro_file = neuro_file.resolve() if neuro_file else None

    def Handler(*args, **kwargs):
        return LiveVizHandler(*args, frame_file=frame_file, neuro_file=neuro_file, **kwargs)  # noqa

    httpd = socketserver.TCPServer(("", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-file", type=Path, default=Path("logs/live_frame.json"))
    ap.add_argument("--neuro-file", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    args.frame_file.parent.mkdir(parents=True, exist_ok=True)
    server, thread = serve(args.frame_file, args.neuro_file, args.port)
    url = f"http://localhost:{args.port}/"
    print(f"Live viz at {url}, reading {args.frame_file}")
    if args.open:
        webbrowser.open(url)
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
