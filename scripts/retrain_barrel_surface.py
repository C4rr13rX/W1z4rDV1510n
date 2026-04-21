#!/usr/bin/env python3
"""
Retrain the cow surface grid with barrel-shaped cross-section Z values.
Sends 300 EnvironmentSnapshot frames with the updated surface grid to shift
existing cow_surf_XX_XX centroids toward the correct barrel shape.
"""
import json, math, random, time, urllib.request

NODE = "http://localhost:8090"
WITHER_H = 1.42
HALF_L   = 1.585
HALF_W   = 0.32
NX, NY   = 14, 10

def barrel_hw(ty):
    """Half-width at normalised height ty (0=withers, 1=ground). Barrel cross-section."""
    return HALF_W * (0.25 + 0.75 * math.sin(math.pi * min(ty * 1.5, 1.0)) ** 0.7)

def jit(scale=0.018):
    return random.gauss(0, scale)

def make_frame():
    symbols = []
    for ix in range(NX):
        tx = ix / (NX - 1)
        x  = HALF_L * (1.0 - 2 * tx) + jit()
        for iy in range(NY):
            ty = iy / (NY - 1)
            y  = WITHER_H * (1.0 - ty) + jit(0.012)
            hw = barrel_hw(ty)
            label = f"cow_surf_{iy:02d}_{ix:02d}"
            symbols += [
                {"id": label + "_L", "type": "OBJECT",
                 "position": {"x": x, "y": max(0.0, y), "z":  hw + jit(0.015)}},
                {"id": label + "_R", "type": "OBJECT",
                 "position": {"x": x, "y": max(0.0, y), "z": -hw + jit(0.015)}},
                {"id": label,        "type": "OBJECT",
                 "position": {"x": x, "y": max(0.0, y), "z":  jit(0.012)}},
            ]
    return {
        "snapshot": {
            "timestamp": {"unix": int(time.time())},
            "bounds": {"x": 3.2, "y": 1.8, "z": 1.0},
            "symbols": symbols,
            "metadata": {"scene": "barrel_surface_retrain", "species": "holstein_dairy_cow"},
        },
        "extra_labels": ["txt:word_cow", "txt:word_bovine", "cow_body", "bovine_anatomy"],
    }

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{NODE}{path}", data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

N = 300
print(f"Retraining {N} barrel-surface frames...", flush=True)
ok = 0
for i in range(N):
    try:
        post("/neuro/train", make_frame())
        ok += 1
    except Exception as e:
        print(f"  frame {i}: {e}", flush=True)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N}", flush=True)
    time.sleep(0.03)
print(f"Done: {ok}/{N}")
