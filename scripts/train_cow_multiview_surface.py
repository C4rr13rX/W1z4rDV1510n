#!/usr/bin/env python3
"""
scripts/train_cow_multiview_surface.py

Seeds the neural fabric with DENSE 3D surface points covering ALL cow perspectives:
  - Left side, right side, back, front, dorsal (top), ventral (bottom)
  - Body barrel surface ~200 pts, head ~30 pts, 4 legs ~80 pts, neck ~20 pts

Labels use anatomy keyword substrings so /mesh/synthesize body/head/leg queries
match many more centroids -> richer convex hull -> recognisable bovine shape.

The user insight: "it's got to have seen the back and the side and the underside
and the top side to know what a cow looks like" -- XYZ labels are the 3D equivalent
of multi-view understanding from 2D video observations.

Usage:
  python scripts/train_cow_multiview_surface.py              # 600 frames
  python scripts/train_cow_multiview_surface.py --frames 300 # quick run
  python scripts/train_cow_multiview_surface.py --check      # inspect centroids
"""
import argparse, json, math, random, time, urllib.request

NODE  = "http://localhost:8090"
# --- Cow body dimensions (metres, Y-up, X=head->tail) -------------------------
WITHER_H  = 1.42   # withers height
HALF_L    = 1.00   # body half-length (shoulder x~0.82 -> hip x~-0.80 -> half=1.0)
# Barrel cross-section width at normalised height ty (0=withers, 1=ground-belly)
def barrel_hw(ty):
    """Half-width at normalised height ty. Widest ~ mid-rib."""
    return 0.34 * (0.25 + 0.75 * math.sin(math.pi * min(ty * 1.5, 1.0)) ** 0.7)

# --- Body surface points ------------------------------------------------------
# Generate a dense grid of points on the cow body barrel surface.
# 16 columns along X (head->tail), 18 rows around the oval cross-section.
# Label each point with the nearest anatomy keyword so the /mesh/synthesize
# body query matches it.

def _body_label(x_norm, angle_deg):
    """Choose an anatomy keyword label for a surface point by position."""
    if angle_deg < 30 or angle_deg > 330:   # top
        region = "back" if x_norm < 0.6 else "withers"
        side   = "center"
    elif angle_deg < 90:                    # upper-left
        region = "shoulder" if x_norm > 0.5 else ("spine" if x_norm > 0.0 else "sacrum")
        side   = "L"
    elif angle_deg < 150:                   # lower-left
        region = "flank" if x_norm < 0.7 else "thorax"
        side   = "L"
    elif angle_deg < 210:                   # bottom
        region = "belly" if x_norm > 0.3 else "udder"
        side   = "center"
    elif angle_deg < 270:                   # lower-right
        region = "flank" if x_norm < 0.7 else "thorax"
        side   = "R"
    else:                                   # upper-right
        region = "shoulder" if x_norm > 0.5 else ("hip" if x_norm < 0.2 else "spine")
        side   = "R"
    return region, side

NX_BODY = 14   # columns along body length
NA_BODY = 18   # angular divisions around the cross-section

BODY_SURFACE = []
for ix in range(NX_BODY):
    tx      = ix / (NX_BODY - 1)
    x_body  = HALF_L * (1.0 - 2 * tx)  # +1.0 at shoulder, -1.0 at rump

    # Oval cross-section: height 0.30 (belly) -> 1.42 (withers)
    belly_y  = 0.30
    top_y    = 1.42
    mid_y    = (belly_y + top_y) / 2   # 0.86
    h_half   = (top_y - belly_y) / 2   # 0.56

    for ia in range(NA_BODY):
        angle = ia / NA_BODY * 2 * math.pi
        # Ellipse: width uses barrel_hw, height uses h_half
        # angle=0 -> top, angle=pi -> bottom
        raw_y = mid_y + h_half * math.cos(angle)
        ty    = 1.0 - (raw_y - belly_y) / (top_y - belly_y)  # 0=top, 1=bottom
        hw    = barrel_hw(max(0, min(1, ty)))
        raw_z = hw * math.sin(angle)

        angle_deg = math.degrees(angle) % 360
        region, side = _body_label(tx, angle_deg)
        # Include x-axis keyword for front/rear anatomical labels
        if x_body > 0.5:
            prefix = "thorax" if side != "center" else "chest"
        elif x_body > -0.1:
            prefix = "rumen" if side == "L" else ("liver" if side == "R" else "diaphragm")
        else:
            prefix = "hip" if side != "center" else "sacrum"
        label = f"cow_{prefix}_{region}_{side}_{ix:02d}_{ia:02d}"
        BODY_SURFACE.append((label, x_body, raw_y, raw_z))

# --- Head surface points ------------------------------------------------------
# Ellipsoid centred at (1.45, 1.10, 0), elongated along X.
HEAD_SURFACE = []
NX_HEAD = 6; NA_HEAD = 12
for ix in range(NX_HEAD):
    tx    = ix / (NX_HEAD - 1)
    x_h   = 1.28 + 0.35 * tx           # 1.28 -> 1.63 (back of skull -> snout)
    for ia in range(NA_HEAD):
        angle = ia / NA_HEAD * 2 * math.pi
        y_h   = 1.10 + 0.18 * math.cos(angle)
        z_h   = 0.16 * math.sin(angle)
        angle_deg = math.degrees(angle) % 360
        if angle_deg < 45 or angle_deg > 315:  region = "brain"
        elif angle_deg < 135:                  region = "ear"
        elif angle_deg < 225:                  region = "jaw"
        else:                                  region = "horn"
        side = "L" if z_h > 0 else "R"
        label = f"cow_{region}_{side}_{ix:02d}_{ia:02d}"
        HEAD_SURFACE.append((label, x_h, y_h, z_h))
# snout tip
for ia in range(6):
    angle = ia / 6 * 2 * math.pi
    HEAD_SURFACE.append((f"cow_snout_nostril_{ia:02d}", 1.63, 0.90 + 0.04*math.cos(angle), 0.05*math.sin(angle)))

# --- Neck surface points ------------------------------------------------------
NECK_SURFACE = []
for ix in range(6):
    tx   = ix / 5
    x_n  = 0.85 + 0.45 * tx   # 0.85 -> 1.30 (withers -> skull base)
    y_n  = 0.95 + 0.23 * tx   # rises toward head
    for ia in range(8):
        angle = ia / 8 * 2 * math.pi
        y_off = 0.12 * math.cos(angle)
        z_off = 0.14 * math.sin(angle)
        side  = "L" if z_off > 0 else "R"
        part  = "neck" if x_n < 1.1 else "throat"
        NECK_SURFACE.append((f"cow_{part}_{side}_{ix:02d}_{ia:02d}", x_n, y_n + y_off, z_off))

# --- Leg surface points ------------------------------------------------------
# Each leg is a tapered cylinder from shoulder/hip to hoof.
def _leg_surface(x_base, z_side, labels, prefix):
    """Generate surface points for one leg."""
    pts = []
    NY_LEG = 10
    for iy in range(NY_LEG):
        ty     = iy / (NY_LEG - 1)
        y_leg  = 1.10 * (1.0 - ty)      # 1.10 -> 0 (shoulder to hoof)
        r      = 0.10 - 0.04 * ty        # tapers slightly toward hoof
        for ia in range(8):
            angle = ia / 8 * 2 * math.pi
            y_l   = y_leg + r * math.cos(angle) * 0.3
            z_l   = z_side + r * math.sin(angle)
            side  = "L" if z_side > 0 else "R"
            seg   = labels[min(iy, len(labels)-1)]
            pts.append((f"cow_{seg}_{side}_{iy:02d}_{ia:02d}", x_base, y_l, z_l))
    return pts

FL_LABELS = ["shoulder","shoulder","elbow","elbow","forearm","forearm","knee","cannon","fetlock","hoof"]
RL_LABELS = ["hip","hip","femur","stifle","tibia","tibia","hock","cannon","fetlock","hoof"]
LEG_SURFACE = (
    _leg_surface(0.80,  0.23, FL_LABELS, "fl") +
    _leg_surface(0.80, -0.23, FL_LABELS, "fr") +
    _leg_surface(-0.80, 0.22, RL_LABELS, "rl") +
    _leg_surface(-0.80,-0.22, RL_LABELS, "rr")
)

ALL_SURFACE = BODY_SURFACE + HEAD_SURFACE + NECK_SURFACE + LEG_SURFACE

print(f"Dense surface atlas: {len(BODY_SURFACE)} body + {len(HEAD_SURFACE)} head + "
      f"{len(NECK_SURFACE)} neck + {len(LEG_SURFACE)} legs = {len(ALL_SURFACE)} total points",
      flush=True)

# --- Extra query labels (text tokens) ----------------------------------------
QUERY_LABELS = [
    "txt:word_cow", "txt:word_dairy", "txt:word_bovine", "txt:word_Holstein",
    "txt:word_cattle", "txt:word_anatomy", "txt:word_body", "txt:word_skeleton",
    "txt:word_leg", "txt:word_legs", "txt:word_head", "txt:word_neck",
    "txt:word_spine", "txt:word_rumen", "txt:word_heart", "txt:word_lung",
    "txt:word_hoof", "txt:word_udder", "txt:word_tail", "txt:word_eye",
    "txt:word_ear", "txt:word_shoulder", "txt:word_hip", "txt:word_full",
    "txt:word_liver", "txt:word_kidney", "txt:word_brain", "txt:word_horn",
    "txt:word_flank", "txt:word_back", "txt:word_belly", "txt:word_chest",
    "txt:word_thorax", "txt:word_sacrum", "txt:word_withers", "txt:word_rump",
    "txt:word_elbow", "txt:word_forearm", "txt:word_fetlock", "txt:word_cannon",
    "txt:word_side", "txt:word_surface", "txt:word_view", "txt:word_perspective",
    "cow_body", "bovine_anatomy", "cow_skeleton", "cow_surface",
]

# --- HTTP helper --------------------------------------------------------------
def post_json(path, body, timeout=15):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{NODE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

BATCH_SIZE = 80  # symbols per frame -- keeps each request well under 15s timeout

def make_snapshot(subset, noise=0.015):
    """One EnvironmentSnapshot with a SUBSET of surface points + jitter."""
    symbols = []
    for (part_id, x, y, z) in subset:
        symbols.append({
            "id": part_id, "type": "OBJECT",
            "position": {
                "x": x + random.gauss(0, noise),
                "y": max(0.0, y + random.gauss(0, noise * 0.5)),
                "z": z + random.gauss(0, noise),
            },
            "properties": {"species": "bovine", "category": "cow_surface"},
        })
    return {
        "snapshot": {
            "timestamp": {"unix": int(time.time())},
            "bounds": {"x": 3.4, "y": 1.8, "z": 0.8},
            "symbols": symbols,
            "metadata": {"scene": "bovine_multiview_surface", "source": "dense_3d_atlas"},
        },
        "extra_labels": QUERY_LABELS,
    }

def train(n_frames, delay=0.05):
    print(f"Training {n_frames} frames (batch={BATCH_SIZE} symbols/frame) ...", flush=True)
    shuffled = list(ALL_SURFACE)
    ok = 0
    for i in range(n_frames):
        # Shuffle each "epoch" so all symbols get equal coverage
        if i % (len(shuffled) // BATCH_SIZE + 1) == 0:
            random.shuffle(shuffled)
        start = (i * BATCH_SIZE) % len(shuffled)
        subset = shuffled[start:start + BATCH_SIZE]
        if len(subset) < BATCH_SIZE:
            subset += shuffled[:BATCH_SIZE - len(subset)]
        try:
            post_json("/neuro/train", make_snapshot(subset))
            ok += 1
        except Exception as e:
            print(f"  frame {i}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_frames}", flush=True)
        time.sleep(delay)
    print(f"Done: {ok}/{n_frames} accepted", flush=True)
    return ok

def check(show_n=10):
    req = urllib.request.Request(f"{NODE}/neuro/snapshot", method="GET")
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    c = d.get("centroids", {})
    surface_c = {k: v for k, v in c.items() if "cow_" in k.lower()}
    print(f"Total centroids: {len(c)}   cow-related: {len(surface_c)}", flush=True)
    # Show Z range for body points (should be barrel-shaped ~ +-0.34 max)
    body_pts  = {k: v for k, v in surface_c.items() if any(x in k for x in ["flank","thorax","rumen","belly","back","shoulder"])}
    if body_pts:
        zs = [v["z"] for v in body_pts.values()]
        print(f"Body centroid Z range: [{min(zs):.3f}, {max(zs):.3f}]  (target ~+-0.34)", flush=True)
    for k, v in list(surface_c.items())[:show_n]:
        print(f"  {k}: z={v['z']:.3f}", flush=True)

def test_mesh_body():
    r = post_json("/mesh/synthesize", {
        "query": "thorax spine rumen reticulum omasum abomasum heart lung diaphragm udder belly withers sacrum rump sternum flank back shoulder hip chest liver kidney spleen pancreas hide_dorsal",
        "hops": 3, "min_activation": 0.01, "categories": ["id"],
        "x_range": [-1.5, 0.82],
    })
    vc = r.get("vertex_count", 0)
    fc = r.get("face_count",   0)
    src = r.get("point_source", 0)
    print(f"Body mesh: {vc} verts, {fc} faces, {src} source points", flush=True)
    if r.get("obj"):
        verts = []
        for line in r["obj"].split("\n"):
            p = line.strip().split()
            if p and p[0] == "v":
                verts.append([float(p[1]), float(p[2]), float(p[3])])
        if verts:
            zs = [v[2] for v in verts]
            ys = [v[1] for v in verts]
            print(f"  Z range: [{min(zs):.3f}, {max(zs):.3f}]  (width target ~+-0.34)", flush=True)
            print(f"  Y range: [{min(ys):.3f}, {max(ys):.3f}]  (height target ~0.28-1.42)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--check",  action="store_true")
    args = ap.parse_args()

    if args.check:
        check()
        test_mesh_body()
    else:
        print("=== Phase 1: before training ===", flush=True)
        test_mesh_body()
        train(args.frames)
        print("\n=== Phase 2: after training ===", flush=True)
        check()
        test_mesh_body()
