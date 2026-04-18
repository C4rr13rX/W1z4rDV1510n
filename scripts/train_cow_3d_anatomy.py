#!/usr/bin/env python3
# coding: utf-8
"""
scripts/train_cow_3d_anatomy.py

Seeds the neural fabric with 3D anatomical positions for bovine body parts.
Each training frame is an EnvironmentSnapshot with all major anatomical
landmarks at their correct 3D positions, plus text co-labels that link the
spatial layout to the query vocabulary used by /mesh/synthesize.

After ~500 frames centroids are dense enough for a recognisable bovine mesh.

Usage:
  python scripts/train_cow_3d_anatomy.py            # 500 frames default
  python scripts/train_cow_3d_anatomy.py --frames 800 --check-only
"""
import argparse, json, math, random, sys, time, urllib.request, urllib.error

NODE = "http://localhost:8090"

# ── Bovine anatomy 3D atlas ────────────────────────────────────────────────
# Y-up, X = nose-to-tail, Z = left/right.  All values in cow-body units
# (~1 = 1 m).  Head at X≈+1.5, rump at X≈-1.2, feet at Y=0, withers Y≈1.4.
BOVINE_ANATOMY = [
    # ─ Axial skeleton ─────────────────────────────────────────────────────
    ("cow_head",          1.40, 1.05, 0.00),
    ("cow_snout",         1.62, 0.90, 0.00),
    ("cow_brain",         1.36, 1.22, 0.00),
    ("cow_jaw",           1.52, 0.84, 0.00),
    ("cow_neck_upper",    1.08, 1.18, 0.00),
    ("cow_neck_lower",    1.00, 0.95, 0.00),
    ("cow_withers",       0.55, 1.42, 0.00),
    ("cow_thorax",        0.30, 1.20, 0.00),
    ("cow_spine_T",       0.10, 1.32, 0.00),   # thoracic
    ("cow_spine_L",      -0.30, 1.30, 0.00),   # lumbar
    ("cow_sacrum",       -0.68, 1.20, 0.00),
    ("cow_rump",         -0.80, 1.18, 0.00),
    ("cow_tail_root",    -1.10, 1.12, 0.00),
    ("cow_tail_mid",     -1.35, 0.98, 0.00),
    ("cow_tail_tip",     -1.55, 0.78, 0.00),

    # ─ Head features ──────────────────────────────────────────────────────
    ("cow_eye_L",         1.44, 1.18, 0.19),
    ("cow_eye_R",         1.44, 1.18,-0.19),
    ("cow_ear_L",         1.22, 1.28, 0.23),
    ("cow_ear_R",         1.22, 1.28,-0.23),
    ("cow_horn_L",        1.10, 1.34, 0.17),
    ("cow_horn_R",        1.10, 1.34,-0.17),
    ("cow_nostril_L",     1.64, 0.93, 0.06),
    ("cow_nostril_R",     1.64, 0.93,-0.06),

    # ─ Thoracic organs ────────────────────────────────────────────────────
    ("cow_heart",         0.62, 1.10, 0.00),
    ("cow_lung_L",        0.45, 1.16, 0.24),
    ("cow_lung_R",        0.45, 1.16,-0.24),
    ("cow_trachea",       0.90, 1.08, 0.00),
    ("cow_diaphragm",     0.08, 1.00, 0.00),
    ("cow_sternum",       0.50, 0.68, 0.00),

    # ─ Digestive system ───────────────────────────────────────────────────
    ("cow_rumen",        -0.28, 0.92, 0.24),   # largest — left side
    ("cow_rumen_dorsal", -0.20, 1.02, 0.26),
    ("cow_rumen_ventral",-0.36, 0.72, 0.22),
    ("cow_reticulum",     0.06, 0.88, 0.12),
    ("cow_omasum",        0.14, 0.90,-0.06),
    ("cow_abomasum",      0.06, 0.68, 0.02),
    ("cow_small_int",    -0.08, 0.55, 0.00),
    ("cow_large_int",    -0.40, 0.60, 0.00),
    ("cow_liver",         0.15, 0.94,-0.26),
    ("cow_pancreas",      0.00, 0.80,-0.12),
    ("cow_spleen",       -0.15, 0.88, 0.28),

    # ─ Urogenital ─────────────────────────────────────────────────────────
    ("cow_kidney_L",     -0.24, 1.06, 0.20),
    ("cow_kidney_R",     -0.24, 1.06,-0.20),
    ("cow_bladder",      -0.50, 0.45, 0.00),
    ("cow_udder",        -0.34, 0.28, 0.00),
    ("cow_teat_LF",      -0.28, 0.12, 0.10),
    ("cow_teat_RF",      -0.28, 0.12,-0.10),
    ("cow_teat_LR",      -0.44, 0.12, 0.10),
    ("cow_teat_RR",      -0.44, 0.12,-0.10),

    # ─ Appendicular — front ───────────────────────────────────────────────
    ("cow_shoulder_L",    0.76, 1.12, 0.32),
    ("cow_shoulder_R",    0.76, 1.12,-0.32),
    ("cow_upper_arm_L",   0.78, 0.88, 0.30),
    ("cow_upper_arm_R",   0.78, 0.88,-0.30),
    ("cow_elbow_L",       0.80, 0.74, 0.28),
    ("cow_elbow_R",       0.80, 0.74,-0.28),
    ("cow_forearm_L",     0.80, 0.58, 0.26),
    ("cow_forearm_R",     0.80, 0.58,-0.26),
    ("cow_knee_FL",       0.80, 0.44, 0.24),
    ("cow_knee_FR",       0.80, 0.44,-0.24),
    ("cow_cannon_FL",     0.80, 0.28, 0.23),
    ("cow_cannon_FR",     0.80, 0.28,-0.23),
    ("cow_fetlock_FL",    0.80, 0.14, 0.22),
    ("cow_fetlock_FR",    0.80, 0.14,-0.22),
    ("cow_hoof_FL",       0.80, 0.00, 0.22),
    ("cow_hoof_FR",       0.80, 0.00,-0.22),

    # ─ Appendicular — hind ────────────────────────────────────────────────
    ("cow_pelvis_L",     -0.70, 1.14, 0.34),
    ("cow_pelvis_R",     -0.70, 1.14,-0.34),
    ("cow_hip_L",        -0.80, 1.12, 0.33),
    ("cow_hip_R",        -0.80, 1.12,-0.33),
    ("cow_femur_L",      -0.78, 0.86, 0.30),
    ("cow_femur_R",      -0.78, 0.86,-0.30),
    ("cow_stifle_L",     -0.76, 0.76, 0.28),
    ("cow_stifle_R",     -0.76, 0.76,-0.28),
    ("cow_tibia_L",      -0.78, 0.58, 0.26),
    ("cow_tibia_R",      -0.78, 0.58,-0.26),
    ("cow_hock_L",       -0.80, 0.44, 0.24),
    ("cow_hock_R",       -0.80, 0.44,-0.24),
    ("cow_cannon_BL",    -0.80, 0.28, 0.22),
    ("cow_cannon_BR",    -0.80, 0.28,-0.22),
    ("cow_fetlock_BL",   -0.80, 0.14, 0.21),
    ("cow_fetlock_BR",   -0.80, 0.14,-0.21),
    ("cow_hoof_BL",      -0.80, 0.00, 0.21),
    ("cow_hoof_BR",      -0.80, 0.00,-0.21),

    # ─ Body surface reference zones ───────────────────────────────────────
    ("cow_chest_wall",    0.58, 0.78, 0.00),
    ("cow_belly",         0.08, 0.38, 0.00),
    ("cow_flank_L",      -0.22, 0.82, 0.40),
    ("cow_flank_R",      -0.22, 0.82,-0.40),
    ("cow_back",          0.00, 1.40, 0.00),
    ("cow_hide_dorsal",   0.00, 1.44, 0.00),

    # ─ Alias / shorthand labels (legacy sensor-stream names — reset to correct canonical pos) ─
    ("cow_body",          0.00, 0.80, 0.00),   # body centroid
    ("cow_spine",         0.00, 1.30, 0.00),   # spine midpoint
    ("cow_neck",          1.04, 1.10, 0.00),   # neck midpoint
    ("cow_tail",         -1.35, 0.98, 0.00),   # tail midpoint
    ("cow_chest",         0.50, 0.68, 0.00),   # chest / sternum
    ("cow_leg_FL",        0.80, 0.55, 0.23),   # front-left leg mid
    ("cow_leg_FR",        0.80, 0.55,-0.23),   # front-right leg mid
    ("cow_leg_BL",       -0.80, 0.55, 0.22),   # back-left leg mid
    ("cow_leg_BR",       -0.80, 0.55,-0.22),   # back-right leg mid
]

# ── Text labels co-trained with each frame ────────────────────────────────
# These are the raw word tokens the TextBitsEncoder produces for the
# mesh/synthesize query string, plus semantic synonyms.
QUERY_LABELS = [
    "txt:word_cow", "txt:word_dairy", "txt:word_bovine", "txt:word_Holstein",
    "txt:word_cattle", "txt:word_anatomy", "txt:word_body", "txt:word_skeleton",
    "txt:word_leg", "txt:word_legs", "txt:word_head", "txt:word_neck",
    "txt:word_spine", "txt:word_rumen", "txt:word_heart", "txt:word_lung",
    "txt:word_hoof", "txt:word_udder", "txt:word_tail", "txt:word_eye",
    "txt:word_ear", "txt:word_shoulder", "txt:word_hip", "txt:word_full",
    "txt:word_liver", "txt:word_kidney", "txt:word_brain", "txt:word_horn",
    "txt:word_bone", "txt:word_muscle", "txt:word_organ", "txt:word_whole",
    # Direct semantic labels (trained by media/train on books)
    "cow_body", "bovine_anatomy", "cow_skeleton", "cow_organs",
]


def post_json(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{NODE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def make_snapshot(noise=0.018, ts=None):
    """Build one EnvironmentSnapshot with all bovine landmarks plus jitter."""
    symbols = []
    for (part_id, x, y, z) in BOVINE_ANATOMY:
        jx = x + random.gauss(0, noise)
        jy = max(0.0, y + random.gauss(0, noise * 0.5))
        jz = z + random.gauss(0, noise)
        symbols.append({
            "id": part_id,
            "type": "OBJECT",
            "position": {"x": jx, "y": jy, "z": jz},
            "properties": {
                "species": "bovine",
                "category": "cow_body",
                "part": part_id.replace("cow_", ""),
            }
        })
    return {
        "snapshot": {
            "timestamp": {"unix": ts or int(time.time())},
            "bounds": {"x": 3.2, "y": 1.8, "z": 1.0},
            "symbols": symbols,
            "metadata": {
                "scene":   "bovine_anatomy_atlas",
                "species": "holstein_dairy_cow",
                "source":  "anatomical_3d_training",
            }
        },
        "extra_labels": QUERY_LABELS,
    }


def train_anatomy(n_frames, delay=0.04):
    print(f"Training {n_frames} 3D anatomy frames …", flush=True)
    ok = 0
    for i in range(n_frames):
        try:
            post_json("/neuro/train", make_snapshot())
            ok += 1
        except Exception as e:
            print(f"  frame {i}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_frames}", flush=True)
        time.sleep(delay)
    print(f"Done: {ok}/{n_frames} accepted", flush=True)
    return ok


def check_centroids():
    req = urllib.request.Request(f"{NODE}/neuro/snapshot", method="GET")
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    c = d.get("centroids", {})
    cow_c = {k: v for k, v in c.items()
             if k.startswith("id::cow") or k.startswith("cow_")}
    print(f"Total centroids: {len(c)}   cow centroids: {len(cow_c)}", flush=True)
    for k, v in list(cow_c.items())[:6]:
        print(f"  {k}: {v}", flush=True)
    return len(cow_c)


def test_mesh(min_act=0.005, hops=4):
    r = post_json("/mesh/synthesize", {
        "query": "dairy cow Holstein bovine full body skeleton legs head neck spine anatomy",
        "hops": hops,
        "min_activation": min_act,
        "format": "json"
    })
    if r.get("verts"):
        nv = len(r["verts"])
        nf = len(r.get("faces", []))
        print(f"MESH OK: {nv} verts, {nf} faces  (v_count={r.get('vertex_count')} f_count={r.get('face_count')})",
              flush=True)
    else:
        print(f"MESH null — reason: {r.get('reason')} | activated: {r.get('activated_count')}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()

    print("=== Phase 1: current centroid state ===", flush=True)
    cow_count = check_centroids()

    if args.check_only or args.test_only:
        if not args.check_only:
            print("\n=== Testing mesh synthesis ===", flush=True)
            test_mesh()
        return

    target = args.frames
    if cow_count >= 50:
        print(f"Already have {cow_count} cow centroids — running {target//4} top-up frames", flush=True)
        train_anatomy(target // 4)
    else:
        train_anatomy(target)

    print("\n=== Phase 2: centroid state after training ===", flush=True)
    cow_count2 = check_centroids()

    print("\n=== Phase 3: mesh synthesis test ===", flush=True)
    r = test_mesh()

    if not r.get("verts"):
        print("\n>> Mesh still null — extra 200 frames + lower threshold …", flush=True)
        train_anatomy(200, delay=0.02)
        print("\n=== Phase 4: re-test mesh ===", flush=True)
        test_mesh(min_act=0.001, hops=5)


if __name__ == "__main__":
    main()
