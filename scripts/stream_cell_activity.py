#!/usr/bin/env python3
"""
Cellular Activity Stream
========================
Simulates a mesophyll cell in real-time, feeding EnvironmentSnapshot frames
into the neural fabric at ~10 fps.  Each symbol carries:
  - id          : persistent ID so the net learns continuity (cp0, mito3, ...)
  - position    : x, y, z  (normalised 0.0-1.0, z=depth)
  - velocity    : dx, dy  -- the net learns motion dynamics
  - properties  : label, scale_m, diameter_m  -- semantic + physical context

After training, GET /neuro/stream returns activation frames whose centroids
reconstruct the cell spatial layout from neural state alone.

Usage:
    python3 scripts/stream_cell_activity.py --host 192.168.1.84 --port 8090
"""

import argparse, json, math, random, time, sys
import urllib.request, urllib.error

# -- Simulation parameters ----------------------------------------------------
FPS       = 10          # frames fed per second
DT        = 1.0 / FPS  # physics timestep
BOUNDS    = {'x': 1.0, 'y': 1.0, 'z': 0.15}   # normalised cell volume

# -- Organelle definitions -----------------------------------------------------
# Each entry: id_prefix, label, count, radial_range, z_range, diameter_m, mass
# mass controls Brownian amplitude: low mass -> fast diffusion
ORGANELLE_SPECS = [
    # id_prefix  label                   n   r_min r_max  z_min z_max  diam_m   mass
    ('cp',       'chloroplast',         15,  0.18, 0.42,  0.04, 0.10,  5e-6,   4.0),
    ('mito',     'mitochondria',         8,  0.08, 0.36,  0.04, 0.10,  2e-6,   2.5),
    ('er',       'endoplasmic_reticulum',3,  0.10, 0.22,  0.05, 0.09,  0.5e-6, 6.0),
    ('ribo',     'ribosome',            40,  0.02, 0.48,  0.02, 0.13,  27e-9,  0.3),
    ('vesicle',  'vesicle',             12,  0.08, 0.40,  0.04, 0.10,  100e-9, 1.0),
    ('mt',       'microtubule_track',    8,  0.05, 0.45,  0.05, 0.08,  24e-9,  8.0),
]

# Stationary organelles (single instance each)
STATIC_SPECS = [
    # id         label           x     y     z     diam_m
    ('nucleus',  'nucleus',      0.50, 0.50, 0.07, 7e-6),
    ('nucl',     'nucleolus',    0.50, 0.50, 0.07, 1.8e-6),
    ('vacuole',  'vacuole',      0.50, 0.50, 0.07, 15e-6),
    ('golgi',    'golgi',        0.62, 0.42, 0.07, 1.5e-6),
    ('centrosome','centrosome',  0.48, 0.52, 0.07, 0.5e-6),
]

def make_organelles():
    orgs = []
    for pfx, label, n, rmin, rmax, zmin, zmax, diam_m, mass in ORGANELLE_SPECS:
        for i in range(n):
            a = (i / n) * math.pi * 2 + random.uniform(-0.4, 0.4)
            r = random.uniform(rmin, rmax)
            cx = 0.5 + r * math.cos(a)
            cy = 0.5 + r * math.sin(a)
            cz = random.uniform(zmin, zmax)
            vx = random.uniform(-0.002, 0.002) / max(mass, 1)
            vy = random.uniform(-0.002, 0.002) / max(mass, 1)
            orgs.append({
                'id':     f'{pfx}{i}',
                'label':  label,
                'x': cx, 'y': cy, 'z': cz,
                'vx': vx, 'vy': vy, 'vz': 0.0,
                'diam_m': diam_m,
                'mass':   mass,
                'mobile': True,
                # directed motion track (microtubule guide for chloroplasts)
                'track_angle': a,
            })
    for sid, label, x, y, z, diam_m in STATIC_SPECS:
        orgs.append({'id': sid, 'label': label, 'x': x, 'y': y, 'z': z,
                     'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'diam_m': diam_m,
                     'mass': 999, 'mobile': False, 'track_angle': 0.0})
    return orgs


def step_physics(orgs, t, dt=DT):
    """Advance all organelles by one timestep with biologically-motivated motion."""
    noise_scale = 0.0025  # Brownian amplitude at unit mass

    for o in orgs:
        if not o['mobile']:
            continue

        m = o['mass']

        # Brownian force -- inversely proportional to mass (Stokes-Einstein)
        bx = random.gauss(0, noise_scale / math.sqrt(m))
        by = random.gauss(0, noise_scale / math.sqrt(m))

        # Weak directed motion for chloroplasts -- cytoplasmic streaming
        if o['label'] == 'chloroplast':
            # Follow a circular track around nucleus (simulates cytoplasmic streaming)
            ang = o['track_angle'] + t * 0.04   # slow orbit
            target_x = 0.5 + math.cos(ang) * 0.28
            target_y = 0.5 + math.sin(ang) * 0.28
            bx += (target_x - o['x']) * 0.003
            by += (target_y - o['y']) * 0.003

        # Mitochondria: fission-like split then drift back
        if o['label'] == 'mitochondria':
            bx += random.gauss(0, 0.0008)
            by += random.gauss(0, 0.0008)

        # Vesicles: directed along microtubule tracks (random track assignment)
        if o['label'] == 'vesicle':
            ang = o['track_angle'] + math.sin(t * 0.5 + float(o['id'][-1])) * 0.4
            bx += math.cos(ang) * 0.0015
            by += math.sin(ang) * 0.0015

        # Ribosomes: fast diffusion (cytoplasmic crowd diffusion)
        # velocity already has large Brownian from noise_scale, just damp

        o['vx'] = (o['vx'] + bx) * 0.88   # viscous damping
        o['vy'] = (o['vy'] + by) * 0.88

        # Clamp speed
        spd = math.sqrt(o['vx']**2 + o['vy']**2)
        max_spd = 0.008 / m
        if spd > max_spd:
            o['vx'] *= max_spd / spd
            o['vy'] *= max_spd / spd

        o['x'] = max(0.03, min(0.97, o['x'] + o['vx'] * dt))
        o['y'] = max(0.03, min(0.97, o['y'] + o['vy'] * dt))

        # Z oscillation (organelle bobbing through focal plane depth)
        o['z'] = o['z'] + math.sin(t * 0.8 + hash(o['id']) % 628 * 0.01) * 0.0002

    return orgs


def build_snapshot(orgs, t):
    """Build the EnvironmentSnapshot JSON that the neural fabric understands."""
    symbols = []
    for o in orgs:
        symbols.append({
            'id':   o['id'],
            'type': 'CUSTOM',
            'position': {'x': round(o['x'], 5), 'y': round(o['y'], 5), 'z': round(o['z'], 5)},
            'velocity': {'x': round(o['vx'], 6), 'y': round(o['vy'], 6), 'z': round(o['vz'], 6)},
            'properties': {
                # Semantic label -- primary Hebbian association
                'label':     o['label'],
                # Physical dimensions -- let the EEM calibrate z-axis
                'scale_m':   str(o['diam_m']),
                'diameter_m':str(o['diam_m']),
                # Depth/layer context -- maps to overlay depth model
                'depth_class': _depth_class(o['label']),
                # Continuity -- the net learns that cp0 at t->t+1 is same object
                'track_id':  o['id'],
            }
        })

    return {
        'timestamp': int(t * 1000),
        'bounds': BOUNDS,
        'symbols': symbols,
        'metadata': {
            'context':     'mesophyll_cell',
            'modality':    'confocal_clsm_simulated',
            'frame_t':     str(round(t, 3)),
            # Scale hint for EEM z-calibration
            'cell_width_m': '20e-6',
        }
    }


def _depth_class(label):
    """Map label to conceptual depth layer -- reinforces the overlay depth model."""
    outer = {'cell_wall', 'plasma_membrane', 'vacuole'}
    membrane = {'endoplasmic_reticulum', 'mitochondria', 'golgi', 'centrosome'}
    molecular = {'ribosome', 'vesicle', 'microtubule_track'}
    if label in outer:    return 'd0_surface'
    if label == 'nucleus' or label == 'nucleolus': return 'd1_nuclear'
    if label == 'chloroplast': return 'd2_plastid'
    if label in membrane:  return 'd2_organelle'
    if label in molecular: return 'd3_molecular'
    return 'd2_organelle'


# -- HTTP ------------------------------------------------------------------------
def post(host, port, path, body):
    url  = f'http://{host}:{port}{path}'
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f'  HTTP {e.code} {path}: {e.read().decode()[:60]}', flush=True)
        return {}
    except Exception as e:
        print(f'  ERR {path}: {e}', flush=True)
        return {}


def check_connection(host, port):
    try:
        url = f'http://{host}:{port}/health'
        with urllib.request.urlopen(url, timeout=4):
            return True
    except:
        return False


# -- Main loop ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='192.168.1.84')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--fps',  type=int, default=FPS)
    args = ap.parse_args()

    print(f'Connecting to {args.host}:{args.port} ...', flush=True)
    if not check_connection(args.host, args.port):
        print('Node not reachable. Start the node first.', flush=True)
        sys.exit(1)
    print('Connected. Starting cell activity stream.', flush=True)
    print(f'  {sum(s[2] for s in ORGANELLE_SPECS) + len(STATIC_SPECS)} organelles tracked', flush=True)
    print(f'  {args.fps} fps  --  GET /neuro/stream for activation state', flush=True)
    print('Press Ctrl+C to stop.', flush=True)

    orgs = make_organelles()
    frame = 0
    t     = 0.0
    interval = 1.0 / args.fps
    drift = 0.0

    while True:
        t0 = time.perf_counter()

        # Advance physics
        orgs = step_physics(orgs, t)

        # Build and send snapshot
        snap = build_snapshot(orgs, t)
        result = post(args.host, args.port, '/neuro/train', {'snapshot': snap})

        # Progress readout every 50 frames
        if frame % 50 == 0:
            lc = result.get('label_count', '?')
            print(f'  frame {frame:06d}  t={t:.1f}s  labels={lc}', flush=True)

        frame += 1
        t     += DT

        # Rate limiting -- maintain target fps
        elapsed = time.perf_counter() - t0
        wait    = interval - elapsed - drift
        if wait > 0:
            time.sleep(wait)
            drift = 0.0
        else:
            drift = -wait * 0.1  # bleed off accumulated latency


if __name__ == '__main__':
    main()
