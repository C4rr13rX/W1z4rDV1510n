#!/usr/bin/env python3
"""
Video Scene Simulator
=====================
Simulates what a real video-to-neural pipeline would produce -- sends
EnvironmentSnapshot frames as if frames of video are being decoded and
the detected regions are being forwarded to the neural fabric.

Two built-in scenes:
  --scene field   : cow grazing in a field (tests object + motion learning)
  --scene cell    : (use stream_cell_activity.py for this instead)

The pipeline is identical to what a real video stream would use:
  Video frame -> detect regions -> extract (id, x, y, velocity, label, scale_m)
  -> POST /neuro/train

After training, GET /neuro/stream returns centroid positions that reconstruct
the spatial layout -- positions ARE learned from the stream, not echoed.
The world_viewer.html reads that stream and builds the 3D world from it.

Usage:
    python3 scripts/stream_video_sim.py --host 192.168.1.84 --port 8090 --scene field
"""

import argparse, json, math, random, time, sys
import urllib.request, urllib.error

FPS      = 10
DT       = 1.0 / FPS
BOUNDS   = {'x': 1.0, 'y': 1.0, 'z': 0.30}   # normalised scene volume (z = depth)


# -- Scene definitions ---------------------------------------------------------
# Each entry: (id, label, x0, y0, z0, diam_m, mass, mobile, behaviour)
# behaviour: string key used in step_physics

FIELD_SCENE = [
    # Cow body -- large, walks in a slow arc
    ('cow_body',    'cow_body',           0.50, 0.60, 0.18, 2.0,   10.0, True,  'cow_walk'),
    ('cow_head',    'cow_head',           0.50, 0.60, 0.22, 0.5,    8.0, True,  'cow_head'),
    ('cow_tail',    'cow_tail',           0.50, 0.60, 0.14, 0.1,    6.0, True,  'cow_tail'),
    ('cow_leg_fl',  'cow_leg',            0.50, 0.60, 0.06, 0.08,   5.0, True,  'leg_fl'),
    ('cow_leg_fr',  'cow_leg',            0.50, 0.60, 0.06, 0.08,   5.0, True,  'leg_fr'),
    ('cow_leg_rl',  'cow_leg',            0.50, 0.60, 0.06, 0.08,   5.0, True,  'leg_rl'),
    ('cow_leg_rr',  'cow_leg',            0.50, 0.60, 0.06, 0.08,   5.0, True,  'leg_rr'),
    ('cow_eye_l',   'cow_eye',            0.50, 0.60, 0.24, 0.04,   4.0, True,  'eye'),
    ('cow_eye_r',   'cow_eye',            0.50, 0.60, 0.24, 0.04,   4.0, True,  'eye'),
    # Grass patches -- many, stationary with subtle wind sway
    ('grass_0',     'grass',              0.10, 0.80, 0.02, 0.3,   20.0, True,  'grass'),
    ('grass_1',     'grass',              0.22, 0.82, 0.02, 0.25,  20.0, True,  'grass'),
    ('grass_2',     'grass',              0.34, 0.78, 0.02, 0.28,  20.0, True,  'grass'),
    ('grass_3',     'grass',              0.48, 0.85, 0.02, 0.30,  20.0, True,  'grass'),
    ('grass_4',     'grass',              0.62, 0.80, 0.02, 0.26,  20.0, True,  'grass'),
    ('grass_5',     'grass',              0.75, 0.83, 0.02, 0.29,  20.0, True,  'grass'),
    ('grass_6',     'grass',              0.88, 0.79, 0.02, 0.27,  20.0, True,  'grass'),
    ('grass_7',     'grass',              0.15, 0.92, 0.02, 0.24,  20.0, True,  'grass'),
    ('grass_8',     'grass',              0.42, 0.90, 0.02, 0.31,  20.0, True,  'grass'),
    ('grass_9',     'grass',              0.70, 0.88, 0.02, 0.25,  20.0, True,  'grass'),
    # Trees -- tall, anchored, slight sway
    ('tree_0',      'tree',               0.08, 0.40, 0.25, 0.8,   30.0, True,  'tree'),
    ('tree_1',      'tree',               0.88, 0.35, 0.28, 0.9,   30.0, True,  'tree'),
    ('tree_2',      'tree',               0.50, 0.15, 0.30, 0.7,   30.0, True,  'tree'),
    # Sky gradient
    ('sky',         'sky',                0.50, 0.10, 0.29, 5.0,   50.0, False, 'static'),
    # Sun / light source
    ('sun',         'sun',                0.80, 0.08, 0.29, 0.8,   50.0, False, 'static'),
    # Ground / horizon
    ('ground',      'ground',             0.50, 0.95, 0.00, 8.0,   50.0, False, 'static'),
    # Distant hill
    ('hill',        'hill',               0.30, 0.65, 0.12, 3.0,   40.0, True,  'sway_slow'),
    # Bird (occasional, appears and disappears)
    ('bird_0',      'bird',               0.20, 0.20, 0.28, 0.15,   2.0, True,  'bird'),
    ('bird_1',      'bird',               0.25, 0.18, 0.28, 0.12,   2.0, True,  'bird'),
]

DEPTH_MAP = {
    'sky': 'd0_sky', 'sun': 'd0_sky', 'hill': 'd0_background',
    'ground': 'd0_surface', 'grass': 'd1_ground',
    'tree': 'd1_midground', 'cow_body': 'd2_subject',
    'cow_head': 'd2_subject', 'cow_tail': 'd2_subject',
    'cow_leg': 'd2_subject', 'cow_eye': 'd2_detail',
    'bird': 'd1_midground',
}

def depth_class(label):
    for k, v in DEPTH_MAP.items():
        if label.startswith(k): return v
    return 'd2_subject'


# -- Physics -------------------------------------------------------------------
def make_objects(scene_spec):
    objs = []
    for sid, label, x, y, z, diam_m, mass, mobile, beh in scene_spec:
        objs.append({
            'id':      sid, 'label': label,
            'x': x,   'y': y,  'z': z,
            'vx': 0.0,'vy': 0.0,'vz': 0.0,
            'diam_m': diam_m, 'mass': mass,
            'mobile': mobile, 'beh': beh,
        })
    return objs


def step_physics(objs, t, dt=DT):
    """Biologically / physically motivated motion for each entity type."""
    # Cow walk orbit parameters
    cow_orbit_speed = 0.06   # radians per second
    cow_orbit_r     = 0.22   # radius around (0.5, 0.6)
    cow_body_x = 0.5 + cow_orbit_r * math.cos(t * cow_orbit_speed)
    cow_body_y = 0.6 + cow_orbit_r * math.sin(t * cow_orbit_speed) * 0.4

    for o in objs:
        if not o['mobile']:
            continue
        beh = o['beh']

        if beh == 'cow_walk':
            # Smooth orbit
            target_x = cow_body_x
            target_y = cow_body_y
            o['vx'] = (target_x - o['x']) * 0.15
            o['vy'] = (target_y - o['y']) * 0.15
            o['x'] = max(0.02, min(0.98, o['x'] + o['vx'] * dt))
            o['y'] = max(0.02, min(0.98, o['y'] + o['vy'] * dt))
            # Slight breathing oscillation in z
            o['z'] = 0.18 + math.sin(t * 1.2) * 0.004

        elif beh == 'cow_head':
            # Follows body + bobbing
            target_x = cow_body_x + math.cos(t * cow_orbit_speed) * 0.065
            target_y = cow_body_y + math.sin(t * 1.8) * 0.018 - 0.04
            o['vx'] = (target_x - o['x']) * 0.18
            o['vy'] = (target_y - o['y']) * 0.18
            o['x'] = max(0.02, min(0.98, o['x'] + o['vx'] * dt))
            o['y'] = max(0.02, min(0.98, o['y'] + o['vy'] * dt))
            o['z'] = 0.22 + math.sin(t * 1.8) * 0.008  # nod

        elif beh == 'cow_tail':
            target_x = cow_body_x - math.cos(t * cow_orbit_speed) * 0.060
            target_y = cow_body_y + math.sin(t * 2.5) * 0.020
            o['vx'] = (target_x - o['x']) * 0.20
            o['vy'] = (target_y - o['y']) * 0.20
            o['x'] = max(0.02, min(0.98, o['x'] + o['vx'] * dt))
            o['y'] = max(0.02, min(0.98, o['y'] + o['vy'] * dt))
            o['z'] = 0.14 + math.sin(t * 3.0) * 0.015  # swish

        elif beh.startswith('leg_'):
            # 4-beat walking gait: FL, RR in phase; FR, RL in phase
            phase_offset = {'leg_fl': 0.0, 'leg_fr': math.pi, 'leg_rl': math.pi, 'leg_rr': 0.0}
            phase = phase_offset.get(beh, 0.0)
            stride_freq = 1.4   # steps per second
            stride_amp  = 0.022  # stride width in normalised units
            body_offset = {'leg_fl': (-0.04, 0.04), 'leg_fr': (0.04, 0.04),
                           'leg_rl': (-0.04, -0.04), 'leg_rr': (0.04, -0.04)}
            bx, by = body_offset.get(beh, (0,0))
            target_x = cow_body_x + bx + math.sin(t * stride_freq * math.pi * 2 + phase) * stride_amp
            target_y = cow_body_y + by
            o['vx'] = (target_x - o['x']) * 0.25
            o['vy'] = (target_y - o['y']) * 0.25
            o['x'] = max(0.02, min(0.98, o['x'] + o['vx'] * dt))
            o['y'] = max(0.02, min(0.98, o['y'] + o['vy'] * dt))
            o['z'] = max(0.01, 0.06 + abs(math.sin(t * stride_freq * math.pi * 2 + phase)) * 0.04)

        elif beh == 'eye':
            is_left = 'l' in o['id']
            ex_off = -0.025 if is_left else 0.025
            ey_off = 0.016
            target_x = cow_body_x + math.cos(t * cow_orbit_speed) * 0.058 + ex_off
            target_y = cow_body_y + ey_off + math.sin(t * 1.8) * 0.012
            o['vx'] = (target_x - o['x']) * 0.20
            o['vy'] = (target_y - o['y']) * 0.20
            o['x'] = max(0.02, min(0.98, o['x'] + o['vx'] * dt))
            o['y'] = max(0.02, min(0.98, o['y'] + o['vy'] * dt))
            o['z'] = 0.24

        elif beh == 'grass':
            # Barely moves -- wind sway in z
            seed = sum(ord(c) for c in o['id'])
            o['z'] = 0.02 + math.sin(t * 0.8 + seed * 0.1) * 0.003
            o['x'] = o['x'] + math.sin(t * 1.1 + seed * 0.2) * 0.0003
            o['vx'] = 0; o['vy'] = 0

        elif beh == 'tree':
            seed = sum(ord(c) for c in o['id'])
            o['z'] = o['z'] + math.sin(t * 0.5 + seed * 0.15) * 0.0005
            o['x'] = o['x'] + math.sin(t * 0.4 + seed * 0.1) * 0.0002
            o['vx'] = 0; o['vy'] = 0

        elif beh == 'bird':
            seed = sum(ord(c) for c in o['id'])
            # Birds drift in arcs across the sky
            o['x'] = 0.5 + 0.4 * math.cos(t * 0.12 + seed)
            o['y'] = 0.15 + 0.08 * math.sin(t * 0.25 + seed)
            o['z'] = 0.28 + math.sin(t * 0.3 + seed) * 0.01
            o['vx'] = -math.sin(t * 0.12 + seed) * 0.12 * 0.12
            o['vy'] = math.cos(t * 0.25 + seed) * 0.08 * 0.25

        elif beh == 'sway_slow':
            seed = sum(ord(c) for c in o['id'])
            o['x'] = o['x'] + math.sin(t * 0.3 + seed) * 0.0003
            o['vx'] = 0; o['vy'] = 0

    return objs


def build_snapshot(objs, t, scene_name='field'):
    symbols = []
    for o in objs:
        symbols.append({
            'id':   o['id'],
            'type': 'CUSTOM',
            'position': {
                'x': round(o['x'], 5),
                'y': round(o['y'], 5),
                'z': round(o['z'], 5),
            },
            'velocity': {
                'x': round(o['vx'], 6),
                'y': round(o['vy'], 6),
                'z': round(o.get('vz', 0.0), 6),
            },
            'properties': {
                # Only string values that should become neural labels
                'label':    o['label'],
                'scene':    scene_name,
                # Numeric values are ignored by the label extractor
                'scale_m':    o['diam_m'],
                'diameter_m': o['diam_m'],
                # track_id kept for world viewer identification (string but prefixed
                # with entity id so it won't match any SCENE_RENDERERS key)
                'track_id': o['id'],
            }
        })

    return {
        'timestamp': {'unix': int(t * 1000)},
        'bounds':    BOUNDS,
        'symbols':   symbols,
        'metadata': {
            'context':  scene_name,
            'modality': 'video_stream_simulated',
        }
    }


# -- HTTP ----------------------------------------------------------------------
def post(host, port, path, body):
    url  = f'http://{host}:{port}{path}'
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f'  HTTP {e.code}: {e.read().decode()[:60]}', flush=True)
        return {}
    except Exception as e:
        print(f'  ERR: {e}', flush=True)
        return {}


def check_connection(host, port):
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/health', timeout=4):
            return True
    except:
        return False


# -- Main ----------------------------------------------------------------------
SCENES = {'field': FIELD_SCENE}

def main():
    ap = argparse.ArgumentParser(description='Video scene simulator for W1z4rD neural fabric')
    ap.add_argument('--host',  default='192.168.1.84')
    ap.add_argument('--port',  type=int, default=8090)
    ap.add_argument('--fps',   type=int, default=FPS)
    ap.add_argument('--scene', choices=list(SCENES.keys()), default='field')
    args = ap.parse_args()

    print(f'Connecting to {args.host}:{args.port} ...', flush=True)
    if not check_connection(args.host, args.port):
        print('Node not reachable. Start the node first.', flush=True)
        sys.exit(1)

    spec = SCENES[args.scene]
    print(f'Connected. Starting [{args.scene}] simulation.', flush=True)
    print(f'  {len(spec)} tracked entities', flush=True)
    print(f'  {args.fps} fps -> POST /neuro/train each frame', flush=True)
    print(f'  GET /neuro/stream -> world_viewer.html reads this live', flush=True)
    print('Press Ctrl+C to stop.', flush=True)

    objs     = make_objects(spec)
    frame    = 0
    t        = 0.0
    interval = 1.0 / args.fps
    drift    = 0.0

    while True:
        t0 = time.perf_counter()
        objs = step_physics(objs, t)
        snap = build_snapshot(objs, t, args.scene)
        result = post(args.host, args.port, '/neuro/train', {'snapshot': snap})

        if frame % 50 == 0:
            lc = result.get('label_count', '?')
            print(f'  frame {frame:06d}  t={t:.1f}s  labels={lc}', flush=True)

        frame += 1
        t     += DT
        elapsed = time.perf_counter() - t0
        wait    = interval - elapsed - drift
        if wait > 0:
            time.sleep(wait)
            drift = 0.0
        else:
            drift = -wait * 0.1

if __name__ == '__main__':
    main()
