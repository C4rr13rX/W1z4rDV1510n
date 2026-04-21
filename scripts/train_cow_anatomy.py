#!/usr/bin/env python3
# coding: utf-8
"""
Cow anatomy + behaviour + lifecycle training corpus for the W1z4rD neural fabric.

Posts structured EnvironmentSnapshot data to /neuro/train so the Hebbian fabric
learns:
  - Complete 3D skeleton (50+ landmarks, normalised 0-1 XYZ per body part)
  - Multiple behavioural poses  (standing, grazing, walking, lying, running)
  - Behavioural sequences and movement physics
  - Environment context (field, barn, pasture, water)
  - Biology / taxonomy (class, order, anatomy, physiology)
  - Lifecycle stages (calf -> heifer -> cow -> elder)
  - Physical constraints (scale_m, mass_kg, speed_ms, gait)

The fabric learns centroid positions for every label, building a spatial model
of what a cow IS -- so inference can reconstruct a full 3D cow from partial
sensor data (e.g. a 2-D video frame identifying position + orientation).

Usage:
    python train_cow_anatomy.py [--node HOST:PORT] [--reps N]
"""

import argparse
import json
import time
import urllib.request

DEFAULT_NODE = 'localhost:8090'
DEFAULT_REPS = 3     # repeat full corpus N times to strengthen associations


# -- Cow anatomy: body part positions in 3 poses -------------------------------
# Coordinate frame: X=left/right (0=left, 0.5=centre, 1=right)
#                   Y=up/down   (0=ground, 1=tallest point)
#                   Z=front/back (0=rear, 1=nose)
# All positions are for an average adult Holstein/Hereford dairy/beef cow
# facing +Z (nose toward viewer), body centred at (0.5, ?, 0.5).

# Standing reference pose
_STAND = {
    # -- Head / face --------------------------------------------------------
    'head':           (0.50, 0.86, 0.88),
    'muzzle':         (0.50, 0.76, 0.97),
    'lower_jaw':      (0.50, 0.74, 0.95),
    'nostrils_L':     (0.46, 0.78, 0.97),
    'nostrils_R':     (0.54, 0.78, 0.97),
    'eye_L':          (0.40, 0.87, 0.92),
    'eye_R':          (0.60, 0.87, 0.92),
    'ear_L':          (0.37, 0.94, 0.85),
    'ear_R':          (0.63, 0.94, 0.85),
    'forehead':       (0.50, 0.92, 0.88),
    'poll':           (0.50, 0.96, 0.85),
    'horn_L':         (0.41, 0.98, 0.83),   # not all breeds
    'horn_R':         (0.59, 0.98, 0.83),
    # -- Neck / throat ------------------------------------------------------
    'neck':           (0.50, 0.86, 0.77),
    'throat':         (0.50, 0.79, 0.80),
    'dewlap':         (0.50, 0.70, 0.80),
    'crest':          (0.50, 0.92, 0.73),
    # -- Trunk / spine ------------------------------------------------------
    'withers':        (0.50, 0.94, 0.68),
    'back':           (0.50, 0.92, 0.53),
    'loin':           (0.50, 0.90, 0.40),
    'rump':           (0.50, 0.88, 0.28),
    'tail_root':      (0.50, 0.84, 0.22),
    'tail_mid':       (0.50, 0.70, 0.15),
    'tail_switch':    (0.50, 0.54, 0.10),
    # -- Chest / belly ------------------------------------------------------
    'brisket':        (0.50, 0.68, 0.72),
    'sternum':        (0.50, 0.62, 0.65),
    'flank_L':        (0.36, 0.68, 0.42),
    'flank_R':        (0.64, 0.68, 0.42),
    'belly':          (0.50, 0.54, 0.48),
    'navel':          (0.50, 0.52, 0.50),
    # -- Udder / reproduction -----------------------------------------------
    'udder':          (0.50, 0.44, 0.42),
    'teat_LF':        (0.44, 0.38, 0.45),
    'teat_RF':        (0.56, 0.38, 0.45),
    'teat_LR':        (0.44, 0.38, 0.38),
    'teat_RR':        (0.56, 0.38, 0.38),
    # -- Front legs (L = left, R = right) -----------------------------------
    'shoulder_L':     (0.35, 0.84, 0.68),
    'shoulder_R':     (0.65, 0.84, 0.68),
    'elbow_L':        (0.35, 0.68, 0.68),
    'elbow_R':        (0.65, 0.68, 0.68),
    'front_cannon_L': (0.36, 0.50, 0.67),
    'front_cannon_R': (0.64, 0.50, 0.67),
    'front_fetlock_L':(0.36, 0.30, 0.66),
    'front_fetlock_R':(0.64, 0.30, 0.66),
    'front_hoof_L':   (0.36, 0.04, 0.66),
    'front_hoof_R':   (0.64, 0.04, 0.66),
    # -- Rear legs ----------------------------------------------------------
    'hip_L':          (0.37, 0.84, 0.30),
    'hip_R':          (0.63, 0.84, 0.30),
    'stifle_L':       (0.37, 0.66, 0.30),
    'stifle_R':       (0.63, 0.66, 0.30),
    'hock_L':         (0.37, 0.44, 0.26),
    'hock_R':         (0.63, 0.44, 0.26),
    'rear_cannon_L':  (0.37, 0.28, 0.24),
    'rear_cannon_R':  (0.63, 0.28, 0.24),
    'rear_fetlock_L': (0.37, 0.18, 0.22),
    'rear_fetlock_R': (0.63, 0.18, 0.22),
    'rear_hoof_L':    (0.37, 0.04, 0.20),
    'rear_hoof_R':    (0.63, 0.04, 0.20),
    # -- Centre of mass -----------------------------------------------------
    'body_centre':    (0.50, 0.70, 0.50),
    'centre_of_mass': (0.50, 0.64, 0.50),
}

def _pose_delta(base: dict, deltas: dict) -> dict:
    """Return a copy of base with per-key coordinate deltas applied."""
    p = dict(base)
    for k, (dx, dy, dz) in deltas.items():
        x, y, z = p[k]
        p[k] = (min(1, max(0, x+dx)), min(1, max(0, y+dy)), min(1, max(0, z+dz)))
    return p

# -- Grazing pose: head / neck dropped to ground level ------------------------
_GRAZE = _pose_delta(_STAND, {
    'head':           ( 0.00, -0.42,  0.06),
    'muzzle':         ( 0.00, -0.70,  0.02),
    'lower_jaw':      ( 0.00, -0.70,  0.02),
    'nostrils_L':     ( 0.00, -0.69,  0.01),
    'nostrils_R':     ( 0.00, -0.69,  0.01),
    'eye_L':          ( 0.00, -0.44,  0.05),
    'eye_R':          ( 0.00, -0.44,  0.05),
    'ear_L':          ( 0.00, -0.34,  0.03),
    'ear_R':          ( 0.00, -0.34,  0.03),
    'forehead':       ( 0.00, -0.40,  0.04),
    'poll':           ( 0.00, -0.32,  0.02),
    'neck':           ( 0.00, -0.24,  0.04),
    'throat':         ( 0.00, -0.20,  0.03),
    'crest':          ( 0.00, -0.18,  0.02),
    # front legs spread for balance
    'front_hoof_L':   (-0.04,  0.00,  0.06),
    'front_hoof_R':   ( 0.04,  0.00,  0.06),
    'front_fetlock_L':(-0.03,  0.00,  0.04),
    'front_fetlock_R':( 0.03,  0.00,  0.04),
})

# -- Walking step A: left-front / right-rear forward --------------------------
_WALK_A = _pose_delta(_STAND, {
    # Left front forward
    'front_hoof_L':   (-0.01, +0.08, +0.16),
    'front_fetlock_L':(-0.01, +0.08, +0.12),
    'front_cannon_L': ( 0.00, +0.04, +0.08),
    'elbow_L':        ( 0.00, +0.04, +0.04),
    # Right rear forward
    'rear_hoof_R':    ( 0.00, +0.08, +0.12),
    'rear_fetlock_R': ( 0.00, +0.06, +0.09),
    'rear_cannon_R':  ( 0.00, +0.04, +0.06),
    # Body slight tilt
    'body_centre':    ( 0.00,  0.00,  0.00),
})

# -- Walking step B: right-front / left-rear forward --------------------------
_WALK_B = _pose_delta(_STAND, {
    'front_hoof_R':   ( 0.01, +0.08, +0.16),
    'front_fetlock_R':( 0.01, +0.08, +0.12),
    'front_cannon_R': ( 0.00, +0.04, +0.08),
    'elbow_R':        ( 0.00, +0.04, +0.04),
    'rear_hoof_L':    ( 0.00, +0.08, +0.12),
    'rear_fetlock_L': ( 0.00, +0.06, +0.09),
    'rear_cannon_L':  ( 0.00, +0.04, +0.06),
})

# -- Lying down ----------------------------------------------------------------
_LIE = _pose_delta(_STAND, {
    'body_centre':    ( 0.00, -0.44,  0.00),
    'centre_of_mass': ( 0.00, -0.44,  0.00),
    'belly':          ( 0.00, -0.44,  0.00),
    'brisket':        ( 0.00, -0.44,  0.00),
    # legs folded under
    'front_hoof_L':   ( 0.06, +0.38, -0.10),
    'front_hoof_R':   (-0.06, +0.38, -0.10),
    'front_fetlock_L':( 0.04, +0.24, -0.08),
    'front_fetlock_R':(-0.04, +0.24, -0.08),
    'front_cannon_L': ( 0.02, +0.16, -0.04),
    'front_cannon_R': (-0.02, +0.16, -0.04),
    'rear_hoof_L':    ( 0.06, +0.32, +0.08),
    'rear_hoof_R':    (-0.06, +0.32, +0.08),
    'rear_fetlock_L': ( 0.04, +0.18, +0.06),
    'rear_fetlock_R': (-0.04, +0.18, +0.06),
    # head slightly raised
    'head':           ( 0.00, +0.04,  0.00),
})

POSES = {
    'standing': _STAND,
    'grazing':  _GRAZE,
    'walking_a': _WALK_A,
    'walking_b': _WALK_B,
    'lying':    _LIE,
}


# -- Biology / taxonomy knowledge ----------------------------------------------
BIOLOGY = [
    # Taxonomy
    ('taxonomy_kingdom',  'Animalia',    (0.50, 0.99, 0.50)),
    ('taxonomy_phylum',   'Chordata',    (0.50, 0.98, 0.50)),
    ('taxonomy_class',    'Mammalia',    (0.50, 0.97, 0.50)),
    ('taxonomy_order',    'Artiodactyla',(0.50, 0.96, 0.50)),
    ('taxonomy_family',   'Bovidae',     (0.50, 0.95, 0.50)),
    ('taxonomy_genus',    'Bos',         (0.50, 0.94, 0.50)),
    ('taxonomy_species',  'Bos_taurus',  (0.50, 0.93, 0.50)),
    # Anatomy facts (stored as spatial references to body regions)
    ('organ_rumen',       'digestive',   (0.50, 0.60, 0.45)),
    ('organ_reticulum',   'digestive',   (0.50, 0.58, 0.55)),
    ('organ_omasum',      'digestive',   (0.50, 0.56, 0.52)),
    ('organ_abomasum',    'digestive',   (0.50, 0.54, 0.48)),
    ('organ_heart',       'circulatory', (0.50, 0.72, 0.62)),
    ('organ_lungs',       'respiratory', (0.50, 0.78, 0.60)),
    ('organ_liver',       'digestive',   (0.48, 0.64, 0.46)),
    ('organ_kidneys',     'excretory',   (0.50, 0.68, 0.38)),
    ('organ_brain',       'nervous',     (0.50, 0.93, 0.88)),
    ('organ_spinalcord',  'nervous',     (0.50, 0.90, 0.50)),
    # Physical characteristics
    ('adult_height_m',    '1.4',         (0.50, 1.00, 0.50)),
    ('adult_length_m',    '2.4',         (0.50, 0.50, 1.00)),
    ('adult_mass_kg',     '600',         (0.50, 0.64, 0.50)),
    ('lifespan_years',    '20',          (0.50, 0.50, 0.50)),
    ('gestation_days',    '283',         (0.50, 0.44, 0.44)),
    ('gait_walk_ms',      '1.4',         (0.50, 0.20, 0.80)),
    ('gait_trot_ms',      '3.5',         (0.50, 0.30, 0.90)),
    ('gait_gallop_ms',    '8.0',         (0.50, 0.40, 0.95)),
    # Sensory
    ('vision_fov_deg',    '330',         (0.50, 0.87, 0.92)),
    ('hearing_hz_max',    '35000',       (0.50, 0.94, 0.85)),
    ('smell_range_m',     '10000',       (0.50, 0.76, 0.97)),
]

# -- Lifecycle stages -----------------------------------------------------------
LIFECYCLE = [
    # (stage, description, size_factor relative to adult, y_scale)
    ('lifecycle_embryo',   'embryo',         (0.50, 0.50, 0.50)),
    ('lifecycle_fetus',    'fetus',          (0.50, 0.52, 0.52)),
    ('lifecycle_newborn',  'calf_newborn',   (0.50, 0.40, 0.50)),
    ('lifecycle_calf',     'calf_3mo',       (0.50, 0.50, 0.50)),
    ('lifecycle_weaned',   'calf_6mo',       (0.50, 0.60, 0.50)),
    ('lifecycle_yearling', 'yearling',       (0.50, 0.75, 0.50)),
    ('lifecycle_heifer',   'heifer_18mo',    (0.50, 0.88, 0.50)),
    ('lifecycle_adult',    'cow_adult',      (0.50, 0.94, 0.50)),
    ('lifecycle_elder',    'cow_elder',      (0.50, 0.90, 0.50)),
]

# -- Environment contexts -------------------------------------------------------
ENVIRONMENTS = [
    # (env_id, description, x, y, z) -- position in the world space
    ('env_grass_field',   'open_pasture',    (0.50, 0.02, 0.50)),
    ('env_sky',           'sky_above',       (0.50, 0.95, 0.50)),
    ('env_trees_distant', 'treeline_far',    (0.50, 0.55, 0.05)),
    ('env_fence',         'pasture_fence',   (0.50, 0.30, 0.02)),
    ('env_water_trough',  'water_source',    (0.20, 0.10, 0.30)),
    ('env_barn',          'shelter',         (0.80, 0.50, 0.10)),
    ('env_mud_patch',     'wet_ground',      (0.50, 0.01, 0.60)),
    ('env_shade_tree',    'shade',           (0.30, 0.60, 0.70)),
    ('env_salt_lick',     'mineral_source',  (0.35, 0.12, 0.40)),
    ('env_hay_feeder',    'feed_source',     (0.25, 0.35, 0.40)),
]

# -- Behaviours -----------------------------------------------------------------
BEHAVIOURS = [
    # (behaviour_id, context_description, position_hint)
    ('behaviour_grazing',    'eating_grass_head_down',  (0.50, 0.30, 0.80)),
    ('behaviour_ruminating', 'chewing_cud_standing',    (0.50, 0.70, 0.50)),
    ('behaviour_walking',    'slow_patrol_field',       (0.50, 0.50, 0.70)),
    ('behaviour_trotting',   'alert_medium_speed',      (0.50, 0.55, 0.80)),
    ('behaviour_galloping',  'fleeing_high_speed',      (0.50, 0.60, 0.95)),
    ('behaviour_lying',      'resting_ruminating',      (0.50, 0.25, 0.40)),
    ('behaviour_drinking',   'head_at_water_trough',    (0.30, 0.40, 0.40)),
    ('behaviour_licking',    'salt_lick_minerals',      (0.38, 0.55, 0.45)),
    ('behaviour_nursing',    'calf_drinking_milk',      (0.50, 0.40, 0.45)),
    ('behaviour_socialising','herd_proximity',          (0.50, 0.65, 0.50)),
    ('behaviour_grooming',   'self_grooming_tongue',    (0.50, 0.75, 0.70)),
    ('behaviour_scratching', 'rubbing_post_skin_care',  (0.50, 0.70, 0.30)),
    ('behaviour_alert',      'head_up_scanning',        (0.50, 0.90, 0.80)),
    ('behaviour_defecating', 'tail_raised_waste',       (0.50, 0.72, 0.18)),
    ('behaviour_mooing',     'vocalisation_contact',    (0.50, 0.76, 0.97)),
    ('behaviour_bellowing',  'vocalisation_distress',   (0.50, 0.76, 0.97)),
    ('behaviour_calving',    'parturition',             (0.50, 0.20, 0.35)),
    ('behaviour_mounting',   'reproductive_behaviour',  (0.50, 0.80, 0.45)),
    ('behaviour_head_butt',  'dominance_fight',         (0.50, 0.90, 0.90)),
]

# -- Skin / coat colour patterns ------------------------------------------------
COAT_PATTERNS = [
    ('coat_colour_black',    'Holstein_Friesian',  (0.10, 0.10, 0.10)),
    ('coat_colour_brown',    'Hereford_body',      (0.60, 0.30, 0.10)),
    ('coat_colour_white',    'Hereford_face',      (0.95, 0.95, 0.90)),
    ('coat_colour_red',      'Angus_Red',          (0.55, 0.20, 0.10)),
    ('coat_colour_dun',      'Highland_dun',       (0.70, 0.55, 0.30)),
    ('coat_colour_brindle',  'mixed_pattern',      (0.40, 0.30, 0.20)),
    ('coat_texture_short',   'summer_coat',        (0.50, 0.50, 0.50)),
    ('coat_texture_long',    'winter_coat_thick',  (0.50, 0.50, 0.50)),
]


# -- Helpers -------------------------------------------------------------------
def _sym(sym_id: str, sym_type: str, x: float, y: float, z: float,
         props: dict | None = None) -> dict:
    p = {'label': sym_id, 'type': sym_type}
    if props:
        p.update(props)
    return {
        'id':         sym_id,
        'type':       'CUSTOM',
        'position':   {'x': x, 'y': y, 'z': z},
        'velocity':   {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'properties': p,
    }


def build_snapshot(symbols: list, context: str, scene: str) -> dict:
    return {
        'timestamp': {'unix': int(time.time() * 1000)},
        'bounds':    {'x': 1.0, 'y': 1.0, 'z': 1.0},
        'symbols':   symbols,
        'metadata':  {'context': context, 'scene': scene, 'modality': 'anatomy_training'},
    }


def post_snapshot(node_host: str, snapshot: dict) -> bool:
    body = json.dumps({'snapshot': snapshot}).encode()
    req  = urllib.request.Request(
        f'http://{node_host}/neuro/train', data=body,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception as e:
        print(f'  POST failed: {e}')
        return False


# -- Build training batches -----------------------------------------------------
def anatomy_snapshots() -> list:
    snaps = []

    for pose_name, pose in POSES.items():
        syms = []
        for part, (x, y, z) in pose.items():
            syms.append(_sym(
                f'cow_{part}', 'ANATOMY', x, y, z,
                {'part': part, 'pose': pose_name, 'species': 'bos_taurus',
                 'scale_m': 1.4, 'type_label': 'body_part'},
            ))
        # Top-level cow identity symbol at centre of mass
        cm = pose.get('centre_of_mass', (0.5, 0.64, 0.5))
        syms.append(_sym('cow', 'ENTITY', *cm,
            {'label': 'cow', 'pose': pose_name, 'species': 'bos_taurus',
             'scale_m': 2.4, 'type_label': 'animal'}))
        snaps.append(build_snapshot(syms, pose_name, 'anatomy'))

    return snaps


def biology_snapshots() -> list:
    syms = [_sym(k, 'BIOLOGY', x, y, z, {'label': k, 'value': v, 'type_label': 'biology'})
            for k, v, (x, y, z) in BIOLOGY]
    return [build_snapshot(syms, 'biology', 'taxonomy')]


def lifecycle_snapshots() -> list:
    snaps = []
    for stage, desc, (x, y, z) in LIFECYCLE:
        syms = [_sym(stage, 'LIFECYCLE', x, y, z,
                     {'label': stage, 'description': desc, 'type_label': 'lifecycle'})]
        snaps.append(build_snapshot(syms, stage, 'lifecycle'))
    return snaps


def environment_snapshots() -> list:
    syms = [_sym(env_id, 'ENVIRONMENT', x, y, z,
                 {'label': env_id, 'description': desc, 'type_label': 'environment'})
            for env_id, desc, (x, y, z) in ENVIRONMENTS]
    return [build_snapshot(syms, 'field_environment', 'environment')]


def behaviour_snapshots() -> list:
    snaps = []
    for beh_id, desc, (x, y, z) in BEHAVIOURS:
        syms = [_sym(beh_id, 'BEHAVIOUR', x, y, z,
                     {'label': beh_id, 'description': desc, 'type_label': 'behaviour'})]
        snaps.append(build_snapshot(syms, beh_id, 'behaviour'))
    return snaps


def coat_snapshots() -> list:
    syms = [_sym(cid, 'VISUAL', x, y, z,
                 {'label': cid, 'description': desc, 'type_label': 'coat'})
            for cid, desc, (x, y, z) in COAT_PATTERNS]
    return [build_snapshot(syms, 'coat_patterns', 'visual')]


def movement_sequence_snapshots() -> list:
    """
    Post a walking gait cycle as a rapid sequence so the fabric learns
    temporal associations between walking_a <-> walking_b steps.
    """
    snaps = []
    for _ in range(8):   # 4 full strides
        for pose_name in ('walking_a', 'standing', 'walking_b', 'standing'):
            pose = POSES.get(pose_name, POSES['standing'])
            syms = []
            for part, (x, y, z) in pose.items():
                syms.append(_sym(f'cow_{part}', 'ANATOMY', x, y, z,
                    {'part': part, 'pose': pose_name, 'motion': 'walking', 'type_label': 'body_part'}))
            syms.append(_sym('cow_motion', 'ENTITY',
                             *pose.get('centre_of_mass', (0.5, 0.64, 0.5)),
                             {'label': 'cow_motion', 'pose': pose_name, 'type_label': 'motion'}))
            snaps.append(build_snapshot(syms, f'walk_cycle_{pose_name}', 'motion_sequence'))
    return snaps


# -- Main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--node', default=DEFAULT_NODE)
    ap.add_argument('--reps', type=int, default=DEFAULT_REPS)
    args = ap.parse_args()

    all_snaps = (
        anatomy_snapshots()          # 5 poses x 50+ parts
        + biology_snapshots()        # taxonomy, organs, physics
        + lifecycle_snapshots()      # calf to elder
        + environment_snapshots()    # field, barn, water
        + behaviour_snapshots()      # 19 behaviours
        + coat_snapshots()           # colour / texture patterns
        + movement_sequence_snapshots()  # walking gait cycles
    )

    total = len(all_snaps)
    print(f'Training corpus: {total} snapshots x {args.reps} reps'
          f' = {total * args.reps} POSTs  ->  http://{args.node}/neuro/train')

    ok = fail = 0
    for rep in range(args.reps):
        for i, snap in enumerate(all_snaps):
            if post_snapshot(args.node, snap):
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 50 == 0:
                print(f'  rep {rep+1}/{args.reps}  snap {i+1}/{total}'
                      f'  ok={ok}  fail={fail}')

    print(f'\nDone.  ok={ok}  fail={fail}')
    print('Neural fabric now carries cow anatomy, biology, behaviour, and environment.')


if __name__ == '__main__':
    main()
