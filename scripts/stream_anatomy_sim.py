#!/usr/bin/env python3
"""
Human Anatomy CT Stream Simulator
===================================
Simulates a virtual CT scanner doing continuous fly-throughs of human anatomy
and streams EnvironmentSnapshot frames into the neural fabric at 10 fps.

120+ anatomical structures span skull→femoral heads in accurate 3-D body
coordinates:
  x  0 = patient left  →  1 = patient right
  y  0 = feet          →  1 = top of head
  z  0 = anterior      →  1 = posterior

Four scan phases cycle automatically:
  full_body  (120 s) — axial sweep head→feet→head, ±6% y-slab
  cardiac    ( 60 s) — heart + great vessels + lung bases zoomed
  brain      ( 60 s) — full cranial contents
  abdomen    ( 60 s) — liver, kidneys, gut, retroperitoneum

Physiological motion on every frame:
  Cardiac  72 bpm   — chambers dilate/contract, aorta pulses
  Resp     15 /min  — lungs translate, diaphragm descends, subdiaphragmatic
                       organs follow

Usage:
    python scripts/stream_anatomy_sim.py --host 192.168.1.84 --port 8090
    python scripts/stream_anatomy_sim.py --phase cardiac
    python scripts/stream_anatomy_sim.py --fps 10 --no-random-noise
"""

import argparse, json, math, random, sys, time
import urllib.request, urllib.error

# ── Timing constants ──────────────────────────────────────────────────────────
FPS           = 10
DT            = 1.0 / FPS
CARDIAC_BPM   = 72
CARDIAC_T     = 60.0 / CARDIAC_BPM   # 0.833 s per beat
RESP_RATE     = 15                    # breaths per minute
RESP_T        = 60.0 / RESP_RATE     # 4.0 s per breath
BOUNDS        = {'x': 1.0, 'y': 1.0, 'z': 1.0}

# ── Phase schedule ────────────────────────────────────────────────────────────
PHASE_SCHEDULE = [
    ('full_body',  120),
    ('cardiac',     60),
    ('brain',       60),
    ('abdomen',     60),
]
PHASE_TOTAL = sum(d for _, d in PHASE_SCHEDULE)

# ── Depth class mapping ───────────────────────────────────────────────────────
def _depth(system):
    return {
        'bone':    'd0_skeletal',
        'surface': 'd0_surface',
        'vessel':  'd1_vascular',
        'neural':  'd1_neural',
        'organ':   'd2_organ',
        'hollow':  'd2_cavity',
        'gland':   'd2_organ',
    }.get(system, 'd2_organ')

# ── Anatomical structure database ─────────────────────────────────────────────
# (id, label, x, y, z, slice_r, scale_m, system)
# slice_r  — half-width used for y-slab inclusion in full_body sweep
# scale_m  — physical diameter in metres
STRUCTURES = [
    # ── SKULL & SCALP ──────────────────────────────────────────────────────
    ('skull',            'skull',                  0.500, 0.910, 0.500, 0.115, 0.200, 'bone'),
    ('scalp',            'scalp',                  0.500, 0.910, 0.500, 0.115, 0.210, 'surface'),
    ('dura',             'dura_mater',             0.500, 0.905, 0.500, 0.105, 0.185, 'neural'),

    # ── BRAIN ──────────────────────────────────────────────────────────────
    ('cerebrum',         'cerebral_cortex',        0.500, 0.905, 0.500, 0.085, 0.140, 'neural'),
    ('frontal_L',        'frontal_lobe',           0.415, 0.918, 0.360, 0.050, 0.070, 'neural'),
    ('frontal_R',        'frontal_lobe',           0.585, 0.918, 0.360, 0.050, 0.070, 'neural'),
    ('parietal_L',       'parietal_lobe',          0.400, 0.918, 0.555, 0.048, 0.065, 'neural'),
    ('parietal_R',       'parietal_lobe',          0.600, 0.918, 0.555, 0.048, 0.065, 'neural'),
    ('temporal_L',       'temporal_lobe',          0.355, 0.888, 0.520, 0.040, 0.060, 'neural'),
    ('temporal_R',       'temporal_lobe',          0.645, 0.888, 0.520, 0.040, 0.060, 'neural'),
    ('occipital',        'occipital_lobe',         0.500, 0.898, 0.720, 0.042, 0.060, 'neural'),
    ('insula_L',         'insular_cortex',         0.372, 0.896, 0.500, 0.022, 0.030, 'neural'),
    ('insula_R',         'insular_cortex',         0.628, 0.896, 0.500, 0.022, 0.030, 'neural'),
    ('cerebellum',       'cerebellum',             0.500, 0.875, 0.720, 0.048, 0.085, 'neural'),
    ('cerebellar_L',     'cerebellar_hemisphere',  0.440, 0.873, 0.725, 0.032, 0.050, 'neural'),
    ('cerebellar_R',     'cerebellar_hemisphere',  0.560, 0.873, 0.725, 0.032, 0.050, 'neural'),
    ('midbrain',         'midbrain',               0.500, 0.866, 0.600, 0.018, 0.025, 'neural'),
    ('pons',             'pons',                   0.500, 0.860, 0.612, 0.018, 0.025, 'neural'),
    ('medulla',          'medulla_oblongata',      0.500, 0.852, 0.620, 0.016, 0.020, 'neural'),
    ('thalamus_L',       'thalamus',               0.455, 0.892, 0.520, 0.017, 0.025, 'neural'),
    ('thalamus_R',       'thalamus',               0.545, 0.892, 0.520, 0.017, 0.025, 'neural'),
    ('hypothalamus',     'hypothalamus',           0.500, 0.886, 0.500, 0.010, 0.014, 'neural'),
    ('hippocampus_L',    'hippocampus',            0.392, 0.884, 0.562, 0.012, 0.018, 'neural'),
    ('hippocampus_R',    'hippocampus',            0.608, 0.884, 0.562, 0.012, 0.018, 'neural'),
    ('amygdala_L',       'amygdala',               0.380, 0.882, 0.498, 0.008, 0.012, 'neural'),
    ('amygdala_R',       'amygdala',               0.620, 0.882, 0.498, 0.008, 0.012, 'neural'),
    ('corpus_callosum',  'corpus_callosum',        0.500, 0.897, 0.522, 0.015, 0.080, 'neural'),
    ('lat_vent_L',       'lateral_ventricle',      0.442, 0.894, 0.520, 0.018, 0.028, 'hollow'),
    ('lat_vent_R',       'lateral_ventricle',      0.558, 0.894, 0.520, 0.018, 0.028, 'hollow'),
    ('third_vent',       'third_ventricle',        0.500, 0.890, 0.518, 0.006, 0.008, 'hollow'),
    ('fourth_vent',      'fourth_ventricle',       0.500, 0.867, 0.630, 0.008, 0.012, 'hollow'),
    ('basal_gang_L',     'basal_ganglia',          0.438, 0.892, 0.500, 0.015, 0.022, 'neural'),
    ('basal_gang_R',     'basal_ganglia',          0.562, 0.892, 0.500, 0.015, 0.022, 'neural'),
    ('substantia_n_L',   'substantia_nigra',       0.458, 0.868, 0.595, 0.008, 0.012, 'neural'),
    ('substantia_n_R',   'substantia_nigra',       0.542, 0.868, 0.595, 0.008, 0.012, 'neural'),
    ('pituitary',        'pituitary_gland',        0.500, 0.878, 0.500, 0.006, 0.008, 'gland'),
    ('pineal',           'pineal_gland',           0.500, 0.888, 0.570, 0.004, 0.006, 'gland'),
    ('int_carotid_L',    'internal_carotid_artery',0.438, 0.878, 0.440, 0.004, 0.006, 'vessel'),
    ('int_carotid_R',    'internal_carotid_artery',0.562, 0.878, 0.440, 0.004, 0.006, 'vessel'),
    ('basilar_a',        'basilar_artery',         0.500, 0.873, 0.572, 0.003, 0.004, 'vessel'),
    ('mca_L',            'middle_cerebral_artery', 0.375, 0.892, 0.480, 0.003, 0.004, 'vessel'),
    ('mca_R',            'middle_cerebral_artery', 0.625, 0.892, 0.480, 0.003, 0.004, 'vessel'),
    ('aca_L',            'anterior_cerebral_artery',0.468,0.902, 0.462, 0.002, 0.003, 'vessel'),
    ('aca_R',            'anterior_cerebral_artery',0.532,0.902, 0.462, 0.002, 0.003, 'vessel'),
    ('pca_L',            'posterior_cerebral_artery',0.448,0.884,0.570, 0.002, 0.003, 'vessel'),
    ('pca_R',            'posterior_cerebral_artery',0.552,0.884,0.570, 0.002, 0.003, 'vessel'),
    ('eye_L',            'eye',                    0.378, 0.919, 0.302, 0.013, 0.024, 'organ'),
    ('eye_R',            'eye',                    0.622, 0.919, 0.302, 0.013, 0.024, 'organ'),
    ('optic_n_L',        'optic_nerve',            0.418, 0.912, 0.352, 0.004, 0.006, 'neural'),
    ('optic_n_R',        'optic_nerve',            0.582, 0.912, 0.352, 0.004, 0.006, 'neural'),
    ('cochlea_L',        'cochlea',                0.298, 0.886, 0.548, 0.008, 0.008, 'organ'),
    ('cochlea_R',        'cochlea',                0.702, 0.886, 0.548, 0.008, 0.008, 'organ'),
    ('mastoid_L',        'mastoid_air_cells',      0.312, 0.882, 0.612, 0.015, 0.022, 'hollow'),
    ('mastoid_R',        'mastoid_air_cells',      0.688, 0.882, 0.612, 0.015, 0.022, 'hollow'),

    # ── NECK ───────────────────────────────────────────────────────────────
    ('thyroid_L',        'thyroid_gland',          0.448, 0.826, 0.400, 0.018, 0.025, 'gland'),
    ('thyroid_R',        'thyroid_gland',          0.552, 0.826, 0.400, 0.018, 0.025, 'gland'),
    ('parathyroid_LL',   'parathyroid_gland',      0.440, 0.822, 0.412, 0.005, 0.006, 'gland'),
    ('parathyroid_LU',   'parathyroid_gland',      0.440, 0.830, 0.412, 0.005, 0.006, 'gland'),
    ('parathyroid_RL',   'parathyroid_gland',      0.560, 0.822, 0.412, 0.005, 0.006, 'gland'),
    ('parathyroid_RU',   'parathyroid_gland',      0.560, 0.830, 0.412, 0.005, 0.006, 'gland'),
    ('trachea_neck',     'trachea',                0.500, 0.825, 0.430, 0.010, 0.018, 'hollow'),
    ('esoph_neck',       'esophagus',              0.500, 0.825, 0.470, 0.008, 0.014, 'hollow'),
    ('carotid_L',        'common_carotid_artery',  0.432, 0.826, 0.442, 0.004, 0.007, 'vessel'),
    ('carotid_R',        'common_carotid_artery',  0.568, 0.826, 0.442, 0.004, 0.007, 'vessel'),
    ('jugular_L',        'internal_jugular_vein',  0.392, 0.826, 0.460, 0.005, 0.009, 'vessel'),
    ('jugular_R',        'internal_jugular_vein',  0.608, 0.826, 0.460, 0.005, 0.009, 'vessel'),
    ('vertebral_a_L',    'vertebral_artery',       0.458, 0.826, 0.618, 0.003, 0.005, 'vessel'),
    ('vertebral_a_R',    'vertebral_artery',       0.542, 0.826, 0.618, 0.003, 0.005, 'vessel'),
    ('cervical_spine',   'cervical_vertebra',      0.500, 0.826, 0.652, 0.016, 0.025, 'bone'),
    ('cervical_cord',    'spinal_cord',            0.500, 0.826, 0.622, 0.007, 0.012, 'neural'),
    ('scm_L',            'sternocleidomastoid',    0.398, 0.822, 0.440, 0.022, 0.035, 'organ'),
    ('scm_R',            'sternocleidomastoid',    0.602, 0.822, 0.440, 0.022, 0.035, 'organ'),

    # ── THORAX — HEART & GREAT VESSELS ────────────────────────────────────
    ('pericardium',      'pericardium',            0.460, 0.685, 0.522, 0.065, 0.135, 'organ'),
    ('heart',            'heart',                  0.455, 0.680, 0.530, 0.060, 0.120, 'organ'),
    ('lv',               'left_ventricle',         0.430, 0.672, 0.550, 0.040, 0.080, 'organ'),
    ('rv',               'right_ventricle',        0.500, 0.680, 0.510, 0.035, 0.060, 'organ'),
    ('la',               'left_atrium',            0.432, 0.710, 0.600, 0.030, 0.055, 'organ'),
    ('ra',               'right_atrium',           0.522, 0.712, 0.568, 0.030, 0.055, 'organ'),
    ('mvl',              'mitral_valve',           0.438, 0.692, 0.572, 0.012, 0.015, 'organ'),
    ('avl',              'aortic_valve',           0.460, 0.700, 0.540, 0.010, 0.013, 'organ'),
    ('tvl',              'tricuspid_valve',        0.510, 0.694, 0.545, 0.012, 0.015, 'organ'),
    ('pvl',              'pulmonary_valve',        0.490, 0.702, 0.510, 0.010, 0.013, 'organ'),
    ('interventricular_septum','interventricular_septum',0.465,0.680,0.530,0.025,0.010,'organ'),
    ('myocardium',       'myocardium',             0.455, 0.680, 0.530, 0.055, 0.010, 'organ'),
    ('lad',              'left_anterior_descending_artery',0.440,0.674,0.498,0.003,0.004,'vessel'),
    ('lcx',              'left_circumflex_artery', 0.428, 0.678, 0.565, 0.003, 0.004, 'vessel'),
    ('rca',              'right_coronary_artery',  0.510, 0.675, 0.562, 0.003, 0.004, 'vessel'),
    ('ascending_aorta',  'ascending_aorta',        0.478, 0.720, 0.510, 0.018, 0.030, 'vessel'),
    ('aortic_arch',      'aortic_arch',            0.500, 0.740, 0.530, 0.015, 0.025, 'vessel'),
    ('desc_aorta',       'descending_aorta',       0.500, 0.672, 0.638, 0.013, 0.022, 'vessel'),
    ('svc',              'superior_vena_cava',     0.540, 0.726, 0.510, 0.011, 0.018, 'vessel'),
    ('ivc_thorax',       'inferior_vena_cava',     0.520, 0.660, 0.600, 0.011, 0.018, 'vessel'),
    ('pulm_a',           'pulmonary_artery',       0.488, 0.718, 0.500, 0.014, 0.022, 'vessel'),
    ('pulm_a_L',         'left_pulmonary_artery',  0.448, 0.716, 0.550, 0.009, 0.014, 'vessel'),
    ('pulm_a_R',         'right_pulmonary_artery', 0.552, 0.716, 0.548, 0.009, 0.014, 'vessel'),
    ('pulm_v_LL',        'pulmonary_vein',         0.418, 0.712, 0.614, 0.008, 0.012, 'vessel'),
    ('pulm_v_LU',        'pulmonary_vein',         0.422, 0.718, 0.608, 0.008, 0.012, 'vessel'),
    ('pulm_v_RL',        'pulmonary_vein',         0.568, 0.712, 0.610, 0.008, 0.012, 'vessel'),
    ('pulm_v_RU',        'pulmonary_vein',         0.572, 0.718, 0.605, 0.008, 0.012, 'vessel'),
    ('brachioceph_a',    'brachiocephalic_artery', 0.528, 0.748, 0.512, 0.010, 0.016, 'vessel'),
    ('subclavian_L',     'left_subclavian_artery', 0.452, 0.748, 0.528, 0.008, 0.012, 'vessel'),
    ('common_carotid_aorta','left_common_carotid_artery',0.472,0.748,0.518,0.007,0.010,'vessel'),

    # ── THORAX — LUNGS ────────────────────────────────────────────────────
    ('lung_R',           'lung',                   0.672, 0.690, 0.518, 0.090, 0.140, 'organ'),
    ('lung_L',           'lung',                   0.328, 0.690, 0.518, 0.080, 0.120, 'organ'),
    ('rul',              'right_upper_lobe',       0.668, 0.728, 0.502, 0.050, 0.070, 'organ'),
    ('rml',              'right_middle_lobe',      0.680, 0.700, 0.510, 0.040, 0.055, 'organ'),
    ('rll',              'right_lower_lobe',       0.668, 0.656, 0.558, 0.050, 0.070, 'organ'),
    ('lul',              'left_upper_lobe',        0.330, 0.726, 0.502, 0.048, 0.068, 'organ'),
    ('lingula',          'lingula',                0.340, 0.702, 0.510, 0.030, 0.045, 'organ'),
    ('lll',              'left_lower_lobe',        0.330, 0.654, 0.552, 0.048, 0.066, 'organ'),
    ('trachea',          'trachea',                0.500, 0.730, 0.500, 0.010, 0.018, 'hollow'),
    ('carina',           'carina',                 0.500, 0.712, 0.528, 0.010, 0.015, 'hollow'),
    ('r_bronchus',       'right_main_bronchus',    0.568, 0.710, 0.532, 0.008, 0.012, 'hollow'),
    ('l_bronchus',       'left_main_bronchus',     0.432, 0.710, 0.532, 0.007, 0.010, 'hollow'),
    ('r_lobar_bronchus', 'lobar_bronchus',         0.592, 0.706, 0.535, 0.005, 0.008, 'hollow'),
    ('l_lobar_bronchus', 'lobar_bronchus',         0.408, 0.706, 0.535, 0.005, 0.008, 'hollow'),
    ('diaphragm',        'diaphragm',              0.500, 0.635, 0.520, 0.015, 0.300, 'organ'),
    ('esoph_thorax',     'esophagus',              0.500, 0.690, 0.590, 0.008, 0.014, 'hollow'),
    ('thoracic_spine',   'thoracic_vertebra',      0.500, 0.690, 0.702, 0.016, 0.022, 'bone'),
    ('thoracic_cord',    'spinal_cord',            0.500, 0.690, 0.672, 0.007, 0.010, 'neural'),
    ('sternum',          'sternum',                0.500, 0.692, 0.410, 0.015, 0.180, 'bone'),
    ('thymus',           'thymus',                 0.500, 0.730, 0.470, 0.025, 0.040, 'gland'),

    # ── ABDOMEN ───────────────────────────────────────────────────────────
    ('liver',            'liver',                  0.650, 0.565, 0.500, 0.090, 0.150, 'organ'),
    ('liver_R',          'liver_right_lobe',       0.680, 0.562, 0.510, 0.065, 0.120, 'organ'),
    ('liver_L',          'liver_left_lobe',        0.540, 0.568, 0.490, 0.038, 0.070, 'organ'),
    ('caudate_lobe',     'caudate_lobe',           0.522, 0.576, 0.558, 0.015, 0.022, 'organ'),
    ('gallbladder',      'gallbladder',            0.638, 0.550, 0.470, 0.018, 0.035, 'hollow'),
    ('bile_duct',        'common_bile_duct',       0.548, 0.558, 0.545, 0.006, 0.008, 'hollow'),
    ('portal_vein',      'portal_vein',            0.520, 0.570, 0.568, 0.007, 0.011, 'vessel'),
    ('hepatic_a',        'hepatic_artery',         0.512, 0.572, 0.560, 0.005, 0.007, 'vessel'),
    ('hepatic_v_R',      'hepatic_vein',           0.618, 0.582, 0.558, 0.006, 0.010, 'vessel'),
    ('hepatic_v_M',      'hepatic_vein',           0.560, 0.584, 0.555, 0.005, 0.008, 'vessel'),
    ('stomach',          'stomach',                0.442, 0.572, 0.480, 0.058, 0.120, 'hollow'),
    ('gastric_fundus',   'gastric_fundus',         0.412, 0.584, 0.478, 0.030, 0.060, 'hollow'),
    ('pylorus',          'pylorus',                0.522, 0.558, 0.490, 0.018, 0.030, 'hollow'),
    ('duodenum',         'duodenum',               0.542, 0.558, 0.530, 0.025, 0.040, 'hollow'),
    ('pancreas',         'pancreas',               0.510, 0.563, 0.558, 0.030, 0.060, 'organ'),
    ('pancreas_head',    'pancreas_head',          0.562, 0.560, 0.548, 0.022, 0.035, 'organ'),
    ('pancreas_body',    'pancreas_body',          0.498, 0.563, 0.562, 0.018, 0.028, 'organ'),
    ('pancreas_tail',    'pancreas_tail',          0.402, 0.567, 0.572, 0.015, 0.025, 'organ'),
    ('spleen',           'spleen',                 0.338, 0.578, 0.572, 0.040, 0.070, 'organ'),
    ('kidney_R',         'kidney',                 0.660, 0.536, 0.630, 0.038, 0.055, 'organ'),
    ('renal_cortex_R',   'renal_cortex',           0.660, 0.538, 0.630, 0.025, 0.040, 'organ'),
    ('renal_pelvis_R',   'renal_pelvis',           0.662, 0.534, 0.632, 0.012, 0.018, 'hollow'),
    ('kidney_L',         'kidney',                 0.340, 0.546, 0.640, 0.038, 0.055, 'organ'),
    ('renal_cortex_L',   'renal_cortex',           0.340, 0.548, 0.640, 0.025, 0.040, 'organ'),
    ('renal_pelvis_L',   'renal_pelvis',           0.338, 0.544, 0.642, 0.012, 0.018, 'hollow'),
    ('ureter_R',         'ureter',                 0.638, 0.478, 0.610, 0.025, 0.006, 'hollow'),
    ('ureter_L',         'ureter',                 0.362, 0.478, 0.618, 0.025, 0.006, 'hollow'),
    ('adrenal_R',        'adrenal_gland',          0.650, 0.570, 0.640, 0.012, 0.020, 'gland'),
    ('adrenal_L',        'adrenal_gland',          0.360, 0.582, 0.650, 0.012, 0.020, 'gland'),
    ('abd_aorta',        'abdominal_aorta',        0.500, 0.525, 0.642, 0.010, 0.018, 'vessel'),
    ('ivc',              'inferior_vena_cava',     0.520, 0.525, 0.620, 0.012, 0.020, 'vessel'),
    ('celiac',           'celiac_axis',            0.500, 0.580, 0.642, 0.005, 0.008, 'vessel'),
    ('sma',              'superior_mesenteric_artery',0.500,0.562,0.642,0.004,0.007,'vessel'),
    ('ima',              'inferior_mesenteric_artery',0.500,0.422,0.640,0.004,0.006,'vessel'),
    ('renal_a_R',        'renal_artery',           0.560, 0.538, 0.638, 0.004, 0.006, 'vessel'),
    ('renal_a_L',        'renal_artery',           0.440, 0.548, 0.640, 0.004, 0.006, 'vessel'),
    ('small_bowel',      'small_intestine',        0.500, 0.508, 0.480, 0.120, 0.025, 'hollow'),
    ('jejunum',          'jejunum',                0.462, 0.528, 0.462, 0.060, 0.025, 'hollow'),
    ('ileum',            'ileum',                  0.538, 0.488, 0.498, 0.060, 0.022, 'hollow'),
    ('ascend_colon',     'ascending_colon',        0.678, 0.502, 0.500, 0.025, 0.050, 'hollow'),
    ('transv_colon',     'transverse_colon',       0.500, 0.548, 0.468, 0.025, 0.050, 'hollow'),
    ('descend_colon',    'descending_colon',       0.322, 0.502, 0.500, 0.025, 0.050, 'hollow'),
    ('lumbar_spine',     'lumbar_vertebra',        0.500, 0.502, 0.702, 0.016, 0.022, 'bone'),
    ('lumbar_cord',      'cauda_equina',           0.500, 0.502, 0.672, 0.007, 0.008, 'neural'),
    ('psoas_R',          'psoas_muscle',           0.572, 0.502, 0.648, 0.030, 0.045, 'organ'),
    ('psoas_L',          'psoas_muscle',           0.428, 0.502, 0.648, 0.030, 0.045, 'organ'),

    # ── PELVIS ────────────────────────────────────────────────────────────
    ('bladder',          'urinary_bladder',        0.500, 0.305, 0.440, 0.040, 0.080, 'hollow'),
    ('rectum',           'rectum',                 0.500, 0.285, 0.620, 0.025, 0.040, 'hollow'),
    ('sigmoid_colon',    'sigmoid_colon',          0.440, 0.322, 0.540, 0.022, 0.040, 'hollow'),
    ('uterus',           'uterus',                 0.500, 0.295, 0.520, 0.035, 0.070, 'organ'),
    ('ovary_L',          'ovary',                  0.430, 0.302, 0.518, 0.015, 0.025, 'organ'),
    ('ovary_R',          'ovary',                  0.570, 0.302, 0.518, 0.015, 0.025, 'organ'),
    ('iliac_a_R',        'external_iliac_artery',  0.582, 0.338, 0.580, 0.008, 0.012, 'vessel'),
    ('iliac_a_L',        'external_iliac_artery',  0.418, 0.338, 0.580, 0.008, 0.012, 'vessel'),
    ('iliac_v_R',        'external_iliac_vein',    0.572, 0.338, 0.590, 0.009, 0.014, 'vessel'),
    ('iliac_v_L',        'external_iliac_vein',    0.428, 0.338, 0.590, 0.009, 0.014, 'vessel'),
    ('sacrum',           'sacrum',                 0.500, 0.282, 0.700, 0.040, 0.070, 'bone'),
    ('ilium_R',          'ilium',                  0.660, 0.305, 0.572, 0.060, 0.110, 'bone'),
    ('ilium_L',          'ilium',                  0.340, 0.305, 0.572, 0.060, 0.110, 'bone'),
    ('femoral_head_R',   'femoral_head',           0.658, 0.245, 0.528, 0.025, 0.045, 'bone'),
    ('femoral_head_L',   'femoral_head',           0.342, 0.245, 0.528, 0.025, 0.045, 'bone'),
    ('acetabulum_R',     'acetabulum',             0.648, 0.255, 0.540, 0.025, 0.050, 'bone'),
    ('acetabulum_L',     'acetabulum',             0.352, 0.255, 0.540, 0.025, 0.050, 'bone'),
]

# ── Phase focus sets ──────────────────────────────────────────────────────────
CARDIAC_FOCUS = frozenset({
    'pericardium','heart','lv','rv','la','ra','mvl','avl','tvl','pvl',
    'interventricular_septum','myocardium',
    'lad','lcx','rca','ascending_aorta','aortic_arch','desc_aorta',
    'svc','ivc_thorax','pulm_a','pulm_a_L','pulm_a_R',
    'pulm_v_LL','pulm_v_LU','pulm_v_RL','pulm_v_RU',
    'lung_R','lung_L','diaphragm','trachea','esoph_thorax',
})

BRAIN_FOCUS = frozenset({
    'skull','dura','cerebrum','frontal_L','frontal_R',
    'parietal_L','parietal_R','temporal_L','temporal_R','occipital',
    'insula_L','insula_R','cerebellum','cerebellar_L','cerebellar_R',
    'midbrain','pons','medulla','thalamus_L','thalamus_R','hypothalamus',
    'hippocampus_L','hippocampus_R','amygdala_L','amygdala_R',
    'corpus_callosum','lat_vent_L','lat_vent_R','third_vent','fourth_vent',
    'basal_gang_L','basal_gang_R','substantia_n_L','substantia_n_R',
    'pituitary','pineal','int_carotid_L','int_carotid_R','basilar_a',
    'mca_L','mca_R','aca_L','aca_R','pca_L','pca_R',
    'eye_L','eye_R','optic_n_L','optic_n_R','cochlea_L','cochlea_R',
})

ABDOMEN_FOCUS = frozenset({
    'liver','liver_R','liver_L','caudate_lobe','gallbladder','bile_duct',
    'portal_vein','hepatic_a','hepatic_v_R','hepatic_v_M',
    'stomach','gastric_fundus','pylorus','duodenum',
    'pancreas','pancreas_head','pancreas_body','pancreas_tail',
    'spleen','kidney_R','renal_cortex_R','renal_pelvis_R',
    'kidney_L','renal_cortex_L','renal_pelvis_L',
    'adrenal_R','adrenal_L','abd_aorta','ivc','celiac','sma',
    'renal_a_R','renal_a_L','small_bowel','jejunum','ileum',
    'ascend_colon','transv_colon','descend_colon',
    'lumbar_spine','psoas_R','psoas_L',
})

# ── Physiological motion ──────────────────────────────────────────────────────
# Structures that move with the cardiac cycle
_CARDIAC_MOBILE = {
    'lv':              ( 0.008, -0.010,  0.005),   # (dx_systole, dy_systole, dz_systole)
    'rv':              (-0.006, -0.008,  0.004),
    'la':              ( 0.003,  0.005, -0.004),
    'ra':              (-0.003,  0.005, -0.003),
    'ascending_aorta': ( 0.002,  0.003,  0.000),
    'pulm_a':          ( 0.002,  0.002,  0.000),
    'abd_aorta':       ( 0.002,  0.000,  0.001),
    'ivc':             ( 0.001,  0.000,  0.001),
}
# Structures that move with the respiratory cycle
_RESP_MOBILE = {
    'lung_R':   (-0.005, -0.018, -0.002),  # (dx_insp, dy_insp, dz_insp) — descend & expand
    'lung_L':   ( 0.005, -0.018, -0.002),
    'rul':      (-0.004, -0.012, -0.002),
    'rml':      (-0.004, -0.015, -0.002),
    'rll':      (-0.004, -0.020, -0.002),
    'lul':      ( 0.004, -0.012, -0.002),
    'lll':      ( 0.004, -0.020, -0.002),
    'lingula':  ( 0.004, -0.015, -0.002),
    'diaphragm':( 0.000, -0.022,  0.000),
    'liver':    ( 0.000, -0.016,  0.000),
    'liver_R':  ( 0.000, -0.016,  0.000),
    'liver_L':  ( 0.000, -0.014,  0.000),
    'spleen':   ( 0.000, -0.014,  0.000),
    'kidney_R': ( 0.000, -0.012,  0.000),
    'kidney_L': ( 0.000, -0.012,  0.000),
    'adrenal_R':( 0.000, -0.010,  0.000),
    'adrenal_L':( 0.000, -0.010,  0.000),
    'gallbladder':( 0.000,-0.014,  0.000),
}

def _cardiac_phase(t):
    """Return systole fraction [0,1]. Sharp rise, slow fall (mimics ECG)."""
    phi = (t % CARDIAC_T) / CARDIAC_T
    if phi < 0.38:                         # systole
        s = math.sin(phi / 0.38 * math.pi)
        return s
    return 0.0                             # diastole

def _resp_phase(t):
    """Inspiration fraction [0,1]. Sinusoidal, slight I:E asymmetry."""
    phi = (t % RESP_T) / RESP_T
    return max(0.0, math.sin(phi * math.pi))

def get_position(struct_id, rx, ry, rz, t, noise):
    """Rest position + cardiac + respiratory + Brownian noise."""
    sys  = _cardiac_phase(t)
    insp = _resp_phase(t)
    dx = dy = dz = 0.0
    if struct_id in _CARDIAC_MOBILE:
        cx, cy, cz = _CARDIAC_MOBILE[struct_id]
        dx += cx * sys; dy += cy * sys; dz += cz * sys
    if struct_id in _RESP_MOBILE:
        rx2, ry2, rz2 = _RESP_MOBILE[struct_id]
        dx += rx2 * insp; dy += ry2 * insp; dz += rz2 * insp
    if noise:
        dx += random.gauss(0, 0.0006)
        dy += random.gauss(0, 0.0006)
        dz += random.gauss(0, 0.0004)
    return (
        max(0.01, min(0.99, rx + dx)),
        max(0.01, min(0.99, ry + dy)),
        max(0.01, min(0.99, rz + dz)),
    )

def get_velocity(struct_id, t):
    """Instantaneous velocity from derivative of physiological motion."""
    eps = 0.01
    x0, y0, z0 = get_position(struct_id, 0, 0, 0, t,       False)
    x1, y1, z1 = get_position(struct_id, 0, 0, 0, t + eps, False)
    return (
        round((x1 - x0) / eps, 5),
        round((y1 - y0) / eps, 5),
        round((z1 - z0) / eps, 5),
    )

# ── Phase determination ───────────────────────────────────────────────────────
def current_phase(t):
    cycle = t % PHASE_TOTAL
    accum = 0.0
    for name, dur in PHASE_SCHEDULE:
        if cycle < accum + dur:
            return name, cycle - accum
        accum += dur
    return PHASE_SCHEDULE[-1]

def scan_y_position(phase, phase_t):
    """Return current y-slice centre for full_body sweep."""
    period = 60.0
    sweep  = phase_t % period
    half   = period / 2.0
    if sweep < half:
        return 0.96 - (0.90 / half) * sweep
    else:
        return 0.06 + (0.90 / half) * (sweep - half)

def visible_structures(phase, phase_t, t, noise):
    """Yield (id, label, x, y, z, size_m, system) for current scan state."""
    results = []
    scan_y  = scan_y_position(phase, phase_t)
    slab    = 0.07   # ±7% body-height slice for full_body

    for row in STRUCTURES:
        sid, label, rx, ry, rz, sr, size_m, system = row
        x, y, z = get_position(sid, rx, ry, rz, t, noise)

        if phase == 'full_body':
            if abs(y - scan_y) > slab + sr:
                continue
        elif phase == 'cardiac':
            if sid not in CARDIAC_FOCUS:
                continue
        elif phase == 'brain':
            if sid not in BRAIN_FOCUS:
                continue
        elif phase == 'abdomen':
            if sid not in ABDOMEN_FOCUS:
                continue

        results.append((sid, label, x, y, z, size_m, system))
    return results

# ── Snapshot builder ──────────────────────────────────────────────────────────
def build_snapshot(structs, prev_pos, t, phase, scan_y):
    symbols = []
    for sid, label, x, y, z, size_m, system in structs:
        vx, vy, vz = get_velocity(sid, t)
        symbols.append({
            'id':   sid,
            'type': 'CUSTOM',
            'position': {'x': round(x, 5), 'y': round(y, 5), 'z': round(z, 5)},
            'velocity': {'x': vx, 'y': vy, 'z': vz},
            'properties': {
                'label':       label,
                'scale_m':     str(size_m),
                'diameter_m':  str(size_m),
                'depth_class': _depth(system),
                'track_id':    sid,
                'system':      system,
            }
        })
    return {
        'timestamp': int(t * 1000),
        'bounds':    {'x': 1.0, 'y': 1.0, 'z': 1.0},
        'symbols':   symbols,
        'metadata':  {
            'context':       'ct_body_scan',
            'modality':      'ct_simulated',
            'frame_t':       str(round(t, 3)),
            'scan_phase':    phase,
            'scan_y':        str(round(scan_y, 3)),
            'symbol_count':  str(len(symbols)),
            'cardiac_bpm':   str(CARDIAC_BPM),
            'resp_rate':     str(RESP_RATE),
        }
    }

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def post(host, port, path, body):
    url  = f'http://{host}:{port}{path}'
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={'Content-Type': 'application/json'})
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
    except Exception:
        return False

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='Stream virtual CT anatomy scan into the neural fabric.')
    ap.add_argument('--host',          default='192.168.1.84')
    ap.add_argument('--port',          type=int, default=8090)
    ap.add_argument('--fps',           type=int, default=FPS)
    ap.add_argument('--phase',         default=None,
                    choices=['full_body','cardiac','brain','abdomen'],
                    help='Lock to a single scan phase instead of cycling')
    ap.add_argument('--no-noise',      action='store_true',
                    help='Disable Brownian positional noise')
    args = ap.parse_args()

    print(f'Connecting to {args.host}:{args.port} ...', flush=True)
    if not check_connection(args.host, args.port):
        print('Node not reachable. Start the node first.', flush=True)
        sys.exit(1)
    print('Connected. Starting anatomy CT stream.', flush=True)
    print(f'  {len(STRUCTURES)} structures  |  {args.fps} fps', flush=True)
    print(f'  Phases: {" → ".join(n for n,_ in PHASE_SCHEDULE)} (cycling every {PHASE_TOTAL}s)', flush=True)
    if args.phase:
        print(f'  Locked to phase: {args.phase}', flush=True)
    print('Press Ctrl+C to stop.', flush=True)

    interval = 1.0 / args.fps
    frame    = 0
    t        = 0.0
    drift    = 0.0
    noise    = not args.no_noise

    while True:
        t0 = time.perf_counter()

        if args.phase:
            phase, phase_t = args.phase, t % PHASE_TOTAL
        else:
            phase, phase_t = current_phase(t)

        scan_y  = scan_y_position(phase, phase_t)
        structs = visible_structures(phase, phase_t, t, noise)
        snap    = build_snapshot(structs, {}, t, phase, scan_y)
        result  = post(args.host, args.port, '/neuro/train', {'snapshot': snap})

        if frame % (args.fps * 5) == 0:
            lc = result.get('label_count', '?')
            print(f'  frame {frame:06d}  t={t:.1f}s  phase={phase:<10}  '
                  f'scan_y={scan_y:.3f}  structs={len(structs):3d}  labels={lc}', flush=True)

        frame += 1
        t     += interval

        elapsed = time.perf_counter() - t0
        wait    = interval - elapsed - drift
        if wait > 0:
            time.sleep(wait)
            drift = 0.0
        else:
            drift = -wait * 0.1

if __name__ == '__main__':
    main()
