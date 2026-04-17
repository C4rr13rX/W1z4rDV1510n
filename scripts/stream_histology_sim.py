#!/usr/bin/env python3
"""
Histology Microscopy Stream Simulator
======================================
Simulates a motorised virtual microscope scanning across stained tissue slides,
feeding EnvironmentSnapshot symbols into the neural fabric at 10 fps.

Six tissue types cycle automatically (90 s each):
  1. cardiac_muscle      — H&E cross/longitudinal section
  2. cerebral_cortex     — Nissl-stained, all 6 cortical layers
  3. kidney_cortex       — PAS-stained, glomeruli + tubules
  4. liver_lobule        — H&E, hepatic plates + portal triads
  5. lung_alveoli        — H&E, type I/II pneumocytes + capillaries
  6. cortical_bone       — Ground section, Haversian systems

All positions are normalised to the current microscope field of view (0–1).
scale_m encodes actual physical diameter in metres so the EEM can calibrate
the z-axis to cellular/subcellular resolution.

Virtual stage motion:
  - Slow raster scan at ~18–25 µm/s (≈ motorised stage at 40×)
  - Slight z-focus oscillation mimics depth-of-field stepping
  - Red blood cells in capillaries drift with simulated blood velocity

Usage:
    python scripts/stream_histology_sim.py --host 192.168.1.84 --port 8090
    python scripts/stream_histology_sim.py --tissue kidney_cortex
    python scripts/stream_histology_sim.py --fps 10 --mag 100
"""

import argparse, json, math, random, sys, time
import urllib.request, urllib.error

# ── Simulation parameters ─────────────────────────────────────────────────────
FPS            = 10
DT             = 1.0 / FPS
TISSUE_DURATION = 90          # seconds per tissue before cycling
BOUNDS         = {'x': 1.0, 'y': 1.0, 'z': 0.30}   # z = depth through section

# Objective field-of-view widths (µm)
FOV_UM = {10: 1800, 20: 900, 40: 450, 100: 180}

# ── Tissue type definitions ───────────────────────────────────────────────────
# Each tissue spec:
#   'fov_um'       : field-of-view width in µm at chosen magnification
#   'slide_um'     : virtual slide width in µm (stage scans across this)
#   'cells'        : list of cell-type dicts — see below
#   'structures'   : list of large structural features (constant in every frame)
#   'motile'       : list of motile element specs (e.g. RBCs in capillaries)

TISSUES = {

    # ── 1. Cardiac muscle (H&E, 40× equivalent) ───────────────────────────
    'cardiac_muscle': {
        'fov_um':     450,
        'slide_um':   2800,
        'context':    'cardiac_histology',
        'description':'Ventricular myocardium — cardiomyocytes, intercalated discs, mitochondria',
        'cells': [
            # (label, count_per_fov, width_um, height_um, z_mean, depth_class, system)
            ('cardiomyocyte',        12, 18.0, 92.0, 0.12, 'd2_organ',     'muscle'),
            ('intercalated_disc',    24,  1.0, 18.0, 0.14, 'd3_membrane',  'muscle'),
            ('cardiac_nucleus',      12,  6.0, 10.0, 0.12, 'd1_nuclear',   'muscle'),
            ('cardiac_mitochondria', 60,  1.8,  2.8, 0.11, 'd3_molecular', 'organelle'),
            ('myofibril',            80,  0.5, 80.0, 0.12, 'd3_molecular', 'organelle'),
            ('sarcomere_z_line',    180,  0.2,  1.0, 0.12, 'd3_molecular', 'organelle'),
            ('t_tubule',             45,  0.3,  0.3, 0.13, 'd3_molecular', 'organelle'),
            ('sarcoplasmic_reticulum',30, 0.5,  2.0, 0.13, 'd3_molecular', 'organelle'),
            ('lipofuscin_granule',    8,  1.5,  1.5, 0.11, 'd3_molecular', 'organelle'),
        ],
        'structures': [
            # (label, rel_x, rel_y, size_um, depth_class)  — relative to slide segment
            ('capillary',   0.25, 0.50, 12.0, 'd2_organ'),
            ('capillary',   0.72, 0.30, 10.0, 'd2_organ'),
            ('endomysium',  0.50, 0.50,  4.0, 'd2_organ'),
            ('perimysium',  0.50, 0.05, 18.0, 'd0_surface'),
        ],
        'motile': [
            ('erythrocyte', 8, 8.0, 2.5, 0.14, 0.8),  # (label, n, diam_um, h_um, z, blood_speed_norm)
        ],
        'scan_speed_um_s': 22,
    },

    # ── 2. Cerebral cortex (Nissl stain, 40×) ─────────────────────────────
    'cerebral_cortex': {
        'fov_um':     450,
        'slide_um':   3000,
        'context':    'brain_histology',
        'description':'Cerebral cortex layers I–VI — pyramidal neurons, glia, neuropil',
        'cells': [
            ('pyramidal_neuron_III',  4, 22.0, 30.0, 0.10, 'd1_neural',    'neuron'),
            ('pyramidal_neuron_V',    3, 35.0, 55.0, 0.10, 'd1_neural',    'neuron'),
            ('stellate_neuron',       8, 12.0, 12.0, 0.12, 'd1_neural',    'neuron'),
            ('basket_cell',           3, 15.0, 15.0, 0.11, 'd1_neural',    'neuron'),
            ('chandelier_cell',       2, 12.0, 18.0, 0.11, 'd1_neural',    'neuron'),
            ('neuronal_nucleus',      7,  8.0, 10.0, 0.09, 'd1_nuclear',   'neuron'),
            ('nucleolus',             7,  2.5,  2.5, 0.09, 'd1_nuclear',   'neuron'),
            ('astrocyte',            12, 10.0, 10.0, 0.12, 'd2_glia',      'glia'),
            ('oligodendrocyte',      18,  7.0,  8.0, 0.12, 'd2_glia',      'glia'),
            ('microglia',            10,  6.0,  9.0, 0.13, 'd2_glia',      'glia'),
            ('neuropil',             40,  3.0,  4.0, 0.14, 'd3_molecular', 'neuropil'),
            ('dendrite',             25,  1.5, 35.0, 0.11, 'd3_molecular', 'neuron'),
            ('axon_myelinated',      30,  2.0, 40.0, 0.13, 'd3_molecular', 'neuron'),
            ('synaptic_bouton',      60,  1.0,  1.0, 0.15, 'd3_molecular', 'synapse'),
            ('dendritic_spine',      80,  0.8,  1.5, 0.15, 'd3_molecular', 'synapse'),
        ],
        'structures': [
            ('cortex_layer_I',   0.50, 0.04, 400.0, 'd0_surface'),
            ('cortex_layer_II',  0.50, 0.12, 400.0, 'd2_organ'),
            ('cortex_layer_III', 0.50, 0.25, 400.0, 'd2_organ'),
            ('cortex_layer_IV',  0.50, 0.45, 400.0, 'd2_organ'),
            ('cortex_layer_V',   0.50, 0.65, 400.0, 'd2_organ'),
            ('cortex_layer_VI',  0.50, 0.85, 400.0, 'd2_organ'),
            ('capillary',        0.35, 0.40, 10.0,  'd2_organ'),
            ('capillary',        0.78, 0.62, 12.0,  'd2_organ'),
            ('blood_brain_barrier', 0.35, 0.40, 10.0,'d1_vascular'),
        ],
        'motile': [
            ('erythrocyte', 5, 8.0, 2.5, 0.15, 0.5),
        ],
        'scan_speed_um_s': 20,
    },

    # ── 3. Kidney cortex (PAS stain, 40×) ────────────────────────────────
    'kidney_cortex': {
        'fov_um':     450,
        'slide_um':   2500,
        'context':    'renal_histology',
        'description':'Renal cortex — glomeruli, proximal/distal tubules, Bowman capsule',
        'cells': [
            ('podocyte',             24,  8.0,  8.0, 0.10, 'd2_organ',     'kidney'),
            ('podocyte_foot_process', 48, 1.5,  4.0, 0.11, 'd3_molecular', 'kidney'),
            ('mesangial_cell',        8,  7.0,  7.0, 0.12, 'd2_organ',     'kidney'),
            ('glomerular_endothelium',30,  3.0,  5.0, 0.13, 'd1_vascular',  'kidney'),
            ('bowman_parietal_cell',  16,  6.0,  8.0, 0.09, 'd2_organ',     'kidney'),
            ('proximal_tubule_cell',  28, 10.0, 12.0, 0.11, 'd2_organ',     'kidney'),
            ('brush_border',          28,  1.0,  3.0, 0.12, 'd3_molecular', 'kidney'),
            ('distal_tubule_cell',    20,  8.0, 10.0, 0.12, 'd2_organ',     'kidney'),
            ('collecting_duct_cell',  12,  7.0,  9.0, 0.12, 'd2_organ',     'kidney'),
            ('juxtaglomerular_cell',   6,  8.0,  9.0, 0.11, 'd2_organ',     'kidney'),
            ('macula_densa_cell',      6,  6.0,  8.0, 0.11, 'd2_organ',     'kidney'),
            ('renal_tubule_nucleus',  28,  5.0,  7.0, 0.10, 'd1_nuclear',   'kidney'),
            ('tubular_mitochondria',  80,  1.2,  2.0, 0.11, 'd3_molecular', 'organelle'),
        ],
        'structures': [
            ('glomerulus',           0.50, 0.50, 200.0, 'd2_organ'),
            ('bowman_capsule',       0.50, 0.50, 215.0, 'd2_organ'),
            ('glomerular_capillary_tuft', 0.50, 0.50, 180.0, 'd1_vascular'),
            ('afferent_arteriole',   0.32, 0.42,  20.0, 'd1_vascular'),
            ('efferent_arteriole',   0.68, 0.58,  15.0, 'd1_vascular'),
            ('peritubular_capillary',0.72, 0.28,  12.0, 'd1_vascular'),
        ],
        'motile': [
            ('erythrocyte', 12, 8.0, 2.5, 0.14, 1.2),
        ],
        'scan_speed_um_s': 18,
    },

    # ── 4. Liver lobule (H&E, 20×) ────────────────────────────────────────
    'liver_lobule': {
        'fov_um':     900,
        'slide_um':   4000,
        'context':    'hepatic_histology',
        'description':'Liver lobule — hepatic plates, sinusoids, portal triads, central vein',
        'cells': [
            ('hepatocyte',           80, 22.0, 25.0, 0.11, 'd2_organ',     'liver'),
            ('binucleate_hepatocyte', 12, 24.0, 26.0, 0.11, 'd2_organ',    'liver'),
            ('hepatocyte_nucleus',   80,  8.0,  9.0, 0.10, 'd1_nuclear',   'liver'),
            ('kupffer_cell',         18, 12.0, 14.0, 0.13, 'd2_organ',     'liver'),
            ('stellate_cell',        10,  8.0, 10.0, 0.14, 'd2_organ',     'liver'),
            ('sinusoidal_endothelium',40, 2.0,  5.0, 0.14, 'd1_vascular',  'liver'),
            ('space_of_disse',       40,  1.0,  3.0, 0.15, 'd3_molecular', 'liver'),
            ('bile_canaliculus',     30,  1.5,  2.0, 0.15, 'd3_molecular', 'liver'),
            ('hepatocyte_mitochondria',150, 1.0, 1.8, 0.11,'d3_molecular', 'organelle'),
            ('smooth_er',            60,  0.5,  1.5, 0.12, 'd3_molecular', 'organelle'),
            ('rough_er',             60,  0.5,  2.0, 0.12, 'd3_molecular', 'organelle'),
            ('glycogen_granule',    200,  0.3,  0.3, 0.12, 'd3_molecular', 'organelle'),
        ],
        'structures': [
            ('central_vein',         0.50, 0.50, 80.0,  'd1_vascular'),
            ('portal_vein_branch',   0.18, 0.22, 60.0,  'd1_vascular'),
            ('hepatic_artery_branch',0.18, 0.22, 30.0,  'd1_vascular'),
            ('bile_duct_branch',     0.18, 0.22, 25.0,  'd2_cavity'),
            ('portal_triad',         0.18, 0.22, 75.0,  'd2_organ'),
            ('portal_triad',         0.82, 0.78, 75.0,  'd2_organ'),
            ('sinusoid',             0.40, 0.60, 12.0,  'd1_vascular'),
            ('hepatic_plate',        0.60, 0.40, 22.0,  'd2_organ'),
        ],
        'motile': [
            ('erythrocyte',  20, 8.0, 2.5, 0.14, 0.6),
            ('leukocyte',     4, 12.0,12.0, 0.13, 0.3),
            ('platelet',     15,  2.5, 2.5, 0.15, 0.5),
        ],
        'scan_speed_um_s': 25,
    },

    # ── 5. Lung alveoli (H&E, 40×) ───────────────────────────────────────
    'lung_alveoli': {
        'fov_um':     450,
        'slide_um':   2600,
        'context':    'pulmonary_histology',
        'description':'Lung parenchyma — alveoli, type I/II pneumocytes, capillary wall, surfactant',
        'cells': [
            ('type_I_pneumocyte',    20,  0.5, 80.0, 0.08, 'd2_organ',     'lung'),
            ('type_II_pneumocyte',    8,  8.0,  8.0, 0.11, 'd2_organ',     'lung'),
            ('alveolar_macrophage',   6, 14.0, 14.0, 0.13, 'd2_organ',     'lung'),
            ('capillary_endothelium',40,  1.5,  4.0, 0.13, 'd1_vascular',  'lung'),
            ('lamellar_body',        16,  3.0,  3.0, 0.12, 'd3_molecular', 'organelle'),
            ('surfactant_layer',     20,  2.0, 20.0, 0.07, 'd3_molecular', 'lung'),
            ('alveolar_nucleus',     12,  5.0,  7.0, 0.10, 'd1_nuclear',   'lung'),
            ('pneumocyte_II_nucleus', 8,  4.5,  5.5, 0.10, 'd1_nuclear',   'lung'),
            ('fibroblast',            5, 10.0, 22.0, 0.14, 'd2_organ',     'lung'),
            ('elastic_fibre',        30,  0.5, 12.0, 0.15, 'd3_molecular', 'lung'),
        ],
        'structures': [
            ('alveolus',          0.28, 0.42, 250.0, 'd2_cavity'),
            ('alveolus',          0.72, 0.58, 220.0, 'd2_cavity'),
            ('alveolar_duct',     0.50, 0.20,  80.0, 'd2_cavity'),
            ('alveolar_septum',   0.50, 0.50,   2.0, 'd2_organ'),
            ('alveolar_capillary',0.38, 0.48,  10.0, 'd1_vascular'),
            ('alveolar_capillary',0.65, 0.55,  10.0, 'd1_vascular'),
        ],
        'motile': [
            ('erythrocyte', 8, 8.0, 2.5, 0.14, 0.9),
        ],
        'scan_speed_um_s': 20,
    },

    # ── 6. Cortical bone — ground section (Haversian system) ─────────────
    'cortical_bone': {
        'fov_um':     450,
        'slide_um':   3200,
        'context':    'bone_histology',
        'description':'Compact bone — osteons, Haversian canals, lacunae, canaliculi, lamellae',
        'cells': [
            ('osteocyte',            20,  8.0, 10.0, 0.12, 'd2_organ',     'bone'),
            ('osteocyte_lacuna',     20, 12.0, 15.0, 0.12, 'd2_organ',     'bone'),
            ('canaliculus',         120,  0.4,  8.0, 0.12, 'd3_molecular', 'bone'),
            ('osteoblast',           12,  9.0, 12.0, 0.09, 'd2_organ',     'bone'),
            ('osteoclast',            4, 50.0, 30.0, 0.09, 'd2_organ',     'bone'),
            ('bone_lining_cell',     25,  8.0,  4.0, 0.08, 'd0_surface',   'bone'),
            ('mineralized_matrix',  100,  3.0,  3.0, 0.15, 'd3_molecular', 'bone'),
            ('collagen_fibre',       80,  0.5, 20.0, 0.14, 'd3_molecular', 'bone'),
            ('hydroxyapatite_crystal',200, 0.1, 0.6, 0.15, 'd3_molecular', 'bone'),
        ],
        'structures': [
            ('osteon',              0.32, 0.38, 260.0, 'd2_organ'),
            ('osteon',              0.72, 0.62, 240.0, 'd2_organ'),
            ('haversian_canal',     0.32, 0.38,  52.0, 'd1_vascular'),
            ('haversian_canal',     0.72, 0.62,  48.0, 'd1_vascular'),
            ('volkmann_canal',      0.52, 0.50,  38.0, 'd1_vascular'),
            ('cement_line',         0.32, 0.38, 262.0, 'd2_organ'),
            ('interstitial_lamella',0.62, 0.20, 120.0, 'd2_organ'),
            ('circumferential_lamella',0.50,0.05,400.0,'d0_surface'),
            ('periosteum',          0.50, 0.02, 400.0, 'd0_surface'),
            ('endosteum',           0.50, 0.99, 400.0, 'd0_surface'),
        ],
        'motile': [
            ('erythrocyte', 4, 8.0, 2.5, 0.13, 0.4),
        ],
        'scan_speed_um_s': 18,
    },
}

TISSUE_ORDER = list(TISSUES.keys())

# ── Virtual stage state ───────────────────────────────────────────────────────
class VirtualStage:
    """Motorised XY stage with sinusoidal raster scan and z-focus dithering."""

    def __init__(self, slide_um: float, fov_um: float, scan_speed_um_s: float):
        self.slide_um  = slide_um
        self.fov_um    = fov_um
        self.speed     = scan_speed_um_s
        # Start at random position
        self.x_um      = random.uniform(0, slide_um)
        self.y_um      = random.uniform(0, slide_um)
        self.z_focus   = 0.12    # nominal focal plane (0–0.30)
        self.dir_x     = 1.0
        self.row_step  = fov_um * 0.8

    def step(self, dt: float):
        self.x_um += self.speed * dt * self.dir_x
        # Raster: reverse direction at slide edges, advance row
        if self.x_um > self.slide_um or self.x_um < 0:
            self.dir_x *= -1
            self.x_um   = max(0, min(self.slide_um, self.x_um))
            self.y_um  += self.row_step
            if self.y_um > self.slide_um:
                self.y_um = 0.0
        # Gentle z-focus oscillation (±0.005)
        self.z_focus = 0.12 + 0.005 * math.sin(time.perf_counter() * 0.7)

    @property
    def fov_x0(self): return self.x_um - self.fov_um / 2
    @property
    def fov_x1(self): return self.x_um + self.fov_um / 2
    @property
    def fov_y0(self): return self.y_um - self.fov_um / 2
    @property
    def fov_y1(self): return self.y_um + self.fov_um / 2

    def in_fov(self, abs_x, abs_y, margin=0):
        return (self.fov_x0 - margin <= abs_x <= self.fov_x1 + margin and
                self.fov_y0 - margin <= abs_y <= self.fov_y1 + margin)

    def to_norm(self, abs_x, abs_y):
        """Map absolute slide coords to normalised FOV coords (0–1)."""
        nx = (abs_x - self.fov_x0) / self.fov_um
        ny = (abs_y - self.fov_y0) / self.fov_um
        return round(nx, 5), round(ny, 5)


# ── Tissue population generators ─────────────────────────────────────────────
def generate_tissue_layout(tissue_name: str, slide_um: float):
    """
    Pre-generate a fixed spatial layout for a tissue type — a list of
    (label, abs_x_um, abs_y_um, size_m, depth_class) entries.
    This is called once per tissue and cached for the 90 s duration.
    """
    spec   = TISSUES[tissue_name]
    fov_um = spec['fov_um']
    layout = []

    # Tile cells across the slide
    for cell in spec['cells']:
        lbl, n_fov, w_um, h_um, z_mean, depth, system = cell
        # Density: n_fov cells per FOV area -> scale to full slide
        fov_area   = fov_um ** 2
        slide_area = slide_um ** 2
        count      = int(n_fov * slide_area / fov_area * random.uniform(0.7, 1.3))
        size_m     = (w_um + h_um) / 2 * 1e-6

        for _ in range(count):
            ax = random.uniform(0, slide_um)
            ay = random.uniform(0, slide_um)
            uid = f'{lbl[:8]}_{len(layout)}'
            layout.append({
                'uid': uid, 'label': lbl,
                'ax': ax, 'ay': ay,
                'size_m': size_m, 'depth': depth,
                'z': z_mean + random.gauss(0, 0.01),
                'mobile': False,
                'vx': 0.0, 'vy': 0.0,
            })

    # Fixed structural features at fractional positions
    for feat in spec['structures']:
        lbl, rel_x, rel_y, size_um, depth = feat
        # Repeat each structural feature across slide in a grid
        grid = max(1, int(math.sqrt(slide_um / (size_um * 3))))
        for gi in range(grid):
            for gj in range(grid):
                ax = (rel_x + gi) / grid * slide_um
                ay = (rel_y + gj) / grid * slide_um
                uid = f'{lbl[:8]}_{len(layout)}'
                layout.append({
                    'uid': uid, 'label': lbl,
                    'ax': ax, 'ay': ay,
                    'size_m': size_um * 1e-6, 'depth': depth,
                    'z': 0.12, 'mobile': False,
                    'vx': 0.0, 'vy': 0.0,
                })

    # Motile elements (RBCs, leukocytes) placed along capillary paths
    for mot in spec.get('motile', []):
        lbl, n, diam_um, h_um, z_mot, speed = mot
        size_m = diam_um * 1e-6
        # Several capillary tracks at random angles
        for track_i in range(max(1, n // 3)):
            base_x = random.uniform(0.1, 0.9) * slide_um
            base_y = random.uniform(0.1, 0.9) * slide_um
            angle  = random.uniform(0, math.pi * 2)
            for ci in range(3):
                uid = f'{lbl[:6]}_t{track_i}_{ci}_{len(layout)}'
                offset = (ci / 3) * slide_um * 0.8
                ax = base_x + math.cos(angle) * offset
                ay = base_y + math.sin(angle) * offset
                layout.append({
                    'uid': uid, 'label': lbl,
                    'ax': ax % slide_um, 'ay': ay % slide_um,
                    'size_m': size_m, 'depth': 'd1_vascular',
                    'z': z_mot,
                    'mobile': True,
                    'vx': math.cos(angle) * speed * spec['fov_um'] / 1000,
                    'vy': math.sin(angle) * speed * spec['fov_um'] / 1000,
                    '_angle': angle, '_speed': speed * spec['fov_um'] / 1000,
                })

    return layout


def step_motile(layout, dt, slide_um):
    """Advance motile elements (RBCs etc.) along their track."""
    for e in layout:
        if not e['mobile']:
            continue
        e['ax'] = (e['ax'] + e['vx'] * dt) % slide_um
        e['ay'] = (e['ay'] + e['vy'] * dt) % slide_um


# ── Snapshot builder ──────────────────────────────────────────────────────────
def build_snapshot(tissue_name, layout, stage, t, prev_pos):
    spec     = TISSUES[tissue_name]
    slide_um = spec['slide_um']
    symbols  = []
    margin   = spec['fov_um'] * 0.15   # slight overshoot for fade-in

    for e in layout:
        if not stage.in_fov(e['ax'], e['ay'], margin):
            continue
        nx, ny = stage.to_norm(e['ax'], e['ay'])
        nz     = round(max(0.01, min(0.28, e['z'])), 4)

        uid = e['uid']
        if uid in prev_pos:
            pnx, pny = prev_pos[uid]
            vx = round((nx - pnx) / DT, 5)
            vy = round((ny - pny) / DT, 5)
        else:
            vx = round(e['vx'] / spec['fov_um'], 5)
            vy = round(e['vy'] / spec['fov_um'], 5)

        symbols.append({
            'id':   uid,
            'type': 'CUSTOM',
            'position': {'x': nx, 'y': ny, 'z': nz},
            'velocity': {'x': vx, 'y': vy, 'z': 0.0},
            'properties': {
                'label':       e['label'],
                'scale_m':     str(round(e['size_m'], 9)),
                'diameter_m':  str(round(e['size_m'], 9)),
                'depth_class': e['depth'],
                'track_id':    uid,
                'tissue':      tissue_name,
                'fov_um':      str(spec['fov_um']),
            }
        })

    return {
        'timestamp': {'unix': int(t * 1000)},
        'bounds':    BOUNDS,
        'symbols':   symbols,
        'metadata':  {
            'context':       spec['context'],
            'modality':      'virtual_microscopy',
            'frame_t':       str(round(t, 3)),
            'tissue':        tissue_name,
            'fov_um':        str(spec['fov_um']),
            'stage_x_um':    str(round(stage.x_um, 1)),
            'stage_y_um':    str(round(stage.y_um, 1)),
            'z_focus':       str(round(stage.z_focus, 4)),
            'symbol_count':  str(len(symbols)),
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
        description='Stream virtual histology microscopy into the neural fabric.')
    ap.add_argument('--host',    default='192.168.1.84')
    ap.add_argument('--port',    type=int, default=8090)
    ap.add_argument('--fps',     type=int, default=FPS)
    ap.add_argument('--tissue',  default=None, choices=list(TISSUES.keys()),
                    help='Lock to one tissue type instead of cycling')
    ap.add_argument('--duration',type=int, default=TISSUE_DURATION,
                    help='Seconds per tissue before cycling (default 90)')
    args = ap.parse_args()

    print(f'Connecting to {args.host}:{args.port} ...', flush=True)
    if not check_connection(args.host, args.port):
        print('Node not reachable. Start the node first.', flush=True)
        sys.exit(1)
    print('Connected. Starting histology stream.', flush=True)
    print(f'  Tissues: {" -> ".join(TISSUE_ORDER)} ({args.duration}s each)', flush=True)
    if args.tissue:
        print(f'  Locked to: {args.tissue}', flush=True)
    print('Press Ctrl+C to stop.', flush=True)

    interval      = 1.0 / args.fps
    frame         = 0
    t             = 0.0
    drift         = 0.0
    prev_pos      = {}
    layout        = None
    stage         = None
    cur_tissue    = None

    while True:
        t0 = time.perf_counter()

        # ── Tissue cycling ───────────────────────────────────────────────
        if args.tissue:
            tissue_name = args.tissue
        else:
            idx         = int(t // args.duration) % len(TISSUE_ORDER)
            tissue_name = TISSUE_ORDER[idx]

        if tissue_name != cur_tissue:
            print(f'\n  -> Switching to: {tissue_name}', flush=True)
            print(f'    {TISSUES[tissue_name]["description"]}', flush=True)
            spec        = TISSUES[tissue_name]
            layout      = generate_tissue_layout(tissue_name, spec['slide_um'])
            stage       = VirtualStage(spec['slide_um'], spec['fov_um'],
                                       spec['scan_speed_um_s'])
            prev_pos    = {}
            cur_tissue  = tissue_name
            print(f'    Layout: {len(layout)} objects in {spec["slide_um"]} µm slide',
                  flush=True)

        # ── Step physics ─────────────────────────────────────────────────
        step_motile(layout, interval, spec['slide_um'])
        stage.step(interval)

        # ── Build + send ─────────────────────────────────────────────────
        snap   = build_snapshot(tissue_name, layout, stage, t, prev_pos)
        result = post(args.host, args.port, '/neuro/train', {'snapshot': snap})

        # Update previous positions
        for sym in snap['symbols']:
            p = sym['position']
            prev_pos[sym['id']] = (p['x'], p['y'])

        if frame % (args.fps * 5) == 0:
            lc = result.get('label_count', '?')
            print(f'  frame {frame:06d}  t={t:.1f}s  tissue={tissue_name:<20}  '
                  f'symbols={len(snap["symbols"]):3d}  labels={lc}', flush=True)

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
