#!/usr/bin/env python3
"""
Train the neural fabric on a plant cell (Elodea mesophyll) at 6 physical scales.

Scales trained:
  0  plant_cell       ~20 µm  whole cell phenotype
  1  cell_wall        ~1  µm  cellulose wall + membrane
  2  chloroplast      ~5  µm  photosynthetic organelle
  3  thylakoid        ~10 nm  membrane + pigment complexes
  4  chlorophyll      ~1  nm  chlorophyll a molecule
  5  carbon_atom      ~154 pm atomic structure

Run:
  python scripts/train_cell_layers.py --host 192.168.1.84 --port 8090 --passes 5
"""
import argparse, json, time, sys
import urllib.request, urllib.error

# ── Helpers ──────────────────────────────────────────────────────────────────

def post(host, port, path, body):
    url  = f"http://{host}:{port}{path}"
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} {path}: {e.read().decode()[:120]}")
        return {}

def train_text(host, port, text):
    r = post(host, port, "/media/train", {"modality": "text", "text": text.strip()})
    return r.get("label_count", 0)

def calibrate(host, port, entity_type, scale_m):
    r = post(host, port, "/overlay/layers/calibrate",
             {"entity_type": entity_type, "known_scale_m": scale_m})
    return r.get("ok", False)

# ── Training corpus ──────────────────────────────────────────────────────────

SCALES = [
    # (entity_type, known_scale_m)
    ("plant_cell",    20e-6),
    ("cell_wall",     0.5e-6),
    ("chloroplast",   5e-6),
    ("thylakoid",     10e-9),
    ("chlorophyll",   1e-9),
    ("carbon_atom",   0.154e-9),
]

CORPUS = [
    # ── 0: Whole plant cell ───────────────────────────────────────────────────
    ("plant_cell", """
The Elodea mesophyll plant cell is a rectangular box-shaped cell 20 to 50 micrometres
long and 10 to 20 micrometres wide. The outer surface is a rigid cell wall made of
cellulose giving the cell its boxy rectangular shape. The green colour comes from
chloroplasts distributed in the cytoplasm just inside the wall. A large central vacuole
occupies up to 90 percent of the cell volume and appears as a clear region inside.
The plasma membrane lies just inside the cell wall. The nucleus is a spherical body
5 to 10 micrometres in diameter. Multiple chloroplasts each 4 to 6 micrometres long
appear as bright green oval bodies. Cytoplasmic streaming moves chloroplasts in a
circular path. The outer wall appears dark under light microscopy. The cell interior
is green with moving oval organelles. Four walls form the rectangular border.
"""),
    ("plant_cell", """
The plant cell outer layer is a cell wall made of cellulose fibres. Inside the wall
is the plasma membrane. The cell interior cytoplasm contains organelles. The nucleus
stores DNA and controls cell activity. Chloroplasts carry out photosynthesis. The
large vacuole stores water and maintains turgor pressure. Mitochondria provide energy.
The endoplasmic reticulum forms a network of membranes. Ribosomes make proteins.
The cell is enclosed by the outer cell wall surface exterior. Looking at the cell from
outside you see the flat rectangular face of the wall. The corners of the box cell
are sharp. The cell surface is the outermost visible layer of the plant cell.
"""),

    # ── 1: Cell wall ──────────────────────────────────────────────────────────
    ("cell_wall", """
The plant cell wall is a layered composite structure. The primary cell wall is
0.1 to 0.5 micrometres thick made of cellulose microfibrils in a hemicellulose
matrix. Each cellulose microfibril is 5 to 12 nanometres in diameter and several
micrometres long. The microfibrils cross each other in different orientations
forming a woven mesh. Pectin fills the spaces between microfibrils. The middle
lamella between two adjacent cells contains pectin and calcium. Plasmodesmata
are cytoplasmic channels 50 to 60 nanometres wide passing through the wall.
The plasma membrane beneath the wall is a lipid bilayer 7 to 10 nanometres thick.
The cell wall provides structural support and protection. Its surface appears
fibrous and layered when seen by electron microscopy. The outer face of the wall
is the exterior surface of the plant cell.
"""),
    ("cell_wall", """
Cellulose fibres in the cell wall are arranged in layers. Each layer has fibres
running in a different direction creating strength. The middle lamella is the first
layer deposited during cell division. The primary wall is thin and flexible. Some
cells deposit a secondary wall inside the primary wall. The secondary wall is much
thicker with lignin. Lignin makes wood cells rigid and impermeable. The wall layers
from outside to inside are: middle lamella outer layer, primary wall layer, secondary
wall inner layer, plasma membrane. Each layer has different chemistry and thickness.
The fibrous texture of the wall surface shows under electron microscopy.
"""),

    # ── 2: Chloroplast ────────────────────────────────────────────────────────
    ("chloroplast", """
The chloroplast is an oval organelle 3 to 10 micrometres long and 1 to 3 micrometres
wide enclosed by a double outer envelope membrane. The stroma is a fluid matrix inside
filled with enzymes for the Calvin cycle carbon fixation. Within the stroma stacks of
flat membrane discs called grana contain chlorophyll. Each granum is a stack of
thylakoid discs 300 to 600 nanometres in diameter. A mature chloroplast has 10 to 100
grana. Thylakoid membranes hold photosystems and the electron transport chain.
The chloroplast contains its own circular DNA. Plastoglobules are small lipid droplets
100 to 500 nanometres across in the stroma. The inner envelope membrane separates
stroma from the intermembrane space. The outer envelope is smooth. The thylakoid
lumen is the space inside each disc 10 to 30 nanometres wide.
"""),
    ("chloroplast", """
Inside the chloroplast the stacked grana look like piles of coins under electron
microscopy. Each coin-shaped thylakoid disc is a flattened membrane sac. The grana
are interconnected by stromal lamellae flat membrane tubes running through the stroma.
Photosynthesis happens in two stages. The light reactions happen in the thylakoid
membranes. The dark reactions happen in the stroma. The green colour of chloroplasts
comes from chlorophyll pigment in the thylakoid membranes. Chloroplasts were once
free-living cyanobacteria that were incorporated into eukaryotic cells by endosymbiosis.
The outer envelope of the chloroplast is the surface boundary of this organelle.
"""),

    # ── 3: Thylakoid ──────────────────────────────────────────────────────────
    ("thylakoid", """
The thylakoid membrane is a lipid bilayer 7 to 8 nanometres thick. Large protein
complexes are embedded in it: Photosystem II is 20 to 25 nanometres wide. The
cytochrome b6f complex transfers electrons. Photosystem I reduces NADP+ to NADPH.
ATP synthase makes ATP using the proton gradient. Photosystem II contains 35
chlorophyll molecules and the P680 reaction centre absorbing at 680 nanometres.
Light-harvesting complex LHCII holds 14 chlorophyll molecules. Photosystem I
holds 100 chlorophyll molecules and P700 absorbing at 700 nanometres. The
thylakoid lumen is acidic at pH 5 during photosynthesis. Electrons flow from
water through PSII along the electron transport chain to PSI. The outer surface
of the thylakoid disc faces the stroma. Protein bumps visible on the outer surface
are the photosystem complexes.
"""),
    ("thylakoid", """
Chlorophyll molecules in the thylakoid membrane are arranged in antenna arrays.
Light hits an antenna chlorophyll and the energy is passed from molecule to molecule
until it reaches the reaction centre. Energy transfer in the antenna complex
happens in femtoseconds. Each chlorophyll absorbs specific wavelengths of light.
Chlorophyll a absorbs red 662 nm and blue 430 nm light most strongly. The
porphyrin head of chlorophyll is held flat in the membrane by the protein scaffold.
The phytol tail anchors chlorophyll into the lipid bilayer. The magnesium ion at
the centre of the porphyrin ring is essential for light absorption. Carotenoid
pigments also absorb light in the 400 to 500 nm range.
"""),

    # ── 4: Chlorophyll molecule ───────────────────────────────────────────────
    ("chlorophyll", """
Chlorophyll a has molecular formula C55H72O5N4Mg and molecular weight 893 daltons.
The molecule has a flat porphyrin ring head approximately 1.2 nanometres across.
The porphyrin ring contains four pyrrole rings connected by methine bridges in a
macrocycle. A central magnesium atom is coordinated by four nitrogen atoms from
the pyrrole rings. The phytol tail is a 20-carbon isoprenoid chain approximately
2.8 nanometres long. Carbon-carbon bond lengths in the ring are 0.140 nanometres
for aromatic bonds. Nitrogen-carbon bonds are 0.134 nanometres. The magnesium-nitrogen
bond is 0.208 nanometres. The pi electron system is delocalized over the entire
porphyrin ring. The molecule is essentially planar with the phytol tail extending
perpendicular from the edge of the ring. Absorption peaks are at 430 nm and 662 nm.
"""),
    ("chlorophyll", """
The porphyrin ring of chlorophyll is similar to haem in haemoglobin but contains
magnesium instead of iron. The ring system is aromatic with 18 pi electrons following
Huckel's rule. Four nitrogen atoms point inward holding the magnesium. The ring
carbons are approximately 0.14 nanometres apart. Carbon atoms in the ring are sp2
hybridized forming a planar network. The ester linkage connects the phytol tail
to the ring. Oxygen atoms at the ester group are 0.122 nanometres from carbon.
The entire molecule from ring to tail end is approximately 3 to 4 nanometres long.
The molecule is a flat disc with a long flexible tail.
"""),

    # ── 5: Carbon atom ────────────────────────────────────────────────────────
    ("carbon_atom", """
Carbon has atomic number 6 with 6 protons 6 neutrons and 6 electrons. The nuclear
radius is approximately 3 femtometres. The covalent radius is 77 picometres.
The van der Waals radius is 170 picometres. Electron configuration is 1s2 2s2 2p2.
The 1s shell holds 2 electrons close to the nucleus. The 2s orbital holds 2 electrons.
The 2p orbital holds 2 electrons in lobed orbitals above and below. In sp2 hybridization
three hybrid orbitals point in a plane at 120 degrees forming sigma bonds. The
unhybridized p orbital forms a pi bond perpendicular to the plane. Bond length
for a C-C single bond is 154 picometres. A C=C double bond is 134 picometres.
The C-H bond is 109 picometres. First ionization energy is 11.26 electron volts.
"""),
    ("carbon_atom", """
The quantum mechanical model of carbon describes electrons as wavefunctions.
The 1s orbital is a sphere close to the nucleus. The 2s orbital is a larger sphere.
The three 2p orbitals are dumbbell-shaped lobes oriented along x y z axes. In the
ground state the two 2p electrons occupy different orbitals with parallel spins by
Hund's rule. The de Broglie wavelength of a 2p electron is approximately 0.3 nm.
The Bohr radius is 52.9 picometres defining the most probable electron position
in hydrogen. Carbon nuclear charge 6 pulls electrons closer. Electron probability
density maps show a fuzzy cloud around the nucleus. Quantum numbers n l m s
describe each electron state. The nucleus consists of 6 protons and 6 neutrons
bound by the strong nuclear force. Proton diameter is approximately 1.7 femtometres.
"""),
]

# ── Quality check ─────────────────────────────────────────────────────────────

def fz(v):
    if v is None: return "?"
    if v < 1e-12: return f"{v*1e15:.1f}fm"
    if v < 1e-9:  return f"{v*1e12:.0f}pm"
    if v < 1e-6:  return f"{v*1e9:.2f}nm"
    if v < 1e-3:  return f"{v*1e6:.1f}um"
    if v < 1:     return f"{v*1e3:.1f}mm"
    return f"{v:.3f}m"

PROBE_LABELS = {
    "plant_cell":  ["outer","wall","surface","membrane","cell","nucleus","vacuole",
                    "chloroplast","cytoplasm","layer","organelle","interior"],
    "cell_wall":   ["wall","layer","cellulose","fibril","membrane","surface","outer",
                    "middle","primary","secondary","pectin","lamella"],
    "chloroplast": ["organelle","membrane","stroma","thylakoid","disc","granum",
                    "stack","inner","outer","envelope","lumen"],
    "thylakoid":   ["membrane","layer","protein","complex","surface","channel",
                    "electron","light","photosystem","chlorophyll"],
    "chlorophyll": ["molecule","ring","atom","bond","carbon","nitrogen","magnesium",
                    "oxygen","hydrogen","chain","tail","porphyrin"],
    "carbon_atom": ["atom","nucleus","electron","orbital","quantum","bond",
                    "proton","neutron","spin","wavefunction","shell"],
}

def quality_check(host, port):
    print("\n=== Layer decomposition quality check ===")
    print(f"{'ENTITY':<16} {'CAL':3} {'N':>2}  {'Z_MIN':>10} {'Z_MAX':>10}  LAYERS")
    print("-"*80)
    for entity, scale_m in SCALES:
        labels = PROBE_LABELS[entity]
        n = len(labels)
        scores = {l: 1.0 - i/n for i,l in enumerate(labels)}
        body = {"entities":[{
            "entity_id":     entity,
            "phenotype":     entity,
            "active_labels": labels,
            "label_scores":  scores,
            "x_frac": 0.5, "y_frac": 0.5,
            "size_est": scale_m,
        }]}
        r = post(host, port, "/overlay/layers", body)
        reps = r.get("layers", [])
        if not reps:
            print(f"  {entity:<16}: NO REPORT")
            continue
        rep = reps[0]
        sp = rep["scale_profile"]
        lyrs = rep["layers"]
        if not lyrs:
            print(f"  {entity:<16}: 0 layers")
            continue
        zvs = [l["position"]["z_est_m"] for l in lyrs]
        layer_brief = "  ".join(f"d{l['depth']}:{l['label'][:10]}" for l in sorted(lyrs, key=lambda x: x["depth"])[:4])
        print(f"  {entity:<16} {'T' if sp['calibrated'] else 'F':3} {len(lyrs):>2}  {fz(min(zvs)):>10} {fz(max(zvs)):>10}  {layer_brief}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",  default="192.168.1.84")
    ap.add_argument("--port",  type=int, default=8090)
    ap.add_argument("--passes",type=int, default=5)
    args = ap.parse_args()
    h, p = args.host, args.port

    # Connectivity check — snapshot is GET, not POST.
    url = f"http://{h}:{p}/neuro/snapshot"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            r = json.loads(resp.read())
    except Exception as e:
        print(f"ERROR: cannot reach {h}:{p} — {e}"); sys.exit(1)
    print(f"Connected to {h}:{p}  pool labels: {len(r.get('active_labels',[]))}")

    # Register calibrations first.
    print("\n=== Registering physical scale calibrations ===")
    for entity, scale_m in SCALES:
        ok = calibrate(h, p, entity, scale_m)
        print(f"  {'OK' if ok else '!!'} {entity:<16} -> {fz(scale_m)}")

    # Training passes.
    total = 0
    for pass_idx in range(args.passes):
        print(f"\n=== Training pass {pass_idx+1}/{args.passes} ===")
        for i, (entity, text) in enumerate(CORPUS):
            n = train_text(h, p, text)
            total += n
            print(f"  [{i+1:02d}/{len(CORPUS)}] {entity:<16} +{n:4d} labels")
            time.sleep(0.15)

    print(f"\nTotal labels trained across {args.passes} passes: {total}")
    quality_check(h, p)
    print("\nDone. Open cell_viewer.html to explore.")

if __name__ == "__main__":
    main()
