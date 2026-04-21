#!/usr/bin/env python3
"""
Multi-scale anatomy ingestion for layered physiology demo.

Ingests text describing a game console at 5 biological/physical scales,
trains the neuro pool, then registers calibration scales via the API.

Scales (outermost -> deepest):
  0  -- phenotype  (whole device, ~30 cm)
  1  -- subsystem  (PCB/board level, ~10 cm)
  2  -- component  (chip/IC, ~1 cm)
  3  -- circuit    (transistor/gate, ~10 µm -> 100 nm)
  4  -- atomic     (silicon lattice, ~0.5 nm)

Usage:
  python scripts/ingest_anatomy_layers.py [--host 192.168.1.84] [--port 8090]
"""
import argparse, json, time, sys
import urllib.request, urllib.error

def post(host, port, path, body):
    url  = f"http://{host}:{port}{path}"
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} on {path}: {e.read().decode()[:200]}")
        return {}
    except Exception as e:
        print(f"  ERROR {path}: {e}")
        return {}

def get(host, port, path):
    url = f"http://{host}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  ERROR GET {path}: {e}")
        return {}

# -- Scale-annotated training corpus -----------------------------------------
# Each block has: title, text (rich prose), scale label, known_scale_m
CORPUS = [
    # -- Scale 0: phenotype (whole device) -----------------------------------
    {
        "scale_label": "game_console",
        "known_scale_m": 0.30,
        "depth_hint": 0,
        "title": "Game console -- exterior design and housing",
        "text": """
A gaming console is an electronic device designed to output a video signal or image
to display a video game. The outer shell and housing is typically injection-molded
plastic, providing structural rigidity and aesthetic form. The exterior casing
measures approximately 30 cm in its longest dimension. The surface of the console
has ventilation slots, USB ports, disc tray, and power button embedded in the
external plastic shell. The housing color is typically matte black or glossy white.
Rubber feet on the underside prevent sliding. The chassis holds all internal
components in spatial alignment. Heat dissipation requires that the surface have
carefully designed vent openings. The outer shell protects the circuit board from
dust, moisture, and physical damage. The exterior is the primary phenotypic feature
observed by the user -- shape, color, surface texture, and port arrangement all
constitute the visible phenotype of the device.
"""
    },
    {
        "scale_label": "game_console",
        "known_scale_m": 0.30,
        "depth_hint": 0,
        "title": "Console subsystems and internal layout",
        "text": """
Inside a gaming console the main subsystems are spatially arranged on a single
printed circuit board (PCB) or motherboard. The PCB acts as the structural and
electrical backbone. The cooling subsystem includes a heat sink and fan assembly
positioned over the main processor. The optical disk drive occupies a significant
portion of the interior volume. The power supply unit converts mains AC to DC
voltages required by the components. The hard disk drive or SSD storage module
is mounted in a bay on the chassis interior. The HDMI output board, USB controller
board, and Bluetooth/Wi-Fi radio module are subsidiary boards connected to the
motherboard via flex cables and connectors. The internal layout follows thermal
management principles -- heat-producing components are near vents, and airflow
channels direct heat away from sensitive flash memory and analog components.
"""
    },

    # -- Scale 1: subsystem / PCB ---------------------------------------------
    {
        "scale_label": "pcb",
        "known_scale_m": 0.10,
        "depth_hint": 1,
        "title": "Printed circuit board -- traces, pads, and layers",
        "text": """
The printed circuit board (PCB) in a gaming console is a multi-layer fiberglass
substrate with copper traces etched into each layer. Typical designs use 6 to 12
copper layers separated by insulating prepreg. Surface-mount components are soldered
onto pads on the top and bottom layers. The board dimensions are roughly 10 cm x 15 cm.
Copper traces carry signals between the CPU, GPU, memory, and peripheral controllers.
Ground and power planes occupy dedicated internal layers to minimize impedance and
electromagnetic interference. Via holes connect traces between layers using copper-
plated drill holes with diameters typically 0.1 to 0.3 mm. The solder mask covers
all exposed copper except pads, preventing shorts. Silkscreen layers mark component
designations. Signal integrity requires careful impedance-controlled routing for
high-speed DDR memory lanes and PCIe lanes connecting the CPU and GPU.
"""
    },
    {
        "scale_label": "pcb",
        "known_scale_m": 0.10,
        "depth_hint": 1,
        "title": "Board-level components: capacitors, resistors, inductors",
        "text": """
On the PCB surface hundreds of passive components populate the board. Bypass
capacitors filter power supply noise at each IC power pin -- typical values
range from 100 nF to 10 µF in 0402 and 0201 package sizes. Bulk capacitors
near the voltage regulators store charge for transient current demand during
processor load spikes. Ferrite bead inductors on power rails filter high-
frequency noise. Pull-up and pull-down resistors set logic levels on I2C,
SPI, and GPIO lines. Crystal oscillators provide the reference clock frequency --
typically 25 MHz for Ethernet PHY and 27 MHz for video clock generation.
Series resistors dampen signal ringing on high-speed traces. Electrostatic
discharge protection diodes guard the USB and HDMI connectors. The board is
comprehensively decoupled with a capacitance network across the power delivery
network to maintain stable voltages under dynamic load conditions.
"""
    },

    # -- Scale 2: chip / IC ---------------------------------------------------
    {
        "scale_label": "integrated_circuit",
        "known_scale_m": 0.012,
        "depth_hint": 2,
        "title": "System-on-chip (SoC): CPU, GPU, memory interface",
        "text": """
The heart of a modern gaming console is a custom system-on-chip (SoC) fabricated
in a 7 nm or 5 nm CMOS process. The SoC integrates a multi-core CPU based on the
ARM Cortex-A or x86-64 architecture with a high-performance GPU sharing an on-
package memory subsystem. The die size is approximately 200 to 300 mm^2 for high-
end designs. Unified memory architecture places GDDR6 or LPDDR5 memory in close
proximity to the processor die using a high-bandwidth memory interface with data
rates exceeding 400 GB/s. The SoC die contains billions of transistors laid out
in functional blocks: CPU cores, GPU shader clusters, video encode/decode engines,
display engine, audio DSP, security engine, and IO controllers. The package substrate
bonds the die to the PCB via a ball grid array (BGA) with hundreds of solder balls
on a 0.65 mm pitch. Thermal interface material and a copper heat spreader cap the
die package.
"""
    },
    {
        "scale_label": "integrated_circuit",
        "known_scale_m": 0.012,
        "depth_hint": 2,
        "title": "Memory die: GDDR6 high-bandwidth architecture",
        "text": """
High-bandwidth memory in a gaming console uses GDDR6 DRAM organized in banks and
sub-arrays. The memory die is fabricated on a dedicated DRAM process node optimized
for cell density rather than logic speed. Each memory cell consists of a capacitor
and an access transistor. The capacitor stores charge representing a logic 0 or 1.
The cell pitch in modern GDDR6 is approximately 50 nm in the bit line direction and
70 nm in the word line direction. Row and column decoders translate addresses to
specific wordlines and bitlines. Sense amplifiers detect the tiny voltage swing
when a cell is read. Refresh cycles restore charge to prevent data loss from leakage.
The data bus interface uses differential signaling on a 256-bit bus with source-
synchronous clocking. Die-to-die connections in stacked configurations use through-
silicon vias (TSV) to minimize package latency.
"""
    },

    # -- Scale 3: circuit / transistor (10 nm range) -------------------------
    {
        "scale_label": "transistor",
        "known_scale_m": 7e-9,
        "depth_hint": 3,
        "title": "FinFET transistor geometry at 7 nm node",
        "text": """
At the 7 nm process node, transistors are FinFET (fin field-effect transistor)
structures with silicon fins approximately 5-7 nm wide. The fin height is
approximately 50 nm. The gate wraps around three sides of the fin to improve
electrostatic control and reduce short-channel leakage. The gate dielectric is
hafnium oxide (HfO2) with an equivalent oxide thickness below 1 nm. The gate
length (Lg) is approximately 12 nm. Metal gate electrodes replace polysilicon to
reduce gate resistance at these dimensions. Source and drain regions are strained
silicon-germanium epitaxially grown to enhance carrier mobility. The contacted
poly pitch (CPP) determines transistor density -- at 7 nm this is approximately
56 nm. Local interconnects connect transistors within a standard cell using
cobalt or ruthenium fill metals at M0 and M1 layers. The gate cut is defined by
extreme ultraviolet (EUV) lithography with a wavelength of 13.5 nm.
"""
    },
    {
        "scale_label": "transistor",
        "known_scale_m": 7e-9,
        "depth_hint": 3,
        "title": "Logic gate implementation: NAND, NOR, inverter standard cells",
        "text": """
Standard cells in a 7 nm library implement logic gates using networks of NMOS and
PMOS FinFET transistors. An inverter uses one NMOS and one PMOS transistor in
series between VDD and VSS. A NAND2 gate uses two NMOS transistors in series and
two PMOS in parallel. Standard cell height is typically 6 to 8 metal track heights --
at 7 nm this corresponds to approximately 240 to 320 nm. The transistor gate pitch
within a cell is the contacted poly pitch of 56 nm. Cells are abutted horizontally
to share power rails. Placement and route tools fill the die with millions of
standard cells and connect them via copper interconnect at metal layers M1 through M12.
Via arrays connect metal layers with tungsten-filled contacts at M0 and copper or
cobalt at higher layers. The routing pitch decreases from 48 nm at M1 to 80 nm at
upper layers where resistance per length is the constraint. Signal timing closure
requires careful buffering of long interconnects to meet setup and hold constraints.
"""
    },

    # -- Scale 4: atomic / crystal lattice -----------------------------------
    {
        "scale_label": "silicon_crystal",
        "known_scale_m": 5.43e-10,
        "depth_hint": 4,
        "title": "Silicon crystal lattice and atomic spacing",
        "text": """
Silicon crystallizes in the diamond cubic structure with a lattice constant of
5.43 Å (0.543 nm). Each silicon atom forms four covalent bonds with nearest
neighbors in a tetrahedral geometry with a bond length of 2.35 Å. The (100) surface
planes are used for CMOS device fabrication because they support the highest
carrier mobility. Crystal defects such as dislocations, vacancies, and interstitials
affect dopant diffusion and carrier lifetime. Dopant atoms -- phosphorus (n-type)
and boron (p-type) -- occupy substitutional sites in the silicon lattice. The
covalent radius of silicon is 1.11 Å. The electron mean free path in bulk silicon
at room temperature is approximately 10 to 40 nm depending on doping concentration.
Phonon scattering limits carrier mobility. Quantum confinement effects emerge when
silicon dimensions approach the de Broglie wavelength of electrons (~5-10 nm at
room temperature), leading to quantized energy subbands in ultra-thin fin structures.
"""
    },
    {
        "scale_label": "silicon_crystal",
        "known_scale_m": 5.43e-10,
        "depth_hint": 4,
        "title": "Electron behavior and quantum effects in nanoscale silicon",
        "text": """
In nanometer-scale silicon transistors quantum mechanical effects dominate device
behavior. The electron wavefunction extends beyond the classical depletion region
boundary -- this quantum tunneling allows gate leakage current even through thin
gate dielectrics. The de Broglie wavelength of an electron in silicon at room
temperature is approximately 5 to 12 nm, comparable to the channel length at
advanced nodes. Energy quantization splits the conduction band into discrete
subbands. Band-to-band tunneling in highly doped junctions generates leakage
current that scales adversely with voltage scaling. The Heisenberg uncertainty
principle limits the simultaneous precision of electron momentum and position --
at the scale of a 5 nm transistor this fundamentally constrains device design.
Coulomb blockade can occur in the smallest nanowire transistors, where single
electrons control the channel conductance. Spin-orbit coupling influences carrier
transport in strained silicon and germanium channels at these atomic dimensions.
"""
    },
]

# -- Calibration scales for each entity type ---------------------------------
CALIBRATIONS = [
    ("game_console",      0.30),
    ("pcb",               0.10),
    ("integrated_circuit",0.012),
    ("transistor",        7e-9),
    ("silicon_crystal",   5.43e-10),
]

_DOC_COUNTER = [0]

def ingest_doc(host, port, title, text, tags=None):
    _DOC_COUNTER[0] += 1
    doc_id = f"anatomy_{_DOC_COUNTER[0]:04d}"
    body = {
        "document": {
            "doc_id": doc_id,
            "source": "ingest_anatomy_layers",
            "title":  title,
            "text_blocks": [
                {
                    "block_id":   f"{doc_id}_b0",
                    "text":       text.strip(),
                    "section":    title,
                    "order":      0,
                    "source":     "anatomy_script",
                    "confidence": 1.0,
                }
            ],
            "figures":  [],
            "metadata": {"tags": tags or []},
        }
    }
    r = post(host, port, "/knowledge/ingest", body)
    return r.get("label_count", 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.84")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--passes", type=int, default=2,
                    help="Training passes over the corpus")
    args = ap.parse_args()
    h, p = args.host, args.port

    # Quick connectivity check.
    status = get(h, p, "/node/info")
    if not status:
        # Fall back to bridge info endpoint.
        status = get(h, p, "/neuro/snapshot") or {}
    if not status:
        print(f"ERROR: cannot reach {h}:{p} -- start the node first.")
        sys.exit(1)
    print(f"Connected to {h}:{p}")

    total_labels = 0
    for pass_idx in range(args.passes):
        print(f"\n=== Training pass {pass_idx+1}/{args.passes} ===")
        for i, doc in enumerate(CORPUS):
            tags = [doc["scale_label"], f"depth_{doc['depth_hint']}"]
            n = ingest_doc(h, p, doc["title"], doc["text"], tags)
            total_labels += n
            print(f"  [{i+1:02d}/{len(CORPUS)}] {doc['title'][:55]:<55}  +{n} labels")
            time.sleep(0.25)

    print(f"\nTotal labels ingested: {total_labels}")

    # Register physical scale calibrations.
    print("\n=== Registering calibration scales ===")
    for entity_type, scale_m in CALIBRATIONS:
        r = post(h, p, "/overlay/layers/calibrate", {
            "entity_type":   entity_type,
            "known_scale_m": scale_m,
        })
        sci = f"{scale_m:.2e} m"
        ok  = "OK" if r.get("ok") else "!!"
        print(f"  [{ok}]  {entity_type:<22} -> {sci}")

    # Test decompose for each scale.
    print("\n=== Layer decomposition quality check ===")
    test_entities = [
        ("game_console", ["outer", "shell", "housing", "chassis", "exterior",
                          "surface", "casing", "plastic", "vent", "port"]),
        ("pcb",          ["pcb", "circuit", "board", "trace", "pad", "capacitor",
                          "resistor", "via", "layer", "solder"]),
        ("integrated_circuit", ["chip", "ic", "die", "transistor", "gate", "core",
                                 "gpu", "cpu", "memory", "bandwidth"]),
        ("transistor",   ["transistor", "gate", "channel", "finfet", "silicon",
                          "dopant", "electron", "oxide", "contact", "lithography"]),
        ("silicon_crystal", ["atom", "crystal", "lattice", "silicon", "bond",
                              "electron", "quantum", "orbital", "spin", "phonon"]),
    ]

    for entity_id, labels in test_entities:
        scores = {l: 1.0 - (i / len(labels)) for i, l in enumerate(labels)}
        body = {
            "entities": [{
                "entity_id":     entity_id,
                "phenotype":     entity_id,
                "active_labels": labels,
                "label_scores":  scores,
                "x_frac":        0.5,
                "y_frac":        0.5,
                "size_est":      0.1,
            }]
        }
        r = post(h, p, "/overlay/layers", body)
        reports = r.get("layers", [])
        if not reports:
            print(f"  {entity_id}: no report")
            continue
        rep = reports[0]
        sp  = rep.get("scale_profile", {})
        layers = rep.get("layers", [])
        max_d  = rep.get("max_depth", 0)
        calibrated = sp.get("calibrated", False)
        z_vals = [l["position"]["z_est_m"] for l in layers]
        eem_count = sp.get("eem_equation_count", 0)
        print(f"  {entity_id:<22}  layers={len(layers)}  max_depth={max_d}"
              f"  calibrated={calibrated}  eem_eqs={eem_count}"
              f"  z_range=[{min(z_vals):.2e}, {max(z_vals):.2e}]m" if z_vals else "  (no layers)")

    print("\nDone. Run layer_viewer.html to explore the 3D view.")

if __name__ == "__main__":
    main()
