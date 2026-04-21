#!/usr/bin/env python3
"""
Dense multi-scale scientific training corpus for plant cell physiology.
Journal-quality text with Latin/Greek nomenclature, crystallographic data,
biochemical pathway parameters, and SI measurements at each scale.

Scales:
  plant_cell     ~20 um  -- mesophyll cell ultrastructure
  cell_wall      ~0.5 um -- cellulose microfibril composite
  chloroplast    ~5 um   -- thylakoid/stroma architecture
  thylakoid      ~10 nm  -- photosynthetic membrane complexes
  chlorophyll    ~1 nm   -- porphyrin chromophore
  carbon_atom    ~154 pm -- sp2 hybridised carbon, quantum structure
"""
import argparse, json, time, sys, urllib.request, urllib.error

# -- HTTP helpers -------------------------------------------------------------

def post(h, p, path, body):
    url  = f"http://{h}:{p}{path}"
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} {path}: {e.read().decode()[:80]}")
        return {}

def train(h, p, text):
    r = post(h, p, "/media/train", {"modality": "text", "text": text.strip()})
    return r.get("label_count", 0)

def calibrate(h, p, entity, scale_m):
    r = post(h, p, "/overlay/layers/calibrate",
             {"entity_type": entity, "known_scale_m": scale_m})
    return r.get("ok", False)

# -- Physical calibrations -----------------------------------------------------

SCALES = [
    ("plant_cell",    20e-6),
    ("cell_wall",     0.5e-6),
    ("chloroplast",   5e-6),
    ("thylakoid",     10e-9),
    ("chlorophyll",   1e-9),
    ("carbon_atom",   0.154e-9),
]

# -- Corpus -- 5 documents per scale, journal-quality prose -------------------

CORPUS = [

# ====================================================================
# SCALE 0 -- Plant Cell Phenotype  (~20 um)
# ====================================================================

("plant_cell", """
The mesophyll parenchyma cell of Elodea canadensis (Michaux 1803) constitutes
the principal photosynthetic unit of the leaf lamina. The cell is bounded by a
primary cell wall (paries cellularis) 0.10-0.50 um in thickness, consisting
of a (1->4)-beta-D-glucan cellulose scaffold reinforced by xyloglucan,
arabinoxylan, and homogalacturonan pectin. The plasmalemma (plasma membrane,
unit membrane) subtends the wall at 7-10 nm thickness, maintaining a resting
electrical potential of −120 to −180 mV (electrogenic H+-ATPase, EC 3.6.3.6).
The central vacuole (vacuolum centrale) occupies 80-95% of the mature cell
volume and is bounded by the tonoplast (vacuolar membrane), which houses
V-type H+-ATPase and H+-PPase for luminal acidification (pH 5.0-5.5).
Cell dimensions: 25-50 um (longitudinal axis) x 15-25 um (transverse) x
10-20 um (depth), yielding a mean cell volume of approximately 5,000-12,000 um^3.
The nucleus (nucleus cellularis), 8-12 um in diameter, is enveloped by the
nuclear envelope (two unit membranes, 20-40 nm total), studded with nuclear pore
complexes (NPC, ~120 nm diameter, ~800 per nucleus). Heterochromatin lines
the inner nuclear membrane; euchromatin occupies the nucleoplasm. The nucleolus
(nucleolus, ~2-4 um) synthesises 18S, 5.8S, and 25S rRNA. Chloroplasts (5-50
per cell, mean ~20) are discoid, 3-8 um long, 1.5-3 um wide. Each chloroplast
contains 30-80 grana stacks. Mitochondria (chondriosome) number 200-2000 per
cell, 0.5-2 um length, with a distinctive double membrane and cristae system.
Rough endoplasmic reticulum (ergastoplasm) forms a continuous lumenal network
with the outer nuclear envelope. The Golgi apparatus (dictyosome) consists of
4-8 cisternae, 0.8-1.2 um diameter per cisterna, processing glycoproteins and
polysaccharides for secretion to the cell wall. Microtubules (24 nm outer
diameter) and actin microfilaments (7 nm) form the cytoskeletal network.
Plasmodesmata (symplastic connections, 40-60 nm diameter) traverse the cell
wall at a frequency of 1-15 per um^2 connecting adjacent cells.
"""),

("plant_cell", """
Mesophyll cell ultrastructure as revealed by transmission electron microscopy
(TEM) and cryogenic electron tomography shows a precisely organised cytoplasm
occupying the cortical layer beneath the plasmalemma (cortical cytoplasm).
Cytosolic ionic composition: K⁺ 80-160 mM, Mg^2⁺ 0.5-5 mM (free), Ca^2⁺
100-200 nM (resting, rises to 1-5 uM upon stimulus), pH 7.2-7.5, osmolarity
300-400 mOsm. The central vacuole contains a dilute solution of organic acids
(malate, citrate, oxalate), inorganic ions, hydrolytic enzymes (alpha-mannosidase,
beta-glucosidase, protease), phenolic compounds, and anthocyanins. Turgor pressure
(pressure potential ψp) of 0.3-1.0 MPa maintains cell rigidity. Chloroplasts
exhibit acropetal streaming (cyclosis) at rates of 2-5 um/s driven by myosin-XI
(plant myosin superfamily) on actin cables. Peroxisomes (0.2-1.5 um, bounded by
a single membrane) participate in the C2 photorespiratory pathway alongside
chloroplasts and mitochondria in a metabolon. Ribosomes (80S, 25 nm diameter)
stud the rough ER and the chloroplast ribosomes (70S, Svedberg unit) reside in
the plastid stroma. The cell plate (phragmoplast, nascent cell wall) forms from
fusing Golgi vesicles during cytokinesis. Cell division is preceded by a
preprophase band of microtubules defining the future division plane.
"""),

("plant_cell", """
Photosynthetic mesophyll cells are the primary site of carbon assimilation in
C3 plants via the Calvin-Benson-Bassham (CBB) cycle, fixing CO₂ through
ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO, EC 4.1.1.39).
RuBisCO constitutes 25-50% of total leaf nitrogen and is the most abundant
enzyme on Earth (~700 Tg globally). The Km(CO₂) of plant RuBisCO is ~10 uM
and kcat ~3 s⁻¹ per active site, necessitating its extreme abundance.
Carboxysomes are absent in eukaryotic chloroplasts; CO₂ concentration is
achieved by carbonic anhydrase (CA, EC 4.2.1.1) activity and the
CO₂/HCO₃⁻ equilibrium. Stomatal conductance to CO₂ (gs,CO₂) is 50-300
mmol m⁻^2 s⁻¹ in well-watered conditions. Mesophyll conductance (gm) limits
CO₂ diffusion from substomatal cavity to chloroplast stroma (~0.1-0.5 mol
m⁻^2 s⁻¹ bar⁻¹). Net photosynthesis (Pn) peaks at 15-30 umol CO₂ m⁻^2 s⁻¹
under light saturation (1000-1500 umol photons m⁻^2 s⁻¹) and 380 uL L⁻¹ CO₂.
The cell outer wall surface (epidermis-facing) is coated with a waxy cuticle
layer of cutin polyester and wax esters providing the hydrophobic exterior
barrier to water loss and pathogen ingress.
"""),

# ====================================================================
# SCALE 1 -- Cell Wall  (~0.5 um)
# ====================================================================

("cell_wall", """
The primary cell wall (paries cellularis primarius) of angiosperm mesophyll
cells is a hydrated, viscoelastic, fibre-reinforced composite. Cellulose
microfibrils (CMF, crystalline core) are synthesised by rosette cellulose
synthase complexes (CSC, CESA1/3/6 trimers in Arabidopsis) embedded in the
plasmalemma, extruding beta-(1->4)-glucan chains that self-assemble into crystalline
elementary fibrils of 18 glucan chains (cross-section ~3 nm x 5 nm) and
macrofibrils of 5-12 nm diameter. The degree of polymerisation (DP) is
6,000-13,000 glucose residues. CMF microfibril angle (MFA) relative to the
cell axis is 0deg-5deg in elongated cells. Cellulose mass fraction in the primary
wall is ~20-30%; hemicelluloses (xyloglucan dominant in eudicots, 20-25%),
pectins (homogalacturonan HG, rhamnogalacturonan RGI/RGII, 35-40%), and
structural glycoproteins (extensin, arabinogalactan-protein) comprise the
matrix. Xyloglucan binds directly to CMF surfaces via van der Waals contacts
and hydrogen bonds (Kd ~0.1 mM). Pectin Ca^2⁺-crosslinking (egg-box model,
HG degree of methylesterification <50%) determines wall stiffness modulus
E ~= 0.3-1.0 GPa. The middle lamella (lamella media), enriched in calcium
pectate, cements adjacent cells and is 0.02-0.20 um thick. Wall-associated
kinases (WAK1-5) sense pectin fragmentation during pathogen attack.
Plasmodesmata (PD) are lined by the desmotubule (compressed appressed ER,
~15 nm diam) passing through a plasma membrane collar, creating cytoplasmic
annuli (~20-30 nm effective aperture for macromolecular transport cutoff ~67 kDa).
The outer face of the wall facing the intercellular airspace is the surface
exterior layer that forms the visible boundary of the cell phenotype.
"""),

("cell_wall", """
Cell wall polysaccharide architecture at nanometre resolution, as resolved by
solid-state ¹^3C NMR spectroscopy and small-angle X-ray scattering (SAXS):
Cellulose I_beta is the dominant allomorph in higher plants, with a monoclinic
unit cell a=7.78 A, b=8.20 A, c=10.38 A, γ=96.5deg. The adjacent cellulose
chains pack in a parallel arrangement (all non-reducing ends facing the same
direction), forming intrachain O3...O5′ and O2...O6′ hydrogen bonds of ~2.5-2.8 A.
Interchain hydrogen bonds (O6...O3′, ~2.6 A) and van der Waals stacking of the
hydrophobic (110) face contribute to microfibril stability. Water occupies
the non-crystalline surface of CMFs (accessible via D₂O exchange, 30-40%
of cellulose hydroxyl groups). The overall wall thickness of 0.1-0.5 um
encompasses 50-250 lamellae of CMFs each ~10 nm apart in the transverse section.
The outer layer of the wall (outer face, lamella prima) was deposited earliest
and contains the highest pectin concentration. The inner layer (nearest
plasmalemma, lamella ultima) is freshest and contains the highest proportion
of newly synthesised xyloglucan. Wall expansion during cell elongation requires
expansin proteins (alpha-expansin, EXPA) disrupting hydrogen bonds between xyloglucan
and cellulose, yielding wall creep at pH 4.5-5.5 (acid growth hypothesis).
"""),

("cell_wall", """
The plasma membrane (plasmalemma) subtending the inner surface of the cell wall
is a Type I fluid mosaic bilayer composed of ~35 mol% phosphatidylcholine (PC),
~28% phosphatidylethanolamine (PE), ~20% phosphatidylserine (PS) and
phosphatidylinositol (PI), ~14% sterols (sitosterol, campesterol), and
~3% sphingolipids (glucocerebrosides, GIPC). Bilayer thickness is 7.5-8.0 nm
(leaflet 3.5-4.0 nm each). Membrane protein density: ~25,000 integral proteins
per um^2, lateral diffusion coefficient D_lat ~0.1-1.0 um^2 s⁻¹. The H⁺-ATPase
(P-type ATPase, AHA1-11 in Arabidopsis) is the primary electrogenic pump;
10-subunit C₁₀ proteolipid ring, stoichiometry 1 ATP per 3.3 H⁺, turnover
100-200 s⁻¹. K⁺ channels (AKT1, AKT2, KAT1) mediate K⁺ uptake and efflux.
Aquaporins (PIP1, PIP2 tetramers, ~30-A pore diameter) allow water flux
(osmotic permeability Pf ~100 um s⁻¹) essential for turgor regulation.
The cell wall-plasma membrane interface is the apoplast-symplast boundary.
"""),

# ====================================================================
# SCALE 2 -- Chloroplast  (~5 um)
# ====================================================================

("chloroplast", """
The chloroplast (plastid) of vascular plant mesophyll cells is a semiautonomous,
double-membrane-bounded organelle of cyanobacterial endosymbiotic origin
(primary endosymbiosis, ~1.5 Ga). The organelle's prolate-spheroid envelope
measures 3-8 um (long axis) x 1.5-3 um (short axis). The outer envelope
membrane (OEM, 30-38% protein) is permeable to metabolites <=10 kDa via the
non-selective outer envelope channel (OEP24, beta-barrel). The inner envelope
membrane (IEM, ~60% protein) is tightly regulated; the triose phosphate/phosphate
translocator (TPT) exports 3-phosphoglycerate (3-PGA) and dihydroxyacetone
phosphate (DHAP) in antiport with Pi (Km,DHAP ~0.5 mM). The intermembrane space
(IMS, ~6 nm) has enzymatic activity including adenylate kinase. The stroma
(pH 8.0 in light, 7.1 in dark) is a concentrated protein solution (~600 mg/mL):
RuBisCO hexadecamer (550 kDa, L8S8) accounts for 30-50% of stromal protein,
forming a paracrystalline array visible in TEM. The plastid-encoded RNA polymerase
(PEP) and nucleus-encoded RNA polymerase (NEP) both transcribe the ~154 kb
plastome (ptDNA) organised in 2-100 copies per organelle in nucleoid bodies
(~500 nm, stainable with DAPI). Translation occurs on 70S ribosomes (50S + 30S
subunits; streptomycin inhibitable) at a density of ~1,000 per um^3 of stroma.
The thylakoid membrane network (thylakoidale Membransystem) is highly folded
into stacked grana (5-25 discs each, mean 10, disc diameter 300-600 nm,
interthylakoid repeat distance 18-20 nm) interconnected by unstacked stromal
lamellae. Total thylakoid membrane area per chloroplast ~= 150-200 um^2.
"""),

("chloroplast", """
Chloroplast ultrastructure as characterised by cryo-electron tomography
(cryo-ET) and three-dimensional reconstruction reveals the following
architecture of the thylakoid system: Grana stacks (granum, pl. grana)
consist of appressed thylakoid membrane pairs held together by the
LHCII-LHCII stacking interaction (van der Waals + electrostatic) mediated
by the N-terminal stromal domain of LHCII trimers (LHCB1-3). The lumenal
space within a granum stack is narrowed to 4-6 nm (compressed by Lhcb
proteins), vs. 20-40 nm in unstacked stromal lamellae and margins.
Stromal lamellae connect adjacent grana at right angles and have a lumenal
width of 20-40 nm. The chloroplast lipid composition is unique: monogalactosyl
diacylglycerol (MGDG, ~50 mol%) is a non-bilayer lipid forming type-II
hexagonal phases; digalactosyl diacylglycerol (DGDG, ~30%) stabilises
the bilayer; sulfoquinovosyl diacylglycerol (SQDG, ~10%) and
phosphatidylglycerol (PG, ~10%). All four lipids carry predominantly
16:0, 18:1Δ9, 18:2Δ9,12, and 18:3Δ9,12,15 fatty acid tails. The thylakoid
membrane is only ~30% lipid by mass; the remaining 70% is protein.
Plastoglobules (osmiophilic droplets, 50-500 nm diam) are semicrystalline
lipid bodies attached to the outer thylakoid face, enriched in plastochromanol
and tocopherol (vitamin E precursors).
"""),

("chloroplast", """
Chloroplast import machinery: Newly synthesised nuclear-encoded chloroplast
proteins (~2,500-3,000 proteins) are imported post-translationally via the
TOC/TIC (Translocon at Outer/Inner envelope membrane of Chloroplast) complexes.
TOC75 (beta-barrel, OEP channel) + TOC34 + TOC159 form the TOC complex (~500 kDa);
TIC110 + TIC40 + TIC20 + TIC214 form the inner TIC complex. Import requires
a stromal targeting sequence (STP, ~30-60 aa, amphipathic helix), GTP
(TOC34/TOC159 GTPases), and ATP (stromal Hsp70/Hsp93/chaperonin-60 unfoldase).
Transit peptide cleavage by stromal processing peptidase (SPP, ~150 kDa Zn^2⁺
metalloprotease) releases mature protein. Chloroplast proteome import rate
~100 proteins min⁻¹ per organelle in high light. The division of chloroplasts
uses FtsZ1/FtsZ2 (prokaryotic tubulin homologues) forming a mid-plastid Z-ring
(50-nm-diameter protofilaments), MinD/MinE positioning system, and DRP5B
dynamin for membrane constriction. Chloroplast number per mesophyll cell doubles
during leaf expansion (1-2 d at 20degC, 400 umol m⁻^2 s⁻¹).
"""),

# ====================================================================
# SCALE 3 -- Thylakoid Membrane  (~10 nm)
# ====================================================================

("thylakoid", """
The thylakoid membrane harbours the photosynthetic electron transport chain
(PETC) embedded in a 7.5 nm lipid bilayer (40% MGDG, 30% DGDG, 10% SQDG,
10% PG, 10% mixed). The four major supramolecular complexes and their
crystallographic structures (PDB):

Photosystem II (PSII, PDB 3ARC, spinach): ~700 kDa, dimeric supercomplex,
~170 A x 110 A footprint. Core subunits D1 (PsbA, 32 kDa) + D2 (PsbD, 34 kDa)
bind the primary donor P680 (chlorophyll a pair, Mg-Mg 8.2 A, special pair at
centre-to-centre distance), pheophytin (PheoD1), and plastoquinone QA/QB.
Mn4CaO5 oxygen-evolving complex (OEC) at the lumenal face oxidises 2H₂O
-> O₂ + 4H⁺ + 4e⁻ (S-state cycle, Kok cycle, S0-S4). OEC Mn-Mn distances
2.7-3.3 A; Ca-Mn 3.3-3.5 A. Edeg(P680⁺/P680) = +1.25 V (strongest biological
oxidant). Tyrosine 161 (TyrZ, YZ) acts as the radical intermediate.
The LHCII major antenna complex (PDB 1RWT): trimer, 3 x 25 kDa (Lhcb1-3),
14 Chl a+b molecules + 4 carotenoids per monomer. Absorption cross-section
~100 A^2 per Chl. Forster resonance energy transfer (FRET) from Chl b
(λabs 650 nm) to Chl a (λabs 680 nm) in <100 fs.
"""),

("thylakoid", """
Cytochrome b6f complex (Cyt b6f, PDB 1Q90, Chlamydomonas): dimeric ~220 kDa,
cytochrome f (31 kDa, c-type haem, Edeg = +0.37 V, lumenal), cytochrome b6
(24 kDa, two b-haems: bp Edeg = −0.18 V, bn Edeg = −0.08 V), Rieske Fe-S protein
(20 kDa, [2Fe-2S] cluster Edeg = +0.30 V), subunit IV + 4 small subunits.
Cyt b6f is the rate-limiting step in PETC, turnover ~200-400 H⁺ s⁻¹.
The Q-cycle (Mitchell chemiosmotic loop) translocates 2H⁺/e⁻ across the
membrane. Plastocyanin (PC, 10.4 kDa, type I copper protein, Edeg = +0.37 V)
is the mobile electron carrier between Cyt b6f and PSI in the lumen.

Photosystem I (PSI, PDB 1JB0): ~540 kDa monomer, 12 protein subunits (PsaA-PsaF).
P700 (Chl a/a′ heterodimer, 8.9 A Mg-Mg) Edeg = +0.49 V; reduced by PC (lumenal).
Electron transfer: P700* -> A0 (Chl a, <5 ps) -> A1 (phylloquinone, 10-25 ps)
-> Fx ([4Fe-4S], ~200 ps) -> FA/FB ([4Fe-4S]/[4Fe-4S], ~600 ns) -> Fd (ferredoxin,
11 kDa, [2Fe-2S], Edeg = −0.42 V). FNR (Fd:NADP⁺ oxidoreductase, EC 1.18.1.2)
reduces NADP⁺ to NADPH. ΔG for full PETC (H₂O -> NADPH) ~= −220 kJ mol⁻¹.

ATP synthase (CF0-CF1, PDB 6FKF): 595 kDa, c14 proteolipid ring (CF0),
alpha3beta3γδε (CF1). Rotary mechanism: 14 c-subunits x 1 H⁺/c-subunit = 14 H⁺
per revolution = ~4.7 ATP synthesised (3 catalytic sites x 120deg each per
revolution x 14/3 = 4.67). ΔpH component ΔΨ ~120 mV + ΔpH ~3 units =
~300 mV total proton motive force (pmf) in steady-state light.
"""),

("thylakoid", """
State transition kinetics and lateral heterogeneity of thylakoid proteins.
PSII is concentrated in appressed grana membrane; PSI and ATP synthase are
restricted to non-appressed stroma lamellae (lateral segregation confirmed by
immuno-gold TEM and single-particle TIRF). Under state-1 conditions (low light),
LHCII trimers are predominantly PSII-associated. Under state-2 (high light /
reduced PQ pool), STN7 kinase (Arabidopsis) phosphorylates Thr3 on Lhcb1/2
(confirmed by phosphoproteomics MS/MS), reducing LHCII-PSII interaction and
causing LHCII migration to stroma lamellae for energy coupling to PSI.
Mg^2⁺ concentration in the thylakoid lumen oscillates between 1 mM (dark)
and 3-5 mM (light) due to H⁺/Mg^2⁺ exchange via the CLCe channel.
The plastoquinone (PQ) pool (4-8 PQ molecules per PSII; hydrophobic, diffuses
in membrane at D_lat ~5 x 10⁻⁹ cm^2 s⁻¹) equilibrates PSII and Cyt b6f.
Non-photochemical quenching (NPQ) up-regulates under excess light:
qE (energy-dependent quenching, PsbS sensor protein + zeaxanthin, fastest,
ΔpH-triggered), qT (state transition), qI (photoinhibition, D1 turnover ~30 min).
Thylakoid protein complex stoichiometry per granum disc: ~180 PSII, ~120 LHCII
trimers, ~80 PSI, ~40 Cyt b6f, ~30 ATP synthase molecules.
"""),

# ====================================================================
# SCALE 4 -- Chlorophyll a Molecule  (~1 nm)
# ====================================================================

("chlorophyll", """
Chlorophyll a (Chl a), systematic IUPAC name: [(2R,3S)-3-[(1Z)-1-[(3E,7R,11R)-
4,8,12-trimethyltridec-3-en-1-yl]ethylidene]... C55H72MgN4O5, CAS 479-61-8,
Mr = 893.49 g mol⁻¹. Crystallographic data (X-ray, P2₁/c, R = 0.039):
porphyrin macrocycle is a chlorin (one pyrrole ring B is reduced:
C17-C18 single bond, sp3 carbons). Porphyrin plane dimension: ~8.5 A x 8.5 A.
Mg coordination: square planar, Mg-N bond lengths 2.080-2.090 A (equatorial
N1, N2, N3, N4). Porphyrin ring bond distances: Calpha-Cbeta (pyrrole alpha to beta)
1.380-1.395 A; Calpha-Cm (meso carbon) 1.370-1.385 A; Cbeta-Cbeta 1.437-1.445 A;
N-Calpha 1.365-1.378 A. The macrocycle deviation from planarity (rms displacement
of heavy atoms) is <0.04 A in crystal but up to 0.3 A in the protein-bound state
(conformational distortion modulating photophysics). Phytol tail (C20H39OH
3,7,11,15-tetramethylhexadec-2-en-1-ol, isoprenoid polyterpenol) is connected
at C17^3 ester to the propionic acid side chain. Extended phytol conformation
spans 18-20 A. The molecule has u=3.2 D permanent dipole (in vacuo);
in protein it modulates to 1-6 D. UV-Vis (diethyl ether): λmax 430 nm (Soret,
ε = 1.11 x 10⁵ M⁻¹cm⁻¹), λmax 662 nm (Qy, ε = 8.6 x 10⁴ M⁻¹cm⁻¹).
Excited state: Qy S₁ lifetime 6 ns (in solution), 100-300 fs in antenna
(Forster/Dexter quenching); triplet yield 64% in O₂-free solvent.
"""),

("chlorophyll", """
Electronic structure and photophysics of chlorophyll a (DFT/TD-DFT analysis,
B3LYP/6-31G*): The HOMO is delocalised over the entire 18-pi-electron macrocycle
(Huckel aromatic, 4n+2 = 18 electrons). The LUMO is concentrated on pyrrole
rings A and C. HOMO-LUMO gap 1.87 eV (Qy transition at 662 nm = 1.87 eV),
Soret band = B-band transition at 430 nm (2.88 eV). Natural transition orbital
(NTO) analysis: Qy NTO pair is delocalised; B-band NTO pair has more localised
character on individual pyrrole rings. Mg^2⁺ is not formally oxidised; it acts
as a 2+ Lewis acid Lewis coordinating N lone pairs, contracting the macrocycle
and raising HOMO energy by ~0.4 eV relative to free-base porphyrin.
Inter-pigment Forster transfer rate: kFRET = 1/τD x (R₀/r)⁶, where τD = 6 ns,
R₀ = 6.5-8.0 nm (Chl a->Chl a), r = 0.9-1.5 nm (nearest-neighbour in LHCII).
Calculated kFRET ~10¹^2 s⁻¹, consistent with femtosecond pump-probe kinetics.
The Mg^2⁺ is coordinated by a fifth axial histidine ligand (protein-bound) in
PSII/PSI, shifting Qy by −5 to −10 nm (red shift). The excited state energy of
P680* in PSII is ~1.83 eV (Qy 680 nm), sufficient to oxidise water
(Edeg(O₂/H₂O) = +0.82 V at pH 7) given that Edeg(P680⁺/P680*) ~= −0.68 V.
"""),

("chlorophyll", """
Biosynthesis of chlorophyll a in higher plants proceeds via 17 enzymatic steps
from 5-aminolevulinic acid (ALA, formed by the C5 pathway: Glu + tRNA^Glu
-> Glu-1-semialdehyde via GluTR, EC 1.2.1.70, then GSAT aminotransferase).
Two ALA molecules condense to porphobilinogen (PBG) via ALA dehydratase (ALAD,
EC 4.2.1.24, Zn^2⁺-dependent, homooctamer). Four PBG -> hydroxymethylbilane
-> uroporphyrinogen III -> coproporphyrinogen III -> protoporphyrinogen IX
-> protoporphyrin IX (Proto IX, the Mg branch point). Mg chelatase
(ChlH + ChlI + ChlD + GUN4, ATP-dependent, 4 Mg^2⁺ per subunit of ChlI ATPase,
kcat ~0.25 Mg^2⁺ s⁻¹) inserts Mg^2⁺ into Proto IX to form Mg-protoporphyrin IX.
Phytol attachment: chlorophyll synthase (CHLG, EC 2.5.1.62) esterifies
chlorophyllide a with geranylgeranyl pyrophosphate (GGPP) then geranylgeranyl
reductase (CHLP) progressively reduces the C17-C18, C13-C14, C9-C10 double bonds
to yield the saturated phytol tail. Chl a:Chl b ratio in mesophyll cell is
~3:1; Chl b is formed from Chl a by chlorophyllide a oxygenase (CAO, EC 1.14.13.-).
"""),

# ====================================================================
# SCALE 5 -- Carbon Atom  (~154 pm)
# ====================================================================

("carbon_atom", """
Carbon (C, ⁶₁₂C, atomic number Z = 6): ground-state electron configuration
1s^2 2s^2 2p₁¹ 2p₂¹ (Hund's rule, maximised spin multiplicity triplet, S = 1).
The 1s orbital (nodeless, spherically symmetric, ψ₁s ∝ e^(−Zr/a₀)) has
a Bohr radius of a₁s = 0.529 / Z_eff A = 0.529 / 5.67 A ~= 0.093 A (Z_eff for 1s
via Slater screening σ₁s = 0.30 x 1 = 0.30; Z_eff = 5.70). The 2s and 2p
orbitals have one radial node (2s) or no radial node but one angular node (2p);
Z_eff(2s/2p) = 6 − σ₂ = 6 − 3.35 = 2.65 (Slater rules: σ from two 1s
electrons = 2 x 0.85 = 1.70; σ from the other 2s/2p = 0.35 x 3 = 1.05;
σ_total = 2.75, Z_eff = 3.25 more precisely by Clementi-Raimondi).
First ionisation energy: I₁ = 11.2603 eV (=1086.5 kJ mol⁻¹). Electron
affinity: EA = 1.2629 eV (121.9 kJ mol⁻¹). Electronegativity: χ (Pauling) = 2.55;
χ (Mulliken) = (I₁ + EA)/2 = 6.26 eV. Covalent radius (sp3): 77 pm; sp2: 73 pm;
sp: 69 pm. Van der Waals radius: 170 pm. C-C single bond: 154 pm; double 134 pm;
triple 120 pm. C-H: 109 pm. In sp2 hybridisation (as in porphyrin macrocycle):
three sp2 hybrid orbitals in a plane at 120deg (σ-bonding framework); one
unhybridised 2pz orbital perpendicular, participating in pi-bonding.
Nuclear: ¹^2C, nuclear spin I=0 (even Z, even N, ¹^3C: I=1/2, natural abundance
1.11%, key NMR nucleus). Binding energy of ¹^2C nucleus: 92.16 MeV (7.68 MeV/nucleon).
"""),

("carbon_atom", """
Quantum mechanical wavefunctions for the carbon atom in the independent-particle
model (Hartree-Fock, HF): The hydrogenic wavefunction ψ_nlm(r, θ, φ) = R_nl(r) x
Y_l^m(θ,φ), where R_nl is the radial wavefunction and Y_l^m the spherical harmonic.
For carbon the effective nuclear charge modifies R_nl. Hartree-Fock eigenvalues
(Koopmans' theorem): ε_1s = −305.9 eV; ε_2s = −16.59 eV; ε_2p = −11.33 eV.
Probability density |ψ|^2 of the 2pz orbital: dumbbell-shaped lobes of maximum
probability at r ~= 2a₀/Z_eff ~= 0.75 A from nucleus (for Z_eff ~= 2.6). Radial
expectation value ⟨r⟩_2p = (3/2)(a₀/Z_eff) x n^2 in hydrogenic approximation
= 1.09 A (for n=2, Z_eff=2.6). The de Broglie wavelength of a 2p electron:
λ_dB = h/p = h/(√2mE) = 1.226/√(E_keV) nm; E_2p ~= 11.3 eV = 0.0113 keV,
λ_dB = 1.226/√0.0113 = 11.5 nm >> atomic dimension (quantum wavelike). The
nucleus occupies radius ~3 fm (3 x 10⁻¹⁵ m) per the nuclear shell model,
comprising 6 protons and 6 neutrons bound by gluon-mediated strong force
(~=200 MeV fm⁻¹ at short range). Proton mass 938.3 MeV/c^2; neutron 939.6 MeV/c^2.
Nuclear binding energy 92.16 MeV ÷ 12 nucleons = 7.68 MeV per nucleon,
consistent with the semi-empirical Bethe-Weizsacker formula.
"""),

("carbon_atom", """
sp2 hybridisation of carbon and the porphyrin pi-system: In the flat porphyrin
macrocycle each meso and pyrrole alpha,beta carbon is sp2-hybridised. The three sp2
orbitals (formed from 2s + 2px + 2py) point at 120deg in the molecular plane.
The unhybridised 2pz orbital is perpendicular, contributing to the delocalized
pi-network. The 18 pi-electrons of chlorophyll's macrocycle satisfy Huckel's rule
(4n+2, n=4). The HOMO (pi) and LUMO (pi*) of the 18pi macrocycle define the
optical gap. In conjugated sp2 carbons the C-C bond length averages 1.40 A
(between single 1.54 A and double 1.34 A, resonance-delocalised). The resonance
energy of the porphyrin macrocycle is ~600 kJ mol⁻¹ relative to the
localised-bond reference. Carbon HOMO orbital lobes (2pz) have maximum amplitude
at +/-a₀/Z_eff ~= +/-0.20 nm from the nucleus perpendicular to the porphyrin plane.
The ionisation energy of a sp2 ring carbon in porphyrin is lowered ~0.5 eV
relative to isolated carbon due to through-space conjugation stabilisation.
Electron probability density at the nucleus: |ψ(0)|^2 = Z^3_eff/(pia₀^3) for s-orbitals;
for the 1s of C: |ψ₁s(0)|^2 = (Z_eff,1s)^3/(pia₀^3) ~= 2.7 x 10⁶ nm⁻^3, accounting
for the Fermi contact interaction measured in ¹^3C NMR hyperfine coupling.
"""),

]  # end CORPUS

# -- Quality check -------------------------------------------------------------

PROBE = {
    "plant_cell":   ["outer","wall","surface","membrane","cell","nucleus","vacuole",
                     "chloroplast","cytoplasm","organelle","mitochondria","plasmalemma"],
    "cell_wall":    ["wall","layer","cellulose","fibril","membrane","surface","outer",
                     "middle","primary","pectin","lamella","plasma"],
    "chloroplast":  ["organelle","membrane","stroma","thylakoid","disc","granum",
                     "stack","envelope","lumen","grana","lamella"],
    "thylakoid":    ["membrane","layer","protein","complex","surface","electron",
                     "photosystem","chlorophyll","bilayer","antenna","lumen"],
    "chlorophyll":  ["molecule","ring","atom","bond","carbon","nitrogen","magnesium",
                     "oxygen","porphyrin","macrocycle","phytol","pyrrole"],
    "carbon_atom":  ["atom","nucleus","electron","orbital","quantum","bond",
                     "proton","neutron","wavefunction","shell","hybridisation","pi"],
}

def fz(v):
    if v is None: return "?"
    if v < 1e-12: return f"{v*1e15:.1f}fm"
    if v < 1e-9:  return f"{v*1e12:.0f}pm"
    if v < 1e-6:  return f"{v*1e9:.2f}nm"
    if v < 1e-3:  return f"{v*1e6:.1f}um"
    if v < 1:     return f"{v*1e3:.1f}mm"
    return f"{v:.4f}m"

def quality_check(h, p):
    print(f"\n{'ENTITY':<16} {'CAL':3} {'N':>2}  {'Z_MIN':>10} {'Z_MAX':>10}  ACTIVATED LAYERS")
    print("-"*85)
    for entity, scale_m in SCALES:
        labels = PROBE[entity]
        n = len(labels)
        scores = {l: 1.0 - i/n for i,l in enumerate(labels)}
        body = {"entities":[{
            "entity_id": entity, "phenotype": entity,
            "active_labels": labels, "label_scores": scores,
            "x_frac": 0.5, "y_frac": 0.5, "size_est": scale_m,
        }]}
        r = post(h, p, "/overlay/layers", body)
        reps = r.get("layers", [])
        if not reps: print(f"  {entity:<16}: NO REPORT"); continue
        rep   = reps[0]
        sp    = rep["scale_profile"]
        lyrs  = rep["layers"]
        if not lyrs: print(f"  {entity:<16}: 0 layers"); continue
        zvs   = [l["position"]["z_est_m"] for l in lyrs]
        brief = "  ".join(f"d{l['depth']}:{l['label'][:14]}"
                          for l in sorted(lyrs, key=lambda x: x["depth"])[:4])
        print(f"  {entity:<16} {'T' if sp['calibrated'] else 'F':3} {len(lyrs):>2}"
              f"  {fz(min(zvs)):>10} {fz(max(zvs)):>10}  {brief}")

# -- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",  default="192.168.1.84")
    ap.add_argument("--port",  type=int, default=8090)
    ap.add_argument("--passes",type=int, default=6)
    args = ap.parse_args()
    h, p = args.host, args.port

    try:
        with urllib.request.urlopen(f"http://{h}:{p}/neuro/snapshot", timeout=10) as r:
            snap = json.loads(r.read())
    except Exception as e:
        print(f"ERROR: cannot reach {h}:{p} -- {e}"); sys.exit(1)
    print(f"Connected  pool_labels={len(snap.get('active_labels',[]))}")

    print("\n=== Calibrations ===")
    for entity, scale_m in SCALES:
        ok = calibrate(h, p, entity, scale_m)
        print(f"  {'OK' if ok else '!!'} {entity:<16} {fz(scale_m)}")

    total = 0
    for pass_idx in range(args.passes):
        print(f"\n=== Pass {pass_idx+1}/{args.passes} ({len(CORPUS)} docs) ===")
        for i, (entity, text) in enumerate(CORPUS):
            n = train(h, p, text)
            total += n
            bar = "#" * min(40, n // 15)
            print(f"  [{i+1:02d}/{len(CORPUS)}] {entity:<16} {bar:<40} +{n}")
            time.sleep(0.12)

    print(f"\nTotal labels: {total:,}  ({total // len(CORPUS)} avg/doc)")
    print("\n=== Quality check ===")
    quality_check(h, p)
    print("\nTraining complete. Open cell_viewer.html")

if __name__ == "__main__":
    main()
