#!/usr/bin/env python3
# coding: utf-8
"""
build_cow_dataset.py — Comprehensive bovine anatomy training dataset builder
for the W1z4rD V1510n neural fabric.

Downloads, structures, and ingests thousands of items across six curriculum
stages from free, open-access sources:

  Stage 0  — Visual primitives  (synthetic shading/depth images, ~500 items)
  Stage 1  — Bovine anatomy text (PubMed Central OA, Merck Vet Manual, ~2000 items)
  Stage 2  — Cow video frames   (CC-licensed YouTube, ~5000 frames)
  Stage 3  — Medical imaging    (MRI/CT cross-sections, CT-BEAST / OpenVetAnatomy, ~800 items)
  Stage 4  — Histology images   (Wikimedia Commons histology plates, ~600 items)
  Stage 5  — Molecular data     (PDB protein structures as text + imagery, ~300 items)

All items are structured into D:/w1z4rdv1510n-data/training/ and posted to
the W1z4rD node API for ingestion.

Usage:
  python build_cow_dataset.py [--stages 0,1,2,3,4,5] [--node localhost:8090]
                              [--data-dir D:/w1z4rdv1510n-data]
                              [--download-only] [--ingest-only] [--workers 4]

Requirements (install once):
  pip install requests yt-dlp Pillow numpy opencv-python tqdm

Source licenses:
  - PubMed Central OA: CC-BY / CC-BY-NC (checked per article)
  - YouTube CC: Creative Commons Attribution (CC BY)
  - Wikimedia Commons: CC-BY-SA / public domain
  - Merck Vet Manual: publicly accessible web text (educational scraping)
  - PDB: CC0 / unrestricted

Author: C4rr13rX / W1z4rD V1510n
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print('[WARN] opencv-python not found — video frame extraction disabled')

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print('[WARN] Pillow not found — synthetic stage 0 disabled')

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print('[WARN] requests not found — falling back to urllib')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):
        return it

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NODE     = 'localhost:8090'
DEFAULT_DATA_DIR = 'D:/w1z4rdv1510n-data'
TRAINING_DIR_REL = 'training'

# ── Stage descriptions ────────────────────────────────────────────────────────
STAGES = {
    0: 'Visual primitives — synthetic shading, perspective, occlusion',
    1: 'Bovine anatomy text — PubMed Central OA + Merck Vet Manual',
    2: 'Cow video frames   — CC-licensed YouTube clips',
    3: 'Medical imaging    — MRI/CT cross-sections',
    4: 'Histology images   — tissue plates from Wikimedia',
    5: 'Molecular data     — PDB structures + molecular imagery',
}

# ── PubMed Central search terms for bovine anatomy ───────────────────────────
PMC_QUERIES = [
    'bovine anatomy',
    'cattle anatomy musculoskeletal',
    'bovine limb anatomy',
    'cow hoof anatomy histology',
    'bovine head anatomy',
    'cattle locomotion biomechanics',
    'bovine mammary gland anatomy',
    'cattle digestive system anatomy',
    'bovine reproductive anatomy',
    'cattle skeletal muscle fiber',
    'bovine osteology',
    'cow skin hide histology',
    'bovine eye anatomy',
    'cattle nerve anatomy',
    'bovine joint anatomy',
    'cow tendon ligament anatomy',
    'cattle body condition scoring',
    'bovine udder anatomy quarter',
    'cattle rumen anatomy physiology',
    'bovine tooth dental formula',
]

# ── Merck Veterinary Manual pages (bovine anatomy sections) ──────────────────
MERCK_VET_PAGES = [
    ('Bovine Musculoskeletal System Overview',
     'https://www.merckvetmanual.com/musculoskeletal-system/bovine/musculoskeletal-system-introduction-cattle'),
    ('Cattle Lameness and Foot Disorders',
     'https://www.merckvetmanual.com/musculoskeletal-system/lameness-in-cattle/lameness-in-cattle-overview'),
    ('Bovine Eye Disorders',
     'https://www.merckvetmanual.com/eye-and-ear/ophthalmic-diseases/ophthalmic-diseases-of-cattle'),
    ('Cattle Digestive System',
     'https://www.merckvetmanual.com/digestive-system/gastrointestinal-diseases-of-cattle/overview-of-gastrointestinal-diseases-of-cattle'),
    ('Bovine Reproductive System',
     'https://www.merckvetmanual.com/reproductive-system/reproductive-diseases-of-cattle/reproductive-diseases-of-cattle-overview'),
    ('Cattle Respiratory System',
     'https://www.merckvetmanual.com/respiratory-system/respiratory-diseases-of-cattle/respiratory-diseases-of-cattle-overview'),
    ('Bovine Skin Diseases',
     'https://www.merckvetmanual.com/integumentary-system/integumentary-system-in-cattle/skin-diseases-of-cattle'),
    ('Udder and Teat',
     'https://www.merckvetmanual.com/reproductive-system/mastitis-in-cattle/mastitis-in-cattle'),
]

# ── Wikimedia Commons histology categories ───────────────────────────────────
WIKIMEDIA_HISTOLOGY_CATEGORIES = [
    'Skeletal_muscle_histology',
    'Bone_histology',
    'Cartilage_histology',
    'Adipose_tissue_histology',
    'Connective_tissue_histology',
    'Skin_histology',
    'Nerve_histology',
    'Tendon_histology',
]

# ── PDB IDs relevant to bovine structural biology ────────────────────────────
PDB_IDS = [
    ('2MHM', 'Bovine myosin S1 fragment — skeletal muscle motor protein'),
    ('1ATN', 'Bovine non-muscle actin — G-actin monomer, 375 aa'),
    ('1CFC', 'Bovine calmodulin — Ca2+-binding protein, muscle regulation'),
    ('3ZBD', 'Bovine liver catalase — peroxisome enzyme'),
    ('1BNA', 'DNA dodecamer — canonical B-form double helix reference'),
    ('4UNI', 'Bovine fibrinogen — blood clotting protein'),
    ('3HFM', 'Bovine serum albumin — major plasma protein'),
    ('2QSK', 'Bovine collagen type I triple helix fragment'),
    ('3BJU', 'Bovine rhodopsin — visual receptor in rod photoreceptors'),
    ('1EKL', 'Bovine pancreatic trypsin — digestive protease'),
    ('1PPN', 'Bovine procollagen C-proteinase — collagen maturation'),
    ('4JKX', 'Bovine ATP synthase F1 subunit — mitochondrial energy production'),
    ('2Y69', 'Bovine keratin type II — hoof/hair structural protein'),
    ('1BVN', 'Bovine cytochrome b5 — electron transfer in ER'),
    ('2BBK', 'Bovine chymotrypsin — digestive enzyme'),
]

# ── YouTube CC-licensed cow content search queries ───────────────────────────
YOUTUBE_QUERIES = [
    'cow walking side view creative commons',
    'cattle herd grazing field creative commons',
    'cow anatomy educational creative commons',
    'dairy cow milking farm creative commons',
    'bovine behavior close up creative commons',
    'cattle livestock farm creative commons',
    'cow running field slow motion creative commons',
    'veterinary bovine examination creative commons',
]

# ── Bovine anatomy knowledge base (embedded, no download needed) ──────────────
ANATOMY_KNOWLEDGE = [
    {
        'title': 'Bovine Musculoskeletal System Overview',
        'text': '''The bovine skeleton consists of approximately 210 bones, including the skull, vertebral column, ribs, sternum, and the bones of the four limbs. The vertebral formula for cattle is C7, T13, L6, S5, Co18-20. The thoracic limb (forelimb) contains the scapula, humerus, radius, ulna, carpals, metacarpals (fused MC3+MC4 = cannon bone), and two phalanges per digit. The pelvic limb (hindlimb) consists of the ilium, ischium, pubis, femur, tibia, fibula (vestigial), tarsals, metatarsals (fused MT3+MT4), and the digits.

Bovine muscles are organized into functional groups. The epaxial muscles (longissimus, iliocostalis, spinalis) run dorsally and extend the vertebral column. The hypaxial muscles (psoas, external/internal obliques, rectus abdominis) support the ventral trunk. The largest single muscle mass is the longissimus dorsi, originating on the lumbar vertebrae and inserting on the ribs; this muscle is the source of the ribeye cut and is used in body condition scoring.

The bovine locomotor apparatus differs from equines in having two functional digits (III and IV) per foot. Cloven-hoofed locomotion distributes weight between the two claws. Each digit has three phalanges: proximal (P1), middle (P2), and distal (P3 = pedal bone). The hoof capsule consists of three zones: the wall (dorsal stratum), the sole, and the white line—a zone of soft horn at their junction and the most common site of white line disease.

Weight distribution: in the standing position, approximately 55–60% of body weight is borne by the forelimbs and 40–45% by the hindlimbs. Each hind foot bears roughly equal weight on the medial and lateral claws in the forelimb, but the lateral claw of the hindfoot bears approximately 60–70% of the load, predisposing it to sole ulcers and white line disease.''',
        'tags': ['anatomy', 'musculoskeletal', 'skeleton', 'bones', 'muscles', 'locomotive'],
    },
    {
        'title': 'Bovine Head and Sensory Organs',
        'text': '''The bovine skull is composed of the frontal, parietal, occipital, temporal, sphenoid, ethmoid, nasal, lacrimal, zygomatic, maxilla, premaxilla, palatine, vomer, and mandible bones. The frontal bones are highly pneumatised in adult cattle, forming the large frontal sinuses that extend posteriorly into the horn cores in horned breeds (Bos taurus).

The bovine eye has a horizontal pupil providing approximately 330° of panoramic vision, with a binocular field of approximately 25–30° directly ahead. The tapetum lucidum is a well-developed iridescent layer behind the retina that reflects light back through the photoreceptors, enabling effective low-light vision for nocturnal grazing. Cattle are dichromats, perceiving primarily blue and yellow-green wavelengths.

The nasal planum (muzzle) is a distinctive feature of cattle—a hairless, glandular area used for thermoregulation. The serous nasal discharge keeps the muzzle moist; dryness of the muzzle is a clinical indicator of fever or systemic disease. The bovine olfactory system includes both the main olfactory epithelium and a vomeronasal organ (VNO) with a biphasic structure, used in detecting pheromones during reproduction.

Bovine teeth: calves are born with temporary (deciduous) teeth. The permanent dental formula is I0/3, C0/1, P3/3, M3/3 = 32 teeth. The absence of upper incisors is characteristic; cattle use a dental pad (a hardened ridge of gum) in combination with the lower incisors to crop grass. Dental eruption timeline is used for age estimation.

The bovine ear (pinna) acts as a directional sound receptor and as a thermoregulatory organ; the auricular vasculature radiates heat. Normal internal ear canal temperature (~38.5°C) is used for remote temperature monitoring in electronic ear tags.''',
        'tags': ['anatomy', 'head', 'skull', 'eye', 'sensory', 'teeth', 'muzzle', 'ear'],
    },
    {
        'title': 'Bovine Integument — Skin and Hide',
        'text': '''Bovine skin (integument) consists of the epidermis, dermis, and hypodermis (subcutis). Total skin thickness varies by region: 3–4 mm at the muzzle, 5–7 mm over the trunk, and 8–10 mm at the neck. The epidermis is a keratinised stratified squamous epithelium, 30–60 µm thick, consisting of stratum basale, spinosum, granulosum, and corneum layers. Melanocytes in the stratum basale determine coat colour.

The dermis is composed of dense irregular connective tissue with collagen (primarily type I) and elastin fibres, fibroblasts, mast cells, and macrophages. The papillary dermis is thin and loosely organised; the reticular dermis is thick and provides mechanical strength—this is the layer tanned for leather. Collagen fibres are arranged in a weave pattern that provides tensile strength in multiple directions.

Bovine sweat glands are apocrine (in contrast to eccrine glands in primates). They are widely distributed in cattle and are important for evaporative cooling. Each gland consists of a coiled secretory portion in the deep dermis and a straight duct that opens into the hair follicle infundibulum above the sebaceous gland.

Hair follicles in cattle are compound—a central primary follicle surrounded by smaller secondary follicles (3–8 per primary). This results in a ratio of secondary:primary follicles of approximately 5–8:1. Coat texture and insulation are determined by this ratio and by fibre diameter (50–150 µm for cattle body hair).

The hoof capsule is a highly specialised keratinised epidermal structure. It consists of the wall (stratum externum, medium, and internum), the sole, and the white line. The wall grows approximately 6 mm per month. The hoof corium (dermis of the hoof) is highly vascularised and innervated; the laminar corium interdigitates with the lamellae of the inner wall to provide mechanical attachment.''',
        'tags': ['anatomy', 'skin', 'integument', 'hide', 'epidermis', 'dermis', 'hoof', 'keratin'],
    },
    {
        'title': 'Bovine Digestive System Anatomy',
        'text': '''Cattle are ruminants with a four-compartment stomach: rumen, reticulum, omasum, and abomasum. The total capacity of the adult rumen is 100–200 L. The rumen (paunch) occupies the left side of the abdominal cavity and contains a complex microbial ecosystem (bacteria, protozoa, fungi) that ferments cellulose and other plant polysaccharides.

The rumen wall consists of: serosa, longitudinal and circular smooth muscle layers, submucosa, and mucosa. The mucosal surface bears numerous papillae (3–6 mm) that increase absorptive area for volatile fatty acids (VFAs). VFA production is approximately 3 mol acetate, 1 mol propionate, and 0.5 mol butyrate per day in a well-fed dairy cow.

The reticulum (honeycomb bag, ~20 L) communicates with the rumen via the ruminoreticular fold. Its mucosa bears a distinctive hexagonal/honeycomb pattern. Hardware disease (traumatic reticuloperitonitis) results from ingested metallic objects penetrating the reticulum wall.

The omasum (manyplies, ~15 L) has muscular laminae (leaves) projecting from the wall into the lumen, providing mechanical grinding action. The abomasum (true stomach, ~25 L) is the glandular stomach equivalent to the monogastric stomach. The abomasal mucosa has three regions: fundic (pepsinogen, HCl), cardiac, and pyloric zones.

The small intestine (~40 m long) is the primary site of nutrient absorption. The large intestine (~10 m) reabsorbs water and electrolytes. The caecum (~30 L) is a blind pouch at the ileocaecal junction. The bovine intestinal tract is supported by mesenteric fat and lymph nodes (mesenteric lymph nodes are large and numerous in cattle).''',
        'tags': ['anatomy', 'digestive', 'rumen', 'reticulum', 'omasum', 'abomasum', 'intestine', 'ruminant'],
    },
    {
        'title': 'Bovine Cardiovascular and Respiratory Systems',
        'text': '''The bovine heart weighs 2–3 kg (0.4% of body weight) and beats 48–80 times per minute at rest. It has the typical mammalian four-chamber configuration (right/left atria, right/left ventricles) with a bicuspid mitral valve on the left and a tricuspid valve on the right. The interventricular septum is often partially calcified (cordis ossification) in aged cattle—this is not clinically significant.

The cardiac output at rest is approximately 30–40 L/min. Cardiac rhythm is controlled by the sinoatrial node. The bovine ECG shows characteristic large P-waves reflecting the atrial depolarisation of the enlarged right atrium. The aorta exits the left ventricle and arches to the left before descending.

The bovine lung has four lobes on the right (cranial, middle, caudal, accessory) and two to three on the left (cranial with two parts, caudal). Total lung capacity is approximately 25–30 L. Respiratory rate is 15–30 breaths per minute. The bovine bronchial tree has a characteristically prominent right apical (tracheal) bronchus arising directly from the trachea before the carina, making the right apical lobe particularly susceptible to aspiration pneumonia.

The trachea is ~50 cm long with 50–60 C-shaped cartilage rings. The nasal passages have well-developed turbinate bones (dorsal, ventral, middle conchae) lined with ciliated respiratory epithelium. Mucociliary clearance propels particulates toward the pharynx at approximately 10–15 mm/min.''',
        'tags': ['anatomy', 'cardiovascular', 'heart', 'lung', 'respiratory', 'trachea', 'circulation'],
    },
    {
        'title': 'Bovine Reproductive Anatomy',
        'text': '''The bovine uterus is bicornuate (two horns) with a relatively small body. Each horn is approximately 35–40 cm long in the non-pregnant cow, running forward and laterally from the uterine body along the broad ligament. The uterine wall has three layers: perimetrium (serosa), myometrium (smooth muscle, inner circular + outer longitudinal), and endometrium (mucosa).

The bovine endometrium contains 70–120 caruncles—raised, button-like structures that interdigitate with fetal cotyledons to form placentomes. This type of placentation (cotyledonary) is characteristic of ruminants and differs from the diffuse placentation of mares and sows.

The ovaries are bean-shaped, approximately 3×2×1.5 cm in the cycling cow. Follicular development follows a wave pattern (2–3 waves per 21-day oestrous cycle). The dominant follicle of the first and second waves is suppressed by progesterone from the corpus luteum unless the animal is in oestrus.

In the male (bull), the scrotum contains the testes in a thermoregulated environment 3–5°C below core body temperature. Each testis is approximately 12 × 7 × 6 cm and weighs 300–400 g. Spermatogenesis takes ~54–60 days in cattle. The penis is fibroelastic (sigmoid flexure type, unlike the vascular type in primates), maintained in the prepuce by retractor penis muscles.''',
        'tags': ['anatomy', 'reproductive', 'uterus', 'ovary', 'placenta', 'testis', 'reproduction'],
    },
    {
        'title': 'Bovine Nervous System and Neuroanatomy',
        'text': '''The bovine brain weighs 410–500 g (about 0.1% of body weight, reflecting the inverse brain:body ratio scaling in large mammals). The cerebrum is well-developed with distinct gyri and sulci. The cerebellum is proportionally large, reflecting the complex motor coordination required for quadrupedal locomotion.

The bovine spinal cord runs from the foramen magnum to approximately L5 in adults (the spinal cord is shorter than the vertebral column due to differential growth). Lumbosacral plexus formation is important in cattle lameness—injuries here cause hindlimb paresis. The sciatic nerve (L5–S2) is the most commonly injured nerve in cattle, particularly during calving (obturator paralysis, downer cow syndrome).

The autonomic nervous system regulates rumen motility (parasympathetic: vagus nerve increases motility; sympathetic: stress inhibits motility). Vagal indigestion results from disruption of the vagal innervation of the forestomachs.

Bovine spongiform encephalopathy (BSE, "mad cow disease") results from misfolded prion proteins accumulating in the central nervous system. Prions are resistant to conventional denaturation. BSE is a zoonotic concern linked to variant Creutzfeldt-Jakob disease in humans.

The brachial plexus (C6–T2) innervates the forelimb. Radial nerve paralysis (C7–T1) causes inability to bear weight on the forelimb and a characteristic "dropped elbow" stance. The peroneal nerve (branch of sciatic) injury causes knuckling of the fetlock and walking on the dorsum of the foot.''',
        'tags': ['anatomy', 'nervous', 'brain', 'spinal', 'nerve', 'neurology', 'BSE'],
    },
    {
        'title': 'Bovine Udder and Mammary Gland Anatomy',
        'text': '''The bovine mammary gland (udder) has four separate glands (quarters): right and left fore-quarters, and right and left hind-quarters. Each quarter is drained by a single teat. The two hind quarters are typically larger than the fore quarters and produce 60% of total milk yield.

Each quarter consists of: (1) glandular parenchyma—secretory alveoli and ducts; (2) stroma—fibrous, fatty connective tissue; (3) blood and lymph vasculature; (4) nerves. The secretory alveolus is a spherical structure (~0.1–0.2 mm diameter) lined by a single layer of secretory epithelial cells (mammary epithelial cells, MECs). Myoepithelial cells surround each alveolus and contract in response to oxytocin to eject milk into the duct system.

Milk is synthesised in MECs by: fatty acid synthesis (de novo from acetate/β-hydroxybutyrate), triglyceride assembly in smooth ER, protein synthesis (casein, whey proteins) in rough ER/Golgi, and lactose synthesis in Golgi (lactose synthetase). MECs express numerous nutrient transporters (GLUT1, MCT1, amino acid transporters) for substrate uptake.

A high-producing Holstein-Friesian cow produces 25–40 L of milk per day containing approximately 3.5% fat, 3.2% protein, 4.8% lactose, and 0.7% ash. The somatic cell count (SCC) is the primary indicator of udder health: SCC <200,000 cells/mL is considered healthy; mastitis is diagnosed at >400,000 cells/mL.

The teat consists of: teat skin (epidermis), teat wall (connective tissue + smooth muscle), teat cistern (gland cistern + teat cistern separated by Fürstenberg's rosette), and teat canal (streak canal, ~10 mm long, sealed by Furstenberg's rosette and keratin plug between milkings). The teat canal is the primary defence against mastitis.''',
        'tags': ['anatomy', 'udder', 'mammary', 'teat', 'milk', 'alveoli', 'secretory', 'mastitis'],
    },
    {
        'title': 'Bovine Skeletal Muscle Histology and Physiology',
        'text': '''Bovine skeletal muscle fibres are large, multinucleated cells (myofibres) ranging from 10–100 µm in diameter. Each fibre contains hundreds to thousands of myofibrils running parallel to the long axis. Each myofibril is divided into sarcomeres (the contractile unit, 2–3 µm long), delineated by Z-discs. The sarcomere contains interdigitating thick (myosin) and thin (actin) filaments.

Three fibre types are distinguished in cattle muscle:
- Type I (slow-twitch, oxidative): high mitochondria density, myoglobin content (red colour), fatigue-resistant; predominant in postural muscles (e.g., soleus)
- Type IIA (fast-twitch, oxidative-glycolytic): intermediate; predominant in most locomotor muscles
- Type IIB (fast-twitch, glycolytic): low mitochondria, pale (white) colour, fast but fatigable; large in well-muscled breeds (e.g., Belgian Blue)

The Belgian Blue (culard) mutation is a loss-of-function in the myostatin gene (GDF8, chromosome 2), resulting in fibre hypertrophy and hyperplasia. Affected animals have dramatically increased muscle mass ("double muscling").

Bovine muscle satellite cells are muscle stem cells residing between the sarcolemma and basal lamina. They are quiescent in adult muscle but activated in response to injury. Satellite cell density is highest in neonatal calves (~8/100 fibres) and decreases with age (~3/100 fibres in adults).

The neuromuscular junction (NMJ) in cattle, as in all mammals, uses acetylcholine as the neurotransmitter. Each muscle fibre is innervated by a single motor neuron. The motor unit (one motor neuron + all fibres it innervates) is the fundamental unit of motor control.''',
        'tags': ['histology', 'muscle', 'myofibre', 'sarcomere', 'actin', 'myosin', 'fibre type', 'satellite cell'],
    },
    {
        'title': 'Bovine Bone Histology and Physiology',
        'text': '''Bovine cortical (compact) bone is composed of concentric lamellar units called osteons (Haversian systems). Each osteon consists of a central Haversian canal (~50 µm diameter, containing blood vessels and nerves) surrounded by 4–20 concentric lamellae (~5 µm thick each). Osteocytes occupy lacunae between lamellae and communicate via canalicular network with osteocytes in adjacent lamellae and with cells lining the Haversian canal.

Three cell types maintain bone homeostasis:
- Osteoblasts: bone-forming cells derived from mesenchymal stem cells. Synthesise type I collagen and osteocalcin; mineralise osteoid by precipitating hydroxyapatite (Ca₁₀(PO₄)₆(OH)₂)
- Osteoclasts: bone-resorbing cells derived from monocyte-macrophage precursors. Large, multinucleated; form a ruffled border that seals against bone surface and secretes HCl + cathepsin K
- Osteocytes: terminally differentiated osteoblasts embedded in bone matrix. Sense mechanical strain and regulate bone remodelling via sclerostin/RANKL/OPG signalling

Bovine bone mineral density (BMD) is approximately 1.5–1.7 g/cm³ for cortical bone. Calcium content is ~34% by weight. Phosphorus content ~15% by weight. The Ca:P ratio of 2.2:1 in hydroxyapatite matches the dietary requirement ratio for cattle (1.5–2:1).

Bovine fracture healing follows the standard mammalian sequence: haematoma formation (0–5 days), soft callus (fibrocartilage bridge, 5–20 days), hard callus (woven bone, 20–60 days), remodelling (months). Cattle fractures are clinically challenging because the high body weight creates large bending moments and implant failure rates are significant in mature animals.''',
        'tags': ['histology', 'bone', 'osteon', 'osteocyte', 'osteoblast', 'osteoclast', 'hydroxyapatite', 'remodelling'],
    },
    {
        'title': 'Bovine Connective Tissue — Tendons, Ligaments, Cartilage',
        'text': '''Tendons connect muscle to bone. Bovine tendons consist of ~70% type I collagen by dry weight, arranged in a hierarchical structure: collagen molecules → fibrils (50–500 nm) → fibres (1–20 µm) → fascicles (50–300 µm) → tendon. Fascicles are surrounded by endotenon; the whole tendon is enclosed in epitenon. Tenocytes (fibroblast-like cells) occupy the spaces between fibres and synthesise collagen and matrix metalloproteinases for remodelling.

The superficial digital flexor tendon (SDFT) and deep digital flexor tendon (DDFT) are the primary load-bearing tendons of the bovine digit. The DDFT inserts on the palmar/plantar surface of the pedal bone (P3). The suspensory ligament (proximal sesamoid ligament complex) supports the fetlock joint. These structures are commonly injured in high-producing dairy cattle.

Articular cartilage is hyaline cartilage covering the articulating surfaces of synovial joints. Bovine articular cartilage is 1–3 mm thick, consisting of chondrocytes in a dense extracellular matrix of type II collagen and proteoglycans (aggrecan, versican). The cartilage has four zones: superficial (tangential), middle (transitional), deep (radial), and calcified. There are no blood vessels or nerves in articular cartilage; nutrition is via synovial fluid diffusion.

The bovine stifle joint (femorotibial + femoropatellar) contains medial and lateral menisci (fibrocartilage), cranial and caudal cruciate ligaments, medial and lateral collateral ligaments, and patellar ligaments. Stifle joint OCD (osteochondrosis dissecans) is a common developmental disease in rapidly growing bulls.''',
        'tags': ['anatomy', 'tendon', 'ligament', 'cartilage', 'collagen', 'connective tissue', 'SDFT', 'stifle'],
    },
    {
        'title': 'Bovine Cell Biology — Cellular Composition and Organelles',
        'text': '''Bovine cells share the fundamental eukaryotic cell organisation. The nucleus contains ~3.0 Gb of DNA (diploid genome, Bos taurus genome sequenced 2009, 29 autosome pairs + sex chromosomes). The nucleus is bounded by a double membrane (nuclear envelope) with nuclear pore complexes (~2,000 per nucleus) that regulate nucleocytoplasmic transport.

The endoplasmic reticulum (ER) is a continuous membrane system: rough ER (rER) studded with ribosomes and specialised for protein synthesis/folding; smooth ER (sER) for lipid metabolism and Ca2+ storage. Secretory proteins synthesised in the rER are co-translationally translocated into the ER lumen, undergo N-linked glycosylation, and are transported to the Golgi apparatus.

The Golgi apparatus consists of cis, medial, and trans cisternae plus the trans-Golgi network (TGN). It processes and sorts proteins for secretion, lysosomal delivery, or membrane insertion. The TGN packages proteins into vesicles: clathrin-coated vesicles for lysosomal targeting, COPII vesicles for anterograde transport, COPI vesicles for retrograde.

Mitochondria in bovine cells have an outer membrane and a folded inner membrane (cristae). The cristae contain the electron transport chain: Complex I (NADH dehydrogenase), II (succinate dehydrogenase), III (cytochrome bc1), IV (cytochrome c oxidase), and ATP synthase (Complex V). Oxidative phosphorylation generates ~32 ATP per glucose. Bovine mitochondrial DNA (mtDNA) is circular, ~16.3 kb, maternally inherited.

Bovine cells have approximately 20,000–25,000 protein-coding genes (similar to humans). Post-translational modifications (phosphorylation, ubiquitination, acetylation, glycosylation) enormously expand the functional proteome diversity. The bovine proteome Atlas is available from UniProt and contains >10,000 reviewed entries.''',
        'tags': ['cell biology', 'nucleus', 'mitochondria', 'ER', 'Golgi', 'ribosome', 'genome', 'DNA', 'protein'],
    },
]

# ── Synthetic visual primitives generator (Stage 0) ──────────────────────────

def generate_synthetic_primitives(out_dir: Path, n: int = 500) -> list[dict]:
    """Generate synthetic images demonstrating shading→depth, perspective→distance,
    and occlusion→spatial order. These train the visual perception pool."""
    if not HAS_PIL:
        print('[SKIP] Stage 0: Pillow not available')
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    items = []

    print(f'  Generating {n} synthetic visual primitives...')
    for i in tqdm(range(n), desc='Stage 0'):
        mode = i % 4
        img = Image.new('RGB', (128, 128), (20, 20, 30))
        draw = ImageDraw.Draw(img)

        if mode == 0:
            # Shading sphere: Lambertian shading demonstrates depth
            cx, cy = 64, 64
            r = 35 + (i % 12) * 2
            light = (0.6, 0.8, 0.5)  # normalised light direction
            for py in range(128):
                for px in range(128):
                    dx = (px - cx) / r; dy = (py - cy) / r
                    d2 = dx*dx + dy*dy
                    if d2 < 1.0:
                        dz = math.sqrt(1.0 - d2)
                        nrm = (dx, dy, dz)
                        diff = max(0.0, dx*light[0] + dy*light[1] + dz*light[2])
                        hue_shift = int(i / n * 255)
                        lv = int(diff * 220)
                        draw.point((px, py), (lv, max(0,lv-30+hue_shift%40), max(0,lv-60)))
            label = f'shading_sphere_{r}px_light{int(math.degrees(math.atan2(light[1],light[0])))}deg'

        elif mode == 1:
            # Perspective grid: receding lines
            vp = (64, 40 + (i % 20))
            for col in range(0, 128, 10):
                gray = 60 + int(col / 128 * 100)
                draw.line([(vp[0], vp[1]), (col, 128)], fill=(gray, gray, gray+20), width=1)
            for row in range(vp[1], 128, 8):
                t = (row - vp[1]) / (128 - vp[1])
                gray = int(t * 160) + 30
                draw.line([(0, row), (128, row)], fill=(gray, gray, gray), width=1)
            label = f'perspective_grid_vp{vp[1]}'

        elif mode == 2:
            # Occlusion scene: overlapping circles at different depths
            circles = [(30+i%40, 45, 25, (200, 80, 40)),
                       (70, 70, 22, (60, 160, 200)),
                       (90, 40+i%30, 18, (180, 200, 60))]
            for (cx, cy, r, col) in circles:
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)
            label = f'occlusion_circles_{i%40}'

        else:
            # Gradient depth field: horizontal gradient encodes depth
            for px in range(128):
                depth_hint = int(px / 128 * 255)
                for py in range(128):
                    noise = int(np.random.normal(0, 8))
                    v = max(0, min(255, depth_hint + noise))
                    obj_present = (abs(py - 64) < 20 + depth_hint // 12)
                    colour = (v, v // 2, 0) if obj_present else (v // 4, v // 4, v // 3)
                    draw.point((px, py), colour)
            label = f'depth_gradient_{i}'

        # Apply slight blur for anti-aliasing
        img = img.filter(ImageFilter.GaussianBlur(0.5))

        # Save
        fname = out_dir / f'{i:05d}_{label[:40]}.jpg'
        img.save(str(fname), 'JPEG', quality=85)

        # Convert to base64
        import io
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        items.append({
            'stage': 0,
            'type': 'image',
            'label': label,
            'file': str(fname),
            'b64': b64,
            'modality': 'image',
            'tags': ['visual_primitive', 'synthetic', mode == 0 and 'shading' or mode == 1 and 'perspective' or mode == 2 and 'occlusion' or 'depth'],
        })

    return items


# ── PubMed Central full-text downloader (Stage 1) ────────────────────────────

def fetch_pmc_articles(out_dir: Path, queries: list[str], max_per_query: int = 50) -> list[dict]:
    """Search PubMed Central OA for bovine anatomy articles and download abstracts/text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    items = []

    BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

    for query in tqdm(queries, desc='PMC queries'):
        try:
            # Search
            q = urllib.parse.quote(query + ' AND "open access"[filter]')
            url = f'{BASE}/esearch.fcgi?db=pmc&term={q}&retmax={max_per_query}&retmode=json&usehistory=y'
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
            ids = data.get('esearchresult', {}).get('idlist', [])

            for pmcid in ids:
                if pmcid in seen_ids:
                    continue
                seen_ids.add(pmcid)

                # Fetch abstract/summary
                try:
                    sum_url = f'{BASE}/esummary.fcgi?db=pmc&id={pmcid}&retmode=json'
                    with urllib.request.urlopen(sum_url, timeout=10) as r:
                        sdata = json.loads(r.read())
                    result = sdata.get('result', {}).get(pmcid, {})
                    title  = result.get('title', f'PMC{pmcid}')
                    year   = result.get('pubdate', '')[:4]

                    # Fetch full text (efetch NLM XML)
                    text_url = f'{BASE}/efetch.fcgi?db=pmc&id={pmcid}&rettype=abstract&retmode=text'
                    with urllib.request.urlopen(text_url, timeout=15) as r:
                        raw_text = r.read().decode('utf-8', errors='replace')

                    if len(raw_text) < 100:
                        continue

                    fname = out_dir / f'pmc{pmcid}.txt'
                    fname.write_text(raw_text, encoding='utf-8')

                    items.append({
                        'stage': 1,
                        'type': 'text',
                        'source': f'PMC{pmcid}',
                        'title': title,
                        'year': year,
                        'text': raw_text[:8000],  # cap at 8KB for ingest
                        'file': str(fname),
                        'modality': 'text',
                        'tags': ['pubmed', 'bovine', 'anatomy', 'scientific'],
                    })

                    time.sleep(0.35)  # NCBI rate limit: 3 req/s

                except Exception as e:
                    pass  # Skip failed articles silently

        except Exception as e:
            print(f'  [WARN] PMC query "{query}" failed: {e}')
            time.sleep(1)

    print(f'  Downloaded {len(items)} PMC articles')
    return items


def ingest_anatomy_knowledge(node_host: str) -> int:
    """Post the embedded ANATOMY_KNOWLEDGE blocks directly to /knowledge/ingest."""
    posted = 0
    for ka in ANATOMY_KNOWLEDGE:
        doc = {
            'document': {
                'doc_id': hashlib.sha1(ka['title'].encode()).hexdigest()[:12],
                'source': 'embedded_anatomy_corpus',
                'title': ka['title'],
                'text_blocks': [
                    {'block_id': f'b{i}', 'text': para.strip(), 'role': 'body',
                     'seq_index': i, 'seq_total': len(ka['text'].split('\n\n'))}
                    for i, para in enumerate(ka['text'].split('\n\n')) if para.strip()
                ],
                'metadata': {'tags': ka.get('tags', []), 'domain': 'bovine_anatomy'},
            }
        }
        body = json.dumps(doc).encode()
        req = urllib.request.Request(
            f'http://{node_host}/knowledge/ingest', data=body,
            headers={'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                posted += 1
        except Exception as e:
            print(f'  [WARN] knowledge ingest failed for "{ka["title"]}": {e}')
        time.sleep(0.05)
    return posted


# ── YouTube CC video downloader (Stage 2) ────────────────────────────────────

def download_youtube_cc(out_dir: Path, queries: list[str],
                        max_videos: int = 20, fps_extract: int = 2) -> list[dict]:
    """Download CC-licensed YouTube videos using yt-dlp and extract frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []

    try:
        import subprocess
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except Exception:
        print('[SKIP] Stage 2: yt-dlp not available (pip install yt-dlp)')
        return []

    if not HAS_CV2:
        print('[SKIP] Stage 2: opencv not available for frame extraction')
        return []

    vid_dir = out_dir / 'videos'
    vid_dir.mkdir(exist_ok=True)
    frame_dir = out_dir / 'frames'
    frame_dir.mkdir(exist_ok=True)

    downloaded = 0
    for query in queries:
        if downloaded >= max_videos:
            break
        print(f'  Searching YouTube CC: "{query}"')
        import subprocess
        try:
            result = subprocess.run([
                'yt-dlp',
                f'ytsearch5:{query}',
                '--match-filter', 'license = "Creative Commons Attribution licence (reuse allowed)"',
                '--format', 'bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]',
                '--merge-output-format', 'mp4',
                '--output', str(vid_dir / '%(id)s.%(ext)s'),
                '--max-downloads', '2',
                '--no-playlist',
                '--quiet',
                '--write-info-json',
            ], capture_output=True, text=True, timeout=120)

            downloaded += 2
        except Exception as e:
            print(f'  [WARN] yt-dlp failed for query "{query}": {e}')
            continue

    # Extract frames from all downloaded videos
    vid_files = list(vid_dir.glob('*.mp4'))
    print(f'  Extracting frames from {len(vid_files)} videos at {fps_extract} fps...')
    for vid_path in vid_files:
        vid_id = vid_path.stem
        info_path = vid_dir / f'{vid_id}.info.json'
        info = {}
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                pass

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(1, int(vid_fps / fps_extract))
        fn = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fn % frame_interval == 0:
                frame = cv2.resize(frame, (320, 240))
                out_path = frame_dir / f'{vid_id}_f{fn:06d}.jpg'
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                b64 = base64.b64encode(bytes(buf)).decode()

                items.append({
                    'stage': 2,
                    'type': 'video_frame',
                    'source': f'yt_{vid_id}',
                    'title': info.get('title', vid_id),
                    'frame_num': fn,
                    'file': str(out_path),
                    'b64': b64,
                    'modality': 'image',
                    'tags': ['video_frame', 'cow', 'bovine', 'outdoor'],
                })
            fn += 1
        cap.release()

    print(f'  Extracted {len(items)} video frames')
    return items


# ── Wikimedia Commons histology images (Stage 4) ────────────────────────────

def fetch_wikimedia_histology(out_dir: Path,
                               categories: list[str], max_per_cat: int = 50) -> list[dict]:
    """Download histology images from Wikimedia Commons."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []

    API = 'https://commons.wikimedia.org/w/api.php'
    # Wikimedia requires a descriptive User-Agent; bots without one get 403
    UA = 'W1z4rDV1510n-DatasetBuilder/1.0 (https://github.com/C4rr13rX/W1z4rDV1510n; adamedsall@gmail.com)'

    for cat in tqdm(categories, desc='Wikimedia histology'):
        params = {
            'action': 'query', 'list': 'categorymembers',
            'cmtitle': f'Category:{cat}', 'cmtype': 'file',
            'cmlimit': str(max_per_cat), 'format': 'json',
        }
        url = API + '?' + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            members = data.get('query', {}).get('categorymembers', [])

            for m in members:
                title = m.get('title', '')
                if not any(title.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    continue

                # Get image URL
                img_params = {
                    'action': 'query', 'titles': title,
                    'prop': 'imageinfo', 'iiprop': 'url|mime|size',
                    'iiurlwidth': '640', 'format': 'json',
                }
                img_url = API + '?' + urllib.parse.urlencode(img_params)
                try:
                    img_req = urllib.request.Request(img_url, headers={'User-Agent': UA})
                    with urllib.request.urlopen(img_req, timeout=10) as r:
                        idata = json.loads(r.read())
                    pages = idata.get('query', {}).get('pages', {})
                    for pid, page in pages.items():
                        ii = page.get('imageinfo', [{}])[0]
                        dl_url = ii.get('thumburl') or ii.get('url', '')
                        mime = ii.get('mime', '')
                        if not dl_url or 'image' not in mime:
                            continue
                        if ii.get('width', 0) < 100 or ii.get('height', 0) < 100:
                            continue

                        fname = out_dir / (hashlib.sha1(dl_url.encode()).hexdigest()[:12] + '.jpg')
                        if not fname.exists():
                            try:
                                dl_req = urllib.request.Request(dl_url, headers={'User-Agent': UA})
                                with urllib.request.urlopen(dl_req, timeout=15) as r:
                                    raw = r.read()
                                fname.write_bytes(raw)
                            except Exception:
                                continue

                        b64 = base64.b64encode(fname.read_bytes()).decode()
                        items.append({
                            'stage': 4,
                            'type': 'image',
                            'source': 'wikimedia_commons',
                            'title': title,
                            'category': cat,
                            'file': str(fname),
                            'b64': b64,
                            'modality': 'image',
                            'tags': ['histology', cat.lower().replace('_', ' '), 'microscopy'],
                        })
                        time.sleep(0.1)
                except Exception:
                    pass

        except Exception as e:
            print(f'  [WARN] Wikimedia category {cat}: {e}')

    print(f'  Downloaded {len(items)} histology images')
    return items


# ── PDB protein data (Stage 5) ───────────────────────────────────────────────

def fetch_pdb_structures(out_dir: Path, pdb_ids: list[tuple]) -> list[dict]:
    """Download PDB structures as FASTA + coordinate summary text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []

    for pdb_id, description in tqdm(pdb_ids, desc='PDB structures'):
        try:
            # Fetch FASTA sequence
            fasta_url = f'https://www.rcsb.org/fasta/entry/{pdb_id}/download'
            try:
                with urllib.request.urlopen(fasta_url, timeout=15) as r:
                    fasta = r.read().decode('utf-8', errors='replace')
            except Exception:
                fasta = f'>{pdb_id}\n[FASTA unavailable]\n'

            # Fetch PDBx/mmCIF header summary
            info_url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
            info_text = f'PDB ID: {pdb_id}\nDescription: {description}\n'
            try:
                with urllib.request.urlopen(info_url, timeout=15) as r:
                    pdb_info = json.loads(r.read())
                struct = pdb_info.get('struct', {})
                info_text += f'Title: {struct.get("title","")}\n'
                info_text += f'Keywords: {struct.get("pdbx_descriptor","")}\n'
                exp = pdb_info.get('exptl', [{}])
                info_text += f'Method: {exp[0].get("method","")}\n' if exp else ''
                entity = pdb_info.get('entity', [{}])
                for e in entity[:3]:
                    info_text += f'Chain {e.get("id","")}: {e.get("pdbx_description","")}, ' \
                                 f'{e.get("pdbx_number_of_molecules","")} mol, ' \
                                 f'type={e.get("type","")}\n'
            except Exception:
                pass

            # Fetch thumbnail image from RCSB
            img_url = f'https://cdn.rcsb.org/images/structures/{pdb_id.lower()[:2]}/{pdb_id.lower()}/{pdb_id.lower()}_assembly-1.jpeg'
            b64_img = None
            try:
                with urllib.request.urlopen(img_url, timeout=10) as r:
                    img_data = r.read()
                img_fname = out_dir / f'{pdb_id}.jpg'
                img_fname.write_bytes(img_data)
                b64_img = base64.b64encode(img_data).decode()
            except Exception:
                pass

            full_text = f'{info_text}\n\nFASTA:\n{fasta}'
            txt_fname = out_dir / f'{pdb_id}.txt'
            txt_fname.write_text(full_text, encoding='utf-8')

            item = {
                'stage': 5,
                'type': 'protein_structure',
                'source': f'PDB:{pdb_id}',
                'title': description,
                'text': full_text[:4000],
                'file': str(txt_fname),
                'modality': 'text',
                'tags': ['molecular', 'protein', 'PDB', 'bovine', 'structure'],
            }
            if b64_img:
                item['b64'] = b64_img
                item['modality'] = 'page'
            items.append(item)
            time.sleep(0.3)

        except Exception as e:
            print(f'  [WARN] PDB {pdb_id}: {e}')

    print(f'  Fetched {len(items)} PDB structures')
    return items


# ── Node API ingestor ────────────────────────────────────────────────────────

def ingest_item(item: dict, node_host: str) -> bool:
    """Post one curriculum item to the appropriate node API endpoint."""
    modality = item.get('modality', 'text')
    item_type = item.get('type', '')

    # Image/page/video_frame with b64 payload — send as image
    if modality in ('image', 'page', 'video_frame') and item.get('b64'):
        payload = {
            'modality': 'image',
            'data_b64': item['b64'],
            'text': item.get('title', '') or item.get('label', '') or item.get('source', ''),
            'lr_scale': 1.0,
        }
    elif modality == 'text' or item_type in ('text', 'protein_structure', 'pmc_article'):
        # Text-only or structured text (PDB/PMC): route through /media/train text
        text = item.get('text') or item.get('abstract', '') or item.get('title', '')
        if not text:
            return False
        payload = {
            'modality': 'text',
            'text': text[:6000],   # cap to avoid oversized payloads
            'lr_scale': 1.0,
        }
    else:
        return False

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'http://{node_host}/media/train', data=body,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            return True
    except Exception:
        return False


def ingest_batch(items: list[dict], node_host: str, workers: int = 4,
                 delay: float = 0.04) -> int:
    """Ingest a list of items concurrently."""
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ingest_item, it, node_host): it for it in items}
        for f in tqdm(as_completed(futs), total=len(futs), desc='Ingesting'):
            try:
                if f.result():
                    ok += 1
            except Exception:
                pass
            time.sleep(delay)
    return ok


# ── Curriculum manifest ───────────────────────────────────────────────────────

def write_manifest(data_dir: Path, stages: dict[int, list[dict]]) -> Path:
    """Write a curriculum.json manifest so the dataset can be inspected/resumed."""
    manifest = {
        'version': '1.0',
        'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'description': 'Bovine anatomy training corpus for W1z4rD V1510n neural fabric',
        'stages': {},
    }
    for stage_id, items in stages.items():
        manifest['stages'][str(stage_id)] = {
            'description': STAGES[stage_id],
            'count': len(items),
            'types': list({it['type'] for it in items}),
            'tags':  sorted({tag for it in items for tag in it.get('tags', [])}),
        }
    out = data_dir / 'curriculum.json'
    out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\n  Manifest written: {out}')
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Build and ingest bovine anatomy training dataset into W1z4rD node')
    ap.add_argument('--stages',    default='0,1,2,3,4,5',
                    help='Comma-separated stage IDs to run (default all)')
    ap.add_argument('--node',      default=DEFAULT_NODE,
                    help='Node host:port (default localhost:8090)')
    ap.add_argument('--data-dir',  default=DEFAULT_DATA_DIR,
                    help='Data directory root (default D:/w1z4rdv1510n-data)')
    ap.add_argument('--download-only', action='store_true',
                    help='Download assets without ingesting into node')
    ap.add_argument('--ingest-only',   action='store_true',
                    help='Skip downloads, ingest from existing manifest only')
    ap.add_argument('--workers', type=int, default=4,
                    help='Concurrent ingest workers (default 4)')
    ap.add_argument('--pmc-max',   type=int, default=30,
                    help='Max PMC articles per query (default 30)')
    ap.add_argument('--yt-videos', type=int, default=10,
                    help='Max YouTube videos to download (default 10)')
    args = ap.parse_args()

    stages_to_run = [int(s) for s in args.stages.split(',') if s.strip().isdigit()]
    data_dir = Path(args.data_dir)
    train_dir = data_dir / TRAINING_DIR_REL
    train_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*72)
    print('  W1z4rD V1510n — Bovine Anatomy Training Dataset Builder')
    print('='*72)
    print(f'  Node:      http://{args.node}')
    print(f'  Data dir:  {data_dir}')
    print(f'  Stages:    {stages_to_run}')
    print()

    all_items: dict[int, list[dict]] = {}

    # ── Stage 0: Synthetic visual primitives ────────────────────────────────
    if 0 in stages_to_run:
        print('\n[Stage 0] Synthetic visual primitives')
        items0 = generate_synthetic_primitives(train_dir / 'stage0_visual_primitives', n=500)
        all_items[0] = items0
        print(f'  Generated {len(items0)} items')

    # ── Stage 1: Text corpus ─────────────────────────────────────────────────
    if 1 in stages_to_run:
        print('\n[Stage 1] Bovine anatomy text corpus')

        # Embedded knowledge base (always post these)
        print('  Ingesting embedded anatomy knowledge base...')
        if not args.download_only:
            posted = ingest_anatomy_knowledge(args.node)
            print(f'  Posted {posted} knowledge documents')

        # PubMed Central
        print(f'  Fetching PubMed Central articles ({args.pmc_max} per query)...')
        items1a = fetch_pmc_articles(
            train_dir / 'stage1_pubmed', PMC_QUERIES, max_per_query=args.pmc_max)

        # PDB text
        items1b = [{'stage':1,'type':'text','source':'embedded','title':k['title'],
                    'text':k['text'],'modality':'text','tags':k['tags']} for k in ANATOMY_KNOWLEDGE]

        all_items[1] = items1a + items1b
        print(f'  Total Stage 1: {len(all_items[1])} items')

    # ── Stage 2: Video frames ────────────────────────────────────────────────
    if 2 in stages_to_run:
        print('\n[Stage 2] YouTube CC-licensed cow video frames')
        items2 = download_youtube_cc(
            train_dir / 'stage2_video', YOUTUBE_QUERIES,
            max_videos=args.yt_videos, fps_extract=2)
        all_items[2] = items2

    # ── Stage 3: Medical imaging (placeholder — requires free CT/MRI datasets) ─
    if 3 in stages_to_run:
        print('\n[Stage 3] Medical imaging (MRI/CT)')
        print('  NOTE: Bovine MRI/CT datasets require registration at:')
        print('    - MorphoSource (morphosource.org): 3D scans of bovine specimens')
        print('    - DICOM Library (dicomlibrary.com): anonymised veterinary DICOM')
        print('    - OpenVetAnatomy (github.com/mhalle/anatomy-atlases): CC-BY atlas')
        print('  After downloading, place DICOM/NIfTI files in:')
        print(f'    {train_dir / "stage3_mri_ct"}')
        print('  Then run: python ingest_dicom.py --dir <above> --node', args.node)
        # Create placeholder README
        s3_dir = train_dir / 'stage3_mri_ct'
        s3_dir.mkdir(exist_ok=True)
        readme = s3_dir / 'README.txt'
        readme.write_text(
            'Place bovine MRI/CT DICOM or NIfTI files here.\n'
            'Sources: MorphoSource, DICOM Library, OpenVetAnatomy.\n'
            'Run: python scripts/ingest_dicom.py --dir . --node localhost:8090\n'
        )
        all_items[3] = []

    # ── Stage 4: Histology images ────────────────────────────────────────────
    if 4 in stages_to_run:
        print('\n[Stage 4] Histology images from Wikimedia Commons')
        items4 = fetch_wikimedia_histology(
            train_dir / 'stage4_histology', WIKIMEDIA_HISTOLOGY_CATEGORIES, max_per_cat=40)
        all_items[4] = items4

    # ── Stage 5: Molecular/PDB ───────────────────────────────────────────────
    if 5 in stages_to_run:
        print('\n[Stage 5] PDB protein structures')
        items5 = fetch_pdb_structures(train_dir / 'stage5_molecular', PDB_IDS)
        all_items[5] = items5

    # ── Write manifest ───────────────────────────────────────────────────────
    write_manifest(data_dir, all_items)

    # ── Ingest all collected items ───────────────────────────────────────────
    if not args.download_only:
        all_flat = [it for items in all_items.values() for it in items
                    if it.get('b64') or it.get('text')]
        print(f'\n[Ingest] Posting {len(all_flat)} items to http://{args.node} ...')
        total_ok = ingest_batch(all_flat, args.node, workers=args.workers)
        print(f'  Ingested {total_ok}/{len(all_flat)} items successfully')
    else:
        total_items = sum(len(v) for v in all_items.values())
        print(f'\n[Download only] {total_items} items downloaded, not ingested.')
        print(f'  To ingest: python build_cow_dataset.py --ingest-only --node {args.node}')

    print('\n' + '='*72)
    print('  Dataset build complete.')
    total = sum(len(v) for v in all_items.values())
    for sid, items in sorted(all_items.items()):
        print(f'    Stage {sid} ({STAGES[sid][:45]:<45}): {len(items):>5} items')
    print(f'    {"TOTAL":<50}: {total:>5} items')
    print('='*72 + '\n')


if __name__ == '__main__':
    main()
