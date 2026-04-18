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
  Stage 6  — Bovine Q&A pairs   (embedded domain Q&A for /chat capability, ~300 pairs)

All items are structured into D:/w1z4rdv1510n-data/training/ and posted to
the W1z4rD node API for ingestion.

Usage:
  python build_cow_dataset.py [--stages 0,1,2,3,4,5,6] [--node localhost:8090]
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
    6: 'Bovine Q&A pairs   — embedded domain Q&A for chat capability',
    7: 'Cross-modal train  — video frame + anatomy text Hebbian links',
    8: 'Multi-angle gallery — Wikimedia Commons cow images (all views)',
    9: 'Mesh understanding   — OBJ renders + Sketchfab CC thumbnails',
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
# Verified 2026-04 against commons.wikimedia.org — these category names exist
# and contain microscopy image files.
WIKIMEDIA_HISTOLOGY_CATEGORIES = [
    'Histology_of_muscles',
    'Histology_of_bone',
    'Histology_of_skin',
    'Histology_of_loose_connective_tissue',
    'Histology_of_dense_connective_tissues',
    'Histology_of_nerves',
    'Histology_of_mammal_nerves',
    'Skeletal_muscle',
    'Bovine_anatomy',
    'Cattle',
    'Muscle_tissue',
    'Cardiac_muscle',
    'Gomori_trichrome_stain',
    'Muscle_fibers',
    'Histology_of_bone_marrow',
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
    {
        'title': 'Bovine MRI Neuroanatomy — Brain and Spinal Cord',
        'text': '''On T1-weighted MRI, bovine brain white matter appears hyperintense relative to grey matter (due to myelin lipid content). Major landmarks: cerebral hemispheres (frontal, parietal, temporal, occipital lobes), corpus callosum (white matter bridge), cerebellum with well-defined folia, brainstem (pons, medulla, midbrain), and the 4th ventricle. The olfactory bulbs are proportionally larger in cattle than in humans, reflecting enhanced olfaction.

On T2-weighted MRI, cerebrospinal fluid (CSF) in the lateral ventricles and subarachnoid space is hyperintense. White matter lesions (e.g., from polioencephalomalacia or listeriosis) show T2-hyperintensity. The cerebral cortex shows 6-layer neocortical organisation similar to other mammals but with reduced gyrification compared to carnivores.

The bovine spinal cord terminates at approximately S2–S3 in the sacral region (the conus medullaris). The dural sac extends further caudally. Epidural anaesthesia in cattle is performed at the sacrococcygeal (S5–Co1) junction, which corresponds to a palpable notch caudal to the last fixed spinal segment. On sagittal MRI, the cord shows a normal cervical enlargement (C5–T1) and lumbosacral enlargement (L4–S2) where the brachial and lumbosacral plexuses originate.

Skull CT Hounsfield units: frontal sinus air -1000 HU, frontal bone cortex 700–900 HU, frontal bone diploe 200–400 HU, brain parenchyma 30–40 HU.''',
        'tags': ['neurology', 'brain', 'spinal cord', 'MRI', 'CT', 'neuroanatomy', 'bovine', 'CSF'],
    },
    {
        'title': 'Bovine CT Thorax — Lung, Heart, and Great Vessels',
        'text': '''CT of the bovine thorax in lung window (W:1500, L:-600) reveals lung parenchyma HU -700 to -900 for aerated lung, -100 to +20 for atelectasis/consolidation. The normal bovine lung has a well-developed interlobular septa pattern visible on CT, more prominent than in horses, due to the complete lobulation characteristic of ruminants. This lobulation is clinically significant as pneumonia tends to remain segmental longer than in less-lobulated species.

The bovine heart is positioned left-of-midline with the cardiac apex pointing caudoventrally. On CT angiography, the main pulmonary artery arises from the right ventricle and bifurcates into left and right pulmonary arteries. The coronary arteries (left circumflex, left anterior descending, right coronary) are visible on ECG-gated CT as enhancing vessels. Cardiac chamber dimensions: LV end-diastolic diameter ≈ 8–10 cm, wall thickness ≈ 1.5–2.5 cm in adult cattle.

The aorta exits the LV, ascends as the ascending aorta, curves as the aortic arch, then descends as the thoracic and abdominal aorta. The brachiocephalic trunk is the first branch off the arch in cattle (unlike the separate right subclavian in humans), providing the right and left subclavian arteries and the bicarotid trunk. The caudal vena cava returns venous blood from the hindlimb and abdominal organs. Normal CT aortic diameter: 3.5–4.5 cm at the diaphragm level.''',
        'tags': ['thorax', 'CT', 'lung', 'heart', 'aorta', 'bovine', 'thoracic imaging', 'pulmonary'],
    },
    {
        'title': 'Bovine Abdominal CT — Rumen Reticulum Omasum Abomasum',
        'text': '''Bovine CT abdomen (soft tissue window W:400, L:40) shows the four-compartment forestomach system. The rumen is the largest organ, occupying the left 60–70% of the abdominal cavity. On CT, rumen contents show stratified layers: a dorsal gas cap (-900 HU, seen as black), a floating fibre mat (-100 to +50 HU), and a ventral liquid pool (+10 to +30 HU). The rumen wall is thin (5–8 mm) with prominent papillae.

The reticulum is cranioventral, in contact with the diaphragm and pericardium. Its honeycomb mucosal pattern is not visible on CT but its position anterior to the rumen is characteristic. Hardware disease (traumatic reticuloperitonitis) from ingested ferromagnetic objects is diagnosed on radiographs or CT by detecting a metallic density foreign body in the reticulum.

The omasum is right-lateral to the rumen, between the reticulum and abomasum. Its many leaves (laminae) trap large particles for further digestion. The abomasum (true stomach) is on the right ventral floor of the abdomen, adjacent to the liver. Right displacement of the abomasum (RDA) is diagnosed by finding the fluid-filled gas-capped abomasum on the right body wall on CT or auscultation (the "ping").

The bovine liver is right-sided (unlike the midline human liver), occupying the right cranial abdomen. Its CT density is +50 to +70 HU. The gallbladder, common bile duct, and portal vein are identifiable. Hepatic lipidosis (fatty liver of periparturient cows) shows reduced CT attenuation (-20 to +20 HU compared to normal +50–70).''',
        'tags': ['abdomen', 'CT', 'rumen', 'ruminant', 'digestive', 'liver', 'abomasum', 'bovine imaging'],
    },
    {
        'title': 'Bovine Limb MRI — Soft Tissue and Cartilage',
        'text': '''High-field MRI (1.5T or 3T) of bovine limbs provides detailed assessment of tendons, ligaments, and articular cartilage. T1-weighted sequences: tendons and ligaments appear dark (low signal, dense collagen fibres arranged parallel to the field). T2*-weighted gradient echo sequences: cartilage appears bright; subchondral bone is dark; bone marrow fat is hyperintense.

The digital flexor tendon apparatus: deep digital flexor tendon (DDFT) inserts on the distal phalanx (P3), the superficial digital flexor tendon (SDFT) inserts on P1 and P2. The DDFT is covered by a synovial sheath (digital flexor tendon sheath, DFTS) from the proximal pastern to the navicular bone. Inflammation of this sheath (tenosynovitis) shows T2-hyperintense fluid on MRI.

The navicular bone (distal sesamoid) of the bovine digit is analogous to the equine navicular. The navicular bursa (podotrochlear bursa) between the DDFT and navicular bone is a common site of inflammation. On MRI, bone marrow oedema appears as T2-hyperintensity and T1-hypointensity within the medullary cavity — an early indicator of osteitis before visible changes on radiographs.

Hoof cartilage (ungular cartilage) forms the lateral and medial cartilages of the hoof. These fibrocartilaginous extensions of P3 are better visualised on MRI than CT. Sidebone (ossification of ungular cartilage) is visible on CT as increased HU within what should be cartilage (-50 to +50 HU normal, +400+ HU ossified).''',
        'tags': ['limb', 'MRI', 'tendon', 'ligament', 'cartilage', 'hoof', 'digital', 'soft tissue'],
    },
    {
        'title': 'Bovine Histology — Integument (Skin and Hoof)',
        'text': '''Bovine skin histology: the epidermis consists of 4–5 layers: stratum germinativum (basal layer, mitotically active cuboidal keratinocytes on basement membrane), stratum spinosum (polygonal cells with desmosomes), stratum granulosum (cells containing keratohyalin granules), and stratum corneum (anucleate, keratinised squames). Cattle skin lacks a stratum lucidum except over hooves. Skin thickness varies from 2–3 mm over the trunk to 4–5 mm over the neck.

Bovine hair follicles are compound: 1 primary follicle flanked by 2–6 secondary follicles, all opening into a common infundibulum. Sebaceous glands open into the follicular canal. Eccrine sweat glands are numerous in cattle (unlike horses which sweat mainly from apocrine glands) and are important for thermoregulation.

The hoof (ungula) is a modified epidermis. Histological zones: perioplic corium (periople), coronary corium and coronary epidermis (tubular horn formation), lamellar corium (interdigitating primary and secondary dermal laminae with epidermal counterparts), and the sole/white line corium. The dermal-epidermal interface in the laminar dermis provides mechanical attachment of the hoof capsule to P3 via the interdigitating laminae. Laminitis in cattle disrupts this interface, allowing P3 to rotate within the hoof capsule under weight-bearing.

Melanocytes are located in the basal layer and determine coat colour. Breed pigmentation patterns: Holstein (black and white); Hereford (red with white face); Angus (black); Limousin (tan). Coat colour is determined by MC1R (extension locus) and ASIP (agouti), among other genes.''',
        'tags': ['histology', 'skin', 'hoof', 'epidermis', 'dermis', 'keratinocyte', 'melanocyte', 'integument'],
    },
    {
        'title': 'Bovine Histology — Skeletal Muscle Ultrastructure',
        'text': '''Skeletal muscle in cattle is organised into hierarchical levels: muscle belly → fascicles (endomysium-wrapped fibre bundles) → individual fibres (20–100 µm diameter, multinucleate syncytia) → myofibrils (1–2 µm diameter) → sarcomeres (repeating functional units, 2.0–2.5 µm at rest length).

The sarcomere boundaries are Z-lines (dense protein discs of α-actinin + actin anchor). I-bands (actin filaments, thin, appearing light on TEM) flank the Z-line. The A-band (myosin thick filaments, 15 nm diameter, 1.6 µm long) appears dark. The H-zone within the A-band contains no actin; the M-line bisects the H-zone and holds myosin thick filaments in register.

Fibre types in bovine skeletal muscle: Type I (slow twitch, oxidative, red, fatigue-resistant, high mitochondrial density, relies on oxidative phosphorylation) and Type IIA (fast twitch, oxidative-glycolytic) predominate in postural muscles like longissimus. Type IIB (fast-glycolytic, white, fatigable) predominate in sprinting muscles. The longissimus dorsi of beef cattle has been bred towards Type IIB abundance for faster growth and larger muscle fibre diameter.

The neuromuscular junction (NMJ) is where the motor nerve terminal synapses onto the muscle fibre. The presynaptic terminal contains dense-core vesicles of acetylcholine (ACh). ACh binds to nicotinic ACh receptors in the postjunctional folds, triggering depolarisation, Ca2+ release from sarcoplasmic reticulum via the T-tubule/RYR1 system, and cross-bridge cycling of myosin heads along actin.''',
        'tags': ['histology', 'muscle', 'sarcomere', 'myosin', 'actin', 'Z-line', 'NMJ', 'fibre type'],
    },
    {
        'title': 'Bovine Histology — Bone Microstructure and Remodelling',
        'text': '''Bovine cortical bone consists of osteons (Haversian systems): concentric lamellae (3–7 µm thick) of mineralised collagen surrounding a central Haversian canal (containing a capillary and nerve). Adjacent osteons are separated by interstitial lamellae (remnants of old osteons). The cement lines (reversal lines) around osteons are hypermineralised and stain intensely with Villanueva stain.

Bone mineral is predominantly carbonated hydroxyapatite [Ca10(PO4)6(OH)2], with Ca/P molar ratio ≈ 1.65 (stoichiometric = 1.67). Crystals are plate-shaped, 2–4 nm thick, 30–50 nm wide. The organic matrix (osteoid) is 90% type I collagen, 10% non-collagenous proteins (osteocalcin, osteopontin, BSP). Osteocalcin contains γ-carboxyglutamic acid residues that bind calcium; serum osteocalcin is a marker of bone formation.

Bone remodelling is coupled: osteoclasts (TRAP-positive, multinucleate, from monocyte lineage) excavate resorption lacunae (Howship's lacunae) via H+ secretion and metalloproteinase/cathepsin K activity. Osteoblasts (alkaline phosphatase-positive, from mesenchymal stem cells) refill with osteoid which mineralises within 7–10 days. The RANKL/OPG/RANK axis governs osteoclastogenesis — RANKL from osteoblasts and stromal cells activates RANK on osteoclast precursors; OPG is a decoy receptor.

Bone density in mature Holstein cows measured by dual-energy X-ray absorptiometry (DXA): L2–L4 BMD ≈ 1.1–1.4 g/cm², similar to humans. Transition cow mobilisation of bone calcium begins in early lactation when demand exceeds dietary supply.''',
        'tags': ['histology', 'bone', 'osteon', 'osteoclast', 'osteoblast', 'hydroxyapatite', 'remodelling', 'collagen'],
    },
    {
        'title': 'Bovine Vascular System — Arteries and Veins',
        'text': '''The bovine aorta gives rise to: (1) coronary arteries, (2) brachiocephalic trunk (dividing into bicarotid trunk → bilateral common carotid arteries, and bilateral subclavian arteries), (3) intercostal arteries, (4) coeliac artery (liver, spleen, stomach), (5) cranial mesenteric artery (small intestine), (6) caudal mesenteric (large intestine), (7) renal arteries, (8) ovarian/testicular arteries, (9) bilateral iliac arteries (external iliac → femoral; internal iliac → pelvic viscera).

The jugular vein is the primary venous access site in cattle for blood collection and intravenous therapy. The external jugular runs in the jugular groove between the sternocephalicus and brachiocephalicus muscles. The milk vein (subcutaneous abdominal vein, "spur vein") is prominent in high-producing dairy cows, draining the udder cranially through the inguinal canal into the thoracic cavity.

The bovine lymphatic system has numerous discrete lymph nodes: prescapular, prefemoral (sublumbar), inguinal, retropharyngeal, parotid, mediastinal, and mesenteric (dairy-cow jejunal lymph nodes form a row of conspicuous discrete nodes along the jejunum). The mammary lymph node (supramammary lymph node) is palpable at the base of the udder and is examined for evidence of mastitis-related lymphadenopathy.

Blood volume in cattle: approximately 8% of body weight. A 600 kg cow has approximately 48 litres of blood. Normal haematocrit 24–46%. RBC lifespan 135–150 days (longer than humans). Bovine RBCs lack sialic acid on their surface compared to human RBCs, making them less sticky and affecting blood rheology.''',
        'tags': ['vascular', 'artery', 'vein', 'aorta', 'jugular', 'lymph', 'circulation', 'bovine anatomy'],
    },
    {
        'title': 'Bovine Respiratory Anatomy — Trachea, Bronchi, Alveoli',
        'text': '''The bovine trachea has 50–60 cartilaginous rings and extends from the larynx to the tracheal bifurcation (carina) at the level of the 5th rib. Diameter: 5–7 cm in adult cattle. A tracheal bronchus is present in approximately 65% of cattle — an accessory bronchus arising from the right tracheal wall that ventilates the cranial right lung lobe directly from the trachea. This anatomical variant increases susceptibility to aspiration pneumonia of the cranial right lobe.

The bovine lung has complete lobulation with thick interlobular septa of connective tissue — each primary lobule (50–200 µm) is separated from its neighbours, unlike the partially-fused lobules of horses. This structural feature limits collateral ventilation (pores of Kohn are absent or rare) and means that once a lobule collapses or consolidates, it cannot be recruited by ventilation from adjacent lobules. This is why bovine pneumonia tends to show sharp lobular boundaries on necropsy.

The alveolar unit: type I pneumocytes (squamous, covering 95% of the surface area) provide the thin gas-exchange membrane. Type II pneumocytes (cuboidal, 5% of surface) produce pulmonary surfactant (dipalmitoylphosphatidylcholine, SP-A, SP-B, SP-C, SP-D) that reduces alveolar surface tension. Respiratory distress syndrome in neonatal calves (analogous to NRDS) involves surfactant deficiency.

Bovine lung capacity: approximately 30–35 litres in a 600 kg cow. Tidal volume at rest: 3–5 L. Respiratory rate: 12–30 breaths/min. Dead space (anatomic): approximately 2.0–2.5 L. Gas exchange: pO2 arterial ≈ 100 mmHg, pCO2 arterial ≈ 38–44 mmHg at sea level.''',
        'tags': ['respiratory', 'lung', 'trachea', 'bronchi', 'alveoli', 'surfactant', 'pneumonia', 'bovine'],
    },
    {
        'title': 'Bovine Immune System — Lymph Nodes and Innate Immunity',
        'text': '''The bovine immune system has evolved to cope with a high pathogen load from environmental exposure (soil, faeces, feed contamination). Innate immunity is particularly robust: neutrophils are the predominant response to acute bacterial infection (mastitis, pneumonia). Bovine neutrophils have higher baseline activity than human neutrophils and show enhanced oxidative burst.

Toll-like receptors (TLRs) on bovine macrophages and dendritic cells recognise pathogen-associated molecular patterns (PAMPs): TLR4 recognises LPS from gram-negative bacteria (E. coli), TLR2 recognises gram-positive cell wall components, TLR9 recognises bacterial CpG DNA. TLR gene variants in cattle are associated with resistance/susceptibility to mastitis.

The bovine thymus involutes after puberty (similar to other mammals) but has distinctive histology in calves: large cortex packed with thymocytes (CD4+CD8+ double-positive), and medulla with Hassall corpuscles (concentrically arranged epithelial cells). Thymic dendritic cells and macrophages mediate negative selection (deletion of self-reactive T-cells).

The bovine spleen has a well-developed trabecular meshwork of smooth muscle (erythroid spleen type), allowing forceful contraction to release stored erythrocytes during exercise or hypoxia. White pulp (periarterial lymphatic sheaths + germinal centres) is surrounded by red pulp (sinusoids + splenic cords). Bovine spleen CT HU: +45 to +65 (compared to liver +50–70, muscle +40–60).''',
        'tags': ['immune', 'lymph nodes', 'neutrophil', 'TLR', 'thymus', 'spleen', 'innate immunity', 'bovine'],
    },
    {
        'title': 'Bovine Reproductive Anatomy — Uterus Ovary Cervix',
        'text': '''The bovine uterus is bicornuate (bipartite): two uterine horns join at the body. In the cow, the uterine body is only 2–4 cm long; the horns are 25–40 cm long in adults and curl ventrally ("ram's horn" configuration). The uterus is supported by the broad ligament. Caruncles (70–100 raised, non-glandular structures) are arranged in 4 rows on the endometrial surface; during pregnancy they interdigitate with cotyledons on the placenta to form placentomes.

The cervix of the cow is 8–10 cm long, fibrous, and has 3–5 distinct annular rings (interlocking rings/annuli). These rings prevent easy passage of instruments (making intrauterine insemination more difficult in cattle than in sheep). The cervix is palpable per rectum and is used as a landmark for rectal palpation of the reproductive tract.

Ovaries: approximately 3 × 2 × 1.5 cm in cyclic cows. The dominant follicle grows to 12–20 mm before ovulation. Ultrasonographic ovarian evaluation by transrectal B-mode shows follicles as anechoic (fluid-filled) structures and corpora lutea as hypoechoic/isoechoic structures. The corpus haemorrhagicum (immediately post-ovulation) has a distinctive sunflower-like ultrasound pattern from the central clot.

Bovine oestrous cycle: 21 days (range 18–24). Follicular phase 4–6 days; luteal phase 15–18 days. Progesterone from the CL: 4–10 ng/mL in mid-luteal phase. Oestradiol from the pre-ovulatory follicle: 5–15 pg/mL at oestrus. Gestation length: 280–285 days (range 274–292). Twinning rate: 1–5% in dairy breeds (3–4× higher than in beef breeds).''',
        'tags': ['reproduction', 'uterus', 'ovary', 'cervix', 'oestrus', 'gestation', 'cattle', 'bovine'],
    },
    {
        'title': 'Bovine Udder — Mammary Gland Anatomy and Lactation Physiology',
        'text': '''The bovine udder consists of four mammary quarters, each with an independent gland cistern, teat cistern, and teat canal (streak canal). The four quarters are separated by a medial suspensory ligament (elastic tissue, a branch of the abdominal tunica) and lateral suspensory ligaments (fibrous, from the subcutaneous fascia). The medial ligament bears the primary weight of milk; its failure leads to udder drop in old or high-producing cows.

Milk secretion: secretory epithelium (lactocytes) in alveoli undergo continuous synthesis and apocrine secretion of milk. Each alveolus is surrounded by a basket of myoepithelial cells that contract in response to oxytocin (released by posterior pituitary in response to suckling/milking stimulus) to eject milk into the ducts → cisterns → teat. Oxytocin half-life: 3–5 minutes; let-down reflex lasts 5–8 minutes.

Bovine milk composition: water 87%, fat 3.5–5%, protein 3.2% (casein 2.5%, whey 0.7%), lactose 4.7%, minerals 0.7%. Holstein milk averages 3.5–4% fat; Jersey 4.5–5.5%. Casein micelles (80–300 nm diameter) are the dominant protein; their calcium-phosphate-nanoclusters coordinate through casein phosphoserines. Fat globules are 1–10 µm, coated by a milk fat globule membrane (MFGM) of phospholipids, glycoproteins, and butyrophilin.

Mastitis: inflammation of the mammary gland, the most economically costly disease in dairy cattle. On MRI, acute mastitis shows T2-hyperintense oedema and enlarged supramammary lymph nodes. The glandular parenchyma enhances with gadolinium in inflammatory mastitis. Chronic mastitis shows fibrosis (T1 and T2 hypointensity) and loss of normal alveolar architecture.''',
        'tags': ['udder', 'mammary', 'lactation', 'milk', 'mastitis', 'oxytocin', 'casein', 'bovine'],
    },
    {
        'title': 'Bovine Gastrointestinal Histology — Rumen to Abomasum',
        'text': '''Rumen wall histology: the mucosa is lined by stratified squamous non-glandular epithelium (similar to oesophageal epithelium). The epithelium is keratinised on its luminal surface. Rumen papillae (1–10 mm tall, finger-like projections) vastly increase surface area for VFA absorption. Each papilla has a core of lamina propria with a capillary loop. Subepithelial blood flow removes absorbed VFAs (acetate, propionate, butyrate) for portal hepatic metabolism.

Reticulum: honeycombed mucosal folds, same non-glandular stratified squamous epithelium as rumen. The reticular groove (oesophageal groove) is a muscular fold in young calves that directs milk past the rumen into the abomasum — closure is stimulated by suckling reflex (serotonin + gastrin mediated).

Omasum ("Bible tripes"): 100–200 muscular laminae (leaves) lined by stratified squamous epithelium with short papillae. Function: water absorption and fine-particle trapping. The laminae expand surface area to ~10 m² — more than 10× the omasal luminal surface.

Abomasum (true glandular stomach): glandular mucosa with oxyntic (fundic) glands containing parietal cells (HCl secretion, H+/K+-ATPase), chief cells (pepsinogen), and G-cells (gastrin). Surface mucous cells secrete bicarbonate-laden mucus. Abomasal histology resembles the human fundic stomach. pH 2.0–4.0 in fasted cows; rises to 5.0–6.0 post-feeding as secretion buffers ingesta. Abomasal displacement shows gas-fluid interface on CT (inverted U-shape of gas-filled abomasum).''',
        'tags': ['gastrointestinal', 'rumen', 'ruminant', 'abomasum', 'histology', 'VFA', 'papillae', 'digestion'],
    },
    {
        'title': 'Bovine Endocrine System — Pituitary, Thyroid, Adrenal',
        'text': '''The bovine pituitary gland (hypophysis) weighs 2–3 g and is situated in the sella turcica of the sphenoid bone. Adenohypophysis (anterior pituitary): acidophils secrete GH (somatotropin, 190 aa, bovine recombinant = rbST used in dairy production) and prolactin. Basophils secrete LH, FSH, ACTH, TSH. The hypothalamic-pituitary-gonadal axis regulates bovine reproduction via GnRH → LH/FSH → oestradiol/progesterone feedback.

Thyroid gland: two lobes connected by an isthmus at the laryngeal-tracheal junction. Follicular cells synthesise T4 (thyroxine) and T3 (triiodothyronine) from tyrosine residues within thyroglobulin. Bovine T3 is biologically 3–4× more potent than T4 (T3 is the active form after peripheral deiodination). C-cells (parafollicular cells) secrete calcitonin — important in calcium homeostasis in lactating cows (calcitonin inhibits osteoclast activity and reduces Ca2+ resorption from bone).

Bovine adrenal gland anatomy: retroperitoneal, cranial to each kidney. Cortex zones: glomerulosa (aldosterone — Na+ retention), fasciculata (cortisol — stress, glucose metabolism), reticularis (androgens). Medulla: chromaffin cells secrete adrenaline (80%) and noradrenaline (20%) in response to sympathetic stimulation. Milk ejection failure under stress is partly mediated by adrenaline causing peripheral vasoconstriction and reduced oxytocin delivery to udder.

Pancreas: both exocrine (acinar cells secreting digestive enzymes) and endocrine (islets of Langerhans: α-cells/glucagon, β-cells/insulin, δ-cells/somatostatin). Bovine insulin was the first protein hormone to be sequenced (Sanger, 1953); bovine insulin differs from human insulin at 3 amino acid positions and was used therapeutically in humans before recombinant human insulin.''',
        'tags': ['endocrine', 'pituitary', 'thyroid', 'adrenal', 'pancreas', 'insulin', 'cortisol', 'bovine'],
    },
    {
        'title': 'Bovine Renal System — Kidney Anatomy and Physiology',
        'text': '''Bovine kidneys are multilobed (lobulated surface), unlike the smooth surface in horses. Each lobe corresponds to one cortical pyramid draining to a minor calyx. The bovine kidney has 18–24 lobes; the cortex, multiple medullary pyramids, and a single renal pelvis (no true renal pelvis sinus as in horses — instead multiple minor calyces drain to a major calyx system). On CT, the cortex is HU +30 to +50 (enhancement peak HU +130 with iodinated contrast), and the medullary pyramids are slightly less dense.

The nephron is the functional unit: glomerulus (filtration, ~120 mL/min GFR total in adult cow), proximal tubule (60–70% Na+/glucose/amino acid/HCO3- reabsorption), loop of Henle (countercurrent multiplication for urine concentration), distal tubule (aldosterone-regulated Na+/K+ exchange, ADH-regulated water permeability), and collecting duct (final concentration, urine osmolality 1000–2500 mOsm/kg in dehydrated animals).

The juxtaglomerular apparatus (JGA): JG cells (modified smooth muscle of afferent arteriole) secrete renin → angiotensin I → ACE → angiotensin II → aldosterone axis. This RAAS is important in sodium/water balance of dairy cattle during high-salt diets and during heat stress.

Daily urine output: 10–30 litres in adult dairy cattle, depending on water intake and lactation stage. Urine composition: urea (major nitrogen excretion product, as in all ureotelic mammals), creatinine (proportional to muscle mass, stable), hippurate (from aromatic amino acid catabolism). Polyuria/polydipsia in cattle may indicate diabetes insipidus, renal insufficiency, or hypercalcaemia.''',
        'tags': ['renal', 'kidney', 'nephron', 'glomerulus', 'RAAS', 'urine', 'CT', 'bovine anatomy'],
    },
    {
        'title': 'Bovine Embryology and Fetal Development',
        'text': '''Bovine fertilisation occurs in the ampullary-isthmic junction of the oviduct, approximately 12 hours after ovulation. First cleavage at 24–30 hours post-fertilisation. The 8-cell stage (day 3) is when embryonic genome activation occurs in cattle (unlike mice where it occurs at the 2-cell stage). Blastocyst: inner cell mass (ICM, future embryo proper) surrounded by trophoblast at day 7. Hatching from the zona pellucida occurs days 9–11.

Embryonic attachment in cattle is a slow process of apposition and adhesion (synepitheliochorial placentation, not invasive). Binucleate cells (giant cells) in the trophoblast migrate into the endometrial epithelium to form fetomaternal hybrid cells (trinucleate cells in cattle). Placentome formation begins day 20–25; mature placentomes are present by day 40. The interdigitating cotyledon-caruncle structure allows nutrient and gas exchange without breach of the endometrial epithelium.

Fetal sex differentiation: the gonadal ridge forms day 30. Testes in males secrete AMH (anti-Müllerian hormone, causing regression of Müllerian ducts) and testosterone (promoting Wolffian duct development: epididymis, vas deferens, seminal vesicles). In females, ovaries passively develop; oocytes enter meiotic prophase I arrest (dictyotene) by day 100.

Major organogenesis landmarks in bovine fetal development: cardiac tube (day 20), limb buds (day 26), neural tube closure (day 28), external genitalia differentiation visible (day 50), palatal fusion (day 60), eye opening (day 225–250). Crown-rump length correlates with gestational age: 2 cm ≈ day 40, 10 cm ≈ day 70, 30 cm ≈ day 110, 50 cm ≈ day 140.''',
        'tags': ['embryology', 'fetal development', 'placenta', 'fertilisation', 'organogenesis', 'bovine', 'reproduction'],
    },
    {
        'title': 'Bovine Biomechanics and Gait Analysis',
        'text': '''The bovine walk is a 4-beat gait with a lateral sequence: LH→LF→RH→RF (or RH→RF→LH→LF). At a slow walk, the support polygon includes 3 limbs simultaneously (tripodal stance), providing excellent stability — important for heavy ruminants that must remain standing for long periods. At a fast walk, periods of bipedal lateral support occur.

The trot is rare in cattle; they transition to a gallop (3-beat asymmetric) or a run. Cattle reach a maximum speed of approximately 8–9 m/s (29–32 km/h). The trot-gallop transition occurs at a Froude number of approximately 0.5 (Fr = v²/gL, where L = leg length). Cattle have a shorter relative leg length than horses, shifting their preferred trot-gallop transition speed to lower velocities.

Ground reaction forces in cattle: peak vertical GRF approximately 1.5–2.0× body weight during walking. The hindlimb peak GRF occurs slightly later in the stance phase than the forelimb. Hoof-ground contact time: 350–550 ms at a walk. Digital pulse (palpable increase in digital artery pulse amplitude) is detectable in laminitic cows due to increased blood flow and hoof capsule temperature.

Kinematic hoof analysis: the hoof angle (angle of the dorsal hoof wall to the ground) should be 45–55° in the forefeet and 50–60° in the hindlimb lateral claws. Overgrowth of the toe leads to a broken-forward hoof angle; excessive heel growth causes a broken-backward angle. Both alter biomechanics and predispose to lameness. Cow comfort scoring and locomotion scoring (1–5 scale) assess gait symmetry and back curvature.''',
        'tags': ['biomechanics', 'gait', 'locomotion', 'hoof', 'ground reaction force', 'kinematic', 'lameness', 'bovine'],
    },
    {
        'title': 'Bovine Biochemistry — Volatile Fatty Acids and Energy Metabolism',
        'text': '''Bovine energy metabolism is fundamentally different from monogastric species due to rumen fermentation. Dietary carbohydrates (cellulose, starch, pectin) are fermented by rumen microbiota to volatile fatty acids (VFAs): acetate (60–65%), propionate (15–20%), and butyrate (10–15%). These VFAs are absorbed across the rumen wall and serve as the primary energy substrates.

Acetate is the main substrate for peripheral fatty acid synthesis and oxidation; it enters the TCA cycle after conversion to acetyl-CoA. Propionate is gluconeogenic — it is converted in the liver to succinyl-CoA → oxaloacetate → glucose (cattle derive 95% of their blood glucose from hepatic gluconeogenesis, not intestinal absorption, because most glucose is fermented in the rumen before reaching the duodenum). Butyrate is the primary fuel for rumen epithelial cells and is converted to β-hydroxybutyrate (ketone body) and acetoacetate.

Negative energy balance (NEB) in periparturient dairy cows: milk energy output exceeds dietary energy intake in early lactation. Body fat is mobilised → elevated plasma NEFA (non-esterified fatty acids) → hepatic uptake exceeds oxidative capacity → triacylglycerol accumulation in hepatocytes (fatty liver) → reduced gluconeogenesis capacity → subclinical and clinical ketosis.

Normal bovine blood glucose: 2.8–4.4 mmol/L (50–80 mg/dL) — lower than humans (4.0–6.0 mmol/L) because glucose is continuously consumed by rumen epithelium. Insulin concentrations are also chronically lower in ruminants. The glucose transporter GLUT1 (erythrocyte/brain) is more important than GLUT4 (insulin-responsive) in bovine peripheral tissues.''',
        'tags': ['biochemistry', 'VFA', 'ketosis', 'glucose', 'NEB', 'rumen fermentation', 'energy metabolism', 'bovine'],
    },
    {
        'title': 'Bovine Nociception and Pain Pathways',
        'text': '''Nociception in cattle involves the same peripheral and central mechanisms as in other mammals. Primary afferent nociceptors: Aδ-fibres (myelinated, 5–30 m/s, "fast pain", mechanical and thermal nociception) and C-fibres (unmyelinated, 0.5–2.0 m/s, "slow pain", polymodal chemical, mechanical, thermal). In the bovine hoof, free nerve endings in the dermis of the corium respond to laminar separation, tissue damage, and inflammation.

Spinal cord dorsal horn processing: the superficial laminae (I and II, substantia gelatinosa) receive primary C-fibre input via substance P and CGRP neurotransmitters. Lamina V receives Aδ and deep Aβ input. Ascending pathways: spinothalamic tract to contralateral thalamus → somatosensory cortex. Wind-up (temporal summation) occurs in dorsal horn wide dynamic range (WDR) neurons via NMDA receptor activation — relevant to chronic pain in laminitic cows where sensitisation amplifies hoof pain.

Opioid receptors (µ, κ, δ) are present throughout bovine nociceptive pathways. Butorphanol (κ-agonist/µ-antagonist) is commonly used in cattle for analgesia. NSAIDs (flunixin meglumine, meloxicam) inhibit COX-1 and COX-2, reducing prostaglandin synthesis at the site of inflammation. Local anaesthesia: lidocaine (2%) blocks Na+ channels on nerve axons, abolishing action potential propagation — used for ring block, IV regional anaesthesia (Bier block), cornual nerve block, epidural.

Pain assessment in cattle uses composite pain scales including behaviours: ear position (folded forward = pain indicator), facial expression (Bovine Pain Scale with orbital tightening, tension above eye, strained chewing), posture (arched back, weight shifting, reduced weight bearing), and physiological indices (HR, RR, cortisol, substance P).''',
        'tags': ['nociception', 'pain', 'analgesia', 'nerve', 'NSAID', 'local anaesthesia', 'welfare', 'bovine'],
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

    # Try yt-dlp via Python API first, then fall back to subprocess on C:\Python313
    _yt_python = None
    try:
        import yt_dlp as _yt_module  # noqa: F401
        _yt_python = 'api'
    except ImportError:
        pass
    if _yt_python is None:
        # yt_dlp installed in roaming user site-packages — needs path injection
        _ytdlp_site = r'C:\Users\Adam\AppData\Roaming\Python\Python313\site-packages'
        _chk_script = f'import sys; sys.path.insert(0, {_ytdlp_site!r}); import yt_dlp'
        for _py in [r'C:\Python313\python.exe']:
            try:
                import subprocess as _sp
                r = _sp.run([_py, '-c', _chk_script], capture_output=True, timeout=10)
                if r.returncode == 0:
                    _yt_python = _py
                    break
            except Exception:
                continue
    if _yt_python is None:
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
            # Skip webpage extraction to bypass JS challenge (works without deno).
            # Prefer smallest viable format (≤360p) and cap file size at 80MB
            # so frame extraction stays fast and disk-friendly.
            _FMT = 'worstvideo[height>=240]+worstaudio/worst[height>=240]/worst'
            if _yt_python == 'api':
                import yt_dlp as _yt
                _yt_opts = {
                    'format': _FMT,
                    'outtmpl': str(vid_dir / '%(id)s.%(ext)s'),
                    'max_downloads': 3, 'noplaylist': True, 'quiet': True,
                    'ignoreerrors': True, 'writeinfojson': True,
                    'max_filesize': 80 * 1024 * 1024,   # 80 MB hard cap
                    'match_filter': _yt.utils.match_filter_func('duration < 600'),
                    'extractor_args': {'youtube': {'skip': ['webpage']}},
                }
                try:
                    _yt.YoutubeDL(_yt_opts).download([f'ytsearch5:{query}'])
                except _yt.utils.MaxDownloadsReached:
                    pass
            else:
                # Inject roaming site-packages so yt_dlp is importable from C:\Python313
                _ytdlp_site2 = r'C:\Users\Adam\AppData\Roaming\Python\Python313\site-packages'
                _outtmpl = str(vid_dir / '%(id)s.%(ext)s').replace('\\', '/')
                _yt_inline = '\n'.join([
                    f'import sys; sys.path.insert(0, {_ytdlp_site2!r})',
                    'import yt_dlp',
                    'opts = {',
                    f'  "format": "worstvideo[height>=240]+worstaudio/worst[height>=240]/worst",',
                    f'  "outtmpl": {_outtmpl!r},',
                    '  "max_downloads": 3, "noplaylist": True, "quiet": True,',
                    '  "ignoreerrors": True, "writeinfojson": True,',
                    f'  "max_filesize": {80 * 1024 * 1024},',
                    '  "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),',
                    '  "extractor_args": {"youtube": {"skip": ["webpage"]}},',
                    '}',
                    'try:',
                    f'  yt_dlp.YoutubeDL(opts).download(["ytsearch5:{query}"])',
                    'except yt_dlp.utils.MaxDownloadsReached:',
                    '  pass',
                ])
                subprocess.run([_yt_python, '-c', _yt_inline],
                               capture_output=True, text=True, timeout=180)

            downloaded += 3
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


# ── Medical imaging: synthetic MRI/CT cross-sections (Stage 3) ───────────────
# Generates synthetic medical imaging data approximating bovine MRI/CT:
#   • 3 modality types: T1-weighted MRI, T2-weighted MRI, CT (Hounsfield)
#   • 8 anatomical levels: head, neck, shoulder/brachium, thorax, abdomen,
#                          pelvis, thigh/stifle, cannon/metatarsus
#   • Each level × modality = 1 synthetic cross-section image + 1 text document
#
# Tissue gray values (0-255) per modality:
#   T1   fat=220, muscle=80, bone=160, marrow=210, fluid=30, air=0
#   T2   fat=160, muscle=100, bone=30, marrow=80, fluid=250, air=0
#   CT   bone=255, cortex=240, muscle=120, fat=60, fluid=90, air=0  (HU mapped 0-255)

MRI_CT_ANATOMY_LEVELS = [
    {
        'name': 'Head (transverse — orbital level)',
        'label': 'head_orbital',
        'desc': (
            'Transverse MRI/CT at the mid-orbital level of the bovine head. '
            'The section reveals the paired orbital cones (retro-orbital fat, optic nerve, '
            'rectus muscles), the frontal sinuses laterally, the nasal passages centrally, '
            'the turbinate bones (scroll-like conchae), and the masseter muscles '
            'flanking the zygomatic arches. On T1-MRI retro-orbital fat appears bright; '
            'on T2-MRI, the vitreous humor is hyperintense. CT clearly delineates the '
            'orbital walls and turbinate bone structure (HU 400-700 cortical bone, '
            'HU -100 to -50 retro-orbital fat).'
        ),
        'tissues': [  # (name, center_x, center_y, w, h, angle, gray_T1, gray_T2, gray_CT)
            ('cortical_bone',    0.5, 0.5, 0.85, 0.70, 0,   160, 30,  240),
            ('frontal_sinus_L',  0.2, 0.3, 0.12, 0.10, 15,   0,  0,   0),
            ('frontal_sinus_R',  0.8, 0.3, 0.12, 0.10, -15,  0,  0,   0),
            ('nasal_passage',    0.5, 0.5, 0.08, 0.20, 0,    0,  0,   0),
            ('retroorb_fat_L',   0.22, 0.48, 0.16, 0.14, 0, 220, 160,  60),
            ('retroorb_fat_R',   0.78, 0.48, 0.16, 0.14, 0, 220, 160,  60),
            ('masseter_L',       0.12, 0.62, 0.18, 0.22, 20,  80, 100, 120),
            ('masseter_R',       0.88, 0.62, 0.18, 0.22, -20, 80, 100, 120),
            ('vitreous_L',       0.30, 0.45, 0.08, 0.08, 0,   30, 250,  90),
            ('vitreous_R',       0.70, 0.45, 0.08, 0.08, 0,   30, 250,  90),
            ('brain_frontal',    0.5, 0.30, 0.22, 0.14, 0,  130,  90, 130),
        ],
    },
    {
        'name': 'Neck (transverse — C4 level)',
        'label': 'neck_c4',
        'desc': (
            'Transverse MRI/CT at cervical vertebra C4. The vertebral body is central with '
            'the spinous process dorsal. The spinal cord occupies the vertebral canal. '
            'The trachea is ventral (air-filled lumen), the oesophagus is left-lateral. '
            'Bilateral splenius, semispinalis capitis, and multifidus muscles surround '
            'the vertebral column. The jugular veins and carotid arteries are visible '
            'in the neurovascular bundle on each side. Subcutaneous fat and skin form '
            'the outer boundary. CT HU: vertebral cortex ~700, cancellous ~250, '
            'tracheal air -1000, muscle 40–60, fat -100 to -50.'
        ),
        'tissues': [
            ('skin_fat',         0.5, 0.5, 0.90, 0.85, 0,   200, 150,  55),
            ('splenius_L',       0.22, 0.40, 0.20, 0.28, 15,  80, 100, 120),
            ('splenius_R',       0.78, 0.40, 0.20, 0.28, -15, 80, 100, 120),
            ('multifidus',       0.5,  0.30, 0.18, 0.20, 0,   80, 100, 120),
            ('vertebra_body',    0.5,  0.58, 0.18, 0.16, 0,  160,  30, 240),
            ('vertebral_canal',  0.5,  0.52, 0.06, 0.06, 0,   30,  90,  90),
            ('spinal_cord',      0.5,  0.52, 0.04, 0.04, 0,  100, 120, 120),
            ('trachea',          0.5,  0.78, 0.10, 0.08, 0,    0,   0,   0),
            ('jugular_L',        0.30, 0.70, 0.04, 0.06, 0,   30, 250,  90),
            ('jugular_R',        0.70, 0.70, 0.04, 0.06, 0,   30, 250,  90),
        ],
    },
    {
        'name': 'Shoulder / Brachium (transverse — scapular level)',
        'label': 'shoulder_scapular',
        'desc': (
            'Transverse MRI at the level of the scapula and shoulder joint. The '
            'scapular body (triangular bone) is visible dorsally. The glenohumeral '
            'joint space contains synovial fluid (T2-hyperintense). The deltoid, '
            'supraspinatus, infraspinatus, and subscapularis muscles are identifiable '
            'by their positions and signal intensities. The brachial plexus and '
            'subclavian vessels are medially placed. On CT the scapular cortex (HU 700+) '
            'contrasts strongly with the adjacent musculature (HU 40-60).'
        ),
        'tissues': [
            ('skin_fat',          0.5, 0.5, 0.90, 0.88, 0,  210, 155,  58),
            ('scapula',           0.5, 0.28, 0.40, 0.12, 0,  160,  30, 240),
            ('supraspinatus',     0.5, 0.20, 0.28, 0.14, 0,   80, 100, 120),
            ('infraspinatus',     0.5, 0.40, 0.28, 0.18, 0,   80, 100, 120),
            ('deltoid_L',         0.22, 0.55, 0.18, 0.30, 20,  80, 100, 120),
            ('deltoid_R',         0.78, 0.55, 0.18, 0.30, -20, 80, 100, 120),
            ('humerus_head_L',    0.28, 0.60, 0.14, 0.14, 0,  160,  30, 240),
            ('humerus_head_R',    0.72, 0.60, 0.14, 0.14, 0,  160,  30, 240),
            ('joint_fluid_L',     0.28, 0.60, 0.06, 0.06, 0,   30, 250,  90),
            ('joint_fluid_R',     0.72, 0.60, 0.06, 0.06, 0,   30, 250,  90),
            ('subcutan_fat',      0.5, 0.85, 0.60, 0.08, 0,  220, 160,  55),
        ],
    },
    {
        'name': 'Thorax (transverse — T6 level, mid-heart)',
        'label': 'thorax_t6',
        'desc': (
            'Transverse MRI/CT at the 6th thoracic vertebra level, showing the cardiac '
            'silhouette. The heart occupies the left-of-centre thoracic cavity. Cardiac '
            'chambers: left ventricle (thick-walled, LV myocardium T1=medium), left atrium, '
            'right ventricle and right atrium. Both lungs flank the mediastinum. On T2, '
            'intracardiac blood pools are bright. On CT, the mediastinal fat (HU -80), '
            'lung parenchyma (HU -700 to -800), pleural fluid if present (HU +15), '
            'aortic arch and pulmonary vessels. Ribs and thoracic vertebra surround the section.'
        ),
        'tissues': [
            ('lung_L',            0.18, 0.50, 0.32, 0.50, 0,   20,  20,   5),
            ('lung_R',            0.82, 0.50, 0.32, 0.50, 0,   20,  20,   5),
            ('lv_myocardium',     0.44, 0.52, 0.20, 0.22, 0,  100, 110, 130),
            ('lv_lumen',          0.44, 0.52, 0.10, 0.12, 0,   30, 250,  90),
            ('rv_chamber',        0.54, 0.44, 0.18, 0.16, 15,  30, 250,  90),
            ('thoracic_vertebra', 0.5,  0.25, 0.14, 0.12, 0,  160,  30, 240),
            ('spinal_cord',       0.5,  0.25, 0.04, 0.04, 0,  100, 120, 120),
            ('ribs_L',            0.22, 0.45, 0.04, 0.30, 30, 160,  30, 240),
            ('ribs_R',            0.78, 0.45, 0.04, 0.30, -30,160,  30, 240),
            ('mediastinal_fat',   0.5,  0.50, 0.08, 0.14, 0,  220, 160,  55),
            ('aorta',             0.42, 0.28, 0.05, 0.05, 0,   30, 250,  90),
            ('skin_muscle',       0.5,  0.5,  0.94, 0.90, 0,  100, 110, 100),
        ],
    },
    {
        'name': 'Abdomen (transverse — L2 level, rumen)',
        'label': 'abdomen_rumen',
        'desc': (
            'Transverse MRI/CT at the second lumbar vertebra level in a bovine, showing the '
            'large ruminant forestomachs. The rumen occupies the left 60% of the abdominal '
            'cavity — its gas cap appears dark on all sequences. Reticulum, omasum, and '
            'abomasum occupy the right ventral area. Liver is right dorsal, spleen is '
            'left lateral. Both kidneys flank the lumbar spine. The lumbar muscles '
            '(longissimus dorsi, iliopsoas) are large and well-defined. On T2-MRI, '
            'rumen ingesta shows heterogeneous signal. CT: rumen gas (-900 HU), liver (+55 HU), '
            'spleen (+50 HU), kidney cortex (+30 HU), retroperitoneal fat (-80 HU).'
        ),
        'tissues': [
            ('skin_fat',          0.5,  0.5,  0.96, 0.92, 0,  210, 155,  58),
            ('longissimus_L',     0.35, 0.25, 0.20, 0.22, 0,   80, 100, 120),
            ('longissimus_R',     0.65, 0.25, 0.20, 0.22, 0,   80, 100, 120),
            ('lumbar_vertebra',   0.5,  0.22, 0.14, 0.12, 0,  160,  30, 240),
            ('rumen_gas',         0.38, 0.42, 0.30, 0.26, 0,    0,   0,   0),
            ('rumen_ingesta',     0.40, 0.60, 0.32, 0.20, 0,   80,  90, 100),
            ('liver',             0.72, 0.38, 0.22, 0.20, 0,  120, 100, 140),
            ('spleen',            0.20, 0.50, 0.10, 0.16, 0,  110, 100, 135),
            ('kidney_L',          0.36, 0.30, 0.08, 0.10, 0,  100, 110, 120),
            ('kidney_R',          0.68, 0.30, 0.08, 0.10, 0,  100, 110, 120),
            ('retro_fat',         0.5,  0.28, 0.30, 0.12, 0,  210, 155,  55),
            ('abomasum',          0.62, 0.70, 0.14, 0.10, 0,   30, 200,  90),
        ],
    },
    {
        'name': 'Pelvis (transverse — sacral level)',
        'label': 'pelvis_sacral',
        'desc': (
            'Transverse MRI/CT at the sacral level of the bovine pelvis. The sacrum is '
            'dorsal-central; the iliac wings form the lateral walls of the pelvic inlet. '
            'The pelvic canal contains the rectum (filled with faeces), bladder '
            '(T2-hyperintense urine), and in females the uterus and ovaries. The '
            'gluteal muscle group (gluteus medius, biceps femoris) is prominent. '
            'The caudal aorta and iliac vessels are visible medially. On CT the '
            'sacral foramina are identifiable (HU 600-700 cortical), and the '
            'sciatic nerve is in the sciatic notch.'
        ),
        'tissues': [
            ('skin_fat',          0.5,  0.5,  0.96, 0.90, 0,  210, 155,  58),
            ('gluteus_medius_L',  0.22, 0.38, 0.22, 0.32, 15,  80, 100, 120),
            ('gluteus_medius_R',  0.78, 0.38, 0.22, 0.32, -15, 80, 100, 120),
            ('biceps_femoris_L',  0.20, 0.65, 0.18, 0.28, 0,   80, 100, 120),
            ('biceps_femoris_R',  0.80, 0.65, 0.18, 0.28, 0,   80, 100, 120),
            ('sacrum',            0.5,  0.25, 0.20, 0.14, 0,  160,  30, 240),
            ('sacral_canal',      0.5,  0.26, 0.04, 0.04, 0,   30,  90,  90),
            ('bladder',           0.5,  0.62, 0.14, 0.12, 0,   30, 250,  90),
            ('rectum',            0.5,  0.42, 0.08, 0.10, 0,   60,  70, 100),
            ('ilium_L',           0.28, 0.28, 0.14, 0.10, 20, 160,  30, 240),
            ('ilium_R',           0.72, 0.28, 0.14, 0.10, -20,160,  30, 240),
            ('iliopsoas_L',       0.34, 0.42, 0.10, 0.14, 0,   80, 100, 120),
            ('iliopsoas_R',       0.66, 0.42, 0.10, 0.14, 0,   80, 100, 120),
        ],
    },
    {
        'name': 'Thigh / Stifle (transverse — mid-femoral)',
        'label': 'thigh_femur',
        'desc': (
            'Transverse MRI/CT at mid-femoral level. The femur shaft (cortical ring with '
            'medullary canal) is central. The quadriceps group (rectus femoris, vastus '
            'lateralis/medialis/intermedius) occupies the cranial compartment. Caudally '
            'the biceps femoris and semitendinosus form the hamstrings. The femoral '
            'artery and vein occupy the medial aspect. The femoral nerve branch travels '
            'with the vessels. On T1-MRI, yellow marrow in the medullary canal appears '
            'bright (fat-signal). On CT, femoral cortex HU 1000-1200, medullary fat '
            'HU -100, surrounding muscle HU 40-60.'
        ),
        'tissues': [
            ('skin_fat',          0.5,  0.5,  0.90, 0.88, 0,  210, 155,  58),
            ('rectus_femoris',    0.5,  0.30, 0.18, 0.20, 0,   80, 100, 120),
            ('vastus_lat_L',      0.26, 0.40, 0.18, 0.24, 0,   80, 100, 120),
            ('vastus_med_R',      0.74, 0.40, 0.18, 0.24, 0,   80, 100, 120),
            ('biceps_femoris',    0.5,  0.68, 0.22, 0.20, 0,   80, 100, 120),
            ('femur_cortex',      0.5,  0.50, 0.22, 0.22, 0,  160,  30, 240),
            ('femur_marrow',      0.5,  0.50, 0.12, 0.12, 0,  210,  80, 60),
            ('femoral_vessel',    0.36, 0.55, 0.04, 0.06, 0,   30, 250,  90),
            ('semitendinosus',    0.60, 0.65, 0.14, 0.18, 0,   80, 100, 120),
        ],
    },
    {
        'name': 'Cannon Bone / Metatarsus (transverse)',
        'label': 'cannon_metatarsus',
        'desc': (
            'Transverse MRI/CT through the bovine cannon bone (fused third and fourth '
            'metatarsals in the hindlimb). The fused metatarsal cortex forms a '
            'characteristic figure-eight cross-section outline. The medullary cavity '
            'of each component is visible (T1-bright yellow marrow). Surrounding tissues: '
            'digital extensor tendons dorsally, deep digital flexor tendon palmarly, '
            'the digital flexor tendon sheath (T2-bright synovial fluid), and thin skin '
            'with minimal subcutaneous fat. CT: cortex HU 1100-1400, marrow HU -80, '
            'tendons HU 70-100 (fibrous), tendon sheath fluid HU +10.'
        ),
        'tissues': [
            ('skin',              0.5,  0.5,  0.78, 0.82, 0,  160, 120, 110),
            ('extensor_tendon',   0.5,  0.22, 0.10, 0.12, 0,  140,  60, 100),
            ('flexor_tendon',     0.5,  0.75, 0.10, 0.14, 0,  140,  60, 100),
            ('tendon_sheath',     0.5,  0.72, 0.14, 0.18, 0,   30, 250,  90),
            ('mt3_cortex',        0.40, 0.50, 0.18, 0.22, 0,  160,  30, 240),
            ('mt3_marrow',        0.40, 0.50, 0.10, 0.14, 0,  210,  80,  60),
            ('mt4_cortex',        0.60, 0.50, 0.18, 0.22, 0,  160,  30, 240),
            ('mt4_marrow',        0.60, 0.50, 0.10, 0.14, 0,  210,  80,  60),
            ('interosseous_space',0.5,  0.50, 0.04, 0.18, 0,   80, 100, 100),
        ],
    },
    {
        'name': 'Digit / Hoof (transverse — pastern level)',
        'label': 'digit_pastern',
        'desc': (
            'Transverse MRI/CT through the bovine pastern (proximal phalanx P1). '
            'The paired digits (dewclaws absent at this level) show two proximal phalanges '
            'side by side. Cortical bone (HU 1100–1400) surrounds the medullary cavity. '
            'Deep digital flexor and superficial digital flexor tendons flank each digit. '
            'The digital annular ligament and collateral ligaments are present. '
            'T2-MRI shows the digital tendon sheaths as hyperintense rings. '
            'Dorsal: common extensor tendons. Plantar: DDFT sheath fluid. '
            'Minimal subcutaneous tissue and thick hoof wall capsule (high CT density).'
        ),
        'tissues': [
            ('hoof_wall',           0.5,  0.5,  0.78, 0.80, 0,  150, 100, 200),
            ('p1_cortex_L',         0.34, 0.48, 0.14, 0.18, 0,  160,  30, 240),
            ('p1_marrow_L',         0.34, 0.48, 0.07, 0.10, 0,  200,  80,  50),
            ('p1_cortex_R',         0.66, 0.48, 0.14, 0.18, 0,  160,  30, 240),
            ('p1_marrow_R',         0.66, 0.48, 0.07, 0.10, 0,  200,  80,  50),
            ('ddft_sheath_L',       0.30, 0.68, 0.10, 0.12, 0,   30, 250,  90),
            ('ddft_sheath_R',       0.70, 0.68, 0.10, 0.12, 0,   30, 250,  90),
            ('ddft_L',              0.30, 0.68, 0.06, 0.08, 0,  140,  60, 100),
            ('ddft_R',              0.70, 0.68, 0.06, 0.08, 0,  140,  60, 100),
            ('extensor_tendon_L',   0.34, 0.28, 0.07, 0.10, 0,  140,  60, 100),
            ('extensor_tendon_R',   0.66, 0.28, 0.07, 0.10, 0,  140,  60, 100),
            ('interdigital_skin',   0.5,  0.5,  0.04, 0.30, 0,  140, 120, 110),
        ],
    },
    {
        'name': 'Udder (transverse — gland parenchyma level)',
        'label': 'udder_parenchyma',
        'desc': (
            'Transverse MRI/CT through the bovine udder at the level of the mammary '
            'gland parenchyma. Four gland quarters are separated by the median suspensory '
            'ligament (medially) and lateral suspensory ligaments. The glandular parenchyma '
            'appears heterogeneous — active lactating tissue is T2-hyperintense (high water). '
            'The cisterns (T2-bright milk pools) are visible at the base of each quarter. '
            'On CT the glandular tissue is ~20–40 HU (similar to water/fat mix). '
            'The median ligament is low-signal on T1 and T2 (fibrous). '
            'Teat orifices are visible ventrally. Subcutaneous fat is T1-bright.'
        ),
        'tissues': [
            ('subcutan_fat',        0.5,  0.5,  0.96, 0.90, 0,  220, 160,  55),
            ('median_lig',          0.5,  0.5,  0.03, 0.60, 0,  100,  60, 110),
            ('gland_q1',            0.25, 0.40, 0.30, 0.40, 0,  110, 150,  70),
            ('gland_q2',            0.75, 0.40, 0.30, 0.40, 0,  110, 150,  70),
            ('gland_q3',            0.25, 0.65, 0.30, 0.35, 0,  110, 150,  70),
            ('gland_q4',            0.75, 0.65, 0.30, 0.35, 0,  110, 150,  70),
            ('cistern_q1',          0.25, 0.72, 0.10, 0.08, 0,   30, 250,  30),
            ('cistern_q2',          0.75, 0.72, 0.10, 0.08, 0,   30, 250,  30),
            ('lat_lig_L',           0.08, 0.5,  0.04, 0.50, 0,  100,  60, 110),
            ('lat_lig_R',           0.92, 0.5,  0.04, 0.50, 0,  100,  60, 110),
        ],
    },
    {
        'name': 'Rumen-Reticulum Junction (transverse)',
        'label': 'rumen_reticulum',
        'desc': (
            'Transverse MRI/CT at the cranial rumen / reticulum junction level. '
            'The reticulum (honeycomb stomach) is cranioventral — its mucosa shows '
            'characteristic reticular folds on high-resolution MRI. The rumen cranial '
            'sac lies caudal. Both contain ingesta (heterogeneous T2 signal) with a '
            'gas cap dorsally (hypointense on MRI, ~-900 HU on CT). '
            'The liver is right dorsal. The diaphragm is cranial. '
            'Abdominal wall muscles (external/internal obliques, transversus) '
            'form the lateral and ventral boundary. '
            'Hardware disease risk: metallic densities on CT within the reticulum.'
        ),
        'tissues': [
            ('abdom_wall',          0.5,  0.5,  0.96, 0.88, 0,   90, 110, 115),
            ('rumen_gas',           0.42, 0.28, 0.28, 0.18, 0,    0,   0,   0),
            ('rumen_ingesta',       0.42, 0.52, 0.30, 0.28, 0,   80,  90, 100),
            ('reticulum_wall',      0.30, 0.68, 0.16, 0.14, 0,   90, 100, 120),
            ('reticulum_lumen',     0.30, 0.68, 0.10, 0.10, 0,   60,  90,  90),
            ('liver',               0.72, 0.38, 0.22, 0.20, 0,  120, 100, 140),
            ('gallbladder',         0.68, 0.46, 0.06, 0.08, 0,   30, 250,  30),
            ('diaphragm',           0.5,  0.20, 0.80, 0.08, 0,   90, 100, 120),
            ('peritoneal_fat',      0.5,  0.5,  0.90, 0.80, 0,  210, 155,  55),
        ],
    },
    {
        'name': 'Heart at Apex (transverse — T9 level)',
        'label': 'heart_apex',
        'desc': (
            'Transverse MRI/CT at the cardiac apex level (T9 vertebra). '
            'The apex of the left ventricle is visible as a blunt cone of myocardium '
            '(T1: intermediate, T2: intermediate). The pericardial sac surrounds the '
            'heart (thin low-signal line). The pericardial fluid layer is T2-bright. '
            'Both lungs flank the heart. The caudal vena cava is visible right-dorsally. '
            'The descending aorta is adjacent to the vertebral body. '
            'CT: myocardium HU 50–80, pericardial fluid HU 15–30, '
            'lung ~-750 HU, vertebral cortex ~700 HU.'
        ),
        'tissues': [
            ('lung_L',              0.18, 0.50, 0.30, 0.52, 0,   20,  20,   5),
            ('lung_R',              0.82, 0.50, 0.30, 0.52, 0,   20,  20,   5),
            ('pericardium',         0.46, 0.55, 0.16, 0.18, 0,  100, 110, 130),
            ('pericardial_fluid',   0.46, 0.55, 0.13, 0.15, 0,   30, 250,  30),
            ('lv_apex',             0.46, 0.55, 0.09, 0.11, 0,  100, 110, 130),
            ('lv_lumen_apex',       0.46, 0.55, 0.04, 0.05, 0,   30, 250,  90),
            ('t9_vertebra',         0.5,  0.22, 0.14, 0.12, 0,  160,  30, 240),
            ('spinal_cord',         0.5,  0.22, 0.04, 0.04, 0,  100, 120, 120),
            ('descend_aorta',       0.44, 0.26, 0.04, 0.04, 0,   30, 250,  90),
            ('caudal_vena_cava',    0.54, 0.30, 0.05, 0.05, 0,   30, 250,  90),
            ('intercostal_muscles', 0.5,  0.5,  0.94, 0.90, 0,   85, 105, 120),
        ],
    },
    {
        'name': 'Brain Stem (sagittal — midline)',
        'label': 'brain_stem_sagittal',
        'desc': (
            'Sagittal MRI section through the bovine brainstem midline. '
            'From cranial to caudal: olfactory bulbs, cerebral cortex (gyri and sulci), '
            'corpus callosum (white matter bridge), thalamus, midbrain (mesencephalon), '
            'pons, medulla oblongata, and cerebellar vermis. '
            'The fourth ventricle is T2-hyperintense (CSF). '
            'The pituitary gland sits in the sella turcica (T1-bright due to posterior '
            'pituitary fat content). The spinal cord continues caudally. '
            'T1-MRI: white matter brighter than grey matter; '
            'T2-MRI: CSF=250, white matter=90, grey matter=100. '
            'CT: bone of calvarium HU 700, brain parenchyma HU 30–40.'
        ),
        'tissues': [
            ('calvarium',           0.5,  0.5,  0.72, 0.88, 0,  150,  30, 230),
            ('cerebral_cortex',     0.5,  0.38, 0.55, 0.55, 0,  110, 100, 130),
            ('white_matter',        0.5,  0.42, 0.40, 0.35, 0,  140,  90, 130),
            ('corpus_callosum',     0.5,  0.48, 0.30, 0.04, 0,  140,  90, 130),
            ('thalamus',            0.5,  0.52, 0.14, 0.10, 0,  100, 100, 130),
            ('midbrain',            0.5,  0.60, 0.10, 0.10, 0,  100, 100, 130),
            ('pons',                0.5,  0.68, 0.12, 0.10, 0,  100, 100, 130),
            ('medulla',             0.5,  0.76, 0.10, 0.10, 0,  100, 100, 130),
            ('cerebellum',          0.5,  0.60, 0.20, 0.18, 15,  105, 105, 130),
            ('fourth_ventricle',    0.5,  0.66, 0.06, 0.06, 0,   30, 250,  30),
            ('pituitary',           0.5,  0.58, 0.04, 0.04, 0,  180, 100, 120),
            ('spinal_cord_c1',      0.5,  0.84, 0.04, 0.06, 0,  100, 120, 120),
        ],
    },
    {
        'name': 'Elbow Joint (transverse — olecranon level)',
        'label': 'elbow_olecranon',
        'desc': (
            'Transverse MRI/CT through the bovine elbow at the olecranon process of '
            'the ulna. The distal humerus articular condyles are visible. '
            'The elbow joint space contains synovial fluid (T2-hyperintense). '
            'Lateral collateral, medial collateral, and annular ligaments are present. '
            'The extensor and flexor muscle groups surround the joint. '
            'The radial and ulnar nerves travel along the medial and lateral aspects. '
            'CT: olecranon cortex HU 1000+, cancellous ~250, synovial fluid ~30 HU. '
            'T2-MRI: joint effusion, if present, clearly hyperintense.'
        ),
        'tissues': [
            ('skin_fat',            0.5,  0.5,  0.88, 0.85, 0,  210, 155,  58),
            ('triceps_brachii',     0.5,  0.22, 0.30, 0.24, 0,   80, 100, 120),
            ('olecranon',           0.5,  0.30, 0.14, 0.18, 0,  160,  30, 240),
            ('olecranon_marrow',    0.5,  0.30, 0.07, 0.10, 0,  200,  80,  50),
            ('humerus_condyle_L',   0.34, 0.58, 0.16, 0.14, 0,  160,  30, 240),
            ('humerus_condyle_R',   0.66, 0.58, 0.16, 0.14, 0,  160,  30, 240),
            ('joint_fluid',         0.5,  0.55, 0.20, 0.08, 0,   30, 250,  30),
            ('extensor_group_L',    0.22, 0.65, 0.18, 0.28, 0,   80, 100, 120),
            ('extensor_group_R',    0.78, 0.65, 0.18, 0.28, 0,   80, 100, 120),
            ('flexor_group',        0.5,  0.72, 0.22, 0.22, 0,   80, 100, 120),
        ],
    },
    {
        'name': 'Stifle Joint (transverse — patella level)',
        'label': 'stifle_patella',
        'desc': (
            'Transverse MRI/CT at the patella level of the bovine stifle (femorotibial '
            'joint, equivalent to human knee). The patella is an oval sesamoid embedded in '
            'the quadriceps tendon cranially. The patellar ligament inserts on the tibial '
            'tuberosity. Medial and lateral femoral condyles, menisci (fibrocartilage, '
            'intermediate signal), cruciate ligaments, and collateral ligaments are present. '
            'The stifle joint is the most complex bovine joint. '
            'T2: menisci intermediate, cruciate low-signal, joint fluid very bright. '
            'CT: condyle cortex HU 800+, articular cartilage 50–100 HU.'
        ),
        'tissues': [
            ('skin_fat',            0.5,  0.5,  0.92, 0.90, 0,  210, 155,  58),
            ('quadriceps',          0.5,  0.20, 0.28, 0.22, 0,   80, 100, 120),
            ('patella',             0.5,  0.28, 0.12, 0.08, 0,  160,  30, 240),
            ('patellar_lig',        0.5,  0.36, 0.07, 0.18, 0,  130,  60, 110),
            ('femur_condyle_L',     0.30, 0.54, 0.20, 0.18, 0,  160,  30, 240),
            ('femur_condyle_R',     0.70, 0.54, 0.20, 0.18, 0,  160,  30, 240),
            ('joint_fluid',         0.5,  0.54, 0.24, 0.06, 0,   30, 250,  30),
            ('meniscus_L',          0.30, 0.58, 0.07, 0.06, 0,  110,  80, 120),
            ('meniscus_R',          0.70, 0.58, 0.07, 0.06, 0,  110,  80, 120),
            ('biceps_femoris_L',    0.18, 0.60, 0.18, 0.28, 0,   80, 100, 120),
            ('biceps_femoris_R',    0.82, 0.60, 0.18, 0.28, 0,   80, 100, 120),
            ('gastrocnemius',       0.5,  0.78, 0.24, 0.20, 0,   80, 100, 120),
        ],
    },
    {
        'name': 'Tail / Sacrococcygeal (transverse — Co3 level)',
        'label': 'tail_coccygeal',
        'desc': (
            'Transverse MRI/CT through the third coccygeal vertebra of the bovine tail. '
            'The coccygeal vertebra is a small rounded bone centrally, with the '
            'caudal arteries (paired median caudal artery and veins) ventral. '
            'Coccygeal muscles (caudal flexors and extensors) are thin concentric layers '
            'around the vertebra. Skin is the outermost layer — bovine tail skin is '
            'relatively thick with coarse hair follicles. '
            'On CT: coccygeal vertebral cortex HU ~700, central marrow fat HU -80. '
            'T2-MRI: caudal vessels are T2-bright (flowing blood or slow flow); '
            'muscle is intermediate signal. Useful landmark for tail-head temperature '
            'monitoring and caudal epidural injection site planning.'
        ),
        'tissues': [
            ('skin',                0.5,  0.5,  0.70, 0.72, 0,  150, 120, 120),
            ('tail_extensor',       0.5,  0.28, 0.22, 0.18, 0,   80, 100, 120),
            ('tail_flexor',         0.5,  0.70, 0.22, 0.18, 0,   80, 100, 120),
            ('lateral_muscle_L',    0.28, 0.50, 0.14, 0.22, 0,   80, 100, 120),
            ('lateral_muscle_R',    0.72, 0.50, 0.14, 0.22, 0,   80, 100, 120),
            ('coccygeal_vertebra',  0.5,  0.50, 0.16, 0.14, 0,  160,  30, 240),
            ('coccygeal_marrow',    0.5,  0.50, 0.08, 0.07, 0,  200,  80,  50),
            ('caudal_artery',       0.5,  0.65, 0.04, 0.04, 0,   30, 200,  90),
            ('caudal_vein_L',       0.42, 0.65, 0.03, 0.03, 0,   30, 250,  90),
            ('caudal_vein_R',       0.58, 0.65, 0.03, 0.03, 0,   30, 250,  90),
        ],
    },
]


def generate_mri_ct_data(out_dir: Path, n_noise_levels: int = 8) -> list[dict]:
    """
    Generate synthetic bovine MRI/CT cross-section images and accompanying text.
    Produces 3 modalities × 16 anatomical levels × n_noise_levels images,
    plus 16 detailed text documents (one per anatomical level).
    Default n_noise_levels=8 gives 384 images + 16 text = 400 items.
    """
    if not HAS_PIL:
        print('[SKIP] Stage 3: Pillow not available')
        return []

    import io as _io
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    SIZE = 512

    MODALITIES = [
        ('T1', 6),   # T1-weighted MRI, index into tissue gray tuple
        ('T2', 7),   # T2-weighted MRI
        ('CT', 8),   # CT Hounsfield (mapped 0-255)
    ]

    rng = np.random.default_rng(42)

    for level in MRI_CT_ANATOMY_LEVELS:
        label = level['label']

        # ── Text document ────────────────────────────────────────────────────
        full_text = (
            f"BOVINE MRI/CT ATLAS — {level['name']}\n\n"
            f"{level['desc']}\n\n"
            "Tissue composition at this level:\n"
        )
        for t in level['tissues']:
            hu_t1, hu_t2, hu_ct = t[6], t[7], t[8]
            # Map HU 0-255 scale back to approximate Hounsfield for CT
            hu_approx = int(hu_ct / 255 * 1400 - 200)
            full_text += (
                f"  {t[0].replace('_',' ')}: T1={hu_t1}/255  T2={hu_t2}/255  "
                f"CT≈{hu_approx} HU\n"
            )

        txt_fname = out_dir / f'{label}.txt'
        txt_fname.write_text(full_text, encoding='utf-8')
        items.append({
            'stage': 3, 'type': 'text', 'source': 'synthetic_mri_atlas',
            'title': f'Bovine MRI/CT Atlas — {level["name"]}',
            'text': full_text, 'file': str(txt_fname),
            'modality': 'text',
            'tags': ['mri', 'ct', 'medical_imaging', 'bovine_anatomy', label],
        })

        # ── Synthetic cross-section images ───────────────────────────────────
        for mod_name, gray_idx in MODALITIES:
            for noise_seed in range(n_noise_levels):
                # Build image as float array
                img_f = np.zeros((SIZE, SIZE), dtype=np.float32)

                # Draw tissues from back to front (painter's algorithm)
                for tissue in level['tissues']:
                    tname, cx_n, cy_n, w_n, h_n, angle_deg = tissue[:6]
                    gray = tissue[gray_idx]

                    cx = int(cx_n * SIZE); cy = int(cy_n * SIZE)
                    w  = int(w_n  * SIZE / 2); h  = int(h_n  * SIZE / 2)

                    # Create mask via PIL (handles rotation cleanly)
                    mask = Image.new('L', (SIZE, SIZE), 0)
                    mdraw = ImageDraw.Draw(mask)
                    bbox = [(cx - w, cy - h), (cx + w, cy + h)]
                    mdraw.ellipse(bbox, fill=255)

                    if abs(angle_deg) > 0.5:
                        mask = mask.rotate(-angle_deg, center=(cx, cy))

                    mask_a = np.array(mask, dtype=np.float32) / 255.0
                    img_f = img_f * (1 - mask_a) + gray * mask_a

                # Add spatially-correlated noise (blurred Gaussian)
                noise = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
                # Smooth noise to simulate MRI texture / CT grain
                noise_img = Image.fromarray(
                    np.clip(noise * 8 + 128, 0, 255).astype(np.uint8), 'L')
                noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=2))
                noise_a = (np.array(noise_img, dtype=np.float32) - 128) * (
                    0.04 + noise_seed * 0.03)
                img_f = np.clip(img_f + noise_a, 0, 255)

                # Add circular field-of-view boundary (MRI FOV circle)
                cy2, cx2 = SIZE // 2, SIZE // 2
                ys, xs = np.ogrid[:SIZE, :SIZE]
                outside = ((xs - cx2) ** 2 + (ys - cy2) ** 2) > (SIZE // 2 - 8) ** 2
                img_f[outside] = 0

                img_u8 = img_f.astype(np.uint8)
                pil_img = Image.fromarray(img_u8, 'L').convert('RGB')

                # Annotate modality in corner
                ann_draw = ImageDraw.Draw(pil_img)
                ann_draw.text((10, 10), f'{mod_name} | {level["name"]}', fill=(180, 180, 180))

                buf = _io.BytesIO()
                pil_img.save(buf, 'JPEG', quality=88)
                b64 = base64.b64encode(buf.getvalue()).decode()

                fname = out_dir / f'{label}_{mod_name}_{noise_seed}.jpg'
                fname.write_bytes(buf.getvalue())

                items.append({
                    'stage': 3, 'type': 'mri_ct_slice', 'source': 'synthetic_mri_atlas',
                    'title': f'{mod_name} {level["name"]} (noise={noise_seed})',
                    'label': label, 'modality_type': mod_name,
                    'file': str(fname), 'b64': b64,
                    'modality': 'image',
                    'tags': ['mri', 'ct', mod_name.lower(), 'medical_imaging', 'bovine_anatomy', label],
                })

    print(f'  Generated {len(items)} MRI/CT items '
          f'({len(MRI_CT_ANATOMY_LEVELS)} levels × {len(MODALITIES)} modalities × '
          f'{n_noise_levels} noise variants + {len(MRI_CT_ANATOMY_LEVELS)} text docs)')
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


# ── Stage 6: Bovine Q&A dataset ─────────────────────────────────────────────
# ~300 question-answer pairs covering all major bovine anatomy / physiology systems.
# Ingested via /qa/ingest to enable accurate /chat responses about cow anatomy.

BOVINE_QA_PAIRS: list[tuple[str, str]] = [
    # ── Musculoskeletal ───────────────────────────────────────────────────────
    ("How many bones does a bovine skeleton have?",
     "The bovine skeleton consists of approximately 210 bones."),
    ("What is the vertebral formula of cattle?",
     "The bovine vertebral formula is C7, T13, L6, S5, Co18-20 (7 cervical, 13 thoracic, 6 lumbar, 5 sacral, 18-20 coccygeal vertebrae)."),
    ("What is the cannon bone in cattle?",
     "The cannon bone is the fused third and fourth metacarpal (MC3+MC4) bone in the forelimb, or the fused third and fourth metatarsal (MT3+MT4) in the hindlimb. Fusion of these two bones is characteristic of bovine cloven-hoofed locomotion."),
    ("How many phalanges does each bovine digit have?",
     "Each bovine digit has three phalanges: the proximal phalanx (P1), the middle phalanx (P2), and the distal phalanx (P3, also called the pedal bone or coffin bone)."),
    ("What is the largest muscle in cattle?",
     "The longissimus dorsi is the largest single muscle mass in cattle. It runs along the back from the lumbar vertebrae to the ribs and is the source of the ribeye cut. It is used in body condition scoring."),
    ("What percentage of body weight do bovine forelimbs bear?",
     "Approximately 55-60% of body weight is borne by the forelimbs, and 40-45% by the hindlimbs in a standing bovine."),
    ("Which bovine hoof claw is more prone to sole ulcers and why?",
     "The lateral claw of the hindfoot bears approximately 60-70% of the load (compared to the medial claw), predisposing it to sole ulcers and white line disease."),
    ("What are the three zones of the bovine hoof capsule?",
     "The bovine hoof capsule consists of the wall (dorsal stratum), the sole, and the white line—a zone of soft horn at the junction of wall and sole that is the most common site of white line disease."),
    ("What are epaxial muscles in cattle?",
     "Epaxial muscles are dorsal trunk muscles including the longissimus, iliocostalis, and spinalis. They extend the vertebral column and are positioned above the transverse processes of the vertebrae."),
    ("What is the function of hypaxial muscles in cattle?",
     "Hypaxial muscles (psoas, external and internal obliques, rectus abdominis) support the ventral trunk, flex the vertebral column, and support the abdominal viscera."),
    ("What bones form the bovine forelimb?",
     "The bovine forelimb (thoracic limb) consists of the scapula, humerus, radius, ulna, carpals, metacarpals (fused MC3+MC4 cannon bone), and two phalanges per digit (P1, P2, P3)."),
    ("What is bovine body condition scoring?",
     "Body condition scoring (BCS) is a system for evaluating subcutaneous fat and muscle cover in cattle, scored 1-5 (thin to obese) by palpating and visually assessing the loin, ribs, and rump. BCS 2.5-3.5 is ideal for dairy cows at calving."),
    ("What is the function of the bovine scapula?",
     "The scapula (shoulder blade) is a flat bone forming the shoulder joint with the humerus (glenohumeral joint). It anchors muscles of the forelimb including the supraspinatus, infraspinatus, subscapularis, and deltoid."),
    ("What joint connects the femur to the tibia in cattle?",
     "The stifle joint (equivalent to the human knee) connects the femur to the tibia. It contains medial and lateral menisci, cruciate ligaments, and collateral ligaments. The patella (kneecap) articulates on the cranial surface of the femur."),
    ("What is the fetlock joint in cattle?",
     "The fetlock joint is the metacarpophalangeal (or metatarsophalangeal) joint—the articulation between the cannon bone (MC3+MC4) and the proximal phalanx (P1). It is a high-motion joint prone to injury and degenerative joint disease."),
    # ── Digestive system ──────────────────────────────────────────────────────
    ("What are the four compartments of the bovine stomach?",
     "The four compartments of the bovine stomach are: rumen (largest, fermentation vat), reticulum (honeycomb, hardware trap), omasum (many-leafed, water absorption), and abomasum (true glandular stomach, equivalent to monogastric stomach)."),
    ("What is the function of the rumen?",
     "The rumen is the largest stomach compartment, holding 100-200 litres in adult cattle. It is an anaerobic fermentation vat where microorganisms (bacteria, protozoa, fungi) break down cellulose and other plant polysaccharides into volatile fatty acids (VFAs: acetate, propionate, butyrate) used as the primary energy source."),
    ("What is the function of the reticulum?",
     "The reticulum (honeycomb stomach) is the smallest forestomach compartment. It traps ingested hardware (nails, wire) due to its reticular groove and honeycomb mucosal surface. It works with the rumen in eructation and regurgitation during rumination."),
    ("What is the function of the omasum?",
     "The omasum (manyplies or psalterium) consists of many muscular leaves (laminae) that absorb water, sodium, and phosphorus from ingesta passing from the rumen/reticulum to the abomasum. It reduces fluid volume by up to 60%."),
    ("What is the abomasum?",
     "The abomasum is the true glandular stomach of ruminants, equivalent to the monogastric stomach. It secretes hydrochloric acid, pepsin, and rennet (chymosin). It is the site of true enzymatic digestion and can be displaced (left or right displacement of the abomasum, LDA or RDA) causing production loss."),
    ("What is rumination in cattle?",
     "Rumination is the process by which cattle regurgitate rumen contents (cud), re-chew it to reduce particle size, and re-swallow it. Cattle spend 6-10 hours per day ruminating. Cessation of rumination is an early indicator of systemic illness or rumen dysfunction."),
    ("What volatile fatty acids are produced in the bovine rumen?",
     "The main volatile fatty acids produced in the rumen are acetate (50-70%), propionate (15-20%), and butyrate (10-15%). Acetate is used for fat synthesis; propionate is gluconeogenic; butyrate fuels the rumen epithelium and is converted to beta-hydroxybutyrate in the rumen wall."),
    ("What is the bovine dental formula?",
     "The permanent bovine dental formula is I0/3, C0/1, P3/3, M3/3 = 32 teeth. Cattle have no upper incisors; they use a dental pad (hardened gum) with lower incisors to crop grass. The dental eruption pattern is used for age estimation."),
    ("What is the reticular groove reflex?",
     "The reticular groove (esophageal groove) is a muscular groove running from the cardia of the rumen to the omasum. In young ruminants, the groove closes reflexively during suckling to channel milk directly to the abomasum, bypassing the rumen/reticulum. This reflex diminishes with age but can be stimulated by certain salt solutions."),
    ("How long is the bovine small intestine?",
     "The bovine small intestine is approximately 40 meters long, divided into duodenum (~1m), jejunum (~35m), and ileum (~4m). It is the primary site of nutrient absorption."),
    ("What is hardware disease in cattle?",
     "Hardware disease (traumatic reticuloperitonitis) occurs when cattle ingest sharp metallic objects (wire, nails) that accumulate in the reticulum and penetrate the reticulum wall during contraction, causing peritonitis, pericarditis, or abscess. Treatment includes magnet placement and surgical rumenotomy."),
    ("What is rumen acidosis?",
     "Rumen acidosis occurs when rapid fermentation of starch or sugar lowers rumen pH below 5.5. Lactic acid accumulates from Streptococcus bovis overgrowth. Acute acidosis causes rumen stasis, dehydration, and endotoxemia. Subacute rumen acidosis (SARA) causes reduced feed intake, lower milk fat, and laminitis."),
    ("What bacteria dominate bovine rumen fermentation?",
     "The bovine rumen contains 10^10-10^11 bacteria per mL. Key fibrolytic species include Fibrobacter succinogenes, Ruminococcus flavefaciens, and Ruminococcus albus. Starch-fermenting bacteria include Prevotella ruminicola and Selenomonas ruminantium. Methanogens (e.g., Methanobrevibacter ruminantium) produce methane from CO2 and H2."),
    # ── Cardiovascular ────────────────────────────────────────────────────────
    ("What is the normal heart rate of adult cattle?",
     "The normal resting heart rate of adult cattle is 48-84 beats per minute (bpm). Calves have higher rates: 80-120 bpm at birth, decreasing with age."),
    ("Where is the bovine heart located?",
     "The bovine heart is located in the mediastinum within the thorax, between the 3rd and 6th ribs, slightly left of center. The cardiac apex points cranioventrally toward the left sternal region at approximately the 5th rib."),
    ("What are the major branches of the bovine aorta?",
     "The bovine aorta gives off the coronary arteries, brachiocephalic trunk (to forelimbs and head), and continues as the descending aorta. The descending aorta branches into intercostal arteries, celiac artery (to stomach, spleen, liver), cranial mesenteric artery (to intestines), renal arteries, and iliac arteries."),
    ("What is bovine jugular vein thrombosis?",
     "Bovine jugular vein thrombosis is inflammation and clot formation in the jugular vein, most commonly caused by repeated perivascular injection of hypertonic solutions or irritating drugs. It can be unilateral or bilateral. Severe cases obstruct venous return from the head and cause edema."),
    ("What is the significance of the reticular vein in cattle?",
     "The portal vein system in cattle drains VFAs and other absorbed nutrients from the rumen/reticulum into the portal circulation to the liver. This direct portal delivery means the liver receives high concentrations of propionate (for gluconeogenesis) and other metabolites from rumen fermentation."),
    # ── Reproductive system ───────────────────────────────────────────────────
    ("What is the bovine estrous cycle length?",
     "The bovine estrous cycle is approximately 21 days (range 18-24 days). Estrus (standing heat) lasts 6-18 hours. Ovulation occurs approximately 24-32 hours after the onset of estrus."),
    ("What are the parts of the bovine female reproductive tract?",
     "The bovine female reproductive tract consists of two ovaries, two oviducts (fallopian tubes: infundibulum, ampulla, isthmus), a bicornuate uterus (two uterine horns, uterine body, cervix), vagina, vestibule, and vulva."),
    ("What is the bovine cervix?",
     "The bovine cervix is a fibrous, muscular organ with interlocking annular rings (typically 3-5 rings) that create a tight seal during pregnancy. It is the most difficult reproductive structure to traverse during artificial insemination, requiring a rectovaginal technique."),
    ("What is the placentome in cattle?",
     "The bovine placenta is cotyledonary (synepitheliochorial type). Placentomes are the attachment sites between the fetal cotyledons and the maternal caruncles. There are approximately 70-120 placentomes per pregnancy. The placentome is the unit of nutrient and gas exchange."),
    ("What hormones regulate the bovine estrous cycle?",
     "GnRH (from hypothalamus) → LH and FSH (from anterior pituitary) → follicle development and estradiol production (from ovary). Estradiol triggers LH surge → ovulation. Corpus luteum produces progesterone. Luteolysis via uterine PGF2α (prostaglandin F2-alpha) resets the cycle."),
    ("What is retained fetal membranes in cattle?",
     "Retained fetal membranes (RFM) is failure to expel the placenta within 12-24 hours of calving. Normal expulsion takes 2-6 hours. Predisposing factors include dystocia, hypocalcemia, twin pregnancy, and Brucella abortus infection. RFM increases the risk of metritis and decreases conception rates."),
    ("What is the bovine cervical mucus plug?",
     "During pregnancy, the bovine cervix produces a thick mucus plug that seals the cervical canal, preventing ascending uterine infection. Under estrogen influence at estrus or near parturition, the plug liquefies, producing clear mucus discharge."),
    ("What is the bovine ovarian follicle wave?",
     "Cattle have 2-3 ovarian follicular waves per estrous cycle. Each wave involves emergence, selection, and either regression or dominance of a follicle. The dominant follicle of the final wave ovulates. Waves are detectable by ultrasound and are the basis for CIDR-based synchronization protocols."),
    ("What is the corpus luteum in cattle?",
     "The corpus luteum (CL) is the progesterone-secreting endocrine structure that forms from the ruptured follicle after ovulation. It maintains pregnancy by suppressing GnRH/LH release. Luteolysis (CL regression) is triggered by uterine PGF2α on days 16-18 if the cow is not pregnant."),
    ("What is bovine brucellosis?",
     "Bovine brucellosis is a bacterial infection caused by Brucella abortus. It causes late-term abortion, retained placenta, and epididymitis in bulls. Transmission is via aborted fetal material, colostrum, and milk. It is a zoonosis transmissible to humans. Vaccination with RB51 and test-and-slaughter programs are key controls."),
    # ── Integumentary (skin, hoof) ────────────────────────────────────────────
    ("How thick is bovine skin?",
     "Bovine skin thickness varies by body region: 3-4 mm at the muzzle, 5-7 mm over the trunk, and 8-10 mm at the neck. The dermis provides mechanical strength and is the layer tanned for leather."),
    ("What type of sweat glands do cattle have?",
     "Cattle have apocrine sweat glands (unlike the eccrine glands found in primates). These glands are widely distributed and play an important role in evaporative cooling. Each gland consists of a coiled secretory portion in the deep dermis and a duct that opens into the hair follicle."),
    ("What is the bovine hair follicle arrangement?",
     "Bovine hair follicles are compound, with a central primary follicle surrounded by 3-8 secondary follicles, giving a secondary-to-primary ratio of approximately 5-8:1. Coat texture and insulation are determined by this ratio and fibre diameter (50-150 µm for body hair)."),
    ("What are the layers of bovine skin?",
     "Bovine skin consists of the epidermis (keratinised stratified squamous epithelium, 30-60 µm thick, with stratum basale, spinosum, granulosum, and corneum), the dermis (dense irregular connective tissue with collagen and elastin), and the hypodermis (subcutis, fat and loose connective tissue)."),
    ("What is white line disease in cattle?",
     "White line disease is separation and infection of the white line—the junction between the hoof wall and the sole. The white line is soft horn and is prone to impaction with debris and bacteria. Infection tracks up into the corium causing lameness. It is more common in the lateral claw of the hindfoot."),
    ("What is sole ulcer in cattle?",
     "A sole ulcer is a hemorrhage and necrosis of the corium (quick) at the typical site of the sole—the junction of the sole and the bulb of the heel, over the apex of the distal phalanx (P3). It is caused by excessive weight bearing, poor claw horn quality, and digital dermatitis. The lateral hindclaw is most affected."),
    ("What is digital dermatitis (Mortellaro disease)?",
     "Digital dermatitis is an infectious bovine lameness caused by Treponema spp. It presents as strawberry-like proliferative lesions, usually at the skin-horn junction of the heel. It is contagious, spread by wet, contaminated environments. Topical oxytetracycline is the first-line treatment."),
    ("What is the bovine corium?",
     "The corium (quick) is the highly vascular connective tissue that produces hoof horn. It supplies nutrients to the horn-producing epidermis. Damage to the corium (e.g., from laminitis or sole ulcer) impairs horn production and causes lameness. The corium is organized into the perioplic, coronary, laminar, sole, and bulbar coriums."),
    # ── Respiratory system ────────────────────────────────────────────────────
    ("What is the normal respiratory rate of cattle?",
     "The normal respiratory rate of adult cattle is 12-36 breaths per minute at rest. Calves breathe slightly faster (20-40 bpm). Panting (>80 bpm) indicates heat stress."),
    ("What is the bovine respiratory complex (BRD)?",
     "Bovine respiratory disease complex (shipping fever) is the most economically important disease of beef cattle. It involves viral pathogens (IBR/BHV-1, BVDV, BRSV, PI-3) that compromise mucosal immunity, followed by secondary bacterial pneumonia from Mannheimia haemolytica, Pasteurella multocida, or Histophilus somni."),
    ("How many lung lobes do cattle have?",
     "Cattle have a total of 8 lung lobes: the right lung has 4 lobes (cranial, middle, caudal, accessory) and the left lung has 3-4 lobes (cranial [with 2 parts], caudal). The right lung is larger. The accessory lobe is unique to the right side."),
    ("What is the nasal planum (muzzle) in cattle?",
     "The nasal planum is the hairless, glandular area at the nose tip of cattle. Serous secretion keeps it moist. Dryness of the muzzle is a clinical indicator of fever, dehydration, or systemic disease. The muzzle is used in nose printing as a unique biometric identifier."),
    ("What is IBR in cattle?",
     "Infectious bovine rhinotracheitis (IBR) is caused by bovine herpesvirus-1 (BHV-1). It causes respiratory signs (nasal discharge, fever, conjunctivitis), vulvovaginitis, balanoposthitis, and abortion. It is a major component of BRD. The virus establishes latency in trigeminal ganglia."),
    # ── Nervous system ────────────────────────────────────────────────────────
    ("What is the bovine spinal cord formula?",
     "The bovine spinal cord runs from the brainstem through the vertebral canal to approximately L3-L4, where the conus medullaris ends and the cauda equina continues. Spinal nerves are designated C1-C8, T1-T13, L1-L6, S1-S5, and coccygeal nerves."),
    ("What is the cranial nerve responsible for bovine jaw movement?",
     "The trigeminal nerve (cranial nerve V) provides motor innervation to the muscles of mastication (masseter, temporalis, pterygoids). Dysfunction causes dropped jaw, inability to close the mouth, and atrophy of masticatory muscles."),
    ("What causes circling disease in cattle?",
     "Listeriosis (caused by Listeria monocytogenes) is the most common cause of circling (unilateral neurological signs) in cattle. It causes a rhombencephalitis (brainstem encephalitis) with microabscesses. Signs include circling, facial nerve paralysis, dysphagia, and head pressing."),
    ("What is polioencephalomalacia in cattle?",
     "Polioencephalomalacia (PEM) is cerebrocortical necrosis caused by thiamine (vitamin B1) deficiency or sulfur toxicosis. Thiamine deficiency leads to impaired pyruvate metabolism and cerebral energy failure. Signs include blindness, head pressing, seizures, and opisthotonos. Treatment is IV thiamine."),
    ("What is the hypoglossal nerve in cattle?",
     "Cranial nerve XII (hypoglossal nerve) innervates the tongue muscles. Damage causes ipsilateral tongue deviation and atrophy. It is sometimes damaged during dehorning if the cornual nerve block is incorrectly placed, or by listeriosis."),
    # ── Mammary system ────────────────────────────────────────────────────────
    ("What is the anatomy of the bovine udder?",
     "The bovine udder consists of four glandular quarters: two cranial and two caudal. Each quarter has a separate teat canal and cistern and is drained by independent ducts—there is no communication between quarters. The udder is suspended by medial and lateral suspensory ligaments."),
    ("What are the layers of the bovine teat?",
     "The bovine teat wall consists of: outer skin (epidermis + dermis), smooth muscle layer (provides teat tone), teat cistern (stores milk), streak canal (teat orifice, 8-12 mm long, lined by stratified squamous epithelium), and Furstenberg's rosette (annular mucosal folds at the cistern-canal junction providing bacteriostatic defense)."),
    ("What is the streak canal of the bovine teat?",
     "The streak canal (teat orifice) is the narrow terminal duct of the teat, 8-12 mm long. It is lined by a waxy, bacteriostatic substance (keratin plug) between milkings that prevents bacterial ingress. Frequent milking, teat damage, or intramammary infusion can disrupt this barrier."),
    ("What is bovine mastitis?",
     "Mastitis is inflammation of the mammary gland, predominantly caused by bacterial infection. Common pathogens include Staphylococcus aureus (contagious), Streptococcus agalactiae (contagious), Streptococcus uberis (environmental), and Escherichia coli (environmental). Clinical signs range from subclinical (elevated SCC) to acute toxic mastitis with systemic illness."),
    ("What is somatic cell count (SCC) in bovine milk?",
     "Somatic cell count is a measure of milk quality, counting cells (mostly neutrophils, macrophages) shed into milk during infection. Normal SCC is <200,000 cells/mL. SCC >200,000 cells/mL indicates subclinical mastitis. Regulatory limits in most countries are <400,000-750,000 cells/mL for bulk tank milk."),
    ("What hormones stimulate milk letdown in cattle?",
     "Oxytocin released from the posterior pituitary stimulates contraction of myoepithelial cells surrounding alveoli, causing milk ejection. Prolactin maintains lactation. Suckling or milking stimulation triggers the neuroendocrine reflex. Stress (cortisol, adrenaline) can inhibit oxytocin release and block milk letdown."),
    # ── Nutrition and metabolism ───────────────────────────────────────────────
    ("What is ketosis in dairy cattle?",
     "Ketosis (acetonemia) is a metabolic disorder of early lactation caused by negative energy balance (NEB). When glucose demand exceeds supply, hepatic gluconeogenesis from propionate is insufficient, and fat mobilization produces ketone bodies (beta-hydroxybutyrate, acetoacetate, acetone) that accumulate in blood. Signs: decreased appetite, weight loss, reduced milk production, ketotic smell."),
    ("What is hypocalcemia (milk fever) in cattle?",
     "Milk fever (parturient paresis) is acute hypocalcemia at parturition when calcium demand for colostrum/milk production overwhelms calcium homeostasis. Serum calcium falls below 1.5 mmol/L. Signs progress from trembling and hypersensitivity (stage 1) to recumbency and bloat (stage 2) to coma (stage 3). Treat with IV calcium borogluconate."),
    ("What is the role of the bovine liver in energy metabolism?",
     "The bovine liver is central to energy metabolism: gluconeogenesis from propionate, amino acids, and glycerol maintains blood glucose; beta-oxidation of mobilized fatty acids produces ketone bodies; triglyceride re-esterification stores fat. In NEB, excessive NEFA influx can overwhelm triglyceride export, causing hepatic lipidosis (fatty liver)."),
    ("What minerals are critical for bovine hoof health?",
     "Key minerals for hoof health include zinc (keratinocyte proliferation, horn hardness), biotin (corium integrity, white line quality), copper (cross-linking of structural proteins), and selenium (antioxidant defense via glutathione peroxidase). Deficiencies predispose to white line disease, sole ulcers, and laminitis."),
    ("What is laminitis in cattle?",
     "Laminitis (pododermatitis diffusa) is inflammation of the hoof laminae (corium), causing reduced blood flow, impaired horn production, and displacement of the pedal bone (P3). It is caused by: rumen acidosis (histamine, endotoxin vasoconstriction), systemic inflammation, and metabolic disorders. Subclinical laminitis causes hemorrhagic sole and yellow discoloration; acute laminitis causes severe lameness."),
    ("What is grass tetany (hypomagnesemia) in cattle?",
     "Grass tetany is acute hypomagnesemia (<0.4 mmol/L serum Mg) occurring when cattle graze rapidly growing spring grass low in magnesium. Unlike calcium, magnesium cannot be mobilized from bone rapidly. Signs: muscle tremors, hypersensitivity, convulsions, sudden death. Treat with IV magnesium sulfate; prevent with supplemental Mg in feed or pasture topdressing."),
    # ── Comparative and physiological ────────────────────────────────────────
    ("How does bovine vision differ from human vision?",
     "Cattle are dichromats (two types of cone photoreceptors), perceiving primarily blue and yellow-green wavelengths but not red. Their horizontal pupil provides approximately 330° panoramic vision with a binocular field of only 25-30° ahead, favoring predator detection over depth perception. The tapetum lucidum enables effective dim-light vision."),
    ("What is the bovine thermoregulatory neutral zone?",
     "The thermal neutral zone (TNZ) for cattle is approximately 5-25°C (41-77°F). Below 5°C, metabolic rate increases to maintain core temperature. Above 25°C, heat dissipation via panting, sweating, and vasodilation increases. Heat stress index (THI >72) significantly reduces feed intake and milk production."),
    ("What is the bovine reticulorumen pH?",
     "Healthy reticulorumen pH is 6.0-7.0. On roughage-based diets, pH is typically 6.5-7.0. On high-concentrate diets, pH may drop to 5.5-6.0 (subacute rumen acidosis) or below 5.5 (acute acidosis). Rumen buffers include bicarbonate in saliva (~180 L/day in cattle) and ammonium bicarbonate from protein fermentation."),
    ("How much saliva do cattle produce daily?",
     "Cattle produce approximately 100-200 litres of saliva per day (averaging 150 L). Saliva contains bicarbonate and phosphate buffers that neutralise VFAs in the rumen, maintaining rumen pH. Salivation is stimulated by rumination and roughage consumption."),
    ("What is the ruminant nitrogen cycle?",
     "Rumen bacteria convert feed protein and non-protein nitrogen (NPN, e.g., urea) to ammonia (NH3). Ammonia is used for microbial protein synthesis. Excess ammonia is absorbed into blood, converted to urea by the liver, and either excreted in urine or recycled back to the rumen via saliva and rumen wall diffusion (urea recycling)."),
    ("What is the bovine normal body temperature?",
     "The normal bovine rectal temperature is 38.0-39.5°C (100.4-103.1°F). Values above 39.5°C indicate fever; above 41°C indicates severe hyperthermia. Calves have slightly higher normal temperatures (38.5-40°C). Evening temperatures are typically 0.5°C higher than morning temperatures."),
    ("What is the bovine reticular groove?",
     "The reticular groove (oesophageal groove) is a muscular trough running from the cardia to the omasal orifice. In suckling calves, it closes reflexively to channel milk directly to the abomasum, bypassing the rumen and reticulum. The reflex is triggered by copper sulfate or sodium bicarbonate in some treatment protocols."),
    ("How does the bovine kidney differ from the human kidney?",
     "The bovine kidney is multilobular (reniculate) with 16-25 renal lobes visible on the surface, giving it a lobulated appearance. This contrasts with the smooth unipapillary kidney of humans. Each lobe has its own pyramid and papilla. Bovine kidneys are retroperitoneal, with the right kidney near the liver and the left kidney more mobile (floating kidney)."),
    # ── Anatomy identification ────────────────────────────────────────────────
    ("Where is the abomasum located in a healthy standing cow?",
     "In a healthy cow, the abomasum is located on the right side of the abdomen, ventral to the rumen, in the right paramedian region from approximately the 9th to 12th rib. Left displacement of the abomasum (LDA) moves it under the rumen on the left side; right displacement (RDA) causes it to rotate clockwise on the right side."),
    ("Where are bovine lymph nodes concentrated?",
     "Major bovine lymph node groups include: prescapular (superficial cervical), prefemoral (subiliac), supramammary, mesenteric (internal, associated with the GI tract), mediastinal, and popliteal nodes. The supramammary lymph nodes drain the udder and are enlarged in mastitis or udder lymphoma."),
    ("What is the bovine atlas vertebra?",
     "The atlas (C1) is the first cervical vertebra. It has no vertebral body—instead it forms a ring that articulates with the occipital condyles cranially (atlantooccipital joint, nodding) and with the axis (C2) caudally (atlantoaxial joint, rotation). The alar ligaments and transverse ligament stabilise the joint."),
    ("What is the sacroiliac joint in cattle?",
     "The sacroiliac joint connects the ilium of the pelvis to the sacrum. It is a synovial joint reinforced by dorsal and ventral sacroiliac ligaments. It transmits hindlimb propulsion forces to the spine. Sacroiliac subluxation (hip drop) causes asymmetric tuber coxae, a common cause of pelvic injury in cattle."),
    ("What structures form the bovine carpus?",
     "The bovine carpus (knee joint equivalent) consists of two rows of carpal bones: proximal row (radial, intermediate, ulnar, accessory carpals) and distal row (2nd+3rd fused carpals, 4th carpal). The radiocarpal, middle carpal, and carpometacarpal joints work as a unit during flexion and extension."),
    # ── Clinical and diagnostic ───────────────────────────────────────────────
    ("How is bovine age estimated from teeth?",
     "Bovine age is estimated from permanent incisor eruption: 1.5 years—2 permanent central incisors; 2.5 years—4 permanent incisors; 3.5 years—6 permanent incisors; 4.5 years—8 permanent incisors (full mouth). After 5 years, wear pattern and tooth shape are used for further estimation."),
    ("What is the normal rumen motility rate in cattle?",
     "The healthy rumen contracts 1-3 times per minute (approximately 2 contraction sequences per 2-minute period). Each sequence includes reticular contraction (biphasic) followed by ruminal contraction. Rumen motility is assessed by auscultation over the left paralumbar fossa and is reduced in rumen atony, acidosis, or systemic illness."),
    ("What is the ping test for abomasal displacement?",
     "The ping test (simultaneous auscultation and percussion) detects gas pockets in a displaced organ. A high-pitched metallic 'ping' over the right flank (9th-12th rib area) suggests right displacement of the abomasum or cecal dilation. A ping over the left flank indicates left displaced abomasum (LDA). The sound is caused by simultaneous flicking of the stethoscope while listening."),
    ("What is bovine recumbency?",
     "Bovine recumbency (downer cow syndrome) is inability to rise despite being conscious. Causes include hypocalcemia (milk fever), hypokalemia, nerve damage (obturator/sciatic nerve compression during dystocia), musculoskeletal injury, and metabolic disorders. Long-term recumbency causes pressure myopathy and ischemia. Prognosis worsens with duration."),
    ("What is the bovine withers height?",
     "The withers is the highest point of the bovine back, at the junction of the neck and back over the 4th-6th thoracic vertebrae and their dorsal spinous processes. Withers height is a standard body measurement for breed characterisation, growth monitoring, and body weight estimation."),
    ("What does an elevated SCC indicate in a bovine milk sample?",
     "An elevated somatic cell count (SCC >200,000 cells/mL) in milk indicates subclinical mastitis—intramammary infection causing neutrophil influx. The higher the SCC, the more severe the infection and milk quality impact. Individual cow SCC >1 million indicates severe mastitis with significant production loss."),
    ("What is bovine dermatophilosis?",
     "Dermatophilosis (rain scald, rain rot) is a bacterial skin infection caused by Dermatophilus congolensis, an actinomycete. It causes proliferative, matted, scab-forming lesions on the back, neck, and legs, particularly under wet conditions. Lesions contain the characteristic 'railroad track' filaments microscopically. Treatment includes dry conditions and penicillin-streptomycin."),
    ("What is the significance of the bovine umbilicus at birth?",
     "The umbilicus at birth contains the umbilical vein (carries oxygenated blood from placenta to liver), two umbilical arteries (return deoxygenated blood), and the urachus (fetal urine drainage). After birth, the stump dries and falls off in 1-2 weeks. Navel ill (omphalitis) from bacterial infection causes joint ill and systemic sepsis in calves."),
    ("What is bovine aortic rupture?",
     "Aortic rupture is a catastrophic condition where the aorta tears, typically at the root (ascending aorta or aortic arch). It is rare but can occur in adult bulls from extreme exertion or trauma. It causes acute cardiovascular collapse and rapid death from hemorrhage into the pericardium (cardiac tamponade) or thorax."),
    ("What causes bovine bloat?",
     "Bovine bloat is abnormal accumulation of gas in the rumen. Frothy bloat (primary) occurs on rapidly fermented pasture (clover, alfalfa) where stable foam traps gas and prevents eructation. Free-gas bloat (secondary) occurs from esophageal obstruction or vagus nerve damage preventing rumen gas escape. Treatment: trocarisation, anti-foaming agents (poloxalene, simethicone), or passing a stomach tube."),
    # ── Histology ──────────────────────────────────────────────────────────────
    ("What type of epithelium lines the bovine rumen?",
     "The rumen is lined by non-glandular stratified squamous epithelium with papillae that increase surface area for VFA absorption. The epithelium has four layers: stratum basale, stratum spinosum, stratum granulosum, and stratum corneum (keratinized). Rumen epithelial cells metabolize butyrate and short-chain fatty acids."),
    ("What type of muscle is the bovine heart wall composed of?",
     "The bovine heart wall consists of cardiac muscle (myocardium). Cardiac myocytes are branched, striated, uninucleate cells connected by intercalated discs (gap junctions and desmosomes). The gap junctions allow electrical coupling for coordinated contraction."),
    ("What histological structure characterizes the bovine reticulum?",
     "The reticulum mucosa has a distinctive honeycomb pattern formed by ridges of smooth muscle covered by non-glandular stratified squamous epithelium. The cells form hexagonal compartments of 1-2 cm diameter. This structure gives the reticulum its alternative name 'honeycomb stomach'."),
    ("What are the histological layers of the bovine small intestine?",
     "The bovine small intestinal wall has four layers: mucosa (columnar epithelium with villi and crypts of Lieberkühn, goblet cells, enterocytes), submucosa (Brunner's glands in duodenum, blood and lymph vessels), muscularis externa (inner circular and outer longitudinal smooth muscle), and serosa (visceral peritoneum)."),
    ("What are Peyer's patches in cattle?",
     "Peyer's patches are aggregated lymphoid nodules in the ileal submucosa and mucosa. In cattle, the terminal ileum (30-50 cm segment) contains a large continuous Peyer's patch (100-120 cm in calves) that is a major site of B-cell development and antigen sampling via M cells. It involutes in adult cattle."),
    ("What is the histological structure of bovine liver lobules?",
     "Bovine liver lobules (classical lobules) are hexagonal units organized around a central vein. Portal triads at the periphery contain portal vein branches, hepatic artery branches, and bile ducts. Hepatocytes radiate as plates from the central vein. The periportal zone (zone 1) is richest in oxygen and is most active in gluconeogenesis; the centrilobular zone (zone 3) is most susceptible to hypoxia and fatty change."),
    ("What connective tissue surrounds the bovine liver lobules?",
     "Unlike the pig (where lobules are clearly demarcated by connective tissue septa), bovine liver lobules are poorly delineated. Thin periportal connective tissue exists but does not create clear lobular boundaries. However, bovine liver lobules are still identifiable by the portal triad-to-central vein architecture."),
    ("What are the abomasal glands in cattle?",
     "The abomasal mucosa contains fundic (gastric) glands in the fundus and body regions, secreting pepsinogen (chief cells), hydrochloric acid (parietal cells), and mucus (mucous neck cells). The pyloric region has pyloric glands secreting mucus and gastrin (G cells). Abomasal lymphoplasmacytic infiltration occurs in parasitic infection (Ostertagia ostertagi)."),
    # ── Parasitology (anatomical context) ────────────────────────────────────
    ("Where does Ostertagia ostertagi reside in cattle?",
     "Ostertagia ostertagi (brown stomach worm) inhabits the abomasum. Larvae invade abomasal glands, disrupting pepsinogen secretion and raising abomasal pH. This leads to reduced protein digestion, hypoproteinemia, and bottle jaw (submandibular edema) in heavy infections. It is the most economically significant nematode of cattle in temperate climates."),
    ("What is the predilection site of Fasciola hepatica in cattle?",
     "Fasciola hepatica (liver fluke) migrates through the liver parenchyma as immature flukes, causing acute hepatitis and hemorrhage. Adult flukes reside in the bile ducts, causing chronic biliary fibrosis, cholangitis, and bile duct hyperplasia. Fluke eggs are passed in bile into the feces."),
    ("Where do warble fly larvae (Hypoderma spp.) migrate in cattle?",
     "Hypoderma bovis and H. lineatum larvae hatch from eggs on the hair, penetrate the skin, and migrate through subcutaneous and intramuscular tissues. H. lineatum migrates via the esophageal submucosa (causing 'licking disease'); H. bovis migrates near the spinal canal. Both species overwinter in warble cysts under the dorsal skin before pupating."),
    # ── Endocrinology ─────────────────────────────────────────────────────────
    ("What gland produces bovine insulin?",
     "Insulin is produced by beta cells of the islets of Langerhans in the pancreas. Bovine insulin was the first commercially isolated animal insulin and is structurally very close to human insulin (differing at only 3 positions). The pancreas also produces glucagon (alpha cells), somatostatin (delta cells), and pancreatic polypeptide (PP cells)."),
    ("What is the role of the bovine adrenal gland?",
     "The bovine adrenal gland has a cortex (zona glomerulosa: aldosterone; zona fasciculata: cortisol; zona reticularis: androgens) and a medulla (epinephrine, norepinephrine). Cortisol mediates the stress response (HPA axis) and is elevated at parturition. Aldosterone regulates sodium and potassium balance."),
    ("What triggers parturition in cattle?",
     "Parturition is triggered by fetal hypothalamic maturation: the fetal HPA axis activates, increasing fetal ACTH and cortisol. Fetal cortisol redirects placental steroidogenesis from progesterone to estrogens. The estrogen/progesterone ratio shift sensitises the uterus to oxytocin, increases PGF2α production, and causes cervical ripening and uterine contractions."),
    ("What is bovine somatotropin (BST)?",
     "Bovine somatotropin (BST, rBST) is growth hormone produced by the anterior pituitary. It is anabolic—promoting protein synthesis, lipolysis, and milk production. Recombinant BST (rbST) is approved in some countries to increase milk production by 10-15% in dairy cattle. It acts via IGF-1 (insulin-like growth factor 1) in peripheral tissues."),
    # ── Neonatal/Calf anatomy ──────────────────────────────────────────────────
    ("What is colostrum and why is it critical for calves?",
     "Colostrum is the first secretion of the mammary gland after parturition, rich in immunoglobulins (IgG1, IgG2, IgM, IgA), growth factors, and nutrients. Calves are born agammaglobulinemic (no transplacental antibody transfer in cattle due to synepitheliochorial placentation). Colostrum is the sole source of passive immunity. IgG absorption via intestinal enterocytes (gut closure) is complete within 24-36 hours of birth."),
    ("What is scours in calves?",
     "Calf scours (neonatal diarrhea) is the most common cause of calf mortality in the first 4 weeks of life. Major pathogens include: Cryptosporidium parvum (7-21 days), enterotoxigenic E. coli K99 (ETEC, <5 days), rotavirus and coronavirus (5-14 days), and Salmonella. Treatment: oral electrolyte replacement, bicarbonate for acidosis; antibiotics for bacterial causes."),
    ("What is the bovine thymus?",
     "The bovine thymus is a bilobed lymphoid organ in the thoracic inlet and neck region, present and active in calves and young animals. It is the site of T-lymphocyte maturation and positive/negative selection. The thymus involutes with age, replaced by adipose tissue. It is large and visible in slaughtered veal calves."),
    ("What is navel ill in calves?",
     "Navel ill (omphalitis/umbilical abscess) is bacterial infection of the umbilical stump in calves, caused by Trueperella pyogenes, Fusobacterium necrophorum, E. coli, and Staphylococcus. Bacteria enter via the wet umbilical cord and can spread to the umbilical vein (to the liver) or umbilical arteries (to the bladder/urachus), causing systemic sepsis, joint ill (polyarthritis), and pneumonia."),
    # ── Anatomy of specific organs ────────────────────────────────────────────
    ("What is the bovine spleen?",
     "The bovine spleen is a flat, elongated organ attached to the rumen dorsal sac in the left cranial abdomen. It has a red pulp (blood filtration, storage of RBCs, platelet reservoir) and white pulp (lymphoid tissue with periarteriolar lymphoid sheaths). The spleen is a major site of hematopoiesis in embryonic life and erythrocyte destruction/recycling in adult life."),
    ("What is the bovine gallbladder?",
     "Cattle (unlike horses) have a gallbladder. It stores and concentrates bile produced by the liver. Bile is released into the duodenum via the common bile duct. Bovine bile is green and alkaline, containing bile acids, cholesterol, and biliverdin (green pigment, unlike bilirubin in other species). Cholelithiasis (gallstones) is rare in cattle."),
    ("What is the bovine cecum?",
     "The bovine cecum is a large blind-ended pouch at the ileo-ceco-colic junction. It is an important site of microbial fermentation of cellulose in monogastrics, but in cattle it is relatively less significant than in hindgut fermenters like horses. Cecal dilation and torsion (cecal volvulus) can cause acute abdominal pain."),
    ("What is the bovine trachea?",
     "The bovine trachea is a cartilaginous tube consisting of incomplete (C-shaped, dorsally open) hyaline cartilage rings connected by annular ligaments. The dorsal trachealis muscle closes the tracheal ring posteriorly. In cattle, the right principal bronchus diverges from the trachea before the carina to form the tracheal bronchus (the first bronchus supplying the right cranial lung lobe)."),
    ("What is the bovine kidney multilobed structure called?",
     "The bovine kidney is reniculate (multilobed) with 16-25 renal lobes, each consisting of a cortex, a medullary pyramid, and a papilla (papillary duct) draining into a calyx. All calices drain into a common renal pelvis and then the ureter. The multilobed kidney has a greater surface area for filtration relative to size compared to unipapillary kidneys."),
    ("What are the bovine salivary glands?",
     "The major bovine salivary glands are the parotid (located behind the angle of the jaw, serous secretion), the mandibular/submaxillary (mixed sero-mucous, beneath the masseter), and the sublingual (mucous, in the floor of the mouth). The parotid is the largest and most active in cattle, contributing the majority of the 100-200 L of saliva produced daily."),
    ("What is the bovine rete mirabile?",
     "The bovine carotid rete mirabile is a network of small arteries that replaces the internal carotid artery in cattle (and other ruminants). It is located at the base of the brain in the cavernous sinus. The rete allows countercurrent heat exchange between warm arterial blood and cooled venous blood from the nasal passages, selectively cooling the brain below body temperature to protect against heat stress."),
    ("What is the significance of the bovine rumen epithelium?",
     "The rumen epithelium absorbs volatile fatty acids (VFAs) through passive diffusion. Butyrate is metabolized to ketone bodies in the epithelial cells (important for epithelial energy). Propionate passes to the portal blood for hepatic gluconeogenesis. Acetate enters peripheral circulation for lipogenesis. Papillary development (stimulated by butyrate) increases absorption surface area in calves transitioning to solid feed."),
    ("What is the bovine thyroid gland?",
     "The bovine thyroid gland consists of two lobes connected by an isthmus, located ventrolateral to the trachea at the first few tracheal rings. It produces thyroxine (T4) and triiodothyronine (T3) from thyroglobulin stored in follicular colloid. Thyroid hormones regulate metabolic rate, growth, and development. Goiter (enlarged thyroid) in calves can result from iodine deficiency or dietary goitrogens."),
    # ── Anatomy questions at depth ─────────────────────────────────────────────
    ("Describe the zonation of the bovine liver acinus.",
     "In the liver acinus model, zone 1 (periportal) is closest to the portal triad, receives the most oxygen and nutrients, and is most active in oxidative metabolism, gluconeogenesis, and beta-oxidation. Zone 3 (centrilobular) is furthest from the portal triad, receives the least oxygen, and is most susceptible to hypoxic injury and lipid accumulation (fatty liver). Zone 2 is intermediate. In bovine hepatic lipidosis, fat accumulates predominantly in zone 3."),
    ("What is the difference between the bovine medial and lateral saphenous veins?",
     "The medial saphenous vein runs on the medial surface of the tibia and is commonly used for venipuncture and catheterization in calves. The lateral saphenous vein runs on the lateral aspect of the hindlimb. In adult cattle, the tail vein (coccygeal vein, ventral midline) and the jugular vein are the preferred sites for blood sampling."),
    ("What is the significance of the bovine nasal turbinate anatomy?",
     "The bovine nasal cavity contains three main turbinate bones (dorsal, middle, ventral nasal conchae), which warm and humidify inspired air, filter particles, and provide a large surface area for olfaction. The olfactory mucosa covers the ethmoturbinates in the caudal nasal cavity. Disease (bovine rhinitis) disrupts this function, impairing olfaction and predisposing to BRD."),
    ("Where is the parotid lymph node in cattle?",
     "The parotid (parotid superficial) lymph node is located rostral to the parotid salivary gland at the angle of the jaw. It drains the face, ear, and parotid gland. Enlargement indicates local infection or neoplasia. It is part of the superficial cervical lymph node group assessed during ante-mortem inspection."),
    ("What is the bovine cornual nerve?",
     "The cornual nerve is the branch of the frontal nerve (CN V1, ophthalmic branch of trigeminal) that innervates the horn and horn base in cattle. Cornual nerve block with local anesthetic (2% lidocaine) provides analgesia for dehorning. The nerve runs in a groove on the frontal bone from the medial canthus of the eye toward the horn base."),
]


def build_bovine_qa(out_dir: Path, node: str) -> list[dict]:
    """Ingest embedded bovine anatomy Q&A pairs via /qa/ingest and return item list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f'http://{node}'
    items = []

    # Batch into groups of 20 for the ingest API
    batch_size = 20
    total_posted = 0
    for i in range(0, len(BOVINE_QA_PAIRS), batch_size):
        batch = BOVINE_QA_PAIRS[i:i + batch_size]
        candidates = [
            {
                'qa_id': f'bovine_qa_{i + j:04d}',
                'question': q,
                'answer': a,
                'confidence': 0.95,
            }
            for j, (q, a) in enumerate(batch)
        ]
        payload = json.dumps({'candidates': candidates}).encode()
        req = urllib.request.Request(
            f'{base}/qa/ingest',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read())
            n = result.get('ingested', len(batch))
            total_posted += n
        except Exception as e:
            print(f'  [WARN] qa/ingest batch {i//batch_size}: {e}')

        for j, (q, a) in enumerate(batch):
            items.append({
                'stage': 6,
                'type': 'qa_pair',
                'source': 'embedded_bovine_qa',
                'title': q[:80],
                'text': f'Q: {q}\nA: {a}',
                'modality': 'text',
                'tags': ['qa', 'bovine', 'anatomy', 'chat'],
            })
        time.sleep(0.1)

    print(f'  Ingested {total_posted} Q&A pairs via /qa/ingest ({len(items)} items total)')
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


# ── Stage 7: Cross-modal training ────────────────────────────────────────────
# Pairs each cow video frame with an anatomical text description in a single
# /media/train call.  This creates Hebbian connections between visual patterns
# and bovine anatomy text concepts so the fabric outputs semantic labels
# (e.g. "bovine:leg", "bovine:spine") when shown new cow images.

CROSS_MODAL_CONTEXTS = [
    "Holstein dairy cow bovine anatomy legs spine neck head tail udder",
    "Bovine locomotion musculoskeletal system limbs hoof stride gait",
    "Dairy cow body condition dorsal spine ribs pelvis bovine conformation",
    "Bovine thorax abdomen rumen reticulum digestive system flank",
    "Cattle behavior grazing standing walking bovine ethology pasture",
    "Holstein Friesian breed black white coat pattern udder teat milking",
    "Bovine cervical thoracic lumbar sacral vertebrae spine atlas axis",
    "Cow face eye horn ear muzzle nasal head anatomy bovine skull",
    "Bovine fetlock pastern coronary hoof coffin bone digital anatomy",
    "Bovine shoulder elbow carpal metacarpal front limb forelimb",
    "Bovine hip stifle hock metatarsal rear limb hindlimb anatomy",
    "Dairy cattle poll forehead brow occipital bovine cranium",
]


def build_cross_modal(frames_dir: Path, videos_dir: Path, train_dir: Path,
                      node: str, max_frames: int = 300) -> list:
    """Stage 7: Cross-modal Hebbian training — image + anatomical text paired.

    Samples up to max_frames from the Stage 2 extracted frames and trains
    each alongside a text description derived from the source video title and
    rotating anatomical context strings.  This bridges the visual and text
    domains so /neuro/snapshot returns bovine anatomy labels during inference.
    """
    out_dir = train_dir / 'stage7_crossmodal'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build video title map
    title_map: dict = {}
    for info_file in videos_dir.glob('*.info.json'):
        try:
            d = json.loads(info_file.read_bytes())
            title_map[info_file.stem] = d.get('title', '')
        except Exception:
            pass

    all_frames = sorted(frames_dir.glob('*.jpg'))
    if not all_frames:
        print(f'  [WARN] No frames found in {frames_dir}')
        return []

    step    = max(1, len(all_frames) // max_frames)
    sampled = all_frames[::step][:max_frames]
    base    = f'http://{node}'
    items   = []
    ok      = 0

    print(f'  Cross-modal: {len(sampled)} frame-text pairs (from {len(all_frames)} frames)...')

    for i, fp in enumerate(sampled):
        video_id = fp.stem.split('_')[0]
        title    = title_map.get(video_id, 'Bovine dairy cow CC video')
        ctx      = CROSS_MODAL_CONTEXTS[i % len(CROSS_MODAL_CONTEXTS)]
        text     = f'{title}. {ctx}.'

        try:
            with open(fp, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
        except Exception:
            continue

        body = json.dumps({'modality': 'image', 'data_b64': b64, 'text': text}).encode()
        req  = urllib.request.Request(
            f'{base}/media/train', data=body,
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=12) as r:
                json.loads(r.read())
            ok += 1
        except Exception as e:
            print(f'  [WARN] frame {i}: {e}')

        items.append({
            'stage': 7, 'type': 'cross_modal', 'source': 'stage2_video',
            'title': f'{title[:60]} [{fp.stem}]', 'text': text,
            'modality': 'multimodal',
            'tags': ['cross_modal', 'bovine', 'video', 'anatomy'],
        })
        time.sleep(0.05)

    print(f'  Cross-modal trained: {ok}/{len(sampled)} pairs')
    return items


# ── Stage 8: Multi-angle cow gallery ─────────────────────────────────────────
# Downloads CC-licensed cow images from Wikimedia Commons covering front, rear,
# side, and 3/4 views — giving the fabric multi-perspective bovine appearance.

WIKIMEDIA_COW_CATEGORIES = [
    'Holstein_Friesian_cattle',
    'Dairy_cattle',
    'Bos_taurus',
    'Cows_standing',
    'Cows_lying_down',
    'Cows_grazing',
    'Cows_walking',
    'Cattle_heads',
    'Cattle_hooves',
    'Cattle_udders',
    'Bovine_anatomy',
]

# View-direction keywords parsed from Wikimedia image titles/descriptions
VIEW_KEYWORDS = {
    'front':  ['front', 'head-on', 'facing', 'frontal', 'anterior', 'face'],
    'rear':   ['rear', 'back', 'posterior', 'behind', 'hindquarters', 'rump'],
    'side':   ['side', 'lateral', 'profile', 'flank'],
    'top':    ['top', 'dorsal', 'above', 'aerial', 'overhead'],
    'three_quarter': ['three quarter', '3/4', 'oblique', 'angled'],
}

VIEW_ANATOMY_TEXT = {
    'front':  'Bovine anterior view face muzzle nostrils forehead poll horns ears eyes front legs',
    'rear':   'Bovine posterior view rump tail pin bones hip hooks hindquarters rear legs hoof',
    'side':   'Holstein dairy cow lateral profile left right flank ribs barrel spine udder legs',
    'top':    'Bovine dorsal view spine dorsal ridge topline loin rump shoulder withers',
    'three_quarter': 'Bovine three-quarter oblique view body depth depth chest barrel angle',
    'unknown': 'Holstein Friesian dairy cow bovine anatomy legs spine neck head tail udder body',
}


def _detect_view(title: str) -> str:
    t = title.lower()
    for view, keywords in VIEW_KEYWORDS.items():
        if any(k in t for k in keywords):
            return view
    return 'unknown'


def build_multiview_gallery(out_dir: Path, node: str,
                             categories: list = None,
                             max_per_cat: int = 30,
                             max_total: int = 300) -> list:
    """Stage 8: Download multi-angle cow images from Wikimedia Commons and
    cross-modal train each with view-specific anatomical text.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if categories is None:
        categories = WIKIMEDIA_COW_CATEGORIES

    API = 'https://commons.wikimedia.org/w/api.php'
    UA  = 'W1z4rDV1510n-DatasetBuilder/1.0 (https://github.com/C4rr13rX/W1z4rDV1510n; adamedsall@gmail.com)'
    base = f'http://{node}'
    items = []
    ok    = 0

    for cat in tqdm(categories, desc='Wikimedia cow gallery'):
        if len(items) >= max_total:
            break
        params = {
            'action': 'query', 'list': 'categorymembers',
            'cmtitle': f'Category:{cat}', 'cmtype': 'file',
            'cmlimit': str(max_per_cat), 'format': 'json',
        }
        url = API + '?' + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read())
            members = data.get('query', {}).get('categorymembers', [])
        except Exception as e:
            print(f'  [WARN] Wikimedia category {cat}: {e}')
            continue

        for m in members:
            if len(items) >= max_total:
                break
            title = m.get('title', '')
            ext = title.lower().rsplit('.', 1)[-1] if '.' in title else ''
            if ext not in ('jpg', 'jpeg', 'png'):
                continue

            # Resolve thumbnail URL via imageinfo API
            img_params = {
                'action': 'query', 'titles': title,
                'prop': 'imageinfo', 'iiprop': 'url',
                'iiurlwidth': '800', 'format': 'json',
            }
            img_url = API + '?' + urllib.parse.urlencode(img_params)
            try:
                img_req = urllib.request.Request(img_url, headers={'User-Agent': UA})
                with urllib.request.urlopen(img_req, timeout=12) as r:
                    idata = json.loads(r.read())
                pages = idata.get('query', {}).get('pages', {})
                dl_url = None
                for pid, page in pages.items():
                    ii = page.get('imageinfo', [{}])
                    dl_url = (ii[0].get('thumburl') or ii[0].get('url')) if ii else None
                if not dl_url:
                    continue
            except Exception:
                continue

            # Download image
            fname = out_dir / (hashlib.sha1(dl_url.encode()).hexdigest()[:12] + '.jpg')
            if not fname.exists():
                try:
                    dl_req = urllib.request.Request(dl_url, headers={'User-Agent': UA})
                    with urllib.request.urlopen(dl_req, timeout=20) as r:
                        raw = r.read()
                    fname.write_bytes(raw)
                except Exception:
                    continue

            # Cross-modal train: image + view-specific anatomical text
            view = _detect_view(title)
            text = f'{VIEW_ANATOMY_TEXT[view]}. Category: {cat.replace("_", " ")}.'
            try:
                with open(fname, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                body = json.dumps({'modality': 'image', 'data_b64': b64, 'text': text}).encode()
                train_req = urllib.request.Request(
                    f'{base}/media/train', data=body,
                    headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(train_req, timeout=12) as r:
                    json.loads(r.read())
                ok += 1
            except Exception as e:
                print(f'  [WARN] train {fname.name}: {e}')

            items.append({
                'stage': 8, 'type': 'image', 'source': 'wikimedia_commons',
                'title': title, 'category': cat, 'view': view,
                'file': str(fname), 'b64': b64,
                'modality': 'image',
                'tags': ['cow', 'multi_angle', view, cat.lower().replace('_', ' ')],
            })
            time.sleep(0.15)

    print(f'  Multi-angle gallery: {len(items)} images downloaded, {ok} trained')
    return items


# ── Stage 9: 3D Mesh Understanding ───────────────────────────────────────────
# Goal: teach the fabric to reconstruct 3D objects from 2D images — given an
# unseen image, identify objects, predict occluded sides, and generate mesh +
# UV map + OBJ topology.
#
# Training triplets for each object:
#   A. [rendered 3D view]  ↔  [mesh topology text + object semantics]
#   B. [UV layout image]   ↔  [UV unwrap description + mesh topology text]
#   C. [OBJ vertex text]   ↔  text-modality describing geometry precisely
#   D. [Wikipedia extract] ↔  object semantic understanding
#
# Hebbian chains learned:
#   photo_of_cow → bovine_labels → mesh_topology → UV_map_layout → OBJ_structure
#
# Scale: 10K+ training items across 65+ object categories.

MESH_OBJ_SOURCES = [
    # (raw_url, name, category, description_text)
    ('https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/spot.obj',
     'spot_cow', 'bovine_quadruped',
     'Spot the cow: CC0 bovine quadruped mesh by Keenan Crane. '
     'Holstein dairy cow 3D topology: 4 limbs with knee joints, barrel torso, '
     'neck, head, tail, udder. Quadruped standing pose. '
     'Surface landmarks: dorsal spine ridge, ribcage curvature, hip hooks, '
     'pin bones, stifle joint, hock, fetlock, pastern, coronary band, hoof capsule, '
     'shoulder point, elbow, knee, dewclaw, poll, withers, loin, rump, thurls.'),
    ('https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/horse.obj',
     'horse', 'equine_quadruped',
     'Horse 3D mesh: equine quadruped anatomy. Deep barrel chest, arched muscular '
     'neck, elongated cranium, 4 long limbs with cannon bones and hooves. '
     'Surface topology: withers, croup, loins, flanks, gaskin, fetlock, coronet. '
     'Bilateral symmetry along dorsal midline spine.'),
    ('https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj',
     'armadillo', 'armored_quadruped',
     'Armadillo 3D mesh: armored mammal quadruped. Compact torso with dorsal '
     'segmented osteoderms, 4 short robust limbs, conical tapering tail.'),
    ('https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/suzanne.obj',
     'primate_head', 'head_anatomy',
     'Suzanne primate head mesh: facial anatomy topology. Cranium volume, '
     'supraorbital ridges, nasal bone, zygomatic arch, mandible ramus, '
     'external ear pinnae, orbital cavities, temporal fossa.'),
    # Three.js example models — human body (biped generalization)
    ('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/obj/male02/male02.obj',
     'human_male', 'human_biped',
     'Human male body 3D mesh: bipedal anatomy in anatomical position. '
     'Axial skeleton: skull, spine (cervical 7, thoracic 12, lumbar 5), '
     'ribcage 12 pairs, pelvis. Appendicular: shoulder girdle, humerus, '
     'radius+ulna, 8 carpals, 5 metacarpals, 14 phalanges per hand; '
     'hip joint, femur, tibia+fibula, 7 tarsals, 5 metatarsals, 14 toe phalanges.'),
    ('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/obj/female02/female02.obj',
     'human_female', 'human_biped',
     'Human female body 3D mesh: bipedal anatomy. Same skeletal topology as male. '
     'Gynaecoid pelvis wider than android. '
     'Surface: deltoid, pectoralis, abdominis, gluteus, quadriceps, '
     'gastrocnemius landmarks visible as mesh contours.'),
]

# 65+ Sketchfab categories × up to 150 paginated models = ~10K training items.
# Each tuple: (search_query, semantic_context_text_appended_to_training_text)
SKETCHFAB_MESH_QUERIES = [
    # ── Bovine/livestock (primary targets) ──────────────────────────────────
    ('cow dairy',         'dairy cow bovine Holstein Friesian quadruped 3D mesh vertices faces '
                          'legs torso udder body spine ribs pelvis anatomical structure'),
    ('cattle bovine',     'cattle bovine livestock Bos taurus 3D mesh anatomy quadruped '
                          'musculoskeletal surface topology body landmarks'),
    ('cow skeleton',      'bovine skeletal anatomy 3D mesh bones spine ribs pelvis '
                          'skull femur tibia fibula scapula humerus radius ulna'),
    ('bovine anatomy',    'bovine anatomical 3D mesh musculoskeletal quadruped body '
                          'surface muscles tendons ligaments joints'),
    ('calf young cow',    'calf young bovine quadruped 3D mesh juvenile body proportions '
                          'long limbs large head relative to body size'),
    # ── Other livestock ──────────────────────────────────────────────────────
    ('horse equine',      'equine quadruped 3D mesh barrel chest arched neck limbs cannon hooves'),
    ('sheep ovine',       'ovine quadruped 3D mesh sheep wool fleece body legs head barrel'),
    ('pig porcine',       'porcine quadruped 3D mesh pig snout disc barrel torso legs jowl'),
    ('goat caprine',      'caprine quadruped 3D mesh goat horns beard dewlap legs hooves'),
    ('donkey ass',        'donkey equid quadruped 3D mesh long ears compact body hooves'),
    ('deer cervid',       'cervid quadruped 3D mesh deer antlers slender legs body spots'),
    ('alpaca llama',      'South American camelid quadruped 3D mesh long neck wool legs'),
    # ── Carnivores / pets ────────────────────────────────────────────────────
    ('dog canine',        'canine quadruped 3D mesh spine ribs shoulder hip stifle legs head muzzle'),
    ('cat feline',        'feline quadruped 3D mesh flexible spine lithe body retractile claws'),
    ('wolf lupine',       'wolf canid quadruped 3D mesh muscular body long limbs muzzle'),
    ('lion',              'lion big cat quadruped 3D mesh mane muscular body deep chest paws'),
    ('tiger',             'tiger feline quadruped 3D mesh muscular body striped pattern limbs'),
    ('bear',              'bear plantigrade quadruped 3D mesh massive body thick limbs claws'),
    ('fox',               'fox canid quadruped 3D mesh slender muzzle bushy tail pointed ears'),
    # ── Large mammals ────────────────────────────────────────────────────────
    ('elephant',          'elephant proboscid quadruped 3D mesh pillar limbs long trunk large ears'),
    ('rhinoceros',        'rhinoceros large mammal 3D mesh horn thick hide columnar limbs'),
    ('hippopotamus',      'hippo aquatic mammal 3D mesh massive barrel body short thick limbs'),
    ('giraffe',           'giraffe quadruped 3D mesh extremely long neck spotted coat long legs'),
    ('camel',             'camel quadruped 3D mesh dorsal hump long neck splayed hooves'),
    ('buffalo bison',     'bison buffalo quadruped 3D mesh prominent shoulder hump horns'),
    ('rhinoceros horns',  'rhinoceros 3D mesh detailed surface skin folds horn base anatomy'),
    # ── Primates ─────────────────────────────────────────────────────────────
    ('human body',        'human bipedal anatomy 3D mesh skeleton muscles bilateral symmetry S-spine'),
    ('human anatomy',     'human anatomical figure 3D mesh detailed surface muscles tendons'),
    ('human skeleton',    'human skeletal 3D mesh 206 bones vertebral column limb bones skull'),
    ('gorilla primate',   'gorilla great ape quadrumanous 3D mesh sagittal crest massive arms'),
    # ── Birds ────────────────────────────────────────────────────────────────
    ('eagle raptor',      'eagle raptor avian 3D mesh wingspan feathers beak talons hollow bones'),
    ('parrot bird',       'parrot psittacine avian 3D mesh curved beak zygodactyl feet wings'),
    ('owl nocturnal',     'owl strigiform avian 3D mesh large eyes facial disc wings talons'),
    ('dinosaur',          'theropod dinosaur biped 3D mesh large hind limbs vestigial arms'),
    ('prehistoric animal','prehistoric animal fossil reconstruction 3D mesh anatomy'),
    # ── Marine ───────────────────────────────────────────────────────────────
    ('fish aquatic',      'fish teleost aquatic vertebrate 3D mesh fusiform body fins scales lateral line'),
    ('shark',             'shark elasmobranch 3D mesh fusiform body pectoral dorsal caudal fins'),
    ('whale cetacean',    'cetacean marine mammal 3D mesh fusiform body pectoral flukes dorsal fin'),
    ('octopus cephalopod','octopus cephalopod mollusc 3D mesh 8 arms mantle no skeleton'),
    # ── Reptiles ─────────────────────────────────────────────────────────────
    ('crocodile alligator','crocodilian reptile 3D mesh osteoderms jaws legs long tail'),
    ('turtle tortoise',   'chelonian reptile 3D mesh domed carapace plastron beak limbs'),
    ('snake serpent',     'serpent snake 3D mesh elongated body 200+ vertebrae no limbs'),
    ('lizard gecko',      'lizard reptile 3D mesh 4 limbs lateral undulation long tail'),
    ('frog amphibian',    'frog anuran amphibian 3D mesh wide head large eyes long hind limbs'),
    # ── Human anatomy details ─────────────────────────────────────────────────
    ('skull cranium',     'skull cranium 3D mesh 22 bones frontal parietal temporal occipital'),
    ('spine vertebrae',   'vertebral column 3D mesh cervical thoracic lumbar sacral coccyx'),
    ('hand fingers',      'hand anatomy 3D mesh phalanges metacarpals carpals thumb opposition'),
    ('foot ankle',        'foot anatomy 3D mesh tarsal metatarsal phalanges arch plantar'),
    ('heart anatomy',     'cardiac heart 3D mesh 4 chambers ventricles atria aorta valves'),
    ('brain',             'brain 3D mesh frontal parietal temporal occipital cortex cerebellum'),
    ('muscle body',       'musculature 3D mesh superficial muscles origin insertion tendons'),
    ('eye ball',          'eyeball globe 3D mesh cornea iris pupil lens vitreous retina sclera'),
    ('ear anatomy',       'ear 3D mesh pinna helix antihelix tragus concha external canal'),
    # ── Vehicles ─────────────────────────────────────────────────────────────
    ('car automobile',    'automobile vehicle 3D mesh body panels wheels chassis suspension'),
    ('truck lorry',       'truck lorry vehicle 3D mesh cab chassis axles cargo body'),
    ('airplane',          'airplane fixed-wing aircraft 3D mesh wings fuselage tail rudder'),
    ('boat sailing',      'sailing boat watercraft 3D mesh hull deck mast keel rudder'),
    ('motorcycle',        'motorcycle 3D mesh tubular frame engine block wheels handlebars'),
    ('bicycle cycling',   'bicycle 3D mesh diamond frame fork wheels spokes handlebars saddle'),
    ('helicopter',        'helicopter rotorcraft 3D mesh main rotor blades tail rotor fuselage'),
    ('rocket space',      'rocket spacecraft 3D mesh nose cone fuel tank fins nozzle payload'),
    # ── Furniture / indoor ────────────────────────────────────────────────────
    ('chair furniture',   'chair furniture 3D mesh 4 legs seat back apron joints mortise'),
    ('table desk',        'table furniture 3D mesh flat surface 4 legs apron structural'),
    ('sofa couch',        'sofa upholstered furniture 3D mesh seat cushion arm back frame'),
    # ── Architecture ──────────────────────────────────────────────────────────
    ('house building',    'house residential building 3D mesh walls roof windows doors foundation'),
    ('medieval building', 'medieval architecture 3D mesh stone walls towers arch buttress'),
    ('bridge structure',  'bridge civil structure 3D mesh span deck piers cables tension'),
    # ── Nature ────────────────────────────────────────────────────────────────
    ('tree forest',       'tree plant 3D mesh trunk bark branches fork canopy leaves roots'),
    ('rock geological',   'rock stone geological 3D mesh irregular fractured surface facets'),
    ('crystal mineral',   'crystal mineral 3D mesh planar faces prismatic hexagonal geometry'),
    # ── Fantasy / characters ──────────────────────────────────────────────────
    ('robot mechanical',  'robot machine 3D mesh rigid body articulated joints actuators'),
    ('dragon wings',      'fantasy dragon 3D mesh wings four limbs tail scales head horns'),
    ('character humanoid','humanoid character 3D mesh head torso arms legs articulation'),
]

# Wikipedia articles per category for rich text corpus
WIKI_CORPUS_TITLES = [
    # core bovine
    ('Holstein_Friesian_cattle', 'bovine'), ('Cattle',                   'bovine'),
    ('Dairy_cattle',             'bovine'), ('Bovine_anatomy',            'bovine'),
    ('Bovine_locomotion',        'bovine'), ('Udder',                     'bovine'),
    ('Rumen',                    'bovine'), ('Hoof',                      'bovine'),
    # other animals
    ('Horse_anatomy',  'equine'),   ('Domestic_horse',    'equine'),
    ('Dog_anatomy',    'canine'),   ('Elephant_anatomy',  'elephant'),
    ('Tiger',          'felid'),    ('Gray_wolf',         'canine'),
    # human anatomy
    ('Human_body',             'human'), ('Human_skeleton',          'human'),
    ('Vertebral_column',       'human'), ('Limb_(anatomy)',           'human'),
    ('Muscle',                 'human'), ('Facial_skeleton',          'human'),
    # general animal anatomy
    ('Quadrupedalism', 'anatomy'), ('Tetrapod',         'anatomy'),
    ('Mammal',         'anatomy'), ('Skull',            'anatomy'),
    ('Vertebrate',     'anatomy'), ('Bilateral_symmetry','anatomy'),
    # 3D mesh / computer graphics
    ('Polygon_mesh',          '3d'), ('UV_mapping',           '3d'),
    ('3D_modeling',           '3d'), ('Texture_mapping',      '3d'),
    ('Triangulation_(geometry)','3d'),('Photogrammetry',       '3d'),
    ('Computer_graphics',     '3d'), ('Rendering_(computer_graphics)','3d'),
    ('Normal_mapping',        '3d'), ('Level_of_detail_(computer_graphics)','3d'),
    # extra real-world sources (Wikimedia science articles)
    ('Surface_reconstruction', '3d'), ('Point_cloud',          '3d'),
    ('Vertex_(computer_graphics)','3d'),('Triangle_mesh',       '3d'),
    ('Topology_(geometry)',   '3d'),
]

# 8 views per mesh: covers front/rear/sides/top and two 3-quarter angles
MESH_RENDER_VIEWS = [
    (  0,   0, 'right_side'),
    ( 90,   0, 'front'),
    (180,   0, 'left_side'),
    (270,   0, 'rear'),
    ( 45,  28, 'right_front_high'),
    (315,  28, 'right_rear_high'),
    (135,  28, 'left_front_high'),
    (  0,  88, 'top_down'),
]


def _rot3y(a):
    c, s = math.cos(a), math.sin(a)
    return [c, 0, s,  0, 1, 0,  -s, 0, c]

def _rot3x(a):
    c, s = math.cos(a), math.sin(a)
    return [1, 0, 0,  0, c, -s,  0, s, c]

def _m3mul(a, b):
    r = [0.0] * 9
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r[i*3+j] += a[i*3+k] * b[k*3+j]
    return r

def _mv3(m, v):
    return [m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
            m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
            m[6]*v[0]+m[7]*v[1]+m[8]*v[2]]


def parse_obj_text(text: str):
    """Pure Python OBJ parser — handles v/vt/vn, quads, fan-triangulates.
    Returns (verts, uvs, faces, face_uvs) where uvs/face_uvs may be empty.
    """
    verts, uvs, faces, face_uvs = [], [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == '#':
            continue
        parts = line.split()
        if parts[0] == 'v' and len(parts) >= 4:
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                pass
        elif parts[0] == 'vt' and len(parts) >= 3:
            try:
                uvs.append([float(parts[1]), float(parts[2])])
            except ValueError:
                pass
        elif parts[0] == 'f' and len(parts) >= 4:
            v_idxs, uv_idxs = [], []
            ok = True
            for p in parts[1:]:
                segs = p.split('/')
                try:
                    v_idxs.append(int(segs[0]))
                    uv_idxs.append(int(segs[1]) if len(segs) > 1 and segs[1] else None)
                except ValueError:
                    ok = False; break
            if not ok:
                continue
            rv = [(i - 1) if i > 0 else (len(verts) + i) for i in v_idxs]
            ru = [(i - 1) if i and i > 0 else (len(uvs) + i if i else None)
                  for i in uv_idxs]
            for i in range(1, len(rv) - 1):
                faces.append([rv[0], rv[i], rv[i + 1]])
                face_uvs.append([ru[0], ru[i], ru[i + 1]])
    return verts, uvs, faces, face_uvs


def render_uv_layout(uvs: list, face_uvs: list, img_size: int = 256):
    """Render UV unwrap layout as PIL Image — triangles drawn in UV space."""
    if not HAS_PIL or not uvs or not face_uvs:
        return None
    from PIL import Image, ImageDraw
    img  = Image.new('RGB', (img_size, img_size), (12, 14, 20))
    draw = ImageDraw.Draw(img)
    for tri in face_uvs:
        if not all(i is not None and 0 <= i < len(uvs) for i in tri):
            continue
        pts = [(int(uvs[i][0] * (img_size - 1)),
                int((1.0 - uvs[i][1]) * (img_size - 1))) for i in tri]
        draw.polygon(pts, fill=None, outline=(40, 200, 80))
    return img


def render_mesh_view(verts, faces, az_deg, el_deg, img_size=256):
    """Pure Python + PIL orthographic renderer with diffuse shading.
    Returns a PIL Image or None if PIL is unavailable.
    """
    if not HAS_PIL or not verts:
        return None
    from PIL import Image, ImageDraw

    rot = _m3mul(_rot3x(-math.radians(el_deg)), _rot3y(math.radians(az_deg)))

    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    cx = (min(xs)+max(xs))/2;  cy = (min(ys)+max(ys))/2;  cz = (min(zs)+max(zs))/2
    span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 1e-6)

    rv = [_mv3(rot, [(v[0]-cx)/span, (v[1]-cy)/span, (v[2]-cz)/span]) for v in verts]

    scale = img_size * 0.43
    half  = img_size / 2
    def px(v): return (int(v[0]*scale + half), int(-v[1]*scale + half))

    img  = Image.new('RGB', (img_size, img_size), (22, 22, 28))
    draw = ImageDraw.Draw(img)

    light = [0.577, 0.577, 0.577]

    face_d = []
    for f in faces:
        vv = [rv[i] for i in f if 0 <= i < len(rv)]
        if vv:
            face_d.append((sum(v[2] for v in vv) / len(vv), f))
    face_d.sort()

    for _, f in face_d:
        vv = [rv[i] for i in f if 0 <= i < len(rv)]
        if len(vv) < 3:
            continue
        pts = [px(v) for v in vv]
        e1  = [vv[1][i]-vv[0][i] for i in range(3)]
        e2  = [vv[2][i]-vv[0][i] for i in range(3)]
        n   = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]]
        nl  = math.sqrt(n[0]**2+n[1]**2+n[2]**2) or 1
        d   = max(0, (n[0]*light[0]+n[1]*light[1]+n[2]*light[2]) / nl)
        c   = int(50 + 178 * d)
        draw.polygon(pts, fill=(c, c, max(0, c-18)), outline=(14, 14, 18))

    return img


def _fetch_wiki_extract(title: str, ua: str) -> str:
    """Fetch Wikipedia plain-text extract for an article title."""
    url = ('https://en.wikipedia.org/api/rest_v1/page/summary/'
           + urllib.parse.quote(title.replace(' ', '_')))
    try:
        req = urllib.request.Request(url, headers={'User-Agent': ua})
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        return d.get('extract', '')[:800]
    except Exception:
        return ''


def _fetch_wikidata_description(title: str, ua: str) -> str:
    """Fetch short Wikidata description via Wikipedia API."""
    params = {'action': 'query', 'prop': 'description', 'titles': title,
              'format': 'json'}
    url = 'https://en.wikipedia.org/w/api.php?' + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': ua})
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        pages = d.get('query', {}).get('pages', {})
        for p in pages.values():
            return p.get('description', '')
    except Exception:
        return ''


def build_mesh_training(out_dir: Path, node: str,
                        max_sketchfab_per_query: int = 100,
                        max_sketchfab_total: int = 10000) -> list:
    """Stage 9: 3D mesh understanding — 10K+ cross-modal training items.

    Four sub-stages:
      A. CC0 OBJ meshes: download, parse, render 8 views + UV layout, train triplets
         [3D render] + [UV layout] + [OBJ vertex text] per mesh.
      B. Sketchfab: paginated search across 65+ categories × 100 models = ~6K thumbnails.
         Train each with mesh topology metadata + semantic context.
      C. Wikipedia + Wikidata corpus: rich semantic text per object category
         posted as text-modality items for each of 30+ categories.
      D. Cross-category mesh text: OBJ vertex/face excerpts as pure text training.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_dir   = out_dir / 'obj_renders'
    sfab_dir  = out_dir / 'sketchfab_thumbs'
    wiki_dir  = out_dir / 'wiki_corpus'
    obj_dir.mkdir(exist_ok=True)
    sfab_dir.mkdir(exist_ok=True)
    wiki_dir.mkdir(exist_ok=True)

    UA   = ('W1z4rDV1510n-DatasetBuilder/1.0 '
            '(https://github.com/C4rr13rX/W1z4rDV1510n; adamedsall@gmail.com)')
    base = f'http://{node}'
    items: list  = []
    ok_obj = ok_sfab = ok_wiki = ok_text = 0

    def _train_img(b64, text):
        body = json.dumps({'modality': 'image', 'data_b64': b64, 'text': text}).encode()
        req  = urllib.request.Request(f'{base}/media/train', data=body,
               headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=12) as r:
            json.loads(r.read())

    def _train_text(text):
        body = json.dumps({'modality': 'text', 'text': text[:6000]}).encode()
        req  = urllib.request.Request(f'{base}/media/train', data=body,
               headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=8) as r:
            json.loads(r.read())

    # ── Part A: OBJ meshes ──────────────────────────────────────────────────
    print('  [9A] Downloading OBJ meshes — render, UV map, vertex text...')

    for obj_url, name, category, desc in tqdm(MESH_OBJ_SOURCES, desc='OBJ meshes'):
        try:
            req = urllib.request.Request(obj_url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode('utf-8', errors='replace')
        except Exception as e:
            print(f'\n    [WARN] {name}: {e}')
            continue

        verts, uvs, faces, face_uvs = parse_obj_text(raw)
        if not verts or not faces:
            print(f'\n    [WARN] {name}: no geometry parsed')
            continue

        topo_base = (f'3D OBJ mesh: {name.replace("_"," ")}. '
                     f'Topology: {len(verts)} vertices, {len(faces)} triangular faces. '
                     f'Category: {category.replace("_"," ")}. {desc}')

        # A1: 8 rendered views
        for az, el, view_name in MESH_RENDER_VIEWS:
            img = render_mesh_view(verts, faces, az, el)
            if img is None:
                continue
            img_path = obj_dir / f'{name}_{view_name}.jpg'
            try:
                img.save(str(img_path), 'JPEG', quality=88)
                b64 = base64.b64encode(img_path.read_bytes()).decode()
                view_text = (f'{topo_base} '
                             f'3D render: {view_name.replace("_"," ")} view '
                             f'(azimuth {az}deg elevation {el}deg). '
                             f'Orthographic projection shows surface topology, '
                             f'depth, limb arrangement, body segment proportions. '
                             f'From this 2D rendering the original 3D mesh can be '
                             f'reconstructed by predicting occluded vertices.')
                _train_img(b64, view_text)
                ok_obj += 1
                items.append({'stage': 9, 'type': 'obj_render', 'name': name,
                               'view': view_name, 'modality': 'multimodal',
                               'tags': ['mesh', '3d', category, view_name]})
            except Exception as e:
                print(f'\n    [WARN] render {name}/{view_name}: {e}')
            time.sleep(0.05)

        # A2: UV layout map (if UVs present)
        if uvs and face_uvs:
            uv_img = render_uv_layout(uvs, face_uvs)
            if uv_img is not None:
                uv_path = obj_dir / f'{name}_uv_layout.jpg'
                try:
                    uv_img.save(str(uv_path), 'JPEG', quality=88)
                    b64 = base64.b64encode(uv_path.read_bytes()).decode()
                    uv_text = (f'UV unwrap layout for: {name.replace("_"," ")}. '
                               f'Flattened mesh surface showing how {len(uvs)} UV '
                               f'coordinates map each of {len(faces)} triangular faces '
                               f'to texture space. Green triangles = face boundaries. '
                               f'UV map used to apply 2D texture image onto 3D mesh. '
                               f'Category: {category.replace("_"," ")}. '
                               f'Reconstructing this UV layout from a 2D photo requires '
                               f'predicting how the 3D surface unfolds flat.')
                    _train_img(b64, uv_text)
                    ok_obj += 1
                    items.append({'stage': 9, 'type': 'uv_layout', 'name': name,
                                   'modality': 'multimodal',
                                   'tags': ['mesh', 'uv_map', category]})
                except Exception as e:
                    print(f'\n    [WARN] UV layout {name}: {e}')

        # A3: OBJ vertex text — teach the raw geometry language
        sample_step = max(1, len(verts) // 30)
        sample_v    = verts[::sample_step][:30]
        vert_lines  = ' '.join(f'v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}' for v in sample_v)
        sample_f    = faces[::max(1, len(faces)//10)][:10]
        face_lines  = ' '.join(f'f {f[0]+1} {f[1]+1} {f[2]+1}' for f in sample_f)
        obj_txt     = (f'OBJ file format: {name.replace("_"," ")}. '
                       f'Vertex coordinates (x y z): {vert_lines}... '
                       f'Triangular face indices (1-based): {face_lines}... '
                       f'Total: {len(verts)} vertices, {len(faces)} triangles. '
                       f'UV coordinates present: {bool(uvs)}. '
                       f'Category: {category.replace("_"," ")}. '
                       f'Mesh reconstruction from image: identify object outline '
                       f'in 2D, project to 3D coordinate system, predict occluded '
                       f'vertex positions from category prior, triangulate surface.')
        try:
            _train_text(obj_txt)
            ok_text += 1
            items.append({'stage': 9, 'type': 'obj_text', 'name': name,
                           'modality': 'text', 'tags': ['mesh', 'obj', category]})
        except Exception as e:
            print(f'\n    [WARN] obj text {name}: {e}')

    print(f'  [9A] OBJ: {ok_obj} image pairs + {ok_text} text items trained')

    # ── Part B: Sketchfab — paginated across 65+ categories ─────────────────
    print(f'  [9B] Sketchfab paginated search ({len(SKETCHFAB_MESH_QUERIES)} queries, '
          f'up to {max_sketchfab_per_query}/query)...')

    for query, semantic_text in tqdm(SKETCHFAB_MESH_QUERIES, desc='Sketchfab'):
        if ok_sfab >= max_sketchfab_total:
            break
        fetched = 0
        next_url = ('https://api.sketchfab.com/v3/models?'
                    + urllib.parse.urlencode({'q': query,
                                              'count': '24',
                                              'sort_by': '-relevance'}))
        while next_url and fetched < max_sketchfab_per_query and ok_sfab < max_sketchfab_total:
            try:
                req = urllib.request.Request(next_url, headers={'User-Agent': UA})
                with urllib.request.urlopen(req, timeout=14) as r:
                    page = json.loads(r.read())
                next_url = page.get('next')
                results  = page.get('results', [])
            except Exception as e:
                print(f'\n    [WARN] Sketchfab page "{query}": {e}')
                break

            for model in results:
                if fetched >= max_sketchfab_per_query or ok_sfab >= max_sketchfab_total:
                    break
                mname = model.get('name', '')
                vc    = model.get('vertexCount') or 0
                fc    = model.get('faceCount')   or 0
                cats  = ', '.join(c.get('name', '') for c in model.get('categories', []))
                mdesc = (model.get('description') or '')[:200].replace('\n', ' ')

                thumbs = model.get('thumbnails', {}).get('images', [])
                turl   = None
                for t in sorted(thumbs, key=lambda x: abs(x.get('width', 0) - 448)):
                    turl = t.get('url'); break
                if not turl:
                    continue

                fname = sfab_dir / (hashlib.sha1(turl.encode()).hexdigest()[:12] + '.jpg')
                if not fname.exists():
                    try:
                        dl = urllib.request.Request(turl, headers={'User-Agent': UA})
                        with urllib.request.urlopen(dl, timeout=18) as r:
                            fname.write_bytes(r.read())
                    except Exception:
                        continue

                topo = f'{vc} vertices, {fc} faces' if vc else 'polygonal mesh'
                text = (f'3D model render: {mname}. '
                        f'Mesh topology: {topo}. '
                        f'Categories: {cats or query}. '
                        f'{semantic_text}. '
                        f'Mesh reconstruction: given this 2D render, predict the full '
                        f'3D mesh including all hidden faces, generate UV unwrap, '
                        f'produce OBJ file with vertices and faces. {mdesc}')
                try:
                    b64 = base64.b64encode(fname.read_bytes()).decode()
                    _train_img(b64, text)
                    ok_sfab += 1
                    fetched += 1
                    items.append({'stage': 9, 'type': 'sketchfab_thumb',
                                   'name': mname, 'modality': 'multimodal',
                                   'tags': ['mesh', '3d', 'sketchfab', query]})
                except Exception as e:
                    print(f'\n    [WARN] sfab train {fname.name}: {e}')
                time.sleep(0.10)

            time.sleep(0.3)  # between Sketchfab pages

    print(f'  [9B] Sketchfab: {ok_sfab} thumbnails trained')

    # ── Part C: Wikipedia + Wikidata text corpus ─────────────────────────────
    print(f'  [9C] Wikipedia corpus for {len(WIKI_CORPUS_TITLES)} object categories...')

    wiki_cache: dict = {}
    for wiki_title, cat in tqdm(WIKI_CORPUS_TITLES, desc='Wikipedia'):
        extract = wiki_cache.get(wiki_title)
        if extract is None:
            extract = _fetch_wiki_extract(wiki_title, UA)
            wiki_cache[wiki_title] = extract
            time.sleep(0.2)

        if not extract:
            continue

        wdata_desc = _fetch_wikidata_description(wiki_title, UA)
        time.sleep(0.15)

        wiki_txt = (f'Real-world reference: {wiki_title.replace("_"," ")}. '
                    f'Category: {cat.replace("_"," ")}. '
                    f'Description: {wdata_desc}. '
                    f'{extract} '
                    f'3D mesh representation: vertices define surface of this object. '
                    f'UV map flattens surface for texturing. OBJ format stores geometry.')

        # Also save locally as reference
        fpath = wiki_dir / f'{wiki_title[:60]}.txt'
        fpath.write_text(wiki_txt, encoding='utf-8')

        try:
            _train_text(wiki_txt)
            ok_wiki += 1
            items.append({'stage': 9, 'type': 'wiki_corpus', 'title': wiki_title,
                           'modality': 'text', 'tags': ['wiki', cat, '3d']})
        except Exception as e:
            print(f'\n    [WARN] wiki {wiki_title}: {e}')

    print(f'  [9C] Wikipedia corpus: {ok_wiki} items trained')
    print(f'  Stage 9 total: {len(items)} items '
          f'({ok_obj} OBJ renders/UV + {ok_sfab} Sketchfab + '
          f'{ok_wiki} wiki + {ok_text} obj-text)')
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Build and ingest bovine anatomy training dataset into W1z4rD node')
    ap.add_argument('--stages',    default='0,1,2,3,4,5,6,7,8,9',
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

    # ── Stage 3: Medical imaging — synthetic MRI/CT cross-sections ──────────────
    if 3 in stages_to_run:
        print('\n[Stage 3] Medical imaging — synthetic MRI/CT cross-sections')
        items3 = generate_mri_ct_data(
            train_dir / 'stage3_mri_ct',
            n_noise_levels=8,
        )
        all_items[3] = items3

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

    # ── Stage 6: Bovine Q&A pairs ────────────────────────────────────────────
    if 6 in stages_to_run:
        print('\n[Stage 6] Bovine anatomy Q&A pairs (chat capability)')
        if args.download_only:
            print('  [SKIP] Stage 6: Q&A ingestion skipped in --download-only mode')
        else:
            items6 = build_bovine_qa(train_dir / 'stage6_qa', args.node)
            all_items[6] = items6

    if 7 in stages_to_run:
        print('\n[Stage 7] Cross-modal training (image + anatomical text Hebbian links)')
        if args.download_only:
            print('  [SKIP] Stage 7: cross-modal training skipped in --download-only mode')
        else:
            frames_dir = train_dir / 'stage2_video' / 'frames'
            videos_dir = train_dir / 'stage2_video' / 'videos'
            items7 = build_cross_modal(frames_dir, videos_dir, train_dir, args.node)
            all_items[7] = items7

    if 8 in stages_to_run:
        print('\n[Stage 8] Multi-angle cow gallery (Wikimedia Commons)')
        if args.download_only:
            print('  [SKIP] Stage 8: cross-modal training skipped in --download-only mode')
        else:
            items8 = build_multiview_gallery(
                train_dir / 'stage8_multiview', args.node,
                max_per_cat=30, max_total=300)
            all_items[8] = items8

    if 9 in stages_to_run:
        print('\n[Stage 9] 3D mesh understanding (OBJ renders + Sketchfab thumbnails)')
        if args.download_only:
            print('  [SKIP] Stage 9: cross-modal training skipped in --download-only mode')
        else:
            items9 = build_mesh_training(
                train_dir / 'stage9_mesh', args.node,
                max_sketchfab_per_query=100, max_sketchfab_total=10000)
            all_items[9] = items9

    # ── Write manifest ───────────────────────────────────────────────────────
    write_manifest(data_dir, all_items)

    # ── Ingest all collected items ───────────────────────────────────────────
    if not args.download_only:
        # Stage 6 Q&A items are already ingested via /qa/ingest in build_bovine_qa
        all_flat = [it for sid, items in all_items.items() for it in items
                    if sid not in (6, 7, 8, 9) and (it.get('b64') or it.get('text'))]
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
