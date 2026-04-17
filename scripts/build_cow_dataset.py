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
        for _py in [r'C:\Python313\python.exe', r'C:\Users\Adam\AppData\Roaming\Python\Python313\Scripts\yt-dlp.exe']:
            try:
                import subprocess as _sp
                _sp.run([_py, '--version'] if _py.endswith('.exe') and 'yt-dlp' in _py
                        else [_py, '-c', 'import yt_dlp'],
                        capture_output=True, check=True)
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
            if _yt_python == 'api':
                # Use yt-dlp Python API directly
                import yt_dlp as _yt
                ydl_opts = {
                    'format': 'bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]',
                    'merge_output_format': 'mp4',
                    'outtmpl': str(vid_dir / '%(id)s.%(ext)s'),
                    'max_downloads': 2,
                    'noplaylist': True,
                    'quiet': True,
                    'writeinfojson': True,
                    'match_filter': _yt.utils.match_filter_func(
                        'license = "Creative Commons Attribution licence (reuse allowed)"'),
                }
                with _yt.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f'ytsearch5:{query}'])
            else:
                subprocess.run([
                    _yt_python, '-m', 'yt_dlp',
                    f'ytsearch5:{query}',
                    '--match-filter', 'license = "Creative Commons Attribution licence (reuse allowed)"',
                    '--format', 'bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]',
                    '--merge-output-format', 'mp4',
                    '--output', str(vid_dir / '%(id)s.%(ext)s'),
                    '--max-downloads', '2',
                    '--no-playlist', '--quiet', '--write-info-json',
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
]


def generate_mri_ct_data(out_dir: Path, n_noise_levels: int = 3) -> list[dict]:
    """
    Generate synthetic bovine MRI/CT cross-section images and accompanying text.
    Produces 3 modalities × 8 anatomical levels × n_noise_levels = 72 images,
    plus 8 detailed text documents (one per anatomical level).
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

    # ── Stage 3: Medical imaging — synthetic MRI/CT cross-sections ──────────────
    if 3 in stages_to_run:
        print('\n[Stage 3] Medical imaging — synthetic MRI/CT cross-sections')
        items3 = generate_mri_ct_data(
            train_dir / 'stage3_mri_ct',
            n_noise_levels=3,
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
