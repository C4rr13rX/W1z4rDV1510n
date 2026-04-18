#!/usr/bin/env python3
"""
build_medical_corpus.py — Stages 30-33
Medical, psychological, genetic, and longevity science corpus.
All content sourced exclusively from peer-reviewed literature via
the NCBI/NLM E-utilities API (PubMed + PubMed Central).

Every training item carries its provenance: PMID, journal, year, article type —
so the model learns both the content and its authoritative source.

Stages:
  30 — Clinical Medicine: full physician-level knowledge (all specialties)
  31 — Clinical Psychology & Psychiatry: DSM-5, therapy modalities, neuro
  32 — Genetic Engineering: CRISPR, gene therapy, genomics, epigenetics
  33 — Reverse Aging & Longevity: hallmarks of aging, senolytics, NAD+, clocks

Data quality:
  • PubMed-indexed = peer-reviewed by definition
  • Filtered to: Review, Systematic Review, Meta-Analysis, Practice Guideline
  • PMC open-access full text for landmark review articles (full body text)
  • Rate-limited to comply with NCBI policy (0.34s between calls, or 0.1s with key)

Usage:
  python scripts/build_medical_corpus.py --node localhost:8090
  python scripts/build_medical_corpus.py --stages 30
  python scripts/build_medical_corpus.py --ncbi-api-key YOUR_KEY_HERE
  (Free NCBI API key: https://www.ncbi.nlm.nih.gov/account/)
"""

import argparse, json, re, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_DATA_DIR = 'D:/w1z4rdv1510n-data'
DEFAULT_NODE     = 'localhost:8090'
NCBI_BASE        = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
TOOL             = 'W1z4rDV1510n'
EMAIL            = 'adamedsall@gmail.com'
UA               = 'W1z4rDV1510n-Medical/1.0 (adamedsall@gmail.com; educational AI)'

STAGES = {
    30: 'Clinical Medicine — physician-level knowledge across all specialties',
    31: 'Clinical Psychology & Psychiatry — DSM-5, therapies, neuroscience',
    32: 'Genetic Engineering — CRISPR, gene therapy, genomics, synthetic biology',
    33: 'Reverse Aging & Longevity — hallmarks of aging, senolytics, epigenetic clocks',
}

# Search filter for high-quality article types only
QUALITY_FILTER = '(Review[ptyp] OR "Systematic Review"[ptyp] OR "Meta-Analysis"[ptyp] OR "Practice Guideline"[ptyp])'

# ── Search term lists ──────────────────────────────────────────────────────────

STAGE30_QUERIES = [
    # ── Cardiology ────────────────────────────────────────────────────────────
    'heart failure[MeSH] AND ' + QUALITY_FILTER,
    'coronary artery disease[MeSH] AND ' + QUALITY_FILTER,
    'atrial fibrillation[MeSH] AND ' + QUALITY_FILTER,
    'acute myocardial infarction[MeSH] AND ' + QUALITY_FILTER,
    'hypertension[MeSH] AND ' + QUALITY_FILTER,
    'valvular heart disease[MeSH] AND ' + QUALITY_FILTER,
    'cardiomyopathy[MeSH] AND ' + QUALITY_FILTER,
    'cardiac arrhythmia[MeSH] AND ' + QUALITY_FILTER,
    # ── Pulmonology ───────────────────────────────────────────────────────────
    'asthma[MeSH] AND ' + QUALITY_FILTER,
    'pulmonary disease chronic obstructive[MeSH] AND ' + QUALITY_FILTER,
    'pneumonia[MeSH] AND ' + QUALITY_FILTER,
    'pulmonary embolism[MeSH] AND ' + QUALITY_FILTER,
    'interstitial lung disease[MeSH] AND ' + QUALITY_FILTER,
    'lung neoplasms[MeSH] AND ' + QUALITY_FILTER,
    'pulmonary hypertension[MeSH] AND ' + QUALITY_FILTER,
    'obstructive sleep apnea[MeSH] AND ' + QUALITY_FILTER,
    # ── Gastroenterology ──────────────────────────────────────────────────────
    'inflammatory bowel disease[MeSH] AND ' + QUALITY_FILTER,
    'liver cirrhosis[MeSH] AND ' + QUALITY_FILTER,
    'gastroesophageal reflux[MeSH] AND ' + QUALITY_FILTER,
    'colorectal neoplasms[MeSH] AND ' + QUALITY_FILTER,
    'hepatocellular carcinoma[MeSH] AND ' + QUALITY_FILTER,
    'pancreatitis[MeSH] AND ' + QUALITY_FILTER,
    'celiac disease[MeSH] AND ' + QUALITY_FILTER,
    'peptic ulcer[MeSH] AND ' + QUALITY_FILTER,
    # ── Nephrology ────────────────────────────────────────────────────────────
    'chronic kidney disease[MeSH] AND ' + QUALITY_FILTER,
    'acute kidney injury[MeSH] AND ' + QUALITY_FILTER,
    'glomerulonephritis[MeSH] AND ' + QUALITY_FILTER,
    'renal replacement therapy[MeSH] AND ' + QUALITY_FILTER,
    'nephrotic syndrome[MeSH] AND ' + QUALITY_FILTER,
    # ── Neurology ─────────────────────────────────────────────────────────────
    'stroke[MeSH] AND ' + QUALITY_FILTER,
    'multiple sclerosis[MeSH] AND ' + QUALITY_FILTER,
    'epilepsy[MeSH] AND ' + QUALITY_FILTER,
    'Alzheimer disease[MeSH] AND ' + QUALITY_FILTER,
    'Parkinson disease[MeSH] AND ' + QUALITY_FILTER,
    'migraine[MeSH] AND ' + QUALITY_FILTER,
    'dementia[MeSH] AND ' + QUALITY_FILTER,
    'amyotrophic lateral sclerosis[MeSH] AND ' + QUALITY_FILTER,
    # ── Endocrinology ─────────────────────────────────────────────────────────
    'diabetes mellitus type 2[MeSH] AND ' + QUALITY_FILTER,
    'diabetes mellitus type 1[MeSH] AND ' + QUALITY_FILTER,
    'thyroid diseases[MeSH] AND ' + QUALITY_FILTER,
    'obesity[MeSH] AND ' + QUALITY_FILTER,
    'metabolic syndrome[MeSH] AND ' + QUALITY_FILTER,
    'adrenal gland diseases[MeSH] AND ' + QUALITY_FILTER,
    'pituitary diseases[MeSH] AND ' + QUALITY_FILTER,
    # ── Hematology / Oncology ─────────────────────────────────────────────────
    'leukemia[MeSH] AND ' + QUALITY_FILTER,
    'lymphoma[MeSH] AND ' + QUALITY_FILTER,
    'anemia[MeSH] AND ' + QUALITY_FILTER,
    'breast neoplasms[MeSH] AND ' + QUALITY_FILTER,
    'prostate neoplasms[MeSH] AND ' + QUALITY_FILTER,
    'immunotherapy antineoplastic[MeSH] AND ' + QUALITY_FILTER,
    'chimeric antigen receptor T-cell therapy[tw] AND ' + QUALITY_FILTER,
    # ── Infectious Disease ────────────────────────────────────────────────────
    'sepsis[MeSH] AND ' + QUALITY_FILTER,
    'HIV infections[MeSH] AND ' + QUALITY_FILTER,
    'tuberculosis[MeSH] AND ' + QUALITY_FILTER,
    'antimicrobial resistance[MeSH] AND ' + QUALITY_FILTER,
    'COVID-19[MeSH] AND ' + QUALITY_FILTER,
    'hepatitis B[MeSH] AND ' + QUALITY_FILTER,
    'hepatitis C[MeSH] AND ' + QUALITY_FILTER,
    # ── Rheumatology / Immunology ─────────────────────────────────────────────
    'rheumatoid arthritis[MeSH] AND ' + QUALITY_FILTER,
    'systemic lupus erythematosus[MeSH] AND ' + QUALITY_FILTER,
    'autoimmune diseases[MeSH] AND ' + QUALITY_FILTER,
    'vasculitis[MeSH] AND ' + QUALITY_FILTER,
    # ── Pharmacology ──────────────────────────────────────────────────────────
    'drug-drug interactions[MeSH] AND ' + QUALITY_FILTER,
    'pharmacokinetics[MeSH] AND ' + QUALITY_FILTER,
    'adverse drug reactions[MeSH] AND ' + QUALITY_FILTER,
    'anticoagulants[MeSH] AND ' + QUALITY_FILTER,
    'antibiotics[MeSH] AND ' + QUALITY_FILTER,
    'antihypertensive agents[MeSH] AND ' + QUALITY_FILTER,
    # ── Anatomy / Physiology ──────────────────────────────────────────────────
    'cardiac physiology[tw] AND ' + QUALITY_FILTER,
    'renal physiology[tw] AND ' + QUALITY_FILTER,
    'neuroanatomy[MeSH] AND ' + QUALITY_FILTER,
    'blood pressure physiology[tw] AND ' + QUALITY_FILTER,
    # ── Diagnostics ───────────────────────────────────────────────────────────
    'clinical laboratory techniques[MeSH] AND ' + QUALITY_FILTER,
    'electrocardiography[MeSH] AND ' + QUALITY_FILTER,
    'magnetic resonance imaging[MeSH] AND ' + QUALITY_FILTER,
    'point of care testing[MeSH] AND ' + QUALITY_FILTER,
    'biomarkers[MeSH] AND ' + QUALITY_FILTER,
    # ── Surgery / Emergency ───────────────────────────────────────────────────
    'surgical procedures operative[MeSH] AND ' + QUALITY_FILTER,
    'critical care[MeSH] AND ' + QUALITY_FILTER,
    'trauma surgery[tw] AND ' + QUALITY_FILTER,
    'resuscitation[MeSH] AND ' + QUALITY_FILTER,
]

STAGE31_QUERIES = [
    # ── DSM-5 Disorders ───────────────────────────────────────────────────────
    'major depressive disorder[MeSH] AND ' + QUALITY_FILTER,
    'bipolar disorder[MeSH] AND ' + QUALITY_FILTER,
    'schizophrenia[MeSH] AND ' + QUALITY_FILTER,
    'anxiety disorders[MeSH] AND ' + QUALITY_FILTER,
    'post-traumatic stress disorder[MeSH] AND ' + QUALITY_FILTER,
    'obsessive-compulsive disorder[MeSH] AND ' + QUALITY_FILTER,
    'attention deficit disorder hyperactivity[MeSH] AND ' + QUALITY_FILTER,
    'autism spectrum disorder[MeSH] AND ' + QUALITY_FILTER,
    'borderline personality disorder[MeSH] AND ' + QUALITY_FILTER,
    'eating disorders[MeSH] AND ' + QUALITY_FILTER,
    'substance use disorders[MeSH] AND ' + QUALITY_FILTER,
    'panic disorder[MeSH] AND ' + QUALITY_FILTER,
    'social phobia[MeSH] AND ' + QUALITY_FILTER,
    'somatic symptom disorder[tw] AND ' + QUALITY_FILTER,
    'dissociative disorders[MeSH] AND ' + QUALITY_FILTER,
    'sleep wake disorders[MeSH] AND ' + QUALITY_FILTER,
    # ── Psychotherapy Modalities ──────────────────────────────────────────────
    'cognitive behavioral therapy[MeSH] AND ' + QUALITY_FILTER,
    'dialectical behavior therapy[tw] AND ' + QUALITY_FILTER,
    'acceptance commitment therapy[tw] AND ' + QUALITY_FILTER,
    'eye movement desensitization reprocessing[MeSH] AND ' + QUALITY_FILTER,
    'psychoanalysis[MeSH] AND ' + QUALITY_FILTER,
    'motivational interviewing[MeSH] AND ' + QUALITY_FILTER,
    'mindfulness-based cognitive therapy[tw] AND ' + QUALITY_FILTER,
    'exposure therapy[tw] AND ' + QUALITY_FILTER,
    # ── Psychiatric Pharmacology ──────────────────────────────────────────────
    'antidepressive agents[MeSH] AND ' + QUALITY_FILTER,
    'antipsychotic agents[MeSH] AND ' + QUALITY_FILTER,
    'mood stabilizers[tw] AND ' + QUALITY_FILTER,
    'anxiolytics[MeSH] AND ' + QUALITY_FILTER,
    'psychostimulants[tw] AND ' + QUALITY_FILTER,
    'ketamine depression treatment[tw] AND ' + QUALITY_FILTER,
    # ── Neuropsychology ───────────────────────────────────────────────────────
    'neuropsychological tests[MeSH] AND ' + QUALITY_FILTER,
    'brain plasticity[MeSH] AND ' + QUALITY_FILTER,
    'default mode network[tw] AND ' + QUALITY_FILTER,
    'prefrontal cortex emotion[tw] AND ' + QUALITY_FILTER,
    'amygdala fear[tw] AND ' + QUALITY_FILTER,
    'hippocampus memory[tw] AND ' + QUALITY_FILTER,
    'dopamine reward[tw] AND ' + QUALITY_FILTER,
    'serotonin depression[tw] AND ' + QUALITY_FILTER,
    # ── Assessment / Diagnosis ────────────────────────────────────────────────
    'psychological tests[MeSH] AND ' + QUALITY_FILTER,
    'psychiatric diagnosis[tw] AND ' + QUALITY_FILTER,
    'mental health assessment[tw] AND ' + QUALITY_FILTER,
    # ── Developmental / Lifespan ──────────────────────────────────────────────
    'child psychology development[tw] AND ' + QUALITY_FILTER,
    'adolescent mental health[tw] AND ' + QUALITY_FILTER,
    'geriatric psychiatry[MeSH] AND ' + QUALITY_FILTER,
    'trauma childhood adverse experiences[tw] AND ' + QUALITY_FILTER,
    # ── Neuroscience of Mental Health ─────────────────────────────────────────
    'neuroinflammation mental health[tw] AND ' + QUALITY_FILTER,
    'gut brain axis psychiatry[tw] AND ' + QUALITY_FILTER,
    'psychoneuroimmunology[MeSH] AND ' + QUALITY_FILTER,
]

STAGE32_QUERIES = [
    # ── CRISPR Technology ─────────────────────────────────────────────────────
    'CRISPR-Cas9 gene editing[tw] AND ' + QUALITY_FILTER,
    'CRISPR base editing[tw] AND ' + QUALITY_FILTER,
    'CRISPR prime editing[tw] AND ' + QUALITY_FILTER,
    'CRISPR Cas12[tw] AND ' + QUALITY_FILTER,
    'CRISPR therapeutic applications[tw] AND ' + QUALITY_FILTER,
    'off-target effects CRISPR[tw] AND ' + QUALITY_FILTER,
    'CRISPR delivery mechanisms[tw] AND ' + QUALITY_FILTER,
    'CRISPR sickle cell disease[tw] AND ' + QUALITY_FILTER,
    # ── Gene Therapy ──────────────────────────────────────────────────────────
    'gene therapy[MeSH] AND ' + QUALITY_FILTER,
    'adeno-associated virus gene therapy[tw] AND ' + QUALITY_FILTER,
    'lentiviral vector gene therapy[tw] AND ' + QUALITY_FILTER,
    'CAR-T cell therapy[tw] AND ' + QUALITY_FILTER,
    'gene therapy clinical trials[tw] AND ' + QUALITY_FILTER,
    'in vivo gene editing[tw] AND ' + QUALITY_FILTER,
    'RNA therapeutics[tw] AND ' + QUALITY_FILTER,
    'mRNA therapy[tw] AND ' + QUALITY_FILTER,
    # ── Genomics ──────────────────────────────────────────────────────────────
    'human genome sequencing[tw] AND ' + QUALITY_FILTER,
    'whole genome sequencing clinical[tw] AND ' + QUALITY_FILTER,
    'genome-wide association studies[MeSH] AND ' + QUALITY_FILTER,
    'single nucleotide polymorphism[MeSH] AND ' + QUALITY_FILTER,
    'structural variation genome[tw] AND ' + QUALITY_FILTER,
    'pharmacogenomics[MeSH] AND ' + QUALITY_FILTER,
    'liquid biopsy circulating DNA[tw] AND ' + QUALITY_FILTER,
    # ── Epigenetics ───────────────────────────────────────────────────────────
    'epigenetics[MeSH] AND ' + QUALITY_FILTER,
    'DNA methylation disease[tw] AND ' + QUALITY_FILTER,
    'histone modification[MeSH] AND ' + QUALITY_FILTER,
    'chromatin remodeling[MeSH] AND ' + QUALITY_FILTER,
    'epigenetic inheritance[tw] AND ' + QUALITY_FILTER,
    'non-coding RNA epigenetics[tw] AND ' + QUALITY_FILTER,
    # ── Synthetic Biology ─────────────────────────────────────────────────────
    'synthetic biology[MeSH] AND ' + QUALITY_FILTER,
    'metabolic engineering[MeSH] AND ' + QUALITY_FILTER,
    'protein engineering[MeSH] AND ' + QUALITY_FILTER,
    'cell-free synthetic biology[tw] AND ' + QUALITY_FILTER,
    'biosafety genetic engineering[tw] AND ' + QUALITY_FILTER,
    # ── Molecular Tools ───────────────────────────────────────────────────────
    'zinc finger nucleases[MeSH] AND ' + QUALITY_FILTER,
    'transcription activator-like effector nucleases[MeSH] AND ' + QUALITY_FILTER,
    'RNA interference therapeutics[tw] AND ' + QUALITY_FILTER,
    'antisense oligonucleotides[MeSH] AND ' + QUALITY_FILTER,
    # ── Applications ──────────────────────────────────────────────────────────
    'gene therapy cancer[tw] AND ' + QUALITY_FILTER,
    'gene editing inherited disease[tw] AND ' + QUALITY_FILTER,
    'genetic engineering agriculture[tw] AND ' + QUALITY_FILTER,
    'genetically modified organisms biosafety[tw] AND ' + QUALITY_FILTER,
    'polygenic risk scores[tw] AND ' + QUALITY_FILTER,
]

STAGE33_QUERIES = [
    # ── Hallmarks of Aging ────────────────────────────────────────────────────
    'hallmarks of aging[tw] AND ' + QUALITY_FILTER,
    'cellular senescence[MeSH] AND ' + QUALITY_FILTER,
    'telomere shortening aging[tw] AND ' + QUALITY_FILTER,
    'mitochondrial dysfunction aging[tw] AND ' + QUALITY_FILTER,
    'proteostasis aging[tw] AND ' + QUALITY_FILTER,
    'stem cell exhaustion aging[tw] AND ' + QUALITY_FILTER,
    'epigenetic alterations aging[tw] AND ' + QUALITY_FILTER,
    'inflammaging chronic inflammation aging[tw] AND ' + QUALITY_FILTER,
    'nutrient sensing aging mTOR[tw] AND ' + QUALITY_FILTER,
    'intercellular communication aging[tw] AND ' + QUALITY_FILTER,
    'genomic instability aging[tw] AND ' + QUALITY_FILTER,
    'disabled macroautophagy aging[tw] AND ' + QUALITY_FILTER,
    # ── Senolytics / Senomorphics ─────────────────────────────────────────────
    'senolytics[tw] AND ' + QUALITY_FILTER,
    'dasatinib quercetin senolytic[tw] AND ' + QUALITY_FILTER,
    'navitoclax senolytic[tw] AND ' + QUALITY_FILTER,
    'senomorphics SASP[tw] AND ' + QUALITY_FILTER,
    'senescent cell clearance[tw] AND ' + QUALITY_FILTER,
    # ── NAD+ and Sirtuins ─────────────────────────────────────────────────────
    'NAD+ aging metabolism[tw] AND ' + QUALITY_FILTER,
    'nicotinamide riboside NMN aging[tw] AND ' + QUALITY_FILTER,
    'sirtuins longevity[tw] AND ' + QUALITY_FILTER,
    'SIRT1 aging metabolism[tw] AND ' + QUALITY_FILTER,
    # ── mTOR and Caloric Restriction ──────────────────────────────────────────
    'rapamycin longevity[tw] AND ' + QUALITY_FILTER,
    'mTOR pathway aging[tw] AND ' + QUALITY_FILTER,
    'caloric restriction lifespan[tw] AND ' + QUALITY_FILTER,
    'intermittent fasting longevity[tw] AND ' + QUALITY_FILTER,
    'dietary restriction aging mechanisms[tw] AND ' + QUALITY_FILTER,
    # ── Epigenetic Reprogramming ───────────────────────────────────────────────
    'epigenetic clock aging[tw] AND ' + QUALITY_FILTER,
    'Horvath clock methylation[tw] AND ' + QUALITY_FILTER,
    'partial reprogramming aging[tw] AND ' + QUALITY_FILTER,
    'Yamanaka factors aging reversal[tw] AND ' + QUALITY_FILTER,
    'biological age reversal[tw] AND ' + QUALITY_FILTER,
    # ── Telomere Biology ──────────────────────────────────────────────────────
    'telomerase aging cancer[tw] AND ' + QUALITY_FILTER,
    'telomere length disease[tw] AND ' + QUALITY_FILTER,
    'telomere restoration therapy[tw] AND ' + QUALITY_FILTER,
    # ── Longevity Pathways ────────────────────────────────────────────────────
    'IGF-1 insulin signaling aging[tw] AND ' + QUALITY_FILTER,
    'AMPK aging exercise[tw] AND ' + QUALITY_FILTER,
    'autophagy aging disease[tw] AND ' + QUALITY_FILTER,
    'klotho anti-aging[tw] AND ' + QUALITY_FILTER,
    'GDF11 aging reversal[tw] AND ' + QUALITY_FILTER,
    'parabiosis heterochronic aging[tw] AND ' + QUALITY_FILTER,
    # ── Clinical / Interventional ─────────────────────────────────────────────
    'metformin aging clinical trial[tw] AND ' + QUALITY_FILTER,
    'TAME trial metformin longevity[tw] AND ' + QUALITY_FILTER,
    'resveratrol aging[tw] AND ' + QUALITY_FILTER,
    'spermidine longevity[tw] AND ' + QUALITY_FILTER,
    'fisetin senolytic clinical[tw] AND ' + QUALITY_FILTER,
    'centenarians longevity genetics[tw] AND ' + QUALITY_FILTER,
    'blue zones longevity lifestyle[tw] AND ' + QUALITY_FILTER,
    'stem cell therapy aging[tw] AND ' + QUALITY_FILTER,
    'exosome therapy aging[tw] AND ' + QUALITY_FILTER,
]

STAGE_QUERIES = {
    30: (STAGE30_QUERIES, 'Clinical Medicine'),
    31: (STAGE31_QUERIES, 'Clinical Psychology & Psychiatry'),
    32: (STAGE32_QUERIES, 'Genetic Engineering'),
    33: (STAGE33_QUERIES, 'Reverse Aging & Longevity'),
}


# ── HTTP session ───────────────────────────────────────────────────────────────

def _make_session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'])))
    s.headers['User-Agent'] = UA
    return s

def _train(text, node, session):
    try:
        r = session.post(f'http://{node}/media/train',
                         data=json.dumps({'modality': 'text', 'text': text}),
                         headers={'Content-Type': 'application/json'}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f'  [WARN] train: {e}', flush=True)
        return False


# ── NCBI E-utilities helpers ───────────────────────────────────────────────────

def _ncbi_params(api_key=None):
    p = {'tool': TOOL, 'email': EMAIL}
    if api_key:
        p['api_key'] = api_key
    return p

def _rate_delay(api_key):
    """NCBI policy: 3 req/s without key, 10 req/s with key."""
    time.sleep(0.12 if api_key else 0.38)


def search_pubmed(query, max_results, api_key, session):
    """Return list of PMIDs for a PubMed query."""
    params = _ncbi_params(api_key)
    params.update({'db': 'pubmed', 'term': query,
                   'retmax': max_results, 'retmode': 'json', 'sort': 'relevance'})
    try:
        r = session.get(f'{NCBI_BASE}/esearch.fcgi', params=params, timeout=20)
        r.raise_for_status()
        return r.json()['esearchresult']['idlist']
    except Exception as e:
        print(f'  [WARN] search "{query[:50]}": {e}', flush=True)
        return []
    finally:
        _rate_delay(api_key)


def fetch_abstracts_text(pmids, api_key, session):
    """
    Fetch formatted abstracts for up to 20 PMIDs in one call.
    Returns the raw NCBI text block (includes title, authors, journal, abstract, PMID).
    """
    params = _ncbi_params(api_key)
    params.update({'db': 'pubmed', 'id': ','.join(pmids),
                   'retmode': 'text', 'rettype': 'abstract'})
    try:
        r = session.get(f'{NCBI_BASE}/efetch.fcgi', params=params, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f'  [WARN] efetch {len(pmids)} PMIDs: {e}', flush=True)
        return ''
    finally:
        _rate_delay(api_key)


def split_abstract_blocks(text):
    """
    Split the NCBI text abstract dump into individual article blocks.
    Articles are separated by blank lines before a numbered entry.
    """
    blocks = re.split(r'\n(?=\d+\. )', text.strip())
    return [b.strip() for b in blocks if b.strip() and len(b) > 100]


def search_pmc_fulltext(query, max_results, api_key, session):
    """Return PMC IDs for open-access full-text articles."""
    params = _ncbi_params(api_key)
    params.update({'db': 'pmc', 'term': query + ' AND open access[filter]',
                   'retmax': max_results, 'retmode': 'json', 'sort': 'relevance'})
    try:
        r = session.get(f'{NCBI_BASE}/esearch.fcgi', params=params, timeout=20)
        r.raise_for_status()
        return r.json()['esearchresult']['idlist']
    except Exception as e:
        print(f'  [WARN] PMC search: {e}', flush=True)
        return []
    finally:
        _rate_delay(api_key)


def fetch_pmc_fulltext(pmc_id, api_key, session):
    """
    Fetch and parse a PMC full-text XML article.
    Returns plain text (title + abstract + body sections).
    """
    params = _ncbi_params(api_key)
    params.update({'db': 'pmc', 'id': pmc_id, 'retmode': 'xml'})
    try:
        r = session.get(f'{NCBI_BASE}/efetch.fcgi', params=params, timeout=40)
        r.raise_for_status()
        return _parse_pmc_xml(r.text)
    except Exception as e:
        print(f'  [WARN] PMC fetch {pmc_id}: {e}', flush=True)
        return ''
    finally:
        _rate_delay(api_key)


def _parse_pmc_xml(xml_text):
    """Extract readable text from PMC XML (title, abstract, body sections)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ''

    parts = []

    # Journal
    for el in root.iter('journal-title'):
        parts.append(f'Journal: {(el.text or "").strip()}')
        break

    # Article title
    for el in root.iter('article-title'):
        title = ''.join(el.itertext()).strip()
        if title:
            parts.append(f'Title: {title}')
        break

    # Year
    for el in root.iter('pub-date'):
        yr = el.find('year')
        if yr is not None:
            parts.append(f'Year: {yr.text}')
        break

    # Abstract
    for abstract in root.iter('abstract'):
        abs_parts = []
        for p in abstract.iter('p'):
            t = ''.join(p.itertext()).strip()
            if t:
                abs_parts.append(t)
        if abs_parts:
            parts.append('\nAbstract:\n' + ' '.join(abs_parts))
        break

    # Body sections
    for body in root.iter('body'):
        for sec in body.iter('sec'):
            sec_title = sec.find('title')
            if sec_title is not None:
                t = ''.join(sec_title.itertext()).strip()
                if t:
                    parts.append(f'\n[{t}]')
            for p in sec.findall('p'):
                t = ''.join(p.itertext()).strip()
                if t and len(t) > 40:
                    parts.append(t)
        break

    return '\n'.join(parts)


# ── Stage builder ──────────────────────────────────────────────────────────────

def build_stage(stage_num, queries, domain_label, out_dir, node,
                max_per_query, api_key, train_pmc_fulltext,
                session, train_session):
    """
    Run one stage: search PubMed for each query, fetch abstracts, train.
    Optionally fetch PMC full text for landmark queries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'

    trained_pmids: set = set()
    trained_pmc:   set = set()

    if ckpt_path.exists():
        try:
            state = json.loads(ckpt_path.read_text(encoding='utf-8'))
            trained_pmids = set(state.get('pmids', []))
            trained_pmc   = set(state.get('pmc', []))
            print(f'  [Stage {stage_num}] Resuming — '
                  f'{len(trained_pmids)} abstracts, {len(trained_pmc)} full texts',
                  flush=True)
        except Exception:
            pass

    def save_ckpt():
        ckpt_path.write_text(json.dumps(
            {'pmids': sorted(trained_pmids), 'pmc': sorted(trained_pmc)}
        ), encoding='utf-8')

    ok = 0
    header = (f'PEER-REVIEWED MEDICAL LITERATURE\n'
              f'Source: National Library of Medicine — PubMed/NCBI\n'
              f'Domain: {domain_label}\n\n')

    for qi, query in enumerate(queries):
        short_q = query.split('[')[0].strip()[:60]
        print(f'  [{stage_num}] ({qi+1}/{len(queries)}) {short_q}...', flush=True)

        # ── PubMed abstracts ──────────────────────────────────────────────────
        pmids = search_pubmed(query, max_per_query, api_key, session)
        new_pmids = [p for p in pmids if p not in trained_pmids]

        for batch_start in range(0, len(new_pmids), 20):
            batch = new_pmids[batch_start: batch_start + 20]
            raw   = fetch_abstracts_text(batch, api_key, session)
            if not raw:
                continue
            for block in split_abstract_blocks(raw):
                text = header + block
                if _train(text, node, train_session):
                    ok += 1
                time.sleep(0.08)
            for pmid in batch:
                trained_pmids.add(pmid)

        # ── PMC full text (selected reviews) ─────────────────────────────────
        if train_pmc_fulltext:
            pmc_ids = search_pmc_fulltext(query, 5, api_key, session)
            for pmc_id in pmc_ids:
                if pmc_id in trained_pmc:
                    continue
                full_text = fetch_pmc_fulltext(pmc_id, api_key, session)
                if full_text and len(full_text) > 500:
                    text = (f'PEER-REVIEWED FULL TEXT — Open Access\n'
                            f'Source: PubMed Central, National Library of Medicine\n'
                            f'Domain: {domain_label}\n'
                            f'PMC: {pmc_id}\n\n'
                            f'{full_text[:12000]}')  # cap at 12K chars per article
                    if _train(text, node, train_session):
                        ok += 1
                    trained_pmc.add(pmc_id)
                    time.sleep(0.08)

        if (qi + 1) % 10 == 0:
            save_ckpt()
            print(f'  [Stage {stage_num}] {ok} items trained so far...', flush=True)

    save_ckpt()
    print(f'  [Stage {stage_num}] Complete — {ok} items trained '
          f'({len(trained_pmids)} abstracts, {len(trained_pmc)} full texts)',
          flush=True)
    return [{'stage': stage_num, 'domain': domain_label, 'trained': ok,
             'type': 'pubmed_literature', 'modality': 'text',
             'tags': ['medical', 'peer-reviewed', 'ncbi', domain_label.lower()]}]


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Stages 30-33: Medical/psychological/genetic/longevity corpus from NCBI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(f'  {n}: {d}' for n, d in STAGES.items()),
    )
    ap.add_argument('--stages',        default='30,31,32,33')
    ap.add_argument('--node',          default=DEFAULT_NODE)
    ap.add_argument('--data-dir',      default=DEFAULT_DATA_DIR)
    ap.add_argument('--max-per-query', type=int, default=80,
                    help='Max PubMed results per search query (default 80)')
    ap.add_argument('--ncbi-api-key',  default=None,
                    help='Free NCBI API key (10 req/s vs 3/s without). '
                         'Get one at: https://www.ncbi.nlm.nih.gov/account/')
    ap.add_argument('--no-fulltext',   action='store_true',
                    help='Skip PMC full-text fetching (abstracts only, much faster)')
    args = ap.parse_args()

    stages    = {int(s.strip()) for s in args.stages.split(',')}
    train_dir = Path(args.data_dir) / 'training'
    api_key   = args.ncbi_api_key
    pmc_ft    = not args.no_fulltext

    print('Medical Corpus Builder — Stages 30-33')
    print(f'  Node        : {args.node}')
    print(f'  Stages      : {sorted(stages)}')
    print(f'  Max/query   : {args.max_per_query}')
    print(f'  NCBI API key: {"YES (10 req/s)" if api_key else "NO (3 req/s — get a free key!)"}')
    print(f'  Full text   : {"yes (PMC open access)" if pmc_ft else "no (abstracts only)"}')
    if not api_key:
        print('\n  TIP: Get a free NCBI API key at https://www.ncbi.nlm.nih.gov/account/')
        print('       Then re-run with --ncbi-api-key YOUR_KEY for 3× faster fetching.\n')

    ncbi_session  = _make_session()
    train_session = _make_session()
    all_items: dict = {}

    for stage_num in sorted(stages):
        if stage_num not in STAGE_QUERIES:
            print(f'  [SKIP] Stage {stage_num} not defined here')
            continue
        queries, label = STAGE_QUERIES[stage_num]
        print(f'\n[Stage {stage_num}] {label}')
        print(f'  {len(queries)} search queries, up to {args.max_per_query} results each')
        all_items[stage_num] = build_stage(
            stage_num, queries, label,
            train_dir / f'stage{stage_num}_{label.lower().split()[0]}',
            args.node, args.max_per_query, api_key, pmc_ft,
            ncbi_session, train_session,
        )

    manifest = [item for items in all_items.values() for item in items]
    mpath = train_dir / 'stage30_33_manifest.json'
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nManifest → {mpath}')
    print('Done.')


if __name__ == '__main__':
    main()
