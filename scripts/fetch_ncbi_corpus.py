#!/usr/bin/env python3
"""
fetch_ncbi_corpus.py — pull measurement-heavy passages from PubMed/PMC
on the topics the node should learn.

Topics (from the user directive):
  - neuroscience
  - biofields
  - genetics
  - ultradian rhythms
  - sub-ultradian rhythms
  - reversing cellular aging

Pipeline:
  1. esearch on each topic to get PMC IDs (rate-limited, max 100/topic).
  2. efetch full text in JATS XML (PMC open-access subset only).
  3. parse abstract + body paragraphs.
  4. score paragraphs by *measurement density* (counts of numbers + units).
  5. keep top-N paragraphs per topic.
  6. split each kept paragraph into (question, answer) pairs by sentence-window
     pattern: a question is one sentence, the answer is the next 1-2 sentences.

The output is JSONL at data/foundation/ncbi_pairs.jsonl with fields:
    {"topic": str, "pmcid": str, "question": str, "answer": str,
     "measurement_density": float}

Polite to NCBI: 3 req/s without API key, 10 req/s with one (set NCBI_API_KEY).
Resumable: skips PMC IDs that are already cached locally.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NCBI_BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_BASE    = "https://www.ncbi.nlm.nih.gov/pmc/articles"

TOPICS = {
    "neuroscience":           "(\"neural circuits\"[Title/Abstract] OR \"hebbian\"[Title/Abstract] OR \"synaptic plasticity\"[Title/Abstract] OR \"action potential\"[Title/Abstract]) AND \"open access\"[Filter]",
    "biofields":              "(\"biofield\"[Title/Abstract] OR \"bioelectromagnetic field\"[Title/Abstract] OR \"endogenous electromagnetic\"[Title/Abstract]) AND \"open access\"[Filter]",
    "genetics":               "(\"gene expression\"[Title/Abstract] OR \"transcription factor\"[Title/Abstract] OR \"epigenetic\"[Title/Abstract] OR \"CRISPR\"[Title/Abstract]) AND \"open access\"[Filter]",
    "ultradian_rhythms":      "(\"ultradian rhythm\"[Title/Abstract] OR \"ultradian oscillation\"[Title/Abstract]) AND \"open access\"[Filter]",
    "sub_ultradian_rhythms":  "(\"infradian\"[Title/Abstract] OR \"sub-ultradian\"[Title/Abstract] OR \"high-frequency oscillation\"[Title/Abstract]) AND \"open access\"[Filter]",
    "cellular_aging_reversal":"(\"aging reversal\"[Title/Abstract] OR \"cellular reprogramming\"[Title/Abstract] OR \"yamanaka factors\"[Title/Abstract] OR \"senescence reversal\"[Title/Abstract]) AND \"open access\"[Filter]",
}

# Number+unit regex: matches "12.3 ms", "5 hz", "0.04 mol/L", "100 mV", etc.
MEASUREMENT_RE = re.compile(
    r"\b(\d+\.\d+|\d+)\s*(?:%|ms|s|min|hr|hz|khz|mhz|ghz|kg|g|mg|ug|µg|ng|"
    r"m|cm|mm|um|µm|nm|mol|mmol|umol|µmol|nmol|m\/s|hz|mv|v|kv|ka|ma|ua|µa|"
    r"na|j|kj|w|kw|n|kn|pa|kpa|mpa|°c|°f|c|k|cd|lm|lx|au|Wb|T)\b",
    flags=re.IGNORECASE,
)

OUT_DIR  = pathlib.Path("data/foundation")
PMC_DIR  = OUT_DIR / "pmc_cache"
JSONL    = OUT_DIR / "ncbi_pairs.jsonl"


def http_get(url: str, timeout: int = 30) -> str:
    api_key = os.environ.get("NCBI_API_KEY", "")
    sep = "&" if "?" in url else "?"
    if api_key:
        url = f"{url}{sep}api_key={api_key}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "w1z4rdv1510n-ncbi-fetch/0.1 (research, contact node owner)"
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def esearch_pmc(query: str, retmax: int = 100) -> list[str]:
    """Return list of PMC IDs (numeric strings, no PMC prefix)."""
    url = (f"{NCBI_BASE}/esearch.fcgi?db=pmc"
           f"&term={urllib.parse.quote(query)}&retmax={retmax}&retmode=json")
    raw = http_get(url)
    try:
        j = json.loads(raw)
        return j.get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def efetch_pmc_xml(pmcid: str) -> str | None:
    cache = PMC_DIR / f"{pmcid}.xml"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    url = f"{NCBI_BASE}/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
    try:
        xml = http_get(url, timeout=60)
    except Exception as e:
        print(f"  efetch {pmcid}: {e}", file=sys.stderr)
        return None
    if not xml or len(xml) < 200:
        return None
    cache.write_text(xml, encoding="utf-8")
    return xml


def extract_paragraphs(xml_text: str) -> list[str]:
    """Pull every <p> text from the JATS body + abstract, flattened."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    paras = []
    for p in root.iter():
        # JATS uses unprefixed tags; some PMC dumps use namespaces.
        tag = p.tag.split("}")[-1] if "}" in p.tag else p.tag
        if tag != "p":
            continue
        txt = "".join(p.itertext()).strip()
        txt = re.sub(r"\s+", " ", txt)
        if 200 <= len(txt) <= 2000:
            paras.append(txt)
    return paras


def measurement_density(text: str) -> float:
    """Count of measurement tokens divided by sentence count."""
    matches = MEASUREMENT_RE.findall(text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    return len(matches) / sentences


def split_into_pairs(text: str) -> list[tuple[str, str]]:
    """Crude sentence-pair extraction:
       a question is sentence i if it ends with '?'; answer is sentence i+1..i+2.
       If no questions, take consecutive sentence pairs as (statement, follow-up)
       which the multi-pool fabric can still learn as bidirectional associations.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
    pairs = []
    if any(s.endswith("?") for s in sents):
        for i, s in enumerate(sents):
            if s.endswith("?") and i + 1 < len(sents):
                ans = " ".join(sents[i+1:i+3])[:400]
                pairs.append((s[:200], ans))
    if not pairs:
        # Sliding window: pair consecutive sentences.
        for i in range(0, len(sents) - 1, 2):
            pairs.append((sents[i][:200], sents[i+1][:400]))
    return pairs[:8]  # cap per paragraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retmax", type=int, default=20,
                    help="max articles per topic (default 20)")
    ap.add_argument("--max-pairs-per-topic", type=int, default=200)
    ap.add_argument("--rate-secs", type=float, default=0.4,
                    help="sleep between requests (0.4=2.5/s; 0.1 with API key)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PMC_DIR.mkdir(exist_ok=True)

    # We append fresh, but keep the record of what we already have for resume.
    seen_ids: set[str] = set()
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            try:
                seen_ids.add(json.loads(line)["pmcid"])
            except Exception:
                pass
    print(f"Resuming from {len(seen_ids)} cached pmcids" if seen_ids else "fresh run")

    written = 0
    with JSONL.open("a", encoding="utf-8") as out:
        for topic, query in TOPICS.items():
            print(f"\n[{topic}] esearch ...")
            ids = esearch_pmc(query, retmax=args.retmax)
            print(f"  {len(ids)} pmcids returned")
            time.sleep(args.rate_secs)

            topic_pairs = 0
            for pmcid in ids:
                if pmcid in seen_ids:
                    continue
                if topic_pairs >= args.max_pairs_per_topic:
                    break
                xml = efetch_pmc_xml(pmcid)
                time.sleep(args.rate_secs)
                if xml is None:
                    continue

                paras = extract_paragraphs(xml)
                # Top-N most measurement-dense paragraphs.
                paras_scored = sorted(paras,
                    key=measurement_density, reverse=True)[:6]
                for para in paras_scored:
                    md = measurement_density(para)
                    if md < 0.3:
                        continue  # not measurement-heavy enough
                    for q, a in split_into_pairs(para):
                        if not q or not a:
                            continue
                        out.write(json.dumps({
                            "topic":   topic,
                            "pmcid":   pmcid,
                            "question": q,
                            "answer":   a,
                            "measurement_density": round(md, 3),
                        }) + "\n")
                        written += 1
                        topic_pairs += 1
                seen_ids.add(pmcid)
            print(f"  topic done: {topic_pairs} new pairs (running total {written})")

    print(f"\nWrote {written} new pairs to {JSONL}")
    if JSONL.exists():
        n_total = sum(1 for _ in JSONL.open(encoding="utf-8"))
        print(f"Corpus size now {n_total} pairs")


if __name__ == "__main__":
    main()
