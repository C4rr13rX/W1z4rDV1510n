#!/usr/bin/env python3
"""Fetch PubMed Central open-access articles whose licence is verifiable.

Textbooks cover the domains a curriculum names, but some subjects have no
commercially usable textbook at all -- nanotechnology is the case that
prompted this. The peer-reviewed literature does have them, and PMC's Open
Access service states each article's licence as a machine-readable attribute
rather than as prose to be parsed:

    <record id="PMC3045931" license="CC BY" ...>

That is a stronger guarantee than any of the PDF heuristics elsewhere in this
package: it is the publisher's own declaration, per article, and it is
checked here for every single article before a word of its text is kept.

Articles whose licence is anything other than CC BY / CC BY-SA / CC0 are
skipped, and so are articles the service declines to describe at all -- an
unknown licence is treated as a refusal, never as permission.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ..row import Row, RowWriter, hash_source

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

#: Licences that permit commercial use. Matched against the OA service's own
#: `license` attribute, which uses these exact short forms.
COMMERCIAL_LICENCES = {
    "CC BY": "cc-by-4.0",
    "CC BY-SA": "cc-by-sa-4.0",
    "CC0": "cc0-1.0",
}

#: NCBI asks for no more than 3 requests/second without an API key.
REQUEST_INTERVAL = 0.4

#: An abstract shorter than this is a stub -- a correction notice or an
#: editorial fragment -- and teaches nothing.
MIN_ABSTRACT = 320
MAX_ABSTRACT = 8000


def fetch(url: str, timeout: int = 90) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def search(term: str, retmax: int) -> list[str]:
    """PMC ids for a query, newest first."""
    url = (f"{EUTILS}/esearch.fcgi?db=pmc&term={urllib.parse.quote(term)}"
           f"&retmax={retmax}&retmode=json&sort=pub+date")
    try:
        payload = json.loads(fetch(url))
    except Exception:
        return []
    return list(payload.get("esearchresult", {}).get("idlist", []))


def licence_of(pmcid: str) -> str | None:
    """The SPDX id for an article, or None when it is not commercial-safe.

    The OA service is the authority. Silence from it -- a missing record, a
    missing attribute, an error -- means the licence is unknown, and unknown
    is refused rather than assumed.
    """
    try:
        body = fetch(f"{OA_SERVICE}?id=PMC{pmcid}")
    except Exception:
        return None
    match = re.search(r'license="([^"]+)"', body)
    if not match:
        return None
    return COMMERCIAL_LICENCES.get(match.group(1).strip().upper().replace(
        "CC-BY", "CC BY"))


def article_text(pmcid: str) -> tuple[str, str] | None:
    """(title, abstract) for one article, from the EFetch XML."""
    try:
        xml = fetch(f"{EUTILS}/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml")
    except Exception:
        return None
    title = re.search(r"<article-title[^>]*>(.*?)</article-title>", xml,
                      re.S | re.I)
    abstract = re.search(r"<abstract[^>]*>(.*?)</abstract>", xml, re.S | re.I)
    if not title or not abstract:
        return None

    def plain(raw: str) -> str:
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = (raw.replace("&amp;", "&").replace("&lt;", "<")
                  .replace("&gt;", ">").replace("&quot;", '"')
                  .replace("&#x000a0;", " ").replace("&#x2019;", "'"))
        return re.sub(r"\s+", " ", raw).strip()

    return plain(title.group(1)), plain(abstract.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", required=True,
                        help='PMC query, e.g. \'"Beilstein J Nanotechnol"[jour]\'')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="pmc")
    parser.add_argument("--script-id", default="domain_pmc_001")
    parser.add_argument("--intent", default="explain")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    ids = search(args.term, args.limit)
    print(f"{len(ids)} articles matched", flush=True)

    written = refused = unusable = 0
    writer = RowWriter(args.out, script_id=args.script_id, source=args.label)
    try:
        for index, pmcid in enumerate(ids, start=1):
            time.sleep(REQUEST_INTERVAL)
            licence = licence_of(pmcid)
            if licence is None:
                refused += 1
                continue
            time.sleep(REQUEST_INTERVAL)
            parsed = article_text(pmcid)
            if parsed is None:
                unusable += 1
                continue
            title, abstract = parsed
            if not (MIN_ABSTRACT <= len(abstract) <= MAX_ABSTRACT) or not title:
                unusable += 1
                continue
            source = f"{args.label}:PMC{pmcid}"
            row = Row(
                prompt=title,
                response=abstract,
                ctx={"lang": "text", "intent": args.intent,
                     "source": args.label},
                license=licence,
                source=source,
                source_hash=hash_source(f"{title}\n{abstract}"),
                script_id=args.script_id,
            )
            if writer.write(row):
                written += 1
            if index % 50 == 0:
                print(f"  {index}/{len(ids)} written={written} "
                      f"refused={refused}", flush=True)
    finally:
        writer.close()

    print(f"wrote {args.out}\n  rows written {written}\n"
          f"  refused (licence unknown or restricted) {refused}\n"
          f"  unusable (no abstract, or out of size bounds) {unusable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
