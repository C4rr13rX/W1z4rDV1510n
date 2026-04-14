"""
W1z4rDV1510n Research Agent
============================
Polls GET /hypothesis/queue for unresolved questions, fetches authoritative
educational sources (Wikipedia API, ArXiv, NCBI E-utilities, LibreTexts),
trains the node with the results via /qa/ingest + /media/train, then marks
the hypothesis resolved via POST /hypothesis/resolve.

Adapted from CoolCryptoUtilities/services/polite_news_crawler.py infrastructure.

Usage:
    python scripts/research_agent.py [--node http://localhost:8090] [--interval 30]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

import aiohttp

# ── Configuration ─────────────────────────────────────────────────────────────

NODE_API = "http://localhost:8090"
POLL_INTERVAL_SECS = 30
MAX_RETRIES = 2
TIMEOUT_SEC = 15

# Educational sources (all explicitly allow API access)
SOURCES = {
    "wikipedia": {
        "name": "Wikipedia REST API",
        "base": "https://en.wikipedia.org/api/rest_v1/page/summary/",
        "search": "https://en.wikipedia.org/w/api.php",
    },
    "arxiv": {
        "name": "ArXiv API",
        "base": "http://export.arxiv.org/api/query",
    },
    "ncbi": {
        "name": "NCBI E-utilities",
        "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        "tool": "W1z4rDV1510n",
        "email": "research@carrierx.dev",
    },
}

USER_AGENT = "W1z4rDResearchAgent/1.0 (educational; contact: research@carrierx.dev)"

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class HypothesisEntry:
    id: str
    question: str
    queued_at_unix: int
    attempts: int
    max_attempts: int
    resolved: bool
    answer: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ResearchResult:
    question: str
    answer: str
    confidence: float
    source: str
    source_url: str = ""


# ── Node API client ───────────────────────────────────────────────────────────

class NodeClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        self.base = base_url.rstrip("/")
        self.s = session

    async def get_hypothesis_queue(self) -> List[HypothesisEntry]:
        try:
            async with self.s.get(f"{self.base}/hypothesis/queue", timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                return [
                    HypothesisEntry(**{k: v for k, v in e.items() if k in HypothesisEntry.__dataclass_fields__})
                    for e in data.get("entries", [])
                    if not e.get("resolved", False)
                ]
        except Exception as exc:
            print(f"[warn] hypothesis queue fetch failed: {exc}", file=sys.stderr)
            return []

    async def resolve_hypothesis(self, hyp_id: str, answer: str, confidence: float) -> bool:
        payload = {"id": hyp_id, "answer": answer, "confidence": confidence}
        try:
            async with self.s.post(
                f"{self.base}/hypothesis/resolve",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                return r.status == 200
        except Exception as exc:
            print(f"[warn] resolve failed for {hyp_id}: {exc}", file=sys.stderr)
            return False

    async def ingest_qa(self, question: str, answer: str, confidence: float, source: str) -> bool:
        payload = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "book_id": source,
            "page_index": 0,
        }
        try:
            async with self.s.post(
                f"{self.base}/qa/ingest",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                return r.status == 200
        except Exception as exc:
            print(f"[warn] qa/ingest failed: {exc}", file=sys.stderr)
            return False

    async def train_text(self, text: str) -> bool:
        payload = {"modality": "text", "text": text}
        try:
            async with self.s.post(
                f"{self.base}/media/train",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                return r.status == 200
        except Exception as exc:
            print(f"[warn] media/train failed: {exc}", file=sys.stderr)
            return False

    async def checkpoint(self) -> bool:
        try:
            async with self.s.post(
                f"{self.base}/qa/checkpoint",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                return r.status == 200
        except Exception:
            return False


# ── Knowledge fetchers ────────────────────────────────────────────────────────

class WikipediaFetcher:
    """Fetches Wikipedia summaries via the MediaWiki REST API."""

    BASE_SEARCH = "https://en.wikipedia.org/w/api.php"
    BASE_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(self, session: aiohttp.ClientSession):
        self.s = session

    async def search(self, query: str) -> Optional[str]:
        """Find the best matching page title for a query."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
            "origin": "*",
        }
        try:
            async with self.s.get(
                self.BASE_SEARCH,
                params=params,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                results = data.get("query", {}).get("search", [])
                if not results:
                    return None
                return results[0]["title"]
        except Exception:
            return None

    async def fetch_summary(self, title: str) -> Optional[str]:
        """Fetch the lead-section summary of a Wikipedia article."""
        url = self.BASE_SUMMARY + quote(title.replace(" ", "_"), safe="")
        try:
            async with self.s.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC)) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                extract = data.get("extract", "").strip()
                # Limit to ~600 chars to avoid giant training blobs
                return extract[:600] if extract else None
        except Exception:
            return None

    async def answer(self, question: str) -> Optional[ResearchResult]:
        # Strip question words for better search
        q = re.sub(r"^(what|how|why|when|where|who|is|are|does|do|can)\s+", "", question, flags=re.I)
        q = q.rstrip("?!.").strip()
        title = await self.search(q)
        if not title:
            return None
        summary = await self.fetch_summary(title)
        if not summary or len(summary) < 40:
            return None
        url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        return ResearchResult(
            question=question,
            answer=summary,
            confidence=0.80,
            source="wikipedia",
            source_url=url,
        )


class ArXivFetcher:
    """Fetches ArXiv abstracts for science/math questions."""

    BASE = "http://export.arxiv.org/api/query"

    def __init__(self, session: aiohttp.ClientSession):
        self.s = session

    async def answer(self, question: str) -> Optional[ResearchResult]:
        q = re.sub(r"^(what|how|why|when|where|who|is|are|does|do)\s+", "", question, flags=re.I)
        q = q.rstrip("?!.").strip()
        params = {
            "search_query": f"all:{q}",
            "start": 0,
            "max_results": 3,
            "sortBy": "relevance",
        }
        try:
            async with self.s.get(
                self.BASE,
                params=params,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
            ) as r:
                if r.status != 200:
                    return None
                text = await r.text()
        except Exception:
            return None

        # Parse Atom XML — find the first entry's summary
        try:
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(text)
            entries = root.findall("atom:entry", ns)
            if not entries:
                return None
            entry = entries[0]
            summary_el = entry.find("atom:summary", ns)
            title_el = entry.find("atom:title", ns)
            if summary_el is None:
                return None
            summary = (summary_el.text or "").strip()[:600]
            title = (title_el.text or "").strip() if title_el is not None else ""
            if len(summary) < 40:
                return None
            link_el = entry.find("atom:id", ns)
            url = link_el.text if link_el is not None else "https://arxiv.org"
            answer_text = f"{title}: {summary}" if title else summary
            return ResearchResult(
                question=question,
                answer=answer_text,
                confidence=0.70,
                source="arxiv",
                source_url=url,
            )
        except Exception:
            return None


# ── Research orchestrator ─────────────────────────────────────────────────────

class ResearchAgent:
    def __init__(self, node_url: str, poll_interval: int):
        self.node_url = node_url
        self.poll_interval = poll_interval

    async def run(self):
        headers = {"User-Agent": USER_AGENT}
        async with aiohttp.ClientSession(headers=headers) as session:
            node = NodeClient(self.node_url, session)
            wiki = WikipediaFetcher(session)
            arxiv = ArXivFetcher(session)
            fetchers = [wiki, arxiv]

            print(f"[research_agent] started — node={self.node_url}  poll={self.poll_interval}s")
            while True:
                await self._cycle(node, fetchers)
                await asyncio.sleep(self.poll_interval)

    async def _cycle(self, node: NodeClient, fetchers):
        hypotheses = await node.get_hypothesis_queue()
        pending = [h for h in hypotheses if not h.resolved and h.attempts < h.max_attempts]
        if not pending:
            return

        print(f"[research_agent] {len(pending)} unresolved hypotheses")
        resolved_any = False

        for hyp in pending[:5]:  # max 5 per cycle to stay polite
            print(f"  → researching: {hyp.question!r}")
            result: Optional[ResearchResult] = None

            for fetcher in fetchers:
                try:
                    result = await fetcher.answer(hyp.question)
                    if result:
                        break
                except Exception as exc:
                    print(f"    [warn] fetcher {fetcher.__class__.__name__} failed: {exc}", file=sys.stderr)

            if not result:
                print(f"    no result found for {hyp.question!r}")
                continue

            print(f"    found ({result.source}): {result.answer[:80]!r}…")

            # Feed into the node
            trained = await node.train_text(result.answer)
            ingested = await node.ingest_qa(hyp.question, result.answer, result.confidence, result.source)
            if not (trained or ingested):
                print(f"    [warn] training failed for {hyp.question!r}")
                continue

            # Mark resolved
            ok = await node.resolve_hypothesis(hyp.id, result.answer, result.confidence)
            if ok:
                print(f"    ✓ resolved {hyp.id}")
                resolved_any = True
            else:
                print(f"    [warn] resolve endpoint failed for {hyp.id}")

            # Rate limit: be polite to external APIs
            await asyncio.sleep(2.0)

        if resolved_any:
            await node.checkpoint()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="W1z4rDV1510n Research Agent")
    parser.add_argument("--node", default=NODE_API, help="Node API base URL")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL_SECS,
                        help="Poll interval in seconds (default: 30)")
    args = parser.parse_args()

    agent = ResearchAgent(args.node, args.interval)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
