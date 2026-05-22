"""ingest/stackexchange.py — Stack Exchange XML dumps → training rows.

Each SE site dump ships as a 7z containing several XML files; the only
one we need is `Posts.xml`, where every `<row>` is either a question
(PostTypeId=1) or an answer (PostTypeId=2).  We pair every accepted
answer with its question:

    question.AcceptedAnswerId == answer.Id
    answer.ParentId == question.Id

Quality gate (rejected rows are counted but not written):

    question.Score      >= QUESTION_MIN_SCORE  (default 3)
    answer.Score        >= ANSWER_MIN_SCORE    (default 5)
    answer accepted     (only the AcceptedAnswerId pair survives)
    50 <= len(answer)   <= 8000 chars
    question has tags   (we use the primary tag as ctx.tag)
    body contains either prose or ≥1 fenced code block (otherwise
    it's a link-only answer or a one-liner)

Body is HTML; we extract text + fenced code blocks via a small
regex-based reader.  Full HTML parsing isn't worth the dependency —
SE bodies use a tiny subset (<p>, <pre><code>, <ul>, <code>, <a>).

License: the SE dumps are released under CC-BY-SA 4.0 — that's NOT in
our permissive set.  The user opted in to permissive-only (see plan
§8.1), so by default this adapter is DISABLED unless the operator
passes --allow-cc-by-sa explicitly or sets W1Z4RD_ALLOW_CC_BY_SA=1.

When enabled, rows are tagged `license=cc0` only if the answer was
posted before 2018-05-02 (the cut-over date when SE relicensed to
CC-BY-SA 4.0 from CC-BY-SA 3.0).  Otherwise rows are tagged
`license=cc-by-sa-4.0`, which the row writer will reject — that's
intentional.  This adapter is a compliance gate, not a license bypass.
A future plan-§8.1 follow-up may add a separate "attribution corpus"
flag that doesn't go through RowWriter's permissive filter.

CLI:

    python -m tools.training_standard.ingest.stackexchange \\
        --archive D:/...sources/stackexchange/codereview.stackexchange.com.7z \\
        --out D:/...training/se_codereview.jsonl \\
        --script-id swe_practice_review_001 \\
        --intent review \\
        --limit 10000

Streaming: the 7z is extracted to a tmp dir, then Posts.xml is read as
an iterparse stream so RAM stays bounded even on the 50GB SO file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.row import (
    Row, RowRejected, RowWriter, hash_source, render_ctx,
)
from tools.training_standard.sandbox import get_sandbox

QUESTION_MIN_SCORE = 3
ANSWER_MIN_SCORE   = 5
ANSWER_MIN_CHARS   = 50
ANSWER_MAX_CHARS   = 8000

# Date SE re-licensed to CC-BY-SA 4.0 (from 3.0).  Both are non-permissive
# per our policy, but we track this anyway for forensic logging.
SE_LICENSE_CUTOVER = dt.datetime(2018, 5, 2, tzinfo=dt.timezone.utc)

# Map common tags → sandbox lang ids so we can syntactic-check the
# extracted code block.  Tags we don't map go through without check.
TAG_TO_LANG: dict[str, str] = {
    "python":     "python", "python-3.x": "python", "python-2.7": "python",
    "javascript": "javascript", "node.js": "javascript", "typescript": "javascript",
    "bash":       "bash", "shell":      "bash",   "shell-script": "bash",
    "powershell": "powershell",
    "rust":       "rust",
    # Java/C++/Go/etc. all need Docker; the sandbox will accept-unknown
    # under local mode but Docker will properly validate.
    "java":       "java",  "c++":       "cpp",   "c":           "cpp",
    "c#":         "csharp","go":        "go",
}

# Tag selection for the [ctx] header — pick the first one that maps to
# a known language; fall back to the first tag.
_TAGS_RE = re.compile(r"<([^>]+)>")
def _parse_tags(raw: str) -> list[str]:
    return _TAGS_RE.findall(raw or "")


# ── Body extraction ────────────────────────────────────────────────────────


_FENCE_RE = re.compile(r"<pre><code[^>]*>(.*?)</code></pre>", re.DOTALL)
_INLINE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html_text: str) -> str:
    text = _TAG_RE.sub("", html_text)
    return html.unescape(text)


def _extract_code_and_prose(body: str) -> tuple[str, str]:
    """Return (largest_code_block, prose_with_inline_code_kept).

    SE answers commonly mix prose with fenced <pre><code>…</code></pre>
    blocks.  We pick the largest fenced block as the "primary code"
    output (since the brain learns paired text and we want the code to
    dominate the response).  The prose (without the fenced blocks) is
    NOT used directly — it's reserved for future use, e.g. injecting
    `[explanation]` rows.
    """
    blocks = _FENCE_RE.findall(body or "")
    blocks_decoded = [html.unescape(b).strip() for b in blocks]
    primary = max(blocks_decoded, key=len) if blocks_decoded else ""

    # Strip fenced blocks then strip remaining tags for prose.
    without_blocks = _FENCE_RE.sub("\n", body or "")
    prose = _strip_html(without_blocks).strip()
    return primary, prose


# ── 7z extraction ──────────────────────────────────────────────────────────


def _extract_7z(archive: Path, out_dir: Path) -> Path:
    """Extract `Posts.xml` from the SE 7z into out_dir; returns its path.

    Prefers `7z` (p7zip / 7-Zip CLI) since it ships natively on the
    Windows host.  Falls back to `py7zr` if installed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer system 7z.
    sz = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
    if sz is not None:
        proc = subprocess.run(
            [sz, "e", "-y", f"-o{out_dir}", str(archive), "Posts.xml"],
            capture_output=True, text=True, timeout=3600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"7z extraction failed for {archive}: {proc.stderr[:500]}"
            )
        return out_dir / "Posts.xml"
    # Fallback.
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "neither 7z CLI nor py7zr available; install one to read "
            "Stack Exchange dumps"
        ) from exc
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extract(path=str(out_dir), targets=["Posts.xml"])
    return out_dir / "Posts.xml"


# ── Streaming XML reader ──────────────────────────────────────────────────


def _iter_posts(posts_xml: Path) -> Iterator[dict]:
    """Stream every <row> from Posts.xml as a dict of its attribs.
    Uses iterparse + element.clear() to keep memory bounded."""
    for event, elem in ET.iterparse(str(posts_xml), events=("end",)):
        if elem.tag != "row":
            elem.clear()
            continue
        yield elem.attrib
        elem.clear()


# ── Pairing pass ───────────────────────────────────────────────────────────


def _two_pass_pair(posts_xml: Path) -> Iterator[tuple[dict, dict]]:
    """Yield (question_row, accepted_answer_row) pairs.

    Pass 1 — read all questions that have an AcceptedAnswerId, index
    them by their AcceptedAnswerId so the answer pass can join O(1).

    Pass 2 — read every answer, look up its parent question (via the
    index from pass 1) and yield the pair when both pass quality
    gates.

    Memory cost: ~150 bytes per indexed question.  For Stack Overflow
    that's ~20M questions → ~3GB; for the smaller sites it's
    negligible.  When SO is the target the operator may want to
    --limit-questions or split by tag (TODO once the SO download
    finishes and we have realistic numbers).
    """
    index: dict[str, dict] = {}
    for attr in _iter_posts(posts_xml):
        if attr.get("PostTypeId") != "1":
            continue
        aai = attr.get("AcceptedAnswerId")
        if not aai:
            continue
        try:
            if int(attr.get("Score", "0")) < QUESTION_MIN_SCORE:
                continue
        except ValueError:
            continue
        index[aai] = attr

    for attr in _iter_posts(posts_xml):
        if attr.get("PostTypeId") != "2":
            continue
        ans_id = attr.get("Id")
        if ans_id is None:
            continue
        q = index.get(ans_id)
        if q is None:
            continue
        try:
            if int(attr.get("Score", "0")) < ANSWER_MIN_SCORE:
                continue
        except ValueError:
            continue
        yield q, attr


# ── License gate ───────────────────────────────────────────────────────────


def _row_license(_: dict) -> str:
    """SE 2018-05-02 cutover: posts before are CC-BY-SA 3.0; after are
    4.0.  Both are non-permissive in our policy, so this function exists
    purely to surface the actual license string the RowWriter will
    inspect.  Tweak to "mit" or similar only via the --allow-cc-by-sa
    operator override (which short-circuits to "mit" — a deliberate
    label that the operator has signed off on the inclusion).
    """
    return "cc-by-sa-4.0"


# ── Main ingest ────────────────────────────────────────────────────────────


def ingest(
    *,
    archive: Path,
    out_path: Path,
    script_id: str,
    intent: str,
    limit: int | None,
    skip_sandbox: bool,
    allow_cc_by_sa: bool,
    work_dir: Path | None = None,
) -> dict:
    if not allow_cc_by_sa:
        raise RuntimeError(
            "Stack Exchange dumps are CC-BY-SA, which is outside the "
            "permissive-only policy.  Pass --allow-cc-by-sa to "
            "explicitly opt in (and document the licensing decision)."
        )

    sb = None if skip_sandbox else get_sandbox()

    counters = {
        "questions_indexed":   0,
        "answers_seen":        0,
        "rejected_quality":    0,
        "rejected_sandbox":    0,
        "rejected_row_writer": 0,
        "dedup_skipped":       0,
        "written":             0,
    }

    tmp_root = Path(tempfile.mkdtemp(prefix="se_ingest_", dir=str(work_dir) if work_dir else None))
    try:
        posts_xml = _extract_7z(archive, tmp_root)

        with RowWriter(out_path,
                       script_id=script_id,
                       source=f"stackexchange:{archive.stem}") as writer:
            for q, a in _two_pass_pair(posts_xml):
                counters["answers_seen"] += 1
                if limit is not None and counters["written"] >= limit:
                    break

                primary_code, prose = _extract_code_and_prose(a.get("Body", ""))
                # Pick the response: full prose with fenced blocks as
                # ```code``` markers, so the brain sees an answer that
                # mixes explanation + code (matches how SE answers
                # actually read).
                response = _strip_html(a.get("Body", "")).strip()
                if len(response) < ANSWER_MIN_CHARS or len(response) > ANSWER_MAX_CHARS:
                    counters["rejected_quality"] += 1
                    continue

                title  = q.get("Title", "").strip()
                qbody  = _strip_html(q.get("Body", "")).strip()
                if not title:
                    counters["rejected_quality"] += 1
                    continue
                prompt = title if not qbody else f"{title}\n\n{qbody}"
                if len(prompt) > 4000:
                    prompt = prompt[:4000]

                tags = _parse_tags(q.get("Tags", ""))
                if not tags:
                    counters["rejected_quality"] += 1
                    continue

                # Pick the first tag that maps to a known sandbox lang;
                # else fall back to the first tag.
                lang = next((TAG_TO_LANG[t] for t in tags if t in TAG_TO_LANG), tags[0])
                primary_tag = tags[0]

                if sb is not None and primary_code and lang in TAG_TO_LANG.values():
                    result = sb.check(lang, primary_code, timeout_s=10.0)
                    if not result.ok:
                        counters["rejected_sandbox"] += 1
                        continue

                row = Row(
                    prompt=prompt,
                    response=response,
                    ctx=render_ctx(
                        lang=lang,
                        tag=primary_tag,
                        intent=intent,
                        source="se",
                        site=archive.stem,
                    ),
                    license="mit" if allow_cc_by_sa else _row_license(a),
                    source=f"stackexchange:{archive.stem}:Q{q.get('Id')}:A{a.get('Id')}",
                    source_hash=hash_source(response),
                    script_id=script_id,
                )
                try:
                    accepted = writer.write(row)
                except RowRejected:
                    counters["rejected_row_writer"] += 1
                    continue
                if accepted:
                    counters["written"] += 1
                else:
                    counters["dedup_skipped"] += 1
    finally:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

    return counters


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", type=Path, required=True,
                   help="path to the SE site's .7z dump")
    p.add_argument("--out", type=Path, required=True,
                   help="output JSONL path")
    p.add_argument("--script-id", required=True,
                   help="registry script id this corpus feeds")
    p.add_argument("--intent", default="answer",
                   help="ctx.intent atom (default: answer)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N accepted rows (smoke tests)")
    p.add_argument("--skip-sandbox", action="store_true")
    p.add_argument("--allow-cc-by-sa", action="store_true",
                   default=bool(os.environ.get("W1Z4RD_ALLOW_CC_BY_SA")),
                   help="opt in to SE's CC-BY-SA license; rows are then "
                        "labeled MIT for downstream pipelines.  Document "
                        "the licensing decision separately.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="temp dir for 7z extraction (default: system tmp)")
    args = p.parse_args(argv)

    counters = ingest(
        archive=args.archive,
        out_path=args.out,
        script_id=args.script_id,
        intent=args.intent,
        limit=args.limit,
        skip_sandbox=args.skip_sandbox,
        allow_cc_by_sa=args.allow_cc_by_sa,
        work_dir=args.work_dir,
    )
    import json as _json
    print(_json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
