#!/usr/bin/env python3
"""Compile a permissively licensed HuggingFace dataset into training rows.

Textbooks state conclusions. Measured across the 30,924 textbook and article
rows already onboarded, the corpus is dense in assertion -- 29,893 quantified
claims, 18,914 causal statements -- and thin in the shapes that DERIVE a
conclusion: 1,507 if/then conditionals, 1,258 pieces of proof language.

That gap is what this path fills. The licence is read from the Hub's
structured metadata rather than parsed out of prose, and a dataset whose
licence is not on the permissive list is refused before a single row is
fetched.

The deduction datasets are the reason this exists. A ProofWriter row gives
plain-English facts and rules, asks whether some statement follows, and
accepts three answers -- True, False, and **Unknown**. `Unknown` is not a
gap in the data; it is the boundary itself, the case where nothing in the
premises reaches the question, and the correct behaviour is to decline. That
is the property the science-scope benchmark measures, with labelled ground
truth attached.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..row import Row, RowWriter, hash_source

HUB = "https://huggingface.co/api/datasets"
SERVER = "https://datasets-server.huggingface.co"

#: Licences that permit commercial use, as the Hub spells them.
PERMISSIVE = {
    "mit": "mit",
    "apache-2.0": "apache-2.0",
    "bsd-3-clause": "bsd-3-clause",
    "bsd-2-clause": "bsd-2-clause",
    "cc0-1.0": "cc0-1.0",
    "cc-by-4.0": "cc-by-4.0",
    "cc-by-3.0": "cc-by-3.0",
    "cc-by-sa-4.0": "cc-by-sa-4.0",
    "odc-by": "odc-by",
}

#: Rows per request the datasets-server allows.
PAGE = 100

#: A quote acting as a list-member delimiter: one at the very start or end of
#: the flattened text, or one adjacent to the whitespace between members.
#: Anchored this way so an apostrophe inside a sentence ("Anne's") survives.
QUOTE_DELIMITER = re.compile(r"^['\"]|['\"]$|['\"](?=\s)|(?<=\s)['\"]")


def fetch(url: str, timeout: int = 120) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def licence_of(dataset: str) -> str | None:
    """The SPDX id a dataset declares, or None when it is not usable.

    Read from the Hub's own tags. A dataset with no licence tag is refused:
    unstated rights are not permissive ones.
    """
    try:
        info = fetch(f"{HUB}/{dataset}")
    except Exception:
        return None
    for tag in info.get("tags") or []:
        if tag.startswith("license:"):
            return PERMISSIVE.get(tag.split(":", 1)[1].strip().lower())
    return None


def as_text(value) -> str:
    """Flatten a cell to prose.

    Array columns arrive two ways from the datasets-server: as a real list,
    and as a numpy array ALREADY STRINGIFIED into "['a.' 'b.']". The second
    form is the one that bites -- flattening a list handles the first, while
    the second slips through as a literal and puts brackets and quotes into
    the prompt, which no reader would ever type.
    """
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(item) for item in value)
    text = str(value or "")
    if text.startswith("[") and text.endswith("]"):
        # Drop the brackets, then the quotes that delimit members.
        #
        # Matching quoted members with a regex looked cleaner but was wrong:
        # "['It is Anne's.' 'Bob is here.']" has an apostrophe INSIDE a
        # member, so a quote-pair pattern stops early and silently returns
        # "It is Anne". Removing only the delimiters cannot lose text that
        # way -- the worst case is a stray quote character, not a truncated
        # sentence.
        text = text[1:-1]
        text = QUOTE_DELIMITER.sub(" ", text)
    return " ".join(text.split())


def parquet_rows(dataset: str, split: str):
    """Every row of a dataset, from its single parquet file.

    Paging `/rows` 100 at a time is what the API offers, but it rate-limits:
    a 16,449-row dataset truncated at offset 5,200 with a 0.1 s delay and at
    2,900 with exponential backoff, silently yielding a third of the data and
    calling it done. The same dataset is one 3 MB parquet download.

    Yields nothing when parquet is unavailable, so the caller can fall back.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return
    try:
        listing = fetch(f"{SERVER}/parquet?dataset={urllib.parse.quote(dataset)}")
    except Exception:
        return
    urls = [f["url"] for f in listing.get("parquet_files", [])
            if f.get("split") == split]
    for url in urls:
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = response.read()
            table = parquet.read_table(io.BytesIO(payload))
        except Exception as error:
            print(f"  parquet read failed: {error}", flush=True)
            return
        for batch in table.to_pylist():
            yield batch


def deduction_pair(row: dict) -> tuple[str, str] | None:
    """A prompt/response pair for a ProofWriter-shaped row.

    The premises are restated in the prompt because a question alone is not
    answerable: the whole point is that the answer depends on exactly the
    facts and rules given, and on nothing else.
    """
    question = as_text(row.get("question"))
    answer = as_text(row.get("answer"))
    if not question or not answer:
        return None
    facts = as_text(row.get("facts") or row.get("theory"))
    rules = as_text(row.get("rules"))
    premises = " ".join(part for part in (facts, rules) if part)
    if not premises:
        return None
    prompt = (f"Given only these statements: {premises} "
              f"Does it follow that: {question}")
    # Spell the verdict out. "Unknown" as a bare token teaches a label;
    # the sentence teaches the behaviour -- that the premises do not reach
    # the question, so the honest answer is to decline.
    verdict = {
        "true": "Yes, that follows from the statements given.",
        "false": "No, that contradicts the statements given.",
        "unknown": ("Unknown. The statements given do not settle this "
                    "question either way."),
        "uncertain": ("Unknown. The statements given do not settle this "
                      "question either way."),
    }.get(answer.strip().lower())
    if verdict is None:
        return None
    return prompt, verdict


def generic_pair(row: dict, prompt_key: str, response_key: str):
    prompt = as_text(row.get(prompt_key))
    response = as_text(row.get(response_key))
    if not prompt or not response:
        return None
    return prompt, response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="e.g. longface/ProofWriter")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--script-id", required=True)
    parser.add_argument("--intent", default="reason")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--shape", choices=("deduction", "generic"),
                        default="deduction")
    parser.add_argument("--prompt-key", default="question")
    parser.add_argument("--response-key", default="answer")
    args = parser.parse_args()

    licence = licence_of(args.dataset)
    if licence is None:
        print(f"refusing {args.dataset}: licence absent or not commercial-safe",
              file=sys.stderr)
        return 2
    print(f"{args.dataset} licence={licence}", flush=True)

    written = skipped = 0
    writer = RowWriter(args.out, script_id=args.script_id, source=args.label)

    def emit(record: dict, index: int) -> bool:
        pair = (deduction_pair(record) if args.shape == "deduction"
                else generic_pair(record, args.prompt_key, args.response_key))
        if pair is None:
            return False
        prompt, response = pair
        return writer.write(Row(
            prompt=prompt,
            response=response,
            ctx={"lang": "text", "intent": args.intent, "source": args.label},
            license=licence,
            source=f"{args.label}:{args.dataset}#{index}",
            source_hash=hash_source(f"{prompt}\n{response}"),
            script_id=args.script_id,
        ))

    try:
        # One parquet download beats paging: the API rate-limits and silently
        # truncated this dataset to a third of its rows twice.
        scanned = 0
        for index, record in enumerate(parquet_rows(args.dataset, args.split)):
            if index >= args.limit:
                break
            scanned += 1
            if emit(record, index):
                written += 1
            else:
                skipped += 1
        if scanned:
            print(f"read {scanned:,} rows from parquet", flush=True)
            print(f"wrote {args.out}\n  rows written {written}\n"
                  f"  skipped {skipped}")
            return 0
        print("parquet unavailable; falling back to paged rows", flush=True)

        for offset in range(0, args.limit, PAGE):
            url = (f"{SERVER}/rows?dataset={urllib.parse.quote(args.dataset)}"
                   f"&config={args.config}&split={args.split}"
                   f"&offset={offset}&length={PAGE}")
            # The datasets-server rate-limits: a flat 0.1 s delay ran into
            # HTTP 429 at offset 5,200 and silently truncated a 16k dataset to
            # 5k. Back off and retry rather than treating a throttle as the
            # end of the data.
            page = None
            for attempt in range(5):
                try:
                    page = fetch(url)
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 429:
                        print(f"  stopped at offset {offset}: {error}",
                              flush=True)
                        break
                    time.sleep(2 ** attempt)
                except Exception as error:
                    print(f"  stopped at offset {offset}: {error}", flush=True)
                    break
            if page is None:
                print(f"  giving up at offset {offset} after retries",
                      flush=True)
                break
            rows = page.get("rows") or []
            if not rows:
                break
            for item in rows:
                record = item.get("row") or {}
                pair = (deduction_pair(record) if args.shape == "deduction"
                        else generic_pair(record, args.prompt_key,
                                          args.response_key))
                if pair is None:
                    skipped += 1
                    continue
                prompt, response = pair
                row = Row(
                    prompt=prompt,
                    response=response,
                    ctx={"lang": "text", "intent": args.intent,
                         "source": args.label},
                    license=licence,
                    source=f"{args.label}:{args.dataset}#{offset + len(rows)}",
                    source_hash=hash_source(f"{prompt}\n{response}"),
                    script_id=args.script_id,
                )
                if writer.write(row):
                    written += 1
                else:
                    skipped += 1
            if offset and offset % 2000 == 0:
                print(f"  {offset} rows scanned, written={written}", flush=True)
            time.sleep(0.35)
    finally:
        writer.close()

    print(f"wrote {args.out}\n  rows written {written}\n  skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
