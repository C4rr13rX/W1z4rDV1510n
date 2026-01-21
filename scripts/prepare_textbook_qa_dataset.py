#!/usr/bin/env python3
"""
Prepare textbook PDFs into page images + text and generate Q&A candidates.

Outputs (under data/textbooks by default):
  - pages/<book_id>/page_0001.png
  - text/<book_id>/page_0001.txt
  - page_manifest.jsonl
  - association_queue.jsonl
  - qa_candidates.jsonl
  - qa_dataset.jsonl (verified-only)

This script is conservative: it emits review queues so humans can validate
text/image alignment before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Tools:
    renderer_cmd: str
    renderer_kind: str
    ocr_cmd: str
    ocr_kind: str


@dataclass
class PageArtifact:
    page_index: int
    image_path: Path
    text: str
    text_source: str
    text_confidence: float


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_payload(payload: bytes) -> str:
    return hashlib.blake2s(payload).hexdigest()


def normalize_book_id(name: str) -> str:
    cleaned = []
    for ch in name.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_", "."}:
            cleaned.append("_")
    collapsed = re.sub(r"_+", "_", "".join(cleaned)).strip("_")
    return collapsed or "book"


def detect_input_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    candidate = ROOT.parent / "StateOfLoci" / "textbooks"
    if candidate.exists():
        return candidate
    return ROOT / "data" / "textbooks" / "raw"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_tool(preferred: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if preferred and preferred != "auto":
        if shutil.which(preferred):
            return preferred
        raise SystemExit(f"Requested tool not found on PATH: {preferred}")
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None


def choose_tools(renderer: str, ocr: str) -> Tools:
    renderer_kind = None
    renderer_cmd = None
    if renderer != "auto":
        renderer_cmd = find_tool(renderer, [renderer])
        renderer_kind = renderer
    else:
        for candidate in ["pdftoppm", "mutool"]:
            cmd = find_tool(None, [candidate])
            if cmd:
                renderer_cmd = cmd
                renderer_kind = candidate
                break
    if not renderer_cmd or not renderer_kind:
        raise SystemExit("Missing PDF renderer. Install pdftoppm or mutool.")
    ocr_kind = None
    ocr_cmd = None
    if ocr != "auto":
        ocr_cmd = find_tool(ocr, [ocr])
        ocr_kind = ocr
    else:
        for candidate in ["pdftotext", "tesseract"]:
            cmd = find_tool(None, [candidate])
            if cmd:
                ocr_cmd = cmd
                ocr_kind = candidate
                break
    if not ocr_cmd or not ocr_kind:
        raise SystemExit("Missing OCR/text tool. Install pdftotext or tesseract.")
    return Tools(
        renderer_cmd=renderer_cmd,
        renderer_kind=renderer_kind,
        ocr_cmd=ocr_cmd,
        ocr_kind=ocr_kind,
    )


def run_command(cmd: Sequence[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return result.stdout


def render_pages(
    pdf_path: Path,
    output_dir: Path,
    renderer_cmd: str,
    renderer_kind: str,
    dpi: int,
    skip_existing: bool,
) -> List[Path]:
    existing = sorted(output_dir.glob("page_*.png"))
    if existing and skip_existing:
        return existing
    ensure_dir(output_dir)
    if renderer_kind == "pdftoppm":
        prefix = output_dir / "page"
        run_command([renderer_cmd, "-png", "-r", str(dpi), str(pdf_path), str(prefix)])
        pages = []
        for path in output_dir.glob("page-*.png"):
            try:
                suffix = path.stem.split("-")[-1]
                index = int(suffix)
            except ValueError:
                continue
            target = output_dir / f"page_{index:04d}.png"
            if target.exists():
                target.unlink()
            path.rename(target)
            pages.append((index, target))
        return [item[1] for item in sorted(pages, key=lambda item: item[0])]
    if renderer_kind == "mutool":
        pattern = str(output_dir / "page_%04d.png")
        run_command([renderer_cmd, "draw", "-r", str(dpi), "-o", pattern, str(pdf_path)])
        return sorted(output_dir.glob("page_*.png"))
    raise RuntimeError(f"Unsupported renderer: {renderer_kind}")


def extract_text_pdftotext(pdf_path: Path, ocr_cmd: str) -> List[str]:
    output = run_command([ocr_cmd, "-layout", "-enc", "UTF-8", str(pdf_path), "-"])
    pages = output.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def extract_text_tesseract(image_path: Path, ocr_cmd: str, dpi: int) -> str:
    return run_command([ocr_cmd, str(image_path), "stdout", "--dpi", str(dpi)])


def estimate_text_confidence(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    printable = sum(1 for ch in stripped if ch.isprintable())
    letters = sum(1 for ch in stripped if ch.isalpha())
    length_score = min(1.0, len(stripped) / 1200.0)
    ratio = letters / max(1, printable)
    return max(0.0, min(1.0, 0.15 + 0.85 * ratio * length_score))


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sent.strip() for sent in sentences if sent.strip()]


def generate_qa_pairs(text: str, max_pairs: int, include_summary: bool) -> List[dict]:
    normalized = normalize_text(text)
    sentences = split_sentences(normalized)
    pairs: List[dict] = []
    patterns = [
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+is\s+(?P<definition>[^.]{5,220})", re.I), "is"),
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+are\s+(?P<definition>[^.]{5,220})", re.I), "are"),
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+refers to\s+(?P<definition>[^.]{5,220})", re.I), "is"),
        (re.compile(r"^(?P<subject>[^.]{3,80}?)\s+is defined as\s+(?P<definition>[^.]{5,220})", re.I), "is"),
    ]
    for sentence in sentences:
        if len(pairs) >= max_pairs:
            break
        for pattern, verb in patterns:
            match = pattern.match(sentence)
            if not match:
                continue
            subject = match.group("subject").strip(" :;-")
            definition = match.group("definition").strip(" .;")
            if not subject or not definition:
                continue
            subject_lower = subject.lower()
            if subject_lower in {"this", "that", "these", "those", "it", "they"}:
                continue
            if len(subject.split()) > 10:
                continue
            question = f"What is {subject}?" if verb != "are" else f"What are {subject}?"
            pairs.append(
                {
                    "question": question,
                    "answer": definition,
                    "evidence": sentence[:240],
                    "confidence": 0.7,
                }
            )
            break
    if not pairs and include_summary and normalized:
        summary = " ".join(sentences[:2])[:240]
        if summary:
            pairs.append(
                {
                    "question": "What is this page about?",
                    "answer": summary,
                    "evidence": summary,
                    "confidence": 0.25,
                }
            )
    return pairs


def excerpt(text: str, max_len: int = 240) -> str:
    text = normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def load_review_statuses(review_file: Optional[Path]) -> dict:
    if not review_file:
        return {}
    statuses: dict = {}
    with review_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            page_id = record.get("page_id") or record.get("task_id")
            status = record.get("status")
            if page_id and status:
                statuses[page_id] = status
    return statuses


def collect_pdfs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.pdf"))


def build_page_artifacts(
    pdf_path: Path,
    book_id: str,
    output_root: Path,
    tools: Tools,
    dpi: int,
    max_pages: Optional[int],
    skip_existing: bool,
) -> List[PageArtifact]:
    pages_dir = output_root / "pages" / book_id
    text_dir = output_root / "text" / book_id
    ensure_dir(pages_dir)
    ensure_dir(text_dir)
    page_images = render_pages(
        pdf_path,
        pages_dir,
        tools.renderer_cmd,
        tools.renderer_kind,
        dpi,
        skip_existing,
    )
    if max_pages:
        page_images = page_images[:max_pages]
    artifacts: List[PageArtifact] = []
    if tools.ocr_kind == "pdftotext":
        texts = extract_text_pdftotext(pdf_path, tools.ocr_cmd)
        if max_pages:
            texts = texts[:max_pages]
        while len(texts) < len(page_images):
            texts.append("")
        texts = texts[: len(page_images)]
        for idx, image_path in enumerate(page_images, start=1):
            text = texts[idx - 1]
            artifacts.append(
                PageArtifact(
                    page_index=idx,
                    image_path=image_path,
                    text=text,
                    text_source="pdftotext",
                    text_confidence=estimate_text_confidence(text),
                )
            )
    else:
        for idx, image_path in enumerate(page_images, start=1):
            text = extract_text_tesseract(image_path, tools.ocr_cmd, dpi)
            artifacts.append(
                PageArtifact(
                    page_index=idx,
                    image_path=image_path,
                    text=text,
                    text_source="tesseract",
                    text_confidence=estimate_text_confidence(text),
                )
            )
    for artifact in artifacts:
        text_path = text_dir / f"page_{artifact.page_index:04d}.txt"
        if not text_path.exists() or not skip_existing:
            text_path.write_text(artifact.text, encoding="utf-8")
    return artifacts


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", help="Folder containing textbook PDFs")
    parser.add_argument("--output-root", default="data/textbooks", help="Output root")
    parser.add_argument("--renderer", default="auto", help="auto, pdftoppm, mutool")
    parser.add_argument("--ocr", default="auto", help="auto, pdftotext, tesseract")
    parser.add_argument("--dpi", type=int, default=220, help="Render DPI (default 220)")
    parser.add_argument("--max-books", type=int, help="Process only N books")
    parser.add_argument("--max-pages", type=int, help="Process only N pages per book")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing outputs")
    parser.add_argument("--max-qa-per-page", type=int, default=3)
    parser.add_argument("--include-summary", action="store_true", help="Add low-confidence summary QA")
    parser.add_argument("--review-file", help="JSONL file with reviewed statuses")
    args = parser.parse_args()

    input_root = detect_input_root(args.input_root)
    output_root = Path(args.output_root).expanduser()
    ensure_dir(output_root)

    pdfs = collect_pdfs(input_root)
    if not pdfs:
        raise SystemExit(f"No PDFs found under {input_root}")
    if args.max_books:
        pdfs = pdfs[: args.max_books]

    tools = choose_tools(args.renderer, args.ocr)
    review_statuses = load_review_statuses(Path(args.review_file)) if args.review_file else {}

    manifest_path = output_root / "page_manifest.jsonl"
    queue_path = output_root / "association_queue.jsonl"
    qa_candidates_path = output_root / "qa_candidates.jsonl"
    qa_verified_path = output_root / "qa_dataset.jsonl"

    manifest_records: List[dict] = []
    queue_records: List[dict] = []
    qa_candidates: List[dict] = []
    qa_verified: List[dict] = []

    for pdf_path in pdfs:
        book_id = normalize_book_id(pdf_path.stem)
        artifacts = build_page_artifacts(
            pdf_path,
            book_id,
            output_root,
            tools,
            args.dpi,
            args.max_pages,
            args.skip_existing,
        )
        for artifact in artifacts:
            page_id = f"{book_id}-p{artifact.page_index:04d}"
            text_path = output_root / "text" / book_id / f"page_{artifact.page_index:04d}.txt"
            image_hash = hash_payload(artifact.image_path.read_bytes())
            text_hash = hash_payload(artifact.text.encode("utf-8"))
            task_id = hash_payload(f"page-task|{page_id}|{image_hash}|{text_hash}".encode("utf-8"))
            work_id = hash_payload(f"work|{task_id}".encode("utf-8"))
            status = review_statuses.get(page_id) or review_statuses.get(task_id) or "PENDING"
            status = status.strip().upper()
            manifest_records.append(
                {
                    "page_id": page_id,
                    "book_id": book_id,
                    "source_pdf": str(pdf_path),
                    "page_index": artifact.page_index,
                    "image_path": str(artifact.image_path),
                    "text_path": str(text_path),
                    "image_hash": image_hash,
                    "text_hash": text_hash,
                    "text_source": artifact.text_source,
                    "text_confidence": round(artifact.text_confidence, 3),
                    "text_chars": len(artifact.text),
                    "text_words": word_count(artifact.text),
                    "review_status": status,
                    "review_task_id": task_id,
                    "created_at": now_iso(),
                }
            )
            queue_records.append(
                {
                    "task_id": task_id,
                    "work_id": work_id,
                    "work_kind": "HUMAN_ANNOTATION",
                    "page_id": page_id,
                    "book_id": book_id,
                    "page_index": artifact.page_index,
                    "image_path": str(artifact.image_path),
                    "text_path": str(text_path),
                    "image_hash": image_hash,
                    "text_hash": text_hash,
                    "text_excerpt": excerpt(artifact.text),
                    "status": status,
                    "reward_score": 0.9,
                    "created_at": now_iso(),
                }
            )
            qa_pairs = generate_qa_pairs(
                artifact.text,
                max_pairs=args.max_qa_per_page,
                include_summary=args.include_summary,
            )
            for pair in qa_pairs:
                qa_id = hash_payload(
                    f"qa|{page_id}|{pair['question']}|{pair['answer']}".encode("utf-8")
                )
                qa_record = {
                    "qa_id": qa_id,
                    "page_id": page_id,
                    "book_id": book_id,
                    "page_index": artifact.page_index,
                    "image_path": str(artifact.image_path),
                    "text_path": str(text_path),
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "evidence": pair["evidence"],
                    "confidence": pair["confidence"],
                    "review_status": status,
                    "image_hash": image_hash,
                    "text_hash": text_hash,
                }
                qa_candidates.append(qa_record)
                if status == "VERIFIED":
                    qa_verified.append(qa_record)

    write_jsonl(manifest_records, manifest_path)
    write_jsonl(queue_records, queue_path)
    write_jsonl(qa_candidates, qa_candidates_path)
    write_jsonl(qa_verified, qa_verified_path)

    print(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "books": len(pdfs),
                "pages": len(manifest_records),
                "qa_candidates": len(qa_candidates),
                "qa_verified": len(qa_verified),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
