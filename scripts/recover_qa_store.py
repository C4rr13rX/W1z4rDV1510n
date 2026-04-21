#!/usr/bin/env python3
"""
Recover QA store after answer_index corruption.

Reads all question/answer pairs from qa_store.json (which is intact)
and re-ingests them via /qa/ingest so the Rust runtime rebuilds the
correct Blake2s256 answer_index from scratch.  Checkpoints afterwards.
"""

import argparse, json, sys, uuid, httpx

def main():
    parser = argparse.ArgumentParser(
        description="Re-ingest QA pairs after answer_index corruption. "
                    "Reads the intact pairs dict and POSTs them to /qa/ingest."
    )
    parser.add_argument("--node", default="http://127.0.0.1:8090")
    parser.add_argument("--qa-store", default="D:/w1z4rdv1510n-data/qa_store.json",
                        help="Path to qa_store.json to recover from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pairs without ingesting")
    args = parser.parse_args()

    NODE = args.node
    QA_PATH = args.qa_store
    with open(QA_PATH, encoding="utf-8") as f:
        qa = json.load(f)

    pairs = qa.get("pairs", {})
    if not pairs:
        sys.exit("No pairs found in qa_store.json")

    if args.dry_run:
        print(f"DRY RUN -- would re-ingest {len(pairs)} pairs from {QA_PATH}")
        for _, p in list(pairs.items())[:5]:
            print(f"  Q: {p['question'][:70]}")
        return

    print(f"Found {len(pairs)} pairs to re-ingest")

    # Build candidate list from the intact pairs dict
    candidates = []
    for _, p in pairs.items():
        candidates.append({
            "qa_id":         p.get("qa_id") or str(uuid.uuid4()),
            "question":      p["question"],
            "answer":        p["answer"],
            "confidence":    0.90,
            "book_id":       "",
            "page_index":    0,
            "evidence":      "",
            "review_status": "recovered",
        })

    # Ingest in batches of 50
    batch_size = 50
    total_ingested = 0
    with httpx.Client(timeout=60) as client:
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            r = client.post(f"{NODE}/qa/ingest", json={"candidates": batch})
            r.raise_for_status()
            d = r.json()
            total_ingested += d.get("ingested", 0)
            print(f"  Batch {i//batch_size+1}: ingested={d.get('ingested',0)}  "
                  f"total_pairs={d.get('total_pairs','?')}  "
                  f"q_neurons={d.get('question_neurons','?')}")

        # Checkpoint to save corrected store
        print("\nCheckpointing...")
        r = client.post(f"{NODE}/neuro/checkpoint", timeout=120)
        r.raise_for_status()
        d = r.json()
        print(f"  pool={d.get('pool_path','?')}")
        print(f"  qa  ={d.get('qa_path','?')}")

    print(f"\nDone -- re-ingested {total_ingested}/{len(candidates)} pairs")

if __name__ == "__main__":
    main()
