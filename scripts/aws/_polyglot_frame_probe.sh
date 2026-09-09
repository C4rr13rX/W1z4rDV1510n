python3 - <<'PY'
"""Compare the per-component recall of the failing canonical against its
passing paraphrase, for the one polyglot row that blocks every admission.

`javascript_go_order_workers/canonical` returns order_service.js + ledger.go
where the case needs order_service.js + dedup.go; the paraphrase of the SAME
case returns all three and passes. `ledger.go` is the Go atomic ledger from
the cross-language transfer suite, so the Go slot is resolving to a
transaction behaviour rather than the deduplication one.

`intent_diagnostics.component_recall` reports each subset's name and the files
it retrieved, so the subset LABELS are readable directly instead of inferred
from the composed manifest. Asking the Go component alone separates two very
different causes: if "go concurrency deduplication" alone returns dedup.go,
the brain holds it and the composite mislabels the slot; if it returns
ledger.go too, the recall itself cannot discriminate.
"""
import json, time, urllib.request

ENDPOINT = "http://127.0.0.1:18095"
out = {"now": time.time()}


def chat(text):
    body = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT + "/brain/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as handle:
        return json.loads(handle.read().decode("utf-8"))


def summarise(text):
    try:
        response = chat(text)
    except Exception as error:
        return {"error": str(error)[:200]}
    reply = str(response.get("reply") or "")
    try:
        files = sorted(json.loads(reply).get("files", {}))
    except Exception:
        files = None
    diag = response.get("intent_diagnostics") or {}
    return {
        "files": files,
        "reply_len": len(reply),
        "answer_branch": diag.get("answer_branch"),
        "labels": diag.get("instruction_labels") or diag.get("labels"),
        "component_recall": diag.get("component_recall"),
        "manifest_composition_ready": diag.get("manifest_composition_ready"),
    }


PROBES = {
    "canonical_composite":
        "Build a polyglot project with a JavaScript transactional-outbox order "
        "service and a Go concurrency-safe work deduplicator.",
    "paraphrase_composite":
        "Create idempotent Node.js outbox-event ordering code plus Golang "
        "synchronization that suppresses duplicate work in one repository.",
    # The Go component asked alone, in the two vocabularies the case uses.
    "go_component_canonical":
        "Implement a Go concurrency-safe deduplicator that accepts each work "
        "key only once.",
    "go_component_paraphrase":
        "Write Golang code using synchronization to suppress duplicate work "
        "across concurrent callers.",
    # What the post-fix frame actually asks, reconstructed: "<language>
    # <canonical behaviour>". If this returns ledger.go the frame is the fault.
    "frame_go_concurrency": "go concurrency deduplication",
    "frame_go_transaction": "go persistence atomic transaction",
}

for name, text in PROBES.items():
    out[name] = summarise(text)
    out[name]["prompt"] = text[:90]

print("PROBEJSON " + json.dumps(out))
PY
