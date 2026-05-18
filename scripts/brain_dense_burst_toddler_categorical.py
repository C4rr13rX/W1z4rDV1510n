#!/usr/bin/env python3
"""scripts/brain_dense_burst_toddler_categorical.py

Re-scores the dense-burst experiment with a category-substring match
instead of exact-category match.  Reports whether the substrate is
firing the right CATEGORY (modulo byte-decoder reorderings) — which
is the empirical signal the substrate's hierarchical-emergence
architecture is supposed to produce.

A reply is a categorical HIT if it contains a 3+ byte contiguous
substring of the trained category.  ANIMAL = 'animal' has substrings
'ani', 'nim', 'ima', 'mal'.  A reply 'nimal' contains 'nim','ima','mal'
→ HIT.  A reply 'difficult' contains none → miss.
"""
import base64
import json
import sys
import urllib.request

BRAIN = "http://127.0.0.1:8095"
POOL_TEXT = 1
POOL_ACTION = 4

PAIRS = [
    ("dog", "animal"), ("cat", "animal"), ("cow", "animal"),
    ("horse", "animal"), ("bird", "animal"), ("fish", "animal"),
    ("apple", "food"), ("banana", "food"), ("bread", "food"),
    ("cake", "food"), ("milk", "food"),
    ("car", "vehicle"), ("truck", "vehicle"), ("bike", "vehicle"),
    ("plane", "vehicle"), ("boat", "vehicle"),
    ("red", "color"), ("blue", "color"), ("green", "color"), ("yellow", "color"),
    ("ball", "toy"), ("doll", "toy"), ("kite", "toy"), ("drum", "toy"),
    ("tree", "nature"), ("flower", "nature"), ("river", "nature"), ("mountain", "nature"),
    ("hand", "body"), ("foot", "body"), ("eye", "body"), ("mouth", "body"),
]


def b64u(b):
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def post(path, body, timeout=15):
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{BRAIN}{path}", data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}


def category_substrings(category: str, min_len: int = 3) -> list[str]:
    """All contiguous substrings of `category` with length >= min_len."""
    out = []
    for L in range(min_len, len(category) + 1):
        for start in range(0, len(category) - L + 1):
            out.append(category[start:start + L])
    return out


def main():
    print(f"  {'prompt':<10} {'expect':<10} {'reply':<40} {'exact':<5} {'cat':<5} bestsub")
    print(f"  {'-'*10} {'-'*10} {'-'*40} {'-'*5} {'-'*5} -------")
    exact_hits = 0
    cat_hits = 0
    by_cat: dict[str, list[int]] = {}  # cat -> [cat_hit, total]
    by_cat_exact: dict[str, list[int]] = {}
    for prompt, expected in PAIRS:
        post("/observe", {"pool_id": POOL_TEXT,
                          "frame": b64u(prompt.encode("utf-8"))})
        r = post("/integrate", {"query_pool": POOL_TEXT,
                                 "target_pool": POOL_ACTION})
        b64ans = r.get("answer") or ""
        try:
            ans_bytes = base64.urlsafe_b64decode(b64ans + "==")
            ans = ans_bytes.decode("utf-8", "replace")
        except Exception:
            ans = ""
        exact = expected in ans.lower()
        cat = False
        best_sub = ""
        for sub in sorted(category_substrings(expected, 3),
                            key=lambda s: -len(s)):
            if sub in ans.lower():
                cat = True
                best_sub = sub
                break
        exact_hits += int(exact)
        cat_hits += int(cat)
        by_cat.setdefault(expected, [0, 0])
        by_cat[expected][1] += 1
        if cat:
            by_cat[expected][0] += 1
        by_cat_exact.setdefault(expected, [0, 0])
        by_cat_exact[expected][1] += 1
        if exact:
            by_cat_exact[expected][0] += 1
        print(f"  {prompt:<10} {expected:<10} {ans[:40]!r:<40} "
                f"{'Y' if exact else '.':<5} "
                f"{'Y' if cat else '.':<5} "
                f"{best_sub!r}")

    n = len(PAIRS)
    print()
    print(f"=== exact-substring (trained-baseline metric): "
            f"{exact_hits}/{n}  ({100.0*exact_hits/n:.1f}%) ===")
    print(f"=== category-substring (architectural-correct metric, min 3 bytes): "
            f"{cat_hits}/{n}  ({100.0*cat_hits/n:.1f}%) ===")
    print()
    print(f"per category (categorical / exact):")
    for cat in sorted(by_cat):
        h, t = by_cat[cat]
        eh, _ = by_cat_exact[cat]
        print(f"  {cat:<10}  cat={h}/{t}   exact={eh}/{t}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
