"""Phase D — self-tuning validation.

Train the brain, snapshot its recall, run multiple /retune steps,
watch the brain hill-climb its own decay_rate using only its self_test
recall as the gradient signal.
"""

import base64, http.client, json, sys

HOST, PORT = "127.0.0.1", 8195
POOL_TEXT, POOL_ACTION = 1, 4


def conn(): return http.client.HTTPConnection(HOST, PORT, timeout=60)


def post(p, j):
    c = conn(); c.request("POST", p, json.dumps(j).encode(), {"Content-Type": "application/json"})
    r = c.getresponse(); out = r.read(); c.close()
    try: return r.status, json.loads(out)
    except: return r.status, out.decode(errors="replace")


def get(p):
    c = conn(); c.request("GET", p)
    r = c.getresponse(); out = r.read(); c.close()
    try: return r.status, json.loads(out)
    except: return r.status, out.decode(errors="replace")


def observe(pool, frame):
    b64 = base64.urlsafe_b64encode(frame.encode()).decode().rstrip("=")
    return post("/observe", {"pool_id": pool, "frame": b64})


def tick(): post("/tick", {})


def train_pair(p, r, n=12):
    for _ in range(n):
        observe(POOL_TEXT, p); observe(POOL_ACTION, r); tick()


CORPUS = [
    ("hi", "hello"), ("hey", "hello"), ("yo", "hello"),
    ("cat", "animal"), ("dog", "animal"),
    ("car", "vehicle"),
    ("red", "color"), ("blue", "color"),
]


def main():
    s, _ = get("/health")
    if s != 200: print("brain not up"); return 1

    print("=== Phase D: self-tuning ===\n")
    print("Training (sparse): each pair only 4 reps so consolidation lock")
    print("                   doesn't fully saturate — leaves real decay sensitivity\n")
    for p, r in CORPUS:
        train_pair(p, r, n=4)
    tick()

    _, qa = get("/qa_db_stats")
    _, lk = get("/consolidation_stats")
    print(f"  qa={qa['count']} locked={lk['locked_terminals']}\n")

    # Initial recall snapshot.
    _, t0 = get("/tuning_state")
    print(f"Initial tuning state: decay={t0['last_decay_rate']:.6f} recall={t0['last_recall']:.3f}\n")

    # Run 12 retune steps and watch the hill climb.
    print(f"{'step':>5} {'recall_before':>14} {'recall_after':>14} {'decay_before':>14} {'decay_after':>14} {'dir':>5} {'best_recall':>12} {'best_decay':>14}")
    print("-" * 110)
    for step in range(12):
        s, rpt = post("/retune", {"sample_count": 16})
        print(f"{step:>5d} "
              f"{rpt['recall_before']:>14.4f} {rpt['recall_after']:>14.4f} "
              f"{rpt['decay_before']:>14.6f} {rpt['decay_after']:>14.6f} "
              f"{rpt['direction_after']:>5.1f} "
              f"{rpt['best_recall']:>12.4f} {rpt['best_decay_rate']:>14.6f}")

    print()
    _, tF = get("/tuning_state")
    print(f"Final tuning state after baseline retune loop:")
    print(f"  steps={tF['steps']}")
    print(f"  last_recall={tF['last_recall']:.4f}  best_recall={tF['best_recall']:.4f}")
    print(f"  last_decay_rate={tF['last_decay_rate']:.6f}  best_decay_rate={tF['best_decay_rate']:.6f}")
    print(f"  condition_best={tF['condition_best']}")

    # Perturbation experiment: spike decay aggressively, let idle ticks
    # damage unlocked terminals, then watch the controller recover.
    print("\n=== Perturbation: spike decay to 0.05, idle 50 ticks, run retune loop ===\n")
    post("/force_decay", {"decay_rate": 0.05})
    post("/idle_ticks", {"n": 50})
    _, lk2 = get("/consolidation_stats")
    print(f"After perturbation: locked_terminals={lk2['locked_terminals']}\n")

    print(f"{'step':>5} {'recall_before':>14} {'recall_after':>14} {'decay_before':>14} {'decay_after':>14} {'dir':>5}")
    print("-" * 80)
    for step in range(15):
        s, rpt = post("/retune", {"sample_count": 16})
        print(f"{step:>5d} "
              f"{rpt['recall_before']:>14.4f} {rpt['recall_after']:>14.4f} "
              f"{rpt['decay_before']:>14.6f} {rpt['decay_after']:>14.6f} "
              f"{rpt['direction_after']:>5.1f}")

    _, tG = get("/tuning_state")
    print(f"\nFinal tuning state after recovery loop:")
    print(f"  steps={tG['steps']}  best_recall={tG['best_recall']:.4f}  best_decay={tG['best_decay_rate']:.6f}")
    print(f"  current decay={tG['last_decay_rate']:.6f}  current recall={tG['last_recall']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
