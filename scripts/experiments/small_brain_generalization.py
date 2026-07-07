#!/usr/bin/env python3
"""Reproducible small-batch brain generalization experiment.

Run only against a disposable/clean brain instance; this script mutates it by
training. It never starts, resets, checkpoints, or points at production state.
"""
from __future__ import annotations

import argparse, base64, json, re, statistics, time, urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "data" / "small_generalization_v1.json"

def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_*]+", s.lower()))

def token_f1(reply: str, expected: list[str]) -> float:
    r = set(norm(reply).split()); e = set(norm(" ".join(expected)).split())
    if not r or not e: return 0.0
    p, q = len(r & e) / len(r), len(r & e) / len(e)
    return 2*p*q/(p+q) if p+q else 0.0

def concept_recall(reply: str, concepts: list[str]) -> float:
    n = norm(reply)
    return sum(norm(x) in n for x in concepts) / len(concepts) if concepts else 0.0

def nearest_baseline(prompt: str, train: list[dict]) -> str:
    q = set(norm(prompt).split())
    def score(row):
        t=set(norm(row["prompt"]).split()); return len(q&t)/len(q|t) if q|t else 0
    return max(train, key=score)["answer"]

class Client:
    def __init__(self, endpoint: str, prefix: str, timeout: float, input_pool: int = 1,
                 output_pool: int = 4, query_pool: int | None = None, settle_ticks: int = 0,
                 inference_mode: str = "integrate"):
        self.root=endpoint.rstrip("/")+prefix.rstrip("/"); self.timeout=timeout
        self.input_pool=input_pool; self.output_pool=output_pool
        self.query_pool=query_pool or input_pool; self.settle_ticks=max(0,settle_ticks)
        self.inference_mode=inference_mode
    def request(self, method: str, path: str, body=None):
        raw=None if body is None else json.dumps(body).encode()
        req=urllib.request.Request(self.root+path, data=raw, method=method,
            headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read() or b"{}")
    def train(self, prompt, answer):
        enc=lambda s: base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
        self.request("POST","/consolidate",{
            "input_pool":self.input_pool,"input_frame":enc(prompt),
            "outcome_pool":self.output_pool,"outcome_frame":enc(answer)})
    def chat(self,prompt):
        if self.inference_mode == "chat":
            return self.request("POST","/chat",{"text":prompt})
        autonomous = None
        if self.inference_mode == "hybrid":
            autonomous = self.request("POST","/chat",{"text":prompt})
            if ((autonomous.get("grounding") or {}).get("outside_grounding")
                    or not str(autonomous.get("reply") or autonomous.get("answer") or "").strip()):
                return autonomous
        enc=lambda s: base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
        response=self.request("POST","/predict",{
            "query_pool":self.query_pool,"target_pool":self.output_pool,"frame":enc(prompt)})
        answer=response.get("answer")
        if answer:
            response["reply"]=base64.urlsafe_b64decode(answer+"="*(-len(answer)%4)).decode(errors="replace")
        if autonomous is not None:
            auto_reply = str(autonomous.get("reply") or autonomous.get("answer") or "")
            response["reply"] = " ".join(x for x in (auto_reply, response.get("reply", "")) if x)
            response["grounding"] = autonomous.get("grounding") or {}
        return response

def evaluate(rows, ask, train):
    out=[]
    for row in rows:
        start=time.perf_counter(); response=ask(row["prompt"]); latency=time.perf_counter()-start
        reply=response if isinstance(response,str) else str(response.get("reply") or response.get("answer") or "")
        grounding={} if isinstance(response,str) else (response.get("grounding") or {})
        oov=bool(grounding.get("outside_grounding")) or not reply.strip()
        out.append({"kind":row["kind"],"prompt":row["prompt"],"reply":reply,
          "concept_recall": concept_recall(reply,row.get("concepts",[])),
          "token_f1": token_f1(reply,row.get("concepts",[])),
          "oov_correct": bool(row.get("expect_oov")) == oov if row["kind"]=="oov" else None,
          "latency_ms":round(latency*1000,3)})
    return out

def summarize(rows):
    by=defaultdict(list)
    for r in rows: by[r["kind"]].append(r)
    return {k:{"count":len(v),"concept_recall":statistics.mean(x["concept_recall"] for x in v),
      "token_f1":statistics.mean(x["token_f1"] for x in v),
      "oov_accuracy":statistics.mean(x["oov_correct"] for x in v) if k=="oov" else None,
      "latency_ms_mean":statistics.mean(x["latency_ms"] for x in v)} for k,v in by.items()}

def numeric_delta(before: dict, after: dict) -> dict:
    """Resource-growth metrics exposed by the brain's /stats contract."""
    return {k: after[k]-before[k] for k in sorted(before.keys() & after.keys())
            if isinstance(before[k], (int,float)) and isinstance(after[k], (int,float))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--endpoint",default="http://127.0.0.1:8090")
    p.add_argument("--prefix",default="/brain")
    p.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS)
    p.add_argument("--output-dir",type=Path,default=HERE/"results")
    p.add_argument("--repeats",type=int,default=8); p.add_argument("--timeout",type=float,default=60)
    p.add_argument("--input-pool",type=int,default=1)
    p.add_argument("--output-pool",type=int,default=4)
    p.add_argument("--query-pool",type=int)
    p.add_argument("--settle-ticks",type=int,default=0)
    p.add_argument("--inference-mode",choices=("integrate","chat"),default="integrate")
    p.add_argument("--confirm-disposable",action="store_true",help="Required: confirms endpoint uses isolated brain data")
    a=p.parse_args()
    if not a.confirm_disposable: p.error("--confirm-disposable is required; training mutates the target brain")
    corpus=json.loads(a.corpus.read_text(encoding="utf-8")); client=Client(
        a.endpoint,a.prefix,a.timeout,a.input_pool,a.output_pool,a.query_pool,a.settle_ticks,
        a.inference_mode)
    before=client.request("GET","/stats")
    started=time.perf_counter(); train_lat=[]
    for _ in range(a.repeats):
        for row in corpus["train"]:
            t=time.perf_counter(); client.train(row["prompt"],row["answer"]); train_lat.append(time.perf_counter()-t)
    brain=evaluate(corpus["test"],client.chat,corpus["train"])
    baseline=evaluate(corpus["test"],lambda q: nearest_baseline(q,corpus["train"]),corpus["train"])
    after=client.request("GET","/stats")
    report={"schema":"small-brain-generalization/v1","endpoint":a.endpoint,"prefix":a.prefix,
      "corpus":str(a.corpus.resolve()),"repeats":a.repeats,"elapsed_s":time.perf_counter()-started,
      "training_latency_ms_mean":statistics.mean(train_lat)*1000,"stats_before":before,"stats_after":after,
      "stats_delta":numeric_delta(before,after),
      "brain_summary":summarize(brain),"nearest_token_baseline_summary":summarize(baseline),
      "brain_results":brain,"nearest_token_baseline_results":baseline}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    dest=a.output_dir/f"small_generalization_{time.strftime('%Y%m%d_%H%M%S')}.json"
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["brain_summary"],indent=2)); print(f"report={dest}")

if __name__=="__main__": main()
