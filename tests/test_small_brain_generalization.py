import importlib.util
from pathlib import Path

P=Path(__file__).parents[1]/"scripts"/"experiments"/"small_brain_generalization.py"
S=importlib.util.spec_from_file_location("small_generalization",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

def test_concept_recall_normalizes_case_and_punctuation():
    assert M.concept_recall("Force = MASS times acceleration!",["force","mass","acceleration"]) == 1

def test_nearest_baseline_is_deterministic():
    rows=[{"prompt":"define density mass volume","answer":"density"},{"prompt":"newton force acceleration","answer":"force"}]
    assert M.nearest_baseline("how do mass and volume define density",rows)=="density"

def test_oov_summary():
    rows=[{"kind":"oov","concept_recall":0,"token_f1":0,"oov_correct":True,"latency_ms":1},
          {"kind":"oov","concept_recall":0,"token_f1":0,"oov_correct":False,"latency_ms":3}]
    assert M.summarize(rows)["oov"]["oov_accuracy"]==0.5

def test_numeric_delta_ignores_metadata():
    assert M.numeric_delta({"neurons":2,"name":"a"},{"neurons":5,"name":"b"}) == {"neurons":3}
