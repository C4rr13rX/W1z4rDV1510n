#!/usr/bin/env bash
# Runs all remaining training stages in sequence.
# Start this after medical (30-33) completes.
# Usage: bash scripts/run_remaining_pipeline.sh >> D:/w1z4rdv1510n-data/training/pipeline_run.log 2>&1 &

export PYTHONUTF8=1
PYTHON="C:/Python313/python.exe"
SCRIPTS="D:/Projects/W1z4rDV1510n/scripts"
NODE="localhost:8090"
DATA="D:/w1z4rdv1510n-data"

echo "=========================================="
echo "Pipeline start: $(date)"
echo "=========================================="

echo ""
echo "--- Stage 34: Mathematics ---"
"$PYTHON" "$SCRIPTS/build_math_corpus.py" --stages 34 --node "$NODE" --data-dir "$DATA"

echo ""
echo "--- Stage 35: Pedagogy ---"
"$PYTHON" "$SCRIPTS/build_pedagogy_corpus.py" --stages 35 --node "$NODE" --data-dir "$DATA"

echo ""
echo "--- Stages 36-40: Foreign Languages (20 languages) ---"
"$PYTHON" "$SCRIPTS/build_foreign_language_corpus.py" --stages 36,37,38,39,40 --node "$NODE" --data-dir "$DATA"

echo ""
echo "--- Foundation training ---"
"$PYTHON" "$SCRIPTS/train_foundation.py" --node "http://$NODE"

echo ""
echo "--- K-12 full curriculum ---"
"$PYTHON" "$SCRIPTS/train_k12.py" --node "http://$NODE" --stages 0,1,2 --resume

echo ""
echo "--- Scoped JSON responses (interleaved, 200 batches) ---"
"$PYTHON" "$SCRIPTS/train_json_responses.py" --node "$NODE" --batches 200

echo ""
echo "--- Stage 41: English Misspellings & Spelling Correction ---"
"$PYTHON" "$SCRIPTS/build_misspellings_corpus.py" --node "$NODE" --repeats 100

echo ""
echo "--- Stage 42: Long-Form, Run-On & Grammar Training ---"
"$PYTHON" "$SCRIPTS/build_communication_corpus.py" --node "$NODE" --repeats 100

echo ""
echo "=========================================="
echo "Pipeline complete: $(date)"
echo "=========================================="
