#!/usr/bin/env bash
# run_all_training.sh — Full curriculum training runner
#
# CURRICULUM ORDER (designed to prevent "untraining"):
#   1. Foundational seeds x3 (anchors core facts before anything else)
#   2. Stage 0: Toddler concepts
#   3. Foundational seeds x2 (reinforce before K-12)
#   4. Stages 1-2: K-12 textbooks
#   5. Foundational seeds x2 (reinforce after textbook load)
#   6. Code/terminal/agent corpus
#   7. Foundational seeds x2
#   8. Stages 10-17: STEM
#   9. Foundational seeds x1
#  10. Stages 18-22: Language/documentation
#  11. Foundational seeds x1
#  12. Stages 23-24: Library docs + Stack Overflow
#  13. Foundational seeds x1
#  14. Stage 25: LibreTexts comprehensive
#  15. Stage 26: Biodiversity
#  16. Stages 27-28: Cognitive/sorting
#  17. Stage 29: Bible
#  18. Stages 30-33: Medical/psychology
#  19. Stage 34: Mathematics
#  20. Stage 35: Pedagogy
#  21. Foundational seeds x3 (final anchor)
#
# Run from project root. Logs to D:/w1z4rdv1510n-data/training/

set -euo pipefail

DATA="D:/w1z4rdv1510n-data"
LOG_DIR="$DATA/training"
NODE="http://localhost:8090"
SCRIPTS="scripts"

mkdir -p "$LOG_DIR"

echo "===== W1z4rD Full Curriculum Training started $(date) =====" | tee -a "$LOG_DIR/run_all.log"

# Helper: run foundational seeds N times
reinforce_foundations() {
    local n="${1:-1}"
    echo "[$(date)] Reinforcing foundations (${n}x)..." | tee -a "$LOG_DIR/run_all.log"
    for i in $(seq 1 "$n"); do
        python "$SCRIPTS/build_foundational_seeds.py" \
            --node "localhost:8090" \
            --repeats 15 \
            2>&1 | tee -a "$LOG_DIR/foundations.log" | tail -1
    done
    echo "[$(date)] Foundation reinforcement done" | tee -a "$LOG_DIR/run_all.log"
}

# Helper: re-anchor conversational identity after each major phase.
# Uses LR=7.5 (near max clamp) to saturate greeting connections so they
# compete with corpus-trained words in kWTA. Without this, the code/docs
# corpus progressively overwrites the "hello" -> greeting associations.
#
# 2026-05-01: dropped --passes 15 -> 5.  Math: passes * lr = 5 * 7.5 = 37.5
# is already 9x the max_weight=4.0 clamp; greetings saturate after pass 1.
# Old (15 passes) was wall-clock waste.  Initial saturation also dropped
# from rounds=20 -> 5 because every later phase already re-anchors 5
# rounds; total greeting reinforcement across the curriculum stays > 50
# rounds, far above what kWTA needs to dominate corpus noise.
reinforce_conversations() {
    local rounds="${1:-5}"
    echo "[$(date)] Reinforcing conversational identity (${rounds} rounds)..." | tee -a "$LOG_DIR/run_all.log"
    PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_conversations.py" \
        --node "$NODE" \
        --rounds "$rounds" \
        --passes 5 \
        --lr 7.5 \
        2>&1 | tee -a "$LOG_DIR/conversations.log" | tail -2
    echo "[$(date)] Conversational reinforcement done" | tee -a "$LOG_DIR/run_all.log"
}

# ─── PHASE 0: CLEAR POOL ─────────────────────────────────────────────────────
echo "[$(date)] Clearing pool for fresh start..." | tee -a "$LOG_DIR/run_all.log"
curl -s -X POST "$NODE/neuro/clear" | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"
sleep 1
echo "[$(date)] Pool cleared — starting fresh curriculum" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 1: SATURATE CONVERSATIONAL IDENTITY ───────────────────────────────
# 35 pairs trained at LR=7.5 x 5 passes x 5 rounds. Saturation math: each
# pass multiplies edge weight by ~LR until max_weight=4.0 clamps; lr=7.5
# clamps in pass 1.  The 5-round contrastive cycle handles noise suppression.
# Cuts ~10 hours of redundant Hebbian work vs the prior 15x20 schedule.
reinforce_conversations 5

# ─── PHASE 2: ANCHOR FOUNDATIONS ─────────────────────────────────────────────
reinforce_foundations 3
reinforce_conversations 5

# ─── PHASE 3: TODDLER STAGE 0 ────────────────────────────────────────────────
echo "[$(date)] Starting Stage 0 (toddler foundations)..." | tee -a "$LOG_DIR/run_all.log"
python3 "$SCRIPTS/train_k12.py" \
    --node "$NODE" \
    --stages 0 \
    --resume \
    2>&1 | tee -a "$LOG_DIR/stage0.log"
echo "[$(date)] Stage 0 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_foundations 2
reinforce_conversations 5

# ─── PHASE 4: K-12 TEXTBOOK CURRICULUM ──────────────────────────────────────
echo "[$(date)] Starting Stages 1-2 (K-12 textbooks)..." | tee -a "$LOG_DIR/run_all.log"
python3 "$SCRIPTS/train_k12.py" \
    --node "$NODE" \
    --stages 1,2 \
    --resume \
    2>&1 | tee -a "$LOG_DIR/k12.log"
echo "[$(date)] Stages 1-2 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_foundations 2
reinforce_conversations 5

# ─── PHASE 5: CODE / TERMINAL / AGENT CORPUS ────────────────────────────────
echo "[$(date)] Starting code+terminal+agent corpus..." | tee -a "$LOG_DIR/run_all.log"
python3 "$SCRIPTS/build_code_corpus.py" \
    --node "localhost:8090" \
    --repeats 20 \
    2>&1 | tee -a "$LOG_DIR/code_corpus.log"
echo "[$(date)] Code corpus done" | tee -a "$LOG_DIR/run_all.log"

reinforce_foundations 2
reinforce_conversations 5

# ─── PHASE 6: STEM (stages 10-17) ────────────────────────────────────────────
echo "[$(date)] Starting Stages 10-17 (STEM training)..." | tee -a "$LOG_DIR/run_all.log"
python3 "$SCRIPTS/build_stem_dataset.py" \
    --stages 10,11,12,13,14,15,16,17 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --wiki-max 5000 \
    --libretexts-max 30 \
    --arxiv-max 200 \
    2>&1 | tee -a "$LOG_DIR/stem.log"
echo "[$(date)] Stages 10-17 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_foundations 1
reinforce_conversations 5

# ─── PHASE 7: LANGUAGE / DOCUMENTATION CORPUS (stages 18-22) ────────────────
echo "[$(date)] Starting Stages 18-22 (language + documentation)..." | tee -a "$LOG_DIR/run_all.log"
python3 "$SCRIPTS/build_language_corpus.py" \
    --stages 18,19,20,21,22 \
    --node localhost:8090 \
    --gutenberg-chars 60000 \
    --gutenberg-passages 20 \
    --wikibooks-chars 15000 \
    2>&1 | tee -a "$LOG_DIR/language.log"
echo "[$(date)] Stages 18-22 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_foundations 1
reinforce_conversations 5

# ─── PHASE 8: LIBRARY DOCS + STACK OVERFLOW (stages 23-24) ──────────────────
echo "[$(date)] Starting Stages 23-24 (library docs + Stack Overflow)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_library_corpus.py" \
    --stages 23,24 \
    --node localhost:8090 \
    --lib-pairs-per-file 30 \
    --lib-prose-chars 8000 \
    --so-per-tag 50 \
    --so-min-votes 5 \
    2>&1 | tee -a "$LOG_DIR/library.log"
echo "[$(date)] Stages 23-24 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_foundations 1
reinforce_conversations 5

# ─── PHASE 9: LIBRETEXTS COMPREHENSIVE (stage 25) ───────────────────────────
echo "[$(date)] Starting Stage 25 (LibreTexts comprehensive corpus)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_libretexts_corpus.py" \
    --stages 25 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-pages 5000 \
    2>&1 | tee -a "$LOG_DIR/libretexts.log"
echo "[$(date)] Stage 25 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 10: BIODIVERSITY (stage 26) ────────────────────────────────────────
echo "[$(date)] Starting Stage 26 (biodiversity visual-ID corpus)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_biodiversity_corpus.py" \
    --stages 26 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-per-group 2000 \
    2>&1 | tee -a "$LOG_DIR/biodiversity.log"
echo "[$(date)] Stage 26 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 11: COGNITIVE + SORTING (stages 27-28) ───────────────────────────
echo "[$(date)] Starting Stages 27-28 (cognitive + sorting)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_cognitive_corpus.py" \
    --stages 27,28 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/cognitive.log"
echo "[$(date)] Stages 27-28 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 12: BIBLE (stage 29) ─────────────────────────────────────────────
echo "[$(date)] Starting Stage 29 (World English Bible)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_bible_corpus.py" \
    --stages 29 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/bible.log"
echo "[$(date)] Stage 29 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 13: MEDICAL / PSYCHOLOGY (stages 30-33) ──────────────────────────
echo "[$(date)] Starting Stages 30-33 (medical corpus from NCBI/NLM)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_medical_corpus.py" \
    --stages 30,31,32,33 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-per-query 80 \
    2>&1 | tee -a "$LOG_DIR/medical.log"
echo "[$(date)] Stages 30-33 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 14: MATHEMATICS (stage 34) ───────────────────────────────────────
echo "[$(date)] Starting Stage 34 (mathematics corpus)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_math_corpus.py" \
    --stages 34 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/math.log"
echo "[$(date)] Stage 34 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 15: PEDAGOGY (stage 35) ──────────────────────────────────────────
echo "[$(date)] Starting Stage 35 (pedagogy & curriculum design)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_pedagogy_corpus.py" \
    --stages 35 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-per-query 30 \
    2>&1 | tee -a "$LOG_DIR/pedagogy.log"
echo "[$(date)] Stage 35 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 16: COW MESH (stage 9) ────────────────────────────────────────────
echo "[$(date)] Starting Stage 9 (mesh training)..." | tee -a "$LOG_DIR/run_all.log"
python "$SCRIPTS/build_cow_dataset.py" \
    --stages 9 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/stage9.log" | tail -1
echo "[$(date)] Stage 9 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 17: MULTI-POOL CONCEPT BINDING (GA-tuned path) ────────────────────
# After the slow-pool char-chain decoder is trained, bind the high-priority
# Q->A pairs (greetings + identity facts) through /multi_pool/train_pair so
# the GA-experimental winning genome (use_trigrams=true, lr=0.825, passes=35,
# mp_confidence_threshold=0.345) actually fires at chat time.  Without this
# step the GA gain is dormant — the curriculum trains the slow pool only.
#
# /query/integrated routes to multi-pool first; high-confidence greeting
# bindings here mean instant high-confidence chat replies for these phrases
# without disturbing the broader corpus knowledge in the slow pool.
echo "[$(date)] Binding key Q->A concepts in multi-pool fabric..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_concept_bindings.py" \
    --node "$NODE" \
    2>&1 | tee -a "$LOG_DIR/concept_bindings.log" | tail -5
echo "[$(date)] Multi-pool concept bindings done" | tee -a "$LOG_DIR/run_all.log"

# ─── FINAL: RE-ANCHOR FOUNDATIONS + CONVERSATIONS ────────────────────────────
reinforce_foundations 3
reinforce_conversations 10

echo "===== Full curriculum training complete $(date) =====" | tee -a "$LOG_DIR/run_all.log"
