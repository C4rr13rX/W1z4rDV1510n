#!/usr/bin/env bash
# run_all_training.sh — Full curriculum training runner
#
# CURRICULUM ORDER (designed to prevent "untraining"):
#   Phase  1  Saturate conversational identity (5 rounds × 5 passes @ LR=7.5)
#   Phase  2  Foundational seeds ×3 (anchors core facts before anything else)
#   Phase  3  Stage  0: toddler concepts
#   Phase  4  Stages 1-2: K-12 textbooks
#   Phase  5  Stage 5: code / terminal / agent corpus (113 Q&A across 29 langs/tools)
#   Phase  6  Stages 10-17: STEM
#   Phase  7  Stages 18-22: language / documentation
#   Phase  8  Stages 23-24: library docs + Stack Overflow
#   Phase  9  Stage 25:    LibreTexts comprehensive
#   Phase 10  Stage 26:    biodiversity
#   Phase 11  Stages 27-28: cognitive + sorting
#   Phase 12  Stage 29:    Bible
#   Phase 13  Stages 30-33: medical / psychology
#   Phase 14  Stage 34:    mathematics
#   Phase 15  Stage 35:    pedagogy
#   Phase 16  Stage  9:    cow mesh
#   Phase 17  English foundation full-architecture pipeline (--pass all)
#   Phase 18  Multi-modal concept streams (image+text+keyboard+temporal)
#   Phase 19  Plant cell physiology (multi-scale)
#   Phase 20  JSON responses (scope-marked, NL-balanced)
#   Phase 21  Natural-language rebalance (counterweight to JSON)
#   Phase 22  Cow anatomy + behaviour + lifecycle
#   Phase 23  Cow 3D anatomical positions
#   Phase 24  Cow multi-view dense surface
#   Phase 25  Cow barrel-surface refinement
#   Phase 26  Foreign language corpus (stages 36-40, 20 languages)
#   Phase 27  Chess outcome + move prediction (5 bounded iterations)
#   Phase 28  Obstacle course (Playwright; best effort)
#   Phase 29  Multi-pool concept binding (GA-tuned /multi_pool/train_pair path)
#   FINAL    Foundational seeds ×3 + conversations ×10 (final anchor)
#
# Foundational seeds + conversational reanchor run between major phases
# throughout to prevent corpus drift from overwriting greetings/identity.
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

# Helper: verifier-driven consolidation.  Runs the (Q, expected_A) battery
# against /chat; for every pair that passes, fires the dopamine-gated LTP
# capture path on the synapses that produced the correct answer.  Failed
# pairs are logged to regression_queue.jsonl and NOT reinforced (would
# burn in a wrong answer).  Exit code 0 = clean, 3 = some regressions —
# we tolerate regressions during the curriculum and only fail-hard on the
# final verifier call (with --check-only) so a single phase drift doesn't
# abort hours of training.
verify_and_reinforce() {
    local label="${1:-mid-curriculum}"
    local extra_args="${2:-}"
    echo "[$(date)] Verifier (${label})..." | tee -a "$LOG_DIR/run_all.log"
    PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/verify_and_reinforce.py" \
        --node "$NODE" $extra_args \
        2>&1 | tee -a "$LOG_DIR/verifier.log" | tail -10 \
        || echo "[$(date)] Verifier (${label}) reported regressions — see regression_queue.jsonl" \
            | tee -a "$LOG_DIR/run_all.log"
    echo "[$(date)] Verifier (${label}) done" | tee -a "$LOG_DIR/run_all.log"
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

# ─── PHASE 17: ENGLISH FOUNDATION CORPUS (full architecture) ─────────────────
# train_foundation.py exercises EVERY core endpoint per fact:
#   /media/train_sequence + /equations/ingest + /knowledge/ingest +
#   /qa/ingest + /neuro/record_episode + /neuro/checkpoint
# --pass all runs concepts -> text -> images sub-passes back to back.
echo "[$(date)] Starting Phase 17 (English foundation, full architecture)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_foundation.py" \
    --node "$NODE" \
    --pass all \
    2>&1 | tee -a "$LOG_DIR/foundation_pipeline.log" | tail -5
echo "[$(date)] Phase 17 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 18: MULTI-MODAL CONCEPT STREAMS ──────────────────────────────────
# Cross-stream concept neurons that fire across image / text / keyboard /
# audio / temporal modalities.  Skips audio when no AWS Polly credentials
# (Stack Overflow / cached probes still useful).  Best-effort.
echo "[$(date)] Starting Phase 18 (multi-modal concept streams)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_concept_streams.py" \
    --node "$NODE" \
    --no-audio \
    2>&1 | tee -a "$LOG_DIR/concept_streams.log" | tail -5 \
    || echo "[$(date)] Phase 18 skipped (missing optional deps)" | tee -a "$LOG_DIR/run_all.log"
echo "[$(date)] Phase 18 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 19: PLANT CELL PHYSIOLOGY (multi-scale) ──────────────────────────
# Dense scientific corpus: plant_cell -> cell_wall -> chloroplast ->
# thylakoid -> chlorophyll -> carbon_atom across 6 length scales.
echo "[$(date)] Starting Phase 19 (plant cell physiology)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_cell_layers.py" \
    --host localhost --port 8090 \
    --passes 6 \
    2>&1 | tee -a "$LOG_DIR/cell_layers.log" | tail -5
echo "[$(date)] Phase 19 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 20: JSON RESPONSES (scope-marked, NL-balanced) ────────────────────
# Every batch interleaves JSON-asked + NL-asked pairs so JSON cannot dominate.
echo "[$(date)] Starting Phase 20 (JSON response training)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_json_responses.py" \
    --node localhost:8090 \
    2>&1 | tee -a "$LOG_DIR/json_responses.log" | tail -5
echo "[$(date)] Phase 20 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 21: NATURAL-LANGUAGE REBALANCE ───────────────────────────────────
# Counterweight to JSON-heavy training: 200 repeats of plain-English Q&A
# across science / history / geography / math / everyday.
echo "[$(date)] Starting Phase 21 (NL rebalance)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/retrain_natural_language.py" \
    --node localhost:8090 \
    2>&1 | tee -a "$LOG_DIR/nl_rebalance.log" | tail -5
echo "[$(date)] Phase 21 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 22: COW ANATOMY + BEHAVIOUR + LIFECYCLE ──────────────────────────
echo "[$(date)] Starting Phase 22 (cow anatomy + behaviour)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_cow_anatomy.py" \
    --node "$NODE" \
    2>&1 | tee -a "$LOG_DIR/cow_anatomy.log" | tail -5
echo "[$(date)] Phase 22 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 23: COW 3D ANATOMICAL POSITIONS ──────────────────────────────────
echo "[$(date)] Starting Phase 23 (cow 3D anatomy)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_cow_3d_anatomy.py" \
    2>&1 | tee -a "$LOG_DIR/cow_3d_anatomy.log" | tail -5
echo "[$(date)] Phase 23 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 24: COW MULTI-VIEW DENSE SURFACE ─────────────────────────────────
echo "[$(date)] Starting Phase 24 (cow multi-view surface)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_cow_multiview_surface.py" \
    2>&1 | tee -a "$LOG_DIR/cow_multiview.log" | tail -5
echo "[$(date)] Phase 24 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 25: COW BARREL-SURFACE REFINEMENT ────────────────────────────────
# Shifts existing cow_surf_XX_XX centroids toward the correct barrel cross-section.
echo "[$(date)] Starting Phase 25 (cow barrel-surface retrain)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/retrain_barrel_surface.py" \
    2>&1 | tee -a "$LOG_DIR/cow_barrel.log" | tail -5
echo "[$(date)] Phase 25 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 26: FOREIGN LANGUAGE CORPUS (stages 36-40) ───────────────────────
# 20 languages: Project Gutenberg native + English language-learning books +
# Wikibooks language courses + built-in alphabets/grammar/phrases + Wiktionary.
echo "[$(date)] Starting Phase 26 (foreign languages, stages 36-40)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/build_foreign_language_corpus.py" \
    --stages 36,37,38,39,40 \
    --node localhost:8090 \
    2>&1 | tee -a "$LOG_DIR/foreign_lang.log" | tail -5
echo "[$(date)] Phase 26 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 27: CHESS OUTCOME + MOVE PREDICTION ──────────────────────────────
# Bounded: 5 iterations on up to 1000 games to avoid the infinite loop default.
# Skips cleanly if the PGN dataset is missing.
echo "[$(date)] Starting Phase 27 (chess training, 5 bounded iterations)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/chess_training_loop.py" \
    --max-games 1000 \
    --max-iterations 5 \
    2>&1 | tee -a "$LOG_DIR/chess.log" | tail -5 \
    || echo "[$(date)] Phase 27 skipped (chess dataset not preprocessed?)" | tee -a "$LOG_DIR/run_all.log"
echo "[$(date)] Phase 27 done" | tee -a "$LOG_DIR/run_all.log"

# ─── PHASE 28: OBSTACLE COURSE (Playwright; best effort) ────────────────────
# Requires Playwright + a viewer of the obstacle UI.  Skipped on environments
# without the browser drivers; the rest of the curriculum still runs.
echo "[$(date)] Starting Phase 28 (obstacle course, best effort)..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_obstacle.py" \
    --node "$NODE" \
    --reps 8 \
    2>&1 | tee -a "$LOG_DIR/obstacle.log" | tail -5 \
    || echo "[$(date)] Phase 28 skipped (Playwright not available?)" | tee -a "$LOG_DIR/run_all.log"
echo "[$(date)] Phase 28 done" | tee -a "$LOG_DIR/run_all.log"

reinforce_conversations 3

# ─── PHASE 29: MULTI-POOL CONCEPT BINDING (GA-tuned path) ────────────────────
# After the slow-pool char-chain decoder is trained, bind the high-priority
# Q->A pairs (greetings + identity facts + 113 code Q&A) through
# /multi_pool/train_pair so the GA-experimental winning genome
# (use_trigrams=true, lr=0.825, passes=35, mp_confidence_threshold=0.345)
# actually fires at chat time.  Without this step the GA gain is dormant —
# the curriculum trains the slow pool only.
#
# /query/integrated routes to multi-pool first; high-confidence bindings
# here mean instant high-confidence chat replies for greetings, identity,
# and code questions without disturbing the broader corpus knowledge in
# the slow pool.
echo "[$(date)] Binding key Q->A concepts in multi-pool fabric..." | tee -a "$LOG_DIR/run_all.log"
PYTHONIOENCODING=utf-8 python3 "$SCRIPTS/train_concept_bindings.py" \
    --node "$NODE" \
    2>&1 | tee -a "$LOG_DIR/concept_bindings.log" | tail -5
echo "[$(date)] Multi-pool concept bindings done" | tee -a "$LOG_DIR/run_all.log"

# Verifier round 1 (post-Phase 29): runs the Q->A battery and reinforces
# every pair that passes — locks in the dopamine-gated capture on the
# synapses that just produced the correct answer.  Failures here are
# expected for any pair that the curriculum hasn't covered yet; they
# flow into the regression queue and get retried after the final
# re-anchor below.
verify_and_reinforce "post-phase29"

# ─── FINAL: RE-ANCHOR FOUNDATIONS + CONVERSATIONS ────────────────────────────
reinforce_foundations 3
reinforce_conversations 10

# Verifier round 2 (post-final-anchor): the final re-anchor should have
# pushed any straggler pairs into a recallable state.  This pass picks
# up the second-tier reinforcement and produces the canonical pass-rate
# metric for the whole curriculum run.
verify_and_reinforce "post-final-anchor"

# ─── CHECKPOINT: PERSIST SLOW-POOL + MULTI-POOL TO DISK ──────────────────────
echo "[$(date)] Saving pools to disk..." | tee -a "$LOG_DIR/run_all.log"
curl -s -X POST "$NODE/neuro/checkpoint" | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

# Verifier round 3 (post-checkpoint, --check-only): final pass-rate metric
# with no further training side-effects — what you see here is what gets
# served to /chat from now on.  Exit code propagates: 0 = clean, 3 = some
# regressions remain in the regression queue.
verify_and_reinforce "final-readout" "--check-only"

echo "===== Full curriculum training complete $(date) =====" | tee -a "$LOG_DIR/run_all.log"
