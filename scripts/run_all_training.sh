#!/usr/bin/env bash
# run_all_training.sh — sequential training runner
# Stage 9 (cow mesh) → Stages 10-17 (STEM) → Stages 18-22 (language/docs)
#                     → Stages 23-24 (library docs + Stack Overflow)
# Run from project root. Logs to D:/w1z4rdv1510n-data/training/

set -euo pipefail

DATA="D:/w1z4rdv1510n-data"
LOG_DIR="$DATA/training"
mkdir -p "$LOG_DIR"

echo "===== W1z4rD Training Runner started $(date) =====" | tee -a "$LOG_DIR/run_all.log"

# Stage 9: cow mesh (resume — Sketchfab + Wikipedia parts)
echo "[$(date)] Starting Stage 9 (mesh training)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_cow_dataset.py \
    --stages 9 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/stage9_run2.log" | tail -1
echo "[$(date)] Stage 9 done" | tee -a "$LOG_DIR/run_all.log"

# Stages 10-17: STEM + Electronics + Embedded + Quantum + Astrophysics
echo "[$(date)] Starting Stages 10-17 (STEM training)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_stem_dataset.py \
    --stages 10,11,12,13,14,15,16,17 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --wiki-max 5000 \
    --libretexts-max 30 \
    --arxiv-max 200 \
    2>&1 | tee -a "$LOG_DIR/stem_run.log"
echo "[$(date)] Stages 10-17 done" | tee -a "$LOG_DIR/run_all.log"

# Stages 18-22: Language corpus (Gutenberg, Wikibooks, MDN, official docs, RFCs)
echo "[$(date)] Starting Stages 18-22 (language + documentation corpus)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_language_corpus.py \
    --stages 18,19,20,21,22 \
    --node localhost:8090 \
    --gutenberg-chars 60000 \
    --gutenberg-passages 20 \
    --wikibooks-chars 15000 \
    2>&1 | tee -a "$LOG_DIR/language_run.log"
echo "[$(date)] Stages 18-22 done" | tee -a "$LOG_DIR/run_all.log"

# Stages 23-24: Library docs + Stack Overflow accepted answers
echo "[$(date)] Starting Stages 23-24 (library docs + Stack Overflow)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_library_corpus.py \
    --stages 23,24 \
    --node localhost:8090 \
    --lib-pairs-per-file 30 \
    --lib-prose-chars 8000 \
    --so-per-tag 50 \
    --so-min-votes 5 \
    2>&1 | tee -a "$LOG_DIR/library_run.log"
echo "[$(date)] Stages 23-24 done" | tee -a "$LOG_DIR/run_all.log"

# Stage 25: LibreTexts comprehensive corpus — all 13 domains, all bookshelves
echo "[$(date)] Starting Stage 25 (LibreTexts comprehensive corpus)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_libretexts_corpus.py \
    --stages 25 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-pages 5000 \
    2>&1 | tee -a "$LOG_DIR/libretexts_run.log"
echo "[$(date)] Stage 25 done" | tee -a "$LOG_DIR/run_all.log"

# Stage 26: Biodiversity visual-ID corpus — plants, animals, fungi
echo "[$(date)] Starting Stage 26 (biodiversity visual-ID corpus)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_biodiversity_corpus.py \
    --stages 26 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-per-group 2000 \
    2>&1 | tee -a "$LOG_DIR/biodiversity_run.log"
echo "[$(date)] Stage 26 done" | tee -a "$LOG_DIR/run_all.log"

# Stages 27-28: Cognitive/IQ reasoning + Sorting algorithms
echo "[$(date)] Starting Stages 27-28 (cognitive + sorting)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_cognitive_corpus.py \
    --stages 27,28 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/cognitive_run.log"
echo "[$(date)] Stages 27-28 done" | tee -a "$LOG_DIR/run_all.log"

# Stage 29: World English Bible — complete 66-book corpus
echo "[$(date)] Starting Stage 29 (World English Bible)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_bible_corpus.py \
    --stages 29 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/bible_run.log"
echo "[$(date)] Stage 29 done" | tee -a "$LOG_DIR/run_all.log"

# Stages 30-33: Medical/Psychology/Genetic Engineering/Reverse Aging (NCBI/NLM)
echo "[$(date)] Starting Stages 30-33 (medical corpus from NCBI/NLM)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_medical_corpus.py \
    --stages 30,31,32,33 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-per-query 80 \
    2>&1 | tee -a "$LOG_DIR/medical_run.log"
echo "[$(date)] Stages 30-33 done" | tee -a "$LOG_DIR/run_all.log"

# Stage 34: Mathematics — arithmetic through topology, symbol dictionary, notation mastery
echo "[$(date)] Starting Stage 34 (mathematics corpus)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_math_corpus.py \
    --stages 34 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    2>&1 | tee -a "$LOG_DIR/math_run.log"
echo "[$(date)] Stage 34 done" | tee -a "$LOG_DIR/run_all.log"

# Stage 35: Evidence-based pedagogy + curriculum design
echo "[$(date)] Starting Stage 35 (pedagogy & curriculum design)..." | tee -a "$LOG_DIR/run_all.log"
python scripts/build_pedagogy_corpus.py \
    --stages 35 \
    --node localhost:8090 \
    --data-dir "$DATA" \
    --max-per-query 30 \
    2>&1 | tee -a "$LOG_DIR/pedagogy_run.log"
echo "[$(date)] Stage 35 done" | tee -a "$LOG_DIR/run_all.log"

echo "===== All training complete $(date) =====" | tee -a "$LOG_DIR/run_all.log"
