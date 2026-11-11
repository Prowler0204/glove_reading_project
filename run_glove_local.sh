#!/usr/bin/env bash
# Run from the root of the ORIGINAL stanfordnlp/GloVe repository under WSL/Linux.
# Usage: bash run_glove_local.sh tiny   OR   bash run_glove_local.sh text8
set -euo pipefail
MODE="${1:-tiny}"
if [[ ! -f src/glove.c || ! -f Makefile ]]; then
  if [[ -d GloVe && -f GloVe/src/glove.c && -f GloVe/Makefile ]]; then
    echo "Switching into GloVe repository directory..."
    cd GloVe
    cp -u ../probe_vectors.py . 2>/dev/null || true
  else
    echo 'ERROR: Run this script inside the stanfordnlp/GloVe repository root.' >&2
    exit 1
  fi
fi

if [[ "$MODE" == "tiny" ]]; then
  CORPUS=toy_corpus.txt
  python3 - <<'PY'
from pathlib import Path
sentences = [
    'the king and the queen rule the kingdom together',
    'the man and the woman walk through the city',
    'the prince and the princess visit the castle',
    'paris is the capital of france and london is the capital of england',
    'rome is the capital of italy and berlin is the capital of germany',
    'a cat is an animal and a dog is an animal',
    'the cat sleeps on the sofa and the dog sleeps on the carpet',
    'the river flows to the ocean and water flows from the mountain',
    'the scientist reads a book about physics and chemistry',
    'the doctor works at a hospital and the teacher works at a school',
]
Path('toy_corpus.txt').write_text(('\n'.join(sentences)+'\n')*300, encoding='utf-8')
PY
  MIN_COUNT=1; WINDOW=4; DIM=25; ITERS=20; THREADS=2; MEMORY=0.25; XMAX=10
elif [[ "$MODE" == "text8" ]]; then
  CORPUS=text8
  if [[ ! -f "$CORPUS" ]]; then
    echo 'ERROR: text8 not found. Download and unzip text8.zip first; see the PDF instructions.' >&2
    exit 1
  fi
  MIN_COUNT=5; WINDOW=15; DIM=50; ITERS=15; THREADS=8; MEMORY=2.0; XMAX=10
else
  echo "Usage: bash run_glove_local.sh [tiny|text8]" >&2; exit 1
fi

mkdir -p logs
printf 'Building the original Stanford C binaries...\n'
make
printf 'Mode=%s  dim=%s  iterations=%s  window=%s  threads=%s\n' "$MODE" "$DIM" "$ITERS" "$WINDOW" "$THREADS"

build/vocab_count -min-count "$MIN_COUNT" -verbose 2 \
  < "$CORPUS" > vocab.txt 2> logs/vocab.log
build/cooccur -memory "$MEMORY" -vocab-file vocab.txt -verbose 2 \
  -window-size "$WINDOW" -symmetric 1 -distance-weighting 1 \
  < "$CORPUS" > cooccurrence.bin 2> logs/cooccur.log
build/shuffle -memory "$MEMORY" -verbose 2 -seed 42 \
  < cooccurrence.bin > cooccurrence.shuf.bin 2> logs/shuffle.log
build/glove -save-file vectors -threads "$THREADS" \
  -input-file cooccurrence.shuf.bin -vocab-file vocab.txt \
  -x-max "$XMAX" -alpha 0.75 -eta 0.05 \
  -iter "$ITERS" -vector-size "$DIM" -binary 2 -model 2 \
  -seed 42 -verbose 2 2> logs/train.log

printf '\nTraining finished; evidence:\n'
ls -lh vocab.txt cooccurrence.bin cooccurrence.shuf.bin vectors.txt vectors.bin
printf '\nLast training log entries:\n'
tail -n 8 logs/train.log
printf '\nTo inspect learned vectors (requires numpy):\n'
printf 'python3 probe_vectors.py vectors.txt king queen paris london\n'
printf '\nFor text8 analogy benchmark (requires numpy):\n'
printf 'python3 eval/python/evaluate.py --vocab_file vocab.txt --vectors_file vectors.txt\n'
