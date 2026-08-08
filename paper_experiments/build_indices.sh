#!/usr/bin/env bash
# Builds the 4 corpora of the paper on Elasticsearch (§5.1).
#
# MIND THE NAMES: main.py calls make_index_name(), which appends "_2", "_3", ... when the
# index already exists. The names must stay EXACTLY as below, because three of them are read
# from .env and one (wf_periodi_aggiornato) is hard-wired in the "all" baseline configs.
# To rebuild an index that already exists, delete it first:
#   curl -X DELETE "$ES_HOST/<index_name>"
#
# Cost: embedding paper_reati_circ_light (1.3M units) is by far the most expensive step.
# Under the paper's configuration (stage3_mode = "sparse") that dense field is NEVER queried,
# so this script indexes it BM25-only. Set EMBED_CIRC=1 only if you want to try dense/hybrid
# stage3_mode variants.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./venv/bin/python}"
EMB="text-embedding-3-large"
EMBED_CIRC="${EMBED_CIRC:-0}"

# The four corpora are the platform's proprietary data and are NOT part of the public
# release (README §2). Fail early and clearly rather than deep inside pandas.
missing=""
for f in data/wf_periodi_aggiornato_paper.csv \
         data/final/to_index/wf_reati_circostanze_flat.csv \
         data/final/to_index/wf_reati.csv \
         data/final/to_index/wf_reati_circostanziati_light.csv; do
  [ -f "$f" ] || missing="$missing  $f"$'\n'
done
if [ -n "$missing" ]; then
  echo "Cannot build the indices: missing corpora" >&2
  printf '%s' "$missing" >&2
  echo "These files are the platform's proprietary data and are NOT part of the public" >&2
  echo "release. See README §2 for what each one is and its expected sha256." >&2
  exit 1
fi

echo "== 1/4 all passages (10,784) -> wf_periodi_aggiornato"
$PY src/main.py index \
  --name_prefix wf_periodi_aggiornato \
  --data_source data \
  --dataset_type periodi_flat \
  --source_file wf_periodi_aggiornato_paper.csv \
  --embedders "$EMB"

echo "== 2/4 restricted passages (2,040) -> paper_periodi_flat"
$PY src/main.py index \
  --name_prefix paper_periodi_flat \
  --data_source data/final/to_index \
  --dataset_type periodi_flat \
  --source_file wf_reati_circostanze_flat.csv \
  --embedders "$EMB"

echo "== 3/4 offense combinations (23,991) -> paper_reati"
$PY src/main.py index \
  --name_prefix paper_reati \
  --data_source data/final/to_index \
  --dataset_type reati \
  --embedders "$EMB"

echo "== 4/4 circumstantiated-offense combinations (1,299,924) -> paper_reati_circ_light"
if [ "$EMBED_CIRC" = "1" ]; then
  $PY src/main.py index \
    --name_prefix paper_reati_circ_light \
    --data_source data/final/to_index \
    --dataset_type reati_circostanziati_light \
    --embedders "$EMB"
else
  echo "   (BM25 only: the paper's stage 3 is sparse. Set EMBED_CIRC=1 to add embeddings)"
  $PY src/main.py index \
    --name_prefix paper_reati_circ_light \
    --data_source data/final/to_index \
    --dataset_type reati_circostanziati_light
fi

echo
echo "Check the expected document counts:"
echo '  curl -s "$ES_HOST/_cat/indices?v&h=index,docs.count" | grep -E "paper_|wf_periodi_aggiornato"'
echo "  wf_periodi_aggiornato 10784 · paper_periodi_flat 2040 · paper_reati 23991 · paper_reati_circ_light 1299924"
