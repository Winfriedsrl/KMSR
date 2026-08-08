#!/usr/bin/env bash
# Re-runs every model of the paper over the 350-query ground truth.
#
# Each config holds ONE model, so each invocation creates exactly one run under
# runs/paper_hard/run_<timestamp>/. Note down the printed IDs ("Run dir"): they are what
# make_table1.py and the Figure 3 plot consume.
#
# Every model is a full sweep of LLM calls over 350 queries (reranker for the baselines;
# classifier + reranker for KMSR), so run it under nohup/tmux. If a run is interrupted,
# resume it without redoing the queries already done:
#   ./venv/bin/python src/main.py evaluate-paper --hard --resume runs/paper_hard/<run_id>
#
# Usage:
#   ./paper_experiments/run_all.sh              # all 10 models
#   ./paper_experiments/run_all.sh B1 K3        # a subset
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./venv/bin/python}"
GT="${GT:-paper_experiments/data/wf_test_set_all.csv}"
CFG_DIR="paper_experiments/configs"

# The ground truth ships with this repository; the corpora behind the Elasticsearch indices,
# and the offense->circumstance mapping loaded at runtime, do not (README §2).
missing=""
for f in "$GT" data/final/wf_key_table.csv; do
  [ -f "$f" ] || missing="$missing  $f"$'\n'
done
if [ -n "$missing" ]; then
  echo "Cannot run: missing input files" >&2
  printf '%s' "$missing" >&2
  echo "data/final/wf_key_table.csv is the platform's proprietary offense->circumstance mapping" >&2
  echo "and is NOT part of the public release. See README §2." >&2
  exit 1
fi

declare -A CONFIGS=(
  [B1]="B1_sparse_all.json"
  [B2]="B2_dense_all.json"
  [B3]="B3_hybrid_all.json"
  [B4]="B4_sparse_restricted.json"
  [B5]="B5_dense_restricted.json"
  [B6]="B6_hybrid_restricted.json"
  [K1]="K1_kmsr_2stage_map.json"
  [K2]="K2_kmsr_query_reduction.json"
  [K3]="K3_kmsr_llm_reranking.json"
  [A1]="A1_kmsr_no_mapping.json"
)
ORDER=(B1 B2 B3 B4 B5 B6 K1 K2 K3 A1)

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=("${ORDER[@]}")
fi

for id in "${TARGETS[@]}"; do
  cfg="${CONFIGS[$id]:-}"
  if [ -z "$cfg" ]; then
    echo "Unknown ID: $id (valid: ${ORDER[*]})" >&2
    exit 1
  fi
  echo
  echo "=============================================================="
  echo "== $id  ($cfg)"
  echo "=============================================================="
  $PY src/main.py evaluate-paper \
    --hard \
    --gt_csv "$GT" \
    --models_config "$CFG_DIR/$cfg"
done

echo
echo "Done. Update the 'run' fields in paper_experiments/runs_manifest.json with the new IDs, then:"
echo "  $PY paper_experiments/make_table1.py --manifest paper_experiments/runs_manifest.json"
