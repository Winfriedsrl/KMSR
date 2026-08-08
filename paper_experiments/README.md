# paper_experiments — reproducibility kit

Everything needed to re-run the experiments of *"The Impact of Domain-Specific Knowledge in
Legal AI: A Case Study on Italian Criminal Law"* (KMSR): one config file per row of Table 1,
the exact command that launches it, and the scripts that rebuild the table and the figure from
the saved runs.

```
paper_experiments/
├── README.md                      this file
├── data/wf_test_set_all.csv       the 350-query ground truth (shipped)
├── configs/                       one config per model (one model per file = one run per file)
├── build_indices.sh               builds the 4 corpora on Elasticsearch (§3)
├── run_all.sh                     runs the 10 models over the ground truth (§4)
├── make_table1.py                 rebuilds Table 1 (+ bootstrap CIs) from the runs (§5)
└── runs_manifest.json             which run produced which row of the paper (provenance)
```

---

## 0. What you can do with what you have

The corpora this system retrieves over are the platform's **proprietary data** and are not part
of this release, and neither are the run directories. That draws a hard line through this kit:

| | Public release | Winfried (full data) |
|---|---|---|
| Read every config and command; audit the parameters of §4 | ✓ | ✓ |
| Inspect the ground truth, recount 350 = 120 A / 227 B / 3 A+B | ✓ | ✓ |
| §3 Build the indices | ✗ needs the corpora | ✓ |
| §4 Run the 10 models | ✗ needs the corpora | ✓ |
| §5 Rebuild Table 1 / Figure 3 | ✗ needs the runs | ✓ |

**If you are reading the paper**: this folder documents exactly *what was run* — every retrieval
parameter, every prompt, every corpus size, the full metric definitions, and the ground truth
itself, which you can open and count. It does not let you recompute the published numbers: that
requires the proprietary corpora (to redo retrieval) or the archived run directories (to redo the
scoring). Sections 3–5 are written for whoever holds that data.

**If you are at Winfried**: you can do all of the above. Note that you do not need to re-run
anything to get Table 1 — the archived runs pinned in `runs_manifest.json` are the artifacts the
paper's numbers were computed from, and §5 regenerates the table from them in seconds.

---

## 1. Prerequisites

Needed for §3 and §4 only. §5 needs nothing beyond Python and the run directories.

| | |
|---|---|
| Elasticsearch | running at `ES_HOST` (default `http://localhost:9201`) — `docker compose -f docker-compose.local.yaml up -d` |
| Python | `./venv` with `requirements.txt` installed |
| OpenAI | API key with access to `gpt-5-mini` and `text-embedding-3-large` |
| Disk | ~56 GB of indices (`paper_reati_circ_light` is nearly all of it) |

`.env` in the repository root, with at least these keys:

```bash
OPENAI_API_KEY="sk-..."
ES_HOST="http://localhost:9201"

# paper indices (the names must be exactly these: see §3)
ES_INDEX_PAPER_PERIODI_FLAT="paper_periodi_flat"
ES_INDEX_PAPER_REATI="paper_reati"
ES_INDEX_PAPER_REATI_CIRC_LIGHT="paper_reati_circ_light"

# required at import time by src/config.py even though the paper does not use them
ES_INDEX_NAME="..."
ES_INDEX_NAME_COMBO="..."
```

> `src/config.py` reads `ES_INDEX_NAME` and `ES_INDEX_NAME_COMBO` via `os.environ[...]`: if they
> are missing, every command dies with a `KeyError` at import, even though the paper never uses
> those indices.

---

## 2. Data

### Included in this release

| Role (paper) | File | Size | sha256 (first 16) |
|---|---|---|---|
| Ground truth, 350 queries | `paper_experiments/data/wf_test_set_all.csv` | 441 rows / 350 queries | `831cf7257257a137` |

Columns `query,result,type`; one query spans several rows when its answer is made of several
provisions. The composition by type is the one reported in the paper: **120 flat (A) · 227
block-structured (B) · 3 mixed (A+B)** = 350.

Internally this is the same file (identical sha256) as `data/final/gt/wf_test_set_all.csv`, which
is the path recorded in the archived runs' `meta.json`.

### Not included — proprietary

The corpora the system retrieves over, and the mapping that drives the *mapper* stage. Winfried
holds them; the digests below are there so you can confirm you have the exact files used in the
paper.

| Role (paper) | File | Rows | sha256 (first 16) |
|---|---|---|---|
| *All passages* (10,784) | `data/wf_periodi_aggiornato_paper.csv` | 10,784 | `df1ddc13d39e3079` |
| *Restricted passages* (2,040) | `data/final/to_index/wf_reati_circostanze_flat.csv` | 2,040 | `ed29d133ad5f1ced` |
| *Offense combinations* (23,991) | `data/final/to_index/wf_reati.csv` | 23,991 | `0126f05a15123e89` |
| *Circumstantiated comb.* (1,299,924) | `data/final/to_index/wf_reati_circostanziati_light.csv` | 1,299,924 | `e1da54fdff359540` |
| Offense→circumstance mapping (the *mapper*) | `data/final/wf_key_table.csv` | 1,323,915 | `a6a5c4b3f768c361` |

Also not released: the run directories under `runs/paper_hard/` (§6), which is why §5 is
Winfried-only. `build_indices.sh` and `run_all.sh` check for these inputs and stop with an
explicit message if they are absent.

---

## 3. Building the indices

> Requires the proprietary corpora of §2.

```bash
./paper_experiments/build_indices.sh
```

| ES index | Corpus in the paper | Docs | Used by |
|---|---|---|---|
| `wf_periodi_aggiornato` | all passages | 10,784 | B1, B2, B3 |
| `paper_periodi_flat` | restricted passages | 2,040 | B4, B5, B6 · and by KMSR with `period_rerank`, to fetch passage text |
| `paper_reati` | offense combinations | 23,991 | KMSR stage 1 |
| `paper_reati_circ_light` | circumstantiated-offense comb. | 1,299,924 | KMSR stage 3 |

⚠️ **The names must match exactly.** `main.py` uses `make_index_name()`, which appends `_2`, `_3`…
when the index already exists: re-indexing on a non-empty Elasticsearch silently creates
`paper_reati_2` while the configs keep pointing elsewhere. Three names come from `.env`;
`wf_periodi_aggiornato` is instead **hard-wired** in the `"index"` field of configs B1–B3 (the
override applied at `evaluation.py:335` on top of the default field profile). To rebuild an
index, delete it first: `curl -X DELETE "$ES_HOST/<name>"`.

💰 Embedding `paper_reati_circ_light` (1.3M units) is the dominant cost. Under the paper's
configuration (`stage3_mode: "sparse"`) the dense field of that index is **never queried**, so the
script indexes it BM25-only. Use `EMBED_CIRC=1 ./paper_experiments/build_indices.sh` to add the
embeddings, if you want to explore dense/hybrid `stage3_mode` variants.

---

## 4. The models

> Requires the indices of §3, hence the proprietary corpora of §2.

One config = one model = one run. Generic command:

```bash
./venv/bin/python src/main.py evaluate-paper \
  --hard \
  --gt_csv paper_experiments/data/wf_test_set_all.csv \
  --models_config paper_experiments/configs/<CONFIG>.json
```

`--hard` selects the ground truth in `query/result/type` format, hence the block metrics and the
query types: **without `--hard` you do not get the paper's metrics**, and the run lands in
`runs/paper/` instead of `runs/paper_hard/`. It must be repeated on `--resume` too.

All of them in sequence (hours — use `tmux`/`nohup`):

```bash
./paper_experiments/run_all.sh          # all 10
./paper_experiments/run_all.sh B1 K3    # a subset
```

### Overview

| ID | Row in the paper | Corpus | Retrieval | Config | Reference run |
|---|---|---|---|---|---|
| B1 | Sparse Baseline / all | all | BM25 | `B1_sparse_all.json` | `run_20260627_161717_311228` |
| B2 | Dense Baseline / all | all | embedding | `B2_dense_all.json` | `run_20260627_173517_190627` |
| B3 | Hybrid Baseline / all | all | RRF | `B3_hybrid_all.json` | `run_20260627_190149_070149` |
| B4 | Sparse Baseline / restricted | restricted | BM25 | `B4_sparse_restricted.json` | `run_20260626_231052_215034` |
| B5 | Dense Baseline / restricted | restricted | embedding | `B5_dense_restricted.json` | `run_20260626_231116_765270` |
| B6 | Hybrid Baseline / restricted | restricted | RRF | `B6_hybrid_restricted.json` | `run_20260627_002814_469184` |
| K1 | KMSR: 2-stage map. | off./circ.off. comb. | dense + sparse | `K1_kmsr_2stage_map.json` | `run_20260629_093549_474218` |
| K2 | + query reduction | off./circ.off. comb. | dense + sparse | `K2_kmsr_query_reduction.json` | `run_20260629_074919_641832` |
| K3 | + LLM reranking (**full system**) | off./circ.off. comb. | dense + sparse | `K3_kmsr_llm_reranking.json` | `run_20260627_014723_777468` |
| A1 | ablation −mapping (0.488, §7 of the paper) | off./circ.off. comb. | dense + sparse | `A1_kmsr_no_mapping.json` | `run_20260629_075017_042397` ¹ |

¹ archived under `runs/paper_hard_altri/`; new runs still land in `runs/paper_hard/`.

### B1–B6 — *retrieve-then-rerank* baselines (§5.2 of the paper)

They retrieve the top-30 passages from the corpus and have `gpt-5-mini` rerank them down to the
top-10. The six rows are 3 methods × 2 corpora; the only difference between the "all" and
"restricted" twins is the `"index"` field in the config.

```bash
# B1 · sparse, full corpus
./venv/bin/python src/main.py evaluate-paper --hard \
  --gt_csv paper_experiments/data/wf_test_set_all.csv \
  --models_config paper_experiments/configs/B1_sparse_all.json

# B2 · dense (text-embedding-3-large)
#   → configs/B2_dense_all.json
# B3 · hybrid RRF (k_RRF=60, window 100)
#   → configs/B3_hybrid_all.json
# B4/B5/B6 · the same three without the "index" field → restricted corpus (paper_periodi_flat)
#   → configs/B4_sparse_restricted.json, B5_dense_restricted.json, B6_hybrid_restricted.json
```

### K1 — KMSR, 2-stage with mapping (no query reduction, no reranker)

Dense stage 1 over `paper_reati` (m=30) → *mapper* (`wf_key_table.csv`) → sparse stage 3 over the
admissible circumstantiated combinations (n=10) → pool → LLM classifier (j=20) → charges dissolved
into passages, truncated to k=10 in combination order.

```bash
./venv/bin/python src/main.py evaluate-paper --hard \
  --gt_csv paper_experiments/data/wf_test_set_all.csv \
  --models_config paper_experiments/configs/K1_kmsr_2stage_map.json
```

### K2 — + query reduction

Same as K1, but the regex *reductor* splits the query: `stage1_input: "q_reato"` (only the
offense core goes to the dense stage), `stage3_input: "q"` (the full query goes to the sparse
stage). The only delta vs K1: `stage1_input` + `split_method`.

```bash
./venv/bin/python src/main.py evaluate-paper --hard \
  --gt_csv paper_experiments/data/wf_test_set_all.csv \
  --models_config paper_experiments/configs/K2_kmsr_query_reduction.json
```

### K3 — + LLM reranking → **the paper's system**

Same as K2 plus `period_rerank: true`: the passages dissolved from the selected charges are
reordered by an `LLMReranker` (`gpt-5-mini`, prompt `prompts/ranker_v1.txt`) before truncating to
k=10. Passage text comes from `paper_periodi_flat`. This is the **0.706 / 0.625** row of Table 1.

```bash
./venv/bin/python src/main.py evaluate-paper --hard \
  --gt_csv paper_experiments/data/wf_test_set_all.csv \
  --models_config paper_experiments/configs/K3_kmsr_llm_reranking.json
```

### A1 — ablation: KMSR without the mapping

`no_mapping: true`: stage 3 becomes a **global** search over the circumstantiated corpus, not
filtered by the combinations admissible for the candidate offense. Everything else is K1. It is
the counterfactual behind the sentence in §7 of the paper: *"without it the two-stage pool stays
at 0.488"*.

```bash
./venv/bin/python src/main.py evaluate-paper --hard \
  --gt_csv paper_experiments/data/wf_test_set_all.csv \
  --models_config paper_experiments/configs/A1_kmsr_no_mapping.json
```

### Fixed parameters (§5.4 of the paper)

| | |
|---|---|
| KMSR | m=30 · n=10 · j=20 · k=10 |
| Baselines | top-30 retrieved → reranked to 10 |
| RRF | `k_RRF`=60, window 100 |
| LLM (classifier + reranker) | `gpt-5-mini` |
| Embeddings | `text-embedding-3-large` (3072 dims) |
| Classifier prompt | `prompts/classifier_v2.txt` (`classifier_version: "v2"`) |
| Reranker prompt | `prompts/ranker_v1.txt` |
| Reductor | regex, `src/splitter.py::_regex_split` (`split_method: "regex"`) |

### If a run is interrupted

`details.csv` is written as a stream, so a run resumes from the next query.

```bash
./venv/bin/python src/main.py evaluate-paper --hard --resume runs/paper_hard/<run_id>
```

---

## 5. From runs to results

> Requires the run directories of §6, which are not part of the public release.

### Table 1

```bash
./venv/bin/python paper_experiments/make_table1.py \
  --manifest paper_experiments/runs_manifest.json            # markdown
./venv/bin/python paper_experiments/make_table1.py \
  --manifest paper_experiments/runs_manifest.json --format latex
```

After re-running the experiments, replace the `run` fields in `runs_manifest.json` with the new
IDs (printed by `evaluate-paper`). The same numbers are also visible in the Streamlit UI
(`streamlit run src/app.py` → "Paper" explorer, *confidence intervals* toggle): this script is the
CLI version of that view.

**Metrics.** Every query contributes the metric suited to its type, averaged over the whole set:

```
Weighted Recall  = (nA·mean_A(Recall) + nB·mean_B(Block Recall) + nAB·s_AB) / n
Weighted Ranking = (nA·mean_A(NDCG)   + nB·mean_B(Block AP)     + nAB·r_AB) / n
```

with A=flat, B=block-structured, A+B=mixed (the mean of the two). CIs: 95% percentile bootstrap
over 1000 resamples of the queries (seed 0), resampling the whole pool — so the interval also
absorbs the variability of the A/B composition. Definitions live in
`evaluation.build_simple_qtype_tables` / `build_ranking_overview_df`; the primitives
(`recall`, `ndcg`, `block_recall`, `block_ap`) in `src/metrics.py`.

### Figure 3 (performance by query category)

```bash
./venv/bin/python paper/plot_ab_detail.py \
  --runs run_20260627_190149_070149 run_20260626_231052_215034 run_20260627_014723_777468 \
  --runs-dir runs/paper_hard \
  --dataset wf_test_set_all.csv \
  --labels "BL-All" "BL-Restricted" "KMSR" \
  --split --out paper/fig_ab_detail.png
```

`--split` writes the two panels separately (`fig_ab_detail_A.png`, `fig_ab_detail_B.png`) = left
and right of Figure 3. The three runs are the **best** baseline of each corpus (B3 hybrid-all,
B4 sparse-restricted) plus the full system (K3). Expected values:

| | Flat Recall | NDCG | Block Recall | Block AP |
|---|---|---|---|---|
| BL-All | 0.97 | 0.89 | 0.21 | 0.15 |
| BL-Restricted | 0.95 | 0.88 | 0.26 | 0.20 |
| KMSR | 0.95 | 0.91 | 0.58 | 0.47 |

> `paper/` is git-ignored, so the plotting script is **not under version control**. If you want
> the reproducibility kit to be self-contained in git, move (or copy) it in here.

---

## 6. What a run contains

`runs/paper_hard/run_<timestamp>/`:

| File | Content |
|---|---|
| `meta.json` | timestamp, ground truth used, model config — the source of truth for *what* was executed |
| `details.csv` | `Query, Query Type, Model, Rank, Periodo, Fonte Normativa, Score, Error` — the raw output, top-10 per query |
| `gt_context.json` | per query: type, expected passages, expected blocks |
| `debug.jsonl` / `debug.json` | KMSR only: pipeline stages (split, stage 1/3 candidates, pool, classifier answer) |
| `status.json` | `running` / `complete` (used by `--resume` and by the UI) |

Metrics are never stored: they are always recomputed from `details.csv` + `gt_context.json`, so a
change in a metric definition applies retroactively to every run.
