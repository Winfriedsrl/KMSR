#!/usr/bin/env python3
"""Rebuild Table 1 of the paper (Weighted Recall / Weighted Ranking) from the saved runs.

Each run under `runs/paper_hard/<run_id>` is one model evaluated on the 350 ground-truth
queries. This script redoes the aggregation of the "Paper" explorer of the Streamlit UI, but
from the CLI, so the paper's table is reproducible without opening the interface.

Definitions (identical to `evaluation.build_simple_qtype_tables` /
`build_ranking_overview_df`, i.e. to the numbers shown in the UI):

    Weighted Recall  = (nA*meanA(Recall) + nB*meanB(Block Recall) + nAB*sAB) / n
    Weighted Ranking = (nA*meanA(NDCG)   + nB*meanB(Block AP)     + nAB*rAB) / n

where A = flat queries, B = block-structured queries, A+B = mixed (mean of both metrics),
and n = nA + nB + nAB.

Confidence intervals are 95% percentile bootstrap (B=1000) resampling the queries over the
whole pool, as in `paper_explorer_view._block_overview_ci` (seed 0): resampling the full set
means the interval also absorbs the variability of the A/B composition.

Examples
--------
  # every row of Table 1, from the manifest
  python paper_experiments/make_table1.py --manifest paper_experiments/runs_manifest.json

  # hand-picked runs (labels follow the same order)
  python paper_experiments/make_table1.py \
      --runs run_20260627_161717_311228 run_20260627_014723_777468 \
      --labels "Sparse Baseline (all)" "KMSR" \
      --format latex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluation import compute_summary_from_details  # noqa: E402

N_BOOT = 1000
ALPHA = 0.05


def _run_arrays(run_dir: Path, include_fonte: bool):
    """Per-query metrics of the run: (query_type[], {flat, block, ndcg, ap})."""
    details = pd.read_csv(run_dir / "details.csv", encoding="utf-8")
    gt = json.loads((run_dir / "gt_context.json").read_text(encoding="utf-8"))
    s = compute_summary_from_details(details, gt, include_fonte=include_fonte)
    if s.empty or "Query Type" not in s.columns:
        raise SystemExit(f"Run without a usable summary: {run_dir}")

    def col(name: str) -> np.ndarray:
        c = f"{name} M1"
        if c not in s.columns:
            return np.zeros(len(s))
        return pd.to_numeric(s[c], errors="coerce").fillna(0.0).to_numpy()

    arrays = {"flat": col("Recall"), "block": col("Block Recall"),
              "ndcg": col("NDCG"), "ap": col("Block AP")}
    return s["Query Type"].to_numpy(), arrays


def _weighted(qt: np.ndarray, a_metric: np.ndarray, b_metric: np.ndarray) -> float:
    """Mean weighted by type cardinality: A uses a_metric, B uses b_metric,
    A+B the mean of the two."""
    mA, mB, mAB = qt == "A", qt == "B", qt == "A+B"
    nA, nB, nAB = int(mA.sum()), int(mB.sum()), int(mAB.sum())
    n = len(qt)
    sA = a_metric[mA].mean() if nA else 0.0
    sB = b_metric[mB].mean() if nB else 0.0
    sAB = ((a_metric[mAB].mean() + b_metric[mAB].mean()) / 2) if nAB else 0.0
    return float((nA * sA + nB * sB + nAB * sAB) / n) if n else 0.0


def _run_scores(qt: np.ndarray, arrays: dict, with_ci: bool, seed: int = 0) -> dict:
    """Point estimates + bootstrap CIs for Weighted Recall and Weighted Ranking."""
    out = {
        "recall": (_weighted(qt, arrays["flat"], arrays["block"]), None, None),
        "ranking": (_weighted(qt, arrays["ndcg"], arrays["ap"]), None, None),
    }
    if not with_ci:
        return out

    n = len(qt)
    rng = np.random.default_rng(seed)
    boot = {"recall": [], "ranking": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        qti = qt[idx]
        boot["recall"].append(_weighted(qti, arrays["flat"][idx], arrays["block"][idx]))
        boot["ranking"].append(_weighted(qti, arrays["ndcg"][idx], arrays["ap"][idx]))
    for key, samples in boot.items():
        v = np.asarray(samples)
        lo = float(np.percentile(v, 100 * ALPHA / 2))
        hi = float(np.percentile(v, 100 * (1 - ALPHA / 2)))
        out[key] = (out[key][0], lo, hi)
    return out


def _cell(score: tuple, with_ci: bool) -> str:
    mean, lo, hi = score
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]" if with_ci else f"{mean:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="JSON listing the models as {id, label, corpus, run}.")
    src.add_argument("--runs", nargs="+", help="Run IDs, in row order.")
    ap.add_argument("--labels", nargs="+", default=None, help="Row labels (default: the run ID).")
    ap.add_argument("--runs-dir", default="runs/paper_hard", help="Directory holding the runs.")
    ap.add_argument("--format", choices=["markdown", "latex", "csv"], default="markdown")
    ap.add_argument("--no-ci", action="store_true", help="Skip confidence intervals.")
    ap.add_argument("--no-fonte", action="store_true", help="Ignore the source act when matching.")
    ap.add_argument("--out", default=None, help="Output file (default: stdout).")
    args = ap.parse_args()

    with_ci = not args.no_ci
    include_fonte = not args.no_fonte

    rows_spec: list[dict] = []
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for m in manifest["models"]:
            if not m.get("run"):
                print(f"  ⚠ {m['id']}: no run in the manifest, row skipped", file=sys.stderr)
                continue
            rows_spec.append({"label": m["label"], "corpus": m.get("corpus", ""),
                              "run": m["run"], "runs_dir": m.get("runs_dir", args.runs_dir)})
    else:
        if args.labels and len(args.labels) != len(args.runs):
            ap.error(f"--labels ({len(args.labels)}) must match the length of --runs ({len(args.runs)}).")
        for i, rid in enumerate(args.runs):
            rows_spec.append({"label": args.labels[i] if args.labels else rid,
                              "corpus": "", "run": rid, "runs_dir": args.runs_dir})

    rows = []
    for spec in rows_spec:
        run_dir = ROOT / spec["runs_dir"] / spec["run"]
        if not (run_dir / "details.csv").exists():
            raise SystemExit(f"Run without details.csv: {run_dir}")
        qt, arrays = _run_arrays(run_dir, include_fonte)
        scores = _run_scores(qt, arrays, with_ci)
        rows.append({
            "Model": spec["label"],
            "Passages Corpus": spec["corpus"],
            "Weighted Recall": _cell(scores["recall"], with_ci),
            "Weighted Ranking": _cell(scores["ranking"], with_ci),
            "#q": len(qt),
            "Run": spec["run"],
        })
        print(f"  ✓ {spec['run']} → {spec['label']}  "
              f"R={scores['recall'][0]:.3f} Rank={scores['ranking'][0]:.3f} (n={len(qt)})",
              file=sys.stderr)

    df = pd.DataFrame(rows)
    cols = ["Model", "Passages Corpus", "Weighted Recall", "Weighted Ranking"]

    if args.format == "csv":
        text = df.to_csv(index=False)
    elif args.format == "latex":
        lines = ["\\begin{tabular}{llrr}", "\\hline",
                 " & ".join(cols) + " \\\\", "\\hline"]
        lines += [" & ".join(str(r[c]) for c in cols) + " \\\\" for _, r in df.iterrows()]
        lines += ["\\hline", "\\end{tabular}"]
        text = "\n".join(lines)
    else:
        text = df[cols + ["Run"]].to_markdown(index=False)

    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nsaved: {args.out}", file=sys.stderr)
    else:
        print("\n" + text)


if __name__ == "__main__":
    main()
