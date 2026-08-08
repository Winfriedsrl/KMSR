"""Evaluation for the 3-stage pipeline on wf_ground_truth.csv.

GT schema: query, combo_reati, combo_reati_circostanziati (may be empty).
Each row = one expected result for that query.
Target key per row = combo_reati_circostanziati if present, else combo_reati.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from runner import run_pipeline_model, run_branching_model
from metrics import ndcg, precision, recall


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    result = str(s).strip()
    return "" if result.lower() == "nan" else result


def build_gt_map(df: pd.DataFrame) -> dict[str, dict]:
    gt_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        query = _norm(row["query"])
        combo_reati = _norm(row.get("combo_reati", ""))
        combo_circ = _norm(row.get("combo_reati_circostanziati", ""))
        target_key = combo_circ if combo_circ else combo_reati
        if not target_key:
            continue
        entry = gt_map.setdefault(query, {
            "target_keys": [],
            "base_reati_keys": [],
            "_has_circ": False,
            "_has_base": False,
        })
        entry["target_keys"].append(target_key)
        if combo_reati:
            entry["base_reati_keys"].append(combo_reati)
        if combo_circ:
            entry["_has_circ"] = True
        else:
            entry["_has_base"] = True

    for entry in gt_map.values():
        entry["target_keys"] = list(dict.fromkeys(entry["target_keys"]))
        entry["base_reati_keys"] = list(dict.fromkeys(entry["base_reati_keys"]))
        has_circ = entry.pop("_has_circ")
        has_base = entry.pop("_has_base")
        if has_circ and has_base:
            entry["type"] = "mixed"
        elif has_circ:
            entry["type"] = "reato_circostanziato"
        else:
            entry["type"] = "reato"

    return gt_map


def extract_hit_key(hit: dict) -> str:
    src = hit.get("_source", {})
    circ = src.get("combo_reati_circostanziati", "")
    if circ:
        return _norm(str(circ))
    return _norm(str(src.get("combo_reati", "")))


def _slim_hit(hit: dict, key_field: str) -> dict:
    src = hit.get("_source", {})
    return {
        "id": hit.get("_id", ""),
        "score": hit.get("_score"),
        "index": hit.get("_index", ""),
        "key": _norm(str(src.get(key_field, ""))),
    }


def evaluate_from_df(
    df: pd.DataFrame,
    model_configs: list[dict],
    es_client: Any,
    index_reati: str,
    index_circ: str,
    key_table: dict,
    on_step=None,
    out_csv_path: Path | None = None,
    out_debug_path: Path | None = None,
    out_errors_path: Path | None = None,
    skip_queries: set | None = None,
    status_path: Path | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    gt_map = build_gt_map(df)
    details_rows = []
    debug_data: dict = {}
    skip_queries = skip_queries or set()
    _status_has_errors = False
    active_queries = {q: p for q, p in gt_map.items() if q not in skip_queries}
    total_steps = len(active_queries) * len(model_configs)
    step = 0
    csv_header_written = out_csv_path is not None and Path(out_csv_path).exists()

    from classifier import LLMClassifier, LLMTypeClassifier
    from splitter import QuerySplitter
    classifiers = []
    splitters = []
    for cfg in model_configs:
        if cfg.get("pipeline_type", "pool") == "pool":
            classifiers.append(LLMClassifier(
                llm_model=cfg["llm_model"],
                classifier_version=cfg.get("classifier_version", "v1"),
            ))
        else:
            classifiers.append(LLMTypeClassifier(
                llm_model=cfg["llm_model"],
                classifier_version=cfg.get("only_classifier_version", "v1"),
            ))
        # Splitter solo se un input non è la query intera.
        needs_split = cfg.get("stage1_input", "q") != "q" or cfg.get("stage3_input", "q") != "q"
        if needs_split:
            splitters.append(QuerySplitter(
                method=cfg.get("split_method", "regex"),
                llm_model=cfg.get("llm_model"),
                version=cfg.get("split_version", "v1"),
            ))
        else:
            splitters.append(None)

    for query_idx, (query, payload) in enumerate(active_queries.items(), start=1):
        debug_data[query] = {}
        for model_idx, config in enumerate(model_configs):
            step += 1
            if on_step:
                on_step(step, total_steps, query_idx, model_idx + 1, query)

            if config.get("pipeline_type", "pool") == "pool":
                result = run_pipeline_model(es_client, query, config, index_reati, index_circ, key_table, classifier=classifiers[model_idx], splitter=splitters[model_idx])
            else:
                result = run_branching_model(es_client, query, config, index_reati, index_circ, key_table, type_classifier=classifiers[model_idx], splitter=splitters[model_idx])
            error = result.get("error")
            model_label = f"M{model_idx + 1}"
            hits = result.get("hits", [])[:int(config.get("k", 10))]

            predicted_type = result.get("predicted_type")
            raw_debug = result.get("debug")
            q_type = payload.get("type", "")

            recall_reati = None
            recall_circostanziati = None
            recall_pool = None
            if raw_debug:
                base_reati_keys = payload.get("base_reati_keys", [])
                target_keys_gt = payload.get("target_keys", [])

                stage1_hit_keys = [_norm(str(h.get("_source", {}).get("combo_reati", ""))) for h in raw_debug.get("stage1_hits", [])]
                recall_reati = round(recall(base_reati_keys, stage1_hit_keys), 3)

                if q_type in ("reato_circostanziato", "mixed"):
                    stage3_keys = [
                        _norm(str(h.get("_source", {}).get("combo_reati_circostanziati", "")))
                        for circ_hits in raw_debug.get("stage3_by_reato", {}).values()
                        for h in circ_hits
                    ]
                    if not stage3_keys:
                        # branching_direct: i circ hits sono nel pool, non in stage3_by_reato
                        stage3_keys = [
                            _norm(str(h.get("_source", {}).get("combo_reati_circostanziati", "")))
                            for h in raw_debug.get("pool", [])
                            if h.get("_source", {}).get("combo_reati_circostanziati", "")
                        ]
                    recall_circostanziati = round(recall(target_keys_gt, stage3_keys), 3)

                pool_keys = [extract_hit_key(h) for h in raw_debug.get("pool", [])]
                recall_pool = round(recall(target_keys_gt, pool_keys), 3)

                pool_key_field = "combo_reati" if raw_debug.get("branch") == "reato" else "combo_reati_circostanziati"
                debug_entry = {
                    "stage1_hits": [_slim_hit(h, "combo_reati") for h in raw_debug.get("stage1_hits", [])],
                    "stage3_by_reato": {
                        reato: [_slim_hit(h, "combo_reati_circostanziati") for h in circ_hits]
                        for reato, circ_hits in raw_debug.get("stage3_by_reato", {}).items()
                    },
                    "pool": [_slim_hit(h, pool_key_field) for h in raw_debug.get("pool", [])],
                    "classifier": {
                        "predicted_type": predicted_type,
                        "reasoning": result.get("classifier_reasoning"),
                        "prompt": result.get("classifier_prompt", ""),
                        "answer": result.get("classifier_answer", ""),
                        "parse_error": result.get("parse_error"),
                    },
                    "splitter": result.get("splitter"),
                }
                debug_data[query][model_label] = debug_entry
            else:
                debug_entry = None

            if out_debug_path:
                with open(out_debug_path, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({"query": query, "model": model_label, "data": debug_entry, "error": error}, ensure_ascii=False) + "\n")

            no_hits = not error and not hits
            if (error or no_hits) and out_errors_path:
                with open(out_errors_path, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "query": query,
                        "model": model_label,
                        "error": error or "no_results",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False) + "\n")
                if not _status_has_errors and status_path and status_path.exists():
                    _status_has_errors = True
                    try:
                        s = json.loads(status_path.read_text(encoding="utf-8"))
                        s["status"] = "running_with_errors"
                        status_path.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass

            row_base = {
                "Query": query,
                "Model": model_label,
                "Error": error,
                "Predicted Type": predicted_type,
                "Recall Reati": recall_reati,
                "Recall Circostanziati": recall_circostanziati,
                "Recall Pool": recall_pool,
            }
            if error:
                new_rows = [{**row_base, "Rank": None, "Score": None, "Hit Key": None, "Source Index": None, "Is Circ": None}]
            elif no_hits:
                new_rows = [{**row_base, "Rank": None, "Score": None, "Hit Key": None, "Source Index": None, "Is Circ": None, "Error": "no_results"}]
            else:
                new_rows = []
                for rank, hit in enumerate(hits, start=1):
                    is_circ = bool(_norm(str(hit.get("_source", {}).get("combo_reati_circostanziati", ""))))
                    new_rows.append({
                        **row_base,
                        "Rank": rank,
                        "Score": hit.get("_score"),
                        "Hit Key": extract_hit_key(hit),
                        "Source Index": hit.get("_index", ""),
                        "Is Circ": is_circ,
                    })

            details_rows.extend(new_rows)
            if out_csv_path and new_rows:
                chunk = pd.DataFrame(new_rows)
                chunk.to_csv(out_csv_path, mode="a", header=not csv_header_written, index=False)
                csv_header_written = True

    gt_context = {
        q: {"target_keys": p["target_keys"], "base_reati_keys": p["base_reati_keys"], "type": p["type"]}
        for q, p in gt_map.items()
    }
    return pd.DataFrame(details_rows), gt_context, debug_data


def compute_summary(details_df: pd.DataFrame, gt_context: dict) -> pd.DataFrame:
    rows = []
    for query, qdf in details_df.groupby("Query", sort=False):
        q_str = str(query)
        target_keys = gt_context.get(q_str, {}).get("target_keys", [])
        gt_type = gt_context.get(q_str, {}).get("type")
        model_labels = sorted(qdf["Model"].dropna().unique())
        row: dict[str, Any] = {"Query": q_str}
        for model_label in model_labels:
            mdf = qdf[qdf["Model"] == model_label]
            has_error = mdf["Error"].notna().any()
            hit_keys = [] if has_error else mdf.dropna(subset=["Rank"]).sort_values("Rank")["Hit Key"].tolist()

            row[f"Final Recall {model_label}"] = round(recall(target_keys, hit_keys), 3)
            row[f"Final Precision {model_label}"] = round(precision(target_keys, hit_keys), 3)
            row[f"NDCG {model_label}"] = round(ndcg(target_keys, hit_keys), 3)

            hits_df = mdf.dropna(subset=["Rank"])
            if not hits_df.empty and "Is Circ" in hits_df.columns:
                n_total = len(hits_df)
                n_circ = int(hits_df["Is Circ"].sum())
                n_reati = n_total - n_circ
                p_circ = n_circ / n_total
                p_reati = n_reati / n_total
                row[f"Purity {model_label}"] = round(abs(p_reati - p_circ), 3)
            else:
                row[f"Purity {model_label}"] = None

            for metric_col in ("Recall Reati", "Recall Circostanziati", "Recall Pool"):
                if metric_col in mdf.columns:
                    vals = mdf[metric_col].dropna()
                    row[f"{metric_col} {model_label}"] = round(float(vals.iloc[0]), 3) if len(vals) else None

            if "Predicted Type" in mdf.columns:
                pt = mdf["Predicted Type"].dropna()
                pred_type = str(pt.iloc[0]) if len(pt) else None
                row[f"Predicted Type {model_label}"] = pred_type
                if pred_type and gt_type:
                    row[f"Type Accuracy {model_label}"] = 1.0 if pred_type == gt_type else 0.0

        rows.append(row)
    return pd.DataFrame(rows)


def save_run(
    details_df: pd.DataFrame | None,
    model_configs: list[dict],
    gt_reference: str,
    gt_context: dict,
    debug_data: dict,
    base_dir: str = "runs/evaluations_pipeline",
    run_dir: Path | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    if run_dir is None:
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        run_dir = base_path / now.strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=False, exist_ok=False)

    if details_df is not None:
        details_df.to_csv(run_dir / "details.csv", index=False)
    if not (run_dir / "meta.json").exists():
        (run_dir / "meta.json").write_text(json.dumps({
            "created_at_utc": now.isoformat(),
            "gt_reference": gt_reference,
            "model_configs": model_configs,
        }, ensure_ascii=False, indent=2))
    if not (run_dir / "gt_context.json").exists():
        (run_dir / "gt_context.json").write_text(json.dumps(gt_context, ensure_ascii=False, indent=2))

    jsonl_path = run_dir / "debug.jsonl"
    if not debug_data and jsonl_path.exists():
        reconstructed: dict = {}
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("data") is not None:
                reconstructed.setdefault(entry["query"], {})[entry["model"]] = entry["data"]
        debug_data = reconstructed
    (run_dir / "debug.json").write_text(json.dumps(debug_data, ensure_ascii=False, indent=2))

    errors_path = run_dir / "errors.jsonl"
    if errors_path.exists():
        resolved: set = set()
        if details_df is not None and "Rank" in details_df.columns:
            resolved = set(details_df.dropna(subset=["Rank"])["Query"].dropna().unique())
        else:
            det_path = run_dir / "details.csv"
            if det_path.exists():
                resolved = set(pd.read_csv(det_path).dropna(subset=["Rank"])["Query"].dropna().unique())
        if resolved:
            remaining_errors = []
            for line in errors_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("query") not in resolved:
                    remaining_errors.append(line)
            errors_path.write_text("\n".join(remaining_errors) + ("\n" if remaining_errors else ""), encoding="utf-8")
        has_errors = bool([l for l in errors_path.read_text(encoding="utf-8").splitlines() if l.strip()])
    else:
        has_errors = False

    status_path = run_dir / "status.json"
    final_status = "complete_with_errors" if has_errors else "complete"
    status_path.write_text(json.dumps({"status": final_status}, ensure_ascii=False), encoding="utf-8")

    return str(run_dir)
