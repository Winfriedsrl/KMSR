from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from config import clean_model_config, resolve_effective_vectorizer, resolve_field_profile
from runner import run_model
from metrics import block_ap, block_recall, ndcg, precision, recall


def _norm_str(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def split_periodo_fonte(text: str):
    """Spezza 'periodo, fonte;' in (periodo, fonte). Fonte vuota se assente."""
    value = str(text).strip().rstrip(";")
    parts = value.rsplit(", ", 1)
    periodo = parts[0]
    fonte = parts[1] if len(parts) == 2 else ""
    return periodo, fonte


def normalize_period(text: str) -> str:
    periodo, fonte = split_periodo_fonte(text)
    return f"{_norm_str(periodo)}\x00{_norm_str(fonte)}"


def extract_hit_key(hit: dict[str, Any]) -> str:
    src = hit.get("_source", {})
    periodo = _norm_str(
        src.get("periodo")
        or src.get("rubriche_con_periodi")
        or src.get("testo")
        or src.get("rubrica_testo")
        or ""
    )
    fonte = _norm_str(src.get("fonte_normativa") or "")
    return f"{periodo}\x00{fonte}"


def extract_hit_period(hit: dict[str, Any]):
    src = hit.get("_source", {})
    return (
        src.get("periodo")
        or src.get("rubriche_con_periodi")
        or src.get("testo")
        or src.get("rubrica_testo")
        or ""
    )


def hit_periodo_fonte(hit: dict[str, Any]):
    """Restituisce (periodo, fonte) per le colonne dei dettagli.

    Sull'indice flat del paper periodo e fonte stanno insieme nel campo 'periodo'
    (es. "609-bis comma 1 periodo primo, codice penale;") e vanno spezzati per
    combaciare con la GT; altrove sono in campi separati (comportamento legacy).
    """
    src = hit.get("_source", {})
    if "periodo_content" in src:
        return split_periodo_fonte(src.get("periodo"))
    return extract_hit_period(hit), (src.get("fonte_normativa") or "")


def query_type_label(type_values: list[str]):
    uniq = sorted(set(type_values))
    if len(uniq) == 1:
        return uniq[0]
    return "+".join(uniq)


def load_models_config(path: str):
    ext = Path(path).suffix.lower()
    text = Path(path).read_text()
    if ext == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def build_gt_map(df: pd.DataFrame):
    gt_map: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        query = str(row["query"])
        query_type = str(row["type"]).strip()
        periods = [normalize_period(p) for p in str(row["result"]).split(";") if p.strip()]
        if query not in gt_map:
            gt_map[query] = {
                "periods": [],
                "blocks": [],
                "types": [],
                "raw_results": [],
            }
        gt_map[query]["periods"].extend(periods)
        if periods:
            gt_map[query]["blocks"].append(periods)
        gt_map[query]["types"].append(query_type)
        gt_map[query]["raw_results"].append(str(row["result"]))

    for query in gt_map:
        uniq_periods = list(dict.fromkeys(gt_map[query]["periods"]))
        gt_map[query]["periods"] = uniq_periods
        # Deduplicate blocks ignoring order of elements inside each block.
        dedup_blocks: dict[tuple[str, ...], list[str]] = {}
        for block in gt_map[query]["blocks"]:
            key = tuple(sorted(set(block)))
            dedup_blocks.setdefault(key, list(key))
        gt_map[query]["blocks"] = list(dedup_blocks.values())
        gt_map[query]["query_type"] = query_type_label(gt_map[query]["types"])
    return gt_map


def build_gt_map_paper(df: pd.DataFrame):
    """GT del paper: un periodo per riga (colonne query/reati_estratti).

    Niente blocchi e niente query_type (la GT non li ha): si valuta sul set
    di periodi sfusi con recall/ndcg/precision.
    """
    gt_map: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        query = str(row["query"])
        if query not in gt_map:
            gt_map[query] = {"periods": [], "blocks": [], "query_type": "", "raw_results": []}
        gt_map[query]["periods"].append(normalize_period(row["reati_estratti"]))
        gt_map[query]["raw_results"].append(str(row["reati_estratti"]))

    for query in gt_map:
        gt_map[query]["periods"] = list(dict.fromkeys(gt_map[query]["periods"]))
    return gt_map


def dissolve_combo_to_periods(hits: list[dict[str, Any]]):
    """Scioglie combo ordinate in una lista (periodo, fonte) deduplicata.

    Ogni combo (campo combo_reati_circostanziati o combo_reati) è una lista di
    periodi separati da ';'. Li appiattiamo in ordine di rank, tenendo la prima
    occorrenza, per valutare i modelli pool (M) sulla GT a periodi sfusi.
    """
    seen = set()
    out = []
    for hit in hits:
        src = hit.get("_source", {})
        combo = src.get("combo_reati_circostanziati") or src.get("combo_reati") or ""
        for piece in str(combo).split(";"):
            if not piece.strip():
                continue
            key = normalize_period(piece)
            if key in seen:
                continue
            seen.add(key)
            out.append(split_periodo_fonte(piece))
    return out


def _pool_debug_entry(result: dict[str, Any], j: int) -> dict[str, Any]:
    """Stadi della pipeline pool per una query (hit con key/score/index), per il Paper Explorer."""
    def slim(hit, key_field):
        src = hit.get("_source", {})
        key = src.get(key_field) or src.get("combo_reati_circostanziati") or src.get("combo_reati") or ""
        return {"key": key, "score": hit.get("_score"), "index": hit.get("_index", "")}

    raw = result.get("debug") or {}
    return {
        "stage1_reati": [slim(h, "combo_reati") for h in raw.get("stage1_hits", [])],
        "stage3_by_reato": {
            reato: [slim(hh, "combo_reati_circostanziati") for hh in hits]
            for reato, hits in raw.get("stage3_by_reato", {}).items()
        },
        "pool": [slim(h, "combo_reati_circostanziati") for h in raw.get("pool", [])],
        "selected": [slim(h, "combo_reati_circostanziati") for h in result.get("hits", [])[:j]],
        "splitter": result.get("splitter"),
        "classifier": {
            "predicted_type": result.get("predicted_type"),
            "reasoning": result.get("classifier_reasoning"),
            "prompt": result.get("classifier_prompt"),
            "answer": result.get("classifier_answer"),
            "parse_error": result.get("parse_error"),
        },
    }


def load_flat_content_map(es_client, flat_index):
    """Mappa chiave_periodo -> periodo_content da paper_periodi_flat (per il period-rerank)."""
    from elasticsearch.helpers import scan
    out = {}
    for h in scan(es_client.es, index=flat_index, _source=["periodo", "periodo_content"], size=1000):
        src = h.get("_source", {})
        out[normalize_period(src.get("periodo", ""))] = src.get("periodo_content", "")
    return out


def evaluate_pool_paper(df, model_configs, es_client, index_reati, index_circ, key_table,
                        on_step=None, gt_map=None, out_csv_path=None, out_debug_path=None,
                        skip_queries=None, flat_index=None):
    """Valuta i modelli pool (M) sulla GT del paper.

    Esegue la pipeline pool, scioglie le combo in periodi e produce lo stesso
    schema di details di evaluate_from_df, così compute_summary_from_details e il
    Paper Explorer trattano B e M allo stesso modo (period-level). Ritorna anche
    debug_data[query][model] con gli stadi della pipeline.

    Variante B (config["period_rerank"]=True): dopo lo scioglimento, riordina i
    periodi con un LLMReranker (content preso da flat_index) prima del taglio a k.

    out_csv_path / out_debug_path : streaming di details e debug (jsonl) per resume.
    skip_queries                  : query già fatte da saltare (resume).
    flat_index                    : indice flat per recuperare il content (period-rerank).
    """
    from runner import run_pipeline_model
    from classifier import LLMClassifier
    from splitter import QuerySplitter
    from reranker import LLMReranker

    if gt_map is None:
        gt_map = build_gt_map_paper(df)
    skip_queries = skip_queries or set()
    active = {q: p for q, p in gt_map.items() if q not in skip_queries}

    classifiers, splitters, rerankers = [], [], []
    for cfg in model_configs:
        classifiers.append(LLMClassifier(
            llm_model=cfg["llm_model"],
            classifier_version=cfg.get("classifier_version", "v1"),
        ))
        needs_split = cfg.get("stage1_input", "q") != "q" or cfg.get("stage3_input", "q") != "q"
        splitters.append(QuerySplitter(
            method=cfg.get("split_method", "regex"),
            llm_model=cfg.get("llm_model"),
            version=cfg.get("split_version", "v1"),
        ) if needs_split else None)
        rerankers.append(LLMReranker(cfg.get("rerank_llm") or cfg["llm_model"]) if cfg.get("period_rerank") else None)

    # Mappa content del flat, caricata una volta sola se serve il period-rerank.
    flat_map = {}
    if flat_index and any(cfg.get("period_rerank") for cfg in model_configs):
        flat_map = load_flat_content_map(es_client, flat_index)

    details_by_query: dict[str, dict[str, Any]] = {}
    debug_data: dict[str, dict[str, Any]] = {}
    details_rows = []
    csv_header_written = out_csv_path is not None and Path(out_csv_path).exists()
    total_steps = len(active) * len(model_configs)
    step = 0

    for query_idx, (query, payload) in enumerate(active.items(), start=1):
        debug_data[query] = {}
        query_rows = []
        for model_idx, config in enumerate(model_configs):
            # j = combo da recuperare; k = periodi finali dopo lo scioglimento.
            j = int(config.get("j", config.get("k", 10)))
            k_periods = int(config.get("k", 10))
            pipe_config = {**config, "k": j}  # la pipeline interpreta k come numero di combo
            result = run_pipeline_model(
                es_client, query, pipe_config, index_reati, index_circ, key_table,
                classifier=classifiers[model_idx], splitter=splitters[model_idx],
            )
            step += 1
            if on_step is not None:
                on_step(step, total_steps, query_idx, model_idx + 1)
            model_label = f"M{model_idx + 1}"
            error = result.get("error")
            entry = _pool_debug_entry(result, j)
            debug_data[query][model_label] = entry
            if out_debug_path:
                with open(out_debug_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"query": query, "model": model_label, "data": entry}, ensure_ascii=False) + "\n")

            if error:
                query_rows.append({
                    "Query": query, "Query Type": payload["query_type"], "Model": model_label,
                    "Rank": None, "Periodo": None, "Fonte Normativa": None, "Score": None,
                    "Error": error,
                })
            else:
                all_periods = dissolve_combo_to_periods(result.get("hits", []))  # ordine combo, dedup
                reranker = rerankers[model_idx]
                if reranker is not None and all_periods:
                    # Variante B: riordina i periodi sciolti col reranker, poi taglia a k.
                    period_hits = [
                        {"_source": {"periodo": p, "fonte_normativa": f,
                                     "periodo_content": flat_map.get(normalize_period(f"{p}, {f}"), "")}}
                        for (p, f) in all_periods
                    ]
                    ranked = reranker.rerank(query, period_hits, rr=len(period_hits), k=k_periods)["ranked_hits"]
                    periods = [(h["_source"]["periodo"], h["_source"]["fonte_normativa"]) for h in ranked]
                    if len(periods) < k_periods:
                        # il reranker ne ha resi < k: completo con i periodi rimanenti (ordine combo).
                        have = {normalize_period(f"{p}, {f}") for (p, f) in periods}
                        for (p, f) in all_periods:
                            if len(periods) >= k_periods:
                                break
                            if normalize_period(f"{p}, {f}") not in have:
                                periods.append((p, f))
                else:
                    # Variante A: ordine dei combo, taglio a k.
                    periods = all_periods[:k_periods]
                if not periods:
                    # Nessun periodo: riga "no_results" così conta come recall 0 (non sparisce).
                    query_rows.append({
                        "Query": query, "Query Type": payload["query_type"], "Model": model_label,
                        "Rank": None, "Periodo": None, "Fonte Normativa": None, "Score": None,
                        "Error": "no_results",
                    })
                for rank, (periodo, fonte) in enumerate(periods, start=1):
                    query_rows.append({
                        "Query": query, "Query Type": payload["query_type"], "Model": model_label,
                        "Rank": rank, "Periodo": periodo, "Fonte Normativa": fonte, "Score": None,
                        "Error": None,
                    })

        details_by_query[query] = {
            "query_type": payload["query_type"],
            "gt_periods": payload["periods"],
            "gt_blocks": payload["blocks"],
            "raw_results": payload["raw_results"],
        }
        details_rows.extend(query_rows)
        if out_csv_path and query_rows:
            pd.DataFrame(query_rows).to_csv(out_csv_path, mode="a", header=not csv_header_written, index=False)
            csv_header_written = True

    return pd.DataFrame(details_rows), details_by_query, debug_data


def run_model_for_query(query: str, config: dict[str, Any], es_client: Any, index_names: dict[str, str]):
    match_field = config["match_field"]
    field_profile = resolve_field_profile(match_field, index_names)
    # index cablato nel config (es. baseline su un corpus diverso) -> override dell'indice
    # risolto dal profilo; campo/analyzer/embedder restano quelli di match_field.
    index_name = config.get("index") or field_profile["index_name"]
    source_fields = field_profile["source_fields"]
    plot_text_field = field_profile["plot_text_field"]

    run_config: dict[str, Any] = dict(config)
    run_config["vectorizer"] = resolve_effective_vectorizer(
        run_config["model_type"],
        run_config["match_field"],
        run_config.get("vectorizer"),
    )

    result = run_model(
        es_client,
        index_name,
        query,
        run_config,
        source_fields=source_fields,
        plot_text_field=plot_text_field,
        compute_plot_vectors=False,
    )
    # Se il reranker è attivo, usiamo ranked_hits; altrimenti hits.
    hits = result.get("ranked_hits") or result.get("hits", [])
    hits = hits[: int(config["k"])]
    return result, hits


def evaluate_from_df(
    df: pd.DataFrame,
    model_configs: list[dict[str, Any]],
    es_client: Any,
    index_names: dict[str, str],
    on_step=None,
    gt_map=None,
    out_csv_path=None,
    skip_queries=None,
):
    """
    Interrogates ES for each (query, model) pair and returns raw hits.
    Metrics are not computed here — use compute_summary_from_details in the Explorer.

    on_step      : callable(step, total, q_idx, m_idx) | None
    gt_map       : se passato, salta build_gt_map (usato dal paper con build_gt_map_paper)
    out_csv_path : se passato, scrive i details in streaming (append per query)
    skip_queries : query già fatte da saltare (resume)
    """
    if gt_map is None:
        gt_map = build_gt_map(df)
    skip_queries = skip_queries or set()
    active = {q: p for q, p in gt_map.items() if q not in skip_queries}
    details_by_query: dict[str, dict[str, Any]] = {}
    details_rows = []
    csv_header_written = out_csv_path is not None and Path(out_csv_path).exists()

    total_steps = len(active) * len(model_configs)
    step = 0

    for query_idx, (query, payload) in enumerate(active.items(), start=1):
        query_type = payload["query_type"]
        query_rows = []

        for model_idx, config in enumerate(model_configs):
            result, hits = run_model_for_query(query, config, es_client, index_names)
            step += 1
            if on_step is not None:
                on_step(step, total_steps, query_idx, model_idx + 1)
            error = result.get("error")
            generation_error = result.get("generation_error")
            model_label = f"M{model_idx + 1}"

            if error:
                query_rows.append({
                    "Query": query, "Query Type": query_type, "Model": model_label,
                    "Rank": None, "Periodo": None, "Fonte Normativa": None, "Score": None,
                    "Error": error, "Generation Error": generation_error,
                })
            elif not hits:
                # Query senza hit: riga "no_results" così conta come recall 0 (non sparisce).
                query_rows.append({
                    "Query": query, "Query Type": query_type, "Model": model_label,
                    "Rank": None, "Periodo": None, "Fonte Normativa": None, "Score": None,
                    "Error": "no_results", "Generation Error": generation_error,
                })
            else:
                for rank, hit in enumerate(hits, start=1):
                    periodo, fonte = hit_periodo_fonte(hit)
                    query_rows.append({
                        "Query": query, "Query Type": query_type, "Model": model_label,
                        "Rank": rank, "Periodo": periodo, "Fonte Normativa": fonte,
                        "Score": hit.get("_score"), "Error": None, "Generation Error": generation_error,
                    })

        details_by_query[query] = {
            "query_type": query_type,
            "gt_periods": payload["periods"],
            "gt_blocks": payload["blocks"],
            "raw_results": payload["raw_results"],
        }
        details_rows.extend(query_rows)
        if out_csv_path and query_rows:
            pd.DataFrame(query_rows).to_csv(out_csv_path, mode="a", header=not csv_header_written, index=False)
            csv_header_written = True

    return pd.DataFrame(details_rows), details_by_query


def compute_summary_from_details(
    details_df: pd.DataFrame,
    gt_context: dict[str, Any],
    include_fonte: bool = True,
) -> pd.DataFrame:
    """Compute per-query recall metrics from raw details + GT context."""
    summary_rows = []

    for query, query_df in details_df.groupby("Query", sort=False):
        query_str = str(query)
        gt = gt_context.get(query_str, {})
        raw_gt_periods = gt.get("gt_periods", [])
        raw_gt_blocks = gt.get("gt_blocks", [])
        query_type = gt.get("query_type", "")
        if not query_type and "Query Type" in query_df.columns:
            query_type = str(query_df["Query Type"].iloc[0])

        if include_fonte:
            gt_periods = raw_gt_periods
            gt_blocks = raw_gt_blocks
        else:
            gt_periods = [p.split("\x00", 1)[0] for p in raw_gt_periods]
            gt_blocks = [[p.split("\x00", 1)[0] for p in block] for block in raw_gt_blocks]

        model_labels = sorted(
            query_df["Model"].dropna().unique(),
            key=lambda x: int(x[1:]) if str(x)[1:].isdigit() else 0,
        )
        flat_recalls: dict[str, float] = {}
        precisions: dict[str, float] = {}
        block_recalls: dict[str, float] = {}
        ndcgs: dict[str, float] = {}
        block_aps: dict[str, float] = {}

        for model_label in model_labels:
            model_df = query_df[query_df["Model"] == model_label]
            has_error = model_df["Error"].notna().any() if "Error" in model_df.columns else False
            if has_error:
                flat_recalls[model_label] = 0.0
                precisions[model_label] = 0.0
                block_recalls[model_label] = 0.0
                ndcgs[model_label] = 0.0
                block_aps[model_label] = 0.0
            else:
                hit_rows = model_df.dropna(subset=["Rank"]).sort_values("Rank")
                predicted_keys = []
                for _, row in hit_rows.iterrows():
                    periodo = _norm_str(str(row["Periodo"]) if pd.notna(row.get("Periodo")) else "")
                    if include_fonte:
                        fonte = _norm_str(str(row["Fonte Normativa"]) if pd.notna(row.get("Fonte Normativa")) else "")
                        predicted_keys.append(f"{periodo}\x00{fonte}")
                    else:
                        predicted_keys.append(periodo)
                flat_recalls[model_label] = recall(gt_periods, predicted_keys)
                precisions[model_label] = precision(gt_periods, predicted_keys)
                block_recalls[model_label] = block_recall(gt_blocks, predicted_keys)
                ndcgs[model_label] = ndcg(gt_periods, predicted_keys)
                block_aps[model_label] = block_ap(gt_blocks, predicted_keys)

        row: dict[str, Any] = {"Query": query_str, "Query Type": query_type}
        for label in model_labels:
            row[f"Recall {label}"] = round(flat_recalls[label], 3)
            row[f"Precision {label}"] = round(precisions[label], 3)
            row[f"Block Recall {label}"] = round(block_recalls[label], 3)
            row[f"NDCG {label}"] = round(ndcgs[label], 3)
            row[f"Block AP {label}"] = round(block_aps[label], 3)
        flat_vals = list(flat_recalls.values())
        prec_vals = list(precisions.values())
        block_vals = list(block_recalls.values())
        ndcg_vals = list(ndcgs.values())
        block_ap_vals = list(block_aps.values())
        row["AVG Recall"] = round(sum(flat_vals) / len(flat_vals), 3) if flat_vals else 0.0
        row["AVG Precision"] = round(sum(prec_vals) / len(prec_vals), 3) if prec_vals else 0.0
        row["AVG Block Recall"] = round(sum(block_vals) / len(block_vals), 3) if block_vals else 0.0
        row["AVG NDCG"] = round(sum(ndcg_vals) / len(ndcg_vals), 3) if ndcg_vals else 0.0
        row["AVG Block AP"] = round(sum(block_ap_vals) / len(block_ap_vals), 3) if block_ap_vals else 0.0
        row["Max Recall"] = round(max(flat_vals), 3) if flat_vals else 0.0
        row["Max Block Recall"] = round(max(block_vals), 3) if block_vals else 0.0
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def evaluate_from_csv(
    gt_csv_path: str,
    model_configs: list[dict[str, Any]],
    es_client: Any,
    index_names: dict[str, str],
    on_step=None,
):
    df = pd.read_csv(gt_csv_path)
    return evaluate_from_df(df, model_configs, es_client, index_names, on_step=on_step)


def get_recall_columns(summary_df: pd.DataFrame):
    return [col for col in summary_df.columns if col.startswith("Recall M")]


def get_block_recall_columns(summary_df: pd.DataFrame):
    return [col for col in summary_df.columns if col.startswith("Block Recall M")]


def get_ndcg_columns(summary_df: pd.DataFrame):
    return [col for col in summary_df.columns if col.startswith("NDCG M")]


def get_block_ap_columns(summary_df: pd.DataFrame):
    return [col for col in summary_df.columns if col.startswith("Block AP M")]


def build_ranking_overview_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Macro e Weighted Score calcolati su NDCG (Tipo A) e Block AP (Tipo B)."""
    ndcg_cols = get_ndcg_columns(summary_df)
    if not ndcg_cols:
        return pd.DataFrame()

    model_labels = [col.replace("NDCG ", "") for col in ndcg_cols]
    rows = []
    for model_label in model_labels:
        scores_by_type: dict[str, float] = {}
        counts_by_type: dict[str, int] = {}
        for qtype in ["A", "B", "A+B"]:
            qtype_df = summary_df[summary_df["Query Type"] == qtype]
            n = len(qtype_df)
            counts_by_type[qtype] = n
            ndcg_col = f"NDCG {model_label}"
            block_ap_col = f"Block AP {model_label}"
            ndcg_avg = float(qtype_df[ndcg_col].mean()) if (n and ndcg_col in qtype_df.columns) else 0.0
            block_ap_avg = float(qtype_df[block_ap_col].mean()) if (n and block_ap_col in qtype_df.columns) else 0.0
            if qtype == "A":
                scores_by_type[qtype] = ndcg_avg
            elif qtype == "B":
                scores_by_type[qtype] = block_ap_avg
            else:
                scores_by_type[qtype] = (ndcg_avg + block_ap_avg) / 2.0

        parts = list(scores_by_type.values())
        macro = sum(parts) / len(parts) if parts else 0.0
        total = sum(counts_by_type.values())
        weighted = sum(scores_by_type[qt] * counts_by_type[qt] / total for qt in scores_by_type) if total > 0 else 0.0
        rows.append({
            "Model": model_label,
            "Ranking Macro": round(macro, 3),
            "Ranking Weighted": round(weighted, 3),
        })
    return pd.DataFrame(rows)


def build_ranking_scoring_heatmap_df(summary_df: pd.DataFrame):
    """Heatmap dei ranking score: NDCG per Tipo A, Block AP per Tipo B."""
    ndcg_cols = get_ndcg_columns(summary_df)
    if not ndcg_cols:
        return pd.DataFrame(), pd.DataFrame()

    model_labels = [col.replace("NDCG ", "") for col in ndcg_cols]
    score_rows: list[dict[str, Any]] = []
    n_by_qtype: dict[str, int] = {}

    for qtype in ["A", "B", "A+B"]:
        qtype_df = summary_df[summary_df["Query Type"] == qtype]
        n_query = int(len(qtype_df))
        n_by_qtype[qtype] = n_query
        row: dict[str, Any] = {"Query Type": qtype}
        for model_label in model_labels:
            ndcg_col = f"NDCG {model_label}"
            block_ap_col = f"Block AP {model_label}"
            ndcg_avg = float(qtype_df[ndcg_col].mean()) if (n_query and ndcg_col in qtype_df.columns) else 0.0
            block_ap_avg = float(qtype_df[block_ap_col].mean()) if (n_query and block_ap_col in qtype_df.columns) else 0.0
            if qtype == "A":
                score = ndcg_avg
            elif qtype == "B":
                score = block_ap_avg
            else:
                score = (ndcg_avg + block_ap_avg) / 2.0
            row[model_label] = round(score, 3)
        score_rows.append(row)

    score_df = pd.DataFrame(score_rows).set_index("Query Type")
    n_rows = [{"Query Type": qt, **{m: n_by_qtype.get(qt, 0) for m in score_df.columns}} for qt in score_df.index]
    n_df = pd.DataFrame(n_rows).set_index("Query Type")
    return score_df, n_df


def build_query_model_heatmap_df(summary_df: pd.DataFrame, query_type: str | None = None):
    recall_cols = get_recall_columns(summary_df)
    if not recall_cols:
        return pd.DataFrame()

    filtered_df = summary_df
    if query_type is not None:
        filtered_df = summary_df[summary_df["Query Type"] == query_type]

    if filtered_df.empty:
        return pd.DataFrame()

    return filtered_df.set_index("Query")[recall_cols]


def build_query_type_heatmap_df(summary_df: pd.DataFrame):
    recall_cols = get_recall_columns(summary_df)
    if not recall_cols:
        return pd.DataFrame()
    return summary_df.groupby("Query Type", as_index=True)[recall_cols].mean().round(3)


def build_model_type_overview_df(summary_df: pd.DataFrame):
    recall_cols = get_recall_columns(summary_df)
    block_cols = get_block_recall_columns(summary_df)
    if not recall_cols:
        return pd.DataFrame()
    model_labels = [col.replace("Recall ", "") for col in recall_cols]
    grouped_rows = []
    for model_label in model_labels:
        flat_col = f"Recall {model_label}"
        block_col = f"Block Recall {model_label}"
        cols = ["Query Type", flat_col]
        if block_col in block_cols:
            cols.append(block_col)
        tmp = summary_df[cols].copy()
        tmp = tmp.rename(columns={flat_col: "Flat Recall"})
        if block_col in tmp.columns:
            tmp = tmp.rename(columns={block_col: "Block Recall"})
        else:
            tmp["Block Recall"] = 0.0

        grouped = (
            tmp.groupby("Query Type", as_index=False)
            .agg(
                n_query=("Flat Recall", "size"),
                mean_flat_recall=("Flat Recall", "mean"),
                count_flat_recall_eq_0=("Flat Recall", lambda s: int((s == 0).sum())),
                mean_block_recall=("Block Recall", "mean"),
                count_block_recall_eq_0=("Block Recall", lambda s: int((s == 0).sum())),
            )
        )
        grouped["Model"] = model_label
        grouped_rows.append(grouped)

    grouped = pd.concat(grouped_rows, ignore_index=True)
    grouped["mean_flat_recall"] = grouped["mean_flat_recall"].round(3)
    grouped["mean_block_recall"] = grouped["mean_block_recall"].round(3)
    grouped = grouped.rename(
        columns={
            "n_query": "N Query",
            "mean_flat_recall": "AVG Recall",
            "count_flat_recall_eq_0": "Count Recall = 0",
            "mean_block_recall": "AVG Block Recall",
            "count_block_recall_eq_0": "Count Block Recall = 0",
        }
    )

    wide_df = grouped.pivot(index="Model", columns="Query Type")
    wide_df.columns = [f"{qtype} {metric}" for metric, qtype in wide_df.columns]
    wide_df = wide_df.reset_index()

    metric_order = [
        "N Query",
        "AVG Recall",
        "Count Recall = 0",
        "AVG Block Recall",
        "Count Block Recall = 0",
    ]
    ordered_cols = ["Model"]
    query_types = sorted(grouped["Query Type"].unique())
    for qtype in query_types:
        for metric in metric_order:
            if qtype == "A" and metric.startswith(("Mean Block", "Count Block")):
                continue
            col = f"{qtype} {metric}"
            if col in wide_df.columns:
                ordered_cols.append(col)
    trailing_cols = [
        c
        for c in wide_df.columns
        if c not in ordered_cols
        and not (
            c.startswith("A AVG Block Recall")
            or c.startswith("A Count Block Recall = 0")
        )
    ]
    return wide_df[ordered_cols + trailing_cols]


def build_simple_qtype_tables(summary_df: pd.DataFrame):
    recall_cols = get_recall_columns(summary_df)
    if not recall_cols:
        return {}, pd.DataFrame()

    model_labels = [col.replace("Recall ", "") for col in recall_cols]
    qtype_tables: dict[str, pd.DataFrame] = {}
    overview_acc: dict[str, dict[str, Any]] = {
        model_label: {
            "macro_parts": [],
            "query_count_total": 0,
        }
        for model_label in model_labels
    }

    for qtype in ["A", "B", "A+B"]:
        qtype_df = summary_df[summary_df["Query Type"] == qtype]
        rows = []
        for model_label in model_labels:
            flat_col = f"Recall {model_label}"
            block_col = f"Block Recall {model_label}"
            ndcg_col = f"NDCG {model_label}"
            block_ap_col = f"Block AP {model_label}"
            if flat_col not in qtype_df.columns:
                continue

            n_query = int(len(qtype_df))
            flat_values = qtype_df[flat_col] if n_query else pd.Series(dtype=float)
            flat_avg = float(flat_values.mean()) if n_query else 0.0
            flat_zero_count = int((flat_values == 0).sum()) if n_query else 0

            ndcg_values = qtype_df[ndcg_col] if (n_query and ndcg_col in qtype_df.columns) else pd.Series([0.0] * n_query, dtype=float)
            ndcg_avg = float(ndcg_values.mean()) if n_query else 0.0

            block_values = qtype_df[block_col] if (n_query and block_col in qtype_df.columns) else pd.Series([0.0] * n_query, dtype=float)
            block_avg = float(block_values.mean()) if n_query else 0.0
            block_zero_count = int((block_values == 0).sum()) if n_query else 0

            block_ap_values = qtype_df[block_ap_col] if (n_query and block_ap_col in qtype_df.columns) else pd.Series([0.0] * n_query, dtype=float)
            block_ap_avg = float(block_ap_values.mean()) if n_query else 0.0

            if qtype == "A":
                rows.append(
                    {
                        "Model": model_label,
                        "Flat Recall AVG": round(flat_avg, 3),
                        "NDCG AVG": round(ndcg_avg, 3),
                        "#Query": n_query,
                        "#Query Recall=0": flat_zero_count,
                    }
                )
                qtype_metric = flat_avg
            elif qtype == "B":
                rows.append(
                    {
                        "Model": model_label,
                        "Block Recall AVG": round(block_avg, 3),
                        "Block AP AVG": round(block_ap_avg, 3),
                        "#Query": n_query,
                        "#Query Block Recall=0": block_zero_count,
                    }
                )
                qtype_metric = block_avg
            else:
                rows.append(
                    {
                        "Model": model_label,
                        "Flat Recall AVG": round(flat_avg, 3),
                        "NDCG AVG": round(ndcg_avg, 3),
                        "Block Recall AVG": round(block_avg, 3),
                        "Block AP AVG": round(block_ap_avg, 3),
                        "#Query": n_query,
                        "#Query Recall=0": flat_zero_count,
                        "#Query Block Recall=0": block_zero_count,
                    }
                )
                qtype_metric = (flat_avg + block_avg) / 2.0

            overview_acc[model_label]["macro_parts"].append(qtype_metric)
            overview_acc[model_label]["query_count_total"] += n_query

        qtype_tables[qtype] = pd.DataFrame(rows)

    overview_rows: list[dict[str, Any]] = []
    for model_label in model_labels:
        acc = overview_acc[model_label]
        macro_parts = acc["macro_parts"]
        macro_score = sum(macro_parts) / len(macro_parts) if macro_parts else 0.0

        total_queries = float(acc["query_count_total"])
        if total_queries > 0:
            w_a = sum(1 for _ in summary_df[summary_df["Query Type"] == "A"].index) / total_queries
            w_b = sum(1 for _ in summary_df[summary_df["Query Type"] == "B"].index) / total_queries
            w_ab = sum(1 for _ in summary_df[summary_df["Query Type"] == "A+B"].index) / total_queries
            # Weighted score with wa/wb/wa+b based on query-type cardinality.
            s_a = acc["macro_parts"][0] if len(acc["macro_parts"]) > 0 else 0.0
            s_b = acc["macro_parts"][1] if len(acc["macro_parts"]) > 1 else 0.0
            s_ab = acc["macro_parts"][2] if len(acc["macro_parts"]) > 2 else 0.0
            weighted_score = (w_a * s_a) + (w_b * s_b) + (w_ab * s_ab)
        else:
            weighted_score = 0.0

        overview_rows.append(
            {
                "Model": model_label,
                "Macro Score": round(float(macro_score), 3),
                "Weighted Score": round(float(weighted_score), 3),
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    if not overview_df.empty:
        overview_df = overview_df[["Model", "Macro Score", "Weighted Score"]]

    return qtype_tables, overview_df


def build_overview_mean_recall_heatmap_df(overview_df: pd.DataFrame):
    if overview_df.empty or "Model" not in overview_df.columns:
        return pd.DataFrame()

    mean_cols = [col for col in overview_df.columns if col.endswith(" AVG Recall")]
    if not mean_cols:
        return pd.DataFrame()

    heatmap_df = overview_df[["Model"] + mean_cols].copy()
    rename_map = {}
    for col in mean_cols:
        qtype = col[: -len(" AVG Recall")].strip()
        rename_map[col] = qtype
    heatmap_df = heatmap_df.rename(columns=rename_map)
    return heatmap_df.set_index("Model")


def build_scoring_heatmap_df(summary_df: pd.DataFrame):
    recall_cols = get_recall_columns(summary_df)
    if not recall_cols:
        return pd.DataFrame(), pd.DataFrame()

    model_labels = [col.replace("Recall ", "") for col in recall_cols]
    score_rows: list[dict[str, Any]] = []
    n_by_qtype: dict[str, int] = {}

    for qtype in ["A", "B", "A+B"]:
        qtype_df = summary_df[summary_df["Query Type"] == qtype]
        n_query = int(len(qtype_df))
        n_by_qtype[qtype] = n_query
        row: dict[str, Any] = {"Query Type": qtype}

        for model_label in model_labels:
            flat_col = f"Recall {model_label}"
            block_col = f"Block Recall {model_label}"
            if flat_col not in qtype_df.columns:
                continue

            flat_avg = float(qtype_df[flat_col].mean()) if n_query else 0.0
            block_avg = (
                float(qtype_df[block_col].mean())
                if (n_query and block_col in qtype_df.columns)
                else 0.0
            )

            if qtype == "A":
                score = flat_avg
            elif qtype == "B":
                score = block_avg
            else:
                score = (flat_avg + block_avg) / 2.0
            row[model_label] = round(score, 3)

        score_rows.append(row)

    score_df = pd.DataFrame(score_rows)
    if score_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    score_df = score_df.set_index("Query Type")

    n_rows: list[dict[str, Any]] = []
    for qtype in score_df.index.tolist():
        n_row: dict[str, Any] = {"Query Type": qtype}
        for model_label in score_df.columns.tolist():
            n_row[model_label] = n_by_qtype.get(str(qtype), 0)
        n_rows.append(n_row)
    n_df = pd.DataFrame(n_rows).set_index("Query Type")

    return score_df, n_df


def build_gt_context(details_by_query: dict[str, Any]) -> dict[str, Any]:
    return {
        query: {
            "query_type": details["query_type"],
            "gt_periods": details["gt_periods"],
            "gt_blocks": details["gt_blocks"],
            "raw_results": details["raw_results"],
        }
        for query, details in details_by_query.items()
    }


def save_evaluation_run(
    details_df: pd.DataFrame,
    model_configs: list[dict[str, Any]],
    gt_reference: str,
    base_dir: str = "runs/evaluations",
    gt_context: dict[str, Any] | None = None,
):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    run_name = now.strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = base_path / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    details_path = run_dir / "details.csv"
    meta_path = run_dir / "meta.json"

    details_df.to_csv(details_path, index=False)

    meta = {
        "created_at_utc": now.isoformat(),
        "gt_reference": gt_reference,
        "model_configs": [clean_model_config(c) for c in model_configs],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    result: dict[str, str] = {
        "run_dir": str(run_dir),
        "details_csv": str(details_path),
        "meta_json": str(meta_path),
    }

    if gt_context is not None:
        gt_context_path = run_dir / "gt_context.json"
        gt_context_path.write_text(json.dumps(gt_context, ensure_ascii=False, indent=2))
        result["gt_context_json"] = str(gt_context_path)

    return result
