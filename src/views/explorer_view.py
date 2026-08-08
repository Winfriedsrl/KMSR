import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from evaluation import (
    build_scoring_heatmap_df,
    build_ranking_scoring_heatmap_df,
    build_ranking_overview_df,
    build_simple_qtype_tables,
    compute_summary_from_details,
    extract_hit_key,
)
from metrics import block_ap, block_recall, ndcg, recall
from views.explorer_components import (
    _preview_heatmap_height,
    _render_results_by_query_type,
)

try:
    import plotly.express as px
except Exception:
    px = None

RUNS_DIR = Path("runs/evaluations")

# Bump this when the metrics computation changes so @st.cache_data caches are automatically busted.
_METRICS_VERSION = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)


def _load_meta(run_dir: Path) -> dict:
    path = run_dir / "meta.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _load_run_summary(run_dir: Path, include_fonte: bool = True, metrics_version: int = _METRICS_VERSION) -> pd.DataFrame:
    details_path = run_dir / "details.csv"
    gt_context_path = run_dir / "gt_context.json"
    if not details_path.exists():
        return pd.DataFrame()
    details_df = pd.read_csv(details_path, encoding="utf-8")
    gt_context: dict = {}
    if gt_context_path.exists():
        with open(gt_context_path, encoding="utf-8") as f:
            gt_context = json.load(f)
    return compute_summary_from_details(details_df, gt_context, include_fonte=include_fonte)


def _fmt_ts(ts: str, fmt: str = "%Y-%m-%d %H:%M") -> str:
    try:
        return datetime.fromisoformat(ts).strftime(fmt)
    except Exception:
        return ts


def _model_short_label(config: dict) -> str:
    parts = [config.get("model_type", "Unknown")]
    if vectorizer := config.get("vectorizer"):
        parts.append(vectorizer)
    if reranker := config.get("reranker_config"):
        parts.append(f"+Reranker({reranker.get('llm_model', '')})")
    return " · ".join(parts)


def _single_retriever_type(r: dict) -> str:
    kind = r.get("kind") or r.get("model_type", "?")
    if kind == "Dense search":
        return f"Dense/{r.get('vectorizer', '?')}"
    if kind == "Sparse search":
        return "Sparse"
    if kind == "Sparse search (weighted)":
        return "Sparse(w)"
    return kind


def _single_retriever_field(r: dict) -> str:
    kind = r.get("kind") or r.get("model_type", "?")
    if kind in ("Dense search", "Sparse search"):
        return r.get("match_field", "?")
    if kind == "Sparse search (weighted)":
        fields = r.get("fields", [])
        weights = r.get("weights", [])
        return " + ".join(f"{f}×{w}" for f, w in zip(fields, weights)) if fields else "?"
    return "?"


def _model_columns(config: dict) -> dict:
    mtype = config.get("model_type", "Unknown")
    reranker = config.get("reranker_config")
    k_ret = config.get("k", "?")

    if mtype == "Fusion (RRF)":
        rrf = config.get("rrf_config", {})
        ra = rrf.get("retriever_a", {})
        rb = rrf.get("retriever_b", {})
        tipo = "Hybrid"
        ret1 = _single_retriever_type(ra)
        campo1 = _single_retriever_field(ra)
        ret2 = f"{_single_retriever_type(rb)} · rrf_k={rrf.get('rrf_k', '?')}"
        campo2 = _single_retriever_field(rb)
    else:
        tipo = "Single"
        ret1 = _single_retriever_type(config)
        campo1 = _single_retriever_field(config)
        ret2 = "—"
        campo2 = "—"

    if reranker:
        llm = reranker.get("llm_model", "?")
        rr = reranker.get("reranker_results", "?")
        reranker_desc = f"{llm} · rr={rr}"
        k_reranker = reranker.get("k", "?")
    else:
        reranker_desc = "—"
        k_reranker = "—"

    return {
        "Tipo": tipo,
        "Ret1": ret1,
        "Campo1": campo1,
        "Ret2": ret2,
        "Campo2": campo2,
        "k": k_ret,
        "Reranker": reranker_desc,
        "k_rr": k_reranker,
    }


def _model_config_description(config: dict) -> str:
    cols = _model_columns(config)
    if cols["Tipo"] == "Hybrid":
        ret = f"Fusion(RRF) [{cols['Ret1']}:{cols['Campo1']} + {cols['Ret2']}:{cols['Campo2']}]"
    else:
        ret = f"{cols['Ret1']} · {cols['Campo1']}"
    base = f"{ret} · k={cols['k']}"
    if cols["Reranker"] != "—":
        base += f" + rerank({cols['Reranker']} · k={cols['k_rr']})"
    return base


def _run_short_label(run_dir: Path, meta: dict) -> str:
    ts = meta.get("created_at_utc", "")
    parts = run_dir.name.split("_")
    micro = parts[-1] if len(parts) >= 4 else ""
    time_label = _fmt_ts(ts, "%Y-%m-%d %H:%M:%S") if ts else run_dir.name
    return f"{time_label}.{micro}" if micro else time_label


@st.cache_data
def _load_details_by_query(run_dir: Path, metrics_version: int = _METRICS_VERSION) -> dict[str, Any]:
    details_path = run_dir / "details.csv"
    gt_context_path = run_dir / "gt_context.json"
    if not details_path.exists():
        return {}

    details_df = pd.read_csv(details_path, encoding="utf-8")

    gt_context: dict[str, Any] = {}
    if gt_context_path.exists():
        with open(gt_context_path, encoding="utf-8") as f:
            gt_context = json.load(f)

    details_by_query: dict[str, Any] = {}
    for query, query_df in details_df.groupby("Query", sort=False):
        query = str(query)
        gt = gt_context.get(query, {})
        models: dict[int, Any] = {}

        gt_periods = gt.get("gt_periods", [])
        gt_blocks = gt.get("gt_blocks", [])

        for model_str, model_df in query_df.groupby("Model", sort=False):
            m_str = str(model_str)
            m_idx = int(m_str[1:]) - 1 if m_str.startswith("M") and m_str[1:].isdigit() else 0
            first = model_df.iloc[0]

            def _val(col: str, _first=first):
                v = _first.get(col)
                return str(v) if pd.notna(v) and v != "" else None

            hits = []
            for _, row in model_df.dropna(subset=["Rank"]).sort_values("Rank").iterrows():
                periodo = row["Periodo"] if pd.notna(row.get("Periodo")) else ""
                fonte = row["Fonte Normativa"] if pd.notna(row.get("Fonte Normativa")) else ""
                hits.append({
                    "_score": row["Score"] if pd.notna(row.get("Score")) else None,
                    "_source": {"periodo": str(periodo), "fonte_normativa": str(fonte)},
                })

            has_error = _val("Error") is not None
            if has_error:
                computed_flat, computed_block, computed_ndcg, computed_block_ap = 0.0, 0.0, 0.0, 0.0
            else:
                predicted_keys = [extract_hit_key(h) for h in hits]
                computed_flat = recall(gt_periods, predicted_keys)
                computed_block = block_recall(gt_blocks, predicted_keys)
                computed_ndcg = ndcg(gt_periods, predicted_keys)
                computed_block_ap = block_ap(gt_blocks, predicted_keys)

            models[m_idx] = {
                "flat_recall": computed_flat,
                "block_recall": computed_block,
                "ndcg": computed_ndcg,
                "block_ap": computed_block_ap,
                "error": _val("Error"),
                "generation_error": _val("Generation Error"),
                "hits": hits,
            }

        details_by_query[query] = {
            "query_type": gt.get("query_type", ""),
            "gt_periods": gt.get("gt_periods", []),
            "gt_blocks": gt.get("gt_blocks", []),
            "raw_results": gt.get("raw_results", []),
            "models": models,
        }

    return details_by_query


# ── model table ───────────────────────────────────────────────────────────────

# Each entry in row_refs: (run_dir, meta, model_idx_1based)
ModelRef = tuple[Path, dict, int]


def _build_model_table(
    candidates: list[tuple[Path, dict]],
) -> tuple[pd.DataFrame, list[ModelRef]]:
    rows: list[dict[str, Any]] = []
    refs: list[ModelRef] = []

    for rd, m in candidates:
        configs: list[dict] = m.get("model_configs", [])

        scores: dict[str, tuple] = {}
        try:
            summary_df = _load_run_summary(rd)
            if not summary_df.empty:
                _, overview_df = build_simple_qtype_tables(summary_df)
                ranking_df = build_ranking_overview_df(summary_df)
                for _, row in overview_df.iterrows():
                    model_key = str(row["Model"])
                    cov_macro = round(float(row["Macro Score"]), 3)
                    cov_weighted = round(float(row["Weighted Score"]), 3)
                    rank_row = ranking_df[ranking_df["Model"] == model_key] if not ranking_df.empty else pd.DataFrame()
                    rank_macro = round(float(rank_row.iloc[0]["Ranking Macro"]), 3) if not rank_row.empty else None
                    rank_weighted = round(float(rank_row.iloc[0]["Ranking Weighted"]), 3) if not rank_row.empty else None
                    scores[model_key] = (cov_macro, cov_weighted, rank_macro, rank_weighted)
        except Exception:
            pass

        ts = m.get("created_at_utc", "")
        time_label = _fmt_ts(ts, "%Y-%m-%d %H:%M:%S") if ts else ""

        for i, cfg in enumerate(configs, 1):
            cov_macro, cov_weighted, rank_macro, rank_weighted = scores.get(f"M{i}", (None, None, None, None))
            rows.append({
                "Run": rd.name,
                "Time": time_label,
                **_model_columns(cfg),
                "Recall Macro": cov_macro if cov_macro is not None else "—",
                "Recall Weighted": cov_weighted if cov_weighted is not None else "—",
                "Rank. Macro": rank_macro if rank_macro is not None else "—",
                "Rank. Weighted": rank_weighted if rank_weighted is not None else "—",
            })
            refs.append((rd, m, i))

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Sort by Recall Weighted descending (put "—" last)
    if not df.empty and "Recall Weighted" in df.columns:
        df["_sort"] = pd.to_numeric(df["Recall Weighted"], errors="coerce")
        sorted_idx = df.sort_values("_sort", ascending=False).index.tolist()
        df = df.loc[sorted_idx].drop(columns="_sort").reset_index(drop=True)
        refs = [refs[i] for i in sorted_idx]

    return df, refs


# ── merge by selected models ──────────────────────────────────────────────────

def _merge_summaries_by_models(
    selected_models: list[tuple[ModelRef, str]],  # (ref, run_label)
    include_fonte: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    dfs: list[pd.DataFrame] = []
    legend: list[tuple[str, str, str]] = []

    for new_idx, ((run_dir, meta, m_idx), run_label) in enumerate(selected_models, 1):
        df = _load_run_summary(run_dir, include_fonte=include_fonte)
        if df.empty:
            continue
        configs: list[dict] = meta.get("model_configs", [])
        cfg = configs[m_idx - 1] if m_idx - 1 < len(configs) else {}

        recall_col = f"Recall M{m_idx}"
        block_col = f"Block Recall M{m_idx}"
        ndcg_col = f"NDCG M{m_idx}"
        block_ap_col = f"Block AP M{m_idx}"
        keep = [c for c in ["Query", "Query Type", recall_col, block_col, ndcg_col, block_ap_col] if c in df.columns]
        df = df[keep].rename(columns={
            recall_col: f"Recall M{new_idx}",
            block_col: f"Block Recall M{new_idx}",
            ndcg_col: f"NDCG M{new_idx}",
            block_ap_col: f"Block AP M{new_idx}",
        })
        dfs.append(df)
        legend.append((f"M{new_idx}", run_label, _model_short_label(cfg)))

    if not dfs:
        return pd.DataFrame(), []

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=["Query", "Query Type"], how="outer")

    recall_cols = [c for c in merged.columns if re.match(r"Recall M\d+$", c)]
    block_cols = [c for c in merged.columns if re.match(r"Block Recall M\d+$", c)]
    ndcg_cols = [c for c in merged.columns if re.match(r"NDCG M\d+$", c)]
    block_ap_cols = [c for c in merged.columns if re.match(r"Block AP M\d+$", c)]
    merged["AVG Recall"] = merged[recall_cols].mean(axis=1).round(3)
    merged["AVG Block Recall"] = merged[block_cols].mean(axis=1).round(3)
    merged["Max Recall"] = merged[recall_cols].max(axis=1).round(3)
    merged["Max Block Recall"] = merged[block_cols].max(axis=1).round(3)
    if ndcg_cols:
        merged["AVG NDCG"] = merged[ndcg_cols].mean(axis=1).round(3)
    if block_ap_cols:
        merged["AVG Block AP"] = merged[block_ap_cols].mean(axis=1).round(3)

    return merged, legend


def _merge_details_by_models(
    selected_models: list[tuple[ModelRef, str]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for new_idx, ((run_dir, meta, m_idx), _) in enumerate(selected_models, 1):
        details = _load_details_by_query(run_dir)
        for query, d in details.items():
            if query not in merged:
                merged[query] = {
                    "query_type": d["query_type"],
                    "gt_periods": d["gt_periods"],
                    "gt_blocks": d["gt_blocks"],
                    "raw_results": d["raw_results"],
                    "models": {},
                }
            if (m_idx - 1) in d["models"]:
                merged[query]["models"][new_idx - 1] = d["models"][m_idx - 1]

    return merged


# ── analysis rendering ────────────────────────────────────────────────────────

def _render_overview_heatmap(overview_heatmap_df: pd.DataFrame, heatmap_n_df: pd.DataFrame):
    if overview_heatmap_df.empty or px is None:
        if not overview_heatmap_df.empty:
            st.dataframe(overview_heatmap_df, use_container_width=True)
        return
    fig = px.imshow(
        overview_heatmap_df,
        color_continuous_scale="RdYlGn", zmin=0.0, zmax=1.0, aspect="auto",
        labels={"x": "Model", "y": "Query Type", "color": "Score"},
    )
    text_matrix = [
        [f"{float(s):.3f} (n={int(n)})" for s, n in zip(sr, nr)]
        for sr, nr in zip(overview_heatmap_df.values, heatmap_n_df.values)
    ]
    fig.update_traces(
        text=text_matrix, texttemplate="%{text}", customdata=heatmap_n_df.values,
        hovertemplate="Query Type: %{y}<br>Model: %{x}<br>Score: %{z:.3f}<br>n: %{customdata}<extra></extra>",
    )
    height = _preview_heatmap_height(len(overview_heatmap_df.index), max_height_px=280)
    fig.update_layout(height=height, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def _render_analysis(selected_models: list[tuple[ModelRef, str]], include_fonte: bool = True):
    merged_summary, legend = _merge_summaries_by_models(selected_models, include_fonte=include_fonte)
    if merged_summary.empty:
        st.warning("Nessun dato disponibile per i modelli selezionati.")
        return

    # Build unified scores table (recall + ranking affiancati)
    score_rows: list[dict[str, Any]] = []
    for (run_dir, _, m_idx), run_label in selected_models:
        macro, weighted, rank_macro, rank_weighted = None, None, None, None
        try:
            run_summary = _load_run_summary(run_dir, include_fonte=include_fonte)
            if not run_summary.empty:
                _, overview_df = build_simple_qtype_tables(run_summary)
                row = overview_df[overview_df["Model"] == f"M{m_idx}"]
                if not row.empty:
                    macro = round(float(row.iloc[0]["Macro Score"]), 3)
                    weighted = round(float(row.iloc[0]["Weighted Score"]), 3)
                ranking_df = build_ranking_overview_df(run_summary)
                rank_row = ranking_df[ranking_df["Model"] == f"M{m_idx}"]
                if not rank_row.empty:
                    rank_macro = round(float(rank_row.iloc[0]["Ranking Macro"]), 3)
                    rank_weighted = round(float(rank_row.iloc[0]["Ranking Weighted"]), 3)
        except Exception:
            pass
        score_rows.append({
            "Run": run_label,
            "Recall Macro": macro if macro is not None else "—",
            "Recall Weighted": weighted if weighted is not None else "—",
            "Rank. Macro": rank_macro if rank_macro is not None else "—",
            "Rank. Weighted": rank_weighted if rank_weighted is not None else "—",
        })

    unified_rows = [
        {"Modello": new_lbl, **sr}
        for (new_lbl, _, _), sr in zip(legend, score_rows)
    ]
    unified_df = pd.DataFrame(unified_rows)

    st.markdown("## Performance Globali")
    top_left, top_right = st.columns([3, 2], gap="large")
    with top_left:
        st.dataframe(unified_df, use_container_width=True, hide_index=True)
        if px is not None:
            bar_cols = [c for c in ["Recall Weighted", "Rank. Weighted"] if c in unified_df.columns]
            bar_df = unified_df[["Modello"] + bar_cols].melt(
                id_vars="Modello", var_name="Metrica", value_name="Score"
            )
            bar_df["Score"] = pd.to_numeric(bar_df["Score"], errors="coerce")
            fig = px.bar(
                bar_df, x="Modello", y="Score", color="Metrica",
                barmode="group",
                color_discrete_map={"Recall Weighted": "#4c72b0", "Rank. Weighted": "#dd8452"},
            )
            fig.update_layout(
                yaxis=dict(range=[0, 1], title="Score"),
                margin={"b": 40, "l": 0, "r": 0, "t": 20},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    with top_right:
        overview_heatmap_df, heatmap_n_df = build_scoring_heatmap_df(merged_summary)
        if not overview_heatmap_df.empty:
            st.markdown("### Recall")
            st.caption("A = Flat Recall, B = Block Recall, A+B = (Flat+Block)/2")
            _render_overview_heatmap(overview_heatmap_df, heatmap_n_df)

        ranking_heatmap_df, ranking_n_df = build_ranking_scoring_heatmap_df(merged_summary)
        if not ranking_heatmap_df.empty:
            st.markdown("### Ranking")
            st.caption("A = NDCG, B = Block AP, A+B = (NDCG+Block AP)/2")
            _render_overview_heatmap(ranking_heatmap_df, ranking_n_df)

    simple_qtype_tables, _ = build_simple_qtype_tables(merged_summary)
    st.divider()
    st.markdown("## Performance per Categoria")
    detail_cols = st.columns(3, gap="large")
    for col, qtype in zip(detail_cols, ["A", "B", "A+B"]):
        with col:
            st.markdown(f"### Tipo: {qtype}")
            t = simple_qtype_tables.get(qtype, pd.DataFrame())
            if not t.empty:
                st.dataframe(t, use_container_width=True, hide_index=True)
            else:
                st.info("Nessun dato.")

    st.divider()
    st.markdown("## Dettaglio per Query")
    merged_details = _merge_details_by_models(selected_models)
    if merged_details:
        _render_results_by_query_type(merged_summary, merged_details, key_prefix="merged_")
    else:
        st.caption("Dettaglio non disponibile (gt_context.json mancante in uno o più run).")


# ── entry point ───────────────────────────────────────────────────────────────

def _render_runs_explorer_view():
    st.title("Storico")

    run_dirs = _list_runs()
    if not run_dirs:
        st.info(f"Nessun run trovato in `{RUNS_DIR}`.")
        return

    runs_with_meta = [(rd, _load_meta(rd)) for rd in run_dirs]
    runs_with_meta = [(rd, m) for rd, m in runs_with_meta if m]
    if not runs_with_meta:
        st.info("Nessun run con meta.json trovato.")
        return

    by_dataset: dict[str, list[tuple[Path, dict]]] = {}
    for rd, m in runs_with_meta:
        by_dataset.setdefault(m.get("gt_reference", "unknown"), []).append((rd, m))

    datasets = sorted(by_dataset.keys())
    selected_dataset = st.selectbox("Dataset", datasets, key="explorer_dataset")
    candidates = by_dataset[selected_dataset]

    model_df, row_refs = _build_model_table(candidates)
    if model_df.empty:
        st.info("Nessun modello disponibile.")
        return

    st.markdown(f"**{len(model_df)} modelli disponibili** — seleziona le righe da confrontare:")
    table_event = st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="explorer_model_table",
    )

    selected_indices = table_event.selection.rows
    if not selected_indices:
        st.info("Seleziona almeno un modello dalla tabella.")
        st.session_state.pop("explorer_active_models", None)
        st.session_state.pop("explorer_last_selection", None)
        return

    # Invalidate results if selection changed since last Esplora
    if selected_indices != st.session_state.get("explorer_last_selection"):
        st.session_state.pop("explorer_active_models", None)

    # Preview JSON configs for selected models
    with st.expander("Config modelli selezionati", expanded=True):
        for new_idx, idx in enumerate(selected_indices, 1):
            run_dir, meta, m_idx = row_refs[idx]
            configs: list[dict] = meta.get("model_configs", [])
            cfg = configs[m_idx - 1] if m_idx - 1 < len(configs) else {}
            st.markdown(f"**M{new_idx}** — {_model_short_label(cfg)}")
            st.json(cfg, expanded=False)

    if st.button("Esplora", type="primary", key="explorer_run_button"):
        st.session_state["explorer_active_models"] = [
            (row_refs[i], _run_short_label(row_refs[i][0], row_refs[i][1]))
            for i in selected_indices
        ]
        st.session_state["explorer_last_selection"] = selected_indices

    active_models = st.session_state.get("explorer_active_models")
    if not active_models:
        return

    include_fonte = True  # fonte normativa sempre inclusa (toggle rimosso)

    st.divider()
    _render_analysis(active_models, include_fonte=include_fonte)
