import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import evaluation_pipeline

try:
    import plotly.express as px
except Exception:
    px = None

RUNS_DIR = Path("runs/evaluations_pipeline")
_METRICS_VERSION = 1

# Metriche confrontabili tra tutte le pipeline (sempre mostrate).
_COMPARABLE_PREFIXES = ["Type Accuracy", "Final Recall", "Final Precision", "NDCG", "Purity"]
# Metriche diagnostiche valide per categoria di query (GT type).
# La categoria Reato non ha target circostanziati → niente Recall Circ.
_DIAGNOSTIC_PREFIXES_BY_CAT = {
    "reato": ["Recall Reati", "Recall Pool"],
    "reato_circostanziato": ["Recall Reati", "Recall Circostanziati", "Recall Pool"],
    "mixed": ["Recall Reati", "Recall Circostanziati", "Recall Pool"],
}
# Etichette compatte per le colonne aggregate.
_PREFIX_LABELS = {"Recall Circostanziati": "Recall Circ."}


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


def _run_status(run_dir: Path) -> str:
    path = run_dir / "status.json"
    if not path.exists():
        details_path = run_dir / "details.csv"
        gt_path = run_dir / "gt_context.json"
        if details_path.exists() and gt_path.exists():
            try:
                n_done = pd.read_csv(details_path, usecols=["Query"])["Query"].nunique()
                n_gt = len(json.loads(gt_path.read_text(encoding="utf-8")))
                if n_done >= n_gt:
                    return "completo"
            except Exception:
                pass
        return "sconosciuto"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "sconosciuto"
    status = data.get("status", "")
    if status == "complete":
        return "completo"
    if status == "complete_with_errors":
        return "completo (errori)"
    pid = data.get("pid")
    if pid:
        try:
            os.kill(pid, 0)
            return "in corso (errori)" if status == "running_with_errors" else "in corso"
        except OSError:
            return "interrotto"
    return "sconosciuto"


@st.cache_data
def _load_run_summary(run_dir: Path, metrics_version: int = _METRICS_VERSION) -> pd.DataFrame:
    details_path = run_dir / "details.csv"
    gt_context_path = run_dir / "gt_context.json"
    if not details_path.exists():
        return pd.DataFrame()
    details_df = pd.read_csv(details_path, encoding="utf-8")
    gt_context: dict = {}
    if gt_context_path.exists():
        with open(gt_context_path, encoding="utf-8") as f:
            gt_context = json.load(f)
    return evaluation_pipeline.compute_summary(details_df, gt_context)


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _split_label(cfg: dict) -> str:
    """Tecnica di split usata dal run, o '—' se non rilevante (input = query intera)."""
    s1, s3 = cfg.get("stage1_input", "q"), cfg.get("stage3_input", "q")
    if s1 == "q" and s3 == "q":
        return "—"
    method = cfg.get("split_method", "regex")
    if method == "llm":  # la versione prompt è rilevante solo per lo splitter LLM
        method = f"llm {cfg.get('split_version', 'v1')}"
    return f"{method} (s1={s1}, s3={s3})"


def _config_label(cfg: dict) -> str:
    pt = cfg.get("pipeline_type", "pool")
    s1 = cfg.get("stage1_mode", "dense")[0]
    s3 = cfg.get("stage3_mode", "dense")[0]
    clf_v = cfg.get("classifier_version", "v1") if pt == "pool" else cfg.get("only_classifier_version", "v1")
    label = f"{pt} · s1={s1} s3={s3} · m={cfg.get('m')} n={cfg.get('n')} k={cfg.get('k')} · {cfg.get('index_circ', '?')} · {cfg.get('llm_model', '?')} · clf={clf_v}"
    split = _split_label(cfg)
    if split != "—":
        label += f" · split={split}"
    return label


def _build_model_table(candidates: list[tuple[Path, dict]]) -> tuple[pd.DataFrame, list[tuple[Path, dict, int]]]:
    rows = []
    refs = []

    for rd, meta in candidates:
        configs: list[dict] = meta.get("model_configs", [])
        ts = meta.get("created_at_utc", "")
        time_label = _fmt_ts(ts) if ts else ""

        metrics_by_model: dict[str, dict] = {}
        n_queries = 0
        try:
            summary_df = _load_run_summary(rd)
            if not summary_df.empty:
                n_queries = len(summary_df)
                metric_prefixes = ("Final Recall", "Final Precision", "NDCG", "Purity", "Type Accuracy")
                for ii in range(1, len(configs) + 1):
                    for prefix in metric_prefixes:
                        col = f"{prefix} M{ii}"
                        if col in summary_df.columns:
                            vals = pd.to_numeric(summary_df[col], errors="coerce").dropna()
                            metrics_by_model.setdefault(f"M{ii}", {})[prefix] = round(float(vals.mean()), 3) if len(vals) else None
        except Exception:
            pass

        status = _run_status(rd)
        for i, cfg in enumerate(configs, 1):
            m_key = f"M{i}"
            m = metrics_by_model.get(m_key, {})
            rows.append({
                "Run": rd.name,
                "Time": time_label,
                "Status": status,
                "#queries": n_queries or "—",
                "stage1": cfg.get("stage1_mode", "dense")[0],
                "stage3": cfg.get("stage3_mode", "dense")[0],
                "llm": cfg.get("llm_model", "?"),
                "m": cfg.get("m"),
                "n": cfg.get("n"),
                "k": cfg.get("k"),
                "index_circ": cfg.get("index_circ", "?"),
                "pipeline_type": cfg.get("pipeline_type", "pool"),
                "clf_version": cfg.get("classifier_version", "v1") if cfg.get("pipeline_type", "pool") == "pool" else cfg.get("only_classifier_version", "v1"),
                "split": _split_label(cfg),
                "type_accuracy": m.get("Type Accuracy") if m.get("Type Accuracy") is not None else "—",
                "final_recall": m.get("Final Recall") if m.get("Final Recall") is not None else "—",
                "final_precision": m.get("Final Precision") if m.get("Final Precision") is not None else "—",
                "ndcg": m.get("NDCG") if m.get("NDCG") is not None else "—",
                "purity": m.get("Purity") if m.get("Purity") is not None else "—",
            })
            refs.append((rd, meta, i))

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty and "final_recall" in df.columns:
        df["_sort"] = pd.to_numeric(df["final_recall"], errors="coerce")
        sorted_idx = df.sort_values("_sort", ascending=False).index.tolist()
        df = df.loc[sorted_idx].drop(columns="_sort").reset_index(drop=True)
        refs = [refs[i] for i in sorted_idx]

    return df, refs


@st.cache_data
def _load_details(run_dir: Path, metrics_version: int = _METRICS_VERSION) -> pd.DataFrame:
    path = run_dir / "details.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


@st.cache_data
def _load_debug(run_dir: Path) -> dict:
    path = run_dir / "debug.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _render_category_heatmap(merged: pd.DataFrame, model_labels: list[str], gt_context: dict, metric_col_prefix: str, title: str, exclude_cats: list[str] | None = None, chart_key: str = ""):
    if px is None or not gt_context:
        return
    all_cats = [("reato", "reato"), ("reato_circostanziato", "reato_circ."), ("mixed", "mixed")]
    if exclude_cats:
        all_cats = [(c, l) for c, l in all_cats if c not in exclude_cats]
    cats = [c for c, _ in all_cats]
    cat_labels = [l for _, l in all_cats]
    model_keys = [f"M{i}" for i in range(1, len(model_labels) + 1)]
    merged["_tipo"] = merged["Query"].map(lambda q: gt_context.get(q, {}).get("type", "?"))
    values, counts = [], []
    valid_cats, valid_cat_labels = [], []
    for cat, cat_label in zip(cats, cat_labels):
        cat_df = merged[merged["_tipo"] == cat]
        if cat_df.empty:
            continue
        row_vals, row_counts = [], []
        for mk in model_keys:
            col = f"{metric_col_prefix} {mk}"
            if col in cat_df.columns:
                vals = pd.to_numeric(cat_df[col], errors="coerce").dropna()
                row_vals.append(round(float(vals.mean()), 3) if len(vals) else 0.0)
                row_counts.append(len(vals))
            else:
                row_vals.append(0.0)
                row_counts.append(0)
        values.append(row_vals)
        counts.append(row_counts)
        valid_cats.append(cat)
        valid_cat_labels.append(cat_label)
    merged.drop(columns=["_tipo"], inplace=True)
    if not values:
        return
    val_df = pd.DataFrame(values, index=valid_cat_labels, columns=model_keys)
    n_df = pd.DataFrame(counts, index=valid_cat_labels, columns=model_keys)
    st.markdown(f"**{title}**")
    fig = px.imshow(val_df, color_continuous_scale="RdYlGn", zmin=0.0, zmax=1.0, aspect="auto",
                    labels={"x": "Modello", "y": "Categoria", "color": "Score"})
    text_matrix = [[f"{v:.3f} (n={n})" for v, n in zip(vr, nr)] for vr, nr in zip(val_df.values, n_df.values)]
    fig.update_traces(text=text_matrix, texttemplate="%{text}",
                      hovertemplate="Categoria: %{y}<br>Modello: %{x}<br>Score: %{z:.3f}<extra></extra>")
    fig.update_layout(height=30 + len(valid_cats) * 40, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True, key=chart_key or f"pipeline_cat_heatmap_{metric_col_prefix.replace(' ', '_').lower()}")


def _render_pipeline_query_heatmap(cat_df: pd.DataFrame, model_labels: list[str], metric_prefix: str, chart_key: str = ""):
    if px is None or "Query" not in cat_df.columns:
        return
    model_keys = [f"M{i}" for i in range(1, len(model_labels) + 1)]
    avail_cols = [f"{metric_prefix} {mk}" for mk in model_keys if f"{metric_prefix} {mk}" in cat_df.columns]
    if not avail_cols:
        return
    plot_df = cat_df[["Query"] + avail_cols].set_index("Query")
    plot_df.columns = [c.replace(f"{metric_prefix} ", "") for c in plot_df.columns]
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    full_labels = [str(v) for v in plot_df.index.tolist()]
    display_labels = [l[:38] + "…" if len(l) > 38 else l for l in full_labels]
    plot_df.index = display_labels
    fig = px.imshow(plot_df, color_continuous_scale="RdYlGn", zmin=0.0, zmax=1.0, aspect="auto",
                    labels={"x": "Modello", "y": "Query", "color": metric_prefix})
    customdata = [[fl for _ in plot_df.columns] for fl in full_labels]
    fig.update_traces(customdata=customdata,
                      hovertemplate="Query: %{customdata}<br>Modello: %{x}<br>Score: %{z:.3f}<extra></extra>")
    height = max(120, min(480, 70 + len(plot_df) * 22))
    fig.update_layout(height=height, margin={"l": 220, "r": 0, "t": 20, "b": 0})
    fig.update_yaxes(automargin=True, tickfont={"size": 10})
    st.plotly_chart(fig, use_container_width=True, key=chart_key or f"pipeline_query_heatmap_{metric_prefix.replace(' ', '_').lower()}")


def _render_analysis(selected_models: list[tuple[tuple[Path, dict, int], str]]):
    # Raccoglie summary per ogni modello selezionato
    dfs: list[pd.DataFrame] = []
    model_labels: list[str] = []
    model_pipeline_types: list[str] = []

    for new_idx, ((run_dir, meta, m_idx), run_label) in enumerate(selected_models, 1):
        summary_df = _load_run_summary(run_dir)
        if summary_df.empty:
            continue
        if f"Final Recall M{m_idx}" not in summary_df.columns:
            continue
        metric_prefixes = ("Recall Reati", "Recall Circostanziati", "Recall Pool", "Final Recall", "Final Precision", "NDCG", "Purity", "Type Accuracy", "Predicted Type")
        src_cols = [f"{p} M{m_idx}" for p in metric_prefixes if f"{p} M{m_idx}" in summary_df.columns]
        rename_map = {col: col.replace(f"M{m_idx}", f"M{new_idx}") for col in src_cols}
        renamed = summary_df[["Query"] + src_cols].rename(columns=rename_map)
        dfs.append(renamed)
        configs: list[dict] = meta.get("model_configs", [])
        cfg = configs[m_idx - 1] if m_idx - 1 < len(configs) else {}
        model_labels.append(f"M{new_idx}: {_config_label(cfg)}")
        model_pipeline_types.append(cfg.get("pipeline_type", "pool"))

    if not dfs:
        st.warning("Nessun dato disponibile.")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Query", how="outer")

    gt_context: dict[str, Any] = {}
    if selected_models:
        run_dir_first = selected_models[0][0][0]
        gt_path = run_dir_first / "gt_context.json"
        if gt_path.exists():
            with open(gt_path, encoding="utf-8") as f:
                gt_context = json.load(f)

    # ── maschera metriche non significative per pipeline ────────────────────
    # Interpretazione a display sul dato grezzo: vale anche per run già salvati.
    #  - recall_pool: il "pool" unificato (stage1+stage3 prima del top-k) esiste
    #    solo per la pool pipeline. Nel branching è ridondante (= final / = circ)
    #    o un top-k troncato → NaN per ogni pipeline branching.
    #  - recall_reati (solo branching_direct): lo stage1 gira solo nel branch
    #    reato. Ha senso solo per query realmente "reato" instradate lì; per le
    #    circostanziate (e le reato mal classificate) lo stage1 non è il percorso
    #    voluto → NaN.
    q_gt_type = merged["Query"].map(lambda q: gt_context.get(q, {}).get("type"))
    for new_idx, pt in enumerate(model_pipeline_types, 1):
        if pt == "pool":
            continue
        rr_col = f"Recall Reati M{new_idx}"
        rp_col = f"Recall Pool M{new_idx}"
        ptype_col = f"Predicted Type M{new_idx}"
        if rp_col in merged.columns:
            merged[rp_col] = float("nan")
        if pt == "branching_direct" and rr_col in merged.columns:
            pred_is_reato = merged[ptype_col] == "reato" if ptype_col in merged.columns else False
            keep = (q_gt_type == "reato") & pred_is_reato
            merged.loc[~keep, rr_col] = float("nan")

    # ── performance globali ────────────────────────────────────────────────
    if gt_context:
        merged["_tipo"] = merged["Query"].map(lambda q: gt_context.get(q, {}).get("type", "?"))

    st.markdown("## Performance Globali")
    global_rows = []
    for new_idx, label in enumerate(model_labels, 1):
        def _avg(col: str) -> Any:
            if col not in merged.columns:
                return "—"
            vals = pd.to_numeric(merged[col], errors="coerce").dropna()
            return round(float(vals.mean()), 3) if len(vals) else "—"

        global_rows.append({
            "Modello": f"M{new_idx}",
            "type_accuracy": _avg(f"Type Accuracy M{new_idx}"),
            "final_recall": _avg(f"Final Recall M{new_idx}"),
            "final_precision": _avg(f"Final Precision M{new_idx}"),
            "ndcg": _avg(f"NDCG M{new_idx}"),
            "purity": _avg(f"Purity M{new_idx}"),
            "#Query": len(merged),
        })

    global_df = pd.DataFrame(global_rows)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.dataframe(global_df, use_container_width=True, hide_index=True)
        if px is not None and not global_df.empty:
            numeric_cols = [c for c in ["type_accuracy", "final_recall", "final_precision", "ndcg", "purity"] if c in global_df.columns]
            melted = global_df.melt(id_vars="Modello", value_vars=numeric_cols, var_name="Metrica", value_name="Valore")
            melted["Valore"] = pd.to_numeric(melted["Valore"], errors="coerce")
            fig = px.bar(melted, x="Metrica", y="Valore", color="Modello", barmode="group", range_y=[0, 1])
            fig.update_layout(height=380, margin={"l": 0, "r": 0, "t": 20, "b": 60})
            st.plotly_chart(fig, use_container_width=True, key="pipeline_global_bar")
    with right:
        if gt_context:
            _render_category_heatmap(merged, model_labels, gt_context, "Type Accuracy", "Type Accuracy", chart_key="pipeline_cat_heatmap_type_accuracy")
            _render_category_heatmap(merged, model_labels, gt_context, "Final Recall", "Final Recall", chart_key="pipeline_cat_heatmap_final_recall")

    if "_tipo" in merged.columns:
        merged.drop(columns=["_tipo"], inplace=True)

    # ── performance per categoria ──────────────────────────────────────────
    if gt_context:
        st.divider()
        st.markdown("## Performance per Categoria")
        merged["_tipo"] = merged["Query"].map(lambda q: gt_context.get(q, {}).get("type", "?"))

        cat_labels = {"reato": "Reato", "reato_circostanziato": "Reato Circostanziato", "mixed": "Mixed"}

        cols = st.columns(3, gap="large")
        for col_ui, cat in zip(cols, ["reato", "reato_circostanziato", "mixed"]):
            cat_df = merged[merged["_tipo"] == cat]
            cat_prefixes = _DIAGNOSTIC_PREFIXES_BY_CAT[cat] + _COMPARABLE_PREFIXES
            with col_ui:
                if cat_df.empty:
                    st.markdown(f"### {cat_labels[cat]} (0)")
                    st.info("Nessun dato.")
                    continue
                n_queries = len(cat_df["Query"].dropna().unique()) if "Query" in cat_df.columns else len(cat_df)
                st.markdown(f"### {cat_labels[cat]} ({n_queries})")
                rows = []
                for new_idx, label in enumerate(model_labels, 1):
                    row: dict[str, Any] = {"Modello": f"M{new_idx}", "#Query": n_queries}
                    for col_prefix in cat_prefixes:
                        col_label = _PREFIX_LABELS.get(col_prefix, col_prefix)
                        col_name = f"{col_prefix} M{new_idx}"
                        if col_name in cat_df.columns:
                            vals = pd.to_numeric(cat_df[col_name], errors="coerce").dropna()
                            row[col_label] = round(float(vals.mean()), 3) if len(vals) else "—"
                        else:
                            row[col_label] = "—"
                    fr_col = f"Final Recall M{new_idx}"
                    if fr_col in cat_df.columns:
                        n_zero = int((pd.to_numeric(cat_df[fr_col], errors="coerce").fillna(0) == 0).sum())
                        row["#Recall=0"] = n_zero
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        merged.drop(columns=["_tipo"], inplace=True)

    # ── tabella per query con recall ───────────────────────────────────────
    st.divider()
    st.markdown("## Dettaglio per Query")

    detail_df = merged.copy()
    if gt_context:
        detail_df["_tipo"] = detail_df["Query"].map(lambda q: gt_context.get(q, {}).get("type", "?"))

    _cats = ["reato", "reato_circostanziato", "mixed"]
    _cat_labels = {"reato": "Reato", "reato_circostanziato": "Reato Circostanziato", "mixed": "Mixed"}

    def _cat_metric_cols(cat: str) -> list[str]:
        prefixes = _DIAGNOSTIC_PREFIXES_BY_CAT[cat] + _COMPARABLE_PREFIXES + ["Predicted Type"]
        return [
            f"{prefix} M{new_idx}"
            for new_idx in range(1, len(model_labels) + 1)
            for prefix in prefixes
            if f"{prefix} M{new_idx}" in detail_df.columns
        ]

    cat_dfs: dict[str, pd.DataFrame] = {}
    for cat in _cats:
        cols = _cat_metric_cols(cat)
        if "_tipo" in detail_df.columns:
            sub = detail_df[detail_df["_tipo"] == cat][["Query"] + cols].reset_index(drop=True)
        else:
            sub = detail_df[["Query"] + cols].reset_index(drop=True)
        cat_dfs[cat] = sub

    prev_sels: dict = st.session_state.get("pipeline_detail_prev_sels", {})
    curr_sels: dict = {}

    for cat in _cats:
        n = len(cat_dfs[cat])
        st.markdown(f"### {_cat_labels[cat]} ({n})")
        col_table, col_heatmap = st.columns([3, 2], gap="large")
        with col_table:
            ev = st.dataframe(
                cat_dfs[cat],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"pipeline_query_table_{cat}",
            )
            curr_sels[cat] = ev.selection.rows
        with col_heatmap:
            _render_pipeline_query_heatmap(cat_dfs[cat], model_labels, "Final Recall", chart_key=f"pipeline_query_heatmap_{cat}")

    changed_cat = next((c for c in _cats if curr_sels[c] != prev_sels.get(c, [])), None)
    st.session_state["pipeline_detail_prev_sels"] = curr_sels

    selected_query = None
    if changed_cat and curr_sels[changed_cat]:
        selected_query = cat_dfs[changed_cat].iloc[curr_sels[changed_cat][0]]["Query"]
    elif not changed_cat:
        for cat in _cats:
            if curr_sels[cat]:
                selected_query = cat_dfs[cat].iloc[curr_sels[cat][0]]["Query"]
                break

    # ── inspection hit per query selezionata ──────────────────────────────
    if selected_query is None:
        st.caption("Clicca su una query per ispezionare gli hit restituiti.")
        return

    target_keys = set(gt_context.get(selected_query, {}).get("target_keys", []))

    st.divider()
    st.markdown(f"### Inspection — *{selected_query}*")
    if target_keys:
        st.markdown("**GT target keys:**")
        for k in target_keys:
            st.markdown(f"- `{k}`")
    else:
        st.warning("Nessuna GT target key trovata per questa query.")

    for new_idx, ((run_dir, meta, m_idx), _) in enumerate(selected_models, 1):
        details_df = _load_details(run_dir)
        debug_all = _load_debug(run_dir)
        configs: list[dict] = meta.get("model_configs", [])
        cfg = configs[m_idx - 1] if m_idx - 1 < len(configs) else {}
        model_label_orig = f"M{m_idx}"
        st.markdown(f"**M{new_idx}** — {_config_label(cfg)}")

        debug_query = debug_all.get(selected_query, {}).get(model_label_orig, {})
        stage1_hits = debug_query.get("stage1_hits", [])
        stage3_by_reato = debug_query.get("stage3_by_reato", {})
        pool = debug_query.get("pool", [])
        classifier_debug = debug_query.get("classifier", {})
        splitter_debug = debug_query.get("splitter") or {}
        pred_type = classifier_debug.get("predicted_type", "—")
        gt_type_q = gt_context.get(selected_query, {}).get("type", "?")

        query_df = details_df[(details_df["Query"] == selected_query) & (details_df["Model"] == model_label_orig)] if not details_df.empty else pd.DataFrame()
        error_rows = query_df[query_df["Error"].notna()] if not query_df.empty else pd.DataFrame()
        hits_df = query_df.dropna(subset=["Rank"]).sort_values("Rank") if not query_df.empty else pd.DataFrame()

        total_circ = sum(len(v) for v in stage3_by_reato.values())
        match_icon = "✓" if pred_type == gt_type_q else "✗"

        tab_s1, tab_s3, tab_pool, tab_split, tab_cls, tab_final = st.tabs([
            f"Stage 1 ({len(stage1_hits)})",
            f"Stage 3 ({total_circ})",
            f"Pool ({len(pool)})",
            "Split",
            f"Classificatore {match_icon}",
            f"Risultati finali ({len(hits_df)})",
        ])

        with tab_s1:
            if stage1_hits:
                st.dataframe(pd.DataFrame([
                    {"Rank": i + 1, "Score": round(h["score"], 4) if h["score"] is not None else None, "combo_reati": h["key"]}
                    for i, h in enumerate(stage1_hits)
                ]), use_container_width=True, hide_index=True)
            else:
                st.info("Nessun hit da Stage 1.")

        with tab_s3:
            if stage3_by_reato:
                for reato, circ_hits in stage3_by_reato.items():
                    st.markdown(f"**{reato}** — {len(circ_hits)} hit")
                    if circ_hits:
                        st.dataframe(pd.DataFrame([
                            {"Rank": i + 1, "Score": round(h["score"], 4) if h["score"] is not None else None, "combo_reati_circostanziati": h["key"]}
                            for i, h in enumerate(circ_hits)
                        ]), use_container_width=True, hide_index=True)
            else:
                st.info("Nessun hit da Stage 3.")

        with tab_pool:
            if pool:
                st.dataframe(pd.DataFrame([
                    {"Rank": i + 1, "Score": round(h["score"], 4) if h["score"] is not None else None, "Source Index": h["index"], "Key": h["key"]}
                    for i, h in enumerate(pool)
                ]), use_container_width=True, hide_index=True)
            else:
                st.info("Pool vuoto.")

        with tab_split:
            if not splitter_debug:
                st.info("Nessuno split per questo run (input = q intera).")
            else:
                st.markdown("**Parti della query** (input → output dello splitter):")
                st.dataframe(pd.DataFrame([
                    {"parte": "q", "testo": splitter_debug.get("q", "")},
                    {"parte": "q_reato", "testo": splitter_debug.get("q_reato", "")},
                    {"parte": "q_circostanza", "testo": splitter_debug.get("q_circostanza", "")},
                ]), use_container_width=True, hide_index=True)
                if splitter_debug.get("parse_error"):
                    st.warning(splitter_debug["parse_error"])
                if splitter_debug.get("prompt"):
                    with st.expander("Prompt splitter (LLM)", expanded=False):
                        st.code(splitter_debug["prompt"], language="text")
                if splitter_debug.get("answer"):
                    with st.expander("Risposta raw splitter (LLM)", expanded=False):
                        st.code(splitter_debug["answer"], language="json")

        with tab_cls:
            st.markdown(f"**GT type:** `{gt_type_q}` — **Predicted:** `{pred_type}` {match_icon}")
            if classifier_debug.get("reasoning"):
                st.info(f"Ragionamento: {classifier_debug['reasoning']}")
            if classifier_debug.get("parse_error"):
                st.warning(classifier_debug["parse_error"])
            if classifier_debug.get("prompt"):
                with st.expander("Prompt inviato al LLM", expanded=False):
                    st.code(classifier_debug["prompt"], language="text")
            if classifier_debug.get("answer"):
                with st.expander("Risposta raw LLM", expanded=False):
                    st.code(classifier_debug["answer"], language="json")

        with tab_final:
            if not error_rows.empty:
                st.error(f"Errore: {error_rows.iloc[0]['Error']}")
            if hits_df.empty:
                st.info("Nessun risultato per questa query.")
            else:
                rows = []
                for _, row in hits_df.iterrows():
                    hit_key = str(row.get("Hit Key", "")) if pd.notna(row.get("Hit Key")) else ""
                    match = "✓" if hit_key in target_keys else "✗"
                    rows.append({
                        "Rank": int(row["Rank"]),
                        "Score": round(float(row["Score"]), 4) if pd.notna(row.get("Score")) else None,
                        "Match": match,
                        "Source Index": str(row.get("Source Index", "")),
                        "Hit Key": hit_key,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_explorer_pipeline_view():
    st.title("Explorer Reati")

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
        gt_ref = Path(m.get("gt_reference", "unknown")).name
        by_dataset.setdefault(gt_ref, []).append((rd, m))

    datasets = sorted(by_dataset.keys())
    selected_dataset = st.selectbox("Dataset", datasets, key="pipeline_explorer_dataset")
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
        key="pipeline_explorer_table",
    )

    selected_indices = table_event.selection.rows
    if not selected_indices:
        st.info("Seleziona almeno un modello dalla tabella.")
        st.session_state.pop("pipeline_explorer_active", None)
        return

    if selected_indices != st.session_state.get("pipeline_explorer_last_sel"):
        st.session_state.pop("pipeline_explorer_active", None)

    with st.expander("Config modelli selezionati", expanded=False):
        for new_idx, idx in enumerate(selected_indices, 1):
            run_dir, meta, m_idx = row_refs[idx]
            configs: list[dict] = meta.get("model_configs", [])
            cfg = configs[m_idx - 1] if m_idx - 1 < len(configs) else {}
            st.markdown(f"**M{new_idx}** — {_config_label(cfg)}")
            st.json(cfg, expanded=False)

    if st.button("Esplora", type="primary", key="pipeline_explorer_btn"):
        st.session_state["pipeline_explorer_active"] = [
            (row_refs[i], row_refs[i][0].name)
            for i in selected_indices
        ]
        st.session_state["pipeline_explorer_last_sel"] = selected_indices

    active_models = st.session_state.get("pipeline_explorer_active")
    if not active_models:
        return

    st.divider()
    _render_analysis(active_models)
