"""Paper Explorer: confronta i run prodotti dalla Paper Run (qualsiasi modello).

Tutti i run in runs/paper sono period-level sulla stessa GT, quindi vengono
trattati uniformemente e confrontati su Recall / Precision / NDCG. Riusa
compute_summary_from_details per le metriche.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from evaluation import (
    compute_summary_from_details, _norm_str,
    build_simple_qtype_tables, build_ranking_overview_df,
    build_scoring_heatmap_df, build_ranking_scoring_heatmap_df,
)
from views.explorer_view import _render_overview_heatmap
from config import (
    PAPER_HARD_RUNS_DIR, DEFAULT_INDEX_PAPER_PERIODI_FLAT,
    DEFAULT_INDEX_PAPER_REATI, DEFAULT_INDEX_PAPER_REATI_CIRC_LIGHT,
)
from reranker import RANKER_PROMPT_FILE

# Etichetta indice/i per la colonna Explore: i pool stanno sempre su reati+circ.
_POOL_INDEX_LABEL = f"{DEFAULT_INDEX_PAPER_REATI} + {DEFAULT_INDEX_PAPER_REATI_CIRC_LIGHT}"

try:
    import plotly.express as px
except Exception:
    px = None

RUNS_DIR = Path("runs/paper")
METRICS = ["Recall", "Precision", "NDCG"]
# Variante "hard": GT legacy con blocchi e tipi → si aggiungono le metriche di blocco.
METRICS_HARD = ["Recall", "Block Recall", "NDCG", "Block AP", "Precision"]
# Versione del prompt del reranker, derivata dal file usato (es. "ranker_v1.txt" -> "v1").
RERANK_VERSION = RANKER_PROMPT_FILE.replace("ranker_", "").replace(".txt", "")


def _list_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True)


def _load_meta(run_dir: Path) -> dict:
    path = run_dir / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def _load_summary(run_dir: Path, include_fonte: bool = True) -> pd.DataFrame:
    details_path = run_dir / "details.csv"
    gt_path = run_dir / "gt_context.json"
    if not details_path.exists():
        return pd.DataFrame()
    details_df = pd.read_csv(details_path, encoding="utf-8")
    gt_context = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
    return compute_summary_from_details(details_df, gt_context, include_fonte=include_fonte)


@st.cache_data
def _load_details(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "details.csv"
    return pd.read_csv(p, encoding="utf-8") if p.exists() else pd.DataFrame()


@st.cache_data
def _load_gt_context(run_dir: Path) -> dict:
    p = run_dir / "gt_context.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def _load_debug(run_dir: Path) -> dict:
    p = run_dir / "debug.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _hit_df(hits: list, key_col: str) -> pd.DataFrame:
    """DataFrame Rank/Score/key da una lista di hit (dict {key,score,index} o stringhe)."""
    rows = []
    for i, h in enumerate(hits, 1):
        if isinstance(h, dict):
            key, score = h.get("key", ""), h.get("score")
        else:
            key, score = str(h), None
        rows.append({"Rank": i, "Score": round(score, 4) if score is not None else None, key_col: key})
    df = pd.DataFrame(rows)
    if not df.empty and df["Score"].isna().all():
        df = df.drop(columns="Score")  # run vecchi senza score
    return df


def _render_pool_tabs(debug: dict, final_df: pd.DataFrame):
    """Inspection a tab della pipeline pool, in stile Explorer Reati."""
    clf = debug.get("classifier", {})
    s3 = debug.get("stage3_by_reato", {})
    splitter = debug.get("splitter") or {}
    total_circ = sum(len(v) for v in s3.values())

    tab_s1, tab_s3, tab_pool, tab_split, tab_cls, tab_final = st.tabs([
        f"Stage 1 ({len(debug.get('stage1_reati', []))})",
        f"Stage 3 ({total_circ})",
        f"Pool ({len(debug.get('pool', []))})",
        "Split",
        f"Classificatore ({clf.get('predicted_type', '—')})",
        f"Risultati finali ({len(final_df)})",
    ])
    with tab_s1:
        df = _hit_df(debug.get("stage1_reati", []), "combo_reati")
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("Nessun hit da Stage 1.")
    with tab_s3:
        if s3:
            for reato, hits in s3.items():
                st.markdown(f"**{reato}** — {len(hits)} hit")
                st.dataframe(_hit_df(hits, "combo_reati_circostanziati"), use_container_width=True, hide_index=True)
        else:
            st.info("Nessun hit da Stage 3.")
    with tab_pool:
        df = _hit_df(debug.get("pool", []), "Key")
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("Pool vuoto.")
    with tab_split:
        if not splitter:
            st.info("Nessuno split per questo run (input = query intera).")
        else:
            st.dataframe(pd.DataFrame([
                {"parte": "q", "testo": splitter.get("q", "")},
                {"parte": "q_reato", "testo": splitter.get("q_reato", "")},
                {"parte": "q_circostanza", "testo": splitter.get("q_circostanza", "")},
            ]), use_container_width=True, hide_index=True)
    with tab_cls:
        st.markdown(f"**Tipo predetto:** `{clf.get('predicted_type', '—')}`")
        if clf.get("reasoning"):
            st.info(f"Ragionamento: {clf['reasoning']}")
        if clf.get("prompt"):
            with st.expander("Prompt inviato al LLM", expanded=False):
                st.code(clf["prompt"], language="text")
        if clf.get("answer"):
            with st.expander("Risposta raw LLM", expanded=False):
                st.code(clf["answer"], language="json")
        st.markdown(f"**Combo selezionate ({len(debug.get('selected', []))})**")
        sel = _hit_df(debug.get("selected", []), "combo")
        if not sel.empty:
            st.dataframe(sel, use_container_width=True, hide_index=True)
    with tab_final:
        st.dataframe(final_df, use_container_width=True, hide_index=True) if not final_df.empty else st.info("Nessun risultato.")


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _describe_config(cfg: dict) -> str:
    """Etichetta estesa per legenda/inspection."""
    if cfg.get("pipeline_type") == "pool":
        rw = "rewrite" if cfg.get("stage1_input", "q") != "q" else "no-rewrite"
        prr = " · +period-rerank" if cfg.get("period_rerank") else ""
        nomap = " · no-map" if cfg.get("no_mapping") else ""
        return (f"Pool · s1={cfg.get('stage1_mode')} s3={cfg.get('stage3_mode')}"
                f" · j={cfg.get('j')} k={cfg.get('k')} · {cfg.get('llm_model','')} · {rw}{prr}{nomap}")
    mt = cfg.get("model_type", "?")
    base = {"Sparse search": "Sparse", "Dense search": "Dense", "Fusion (RRF)": "Hybrid"}.get(mt, mt)
    rr = cfg.get("reranker_config")
    k = rr["k"] if rr else cfg.get("k")
    desc = f"{base} · k={k}"
    if rr:
        desc += f" + rerank({rr.get('llm_model')})"
    return desc


def _split_label(cfg: dict) -> str:
    """Dettaglio dello split (come Explorer Reati): metodo + parti usate, o '—'."""
    s1, s3 = cfg.get("stage1_input", "q"), cfg.get("stage3_input", "q")
    if s1 == "q" and s3 == "q":
        return "—"
    method = cfg.get("split_method", "regex")
    if method == "llm":
        method = f"llm {cfg.get('split_version', 'v1')}"
    return f"{method} (s1={s1}, s3={s3})"


def _model_cols(cfg: dict) -> dict:
    """Colonne config de-aggregate (uniformi con Explorer Reati). '—' se non applicabile.

    reranker (on/llm), splitter (dettaglio), clf (classifier version), llm (modello).
    """
    if cfg.get("pipeline_type") == "pool":
        return {
            "Tipo": "Pool",
            "index": _POOL_INDEX_LABEL,
            "stage1": cfg.get("stage1_mode", "—"),
            "stage3": cfg.get("stage3_mode", "—"),
            "m": str(cfg.get("m", "—")),
            "n": str(cfg.get("n", "—")),
            "j": str(cfg.get("j", "—")),
            "k": str(cfg.get("k", "—")),
            "mapping": "no" if cfg.get("no_mapping") else "yes",
            "reranker": RERANK_VERSION if cfg.get("period_rerank") else "—",
            "splitter": _split_label(cfg),
            "clf": cfg.get("classifier_version", "—"),
            "llm": cfg.get("llm_model", "—"),
        }
    mt = cfg.get("model_type", "?")
    tipo = {"Sparse search": "Sparse", "Dense search": "Dense", "Fusion (RRF)": "Hybrid"}.get(mt, mt)
    rr = cfg.get("reranker_config")
    return {
        "Tipo": tipo,
        "index": cfg.get("index") or DEFAULT_INDEX_PAPER_PERIODI_FLAT,
        "stage1": "—", "stage3": "—", "m": "—", "n": "—", "j": "—",
        "k": str(rr["k"] if rr else cfg.get("k", "—")),
        "mapping": "—",
        "reranker": RERANK_VERSION if rr else "—",
        "splitter": "—",
        "clf": "—",
        "llm": rr["llm_model"] if rr else "—",   # reranker LLM
    }


def _run_status(run_dir: Path) -> str:
    """Stato del run: da status.json (+pid) se presente, altrimenti da #query fatte vs GT."""
    sp = run_dir / "status.json"
    if sp.exists():
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
            status = data.get("status", "")
            if status == "complete":
                return "completo"
            if status == "complete_with_errors":
                return "completo (errori)"
            pid = data.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                    return "in corso"
                except OSError:
                    return "interrotto"
            return status or "sconosciuto"
        except Exception:
            pass
    dp, gp = run_dir / "details.csv", run_dir / "gt_context.json"
    if dp.exists() and gp.exists():
        try:
            done = pd.read_csv(dp, usecols=["Query"])["Query"].nunique()
            ngt = len(json.loads(gp.read_text(encoding="utf-8")))
            return "completo" if done >= ngt else f"parziale ({done}/{ngt})"
        except Exception:
            pass
    return "sconosciuto"


def _run_means(run_dir: Path, include_fonte: bool, metrics: list[str]) -> dict | None:
    """Media su tutte le query delle metriche del modello (ogni run = 1 modello, M1)."""
    summary = _load_summary(run_dir, include_fonte)
    if summary.empty:
        return None
    out = {"#q": len(summary)}
    for m in metrics:
        col = f"{m} M1"
        out[m] = round(float(summary[col].mean()), 3) if col in summary.columns else None
    return out


def _run_block_scores(run_dir: Path, include_fonte: bool) -> dict | None:
    """Macro/Weighted (Recall e Ranking) per il singolo run (modello M1), come
    l'Explorer legacy: A→Flat Recall, B→Block Recall (Ranking: NDCG / Block AP)."""
    s = _load_summary(run_dir, include_fonte)
    if s.empty or "Query Type" not in s.columns:
        return None
    _, rec = build_simple_qtype_tables(s)
    if rec.empty:
        return None
    r0 = rec.iloc[0]
    out = {"Recall Macro": round(float(r0["Macro Score"]), 3),
           "Recall Weighted": round(float(r0["Weighted Score"]), 3),
           "Rank Macro": None, "Rank Weighted": None, "#q": len(s)}
    rnk = build_ranking_overview_df(s)
    if not rnk.empty:
        k0 = rnk.iloc[0]
        out["Rank Macro"] = round(float(k0["Ranking Macro"]), 3)
        out["Rank Weighted"] = round(float(k0["Ranking Weighted"]), 3)
    return out


BLOCK_SCORE_COLS = ["Recall Macro", "Recall Weighted", "Rank Macro", "Rank Weighted"]


# ── Intervalli di confidenza (bootstrap sulle query) ─────────────────────────
# Tutte le metriche mostrate sono medie su query: il CI si ottiene ricampionando
# le query con rimpiazzo (B=1000) e ri-aggregando ogni volta, così cattura anche
# la variabilità della composizione A/B. Il bootstrap dei compositi replica
# esattamente i builder legacy (A→Flat, B→Block, A+B→media), quindi il punto
# stimato cade nell'intervallo per costruzione.
CI_N_BOOT = 1000
CI_ALPHA = 0.05

_PERTYPE_CI_SPEC = {
    "A": [("Flat Recall AVG", "flat"), ("NDCG AVG", "ndcg")],
    "B": [("Block Recall AVG", "block"), ("Block AP AVG", "ap")],
    "A+B": [("Flat Recall AVG", "flat"), ("NDCG AVG", "ndcg"),
            ("Block Recall AVG", "block"), ("Block AP AVG", "ap")],
}


def _ci(samples) -> tuple[float, float]:
    """CI percentile (95% di default) da campioni bootstrap."""
    lo = float(np.percentile(samples, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(samples, 100 * (1 - CI_ALPHA / 2)))
    return lo, hi


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}–{hi:.3f}]"


def _mean_ci(values, seed: int = 0) -> tuple[float, float]:
    """CI bootstrap della media semplice (resampling delle query)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if v.size == 0:
        return (0.0, 0.0)
    draws = np.random.default_rng(seed).integers(0, v.size, (CI_N_BOOT, v.size))
    return _ci(v[draws].mean(axis=1))


def _msum_arrays(msum: pd.DataFrame, i: int) -> dict:
    """Array per-query (Recall/Block Recall/NDCG/Block AP) del modello Mi."""
    def col(name: str):
        c = f"{name} M{i}"
        return (pd.to_numeric(msum[c], errors="coerce").fillna(0.0).to_numpy()
                if c in msum.columns else np.zeros(len(msum)))
    return {"flat": col("Recall"), "block": col("Block Recall"),
            "ndcg": col("NDCG"), "ap": col("Block AP")}


def _block_overview_ci(msum: pd.DataFrame, n_models: int, seed: int = 0) -> dict:
    """CI di Recall/Rank Macro e Weighted, resampling le query sull'intero pool.
    Ritorna {Mi: {col: (lo, hi)}} per le colonne in BLOCK_SCORE_COLS."""
    qt = msum["Query Type"].to_numpy()
    n = len(msum)
    arrs = [_msum_arrays(msum, i) for i in range(1, n_models + 1)]
    acc = {mi: {c: [] for c in BLOCK_SCORE_COLS} for mi in range(n_models)}
    rng = np.random.default_rng(seed)
    for _ in range(CI_N_BOOT):
        idx = rng.integers(0, n, n)
        qti = qt[idx]
        mA, mB, mAB = qti == "A", qti == "B", qti == "A+B"
        nA, nB, nAB = int(mA.sum()), int(mB.sum()), int(mAB.sum())
        for mi, a in enumerate(arrs):
            for prefix, flat, block in (("Recall", a["flat"], a["block"]),
                                        ("Rank", a["ndcg"], a["ap"])):
                fi, bi = flat[idx], block[idx]
                sA = fi[mA].mean() if nA else 0.0
                sB = bi[mB].mean() if nB else 0.0
                sAB = ((fi[mAB].mean() + bi[mAB].mean()) / 2) if nAB else 0.0
                acc[mi][f"{prefix} Macro"].append((sA + sB + sAB) / 3)
                acc[mi][f"{prefix} Weighted"].append((nA * sA + nB * sB + nAB * sAB) / n)
    return {f"M{mi+1}": {c: _ci(np.asarray(v)) for c, v in acc[mi].items()}
            for mi in range(n_models)}


def _block_pertype_ci(msum: pd.DataFrame, n_models: int, seed: int = 1) -> dict:
    """CI delle colonne AVG per tipo, resampling DENTRO ciascun tipo (CI condizionato
    al tipo). Ritorna {qtype: {Mi: {col: (lo, hi)}}}."""
    qt = msum["Query Type"].to_numpy()
    arrs = [_msum_arrays(msum, i) for i in range(1, n_models + 1)]
    rng = np.random.default_rng(seed)
    out: dict = {t: {} for t in _PERTYPE_CI_SPEC}
    for t, spec in _PERTYPE_CI_SPEC.items():
        base = np.where(qt == t)[0]
        if base.size == 0:
            continue
        draws = rng.integers(0, base.size, (CI_N_BOOT, base.size))
        for mi, a in enumerate(arrs):
            out[t][f"M{mi+1}"] = {col: _ci(a[key][base][draws].mean(axis=1))
                                  for col, key in spec}
    return out


def _attach_pertype_ci(t: pd.DataFrame, ci_for_type: dict) -> pd.DataFrame:
    """Inserisce dopo ogni colonna AVG la relativa colonna '<col> ±' (stringa CI)."""
    avg_cols = {c for spec in _PERTYPE_CI_SPEC.values() for c, _ in spec}
    out = t.copy()
    ordered: list[str] = []
    for c in t.columns:
        ordered.append(c)
        if c in avg_cols:
            cic = f"{c} ±"
            out[cic] = out["Model"].map(
                lambda m, c=c: _fmt_ci(*ci_for_type[m][c]) if c in ci_for_type.get(m, {}) else "")
            ordered.append(cic)
    return out[ordered]


def _merge_per_query(sub: pd.DataFrame, runs_dir: Path, include_fonte: bool, metric: str):
    """Tabella Query × M1..Mn per una metrica."""
    merged = None
    for i, (_, row) in enumerate(sub.iterrows(), 1):
        s = _load_summary(runs_dir / row["Run"], include_fonte)
        col_name = f"{metric} M1"
        if s.empty or col_name not in s.columns:
            continue
        col = s[["Query", col_name]].rename(columns={col_name: f"M{i}"})
        merged = col if merged is None else merged.merge(col, on="Query", how="outer")
    return merged


def _merge_block_summary(sub: pd.DataFrame, runs_dir: Path, include_fonte: bool) -> pd.DataFrame:
    """Summary unito (Query, Query Type, <metrica> M1..Mn) per riusare i builder a
    blocchi del legacy (Macro/Weighted, heatmap per tipo). Ogni run è il modello Mi."""
    metrics = ["Recall", "Block Recall", "NDCG", "Block AP"]
    merged = None
    for i, (_, row) in enumerate(sub.iterrows(), 1):
        s = _load_summary(runs_dir / row["Run"], include_fonte)
        if s.empty or "Query Type" not in s.columns:
            continue
        keep = ["Query", "Query Type"] + [f"{m} M1" for m in metrics if f"{m} M1" in s.columns]
        part = s[keep].rename(columns={f"{m} M1": f"{m} M{i}" for m in metrics})
        merged = part if merged is None else merged.merge(part, on=["Query", "Query Type"], how="outer")
    return merged if merged is not None else pd.DataFrame()


def _render_block_analysis(sub: pd.DataFrame, runs_dir: Path, include_fonte: bool, ns: str,
                           cfg_cols: list[str], show_ci: bool = False):
    """Performance globali in stile Explorer legacy: Macro/Weighted (Recall+Ranking),
    heatmap A/B/A+B (Recall e Ranking), tabelle per categoria. Riusa i builder a blocchi.
    Con show_ci aggiunge gli intervalli di confidenza bootstrap (95%)."""
    msum = _merge_block_summary(sub, runs_dir, include_fonte)
    qtype_tables, recall_ov = (build_simple_qtype_tables(msum) if not msum.empty else ({}, pd.DataFrame()))
    if recall_ov.empty:
        st.info("Nessun dato per tipo di query nei run selezionati.")
        return

    uni = recall_ov.rename(columns={"Model": "Modello", "Macro Score": "Recall Macro", "Weighted Score": "Recall Weighted"})
    rank_ov = build_ranking_overview_df(msum)
    if not rank_ov.empty:
        uni = uni.merge(
            rank_ov.rename(columns={"Model": "Modello", "Ranking Macro": "Rank Macro", "Ranking Weighted": "Rank Weighted"}),
            on="Modello", how="left")

    ov_ci = _block_overview_ci(msum, len(sub)) if show_ci else {}
    if ov_ci:
        st.caption(f"Intervalli di confidenza: bootstrap {CI_N_BOOT}× sulle query, "
                   f"{int((1 - CI_ALPHA) * 100)}% percentile.")
        ordered = ["Modello"]
        for c in BLOCK_SCORE_COLS:
            if c in uni.columns:
                uni[f"{c} ±"] = uni["Modello"].map(lambda m, c=c: _fmt_ci(*ov_ci[m][c]))
                ordered += [c, f"{c} ±"]
        uni = uni[ordered]

    cfg_df = sub[cfg_cols].copy()
    cfg_df.insert(0, "Modello", [f"M{i+1}" for i in range(len(sub))])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.dataframe(cfg_df.merge(uni, on="Modello", how="left"), use_container_width=True, hide_index=True)
        if px is not None:
            score_cols = [c for c in ["Recall Weighted", "Rank Weighted"] if c in uni.columns]
            bar = uni.melt(id_vars="Modello", value_vars=score_cols, var_name="Metrica", value_name="Score")
            bar["Score"] = pd.to_numeric(bar["Score"], errors="coerce")
            err_kw = {}
            if ov_ci:
                bar["e_plus"] = bar.apply(lambda r: ov_ci[r["Modello"]][r["Metrica"]][1] - r["Score"], axis=1)
                bar["e_minus"] = bar.apply(lambda r: r["Score"] - ov_ci[r["Modello"]][r["Metrica"]][0], axis=1)
                err_kw = {"error_y": "e_plus", "error_y_minus": "e_minus"}
            fig = px.bar(bar, x="Modello", y="Score", color="Metrica", barmode="group", range_y=[0, 1], **err_kw)
            fig.update_layout(height=320, margin={"t": 20, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig, use_container_width=True, key=f"{ns}_block_bar")
    with right:
        rh, rn = build_scoring_heatmap_df(msum)
        if not rh.empty:
            st.caption("Recall — A=Flat, B=Block, A+B=media")
            _render_overview_heatmap(rh, rn)
        kh, kn = build_ranking_scoring_heatmap_df(msum)
        if not kh.empty:
            st.caption("Ranking — A=NDCG, B=Block AP")
            _render_overview_heatmap(kh, kn)

    for i, (_, row) in enumerate(sub.iterrows(), 1):
        st.caption(f"M{i} — {_describe_config(row['_config'])}  ·  {row['Run']}")

    st.markdown("### Performance per categoria")
    pt_ci = _block_pertype_ci(msum, len(sub)) if show_ci else {}
    for col, qt in zip(st.columns(3, gap="large"), ["A", "B", "A+B"]):
        with col:
            st.markdown(f"**Tipo {qt}**")
            t = qtype_tables.get(qt, pd.DataFrame())
            if not t.empty and pt_ci.get(qt):
                t = _attach_pertype_ci(t, pt_ci[qt])
            st.dataframe(t, use_container_width=True, hide_index=True) if not t.empty else st.info("—")


def _render_global_heatmap(global_df: pd.DataFrame, metrics: list[str], key: str):
    """Heatmap riepilogo: modelli × metriche."""
    if px is None or global_df.empty:
        return
    plot = global_df.set_index("Modello")[metrics].apply(pd.to_numeric, errors="coerce")
    fig = px.imshow(plot, color_continuous_scale="RdYlGn", zmin=0.0, zmax=1.0, aspect="auto",
                    labels={"x": "Metrica", "y": "Modello", "color": "Score"})
    fig.update_traces(text=plot.round(3).values, texttemplate="%{text}")
    fig.update_layout(height=max(160, 60 + len(plot) * 36), margin={"l": 0, "r": 0, "t": 20, "b": 0})
    st.plotly_chart(fig, use_container_width=True, key=key)


def _render_query_heatmap(df, value_cols: list[str], color_label: str, key: str):
    """Heatmap query × colonne (modello[/metrica]), come Explorer Reati."""
    if px is None or df is None or not value_cols:
        return
    full = [str(q) for q in df["Query"]]
    labels = [l[:40] + "…" if len(l) > 40 else l for l in full]
    plot_df = df[value_cols].copy()
    plot_df.index = labels
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
    fig = px.imshow(plot_df, color_continuous_scale="RdYlGn", zmin=0.0, zmax=1.0, aspect="auto",
                    labels={"x": "Modello / Metrica", "y": "Query", "color": color_label})
    height = max(220, min(750, 70 + len(plot_df) * 16))
    fig.update_layout(height=height, margin={"l": 160, "r": 0, "t": 20, "b": 0})
    fig.update_yaxes(automargin=True, tickfont={"size": 9})
    fig.update_xaxes(tickfont={"size": 9})
    st.plotly_chart(fig, use_container_width=True, key=key)


def _render_paper_explorer_view(
    *,
    runs_dir: Path = RUNS_DIR,
    title: str = "Paper Explorer",
    show_block: bool = False,
    ns: str = "paper_exp",
):
    """Explorer dei run period-level. show_block=True (variante hard, GT legacy)
    aggiunge le metriche di blocco e una sezione per tipo di query."""
    st.title(title)
    metrics = METRICS_HARD if show_block else METRICS
    # Hard: la tabella di selezione mostra solo i compositi Weighted (A→Flat, B→Block);
    # i Macro restano disponibili nell'analisi globale. Il paper resta sulle medie grezze.
    score_cols = ["Recall Weighted", "Rank Weighted"] if show_block else metrics
    sort_col = "Recall Weighted" if show_block else "Recall"

    runs = _list_runs(runs_dir)
    if not runs:
        st.info(f"Nessun run trovato in `{runs_dir}`.")
        return

    # Filtro per dataset (GT): ogni run salva 'gt_reference' nel meta. Si seleziona
    # un dataset così non si confrontano run su GT diverse.
    runs_meta = [(rd, _load_meta(rd)) for rd in runs]
    runs_meta = [(rd, m) for rd, m in runs_meta if m]
    datasets = sorted({Path(m.get("gt_reference", "?")).name for _, m in runs_meta})
    sel_ds = st.selectbox("Dataset", datasets, key=f"{ns}_dataset")
    runs_meta = [(rd, m) for rd, m in runs_meta if Path(m.get("gt_reference", "?")).name == sel_ds]

    include_fonte = True  # fonte normativa sempre inclusa (toggle rimosso: inerte sui dati)
    show_ci = st.toggle("Intervalli di confidenza (bootstrap 95%)", value=False, key=f"{ns}_ci")

    rows = []
    for rd, meta in runs_meta:
        cfg = (meta.get("model_configs") or [{}])[0]
        scores = _run_block_scores(rd, include_fonte) if show_block else _run_means(rd, include_fonte, metrics)
        if scores is None:
            continue
        rows.append({
            "Run": rd.name,
            "Time": _fmt_ts(meta.get("created_at_utc", "")),
            "Status": _run_status(rd),
            **_model_cols(cfg),
            **{c: scores[c] for c in score_cols},
            "#q": scores["#q"],
            "_config": cfg,  # per la legenda/inspection, non mostrato in tabella
        })

    if not rows:
        st.info("Nessun run valido (manca details.csv/meta.json).")
        return

    df = pd.DataFrame(rows).sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    cfg_cols = ["Tipo", "index", "stage1", "stage3", "m", "n", "j", "k", "mapping", "reranker", "splitter", "clf", "llm"]
    show_cols = ["Run", "Time", "Status", "#q"] + cfg_cols + score_cols

    st.markdown(f"**{len(df)} run** — seleziona le righe da confrontare:")
    event = st.dataframe(
        df[show_cols], use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="multi-row", key=f"{ns}_table",
    )
    selected = event.selection.rows
    if not selected:
        st.info("Seleziona almeno un run dalla tabella.")
        st.session_state.pop(f"{ns}_active", None)
        return

    if selected != st.session_state.get(f"{ns}_last_sel"):
        st.session_state.pop(f"{ns}_active", None)

    with st.expander("Config modelli selezionati", expanded=True):
        for i, idx in enumerate(selected, 1):
            st.markdown(f"**M{i}** — {_describe_config(df.iloc[idx]['_config'])}  ·  {df.iloc[idx]['Run']}")
            st.json(df.iloc[idx]["_config"], expanded=False)

    if st.button("Esplora", type="primary", key=f"{ns}_go"):
        st.session_state[f"{ns}_active"] = [df.iloc[idx]["Run"] for idx in selected]
        st.session_state[f"{ns}_last_sel"] = selected

    active_runs = st.session_state.get(f"{ns}_active")
    if not active_runs:
        return

    sub = df[df["Run"].isin(active_runs)].reset_index(drop=True)
    n = len(sub)

    # ── Performance globali ──────────────────────────────────────────────────
    st.divider()
    st.markdown("## Performance globali")
    if show_block:
        # Hard: stile Explorer legacy (Macro/Weighted + heatmap A/B/A+B + per categoria).
        _render_block_analysis(sub, runs_dir, include_fonte, ns, cfg_cols, show_ci=show_ci)
    else:
        global_df = sub.copy()
        global_df.insert(0, "Modello", [f"M{i+1}" for i in range(n)])
        # CI bootstrap delle medie grezze (per metrica e per modello).
        ci_simple: dict = {}
        if show_ci:
            for mname in metrics:
                mq = _merge_per_query(sub, runs_dir, include_fonte, mname)
                if mq is None:
                    continue
                for i in range(n):
                    c = f"M{i+1}"
                    if c in mq.columns:
                        ci_simple[(c, mname)] = _mean_ci(mq[c].to_numpy(), seed=i)
        left, right = st.columns([3, 2], gap="large")
        with left:
            disp = global_df[["Modello"] + cfg_cols + metrics + ["#q"]].copy()
            if ci_simple:
                st.caption(f"Intervalli di confidenza: bootstrap {CI_N_BOOT}× sulle query, "
                           f"{int((1 - CI_ALPHA) * 100)}% percentile.")
                ordered = ["Modello"] + cfg_cols
                for mname in metrics:
                    disp[f"{mname} ±"] = disp["Modello"].map(
                        lambda m, mm=mname: _fmt_ci(*ci_simple[(m, mm)]) if (m, mm) in ci_simple else "")
                    ordered += [mname, f"{mname} ±"]
                disp = disp[ordered + ["#q"]]
            st.dataframe(disp, use_container_width=True, hide_index=True)
            if px is not None:
                melt = global_df.melt(id_vars="Modello", value_vars=metrics, var_name="Metrica", value_name="Score")
                melt["Score"] = pd.to_numeric(melt["Score"], errors="coerce")
                err_kw = {}
                if ci_simple:
                    melt["e_plus"] = melt.apply(
                        lambda r: ci_simple[(r["Modello"], r["Metrica"])][1] - r["Score"]
                        if (r["Modello"], r["Metrica"]) in ci_simple else 0.0, axis=1)
                    melt["e_minus"] = melt.apply(
                        lambda r: r["Score"] - ci_simple[(r["Modello"], r["Metrica"])][0]
                        if (r["Modello"], r["Metrica"]) in ci_simple else 0.0, axis=1)
                    err_kw = {"error_y": "e_plus", "error_y_minus": "e_minus"}
                fig = px.bar(melt, x="Metrica", y="Score", color="Modello", barmode="group", range_y=[0, 1], **err_kw)
                fig.update_layout(height=320, margin={"t": 20, "b": 0, "l": 0, "r": 0})
                st.plotly_chart(fig, use_container_width=True, key=f"{ns}_global_bar")
        with right:
            st.caption("Heatmap riepilogo — modelli × metriche")
            _render_global_heatmap(global_df, metrics, f"{ns}_global_heatmap")
        for i, (_, row) in enumerate(sub.iterrows(), 1):
            st.caption(f"M{i} — {_describe_config(row['_config'])}  ·  {row['Run']}")

    # ── Dettaglio per query: tabella (sx) + heatmap query × modelli (dx) ──────
    st.divider()
    st.markdown("## Dettaglio per query")
    detail_metric = (
        st.selectbox("Metrica (dettaglio per query)", metrics, key=f"{ns}_detail_metric")
        if show_block else "Recall"
    )
    merged = _merge_per_query(sub, runs_dir, include_fonte, detail_metric)
    if merged is None:
        return

    col_table, col_heat = st.columns([3, 2], gap="large")
    with col_table:
        st.caption(f"Clicca una riga per ispezionare i risultati di quella query ({detail_metric} per modello).")
        ev = st.dataframe(
            merged, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key=f"{ns}_query_table",
        )
    with col_heat:
        st.caption(f"Heatmap query × modelli ({detail_metric})")
        qcols = [c for c in merged.columns if c != "Query"]
        _render_query_heatmap(merged, qcols, detail_metric, f"{ns}_qheat")

    sel_rows = ev.selection.rows
    if not sel_rows:
        return
    q = str(merged.iloc[sel_rows[0]]["Query"])

    gt = _load_gt_context(runs_dir / sub.iloc[0]["Run"]).get(q, {})
    if include_fonte:
        gt_keys = set(gt.get("gt_periods", []))
    else:
        gt_keys = {g.split("\x00", 1)[0] for g in gt.get("gt_periods", [])}

    st.markdown(f"### Inspection — *{q}*")
    st.markdown("**GT attesa:**")
    for r in gt.get("raw_results", []):
        st.markdown(f"- `{r}`")

    for i, (_, row) in enumerate(sub.iterrows(), 1):
        rd = runs_dir / row["Run"]
        qd = _load_details(rd)
        qd = qd[(qd["Query"].astype(str) == q) & (qd["Model"] == "M1")].dropna(subset=["Rank"]).sort_values("Rank")
        final_rows = []
        for _, r in qd.iterrows():
            p = str(r["Periodo"])
            f = str(r.get("Fonte Normativa", "") or "")
            key = f"{_norm_str(p)}\x00{_norm_str(f)}" if include_fonte else _norm_str(p)
            final_rows.append({"Rank": int(r["Rank"]), "Match": "✅" if key in gt_keys else "❌",
                               "Periodo": p, "Fonte": f})
        final_df = pd.DataFrame(final_rows)

        st.markdown(f"**M{i}** — {_describe_config(row['_config'])}")
        debug = _load_debug(rd).get(q, {}).get("M1")
        if debug:
            _render_pool_tabs(debug, final_df)
        elif not final_df.empty:
            st.dataframe(final_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Nessun hit per questa query.")


def _render_paper_hard_explorer_view():
    """Variante hard: stessi modelli sui run in runs/paper_hard, con metriche di
    blocco (Block Recall, Block AP) e breakdown per tipo di query."""
    _render_paper_explorer_view(
        runs_dir=Path(PAPER_HARD_RUNS_DIR),
        title="Paper Hard Explorer",
        show_block=True,
        ns="paperhard_exp",
    )
