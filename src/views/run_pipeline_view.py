from pathlib import Path

import pandas as pd
import streamlit as st
from es import ES
from config import LLM_MODEL_OPTIONS
import evaluation_pipeline

PIPELINE_OPTIONS = ["pool", "branching_mapping", "branching_direct"]
PIPELINE_LABELS = {
    "pool": "Pipeline Pool",
    "branching_mapping": "Branching con Mapping",
    "branching_direct": "Branching Diretto",
}

# Quale parte della query usare per ogni retrieval (query reduction).
INPUT_OPTIONS = ["q", "q_reato", "q_circostanza"]
INPUT_LABELS = {"q": "q (intera)", "q_reato": "q_reato", "q_circostanza": "q_circostanza"}
SPLIT_OPTIONS = ["regex", "llm"]


def _job_label(cfg: dict, idx: int) -> str:
    pt = cfg.get("pipeline_type", "pool")
    s1 = cfg.get("stage1_mode", "dense")[0]
    s3 = cfg.get("stage3_mode", "dense")[0]
    if pt == "pool":
        clf_v = cfg.get("classifier_version", "v1")
    else:
        clf_v = cfg.get("only_classifier_version", "v1")
    return (
        f"M{idx} · {PIPELINE_LABELS[pt]} · s1={s1} s3={s3}"
        f" · m={cfg.get('m')} n={cfg.get('n')} k={cfg.get('k')}"
        f" · {cfg.get('index_circ', '?')} · {cfg.get('llm_model', '?')} · clf={clf_v}"
    )


def _available_classifier_versions() -> list[str]:
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    if not prompts_dir.exists():
        return ["v1"]
    versions = sorted(
        p.stem.replace("classifier_", "")
        for p in prompts_dir.glob("classifier_*.txt")
    )
    return versions or ["v1"]


def _available_only_classifier_versions() -> list[str]:
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    if not prompts_dir.exists():
        return ["v1"]
    versions = sorted(
        p.stem.replace("only_classifier_", "")
        for p in prompts_dir.glob("only_classifier_*.txt")
    )
    return versions or ["v1"]


def _available_splitter_versions() -> list[str]:
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    if not prompts_dir.exists():
        return ["v1"]
    versions = sorted(
        p.stem.replace("splitter_", "")
        for p in prompts_dir.glob("splitter_*.txt")
    )
    return versions or ["v1"]



def _render_pipeline_config_section(session_key: str) -> list[dict]:
    st.markdown("### Configurazione modelli pipeline")

    if session_key not in st.session_state:
        st.session_state[session_key] = [_default_config()]

    configs: list[dict] = st.session_state[session_key]

    for i, cfg in enumerate(configs):
        with st.expander(f"Modello {i + 1}", expanded=True):

            # ── tipo pipeline ──────────────────────────────────────────────
            pt_idx = PIPELINE_OPTIONS.index(cfg.get("pipeline_type", "pool"))
            cfg["pipeline_type"] = st.selectbox(
                "Tipo pipeline",
                PIPELINE_OPTIONS,
                index=pt_idx,
                format_func=lambda x: PIPELINE_LABELS[x],
                key=f"{session_key}_pt_{i}",
            )
            pipeline_type = cfg["pipeline_type"]

            # ── parametri m / n / k ────────────────────────────────────────
            col_m, col_n, col_k = st.columns(3)
            if pipeline_type == "branching_direct":
                cfg["m"] = col_m.number_input("m — top-k reati (branch reato)", min_value=1, max_value=100,
                                               value=cfg.get("m", 10), key=f"{session_key}_m_{i}")
                cfg["n"] = col_n.number_input("n — top-k circ diretti (branch circ)", min_value=1, max_value=100,
                                               value=cfg.get("n", 10), key=f"{session_key}_n_{i}")
            elif pipeline_type == "branching_mapping":
                cfg["m"] = col_m.number_input("m — top-k reati base", min_value=1, max_value=100,
                                               value=cfg.get("m", 10), key=f"{session_key}_m_{i}")
                cfg["n"] = col_n.number_input("n — top-k circ per reato (branch circ)", min_value=1, max_value=100,
                                               value=cfg.get("n", 10), key=f"{session_key}_n_{i}")
            else:
                cfg["m"] = col_m.number_input("m — top-k reati base", min_value=1, max_value=100,
                                               value=cfg.get("m", 10), key=f"{session_key}_m_{i}")
                cfg["n"] = col_n.number_input("n — top-k circostanziati per reato", min_value=1, max_value=100,
                                               value=cfg.get("n", 10), key=f"{session_key}_n_{i}")

            cfg["k"] = col_k.number_input("k — risultati finali", min_value=1, max_value=100,
                                           value=cfg.get("k", 10), key=f"{session_key}_k_{i}")

            # ── retrieval ──────────────────────────────────────────────────
            col_s1, col_s3, col_llm = st.columns(3)

            with col_s1:
                st.markdown("**Retrieval reati base**")
                st.caption("`wf_reati`")
                if pipeline_type == "branching_direct":
                    st.caption("_(usato nel branch reato)_")
                cfg["stage1_mode"] = st.radio("Metodo", ["dense", "sparse"], horizontal=True,
                                               key=f"{session_key}_s1_{i}")
                cfg["stage1_input"] = st.selectbox(
                    "Input", INPUT_OPTIONS,
                    index=INPUT_OPTIONS.index(cfg.get("stage1_input", "q")),
                    format_func=lambda x: INPUT_LABELS[x], key=f"{session_key}_s1in_{i}",
                )

            with col_s3:
                st.markdown("**Retrieval circostanziati**")
                st.caption("`wf_reati_circ`")
                if pipeline_type == "branching_mapping":
                    st.caption("_(filtrato via key_table)_")
                elif pipeline_type == "branching_direct":
                    st.caption("_(retriever diretto, no filtro)_")
                cfg["stage3_mode"] = st.radio("Metodo", ["dense", "sparse"], horizontal=True,
                                               key=f"{session_key}_s3_{i}")
                cfg["stage3_input"] = st.selectbox(
                    "Input", INPUT_OPTIONS,
                    index=INPUT_OPTIONS.index(cfg.get("stage3_input", "q")),
                    format_func=lambda x: INPUT_LABELS[x], key=f"{session_key}_s3in_{i}",
                )
                cfg["index_circ"] = st.radio("Variante", ["light", "full"], horizontal=True,
                                              key=f"{session_key}_circ_{i}")

            # ── LLM ───────────────────────────────────────────────────────
            with col_llm:
                cfg["llm_model"] = st.selectbox(
                    "Modello LLM",
                    LLM_MODEL_OPTIONS,
                    index=LLM_MODEL_OPTIONS.index(cfg.get("llm_model", LLM_MODEL_OPTIONS[0])) if cfg.get("llm_model") in LLM_MODEL_OPTIONS else 0,
                    key=f"{session_key}_llm_{i}",
                )

                if pipeline_type == "pool":
                    st.markdown("**Classificatore + Reranker**")
                    st.caption("classifica il tipo e seleziona i top-k in una chiamata")
                    versions = _available_classifier_versions()
                    current_v = cfg.get("classifier_version", "v1")
                    default_idx = versions.index(current_v) if current_v in versions else 0
                    cfg["classifier_version"] = st.selectbox(
                        "Versione prompt", versions, index=default_idx,
                        key=f"{session_key}_clf_{i}",
                    )
                else:
                    st.markdown("**Classificatore iniziale**")
                    st.caption("classifica il tipo prima del retrieval")
                    only_versions = _available_only_classifier_versions()
                    current_ov = cfg.get("only_classifier_version", "v1")
                    default_oidx = only_versions.index(current_ov) if current_ov in only_versions else 0
                    cfg["only_classifier_version"] = st.selectbox(
                        "Versione prompt", only_versions, index=default_oidx,
                        key=f"{session_key}_oclf_{i}",
                    )

                    use_ranker = st.checkbox(
                        "Abilita reranker LLM (opzionale)",
                        value=bool(cfg.get("use_ranker")),
                        key=f"{session_key}_use_rk_{i}",
                    )
                    cfg["use_ranker"] = use_ranker

            # ── tecnica di split (model-level) ─────────────────────────────
            # Serve solo se almeno un input non è la query intera.
            needs_split = cfg.get("stage1_input", "q") != "q" or cfg.get("stage3_input", "q") != "q"
            cfg["split_method"] = st.selectbox(
                "Tecnica split query", SPLIT_OPTIONS,
                index=SPLIT_OPTIONS.index(cfg.get("split_method", "regex")),
                disabled=not needs_split, key=f"{session_key}_split_{i}",
                help="Usata solo se un Input ≠ q. regex: gratis ~80%. llm: copre tutto, 1 chiamata.",
            )
            # Versione prompt: rilevante solo per lo splitter LLM.
            if needs_split and cfg["split_method"] == "llm":
                split_versions = _available_splitter_versions()
                current_sv = cfg.get("split_version", "v1")
                default_svidx = split_versions.index(current_sv) if current_sv in split_versions else 0
                cfg["split_version"] = st.selectbox(
                    "Versione prompt splitter", split_versions, index=default_svidx,
                    key=f"{session_key}_splitv_{i}",
                )

            # ── embedder ──────────────────────────────────────────────────
            needs_embedder = (
                cfg.get("stage1_mode") == "dense"
                or cfg.get("stage3_mode") == "dense"
            )
            if needs_embedder:
                cfg["embedder"] = st.selectbox(
                    "Embedder", ["text-embedding-3-large"],
                    key=f"{session_key}_emb_{i}",
                )
            else:
                cfg["embedder"] = "text-embedding-3-large"

        if len(configs) > 1:
            if st.button(f"Rimuovi M{i + 1}", key=f"{session_key}_remove_{i}"):
                configs.pop(i)
                st.session_state[session_key] = configs
                st.rerun()

    if st.button("+ Aggiungi modello", key=f"{session_key}_add"):
        configs.append(_default_config())
        st.session_state[session_key] = configs
        st.rerun()

    st.session_state[session_key] = configs
    return configs


def _default_config() -> dict:
    return {
        "pipeline_type": "pool",
        "m": 10, "n": 10, "k": 10,
        "stage1_mode": "dense",
        "stage3_mode": "dense",
        "embedder": "text-embedding-3-large",
        "llm_model": "gpt-4o-mini",
        "index_circ": "light",
        "classifier_version": "v1",
        "only_classifier_version": "v1",
        "use_ranker": False,
        "stage1_input": "q",
        "stage3_input": "q",
        "split_method": "regex",
        "split_version": "v1",
    }


def _render_run_pipeline_view(
    es_client: ES,
    index_reati: str,
    index_circ_light: str,
    index_circ_full: str,
    key_table: dict,
):
    st.title("Run Reati")

    if not index_reati:
        st.error("ES_INDEX_NAME_REATI non configurato nel .env")
        return

    configs = _render_pipeline_config_section("pipeline_models")
    st.divider()

    csv_file = st.file_uploader("Upload CSV GT (wf_ground_truth.csv)", type=["csv"])
    if csv_file is None:
        st.info("Carica un CSV nel formato wf_ground_truth.csv (colonne: query, combo_reati, combo_reati_circostanziati).")
        return

    max_queries = st.number_input(
        "Max query da testare (0 = tutte)",
        min_value=0, max_value=10000, value=0, step=10,
        key="pipeline_max_queries",
    )

    if st.button("Lancia", type="primary", key="pipeline_run"):
        df = pd.read_csv(csv_file)
        if max_queries > 0:
            unique_queries = df["query"].dropna().unique()[:max_queries]
            df = df[df["query"].isin(unique_queries)]
        st.session_state["pipeline_queue"] = [
            {"config": cfg, "status": "pending", "run_dir": None}
            for cfg in configs
        ]
        st.session_state["pipeline_queue_df"] = df
        st.session_state["pipeline_queue_csv_name"] = csv_file.name
        st.rerun()

    queue: list[dict] = st.session_state.get("pipeline_queue", [])
    if not queue:
        return

    df = st.session_state.get("pipeline_queue_df")
    csv_name = st.session_state.get("pipeline_queue_csv_name", "")

    next_pending_idx = next(
        (i for i, item in enumerate(queue) if item["status"] == "pending"), None
    )

    progress_bar = None
    for i, item in enumerate(queue):
        label = _job_label(item["config"], i + 1)
        if item["status"] == "done":
            run_name = Path(item["run_dir"]).name if item["run_dir"] else ""
            st.success(f"{label} — completato · run: {run_name}")
        elif item["status"] == "error":
            st.error(f"{label} — errore: {item.get('error', '')}")
        elif i == next_pending_idx:
            progress_bar = st.progress(0, text=f"{label} — avvio...")
        else:
            st.info(f"{label} — in attesa")

    if next_pending_idx is not None and progress_bar is not None:
        i = next_pending_idx
        cfg = queue[i]["config"]
        label = _job_label(cfg, i + 1)
        index_circ = index_circ_light if cfg.get("index_circ") == "light" else index_circ_full

        def on_step(step, total, q_idx, m_idx, query="", _label=label, _bar=progress_bar):
            _bar.progress(step / total, text=f"{_label}: {step}/{total} query")

        try:
            details_df, gt_context, debug_data = evaluation_pipeline.evaluate_from_df(
                df, [cfg], es_client, index_reati, index_circ, key_table, on_step=on_step,
            )
            run_dir = evaluation_pipeline.save_run(
                details_df=details_df,
                model_configs=[cfg],
                gt_reference=csv_name,
                gt_context=gt_context,
                debug_data=debug_data,
            )
            queue[i]["status"] = "done"
            queue[i]["run_dir"] = run_dir
        except Exception as exc:
            queue[i]["status"] = "error"
            queue[i]["error"] = str(exc)

        st.session_state["pipeline_queue"] = queue
        st.rerun()
    else:
        n_done = sum(1 for item in queue if item["status"] == "done")
        n_err = sum(1 for item in queue if item["status"] == "error")
        if n_err == 0:
            st.success(f"Tutti i {n_done} modelli completati. Vai a **Explorer Reati** per i risultati.")
        else:
            st.warning(f"{n_done} completati, {n_err} con errore.")
