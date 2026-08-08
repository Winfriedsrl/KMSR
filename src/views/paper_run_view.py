"""Paper Run: builder di modelli sull'indice flat (periodo_content) valutati
sulla GT del paper (periodi sfusi). Riusa evaluate_from_df + save_evaluation_run.

Il flat ha un solo campo di contenuto, quindi il match_field è sempre
'periodo_content'; si configurano solo motore, k, candidati e reranker opzionale.
B1 = Sparse, B2 = Dense, B3 = Hybrid (RRF); altri modelli si ottengono variando
i parametri (es. + reranker).
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from es import ES
from config import (
    LLM_MODEL_OPTIONS, VECTORIZER_OPTIONS, load_key_table,
    PAPER_RUNS_DIR, PAPER_HARD_RUNS_DIR, PAPER_KEY_TABLE,
    DEFAULT_INDEX_PAPER_PERIODI_FULL,
)
from evaluation import (
    build_gt_context,
    build_gt_map,
    build_gt_map_paper,
    evaluate_from_df,
    evaluate_pool_paper,
    save_evaluation_run,
)

ENGINE_OPTIONS = ["Sparse", "Dense", "Hybrid (RRF)"]
STAGE_MODE_OPTIONS = ["dense", "sparse", "hybrid"]
# Famiglie: due baseline (stesso builder, indice cablato diverso) + pool.
FAMILY_FULL = "Flat tutti i periodi"
FAMILY_FLAT = "Flat reati/circ"
FAMILY_POOL = "Pool"
FAMILY_OPTIONS = [FAMILY_FULL, FAMILY_FLAT, FAMILY_POOL]
MATCH_FIELD = "periodo_content"
INFO_UPLOAD = "Carica la GT del paper (colonne: query, reati_estratti)."
INFO_UPLOAD_HARD = "Carica la GT in formato legacy (colonne: query, result, type)."


# ---------------------------------------------------------------------------
# Builder modelli
# ---------------------------------------------------------------------------

def _default_paper_model() -> dict:
    return {
        "engine": "Sparse", "k": 10, "data_points": 100, "rrf_k": 60,
        "rerank": False, "rerank_llm": LLM_MODEL_OPTIONS[0],
        "rerank_rr": 30, "rerank_k": 10,
    }


def _paper_model_editor(cfg: dict, idx: int, key_prefix: str) -> dict:
    r = dict(cfg)
    r["engine"] = st.selectbox(
        "Motore", ENGINE_OPTIONS,
        index=ENGINE_OPTIONS.index(cfg.get("engine", "Sparse")),
        key=f"{key_prefix}_{idx}_engine",
    )

    col_k, col_dp, col_rrf = st.columns(3, gap="small")
    with col_k:
        r["k"] = int(st.number_input(
            "k", min_value=1, max_value=2000, value=int(cfg.get("k", 10)),
            key=f"{key_prefix}_{idx}_k",
        ))
    if r["engine"] in ("Dense", "Hybrid (RRF)"):
        with col_dp:
            r["data_points"] = int(st.number_input(
                "#candidati", min_value=10, max_value=5000,
                value=int(cfg.get("data_points", 100)), key=f"{key_prefix}_{idx}_dp",
            ))
    if r["engine"] == "Hybrid (RRF)":
        with col_rrf:
            r["rrf_k"] = int(st.number_input(
                "RRF k", min_value=1, max_value=500, value=int(cfg.get("rrf_k", 60)),
                key=f"{key_prefix}_{idx}_rrfk",
            ))

    st.divider()
    r["rerank"] = st.checkbox(
        "Applica LLM reranker", value=bool(cfg.get("rerank", False)),
        key=f"{key_prefix}_{idx}_rr",
    )
    if r["rerank"]:
        col_llm, col_rr, col_rk = st.columns([3, 1, 1], gap="small")
        with col_llm:
            r["rerank_llm"] = st.selectbox(
                "LLM", LLM_MODEL_OPTIONS,
                index=LLM_MODEL_OPTIONS.index(cfg.get("rerank_llm", LLM_MODEL_OPTIONS[0])),
                key=f"{key_prefix}_{idx}_rrllm",
            )
        with col_rr:
            r["rerank_rr"] = int(st.number_input(
                "rr", min_value=1, max_value=2000, value=int(cfg.get("rerank_rr", 30)),
                key=f"{key_prefix}_{idx}_rrrr",
            ))
        with col_rk:
            r["rerank_k"] = int(st.number_input(
                "k finale", min_value=1, max_value=2000, value=int(cfg.get("rerank_k", 10)),
                key=f"{key_prefix}_{idx}_rrk",
            ))
    return r


def _paper_cfg_to_model_config(cfg: dict) -> dict:
    """Converte lo stato UI nel ModelConfig atteso da run_model/evaluate_from_df.

    Passa solo i campi che il motore usa davvero: lo Sparse non ha embedder, e
    l'ibrido tiene vectorizer/candidati nei sub-retriever, non al top-level.
    """
    vec = VECTORIZER_OPTIONS[0]
    k = int(cfg.get("k", 10))
    dp = int(cfg.get("data_points", 100))
    engine = cfg.get("engine", "Sparse")

    if engine == "Sparse":
        mc = {"model_type": "Sparse search", "match_field": MATCH_FIELD, "k": k}
    elif engine == "Dense":
        mc = {"model_type": "Dense search", "match_field": MATCH_FIELD,
              "vectorizer": vec, "k": k, "data_points": dp}
    else:  # Hybrid (RRF): sparse + dense sullo stesso campo
        mc = {"model_type": "Fusion (RRF)", "match_field": MATCH_FIELD, "k": k,
              "rrf_config": {
                  "rrf_k": int(cfg.get("rrf_k", 60)),
                  "window_size": max(k * 5, 100),
                  "retriever_a": {"kind": "Sparse search", "match_field": MATCH_FIELD},
                  "retriever_b": {"kind": "Dense search", "match_field": MATCH_FIELD, "vectorizer": vec},
              }}

    if cfg.get("rerank"):
        mc["reranker_config"] = {
            "llm_model": cfg.get("rerank_llm", LLM_MODEL_OPTIONS[0]),
            "reranker_results": int(cfg.get("rerank_rr", 30)),
            "k": int(cfg.get("rerank_k", 10)),
        }
    return mc


def _model_title(idx: int, cfg: dict) -> str:
    parts = [f"Modello {idx + 1}", cfg.get("engine", "?"), f"k={cfg.get('k', '?')}"]
    if cfg.get("rerank"):
        parts.append(f"+ rerank({cfg.get('rerank_llm', '?')})")
    return " · ".join(parts)


def _render_paper_models(state_key: str, key_prefix: str) -> list[dict]:
    if state_key not in st.session_state:
        st.session_state[state_key] = [_default_paper_model()]

    current = st.session_state[state_key]
    updated = []

    st.header("Definisci modelli (indice flat · periodo_content)")
    for idx, cfg in enumerate(current):
        col_exp, col_rm = st.columns([30, 1], gap="small")
        with col_exp:
            expander = st.expander(_model_title(idx, cfg), expanded=(idx == 0))
        with col_rm:
            if st.button("🗑️", key=f"{key_prefix}_{idx}_remove", help="Rimuovi"):
                if len(current) > 1:
                    st.session_state[state_key].pop(idx)
                    st.rerun()
                else:
                    st.warning("Deve rimanere almeno un modello.")
        with expander:
            updated.append(_paper_model_editor(cfg, idx, key_prefix))

    st.button("+ aggiungi modello", key=f"{key_prefix}_add",
              on_click=lambda: st.session_state[state_key].append(_default_paper_model()))
    st.session_state[state_key] = updated

    return [_paper_cfg_to_model_config(c) for c in updated]


# ---------------------------------------------------------------------------
# Builder pool (M)
# ---------------------------------------------------------------------------

def _default_pool_model() -> dict:
    return {
        "stage1_mode": "dense", "stage3_mode": "dense", "m": 10, "n": 10, "j": 10, "k": 10,
        "llm_model": LLM_MODEL_OPTIONS[0], "classifier_version": "v1", "rewriting": True,
        "period_rerank": False, "mapping": True,
    }


def _pool_model_editor(cfg: dict, idx: int, key_prefix: str) -> dict:
    r = dict(cfg)
    col_s1, col_s3 = st.columns(2, gap="small")
    with col_s1:
        r["stage1_mode"] = st.selectbox(
            "Stage1 (reati)", STAGE_MODE_OPTIONS,
            index=STAGE_MODE_OPTIONS.index(cfg.get("stage1_mode", "dense")),
            key=f"{key_prefix}_{idx}_s1",
        )
    with col_s3:
        r["stage3_mode"] = st.selectbox(
            "Stage3 (circ)", STAGE_MODE_OPTIONS,
            index=STAGE_MODE_OPTIONS.index(cfg.get("stage3_mode", "dense")),
            key=f"{key_prefix}_{idx}_s3",
        )

    col_m, col_n, col_j, col_k = st.columns(4, gap="small")
    with col_m:
        r["m"] = int(st.number_input("m (reati)", min_value=1, max_value=200,
                                     value=int(cfg.get("m", 10)), key=f"{key_prefix}_{idx}_m"))
    with col_n:
        r["n"] = int(st.number_input("n (circ/reato)", min_value=1, max_value=200,
                                     value=int(cfg.get("n", 10)), key=f"{key_prefix}_{idx}_n"))
    with col_j:
        r["j"] = int(st.number_input("j (combo)", min_value=1, max_value=500,
                                     value=int(cfg.get("j", 10)), key=f"{key_prefix}_{idx}_j",
                                     help="combo recuperate dall'indice, poi sciolte in periodi"))
    with col_k:
        r["k"] = int(st.number_input("k (periodi)", min_value=1, max_value=500,
                                     value=int(cfg.get("k", 10)), key=f"{key_prefix}_{idx}_k",
                                     help="periodi finali considerati dopo lo scioglimento"))

    col_llm, col_clf = st.columns([2, 1], gap="small")
    with col_llm:
        r["llm_model"] = st.selectbox(
            "LLM (classifier)", LLM_MODEL_OPTIONS,
            index=LLM_MODEL_OPTIONS.index(cfg.get("llm_model", LLM_MODEL_OPTIONS[0])),
            key=f"{key_prefix}_{idx}_llm",
        )
    with col_clf:
        r["classifier_version"] = st.text_input(
            "classifier ver.", value=cfg.get("classifier_version", "v1"),
            key=f"{key_prefix}_{idx}_clf",
        )

    r["rewriting"] = st.checkbox(
        "Query rewriting (splitter)", value=bool(cfg.get("rewriting", True)),
        help="On = M; Off = ablation A2 (-rewriting)", key=f"{key_prefix}_{idx}_rw",
    )
    r["mapping"] = st.checkbox(
        "Mapping reato→circostanze (key_table)", value=bool(cfg.get("mapping", True)),
        help="Off = ablation: stage3 fa una ricerca globale sui combo, senza il mapping di dominio",
        key=f"{key_prefix}_{idx}_map",
    )
    r["period_rerank"] = st.checkbox(
        "Reranker LLM sui periodi sciolti", value=bool(cfg.get("period_rerank", False)),
        help="Variante B: dopo lo scioglimento riordina i periodi col reranker (content dal flat) prima del taglio a k",
        key=f"{key_prefix}_{idx}_prr",
    )
    return r


def _pool_cfg_to_model_config(cfg: dict) -> dict:
    rw = cfg.get("rewriting", True)
    mc = {
        "pipeline_type": "pool",
        "stage1_mode": cfg.get("stage1_mode", "dense"),
        "stage3_mode": cfg.get("stage3_mode", "dense"),
        "embedder": VECTORIZER_OPTIONS[0],
        "m": int(cfg.get("m", 10)),
        "n": int(cfg.get("n", 10)),
        "j": int(cfg.get("j", 10)),  # combo da recuperare
        "k": int(cfg.get("k", 10)),  # periodi finali dopo lo scioglimento
        "llm_model": cfg.get("llm_model", LLM_MODEL_OPTIONS[0]),
        "classifier_version": cfg.get("classifier_version", "v1"),
        "stage1_input": "q_reato" if rw else "q",
        "stage3_input": "q" if rw else "q",
        "split_method": "regex",
    }
    if cfg.get("period_rerank"):
        mc["period_rerank"] = True   # il reranker usa lo stesso llm_model della pipeline
    if not cfg.get("mapping", True):
        mc["no_mapping"] = True
    return mc


def _pool_title(idx: int, cfg: dict) -> str:
    rw = "rewrite" if cfg.get("rewriting", True) else "no-rewrite"
    prr = " · +period-rerank" if cfg.get("period_rerank") else ""
    nomap = " · no-map" if not cfg.get("mapping", True) else ""
    return (f"Modello {idx + 1} · Pool · s1={cfg.get('stage1_mode')} s3={cfg.get('stage3_mode')}"
            f" · j={cfg.get('j')} k={cfg.get('k')} · {rw}{prr}{nomap}")


def _render_pool_models(state_key: str, key_prefix: str) -> list[dict]:
    if state_key not in st.session_state:
        st.session_state[state_key] = [_default_pool_model()]

    current = st.session_state[state_key]
    updated = []

    st.header("Definisci modelli pool (reati + circ · key_table)")
    for idx, cfg in enumerate(current):
        col_exp, col_rm = st.columns([30, 1], gap="small")
        with col_exp:
            expander = st.expander(_pool_title(idx, cfg), expanded=(idx == 0))
        with col_rm:
            if st.button("🗑️", key=f"{key_prefix}_{idx}_remove", help="Rimuovi"):
                if len(current) > 1:
                    st.session_state[state_key].pop(idx)
                    st.rerun()
                else:
                    st.warning("Deve rimanere almeno un modello.")
        with expander:
            updated.append(_pool_model_editor(cfg, idx, key_prefix))

    st.button("+ aggiungi modello", key=f"{key_prefix}_add",
              on_click=lambda: st.session_state[state_key].append(_default_pool_model()))
    st.session_state[state_key] = updated

    return [_pool_cfg_to_model_config(c) for c in updated]


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

def _render_paper_run_view(
    es_client: ES,
    paper_index_names: dict[str, str],
    *,
    ns: str = "paper",
    runs_dir: str = PAPER_RUNS_DIR,
    gt_builder=build_gt_map_paper,
    info_upload: str = INFO_UPLOAD,
):
    """Builder + coda di esecuzione. ns/runs_dir/gt_builder permettono di riusare
    la stessa view per la variante "hard" (GT legacy con blocchi/tipi)."""
    if not paper_index_names.get("periodi_flat"):
        st.error("Indice flat non configurato (ES_INDEX_PAPER_PERIODI_FLAT).")
        return

    family = st.radio("Famiglia", FAMILY_OPTIONS, horizontal=True, key=f"{ns}_family")
    if family == FAMILY_POOL:
        models = _render_pool_models(f"{ns}_pool_models", f"{ns}pool")
    else:
        # Due baseline con stesso builder: cambia solo l'indice (corpus), cablato qui.
        base_index = (DEFAULT_INDEX_PAPER_PERIODI_FULL if family == FAMILY_FULL
                      else paper_index_names["periodi_flat"])
        slug = "full" if family == FAMILY_FULL else "flat"
        models = _render_paper_models(f"{ns}_models_{slug}", f"{ns}{slug}")
        for mc in models:
            mc["index"] = base_index

    st.divider()
    csv_file = st.file_uploader("Carica GT", type=["csv"], key=f"{ns}_uploader")
    if csv_file is None:
        st.info(info_upload)
        return

    if st.button("Lancia", type="primary", key=f"{ns}_run"):
        df = pd.read_csv(csv_file)
        st.session_state[f"{ns}_queue"] = [
            {"config": cfg, "status": "pending", "run_dir": None} for cfg in models
        ]
        st.session_state[f"{ns}_queue_df"] = df
        st.session_state[f"{ns}_queue_csv_name"] = csv_file.name
        st.session_state[f"{ns}_queue_family"] = family
        st.rerun()

    queue: list[dict] = st.session_state.get(f"{ns}_queue", [])
    if not queue:
        return

    df = st.session_state.get(f"{ns}_queue_df")
    csv_name = st.session_state.get(f"{ns}_queue_csv_name", "")
    next_pending = next((i for i, it in enumerate(queue) if it["status"] == "pending"), None)

    queue_family = st.session_state.get(f"{ns}_queue_family", FAMILY_OPTIONS[0])

    progress_bar = None
    for i, item in enumerate(queue):
        kind = item["config"].get("model_type") or item["config"].get("pipeline_type", "")
        label = f"Modello {i + 1} · {kind}"
        if item["status"] == "done":
            run_name = Path(item["run_dir"]).name if item["run_dir"] else ""
            st.success(f"{label} — completato · run: {run_name}")
        elif item["status"] == "error":
            st.error(f"{label} — errore: {item.get('error', '')}")
        elif i == next_pending:
            progress_bar = st.progress(0, text=f"{label} — avvio...")
        else:
            st.info(f"{label} — in attesa")

    if next_pending is not None and progress_bar is not None:
        i = next_pending
        cfg = queue[i]["config"]
        label = f"Modello {i + 1}"

        def on_step(step, total, q_idx, m_idx, _bar=progress_bar, _l=label):
            _bar.progress(step / total, text=f"{_l}: {step}/{total} query")

        try:
            gt_map = gt_builder(df)
            debug_data = None
            if queue_family != FAMILY_POOL:
                details_df, details_by_query = evaluate_from_df(
                    df, [cfg], es_client, paper_index_names, on_step=on_step, gt_map=gt_map,
                )
            else:
                details_df, details_by_query, debug_data = evaluate_pool_paper(
                    df, [cfg], es_client,
                    paper_index_names["reati"], paper_index_names["reati_circ_light"],
                    load_key_table(PAPER_KEY_TABLE), on_step=on_step, gt_map=gt_map,
                    flat_index=paper_index_names["periodi_flat"],
                )
            save_info = save_evaluation_run(
                details_df=details_df,
                model_configs=[cfg],
                gt_reference=csv_name,
                base_dir=runs_dir,
                gt_context=build_gt_context(details_by_query),
            )
            if debug_data:
                (Path(save_info["run_dir"]) / "debug.json").write_text(
                    json.dumps(debug_data, ensure_ascii=False, indent=2)
                )
            queue[i]["status"] = "done"
            queue[i]["run_dir"] = save_info["run_dir"]
        except Exception as exc:
            queue[i]["status"] = "error"
            queue[i]["error"] = str(exc)

        st.session_state[f"{ns}_queue"] = queue
        st.rerun()
    else:
        n_done = sum(1 for it in queue if it["status"] == "done")
        n_err = sum(1 for it in queue if it["status"] == "error")
        if n_err == 0:
            st.success(f"Tutti i {n_done} modelli completati. Vai all'Explorer per i risultati.")
        else:
            st.warning(f"{n_done} completati, {n_err} con errore.")


def _render_paper_hard_run_view(es_client: ES, paper_index_names: dict[str, str]):
    """Stessi modelli di Paper Run, ma valutati sulla GT legacy (query/result/type):
    build_gt_map fornisce blocchi e tipi, salvati in runs/paper_hard."""
    _render_paper_run_view(
        es_client, paper_index_names,
        ns="paperhard", runs_dir=PAPER_HARD_RUNS_DIR,
        gt_builder=build_gt_map, info_upload=INFO_UPLOAD_HARD,
    )
