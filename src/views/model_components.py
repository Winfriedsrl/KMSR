from typing import Any

import streamlit as st
from es import ES
import json
from config import (
    LLM_MODEL_OPTIONS,
    MATCH_FIELD_OPTIONS,
    VECTORIZER_OPTIONS,
    clean_model_config,
    ModelConfig,
    resolve_effective_vectorizer,
    resolve_field_profile,
)
from runner import run_model
from plotting import build_hover_texts, build_plot, build_results_table

RETRIEVAL_MODE_OPTIONS = ["Single retriever", "Fusion (RRF)"]
SINGLE_RETRIEVER_TYPE_OPTIONS = ["Sparse search", "Dense search", "Sparse search (weighted)"]
RRF_RETRIEVER_KIND_OPTIONS = ["Dense search", "Sparse search", "Sparse search (weighted)"]

UNSUPPORTED_MODEL_TYPES: set[str] = set()

ERROR_MIN_ONE_MODEL = "Deve rimanere almeno un modello."
INFO_AUTO_VECTORIZER = (
    "Per 'rubriche_con_periodi' uso automaticamente "
    "'text-embedding-3-large' nelle modalità dense."
)
INFO_NO_RESULTS = "Nessun risultato."
INFO_NO_EMBEDDINGS_PLOT = "Nessun embedding per il grafico 3D."


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def _default_model() -> dict:
    """
    Restituisce la configurazione di default per un nuovo modello.
    Il dict contiene sempre entrambi i sotto-dict 'single' e 'fusion_rrf',
    così quando l'utente cambia modalità i valori precedenti sono preservati.
    """
    return {
        "retrieval_mode": "Single retriever",
        "single": {
            "retriever_type": "Sparse search",
            "match_field": "testo",
            "k": 10,
            "vectorizer": VECTORIZER_OPTIONS[0],
            "data_points": 100,
            "field_1": "rubrica",
            "boost_1": 2.0,
            "field_2": "testo",
            "boost_2": 1.0,
            "reranker_enabled": False,
            "reranker_llm_model": LLM_MODEL_OPTIONS[0],
            "reranker_results": 30,
            "reranker_k": 10,
        },
        "fusion_rrf": {
            "rrf_k": 60,
            "k": 10,
            "retriever_a": {
                "kind": "Dense search",
                "vectorizer": VECTORIZER_OPTIONS[0],
                "match_field": "testo",
                "field_1": "rubrica",
                "boost_1": 2.0,
                "field_2": "testo",
                "boost_2": 1.0,
            },
            "retriever_b": {
                "kind": "Sparse search",
                "match_field": "rubrica",
                "field_1": "rubrica",
                "boost_1": 2.0,
                "field_2": "testo",
                "boost_2": 1.0,
            },
            "reranker_enabled": False,
            "reranker_llm_model": LLM_MODEL_OPTIONS[0],
            "reranker_results": 30,
            "reranker_k": 10,
        },
    }


# ---------------------------------------------------------------------------
# Model title (mostrato nell'expander)
# ---------------------------------------------------------------------------

def _rrf_retriever_short_desc(retriever: dict) -> str:
    """Descrizione compatta di un sub-retriever RRF, usata nel titolo dell'expander."""
    kind = retriever.get("kind", "?")
    if kind == "Dense search":
        return f"Dense:{retriever.get('match_field', '?')}"
    if kind == "Sparse search":
        return f"Sparse:{retriever.get('match_field', '?')}"
    f1 = retriever.get("field_1", "?")
    w1 = retriever.get("boost_1", "?")
    f2 = retriever.get("field_2", "?")
    w2 = retriever.get("boost_2", "?")
    return f"Sparse(w):{f1}×{w1}+{f2}×{w2}"


def _reranker_suffix(state: dict) -> str:
    """Restituisce " + rerank(model)" se il reranker è abilitato, altrimenti stringa vuota."""
    if state.get("reranker_enabled", False):
        llm = state.get("reranker_llm_model", "?")
        rk = state.get("reranker_k", "?")
        return f" + rerank({llm} · k={rk})"
    return ""


def _model_title(idx: int, cfg: dict) -> str:
    """Restituisce la stringa di intestazione mostrata nell'expander del modello."""
    mode = cfg.get("retrieval_mode", "Single retriever")

    if mode == "Fusion (RRF)":
        fx = cfg.get("fusion_rrf", {})
        k = fx.get("k", "?")
        a_desc = _rrf_retriever_short_desc(fx.get("retriever_a", {}))
        b_desc = _rrf_retriever_short_desc(fx.get("retriever_b", {}))
        base = f"Modello {idx + 1}: Fusion(RRF) [{a_desc} + {b_desc}] · k={k}"
        return base + _reranker_suffix(fx)

    single = cfg.get("single", {})
    rtype = single.get("retriever_type", "Sparse search")
    field = single.get("match_field", "?")
    k = single.get("k", "?")

    if rtype == "Sparse search":
        base = f"Modello {idx + 1}: Sparse · {field} · k={k}"
    elif rtype == "Dense search":
        vec = single.get("vectorizer", "?")
        base = f"Modello {idx + 1}: Dense / {vec} · {field} · k={k}"
    else:  # Sparse search (weighted)
        f1 = single.get("field_1", "?")
        w1 = single.get("boost_1", "?")
        f2 = single.get("field_2", "?")
        w2 = single.get("boost_2", "?")
        base = f"Modello {idx + 1}: Sparse(weighted) [{f1}×{w1} + {f2}×{w2}] · k={k}"
    return base + _reranker_suffix(single)


# ---------------------------------------------------------------------------
# Config → ModelConfig
# ---------------------------------------------------------------------------

def _build_reranker_config(state: dict) -> dict | None:
    """
    Costruisce il reranker_config da includere nel ModelConfig se il reranker è abilitato.
    Restituisce None se il reranker non è attivo.
    """
    if not state.get("reranker_enabled", False):
        return None
    return {
        "llm_model": state.get("reranker_llm_model", LLM_MODEL_OPTIONS[0]),
        "reranker_results": state.get("reranker_results", 30),
        "k": state.get("reranker_k", 10),
    }


def _cfg_to_model_config(cfg: dict) -> ModelConfig:
    """
    Converte il dizionario della configurazione nel TypedDict ModelConfig
    atteso da run_model() e evaluate_from_df().

    Se il reranker è abilitato, aggiunge reranker_config come chiave extra.
    Il k top-level diventa il k del reranker (quello usato per le metriche e la tabella).
    """
    mode = cfg.get("retrieval_mode", "Single retriever")

    if mode == "Fusion (RRF)":
        fx = cfg.get("fusion_rrf", {})
        reranker_config = _build_reranker_config(fx)
        k = int(fx.get("k", 10))
        window_size = max(int(fx.get("k", 10)) * 5, 100)
        result = {
            "model_type": "Fusion (RRF)",
            "vectorizer": VECTORIZER_OPTIONS[0],
            "match_field": "testo",
            "k": k,
            "data_points": 100,
            "retriever_results": 30,
            "llm_model": LLM_MODEL_OPTIONS[0],
            "rrf_config": {
                "rrf_k": fx.get("rrf_k", 60),
                "window_size": window_size,
                "retriever_a": fx.get("retriever_a", {}),
                "retriever_b": fx.get("retriever_b", {}),
            },
        }
        if reranker_config:
            result["reranker_config"] = reranker_config
        return result

    single = cfg.get("single", {})
    retriever_type = single.get("retriever_type", "Sparse search")
    reranker_config = _build_reranker_config(single)
    k = int(single.get("k", 10))

    if retriever_type == "Sparse search (weighted)":
        result = {
            "model_type": "Sparse search (weighted)",
            "vectorizer": VECTORIZER_OPTIONS[0],
            "match_field": single.get("field_1", "rubrica"),
            "k": k,
            "data_points": 100,
            "retriever_results": 30,
            "llm_model": LLM_MODEL_OPTIONS[0],
            "fields": [single.get("field_1", "rubrica"), single.get("field_2", "testo")],
            "weights": [float(single.get("boost_1", 1.0)), float(single.get("boost_2", 1.0))],
        }
    else:
        result = {
            "model_type": retriever_type,
            "vectorizer": single.get("vectorizer", VECTORIZER_OPTIONS[0]),
            "match_field": single.get("match_field", "testo"),
            "k": k,
            "data_points": single.get("data_points", 100),
            "retriever_results": 30,
            "llm_model": LLM_MODEL_OPTIONS[0],
        }

    if reranker_config:
        result["reranker_config"] = reranker_config
    return result


# ---------------------------------------------------------------------------
# RRF sub-retriever editor
# ---------------------------------------------------------------------------

def _render_rrf_sub_retriever(retriever: dict, idx: int, key_prefix: str, suffix: str) -> dict:
    """
    Disegna i widget per un sub-retriever RRF (A o B) e restituisce il dict aggiornato.

    suffix : "a" o "b", distingue i widget key dei due retriever dello stesso modello.
    """
    kind = retriever.get("kind", "Dense search")
    if kind not in RRF_RETRIEVER_KIND_OPTIONS:
        kind = "Dense search"

    selected_kind = st.selectbox(
        "Kind",
        RRF_RETRIEVER_KIND_OPTIONS,
        index=RRF_RETRIEVER_KIND_OPTIONS.index(kind),
        key=f"{key_prefix}_{idx}_rrf_{suffix}_kind",
    )
    result = {"kind": selected_kind}

    if selected_kind == "Dense search":
        result["vectorizer"] = st.selectbox(
            "Vectorizer",
            VECTORIZER_OPTIONS,
            index=VECTORIZER_OPTIONS.index(retriever.get("vectorizer", VECTORIZER_OPTIONS[0])),
            key=f"{key_prefix}_{idx}_rrf_{suffix}_vec",
        )
        result["match_field"] = st.selectbox(
            "Campo",
            MATCH_FIELD_OPTIONS,
            index=MATCH_FIELD_OPTIONS.index(retriever.get("match_field", "testo")),
            key=f"{key_prefix}_{idx}_rrf_{suffix}_field",
        )

    elif selected_kind == "Sparse search":
        result["match_field"] = st.selectbox(
            "Campo",
            MATCH_FIELD_OPTIONS,
            index=MATCH_FIELD_OPTIONS.index(retriever.get("match_field", "testo")),
            key=f"{key_prefix}_{idx}_rrf_{suffix}_field",
        )

    else:  # Sparse search (weighted)
        col_f1, col_b1, col_f2, col_b2 = st.columns([2, 1, 2, 1], gap="small")
        with col_f1:
            result["field_1"] = st.selectbox(
                "Campo 1", MATCH_FIELD_OPTIONS,
                index=MATCH_FIELD_OPTIONS.index(retriever.get("field_1", "rubrica")),
                key=f"{key_prefix}_{idx}_rrf_{suffix}_f1",
            )
        with col_b1:
            result["boost_1"] = float(st.number_input(
                "Boost 1", min_value=0.0, max_value=20.0,
                value=float(retriever.get("boost_1", 2.0)), step=0.1,
                key=f"{key_prefix}_{idx}_rrf_{suffix}_b1",
            ))
        with col_f2:
            result["field_2"] = st.selectbox(
                "Campo 2", MATCH_FIELD_OPTIONS,
                index=MATCH_FIELD_OPTIONS.index(retriever.get("field_2", "testo")),
                key=f"{key_prefix}_{idx}_rrf_{suffix}_f2",
            )
        with col_b2:
            result["boost_2"] = float(st.number_input(
                "Boost 2", min_value=0.0, max_value=20.0,
                value=float(retriever.get("boost_2", 1.0)), step=0.1,
                key=f"{key_prefix}_{idx}_rrf_{suffix}_b2",
            ))

    return result


# ---------------------------------------------------------------------------
# Reranker section (opzionale, si applica a qualsiasi retriever)
# ---------------------------------------------------------------------------

def _render_reranker_section(state: dict, retriever_k: int, idx: int, key_prefix: str) -> dict:
    """
    Sezione opzionale del reranker. Disegna il checkbox e, se attivo, i parametri.
    Emette un warning se rr > k del retriever.

    state       : dict corrente (single o fusion_rrf) da cui leggere i valori
    retriever_k : k del retriever, usato solo per il warning rr > k
    """
    result = dict(state)

    reranker_enabled = st.checkbox(
        "Applica LLM reranker",
        value=bool(state.get("reranker_enabled", False)),
        key=f"{key_prefix}_{idx}_rrk_enabled",
    )
    result["reranker_enabled"] = reranker_enabled

    if not reranker_enabled:
        return result

    col_llm, col_rr, col_rk = st.columns([3, 1, 1], gap="small")
    with col_llm:
        result["reranker_llm_model"] = st.selectbox(
            "LLM",
            LLM_MODEL_OPTIONS,
            index=LLM_MODEL_OPTIONS.index(state.get("reranker_llm_model", LLM_MODEL_OPTIONS[0])),
            key=f"{key_prefix}_{idx}_rrk_llm",
        )
    with col_rr:
        rr = int(st.number_input(
            "rr", min_value=1, max_value=2000,
            value=int(state.get("reranker_results", 30)),
            key=f"{key_prefix}_{idx}_rrk_rr",
        ))
        result["reranker_results"] = rr
    with col_rk:
        result["reranker_k"] = int(st.number_input(
            "k", min_value=1, max_value=2000,
            value=int(state.get("reranker_k", 10)),
            key=f"{key_prefix}_{idx}_rrk_k",
        ))

    if rr > retriever_k:
        st.warning(
            f"rr ({rr}) > k retriever ({retriever_k}): "
            f"l'LLM riceverà solo {retriever_k} documenti."
        )

    return result


# ---------------------------------------------------------------------------
# Single retriever editor
# ---------------------------------------------------------------------------

def _render_single_editor(single: dict, idx: int, key_prefix: str) -> dict:
    """
    Disegna i widget per un single retriever e restituisce il dict aggiornato.
    I campi mostrati cambiano in base al tipo scelto.
    """
    retriever_type = single.get("retriever_type", "Sparse search")
    if retriever_type not in SINGLE_RETRIEVER_TYPE_OPTIONS:
        retriever_type = "Sparse search"

    retriever_type = st.selectbox(
        "Retriever",
        SINGLE_RETRIEVER_TYPE_OPTIONS,
        index=SINGLE_RETRIEVER_TYPE_OPTIONS.index(retriever_type),
        key=f"{key_prefix}_{idx}_single_type",
    )
    result = {"retriever_type": retriever_type}

    if retriever_type == "Sparse search":
        col_field, col_k = st.columns([4, 1], gap="small")
        with col_field:
            result["match_field"] = st.selectbox(
                "Campo", MATCH_FIELD_OPTIONS,
                index=MATCH_FIELD_OPTIONS.index(single.get("match_field", "testo")),
                key=f"{key_prefix}_{idx}_single_field",
            )
        with col_k:
            result["k"] = int(st.number_input(
                "k", min_value=1, max_value=2000,
                value=int(single.get("k", 10)),
                key=f"{key_prefix}_{idx}_single_k",
            ))

    elif retriever_type == "Dense search":
        col_vec, col_field, col_k, col_dp = st.columns([3, 2, 1, 1], gap="small")
        with col_vec:
            result["vectorizer"] = st.selectbox(
                "Vectorizer", VECTORIZER_OPTIONS,
                index=VECTORIZER_OPTIONS.index(single.get("vectorizer", VECTORIZER_OPTIONS[0])),
                key=f"{key_prefix}_{idx}_single_vec",
            )
        with col_field:
            result["match_field"] = st.selectbox(
                "Campo", MATCH_FIELD_OPTIONS,
                index=MATCH_FIELD_OPTIONS.index(single.get("match_field", "testo")),
                key=f"{key_prefix}_{idx}_single_field",
            )
        with col_k:
            result["k"] = int(st.number_input(
                "k", min_value=1, max_value=2000,
                value=int(single.get("k", 10)),
                key=f"{key_prefix}_{idx}_single_k",
            ))
        with col_dp:
            result["data_points"] = int(st.number_input(
                "#pts", min_value=10, max_value=2000,
                value=int(single.get("data_points", 100)),
                key=f"{key_prefix}_{idx}_single_dp",
            ))

    elif retriever_type == "Sparse search (weighted)":
        col_f1, col_b1, col_f2, col_b2, col_k = st.columns([2, 1, 2, 1, 1], gap="small")
        with col_f1:
            result["field_1"] = st.selectbox(
                "Campo 1", MATCH_FIELD_OPTIONS,
                index=MATCH_FIELD_OPTIONS.index(single.get("field_1", "rubrica")),
                key=f"{key_prefix}_{idx}_single_f1",
            )
        with col_b1:
            result["boost_1"] = float(st.number_input(
                "Boost 1", min_value=0.0, max_value=20.0,
                value=float(single.get("boost_1", 2.0)), step=0.1,
                key=f"{key_prefix}_{idx}_single_b1",
            ))
        with col_f2:
            result["field_2"] = st.selectbox(
                "Campo 2", MATCH_FIELD_OPTIONS,
                index=MATCH_FIELD_OPTIONS.index(single.get("field_2", "testo")),
                key=f"{key_prefix}_{idx}_single_f2",
            )
        with col_b2:
            result["boost_2"] = float(st.number_input(
                "Boost 2", min_value=0.0, max_value=20.0,
                value=float(single.get("boost_2", 1.0)), step=0.1,
                key=f"{key_prefix}_{idx}_single_b2",
            ))
        with col_k:
            result["k"] = int(st.number_input(
                "k", min_value=1, max_value=2000,
                value=int(single.get("k", 10)),
                key=f"{key_prefix}_{idx}_single_k",
            ))

    st.divider()
    result = _render_reranker_section(result, retriever_k=result["k"], idx=idx, key_prefix=key_prefix)
    return result


# ---------------------------------------------------------------------------
# Fusion (RRF) editor
# ---------------------------------------------------------------------------

def _render_fusion_editor(fusion: dict, idx: int, key_prefix: str) -> dict:
    """
    Disegna i widget per la modalità Fusion RRF e restituisce il dict aggiornato.
    """
    result = dict(fusion)

    col_rrfk, col_k = st.columns([1, 1], gap="small")
    with col_rrfk:
        result["rrf_k"] = int(st.number_input(
            "RRF k", min_value=1, max_value=500,
            value=int(fusion.get("rrf_k", 60)),
            key=f"{key_prefix}_{idx}_rrf_rrfk",
        ))
    with col_k:
        result["k"] = int(st.number_input(
            "k finale", min_value=1, max_value=2000,
            value=int(fusion.get("k", 10)),
            key=f"{key_prefix}_{idx}_rrf_kfinal",
        ))

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Retriever A**")
        result["retriever_a"] = _render_rrf_sub_retriever(
            fusion.get("retriever_a", {}), idx, key_prefix, "a"
        )
    with col_b:
        st.markdown("**Retriever B**")
        result["retriever_b"] = _render_rrf_sub_retriever(
            fusion.get("retriever_b", {}), idx, key_prefix, "b"
        )

    st.divider()
    result = _render_reranker_section(result, retriever_k=result["k"], idx=idx, key_prefix=key_prefix)
    return result


# ---------------------------------------------------------------------------
# Top-level model editor (mode selector + dispatch)
# ---------------------------------------------------------------------------

def _render_single_model_editor(cfg: dict, idx: int, key_prefix: str) -> dict:
    """
    Editor di primo livello per un modello.
    Mostra il selettore di modalità e delega all'editor corrispondente.
    """
    result = dict(cfg)

    mode = cfg.get("retrieval_mode", "Single retriever")
    if mode not in RETRIEVAL_MODE_OPTIONS:
        mode = "Single retriever"

    mode = st.selectbox(
        "Retrieval mode",
        RETRIEVAL_MODE_OPTIONS,
        index=RETRIEVAL_MODE_OPTIONS.index(mode),
        key=f"{key_prefix}_{idx}_mode",
    )
    result["retrieval_mode"] = mode

    if mode == "Single retriever":
        result["single"] = _render_single_editor(cfg.get("single", {}), idx, key_prefix)
    else:
        result["fusion_rrf"] = _render_fusion_editor(cfg.get("fusion_rrf", {}), idx, key_prefix)

    return result


# ---------------------------------------------------------------------------
# Models section (lista di editor + aggiungi/rimuovi)
# ---------------------------------------------------------------------------

def _render_models_section(models_state_key: str, key_prefix: str) -> list[ModelConfig]:
    """
    Disegna la lista di editor modelli con i pulsanti aggiungi/rimuovi.
    Ritorna i config convertiti in ModelConfig per run_model() / evaluate_from_df().

    models_state_key : chiave in st.session_state dove vive la lista di config
    key_prefix       : prefisso per i widget key (es. "cmp", "eval")
    """
    if models_state_key not in st.session_state:
        st.session_state[models_state_key] = [_default_model()]

    current_configs = st.session_state[models_state_key]
    updated_configs = []

    st.header("Definisci modelli")

    for idx, cfg in enumerate(current_configs):
        col_expander, col_remove = st.columns([30, 1], gap="small")
        with col_expander:
            expander = st.expander(_model_title(idx, cfg), expanded=(idx == 0))
        with col_remove:
            if st.button("🗑️", key=f"{key_prefix}_{idx}_remove", help="Rimuovi modello"):
                if len(current_configs) > 1:
                    st.session_state[models_state_key].pop(idx)
                    st.rerun()
                else:
                    st.warning(ERROR_MIN_ONE_MODEL)

        with expander:
            updated_cfg = _render_single_model_editor(cfg, idx, key_prefix)
            updated_configs.append(updated_cfg)

    def _add_model():
        st.session_state[models_state_key].append(_default_model())

    st.button("+ aggiungi modello", on_click=_add_model, key=f"{key_prefix}_add")
    st.session_state[models_state_key] = updated_configs

    model_configs = [_cfg_to_model_config(cfg) for cfg in updated_configs]

    with st.expander("Recap configurazione modelli", expanded=False):
        st.caption(
            "Configurazione che la GUI ha interpretato e che verrà usata nella pipeline."
        )
        recap = [clean_model_config(c) for c in model_configs]
        st.json(recap)

    return model_configs


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _result_title(model_index: int, config: ModelConfig) -> str:
    if config["model_type"] == "Sparse search":
        base = f"### Modello {model_index + 1}: Sparse search"
    elif config["model_type"] == "Sparse search (weighted)":
        fields = config.get("fields", [])
        weights = config.get("weights", [])
        total = sum(weights) or 1
        parts = " + ".join(f"{f}×{w/total:.2f}" for f, w in zip(fields, weights))
        base = f"### Modello {model_index + 1}: Sparse search (weighted) — {parts}"
    elif config["model_type"] == "Fusion (RRF)":
        rrf_cfg = config.get("rrf_config", {})
        rrf_k = rrf_cfg.get("rrf_k", 60)
        base = f"### Modello {model_index + 1}: Fusion (RRF) · rrf_k={rrf_k} · k={config['k']}"
    else:
        base = f"### Modello {model_index + 1}: {config['model_type']} / {config['vectorizer']}"

    reranker_config = config.get("reranker_config")
    if reranker_config:
        llm = reranker_config["llm_model"]
        rk = reranker_config["k"]
        return f"{base} + rerank({llm} · k={rk})"
    return base


def _render_model_result(
    model_index: int,
    config: ModelConfig,
    query: str,
    es_client: ES,
    index_names: dict[str, str],
):
    """
    Esegue la query con il modello specificato e mostra i risultati.
    Per i tipi non ancora supportati mostra un avviso e non esegue nulla.
    """
    if config["model_type"] in UNSUPPORTED_MODEL_TYPES:
        st.info(f"Modello {model_index + 1}: '{config['model_type']}' non ancora supportato.")
        return

    placeholder = st.empty()
    with placeholder.container():
        match_field = config["match_field"]
        field_profile = resolve_field_profile(match_field, index_names)
        index_name = field_profile["index_name"]
        source_fields = field_profile["source_fields"]
        plot_text_field = field_profile["plot_text_field"]

        st.markdown(_result_title(model_index, config))
        with st.spinner("In corso..."):
            run_config: dict[str, Any] = dict(config)
            run_config["vectorizer"] = resolve_effective_vectorizer(
                run_config["model_type"],
                run_config["match_field"],
                run_config["vectorizer"],
            )

            if (
                run_config["match_field"] == "rubriche_con_periodi"
                and run_config["vectorizer"] != config["vectorizer"]
            ):
                st.info(INFO_AUTO_VECTORIZER)

            if config["model_type"] == "RAG":
                run_config["source_fields"] = source_fields
                run_config["text_field"] = plot_text_field
                result = run_rag(query, run_config, index_name=index_name)
            else:
                result = run_model(
                    es_client,
                    index_name,
                    query,
                    run_config,
                    source_fields=source_fields,
                    plot_text_field=plot_text_field,
                )

        if result["error"]:
            st.error(result["error"])
            return
        if not result["hits"]:
            st.info(INFO_NO_RESULTS)
            return

        has_reranker = config.get("reranker_config") is not None
        top_k = min(int(config["k"]), len(result["hits"]))
        # Se il reranker ha prodotto ranked_hits, usiamo quelli per la tabella.
        table_hits = result.get("ranked_hits") or result["hits"]
        table = build_results_table(table_hits, top_k, match_field)
        table_column_config = {
            "rank": st.column_config.NumberColumn("rank", width="small"),
            "score": st.column_config.NumberColumn("score", width="small"),
            "id": st.column_config.TextColumn("id", width="small"),
        }

        no_plot = has_reranker or config["model_type"] in ("Sparse search", "Sparse search (weighted)", "Fusion (RRF)")
        if no_plot:
            st.dataframe(table, use_container_width=True, key=f"table_{model_index}",
                         column_config=table_column_config)
            if has_reranker:
                if result.get("generation_error"):
                    st.warning(result["generation_error"])
                if result.get("prompt"):
                    with st.expander("Prompt", expanded=False):
                        st.code(result["prompt"])
                if result.get("answer_pretty"):
                    with st.expander("Response", expanded=False):
                        st.code(result["answer_pretty"], language="json")
            return

        # Dense search: tabella + grafico 3D affiancati
        col_table, col_plot = st.columns([3, 3], gap="large")
        with col_table:
            st.dataframe(table, use_container_width=True, key=f"table_{model_index}",
                         column_config=table_column_config)
        with col_plot:
            if result["embed_error"]:
                st.warning(result["embed_error"])
            if result["query_vec"] is not None and result["doc_vecs"] is not None:
                hover_texts = build_hover_texts(result["hits"], match_field)
                build_plot(result["query_vec"], result["doc_vecs"], top_k,
                           f"plot_{model_index}", hover_texts)
            else:
                st.info(INFO_NO_EMBEDDINGS_PLOT)
