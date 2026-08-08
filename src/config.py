"""Configurazione del progetto: caricamento .env, costanti, opzioni UI, tipi.

Punto unico per le impostazioni d'ambiente (host ES, nomi indici), le opzioni
mostrate nella UI e i tipi condivisi (ModelConfig, ecc.).
"""
import os
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import streamlit as st

from es import ES


def load_env(path: str = ".env") -> None:
    """Carica le variabili da un file .env in os.environ.

    Salta righe vuote, commenti (#) e righe senza '='. Non sovrascrive le
    variabili già presenti in os.environ.
    """
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# UI options
MODEL_OPTIONS = ["Dense search", "RAG", "Sparse search"]
VECTORIZER_OPTIONS = ["text-embedding-3-large"]
RAG_RETRIEVER_OPTIONS = ["text-embedding-3-large", "BM25"]
MATCH_FIELD_OPTIONS = ["testo", "rubrica", "rubrica_testo", "rubriche_con_periodi"]
LLM_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5.1"]

# Search/vectorizer mappings
VECTORIZER_CONFIG = {
    "text-embedding-3-large": {
        "model_name": "text-embedding-3-large",
        "es_field_prefix": "text-embedding-3-large",
    },
}


FIELD_PROFILES = {
    "testo": {
        "index_key": "periodi",
        "source_fields": ["periodo", "fonte_normativa", "rubrica", "testo"],
        "plot_text_field": "testo",
    },
    "rubrica": {
        "index_key": "periodi",
        "source_fields": ["periodo", "fonte_normativa", "rubrica", "testo"],
        "plot_text_field": "testo",
    },
    "rubrica_testo": {
        "index_key": "periodi",
        "source_fields": ["periodo", "fonte_normativa", "rubrica", "testo"],
        "plot_text_field": "testo",
    },
    "rubriche_con_periodi": {
        "index_key": "combo",
        "source_fields": ["rubriche_con_periodi", "rubrica_imputativa"],
        "plot_text_field": "rubriche_con_periodi",
    },
    "periodo_content": {
        "index_key": "periodi_flat",
        "source_fields": ["periodo", "periodo_content"],
        "plot_text_field": "periodo_content",
    },
}


# Environment config
load_env()
DEFAULT_ES_HOST = os.environ["ES_HOST"]
DEFAULT_INDEX_NAME = os.environ["ES_INDEX_NAME"]
DEFAULT_INDEX_NAME_COMBO = os.environ["ES_INDEX_NAME_COMBO"]
DEFAULT_INDEX_NAME_REATI = os.environ.get("ES_INDEX_NAME_REATI", "")
DEFAULT_INDEX_NAME_REATI_CIRC_LIGHT = os.environ.get("ES_INDEX_NAME_REATI_CIRC_LIGHT", "")
DEFAULT_INDEX_NAME_REATI_CIRC_FULL = os.environ.get("ES_INDEX_NAME_REATI_CIRC_FULL", "")

# Indici del paper
DEFAULT_INDEX_PAPER_PERIODI_FLAT = os.environ.get("ES_INDEX_PAPER_PERIODI_FLAT", "")
# Corpus flat completo (tutti i periodi, distrattori inclusi) per le baseline "più baseline".
DEFAULT_INDEX_PAPER_PERIODI_FULL = os.environ.get("ES_INDEX_PAPER_PERIODI_FULL", "wf_periodi_aggiornato")
DEFAULT_INDEX_PAPER_REATI = os.environ.get("ES_INDEX_PAPER_REATI", "")
DEFAULT_INDEX_PAPER_REATI_CIRC_LIGHT = os.environ.get("ES_INDEX_PAPER_REATI_CIRC_LIGHT", "")

# Paper: storage dei run e mapping reato->circostanze
PAPER_RUNS_DIR = "runs/paper"
# Paper "hard": stessi modelli ma su GT in formato legacy (query/result/type),
# con metriche di blocco e tipi di query.
PAPER_HARD_RUNS_DIR = "runs/paper_hard"
PAPER_KEY_TABLE = "data/final/wf_key_table.csv"


ModelType = Literal["Dense search", "RAG", "Sparse search", "Sparse search (weighted)"]
MatchField = Literal["testo", "rubrica", "rubrica_testo", "rubriche_con_periodi"]
SearchHit = dict[str, Any]


class ModelConfig(TypedDict):
    model_type: ModelType
    vectorizer: str
    match_field: MatchField
    data_points: int
    retriever_results: int
    k: int
    llm_model: str


class FieldProfile(TypedDict):
    index_name: str
    source_fields: list[str]
    plot_text_field: str


class RunModelResult(TypedDict):
    error: str | None
    embed_error: str | None
    hits: list[SearchHit]
    query_vec: np.ndarray | None
    doc_vecs: np.ndarray | None


@st.cache_resource
def get_vectorizer(model_name: str):
    # Import locale per evitare un ciclo di import (vectorizer importa config).
    from vectorizer import Vectorizer
    return Vectorizer(model_name)


@st.cache_data
def load_queries(path: str):
    df = pd.read_csv(path)
    if "query" in df.columns:
        values = df["query"]
    else:
        values = df.iloc[:, 0]
    return [str(val) for val in values.dropna().tolist()]


def default_model_config():
    return {
        "model_type": "Dense search",
        "vectorizer": "text-embedding-3-large",
        "match_field": "testo",
        "data_points": 100,
        "retriever_results": 30,
        "k": 10,
        "llm_model": "gpt-4o-mini",
    }


def resolve_field_profile(match_field: str, index_names: dict[str, str]):
    profile = FIELD_PROFILES[match_field]
    target_index = index_names[profile["index_key"]]
    return {
        "index_name": target_index,
        "source_fields": profile["source_fields"],
        "plot_text_field": profile["plot_text_field"],
    }


def resolve_effective_vectorizer(model_type: str, match_field: str, vectorizer_choice: str):
    if match_field != "rubriche_con_periodi":
        return vectorizer_choice

    if model_type == "Sparse search":
        return vectorizer_choice

    if model_type == "RAG" and vectorizer_choice == "BM25":
        return vectorizer_choice

    return "text-embedding-3-large"


def get_available_vectorizers(es_client: ES, index_name: str, model_type: str, match_field: str):
    if model_type == "Sparse search":
        return []

    if match_field == "rubriche_con_periodi":
        if model_type == "RAG":
            return ["BM25", "text-embedding-3-large"]
        return ["text-embedding-3-large"]

    properties = es_client.get_index_properties(index_name)
    available = []
    for choice in VECTORIZER_OPTIONS:
        prefix = VECTORIZER_CONFIG[choice]["es_field_prefix"]
        field_name = f"{prefix}_{match_field}"
        if field_name in properties:
            available.append(choice)

    if model_type == "RAG":
        return available + ["BM25"]
    return available


def clean_model_config(config: ModelConfig) -> dict:
    """
    Restituisce solo i campi rilevanti per il tipo di modello specificato.
    Usato per il recap in UI e per il salvataggio nel meta.json.
    Se il config ha un reranker_config, viene incluso in fondo.
    """
    model_type = config.get("model_type")

    if model_type == "Sparse search":
        result = {
            "model_type": model_type,
            "match_field": config["match_field"],
            "k": config["k"],
        }

    elif model_type == "Sparse search (weighted)":
        result = {
            "model_type": model_type,
            "k": config["k"],
            "fields": config.get("fields", []),
            "weights": config.get("weights", []),
        }

    elif model_type == "Dense search":
        result = {
            "model_type": model_type,
            "vectorizer": config["vectorizer"],
            "match_field": config["match_field"],
            "k": config["k"],
            "data_points": config["data_points"],
        }

    elif model_type == "Fusion (RRF)":
        result = {
            "model_type": model_type,
            "k": config["k"],
            "rrf_config": config.get("rrf_config", {}),
        }

    else:
        result = dict(config)

    if config.get("reranker_config"):
        result["reranker_config"] = config["reranker_config"]
    if config.get("index"):
        result["index"] = config["index"]  # indice cablato (baseline su corpus diverso)

    return result


@st.cache_resource
def load_key_table(path: str = "data/wf_key_table.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["combo_reati_circostanziati"])
    df = df[df["combo_reati_circostanziati"].str.strip().str.lower() != "nan"]
    return df.groupby("combo_reati")["combo_reati_circostanziati"].apply(list).to_dict()
