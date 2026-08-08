"""Esecuzione dei modelli: dato una query, costruisce i retriever e produce gli hit.

Due famiglie di pipeline:
- periodi/combo: build_retriever, build_reranker, run_model
- reati:         build_pipeline_retriever, run_pipeline_model, run_branching_model
"""
import numpy as np

from es import ES
from retriever import (
    Retriever,
    SparseRetriever,
    WeightedSparseRetriever,
    DenseRetriever,
    RRFRetriever,
    FilteredDenseRetriever,
    FilteredSparseRetriever,
    PipelineRetriever,
)
from reranker import LLMReranker
from config import VECTORIZER_CONFIG, VECTORIZER_OPTIONS, ModelConfig, get_vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline periodi/combo
# ─────────────────────────────────────────────────────────────────────────────

def _build_sub_retriever(
    retriever_cfg: dict,
    es_client: ES,
    index_name: str,
    source_fields: list[str],
) -> Retriever:
    """
    Costruisce un Retriever concreto a partire dal dict di configurazione
    di un sub-retriever RRF (chiave "kind" invece di "model_type").
    """
    kind = retriever_cfg.get("kind", "Sparse search")

    if kind == "Sparse search":
        match_field = retriever_cfg.get("match_field", "testo")
        return SparseRetriever(es_client, index_name, match_field, source_fields)

    if kind == "Sparse search (weighted)":
        fields = [retriever_cfg.get("field_1", "rubrica"), retriever_cfg.get("field_2", "testo")]
        weights = [float(retriever_cfg.get("boost_1", 1.0)), float(retriever_cfg.get("boost_2", 1.0))]
        return WeightedSparseRetriever(es_client, index_name, fields, weights, source_fields)

    # Dense search
    vectorizer_choice = retriever_cfg.get("vectorizer", VECTORIZER_OPTIONS[0])
    match_field = retriever_cfg.get("match_field", "testo")
    vectorizer_cfg_map = VECTORIZER_CONFIG[vectorizer_choice]
    vector_field = f"{vectorizer_cfg_map['es_field_prefix']}_{match_field}"
    vectorizer = get_vectorizer(vectorizer_cfg_map["model_name"])
    return DenseRetriever(es_client, index_name, vector_field, vectorizer, source_fields)


def build_retriever(
    config: ModelConfig,
    es_client: ES,
    index_name: str,
    source_fields: list[str],
) -> Retriever:
    """
    Costruisce il Retriever corretto a partire dal ModelConfig.
    Usato da run_model() per disaccoppiare la logica di retrieval dal resto.
    """
    model_type = config["model_type"]
    match_field = config["match_field"]

    if model_type == "Sparse search":
        return SparseRetriever(es_client, index_name, match_field, source_fields)

    if model_type == "Sparse search (weighted)":
        fields = config.get("fields", [match_field])
        weights = config.get("weights", [1.0])
        return WeightedSparseRetriever(es_client, index_name, fields, weights, source_fields)

    if model_type == "Dense search":
        vectorizer_choice = config["vectorizer"]
        vectorizer_cfg = VECTORIZER_CONFIG[vectorizer_choice]
        vector_field = f"{vectorizer_cfg['es_field_prefix']}_{match_field}"
        vectorizer = get_vectorizer(vectorizer_cfg["model_name"])
        return DenseRetriever(es_client, index_name, vector_field, vectorizer, source_fields)

    if model_type == "Fusion (RRF)":
        rrf_config = config.get("rrf_config", {})
        rrf_k = rrf_config.get("rrf_k", 60)
        window_size = rrf_config.get("window_size", 100)
        retriever_a = _build_sub_retriever(rrf_config.get("retriever_a", {}), es_client, index_name, source_fields)
        retriever_b = _build_sub_retriever(rrf_config.get("retriever_b", {}), es_client, index_name, source_fields)
        return RRFRetriever(retriever_a, retriever_b, rrf_k, window_size)

    # RAG in valutazione CSV: usiamo BM25 sul campo configurato come retriever.
    return SparseRetriever(es_client, index_name, match_field, source_fields)


def build_reranker(reranker_config: dict, match_field: str) -> LLMReranker:
    """
    Costruisce un LLMReranker a partire dal reranker_config del ModelConfig.
    Punto di estensione futuro: altri tipi di reranker (cross-encoder, ecc.)
    possono essere aggiunti qui in base a reranker_config["type"].
    """
    return LLMReranker(llm_model=reranker_config["llm_model"], match_field=match_field)


def run_model(
    es_client: ES,
    index_name: str,
    query: str,
    config: ModelConfig,
    source_fields: list[str],
    plot_text_field: str,
    compute_plot_vectors: bool = True,
):
    model_type = config["model_type"]
    top_k = int(config["k"])
    # Per Dense search recuperiamo più candidati (data_points) e poi tagliamo a top_k.
    fetch_k = int(config.get("data_points", top_k)) if model_type == "Dense search" else top_k

    try:
        retriever = build_retriever(config, es_client, index_name, source_fields)
        hits = retriever.search(query, fetch_k)
    except Exception as exc:
        return {"error": str(exc), "embed_error": None, "hits": [], "query_vec": None, "doc_vecs": None}

    if not hits:
        return {"error": None, "embed_error": None, "hits": [], "query_vec": None, "doc_vecs": None}

    # Applica il reranker se configurato (intercetta prima di qualsiasi altra logica).
    reranker_config = config.get("reranker_config")
    if reranker_config:
        reranker = build_reranker(reranker_config, config["match_field"])
        rr = int(reranker_config.get("reranker_results", len(hits)))
        k_final = int(reranker_config.get("k", rr))
        return reranker.rerank(query, hits, rr=rr, k=k_final)

    # Tipi senza grafico 3D: restituiamo solo gli hit.
    if model_type in ("Sparse search", "Sparse search (weighted)", "Fusion (RRF)"):
        return {"error": None, "embed_error": None, "hits": hits, "query_vec": None, "doc_vecs": None}

    if model_type == "Dense search":
        if not compute_plot_vectors:
            return {"error": None, "embed_error": None, "hits": hits, "query_vec": None, "doc_vecs": None}

        # Il DenseRetriever salva il vettore della query dopo la ricerca.
        query_vec = retriever.query_vec

        doc_ids = [str(hit.get("_id")) for hit in hits]
        doc_vecs = None
        embed_error = None
        try:
            vectorizer_cfg = VECTORIZER_CONFIG[config["vectorizer"]]
            vector_field = f"{vectorizer_cfg['es_field_prefix']}_{config['match_field']}"
            vectors_by_id = es_client.get_vectors_by_ids(index_name, doc_ids, vector_field)
            ordered_vectors = [vectors_by_id.get(doc_id) for doc_id in doc_ids]
            if any(vec is None for vec in ordered_vectors):
                raise ValueError(
                    f"Vettore mancante in indice per campo '{vector_field}' su uno o piu documenti"
                )
            doc_vecs = np.array(ordered_vectors, dtype=float)
        except Exception as exc:
            embed_error = f"Lettura vettori da indice fallita: {exc}"

        return {
            "error": None,
            "embed_error": embed_error,
            "hits": hits,
            "query_vec": query_vec,
            "doc_vecs": doc_vecs,
        }

    # RAG (percorso valutazione CSV): BM25 + embedding dei testi recuperati.
    embed_error = None
    query_vec = None
    doc_vecs = None
    vectorizer_choice = config["vectorizer"]
    texts_to_embed = [query] + [
        hit.get("_source", {}).get(plot_text_field) or "" for hit in hits
    ]
    try:
        vectorizer_cfg = VECTORIZER_CONFIG[vectorizer_choice]
        vectorizer = get_vectorizer(vectorizer_cfg["model_name"])
        vectors = np.array(vectorizer.get_embeddings(texts_to_embed), dtype=float)
        query_vec = vectors[0]
        doc_vecs = vectors[1:]
    except Exception as exc:
        embed_error = f"Embedding fallito: {exc}"

    return {
        "error": None,
        "embed_error": embed_error,
        "hits": hits,
        "query_vec": query_vec,
        "doc_vecs": doc_vecs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline reati
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_retriever(config: dict, es_client: ES, index_reati: str, index_circ: str, key_table: dict) -> PipelineRetriever:
    stage1_mode = config.get("stage1_mode", "dense")
    stage3_mode = config.get("stage3_mode", "dense")
    embedder = config.get("embedder", "text-embedding-3-large")
    vectorizer = get_vectorizer(embedder)
    m, n = int(config.get("m", 10)), int(config.get("n", 10))

    r_vec, r_src = f"{embedder}_combo_reati_content", ["combo_reati_content", "combo_reati"]
    c_vec, c_src = f"{embedder}_combo_reati_circostanziati_content", ["combo_reati_circostanziati_content", "combo_reati_circostanziati"]

    # --- Stage 1: retriever su wf_reati (senza filtro), dense/sparse/hybrid ---
    def s1_dense():
        return DenseRetriever(es_client=es_client, index_name=index_reati,
                              vector_field=r_vec, vectorizer=vectorizer, source_fields=r_src)
    def s1_sparse():
        return SparseRetriever(es_client=es_client, index_name=index_reati,
                               match_field="combo_reati_content", source_fields=r_src)
    if stage1_mode == "dense":
        stage1_retriever = s1_dense()
    elif stage1_mode == "hybrid":
        stage1_retriever = RRFRetriever(s1_sparse(), s1_dense(), rrf_k=60, window_size=max(m * 5, 100))
    else:
        stage1_retriever = s1_sparse()

    # --- Stage 3: factory di un retriever filtrato per base reato, dense/sparse/hybrid ---
    def f3_dense(fv):
        return FilteredDenseRetriever(es_client=es_client, index_name=index_circ,
                                      vector_field=c_vec, vectorizer=vectorizer, source_fields=c_src,
                                      filter_field="combo_reati_circostanziati", filter_values=fv)
    def f3_sparse(fv):
        return FilteredSparseRetriever(es_client=es_client, index_name=index_circ,
                                       match_field="combo_reati_circostanziati_content", source_fields=c_src,
                                       filter_field="combo_reati_circostanziati", filter_values=fv)
    if stage3_mode == "dense":
        stage3_factory = f3_dense
    elif stage3_mode == "hybrid":
        def stage3_factory(fv):
            return RRFRetriever(f3_sparse(fv), f3_dense(fv), rrf_k=60, window_size=max(n * 5, 50))
    else:
        stage3_factory = f3_sparse

    # --- Ablation "no mapping": stage3 globale (non filtrato), dense/sparse/hybrid ---
    def g3_dense():
        return DenseRetriever(es_client=es_client, index_name=index_circ,
                              vector_field=c_vec, vectorizer=vectorizer, source_fields=c_src)
    def g3_sparse():
        return SparseRetriever(es_client=es_client, index_name=index_circ,
                               match_field="combo_reati_circostanziati_content", source_fields=c_src)
    stage3_global = None
    if config.get("no_mapping"):
        if stage3_mode == "dense":
            stage3_global = g3_dense()
        elif stage3_mode == "hybrid":
            stage3_global = RRFRetriever(g3_sparse(), g3_dense(), rrf_k=60, window_size=max(m * n * 2, 200))
        else:
            stage3_global = g3_sparse()

    return PipelineRetriever(
        stage1_retriever=stage1_retriever,
        stage3_retriever_factory=stage3_factory,
        key_table=key_table,
        m=m,
        n=n,
        stage3_global=stage3_global,
    )


def run_pipeline_model(es_client: ES, query: str, config: dict, index_reati: str, index_circ: str, key_table: dict, classifier=None, splitter=None) -> dict:
    from classifier import LLMClassifier
    from splitter import resolve_queries
    k = int(config.get("k", 10))
    m = int(config.get("m", 10))
    n = int(config.get("n", 10))
    pool_size = m * (1 + n)

    # Split della query (una volta): sceglie quale parte usare in ogni stage.
    parts, stage1_query, stage3_query = resolve_queries(query, config, splitter)

    try:
        retriever = build_pipeline_retriever(config, es_client, index_reati, index_circ, key_table)
        debug = retriever.search_with_debug(query, pool_size, stage1_query=stage1_query, stage3_query=stage3_query)
    except Exception as exc:
        return {"error": str(exc), "hits": [], "predicted_type": None, "classifier_answer": None, "debug": None, "splitter": parts}

    try:
        if classifier is None:
            classifier = LLMClassifier(llm_model=config["llm_model"], classifier_version=config.get("classifier_version", "v1"))
        # La classificazione usa sempre la query intera, non quella ridotta.
        result = classifier.classify(query, debug["pool"], k)
        return {
            "error": None,
            "hits": result["hits"],
            "predicted_type": result["predicted_type"],
            "classifier_reasoning": result.get("reasoning"),
            "classifier_prompt": result["prompt"],
            "classifier_answer": result["answer"],
            "parse_error": result.get("parse_error"),
            "debug": debug,
            "splitter": parts,
        }
    except Exception as exc:
        return {"error": str(exc), "hits": [], "predicted_type": None, "classifier_answer": None, "debug": debug, "splitter": parts}


def run_branching_model(
    es_client: ES,
    query: str,
    config: dict,
    index_reati: str,
    index_circ: str,
    key_table: dict,
    type_classifier=None,
    ranker=None,
    splitter=None,
) -> dict:
    """
    Pipeline con branching: classifica prima il tipo della query, poi chiama
    il retriever appropriato in base al tipo e al pipeline_type configurato.

    pipeline_type "branching_mapping":
      - reato → stage1 su wf_reati
      - circostanziato → PipelineRetriever (stage1 + key_table + stage3), pool filtrato ai soli circ

    pipeline_type "branching_direct":
      - reato → stage1 su wf_reati
      - circostanziato → retriever diretto su wf_reati_circ senza filtro key_table
    """
    from classifier import LLMTypeClassifier, LLMRanker
    from splitter import resolve_queries

    k = int(config.get("k", 10))
    m = int(config.get("m", 10))
    n = int(config.get("n", 10))
    pipeline_type = config.get("pipeline_type", "branching_mapping")
    embedder = config.get("embedder", "text-embedding-3-large")
    stage1_mode = config.get("stage1_mode", "dense")
    stage3_mode = config.get("stage3_mode", "dense")

    # Split della query (una volta): stage1_query -> indice reati, stage3_query -> indice circ.
    parts, stage1_query, stage3_query = resolve_queries(query, config, splitter)

    # Step 1: classifica il tipo della query (sempre sulla query intera)
    if type_classifier is None:
        type_classifier = LLMTypeClassifier(
            llm_model=config["llm_model"],
            classifier_version=config.get("only_classifier_version", "v1"),
        )
    try:
        clf_result = type_classifier.classify_type(query)
    except Exception as exc:
        return {"error": str(exc), "hits": [], "predicted_type": None,
                "classifier_answer": None, "debug": None, "splitter": parts}

    predicted_type = clf_result.get("predicted_type")

    # Step 2: retrieval in base al tipo e alla variante di pipeline
    try:
        if predicted_type == "reato" or predicted_type is None:
            # Branch reato (uguale per entrambe le varianti): retriever su wf_reati
            vectorizer = get_vectorizer(embedder) if stage1_mode == "dense" else None
            if stage1_mode == "dense":
                retriever = DenseRetriever(
                    es_client=es_client,
                    index_name=index_reati,
                    vector_field=f"{embedder}_combo_reati_content",
                    vectorizer=vectorizer,
                    source_fields=["combo_reati_content", "combo_reati"],
                )
            else:
                retriever = SparseRetriever(
                    es_client=es_client,
                    index_name=index_reati,
                    match_field="combo_reati_content",
                    source_fields=["combo_reati_content", "combo_reati"],
                )
            hits = retriever.search(stage1_query, m)
            debug = {"branch": "reato", "stage1_hits": hits, "stage3_by_reato": {}, "pool": hits[:k]}

        elif pipeline_type == "branching_mapping":
            # Branch circostanziato via mapping: usa PipelineRetriever, filtra il pool ai soli circ
            pipeline_retriever = build_pipeline_retriever(config, es_client, index_reati, index_circ, key_table)
            raw_debug = pipeline_retriever.search_with_debug(query, m * (1 + n), stage1_query=stage1_query, stage3_query=stage3_query)
            circ_hits = [
                h for h in raw_debug["pool"]
                if h.get("_source", {}).get("combo_reati_circostanziati", "")
                and str(h["_source"]["combo_reati_circostanziati"]).strip().lower() != "nan"
            ]
            hits = circ_hits
            debug = {**raw_debug, "branch": "circostanziato_mapping", "pool": circ_hits}

        else:  # branching_direct
            # Branch circostanziato diretto: retriever su wf_reati_circ senza filtro key_table
            vectorizer = get_vectorizer(embedder) if stage3_mode == "dense" else None
            if stage3_mode == "dense":
                retriever = DenseRetriever(
                    es_client=es_client,
                    index_name=index_circ,
                    vector_field=f"{embedder}_combo_reati_circostanziati_content",
                    vectorizer=vectorizer,
                    source_fields=["combo_reati_circostanziati_content", "combo_reati_circostanziati"],
                )
            else:
                retriever = SparseRetriever(
                    es_client=es_client,
                    index_name=index_circ,
                    match_field="combo_reati_circostanziati_content",
                    source_fields=["combo_reati_circostanziati_content", "combo_reati_circostanziati"],
                )
            hits = retriever.search(stage3_query, n)
            debug = {"branch": "circostanziato_direct", "stage1_hits": [], "stage3_by_reato": {}, "pool": hits[:k]}

    except Exception as exc:
        return {"error": str(exc), "hits": [], "predicted_type": predicted_type,
                "classifier_answer": clf_result.get("answer"), "debug": None, "splitter": parts}

    # Step 3 (opzionale): reranking LLM
    if ranker is None and config.get("use_ranker"):
        ranker = LLMRanker(llm_model=config["llm_model"])

    if ranker is not None:
        try:
            rerank_result = ranker.rerank(query, hits, k)
            hits = rerank_result["hits"]
        except Exception:
            hits = hits[:k]
    else:
        hits = hits[:k]

    return {
        "error": None,
        "hits": hits,
        "predicted_type": predicted_type,
        "classifier_reasoning": clf_result.get("reasoning"),
        "classifier_prompt": clf_result.get("prompt"),
        "classifier_answer": clf_result.get("answer"),
        "parse_error": clf_result.get("parse_error"),
        "debug": debug,
        "splitter": parts,
    }
