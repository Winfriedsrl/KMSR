import warnings
import numpy as np
from ranx import Run, fuse as ranx_fuse
from es import ES
from vectorizer import Vectorizer


class Retriever:
    def search(self, query: str, k: int) -> list[dict]:
        raise NotImplementedError


class SparseRetriever(Retriever):
    def __init__(self, es_client: ES, index_name: str, match_field: str, source_fields: list[str]):
        self.es_client = es_client
        self.index_name = index_name
        self.match_field = match_field
        self.source_fields = source_fields

    def search(self, query: str, k: int) -> list[dict]:
        return self.es_client.search_bm25(
            index_name=self.index_name,
            query_text=query,
            size=k,
            source_fields=self.source_fields,
            match_fields=[self.match_field],
        )


class WeightedSparseRetriever(Retriever):
    def __init__(
        self,
        es_client: ES,
        index_name: str,
        fields: list[str],
        weights: list[float],
        source_fields: list[str],
    ):
        self.es_client = es_client
        self.index_name = index_name
        self.fields = fields
        self.weights = weights
        self.source_fields = source_fields

    def search(self, query: str, k: int) -> list[dict]:
        return self.es_client.search_bm25_weighted(
            index_name=self.index_name,
            query_text=query,
            size=k,
            source_fields=self.source_fields,
            fields=self.fields,
            weights=self.weights,
        )


class DenseRetriever(Retriever):
    def __init__(
        self,
        es_client: ES,
        index_name: str,
        vector_field: str,
        vectorizer: Vectorizer,
        source_fields: list[str],
    ):
        self.es_client = es_client
        self.index_name = index_name
        self.vector_field = vector_field
        self.vectorizer = vectorizer
        self.source_fields = source_fields
        # Salvato durante search(), letto da run_model() per il grafico 3D.
        self.query_vec: np.ndarray | None = None

    def search(self, query: str, k: int) -> list[dict]:
        self.query_vec = np.array(self.vectorizer.get_embedding(query), dtype=float)
        return self.search_by_vector(self.query_vec.tolist(), k, query_text=query)

    def search_by_vector(self, query_vec: list, k: int, query_text: str = "") -> list[dict]:
        self.query_vec = np.array(query_vec, dtype=float)
        return self.es_client.search_knn(
            index_name=self.index_name,
            vector_field=self.vector_field,
            query_vector=query_vec,
            k=k,
            num_candidates=500,
            source_fields=self.source_fields,
            query_text=query_text,
        )


def _apply_rrf(ranked_lists: list[list[dict]], rrf_k: int) -> list[dict]:
    """
    Applica Reciprocal Rank Fusion tramite Ranx.

    Costruisce un Run Ranx per ogni lista di hit, chiama ranx.fuse con
    method="rrf" e params={"k": rrf_k}, poi ricostruisce la lista di hit
    ordinata per score RRF decrescente.

    ranked_lists : lista di liste di hit ES, ciascuna già ordinata per rilevanza.
    rrf_k        : costante di smoothing (tipicamente 60). Valori più alti
                   riducono il vantaggio dei documenti in cima.
    """
    QUERY_ID = "q"

    # Raccogliamo tutti gli hit originali per ricostruirli dopo la fusione.
    hits_by_id: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for hit in ranked_list:
            hits_by_id.setdefault(hit["_id"], hit)

    # Scartiamo le liste vuote: un retriever può non restituire hit, e Ranx
    # solleva un errore sia su un Run vuoto sia con meno di due run.
    non_empty = [rl for rl in ranked_lists if rl]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        return list(non_empty[0])  # già ordinata per rilevanza, niente da fondere

    # Creiamo un Run Ranx per ogni lista: {query_id: {doc_id: score}}.
    runs = [
        Run({QUERY_ID: {hit["_id"]: float(hit.get("_score") or 0.0) for hit in ranked_list}})
        for ranked_list in non_empty
    ]

    # Ranx emette un NumbaTypeSafetyWarning innocuo alla prima JIT compilation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fused = ranx_fuse(runs=runs, method="rrf", params={"k": rrf_k})

    # Ricostruiamo la lista di hit con lo score RRF, ordinata decrescente.
    fused_scores: dict[str, float] = dict(fused.run[QUERY_ID])
    result = []
    for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        hit = dict(hits_by_id[doc_id])
        hit["_score"] = float(score)
        result.append(hit)
    return result


class FilteredDenseRetriever(Retriever):
    """DenseRetriever con pre-filtro terms su un campo keyword. Usato nello stage 3 del pipeline."""

    def __init__(
        self,
        es_client: ES,
        index_name: str,
        vector_field: str,
        vectorizer: Vectorizer,
        source_fields: list[str],
        filter_field: str,
        filter_values: list[str],
    ):
        self.es_client = es_client
        self.index_name = index_name
        self.vector_field = vector_field
        self.vectorizer = vectorizer
        self.source_fields = source_fields
        self.filter_field = filter_field
        self.filter_values = filter_values

    def search(self, query: str, k: int) -> list[dict]:
        return self.search_by_vector(self.vectorizer.get_embedding(query), k)

    def search_by_vector(self, query_vec: list, k: int) -> list[dict]:
        return self.es_client.search_knn_with_filter(
            index_name=self.index_name,
            vector_field=self.vector_field,
            query_vector=query_vec,
            k=k,
            num_candidates=500,
            source_fields=self.source_fields,
            filter_field=self.filter_field,
            filter_values=self.filter_values,
        )


class FilteredSparseRetriever(Retriever):
    """SparseRetriever con pre-filtro terms su un campo keyword. Usato nello stage 3 del pipeline."""

    def __init__(
        self,
        es_client: ES,
        index_name: str,
        match_field: str,
        source_fields: list[str],
        filter_field: str,
        filter_values: list[str],
    ):
        self.es_client = es_client
        self.index_name = index_name
        self.match_field = match_field
        self.source_fields = source_fields
        self.filter_field = filter_field
        self.filter_values = filter_values

    def search(self, query: str, k: int) -> list[dict]:
        return self.es_client.search_bm25_with_filter(
            index_name=self.index_name,
            query_text=query,
            size=k,
            source_fields=self.source_fields,
            match_field=self.match_field,
            filter_field=self.filter_field,
            filter_values=self.filter_values,
        )


class PipelineRetriever(Retriever):
    """
    Orchestratore del pipeline a 3 stage:
      1. stage1_retriever              → top-m candidati da wf_reati
      2. key_table lookup              → circostanziati ammessi per ogni base reato
      3. stage3_retriever_factory(...) → top-n da wf_reati_circ per ogni base reato (filtrato)

    Restituisce il pool completo (stage1 + stage3, deduplicato) ordinato per score originale.
    Il reranking finale è responsabilità del chiamante.
    """

    def __init__(
        self,
        stage1_retriever: Retriever,
        stage3_retriever_factory,   # callable: filter_values -> Retriever
        key_table: dict,
        m: int,
        n: int,
        stage3_global: Retriever | None = None,  # se impostato: stage3 globale senza mapping (ablation)
    ):
        self.stage1_retriever = stage1_retriever
        self.stage3_retriever_factory = stage3_retriever_factory
        self.key_table = key_table
        self.m = m
        self.n = n
        self.stage3_global = stage3_global

    def search(self, query: str, k: int, stage1_query: str | None = None, stage3_query: str | None = None) -> list[dict]:
        return self.search_with_debug(query, k, stage1_query, stage3_query)["pool"]

    def search_with_debug(self, query: str, k: int, stage1_query: str | None = None, stage3_query: str | None = None) -> dict:
        """
        Come search(), ma restituisce anche i risultati intermedi:
          stage1_hits      : hit da wf_reati (top-m)
          stage3_by_reato  : { combo_reati_key -> hit da wf_reati_circ (top-n) }
          pool             : stage1 + stage3 deduplicati, ordinati per score, troncati a k

        stage1_query / stage3_query permettono di cercare con query diverse nei due
        stage (es. query reduction). Se non passate, si usa `query` per entrambi.
        """
        stage1_query = stage1_query or query
        stage3_query = stage3_query or query

        if hasattr(self.stage1_retriever, 'vectorizer'):
            stage1_vec = self.stage1_retriever.vectorizer.get_embedding(stage1_query)
            stage1_hits = self.stage1_retriever.search_by_vector(stage1_vec, self.m, query_text=stage1_query)
        else:
            stage1_vec = None
            stage1_hits = self.stage1_retriever.search(stage1_query, self.m)

        # Ablation "no mapping": stage3 = unica ricerca globale sui combo (niente
        # filtro key_table), top m*n per riempire lo stesso pool di m*(1+n).
        if self.stage3_global is not None:
            if hasattr(self.stage3_global, 'search_by_vector'):
                vec = stage1_vec if (stage3_query == stage1_query and stage1_vec is not None) \
                    else self.stage3_global.vectorizer.get_embedding(stage3_query)
                stage3_hits = self.stage3_global.search_by_vector(vec, self.m * self.n)
            else:
                stage3_hits = self.stage3_global.search(stage3_query, self.m * self.n)
            pool = stage1_hits + stage3_hits
            pool.sort(key=lambda h: h.get("_score") or 0.0, reverse=True)
            return {"stage1_hits": stage1_hits, "stage3_by_reato": {}, "pool": pool[:k]}

        stage3_by_reato: dict[str, list[dict]] = {}
        circ_hits_by_key: dict[str, dict] = {}
        stage3_vec = None  # embedding di stage3_query, calcolato una volta sola

        for hit in stage1_hits:
            combo_reati_key = hit.get("_source", {}).get("combo_reati", "")
            allowed_circs = self.key_table.get(combo_reati_key, [])
            if not allowed_circs:
                continue
            stage3 = self.stage3_retriever_factory(allowed_circs)
            if hasattr(stage3, 'search_by_vector'):
                if stage3_vec is None:
                    # riusa l'embedding dello stage1 se la query è la stessa, altrimenti embedda
                    stage3_vec = stage1_vec if (stage3_query == stage1_query and stage1_vec is not None) \
                        else stage3.vectorizer.get_embedding(stage3_query)
                stage3_hits = stage3.search_by_vector(stage3_vec, self.n)
            else:
                stage3_hits = stage3.search(stage3_query, self.n)
            stage3_by_reato[combo_reati_key] = stage3_hits
            for circ_hit in stage3_hits:
                dedup_key = f"{circ_hit.get('_index', '')}/{circ_hit['_id']}"
                score = circ_hit.get("_score") or 0.0
                existing = circ_hits_by_key.get(dedup_key)
                if existing is None or score > (existing.get("_score") or 0.0):
                    circ_hits_by_key[dedup_key] = circ_hit

        pool = stage1_hits + list(circ_hits_by_key.values())
        pool.sort(key=lambda h: h.get("_score") or 0.0, reverse=True)

        return {
            "stage1_hits": stage1_hits,
            "stage3_by_reato": stage3_by_reato,
            "pool": pool[:k],
        }


class RRFRetriever(Retriever):
    def __init__(
        self,
        retriever_a: Retriever,
        retriever_b: Retriever,
        rrf_k: int,
        window_size: int,
    ):
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.rrf_k = rrf_k
        # Quanti documenti recuperare da ogni retriever prima della fusione.
        # Deve essere > k finale per dare a RRF abbastanza candidati.
        self.window_size = window_size

    def search(self, query: str, k: int) -> list[dict]:
        hits_a = self.retriever_a.search(query, self.window_size)
        hits_b = self.retriever_b.search(query, self.window_size)
        fused = _apply_rrf([hits_a, hits_b], self.rrf_k)
        return fused[:k]
