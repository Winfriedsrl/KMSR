import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ES:
    def __init__(self, host="http://localhost:9201", timeout=600, verbose=True):
        self.es = Elasticsearch(host, request_timeout=timeout)
        self.verbose = verbose

    def index_exists(self, index_name):
        return bool(self.es.indices.exists(index=index_name))

    def create_index(self, index_name, mappings):
        kwargs = {"index": index_name, "mappings": mappings}
        self.es.indices.create(**kwargs)

    def insert(self, index_name, doc_id, document):
        self.es.index(index=index_name, id=doc_id, document=document)

    def insert_bulk(self, index_name, items, chunk_size=500, max_retries=2):
        actions = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": doc_id,
                "_source": document,
            }
            for doc_id, document in items
        ]
        return self._bulk_with_retries(actions, chunk_size=chunk_size, max_retries=max_retries)


    def update_embedding(self, index_name, doc_id, embedding_field, embedding):
        update_query = {"doc": {embedding_field: embedding}}
        self.es.update(index=index_name, id=doc_id, body=update_query)

    def update_embeddings_bulk(self, index_name, items, chunk_size=500, max_retries=2):
        actions = [
            {
                "_op_type": "update",
                "_index": index_name,
                "_id": doc_id,
                "doc": {embedding_field: embedding},
            }
            for doc_id, embedding_field, embedding in items
        ]
        return self._bulk_with_retries(actions, chunk_size=chunk_size, max_retries=max_retries)

    def _bulk_with_retries(self, actions, chunk_size=500, max_retries=2):
        if not actions:
            return []

        errors = []
        for _ in range(max_retries + 1):
            success_count, bulk_errors = bulk(
                self.es,
                actions,
                chunk_size=chunk_size,
                raise_on_error=False,
                raise_on_exception=False,
            )
            _ = success_count

            if not bulk_errors:
                return []
            errors = bulk_errors

        return errors

    def search_knn(self, index_name, vector_field, query_vector, k, num_candidates, source_fields, query_text=None):
        if query_text is not None:
            if self.verbose: print(f"[ES] q={query_text}", flush=True, file=sys.stderr)
        response = self.es.search(
            index=index_name,
            size=k,
            knn={
                "field": vector_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
            },
            _source=source_fields,
        )
        return response.get("hits", {}).get("hits", [])

    def search_knn_with_filter(self, index_name, vector_field, query_vector, k, num_candidates, source_fields, filter_field, filter_values):
        response = self.es.search(
            index=index_name,
            size=k,
            knn={
                "field": vector_field,
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
                "filter": {"terms": {filter_field: filter_values}},
            },
            _source=source_fields,
        )
        return response.get("hits", {}).get("hits", [])

    def search_bm25_with_filter(self, index_name, query_text, size, source_fields, match_field, filter_field, filter_values):
        response = self.es.search(
            index=index_name,
            size=size,
            query={
                "bool": {
                    "must": {"match": {match_field: query_text}},
                    "filter": {"terms": {filter_field: filter_values}},
                }
            },
            _source=source_fields,
        )
        return response.get("hits", {}).get("hits", [])

    def search_bm25(self, index_name, query_text, size, source_fields, match_fields):
        if self.verbose: print(f"[ES] q={query_text}", flush=True, file=sys.stderr)
        response = self.es.search(
            index=index_name,
            size=size,
            query={
                "multi_match": {
                    "query": query_text,
                    "fields": match_fields,
                }
            },
            _source=source_fields,
        )
        return response.get("hits", {}).get("hits", [])

    def search_bm25_weighted(self, index_name, query_text, size, source_fields, fields, weights):
        """
        BM25 search su più campi con pesi normalizzati.

        fields  : lista di campi ES, es. ["rubrica", "testo"]
        weights : lista di pesi relativi, es. [2.0, 1.0]
                  vengono normalizzati internamente: ogni peso diventa w / sum(weights),
                  quindi [2.0, 1.0] → [0.667, 0.333]

        Usa multi_match con type=most_fields, che somma i contributi BM25
        di tutti i campi pesati invece di prendere solo il migliore.
        """
        if self.verbose: print(f"[ES] weighted q={query_text}", flush=True, file=sys.stderr)
        active_pairs = [(f, w) for f, w in zip(fields, weights) if w > 0]
        total = sum(w for _, w in active_pairs)
        boosted_fields = [f"{f}^{w / total:.6f}" for f, w in active_pairs]
        response = self.es.search(
            index=index_name,
            size=size,
            query={
                "multi_match": {
                    "query": query_text,
                    "fields": boosted_fields,
                    "type": "most_fields",
                }
            },
            _source=source_fields,
        )
        return response.get("hits", {}).get("hits", [])

    def get_vectors_by_ids(self, index_name, doc_ids, vector_field):
        if not doc_ids:
            return {}
        response = self.es.mget(
            index=index_name,
            ids=doc_ids,
            _source_includes=[vector_field],
        )
        vectors = {}
        for doc in response.get("docs", []):
            doc_id = doc.get("_id")
            source = doc.get("_source", {})
            vectors[doc_id] = source.get(vector_field)
        return vectors

    def get_ids_with_field(self, index_name, field):
        already_indexed = set()
        resp = self.es.search(
            index=index_name,
            query={"exists": {"field": field}},
            source=False,
            size=10000,
            scroll="2m",
        )
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
        while hits:
            for hit in hits:
                already_indexed.add(hit["_id"])
            resp = self.es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp["_scroll_id"]
            hits = resp["hits"]["hits"]
        self.es.clear_scroll(scroll_id=scroll_id)
        return already_indexed

    def get_index_properties(self, index_name):
        response = self.es.indices.get_mapping(index=index_name)
        first_mapping = next(iter(response.values()))
        mappings = first_mapping.get("mappings", {})
        return mappings.get("properties", {})

    
