from tqdm import tqdm
from vectorizer import Vectorizer


class Indexer:
    def __init__(self, es, dataset, embedders, dataset_type="periodi"):
        self.es = es
        self.dataset = dataset
        self.embedders = embedders
        self.dataset_type = dataset_type
        self.profile = self._build_profile(dataset_type)


    def _build_profile(self, dataset_type):
        if dataset_type == "periodi":
            return {
                "bm25_fields": {
                    "periodo": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                    "fonte_normativa": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                    "rubrica": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                    "testo": {"type": "text", "analyzer": "italian", "similarity": "BM25"},
                    "rubrica_testo": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                },
                "string_fields": {},
                "embedding_fields": ["rubrica", "testo", "rubrica_testo"],
            }

        if dataset_type == "periodi_flat":
            return {
                "bm25_fields": {
                    "periodo_content": {"type": "text", "analyzer": "italian", "similarity": "BM25"},
                },
                "string_fields": {
                    "periodo": {"type": "keyword"},
                },
                "embedding_fields": ["periodo_content"],
            }

        if dataset_type == "combo":
            return {
                "bm25_fields": {
                    "rubriche_con_periodi": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                },
                "string_fields": {
                    "rubrica_imputativa": {"type": "keyword"},
                },
                "embedding_fields": ["rubriche_con_periodi"],
            }

        if dataset_type == "reati":
            return {
                "bm25_fields": {
                    "combo_reati_content": {"type": "text", "analyzer": "italian", "similarity": "BM25"},
                },
                "string_fields": {
                    "combo_reati": {"type": "keyword"},
                },
                "embedding_fields": ["combo_reati_content"],
            }

        if dataset_type in ("reati_circostanziati_full", "reati_circostanziati_light"):
            return {
                "bm25_fields": {
                    "combo_reati_circostanziati_content": {"type": "text", "analyzer": "italian", "similarity": "BM25"},
                },
                "string_fields": {
                    "combo_reati_circostanziati": {"type": "keyword"},
                },
                "embedding_fields": ["combo_reati_circostanziati_content"],
            }

        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


    @staticmethod
    def _normalize_text(value):
        return "" if value is None else str(value).strip()


    def _build_periodi_rubrica_testo(self, record):
        rubrica = self._normalize_text(getattr(record, "rubrica", None))
        testo = self._normalize_text(getattr(record, "testo", None))
        return f"{rubrica} {testo}".strip()


    def _build_periodi_rubrica_testo_embedding_input(self, record):
        rubrica = self._normalize_text(getattr(record, "rubrica", None))
        testo = self._normalize_text(getattr(record, "testo", None))
        if not rubrica and not testo:
            return None
        return f"<rubrica> {rubrica} </rubrica> <testo> {testo} </testo>"


    def _get_text_field_value(self, record, field):
        if self.dataset_type == "periodi" and field == "rubrica_testo":
            return self._build_periodi_rubrica_testo(record)
        return getattr(record, field, None)


    def _get_embedding_input(self, record, field):
        if self.dataset_type == "periodi" and field == "rubrica_testo":
            return self._build_periodi_rubrica_testo_embedding_input(record)
        return getattr(record, field, None)


    def build_mappings(self):
        mappings = {
            "dynamic": "strict",
            "properties": {},
        }

        mappings["properties"].update(self.profile["bm25_fields"])
        mappings["properties"].update(self.profile["string_fields"])

        for embedder in self.embedders:
            dims = Vectorizer.AVAILABLE_MODELS[embedder]["dims"]
            for field in self.profile["embedding_fields"]:
                mappings["properties"][f"{embedder}_{field}"] = {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                }

        return mappings


    def index_text(self, index_name):
        text_fields = list(self.profile["bm25_fields"].keys()) + list(self.profile["string_fields"].keys())
        es_bulk_chunk_size = 500
        pending_inserts = []

        with tqdm(total=len(self.dataset.records)) as pbar:
            def flush_inserts():
                if not pending_inserts:
                    return
                errors = self.es.insert_bulk(
                    index_name=index_name,
                    items=pending_inserts,
                    chunk_size=es_bulk_chunk_size,
                )
                if errors:
                    tqdm.write(f"[WARN] insert bulk errors: {len(errors)}")
                pending_inserts.clear()

            for record in self.dataset.records:
                document = {field: self._get_text_field_value(record, field) for field in text_fields}
                pending_inserts.append((record.index, document))
                if len(pending_inserts) >= es_bulk_chunk_size:
                    flush_inserts()
                pbar.update(1)

            flush_inserts()


    def index_embeddings(self, index_name, embedder_model, skip_existing=False):
        vectorizer = Vectorizer(embedder_model)
        embedding_fields = self.profile["embedding_fields"]
        batches = {field: [] for field in embedding_fields}
        batch_ids = {field: [] for field in embedding_fields}
        embedding_batch_size = 128
        es_bulk_chunk_size = 500

        already_indexed = {}
        if skip_existing:
            for field in embedding_fields:
                es_field = f"{embedder_model}_{field}"
                tqdm.write(f"[RESUME] fetching already-indexed IDs for field '{es_field}'...")
                already_indexed[field] = self.es.get_ids_with_field(index_name, es_field)
                tqdm.write(f"[RESUME] {len(already_indexed[field])} documents will be skipped")

        with tqdm(total=len(self.dataset.records)) as pbar:
            def flush_field(field):
                if not batches[field]:
                    return
                vectors = vectorizer.get_embeddings(batches[field])
                updates = [
                    (doc_id, f"{embedder_model}_{field}", vec)
                    for doc_id, vec in zip(batch_ids[field], vectors)
                ]
                errors = self.es.update_embeddings_bulk(
                    index_name=index_name,
                    items=updates,
                    chunk_size=es_bulk_chunk_size,
                )
                if errors:
                    tqdm.write(f"[WARN] update bulk errors ({field}): {len(errors)}")
                batches[field].clear()
                batch_ids[field].clear()

            for record in self.dataset.records:
                for field in embedding_fields:
                    if skip_existing and record.index in already_indexed.get(field, set()):
                        continue
                    value = self._get_embedding_input(record, field)
                    if value:
                        batches[field].append(value)
                        batch_ids[field].append(record.index)
                        if len(batches[field]) >= embedding_batch_size:
                            flush_field(field)

                pbar.update(1)

            for field in embedding_fields:
                flush_field(field)
                


    def run(self, index_name):
        mappings = self.build_mappings()
        self.es.create_index(index_name, mappings)

        self.index_text(index_name)
        for embedder in self.embedders:
            self.index_embeddings(index_name, embedder)
