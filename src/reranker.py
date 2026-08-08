"""
Reranker layer: prende hit già recuperati da un Retriever e li riordina con un LLM.

Allineato al ranker della pipeline reati: stesse direttive (prompt `ranker_v1.txt`),
stesso schema di output `{"ranked": [{"seq": ...}]}` e stesso parsing. Il contenuto
mostrato all'LLM è estratto in modo robusto dal campo giusto a seconda dell'indice
(periodo_content sul flat del paper, testo sui periodi legacy, rubriche sui combo).
"""
from __future__ import annotations

import json

from llm import LLM
from classifier import _load_prompt, _parse_output

RANKER_PROMPT_FILE = "ranker_v1.txt"


def _rerank_content(hit: dict) -> str:
    """Testo del documento da mostrare all'LLM, robusto rispetto all'indice."""
    src = hit.get("_source", {})
    return (
        src.get("periodo_content")          # flat del paper (contiene fonte+rubrica+testo)
        or src.get("rubriche_con_periodi")  # combo
        or src.get("testo")                 # periodi legacy
        or src.get("periodo")               # fallback: solo etichetta
        or ""
    )


def _format_candidates(hits: list[dict]) -> str:
    items = [{"seq": i + 1, "content": _rerank_content(h)} for i, h in enumerate(hits)]
    return json.dumps(items, ensure_ascii=False, indent=2)


class LLMReranker:
    """
    Reranker che usa un LLM per riordinare documenti già recuperati.

    Riceve la lista di hit, ne passa un sottoinsieme (rr) all'LLM e restituisce i
    top-k riordinati. Direttive e formato identici al ranker dei run reati.
    """

    def __init__(self, llm_model: str, match_field: str = "testo"):
        self.llm_model = llm_model
        self.match_field = match_field  # mantenuto per compatibilità di firma
        self._system_prompt = _load_prompt(RANKER_PROMPT_FILE)

    def rerank(self, query: str, hits: list[dict], rr: int, k: int) -> dict:
        """
        query : testo della query
        hits  : tutti gli hit restituiti dal retriever
        rr    : quanti hit passare all'LLM (retriever_results)
        k     : quanti risultati finali restituire
        """
        candidates = hits[:rr]
        context = _format_candidates(candidates)
        prompt = (
            f'Query: "{query}"\n\n'
            f"Documenti (JSON):\n{context}\n\n"
            f"Restituisci i top-{k} seq più rilevanti.\n"
            "Output — SOLO questo JSON:\n"
            '{"ranked": [{"seq": <intero>}, ...]}'
        )

        llm = LLM(model=self.llm_model)
        answer = llm.ask(prompt, system_prompt=self._system_prompt)

        generation_error = None
        ranked_hits: list[dict] = []
        try:
            parsed = _parse_output(answer)
            seen: set[int] = set()
            for item in parsed.get("ranked", []):
                seq = int(item["seq"])
                if seq < 1 or seq > len(candidates) or seq in seen:
                    continue
                seen.add(seq)
                ranked_hits.append(candidates[seq - 1])
        except Exception as exc:
            generation_error = f"Parsing/rerank LLM fallito: {exc}"

        if not ranked_hits:
            ranked_hits = candidates[:k]
            generation_error = (generation_error or "") + " Uso retrieval ranking come fallback."
        ranked_hits = ranked_hits[:k]

        try:
            answer_pretty = json.dumps(_parse_output(answer), ensure_ascii=False, indent=2)
        except Exception:
            answer_pretty = answer

        return {
            "error": None,
            "embed_error": None,
            "generation_error": generation_error,
            "hits": hits,
            "ranked_hits": ranked_hits,
            "query_vec": None,
            "doc_vecs": None,
            "answer": answer,
            "answer_pretty": answer_pretty,
            "prompt": prompt,
        }
