"""
Componenti LLM per classificazione e reranking.

- LLMTypeClassifier: classifica il tipo della query (reato / reato_circostanziato)
                     senza un pool di documenti.
- LLMRanker: ordina un pool omogeneo di documenti per rilevanza rispetto alla query.
- LLMClassifierReranker: versione combinata (pipeline pool) — classifica il tipo
                          e seleziona i top-k documenti del tipo scelto in una sola
                          chiamata LLM. Riceve un pool misto.
"""
from __future__ import annotations

import json
from pathlib import Path

from llm import LLM

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def _hit_type(hit: dict) -> str:
    src = hit.get("_source", {})
    circ = src.get("combo_reati_circostanziati", "")
    return "reato_circostanziato" if circ and str(circ).strip().lower() != "nan" else "reato"


def _hit_content(hit: dict) -> str:
    src = hit.get("_source", {})
    return (
        src.get("combo_reati_circostanziati_content")
        or src.get("combo_reati_content")
        or src.get("combo_reati_circostanziati")
        or src.get("combo_reati")
        or ""
    )


def _format_pool(hits: list[dict]) -> str:
    items = [
        {"seq": i + 1, "type": _hit_type(h), "content": _hit_content(h)}
        for i, h in enumerate(hits)
    ]
    return json.dumps(items, ensure_ascii=False, indent=2)


def _parse_output(answer: str) -> dict:
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start:end + 1])


class LLMTypeClassifier:
    """Classifica il tipo della query (reato / reato_circostanziato) senza pool."""

    def __init__(self, llm_model: str, classifier_version: str = "v1"):
        self.llm = LLM(model=llm_model)
        self._system_prompt = _load_prompt(f"only_classifier_{classifier_version}.txt")

    def classify_type(self, query: str) -> dict:
        prompt = f'Query da classificare: "{query}"'
        answer = self.llm.ask(prompt, system_prompt=self._system_prompt)
        predicted_type = None
        reasoning = None
        parse_error = None
        try:
            parsed = _parse_output(answer)
            predicted_type = parsed.get("type")
            reasoning = parsed.get("reasoning")
        except Exception as exc:
            parse_error = f"Parsing classificatore fallito: {exc}"
        return {
            "predicted_type": predicted_type,
            "reasoning": reasoning,
            "prompt": prompt,
            "answer": answer,
            "parse_error": parse_error,
        }


class LLMRanker:
    """Ordina un pool omogeneo di documenti per rilevanza rispetto alla query."""

    def __init__(self, llm_model: str):
        self.llm = LLM(model=llm_model)
        self._system_prompt = _load_prompt("ranker_v1.txt")

    def rerank(self, query: str, hits: list[dict], k: int) -> dict:
        pool_json = _format_pool(hits)
        prompt = (
            f'Query: "{query}"\n\n'
            f"Documenti (JSON):\n{pool_json}\n\n"
            f"Restituisci i top-{k} seq più rilevanti.\n"
            "Output — SOLO questo JSON:\n"
            '{"ranked": [{"seq": <intero>}, ...]}'
        )
        answer = self.llm.ask(prompt, system_prompt=self._system_prompt)
        ranked_hits: list[dict] = []
        parse_error = None
        try:
            parsed = _parse_output(answer)
            seen: set[int] = set()
            for item in parsed.get("ranked", []):
                seq = int(item["seq"])
                if seq < 1 or seq > len(hits) or seq in seen:
                    continue
                seen.add(seq)
                ranked_hits.append(hits[seq - 1])
        except Exception as exc:
            parse_error = f"Parsing reranker fallito: {exc}"
        if not ranked_hits:
            ranked_hits = hits[:k]
            parse_error = (parse_error or "") + " Fallback: ordine retriever."
        return {
            "hits": ranked_hits[:k],
            "prompt": prompt,
            "answer": answer,
            "parse_error": parse_error,
        }


class LLMClassifierReranker:
    """
    Pipeline pool: classifica il tipo e seleziona i top-k del tipo scelto
    in una singola chiamata LLM. Riceve un pool misto (reati + circostanziati).
    """

    def __init__(self, llm_model: str, classifier_version: str = "v1"):
        self.llm = LLM(model=llm_model)
        self._system_prompt = _load_prompt(f"classifier_{classifier_version}.txt")

    def classify(self, query: str, hits: list[dict], k: int) -> dict:
        pool_json = _format_pool(hits)
        prompt = (
            f"Query da classificare: \"{query}\"\n\n"
            f"Pool di documenti (JSON):\n{pool_json}\n\n"
            "Task:\n"
            "1. Leggi il contenuto dei documenti nel pool: quelli con type 'reato' mostrano come si "
            "presenta un reato base in questo dominio; quelli con type 'reato_circostanziato' mostrano "
            "come si presenta un reato con circostanze specifiche. Usali come riferimento per orientarti, "
            "non per contarli — guarda cosa dicono, non quanti sono.\n"
            "2. Analizza la query confrontandola con questi esempi: assomiglia al contenuto dei reati base "
            "o a quello dei circostanziati?\n"
            "3. Scegli il tipo: 'reato' se la query è nella forma base, 'reato_circostanziato' se "
            "specifica circostanze aggravanti, attenuanti o modalità qualificanti.\n"
            f"4. Ordina per rilevanza TUTTI i documenti del tipo scelto e includi i primi {k} nel campo ranked.\n"
            f"   Se i documenti del tipo scelto sono meno di {k}, includili tutti.\n"
            "   ATTENZIONE: i seq nel campo 'ranked' devono avere tutti type uguale al valore scelto.\n\n"
            "Output — SOLO questo JSON:\n"
            '{"reasoning": "<una frase che spiega la scelta del tipo>", '
            '"type": "reato" oppure "reato_circostanziato", '
            '"ranked": [{"seq": <intero>}, ...]}'
        )
        answer = self.llm.ask(prompt, system_prompt=self._system_prompt)

        predicted_type = None
        ranked_hits: list[dict] = []
        parse_error = None
        reasoning = None

        try:
            parsed = _parse_output(answer)
            predicted_type = parsed.get("type")
            reasoning = parsed.get("reasoning")
            seen: set[int] = set()
            for item in parsed.get("ranked", []):
                try:
                    seq = int(item["seq"])
                except Exception:
                    continue
                if seq < 1 or seq > len(hits) or seq in seen:
                    continue
                seen.add(seq)
                ranked_hits.append(hits[seq - 1])
        except Exception as exc:
            parse_error = f"Parsing classificatore fallito: {exc}"

        if predicted_type:
            seen_ids = {h["_id"] for h in ranked_hits}
            remaining = [
                h for h in hits
                if _hit_type(h) == predicted_type and h["_id"] not in seen_ids
            ]
            ranked_hits = (ranked_hits + remaining)[:k]

        if not ranked_hits:
            ranked_hits = hits[:k]
            parse_error = (parse_error or "") + " Fallback su pool completo: tipo non determinato."

        return {
            "predicted_type": predicted_type,
            "reasoning": reasoning,
            "hits": ranked_hits[:k],
            "prompt": prompt,
            "answer": answer,
            "parse_error": parse_error,
        }


# Alias per compatibilità con il codice esistente
LLMClassifier = LLMClassifierReranker
