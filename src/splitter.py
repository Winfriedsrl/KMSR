"""
Splitter della query: separa una query in reato base + eventuale circostanza.

Serve per la "query reduction": cercare nell'indice dei reati base con il solo
q_reato (senza il rumore della circostanza) migliora il recall dello stage1.

Due metodi:
- regex: taglia ai marker tipici ("aggravato da", "recidiva", ...). Gratis, ~80%.
- llm:   un LLM estrae reato base e circostanza. Copre i casi senza marker
         (es. "rapina a mano armata" -> "rapina"). Una sola chiamata.

Contratto: split(query) -> {"q", "q_reato", "q_circostanza", "prompt", "answer", "parse_error"}.
Se una parte non è estraibile resta stringa vuota; chi la usa fa fallback a "q".
"""
from __future__ import annotations

import json
import re

from llm import LLM
from classifier import _load_prompt, _parse_output

# Marker che introducono una circostanza dopo il nome del reato base.
_MARK = re.compile(
    r"\b(aggravat[oa]\s+(?:da|dall'|dalla|dagli|dalle|per)|recidiva|attenuat[oa]\s+(?:da|dal|dalla))",
    re.IGNORECASE,
)


def _regex_split(query: str) -> tuple[str, str]:
    """Ritorna (q_reato, q_circostanza). Se nessun marker, q_circostanza è vuota."""
    m = _MARK.search(query)
    if not m:
        return query, ""
    q_reato = query[:m.start()].strip()
    q_circostanza = query[m.start():].strip()
    return (q_reato or query), q_circostanza


class QuerySplitter:
    def __init__(self, method: str = "regex", llm_model: str | None = None, version: str = "v1"):
        self.method = method
        if method == "llm":
            self.llm = LLM(model=llm_model)
            self._system_prompt = _load_prompt(f"splitter_{version}.txt")

    def split(self, query: str) -> dict:
        if self.method == "regex":
            q_reato, q_circostanza = _regex_split(query)
            return {"q": query, "q_reato": q_reato, "q_circostanza": q_circostanza,
                    "prompt": None, "answer": None, "parse_error": None}

        # metodo llm
        prompt = f'Query da scomporre: "{query}"'
        answer = self.llm.ask(prompt, system_prompt=self._system_prompt)
        q_reato = ""
        q_circostanza = ""
        parse_error = None
        try:
            parsed = _parse_output(answer)
            q_reato = (parsed.get("q_reato") or "").strip()
            q_circostanza = (parsed.get("q_circostanza") or "").strip()
        except Exception as exc:
            parse_error = f"Parsing splitter fallito: {exc}"
        return {"q": query, "q_reato": q_reato or query, "q_circostanza": q_circostanza,
                "prompt": prompt, "answer": answer, "parse_error": parse_error}


def select_query(parts: dict, which: str) -> str:
    """Sceglie q / q_reato / q_circostanza; fallback a q se la parte è vuota."""
    value = parts.get(which, "")
    return value if value else parts["q"]


def _trivial_parts(query: str) -> dict:
    """Parti quando non c'è splitter: tutto = query intera."""
    return {"q": query, "q_reato": query, "q_circostanza": query,
            "prompt": None, "answer": None, "parse_error": None}


def resolve_queries(query: str, config: dict, splitter: "QuerySplitter | None"):
    """Dato config e splitter (può essere None) ritorna (parts, stage1_query, stage3_query)."""
    parts = splitter.split(query) if splitter is not None else _trivial_parts(query)
    stage1_query = select_query(parts, config.get("stage1_input", "q"))
    stage3_query = select_query(parts, config.get("stage3_input", "q"))
    return parts, stage1_query, stage3_query
