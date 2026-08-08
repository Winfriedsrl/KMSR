"""Grafici 3D e tabelle dei risultati per la UI."""
import numpy as np
import pandas as pd
import streamlit as st
import umap

from config import SearchHit

try:
    import plotly.graph_objects as go
except Exception:
    go = None


def _wrap_text(text: str, width: int):
    if not text:
        return [""]
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current_len + extra > width and current:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return lines


def build_hover_texts(hits: list[SearchHit], match_field: str):
    texts = []
    is_combo = match_field == "rubriche_con_periodi"
    for hit in hits:
        src = hit.get("_source", {})
        if is_combo:
            rubriche_text = src.get("rubriche_con_periodi") or ""
            imputativa = src.get("rubrica_imputativa") or "None"
            wrapped = "<br>".join(_wrap_text(rubriche_text, 120))
            texts.append(
                f"<b>Rubrica imputativa</b>: {imputativa}<br>"
                f"<b>Rubriche con periodi</b>: {wrapped}"
            )
        else:
            rubrica = src.get("rubrica") if src.get("rubrica") is not None else "None"
            testo = src.get("testo") if src.get("testo") is not None else ""
            testo_wrapped = "<br>".join(_wrap_text(testo, 120))
            texts.append(f"<b>Rubrica</b>: {rubrica}<br><b>Testo</b>: {testo_wrapped}")
    return texts


def build_results_table(hits: list[SearchHit], top_k: int, match_field: str):
    rows = []
    is_combo = match_field == "rubriche_con_periodi"
    for rank, hit in enumerate(hits[:top_k], start=1):
        src = hit.get("_source", {})
        base = {
            "rank": rank,
            "score": hit.get("_score"),
            "id": hit.get("_id"),
        }
        if is_combo:
            base["rubriche_con_periodi"] = src.get("rubriche_con_periodi")
            base["rubrica_imputativa"] = src.get("rubrica_imputativa")
        else:
            base["periodo"] = src.get("periodo")
            base["fonte_normativa"] = src.get("fonte_normativa")
            base["rubrica"] = src.get("rubrica")
            base["testo"] = src.get("testo")
        rows.append(base)
    return pd.DataFrame(rows)


def reduce_to_3d(matrix: np.ndarray):
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(matrix).astype(float)


def build_plot(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int, chart_key: str, hover_texts: list[str]):
    if go is None:
        st.warning("Plotly non disponibile. Salta il grafico 3D.")
        return

    if doc_vecs.size == 0:
        st.info("Nessun punto da mostrare nel grafico.")
        return

    all_vecs = np.vstack([query_vec.reshape(1, -1), doc_vecs])
    coords = reduce_to_3d(all_vecs)
    q = coords[0]
    docs = coords[1:]

    top_k = min(top_k, docs.shape[0])
    top = docs[:top_k]
    other = docs[top_k:]
    hover_top = hover_texts[:top_k]
    hover_other = hover_texts[top_k:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[q[0]],
            y=[q[1]],
            z=[q[2]],
            mode="markers",
            marker={"size": 6, "color": "red"},
            name="Query",
        )
    )

    if top.size:
        fig.add_trace(
            go.Scatter3d(
                x=top[:, 0],
                y=top[:, 1],
                z=top[:, 2],
                mode="markers",
                marker={"size": 4, "color": "blue"},
                hovertext=hover_top,
                hovertemplate="%{hovertext}<extra></extra>",
                name="TopK",
            )
        )

    if other.size:
        fig.add_trace(
            go.Scatter3d(
                x=other[:, 0],
                y=other[:, 1],
                z=other[:, 2],
                mode="markers",
                marker={"size": 3, "color": "gold"},
                hovertext=hover_other,
                hovertemplate="%{hovertext}<extra></extra>",
                name="Altri",
            )
        )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        height=500,
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
