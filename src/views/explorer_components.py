import textwrap
from typing import Any

import pandas as pd
import streamlit as st
from evaluation import build_query_model_heatmap_df, extract_hit_period

try:
    import plotly.express as px
except Exception:
    px = None


def _display_period_key(key: str) -> str:
    parts = key.split("\x00", 1)
    if len(parts) == 2 and parts[1]:
        return f"{parts[0]} · {parts[1]}"
    return parts[0]


def _render_eval_details(selected_query: str, details: dict[str, Any]):
    st.markdown("### Dettaglio query")
    st.write(f"Query: {selected_query}")
    st.write(f"Query Type: {details['query_type']}")

    col_gt, col_models = st.columns([2, 3], gap="large")

    with col_gt:
        with st.expander(f"Ground Truth raw ({len(details['raw_results'])})", expanded=False):
            for idx, result_row in enumerate(details["raw_results"], start=1):
                st.write(f"{idx}. {result_row}")

        with st.expander(f"GT periodi flatten ({len(details['gt_periods'])})", expanded=False):
            for idx, period in enumerate(details["gt_periods"], start=1):
                st.write(f"{idx}. {_display_period_key(period)}")

        qt = str(details.get("query_type", "")).strip().upper()
        if qt != "A":
            gt_blocks = details.get("gt_blocks", [])
            with st.expander(f"GT blocchi ({len(gt_blocks)})", expanded=False):
                for idx, block in enumerate(gt_blocks, start=1):
                    st.write(f"{idx}. {' | '.join(_display_period_key(p) for p in block)}")

    with col_models:
        model_indexes = sorted(details["models"].keys())
        tabs = st.tabs([f"M{idx + 1}" for idx in model_indexes])

        qt = str(details.get("query_type", "")).strip().upper()
        show_flat = qt != "B"
        show_block = qt != "A"

        for tab, model_idx in zip(tabs, model_indexes):
            model_detail = details["models"][model_idx]
            with tab:
                metric_cols = st.columns(2 * (show_flat + show_block))
                col_iter = iter(metric_cols)
                if show_flat:
                    with next(col_iter):
                        st.metric("Flat Recall", f"{model_detail['flat_recall']:.3f}")
                    with next(col_iter):
                        st.metric("NDCG", f"{model_detail.get('ndcg', 0.0):.3f}")
                if show_block:
                    with next(col_iter):
                        st.metric("Block Recall", f"{model_detail['block_recall']:.3f}")
                    with next(col_iter):
                        st.metric("Block AP", f"{model_detail.get('block_ap', 0.0):.3f}")

                if model_detail["error"]:
                    st.error(model_detail["error"])
                    continue

                if model_detail.get("generation_error"):
                    st.warning(model_detail["generation_error"])

                detail_rows = []
                for rank, hit in enumerate(model_detail["hits"], start=1):
                    src = hit.get("_source", {})
                    fonte = src.get("fonte_normativa") or ""
                    detail_rows.append(
                        {
                            "rank": rank,
                            "periodo": extract_hit_period(hit),
                            "fonte": fonte if fonte else "—",
                            "score": hit.get("_score"),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(detail_rows),
                    use_container_width=True,
                    hide_index=True,
                )


def _build_eval_results_column_config(results_df: pd.DataFrame):
    column_config = {
        "Query": st.column_config.TextColumn("Query", width="medium"),
        "AVG Recall": st.column_config.NumberColumn("AVG Recall", width="small"),
        "AVG Block Recall": st.column_config.NumberColumn("AVG Block Recall", width="small"),
        "AVG NDCG": st.column_config.NumberColumn("AVG NDCG", width="small"),
        "AVG Block AP": st.column_config.NumberColumn("AVG Block AP", width="small"),
        "Max Recall": st.column_config.NumberColumn("Max Recall", width="small"),
        "Max Block Recall": st.column_config.NumberColumn("Max Block Recall", width="small"),
    }
    for col in results_df.columns:
        if col.startswith(("Recall M", "Block Recall M", "NDCG M", "Block AP M")):
            column_config[col] = st.column_config.NumberColumn(col, width="small")
    return column_config


def _preview_heatmap_height(
    n_rows: int,
    *,
    row_height_px: int = 20,
    base_padding_px: int = 70,
    min_height_px: int = 120,
    max_height_px: int = 320,
) -> int:
    if n_rows <= 0:
        return min_height_px
    height = base_padding_px + (n_rows * row_height_px)
    return max(min_height_px, min(max_height_px, height))


def _format_heatmap_y_labels(
    labels: list[str],
    *,
    wrap_width: int = 34,
    max_lines: int = 2,
) -> list[str]:
    formatted: list[str] = []
    for label in labels:
        lines = textwrap.wrap(
            label,
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not lines:
            formatted.append(label)
            continue
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if not lines[-1].endswith("..."):
                lines[-1] = f"{lines[-1].rstrip('.')}..."
        formatted.append("<br>".join(lines))
    return formatted


def _render_query_type_heatmap(query_type_df: pd.DataFrame, query_type: str):
    heatmap_df = build_query_model_heatmap_df(query_type_df, query_type=query_type)
    if heatmap_df.empty:
        return

    if px is not None:
        full_labels = [str(v) for v in heatmap_df.index.tolist()]
        display_heatmap_df = heatmap_df.copy()
        display_heatmap_df.index = _format_heatmap_y_labels(full_labels)
        fig = px.imshow(
            display_heatmap_df,
            color_continuous_scale="RdYlGn",
            zmin=0.0,
            zmax=1.0,
            aspect="auto",
            labels={"x": "Modello", "y": "Query", "color": "Recall"},
        )
        customdata = [[label for _ in display_heatmap_df.columns] for label in full_labels]
        fig.update_traces(
            customdata=customdata,
            hovertemplate=(
                "Query: %{customdata}<br>"
                "Modello: %{x}<br>"
                "Recall: %{z:.3f}<extra></extra>"
            ),
        )
        preview_height = _preview_heatmap_height(
            len(heatmap_df.index),
            row_height_px=22,
            min_height_px=120,
            max_height_px=480,
        )
        fig.update_layout(
            height=preview_height,
            margin={"l": 220, "r": 0, "t": 20, "b": 0},
        )
        fig.update_yaxes(automargin=True, tickfont={"size": 10})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(heatmap_df, use_container_width=True)


def _filter_columns_by_query_type(df: pd.DataFrame, query_type: str) -> pd.DataFrame:
    qt = str(query_type).strip().upper()
    drop: list[str] = []
    for col in df.columns:
        is_flat = col in {"AVG Recall", "Max Recall"} or col.startswith("Recall M")
        is_ndcg = col == "AVG NDCG" or col.startswith("NDCG M")
        is_block_recall = col in {"AVG Block Recall", "Max Block Recall"} or col.startswith("Block Recall M")
        is_block_ap = col == "AVG Block AP" or col.startswith("Block AP M")
        if qt == "A" and (is_block_recall or is_block_ap):
            drop.append(col)
        elif qt == "B" and (is_flat or is_ndcg):
            drop.append(col)
    return df.drop(columns=drop, errors="ignore")


def _render_query_type_section(
    query_type: str,
    query_type_df: pd.DataFrame,
    column_config: dict[str, Any],
    key_prefix: str = "",
):
    n_q = len(query_type_df)
    st.markdown(f"#### Query Type {query_type} ({n_q})")
    display_df = query_type_df.drop(columns=["Query Type"], errors="ignore")
    display_df = _filter_columns_by_query_type(display_df, query_type)

    col_table, col_heatmap = st.columns([3, 2], gap="large")
    with col_table:
        table_event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            on_select="rerun",
            selection_mode="single-row",
            key=f"{key_prefix}eval_results_dataframe_{query_type}",
        )

    with col_heatmap:
        _render_query_type_heatmap(query_type_df, query_type)

    selected_rows = table_event.selection.rows
    if not selected_rows:
        return None

    return str(query_type_df.iloc[selected_rows[0]]["Query"])


def _render_results_by_query_type(
    results_df: pd.DataFrame,
    details_by_query: dict[str, Any],
    key_prefix: str = "",
):
    column_config = _build_eval_results_column_config(results_df)
    selected_query_key = f"{key_prefix}eval_selected_query"
    selected_query_type_key = f"{key_prefix}eval_selected_query_type"
    prev_selection_by_type_key = f"{key_prefix}eval_prev_selection_by_type"

    query_types = list(results_df["Query Type"].dropna().unique())
    preferred_order = {"A": 0, "B": 1, "A+B": 2}
    query_types.sort(key=lambda value: (preferred_order.get(str(value), 3), str(value)))

    prev_selection_by_type = st.session_state.get(prev_selection_by_type_key, {})
    current_selection_by_type: dict[str, str | None] = {}

    for query_type in query_types:
        query_type_df = results_df[results_df["Query Type"] == query_type].reset_index(drop=True)
        selected_query = _render_query_type_section(
            query_type=query_type,
            query_type_df=query_type_df,
            column_config=column_config,
            key_prefix=key_prefix,
        )
        current_selection_by_type[str(query_type)] = selected_query

        if selected_query is not None and selected_query != prev_selection_by_type.get(str(query_type)):
            st.session_state[selected_query_key] = selected_query
            st.session_state[selected_query_type_key] = str(query_type)

    st.session_state[prev_selection_by_type_key] = current_selection_by_type

    st.divider()
    col_title, col_reset = st.columns([6, 1], gap="small")

    with col_reset:
        if st.button("Reset selezione", key=f"{key_prefix}eval_reset_selection"):
            st.session_state.pop(selected_query_key, None)
            st.session_state.pop(selected_query_type_key, None)
            st.rerun()

    selected_query = st.session_state.get(selected_query_key)
    selected_query_type = st.session_state.get(selected_query_type_key)

    if not selected_query or selected_query not in details_by_query:
        st.info("Seleziona una query da una tabella per vedere il dettaglio.")
        return

    details = details_by_query[selected_query]
    detail_type_label = selected_query_type or details.get("query_type", "")
    with st.expander(
        f"Dettaglio query (Type {detail_type_label}): {selected_query}",
        expanded=True,
    ):
        _render_eval_details(selected_query, details)
