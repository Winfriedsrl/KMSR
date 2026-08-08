import streamlit as st
from es import ES
from config import (
    DEFAULT_ES_HOST,
    DEFAULT_INDEX_NAME,
    DEFAULT_INDEX_NAME_COMBO,
    DEFAULT_INDEX_NAME_REATI,
    DEFAULT_INDEX_NAME_REATI_CIRC_LIGHT,
    DEFAULT_INDEX_NAME_REATI_CIRC_FULL,
    DEFAULT_INDEX_PAPER_PERIODI_FLAT,
    DEFAULT_INDEX_PAPER_REATI,
    DEFAULT_INDEX_PAPER_REATI_CIRC_LIGHT,
    load_queries,
    load_key_table,
)
from views.playground_view import _render_single_query_view
from views.run_view import _render_csv_eval_view
from views.metrics_view import _render_metrics_formulas_view
from views.explorer_view import _render_runs_explorer_view
from views.run_pipeline_view import _render_run_pipeline_view
from views.explorer_pipeline_view import _render_explorer_pipeline_view
from views.classifier_test_view import _render_classifier_run_view
from views.classifier_test_explorer_view import _render_classifier_explorer_view
from views.paper_run_view import _render_paper_run_view, _render_paper_hard_run_view
from views.paper_explorer_view import _render_paper_explorer_view, _render_paper_hard_explorer_view


def main():
    st.set_page_config(page_title="Model comparison", layout="wide")
    es_host = DEFAULT_ES_HOST
    index_names = {
        "periodi": DEFAULT_INDEX_NAME,
        "combo": DEFAULT_INDEX_NAME_COMBO,
    }
    paper_index_names = {
        "periodi_flat": DEFAULT_INDEX_PAPER_PERIODI_FLAT,
        "reati": DEFAULT_INDEX_PAPER_REATI,
        "reati_circ_light": DEFAULT_INDEX_PAPER_REATI_CIRC_LIGHT,
    }
    queries = load_queries("data/wf_queries.csv")
    es_client = ES(host=es_host)
    view_name = st.sidebar.radio(
        "View",
        [
            "Playground",
            "Run",
            "Explorer",
            "Metrics",
            "Run Reati",
            "Explorer Reati",
            "CLF Run",
            "CLF Explorer",
            "Paper Run",
            "Paper Explorer",
            "Paper Hard Run",
            "Paper Hard Explore",
        ],
        index=0,
    )

    if view_name == "Playground":
        _render_single_query_view(es_client, index_names, queries)
    elif view_name == "Run":
        _render_csv_eval_view(es_client, index_names)
    elif view_name == "Explorer":
        _render_runs_explorer_view()
    elif view_name == "Metrics":
        _render_metrics_formulas_view()
    elif view_name == "Run Reati":
        key_table = load_key_table()
        _render_run_pipeline_view(
            es_client,
            DEFAULT_INDEX_NAME_REATI,
            DEFAULT_INDEX_NAME_REATI_CIRC_LIGHT,
            DEFAULT_INDEX_NAME_REATI_CIRC_FULL,
            key_table,
        )
    elif view_name == "Explorer Reati":
        _render_explorer_pipeline_view()
    elif view_name == "CLF Run":
        _render_classifier_run_view()
    elif view_name == "CLF Explorer":
        _render_classifier_explorer_view()
    elif view_name == "Paper Run":
        _render_paper_run_view(es_client, paper_index_names)
    elif view_name == "Paper Explorer":
        _render_paper_explorer_view()
    elif view_name == "Paper Hard Run":
        _render_paper_hard_run_view(es_client, paper_index_names)
    else:
        _render_paper_hard_explorer_view()


if __name__ == "__main__":
    main()
