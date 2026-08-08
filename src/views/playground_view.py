import streamlit as st
from es import ES
from views.model_components import _render_model_result, _render_models_section

ERROR_EMPTY_QUERY = "Inserisci una query."


def _render_single_query_view(es_client: ES, index_names: dict[str, str], queries: list[str]):
    models = _render_models_section("compare_models", "cmp")
    st.divider()
    st.subheader("Query")
    free_query = st.toggle("Query libera", value=False)
    if free_query:
        query = st.text_area("Input utente", value="", height=120)
    else:
        query = st.selectbox("Seleziona query", queries, index=0)

    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color: #d32f2f;
            border-color: #d32f2f;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: #b71c1c;
            border-color: #b71c1c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Avvia confronto", type="primary", key="cmp_run"):
        if not query.strip():
            st.error(ERROR_EMPTY_QUERY)
            return

        st.divider()
        st.subheader("Risultati")
        for i, config in enumerate(models):
            _render_model_result(i, config, query, es_client, index_names)
