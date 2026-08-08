import json
from pathlib import Path

import pandas as pd
import streamlit as st

from views.classifier_test_view import _render_results

RUNS_DIR = Path("runs/classifier_test")


def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)


def _load_meta(run_dir: Path) -> dict:
    path = run_dir / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _render_classifier_explorer_view():
    st.title("Test Classificatore — Explorer")

    runs = _list_runs()
    if not runs:
        st.info(f"Nessun run trovato in `{RUNS_DIR}`.")
        return

    rows = []
    for rd in runs:
        meta = _load_meta(rd)
        if not meta:
            continue
        rows.append({
            "Run": rd.name,
            "Data": meta.get("created_at_utc", "")[:19].replace("T", " "),
            "Prompt": meta.get("prompt_name", "?"),
            "LLM": meta.get("llm_model", "?"),
            "Query": meta.get("n_queries", "?"),
            "Accuracy": f"{100 * meta.get('accuracy', 0):.1f}%",
            "GT": meta.get("gt_reference", "?"),
        })

    table_df = pd.DataFrame(rows)
    ev = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="clf_explorer_table",
    )

    selected = ev.selection.rows
    if not selected:
        st.info("Seleziona un run per ispezionarlo.")
        return

    run_dir = runs[selected[0]]
    meta = _load_meta(run_dir)

    st.divider()
    st.markdown(f"### {run_dir.name}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Prompt:** `{meta.get('prompt_name', '?')}`")
        st.markdown(f"**LLM:** `{meta.get('llm_model', '?')}`")
        st.markdown(f"**GT:** `{meta.get('gt_reference', '?')}`")
    with col2:
        prompt_path = run_dir / "prompt.txt"
        if prompt_path.exists():
            with st.expander("Prompt di sistema usato", expanded=False):
                st.code(prompt_path.read_text(encoding="utf-8"), language="text")

    details_path = run_dir / "details.csv"
    if not details_path.exists():
        st.warning("details.csv non trovato.")
        return

    details_df = pd.read_csv(details_path, encoding="utf-8")
    _render_results(details_df, key_prefix=f"clf_exp_{run_dir.name}")
