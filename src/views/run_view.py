from pathlib import Path

import pandas as pd
import streamlit as st
from es import ES
from evaluation import (
    build_gt_context,
    evaluate_from_df,
    save_evaluation_run,
)
from views.model_components import _render_models_section

INFO_UPLOAD_CSV = "Carica un CSV nel formato wf_training_set.csv."


def _job_label(cfg: dict, idx: int) -> str:
    model_type = cfg.get("model_type", "")
    match_field = cfg.get("match_field", "")
    reranker = cfg.get("reranker_model") or cfg.get("reranker")
    parts = [f"M{idx}", model_type, match_field]
    if reranker:
        parts.append(f"rr:{reranker}")
    return " · ".join(p for p in parts if p)


def _render_csv_eval_view(es_client: ES, index_names: dict[str, str]):
    models = _render_models_section("eval_models", "eval")
    st.divider()
    csv_file = st.file_uploader("Upload CSV GT", type=["csv"])
    if csv_file is None:
        st.info(INFO_UPLOAD_CSV)
        return

    if st.button("Lancia", type="primary", key="eval_run"):
        df = pd.read_csv(csv_file)
        st.session_state["eval_queue"] = [
            {"config": cfg, "status": "pending", "run_dir": None}
            for cfg in models
        ]
        st.session_state["eval_queue_df"] = df
        st.session_state["eval_queue_csv_name"] = csv_file.name
        st.rerun()

    queue: list[dict] = st.session_state.get("eval_queue", [])
    if not queue:
        return

    df = st.session_state.get("eval_queue_df")
    csv_name = st.session_state.get("eval_queue_csv_name", "")

    next_pending_idx = next(
        (i for i, item in enumerate(queue) if item["status"] == "pending"), None
    )

    progress_bar = None
    for i, item in enumerate(queue):
        label = _job_label(item["config"], i + 1)
        if item["status"] == "done":
            run_name = Path(item["run_dir"]).name if item["run_dir"] else ""
            st.success(f"{label} — completato · run: {run_name}")
        elif item["status"] == "error":
            st.error(f"{label} — errore: {item.get('error', '')}")
        elif i == next_pending_idx:
            progress_bar = st.progress(0, text=f"{label} — avvio...")
        else:
            st.info(f"{label} — in attesa")

    if next_pending_idx is not None and progress_bar is not None:
        i = next_pending_idx
        cfg = queue[i]["config"]
        label = _job_label(cfg, i + 1)

        def on_step(step, total, q_idx, m_idx, _label=label, _bar=progress_bar):
            _bar.progress(step / total, text=f"{_label}: {step}/{total} query")

        try:
            details_df, details_by_query = evaluate_from_df(
                df, [cfg], es_client, index_names, on_step=on_step,
            )
            save_info = save_evaluation_run(
                details_df=details_df,
                model_configs=[cfg],
                gt_reference=csv_name,
                gt_context=build_gt_context(details_by_query),
            )
            queue[i]["status"] = "done"
            queue[i]["run_dir"] = save_info["run_dir"]
        except Exception as exc:
            queue[i]["status"] = "error"
            queue[i]["error"] = str(exc)

        st.session_state["eval_queue"] = queue
        st.rerun()
    else:
        n_done = sum(1 for item in queue if item["status"] == "done")
        n_err = sum(1 for item in queue if item["status"] == "error")
        if n_err == 0:
            st.success(f"Tutti i {n_done} modelli completati. Vai a **Storico** per vedere i risultati.")
        else:
            st.warning(f"{n_done} completati, {n_err} con errore. Vai a **Storico** per i risultati.")
