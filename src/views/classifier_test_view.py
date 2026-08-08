import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from llm import LLM
from classifier import _parse_output


PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
RUNS_DIR = Path("runs/classifier_test")
LLM_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5.1"]


def _available_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("only_classifier_*.txt"))


def _build_gt_type_map(df: pd.DataFrame) -> dict[str, str]:
    gt_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        query = str(row.get("query", "")).strip()
        if not query or query.lower() == "nan":
            continue
        combo_circ = str(row.get("combo_reati_circostanziati", "")).strip()
        has_circ = combo_circ and combo_circ.lower() != "nan"
        combo_reati = str(row.get("combo_reati", "")).strip()
        has_base_only = not has_circ and combo_reati and combo_reati.lower() != "nan"
        entry = gt_map.setdefault(query, {"has_circ": False, "has_base_only": False})
        if has_circ:
            entry["has_circ"] = True
        if has_base_only:
            entry["has_base_only"] = True
    type_map = {}
    for q, flags in gt_map.items():
        if flags["has_circ"] and flags["has_base_only"]:
            type_map[q] = "mixed"
        elif flags["has_circ"]:
            type_map[q] = "reato_circostanziato"
        else:
            type_map[q] = "reato"
    return type_map


def _classify_query(llm: LLM, system_prompt: str, query: str) -> dict:
    try:
        answer = llm.ask(f'Classifica questa query: "{query}"', system_prompt=system_prompt)
        parsed = _parse_output(answer)
        return {
            "predicted_type": parsed.get("type"),
            "reasoning": parsed.get("reasoning", ""),
            "error": None,
        }
    except Exception as exc:
        return {"predicted_type": None, "reasoning": "", "error": str(exc)}


def _save_run(details_df: pd.DataFrame, prompt_name: str, prompt_text: str, llm_model: str, gt_reference: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    run_dir = RUNS_DIR / now.strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({
        "created_at_utc": now.isoformat(),
        "prompt_name": prompt_name,
        "llm_model": llm_model,
        "gt_reference": gt_reference,
        "n_queries": len(details_df),
        "accuracy": round(float((details_df["✓"] == "✓").mean()), 4) if len(details_df) else 0.0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    details_df.to_csv(run_dir / "details.csv", index=False)
    return run_dir


def _default_config() -> dict:
    prompts = _available_prompts()
    return {"prompt_name": prompts[0] if prompts else "", "llm_model": "gpt-4o-mini", "custom": False, "custom_text": ""}


def _render_config_list(session_key: str) -> list[dict]:
    if session_key not in st.session_state:
        st.session_state[session_key] = [_default_config()]

    configs: list[dict] = st.session_state[session_key]
    prompts = _available_prompts()

    for i, cfg in enumerate(configs):
        with st.expander(f"Config {i + 1} — {cfg.get('prompt_name','?')} · {cfg.get('llm_model','?')}", expanded=True):
            use_custom = st.checkbox("Prompt custom", value=cfg.get("custom", False), key=f"{session_key}_custom_{i}")
            cfg["custom"] = use_custom

            col1, col2 = st.columns(2)
            with col1:
                cfg["llm_model"] = st.selectbox("Modello LLM", LLM_OPTIONS,
                    index=LLM_OPTIONS.index(cfg["llm_model"]) if cfg["llm_model"] in LLM_OPTIONS else 0,
                    key=f"{session_key}_llm_{i}")
            with col2:
                if not use_custom:
                    cfg["prompt_name"] = st.selectbox("Versione prompt", prompts,
                        index=prompts.index(cfg["prompt_name"]) if cfg["prompt_name"] in prompts else 0,
                        key=f"{session_key}_pname_{i}")
                else:
                    cfg["prompt_name"] = "custom"

            if use_custom:
                default_text = cfg.get("custom_text") or (
                    (PROMPTS_DIR / "only_classifier_v1.txt").read_text(encoding="utf-8").strip()
                    if (PROMPTS_DIR / "only_classifier_v1.txt").exists() else ""
                )
                cfg["custom_text"] = st.text_area("Prompt", value=default_text, height=250, key=f"{session_key}_txt_{i}")

            prompt_text = cfg["custom_text"] if use_custom else (PROMPTS_DIR / f"{cfg['prompt_name']}.txt").read_text(encoding="utf-8").strip()
            with st.expander("Anteprima prompt", expanded=False):
                st.code(prompt_text, language="text")
            st.caption(f'User message: `Classifica questa query: "{{query}}"`')

        if len(configs) > 1:
            if st.button(f"Rimuovi config {i + 1}", key=f"{session_key}_rm_{i}"):
                configs.pop(i)
                st.session_state[session_key] = configs
                st.rerun()

    if st.button("+ Aggiungi config", key=f"{session_key}_add"):
        configs.append(_default_config())
        st.session_state[session_key] = configs
        st.rerun()

    st.session_state[session_key] = configs
    return configs


def _render_classifier_run_view():
    st.title("Test Classificatore — Run")

    configs = _render_config_list("clf_run_configs")
    st.divider()

    max_queries = st.number_input("Max query (0 = tutte)", min_value=0, value=50, step=10, key="clf_run_max")
    csv_file = st.file_uploader("Upload GT CSV", type=["csv"], key="clf_run_csv")
    if csv_file is None:
        st.info("Carica wf_ground_truth.csv per iniziare.")
        return

    if st.button("Lancia tutti", type="primary", key="clf_run_btn"):
        df = pd.read_csv(csv_file)
        type_map = _build_gt_type_map(df)
        queries = list(type_map.keys())
        if max_queries > 0:
            queries = queries[:max_queries]

        st.session_state["clf_run_queue"] = [
            {"config": cfg, "status": "pending", "run_dir": None, "results": None}
            for cfg in configs
        ]
        st.session_state["clf_run_type_map"] = type_map
        st.session_state["clf_run_queries"] = queries
        st.session_state["clf_run_csv_name"] = csv_file.name
        st.rerun()

    queue: list[dict] = st.session_state.get("clf_run_queue", [])
    if not queue:
        return

    type_map = st.session_state.get("clf_run_type_map", {})
    queries = st.session_state.get("clf_run_queries", [])
    csv_name = st.session_state.get("clf_run_csv_name", "")

    next_pending = next((i for i, item in enumerate(queue) if item["status"] == "pending"), None)

    for i, item in enumerate(queue):
        cfg = item["config"]
        label = f"Config {i + 1} — {cfg.get('prompt_name', '?')} · {cfg.get('llm_model', '?')}"
        if item["status"] == "done":
            run_name = Path(item["run_dir"]).name if item["run_dir"] else ""
            st.success(f"{label} — completato · run: `{run_name}`")
        elif item["status"] == "error":
            st.error(f"{label} — errore: {item.get('error', '')}")
        elif i == next_pending:
            bar = st.progress(0, text=f"{label} — avvio...")
            cfg = item["config"]
            prompt_text = cfg["custom_text"] if cfg.get("custom") else (PROMPTS_DIR / f"{cfg['prompt_name']}.txt").read_text(encoding="utf-8").strip()
            llm = LLM(model=cfg["llm_model"])
            rows = []
            try:
                for j, query in enumerate(queries, 1):
                    bar.progress(j / len(queries), text=f"{label}: {j}/{len(queries)}")
                    result = _classify_query(llm, prompt_text, query)
                    gt_type = type_map[query]
                    predicted = result["predicted_type"] or "—"
                    rows.append({
                        "Query": query,
                        "GT Type": gt_type,
                        "Predicted": predicted,
                        "✓": "✓" if predicted == gt_type else "✗",
                        "Reasoning": result["reasoning"],
                        "Error": result["error"] or "",
                    })
                results_df = pd.DataFrame(rows)
                run_dir = _save_run(results_df, cfg["prompt_name"], prompt_text, cfg["llm_model"], csv_name)
                queue[i]["status"] = "done"
                queue[i]["run_dir"] = str(run_dir)
                queue[i]["results"] = results_df
            except Exception as exc:
                queue[i]["status"] = "error"
                queue[i]["error"] = str(exc)
            st.session_state["clf_run_queue"] = queue
            st.rerun()
        else:
            st.info(f"{label} — in attesa")

    if next_pending is None:
        st.divider()
        for i, item in enumerate(queue):
            if item["status"] != "done" or item.get("results") is None:
                continue
            cfg = item["config"]
            st.markdown(f"#### Config {i + 1} — {cfg.get('prompt_name', '?')} · {cfg.get('llm_model', '?')}")
            _render_results(item["results"], key_prefix=f"clf_run_res_{i}")
            st.divider()


def _render_stats_row(label: str, df: pd.DataFrame):
    total = len(df)
    correct = int((df["✓"] == "✓").sum())
    wrong = total - correct
    llm_errors = int((df["Error"].fillna("").str.strip() != "").sum())
    acc = f"{100 * correct / total:.1f}%" if total else "—"
    st.markdown(f"**{label}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Query testate", total)
    c2.metric("Type Accuracy", acc)
    c3.metric("Corrette", correct)
    c4.metric("Errate", wrong)
    c5.metric("Errori LLM", llm_errors)


def _render_results(results_df: pd.DataFrame, key_prefix: str):
    type_labels = {"_all": "Globale", "reato": "Reato", "reato_circostanziato": "Reato Circostanziato", "mixed": "Mixed"}

    for t, label in type_labels.items():
        sub = results_df if t == "_all" else results_df[results_df["GT Type"] == t]
        _render_stats_row(label, sub)

    st.divider()
    filter_opt = st.radio("Mostra", ["Tutte", "Solo errate", "Solo corrette"], horizontal=True, key=f"{key_prefix}_filter")
    show_df = results_df.copy()
    if filter_opt == "Solo errate":
        show_df = results_df[results_df["✓"] == "✗"]
    elif filter_opt == "Solo corrette":
        show_df = results_df[results_df["✓"] == "✓"]

    tab_labels = {k: v for k, v in type_labels.items() if k != "_all"}
    tabs = st.tabs([f"{label} ({len(show_df[show_df['GT Type'] == t])})" for t, label in tab_labels.items()])
    for tab, (t, _) in zip(tabs, tab_labels.items()):
        with tab:
            sub = show_df[show_df["GT Type"] == t]
            if sub.empty:
                st.info("Nessuna query.")
            else:
                st.dataframe(sub, use_container_width=True, hide_index=True)
