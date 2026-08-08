# main.py
import json
import os
from pathlib import Path
from cli import parse_args
from es import ES
from dataset import Dataset
from indexer import Indexer
from evaluation import build_gt_context, evaluate_from_csv, load_models_config, save_evaluation_run
import evaluation_pipeline
from config import load_env
#from rag import RAG


def make_index_name(es, prefix: str):
    if not es.index_exists(prefix):
        return prefix
    i = 2
    while es.index_exists(f"{prefix}_{i}"):
        i += 1
    return f"{prefix}_{i}"


def _finalize_paper_run(run_dir, is_pool):
    """Chiude un run paper: ricostruisce debug.json dal jsonl (pool) e segna status=complete."""
    if is_pool:
        jsonl = run_dir / "debug.jsonl"
        if jsonl.exists():
            recon: dict = {}
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                recon.setdefault(e["query"], {})[e["model"]] = e["data"]
            (run_dir / "debug.json").write_text(json.dumps(recon, ensure_ascii=False))
    (run_dir / "status.json").write_text(json.dumps({"status": "complete"}, ensure_ascii=False))


def main():
    args = parse_args()
    es = ES()

    if args.command == "index":
        dataset = Dataset(args.data_source, dataset_type=args.dataset_type, source_file=args.source_file)
        indexer = Indexer(
            es=es,
            dataset=dataset,
            embedders=args.embedders,
            dataset_type=args.dataset_type,
        )

        index_name = make_index_name(es, args.name_prefix)
        print(f"[INDEX] using index name: {index_name}")
        indexer.run(index_name=index_name)

    if args.command == "resume":
        print("[WARN] Resume basato sulla posizione del CSV.")
        print("[WARN] Assicurati che il file non sia stato modificato (righe aggiunte, rimosse o riordinate) dall'indicizzazione originale.")
        dataset = Dataset(args.data_source, dataset_type=args.dataset_type)
        indexer = Indexer(
            es=es,
            dataset=dataset,
            embedders=args.embedders,
            dataset_type=args.dataset_type,
        )
        for embedder in args.embedders:
            indexer.index_embeddings(args.index_name, embedder, skip_existing=True)

    if args.command == "evaluate":
        load_env()
        es_host = os.environ["ES_HOST"]
        index_names = {
            "periodi": os.environ["ES_INDEX_NAME"],
            "combo": os.environ["ES_INDEX_NAME_COMBO"],
        }
        model_configs = load_models_config(args.models_config)
        eval_es = ES(host=es_host)
        details_df, details_by_query = evaluate_from_csv(
            args.gt_csv,
            model_configs,
            eval_es,
            index_names,
        )
        save_info = save_evaluation_run(
            details_df=details_df,
            model_configs=model_configs,
            gt_reference=args.gt_csv,
            gt_context=build_gt_context(details_by_query),
        )
        print(f"Saved evaluation run in: {save_info['run_dir']}")

    if args.command == "evaluate-pipeline":
        load_env()
        es_host = os.environ["ES_HOST"]
        index_reati = os.environ["ES_INDEX_NAME_REATI"]
        index_circ_light = os.environ.get("ES_INDEX_NAME_REATI_CIRC_LIGHT", "")
        index_circ_full = os.environ.get("ES_INDEX_NAME_REATI_CIRC_FULL", "")

        from config import load_key_table
        from tqdm import tqdm
        from datetime import datetime, timezone
        from pathlib import Path as _Path
        import sys

        eval_es = ES(host=es_host, verbose=False)
        key_table = load_key_table()
        bar_format = "{desc}: {percentage:3.0f}%|{bar:70}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

        if args.resume:
            # --- RESUME MODE ---
            resume_dir = _Path(args.resume)
            meta = json.loads((resume_dir / "meta.json").read_text())
            model_configs = meta["model_configs"]
            gt_csv = meta["gt_reference"]
            if len(model_configs) != 1:
                print("Errore: --resume supporta solo run con un singolo config.")
                sys.exit(1)
            cfg = model_configs[0]
            df = __import__("pandas").read_csv(gt_csv)
            existing_df = __import__("pandas").read_csv(resume_dir / "details.csv")
            skip_queries = set(existing_df.dropna(subset=["Rank"])["Query"].dropna().unique())
            all_queries = set(df["query"].dropna().unique())
            remaining = all_queries - skip_queries
            n_errors = existing_df[existing_df["Error"].notna()]["Query"].nunique()
            print(f"  Resume: {len(skip_queries)} query OK, {n_errors} con errore da ritentare, {len(remaining)} rimanenti")
            index_circ = index_circ_light if cfg.get("index_circ") == "light" else index_circ_full

            (resume_dir / "status.json").write_text(json.dumps({"status": "running", "pid": os.getpid()}, ensure_ascii=False))

            bar = tqdm(total=len(remaining), desc="[resume]", unit="query", leave=True,
                       bar_format=bar_format, file=sys.stdout)
            last_step = [0]

            def on_step(step, total, q_idx, m_idx, query="", _bar=bar, _last=last_step):
                _bar.update(step - _last[0])
                _last[0] = step
                _bar.set_postfix_str(f"q={query}" if query else "")

            details_df, gt_context, _ = evaluation_pipeline.evaluate_from_df(
                df, [cfg], eval_es, index_reati, index_circ, key_table,
                on_step=on_step,
                out_csv_path=resume_dir / "details.csv",
                out_debug_path=resume_dir / "debug.jsonl",
                out_errors_path=resume_dir / "errors.jsonl",
                status_path=resume_dir / "status.json",
                skip_queries=skip_queries,
            )
            bar.close()
            evaluation_pipeline.save_run(
                details_df=None,
                model_configs=[cfg],
                gt_reference=gt_csv,
                gt_context=gt_context,
                debug_data={},
                run_dir=resume_dir,
            )
            print(f"  Completato: {resume_dir}")

        else:
            # --- NORMAL MODE ---
            if not args.gt_csv or not args.models_config:
                print("Errore: --gt_csv e --models_config sono richiesti senza --resume.")
                sys.exit(1)
            model_configs = json.loads(Path(args.models_config).read_text())
            df = __import__("pandas").read_csv(args.gt_csv)
            if args.max_queries > 0:
                unique_queries = df["query"].dropna().unique()[:args.max_queries]
                df = df[df["query"].isin(unique_queries)]
            n_queries = len(df["query"].dropna().unique())

            for i, cfg in enumerate(model_configs, 1):
                index_circ = index_circ_light if cfg.get("index_circ") == "light" else index_circ_full
                desc = f"[{i}/{len(model_configs)}]"
                bar = tqdm(total=n_queries, desc=desc, unit="query", leave=True,
                           bar_format=bar_format, file=sys.stdout)
                last_step = [0]

                now = datetime.now(timezone.utc)
                run_dir = _Path("runs/evaluations_pipeline") / now.strftime("run_%Y%m%d_%H%M%S_%f")
                run_dir.mkdir(parents=True, exist_ok=False)
                out_csv = run_dir / "details.csv"
                (run_dir / "meta.json").write_text(json.dumps({
                    "created_at_utc": now.isoformat(),
                    "gt_reference": args.gt_csv,
                    "model_configs": [cfg],
                    "index_reati": index_reati,
                    "index_circ": index_circ,
                }, ensure_ascii=False, indent=2))
                gt_map = evaluation_pipeline.build_gt_map(df)
                gt_context_initial = {
                    q: {"target_keys": p["target_keys"], "base_reati_keys": p["base_reati_keys"], "type": p["type"]}
                    for q, p in gt_map.items()
                }
                (run_dir / "gt_context.json").write_text(json.dumps(gt_context_initial, ensure_ascii=False, indent=2))
                (run_dir / "status.json").write_text(json.dumps({"status": "running", "pid": os.getpid()}, ensure_ascii=False))
                print(f"  Run dir: {run_dir}", flush=True)

                def on_step(step, total, q_idx, m_idx, query="", _bar=bar, _last=last_step):
                    _bar.update(step - _last[0])
                    _last[0] = step
                    _bar.set_postfix_str(f"q={query}" if query else "")

                details_df, gt_context, _ = evaluation_pipeline.evaluate_from_df(
                    df, [cfg], eval_es, index_reati, index_circ, key_table,
                    on_step=on_step, out_csv_path=out_csv,
                    out_debug_path=run_dir / "debug.jsonl",
                    out_errors_path=run_dir / "errors.jsonl",
                    status_path=run_dir / "status.json",
                )
                bar.close()
                evaluation_pipeline.save_run(
                    details_df=None,
                    model_configs=[cfg],
                    gt_reference=args.gt_csv,
                    gt_context=gt_context,
                    debug_data={},
                    run_dir=run_dir,
                )
                print(f"  Salvato: {run_dir}")

        print("Completato.")

    if args.command == "evaluate-paper":
        load_env()
        import sys
        import pandas as pd
        from datetime import datetime, timezone
        from tqdm import tqdm
        from config import (
            DEFAULT_INDEX_PAPER_PERIODI_FLAT, DEFAULT_INDEX_PAPER_REATI,
            DEFAULT_INDEX_PAPER_REATI_CIRC_LIGHT, load_key_table,
            PAPER_RUNS_DIR, PAPER_HARD_RUNS_DIR, PAPER_KEY_TABLE,
        )
        from evaluation import (
            build_gt_map, build_gt_map_paper, evaluate_from_df, evaluate_pool_paper,
            compute_summary_from_details,
        )

        # --hard: GT legacy (query/result/type) con blocchi+tipi, salvata in runs/paper_hard.
        gt_builder = build_gt_map if args.hard else build_gt_map_paper
        runs_base = PAPER_HARD_RUNS_DIR if args.hard else PAPER_RUNS_DIR

        eval_es = ES(host=os.environ["ES_HOST"], verbose=False)
        flat_idx = {"periodi_flat": DEFAULT_INDEX_PAPER_PERIODI_FLAT}
        bar_format = "{desc}: {percentage:3.0f}%|{bar:60}| {n}/{total} [{elapsed}<{remaining}]"

        def _run_one(cfg, gt_map, df, run_dir, skip_queries=None):
            """Esegue un modello (B o pool) con streaming details/debug nello run_dir."""
            is_pool = cfg.get("pipeline_type") == "pool"
            n_active = len(gt_map) - len(skip_queries or set())
            bar = tqdm(total=n_active, desc=run_dir.name, unit="q", leave=True,
                       bar_format=bar_format, file=sys.stdout)
            last = [0]

            def on_step(step, total, q_idx, m_idx, _bar=bar, _last=last):
                _bar.update(step - _last[0])
                _last[0] = step

            if is_pool:
                evaluate_pool_paper(
                    df, [cfg], eval_es, DEFAULT_INDEX_PAPER_REATI, DEFAULT_INDEX_PAPER_REATI_CIRC_LIGHT,
                    load_key_table(PAPER_KEY_TABLE), on_step=on_step, gt_map=gt_map,
                    out_csv_path=run_dir / "details.csv", out_debug_path=run_dir / "debug.jsonl",
                    skip_queries=skip_queries, flat_index=DEFAULT_INDEX_PAPER_PERIODI_FLAT,
                )
            else:
                evaluate_from_df(
                    df, [cfg], eval_es, flat_idx, on_step=on_step, gt_map=gt_map,
                    out_csv_path=run_dir / "details.csv", skip_queries=skip_queries,
                )
            bar.close()
            _finalize_paper_run(run_dir, is_pool)

        if args.resume:
            resume_dir = Path(args.resume)
            meta = json.loads((resume_dir / "meta.json").read_text())
            model_configs = meta["model_configs"]
            if len(model_configs) != 1:
                print("Errore: --resume supporta solo run con un singolo config.")
                sys.exit(1)
            cfg = model_configs[0]
            df = pd.read_csv(meta["gt_reference"], keep_default_na=False, na_filter=False)
            gt_map = gt_builder(df)
            done = set()
            det_path = resume_dir / "details.csv"
            if det_path.exists():
                done = set(pd.read_csv(det_path, usecols=["Query"])["Query"].dropna().unique())
            print(f"[paper] resume: {len(done)} fatte, {len(gt_map) - len(done)} rimanenti · {resume_dir}")
            (resume_dir / "status.json").write_text(json.dumps({"status": "running", "pid": os.getpid()}, ensure_ascii=False))
            _run_one(cfg, gt_map, df, resume_dir, skip_queries=done)
            print(f"[paper] completato: {resume_dir}")

        else:
            if not args.gt_csv or not args.models_config:
                print("Errore: --gt_csv e --models_config sono richiesti senza --resume.")
                sys.exit(1)
            models = json.loads(Path(args.models_config).read_text())
            df = pd.read_csv(args.gt_csv, keep_default_na=False, na_filter=False)
            if args.max_queries > 0:
                keep = list(dict.fromkeys(df["query"].tolist()))[: args.max_queries]
                df = df[df["query"].isin(keep)].reset_index(drop=True)
            gt_map = gt_builder(df)
            gt_context = {
                q: {"query_type": p["query_type"], "gt_periods": p["periods"],
                    "gt_blocks": p["blocks"], "raw_results": p["raw_results"]}
                for q, p in gt_map.items()
            }
            base = Path(runs_base)
            base.mkdir(parents=True, exist_ok=True)
            print(f"[paper] {len(models)} modelli · {len(gt_map)} query · gt={args.gt_csv}")

            leaderboard = []
            for i, cfg in enumerate(models, 1):
                now = datetime.now(timezone.utc)
                run_dir = base / now.strftime("run_%Y%m%d_%H%M%S_%f")
                run_dir.mkdir()
                (run_dir / "meta.json").write_text(json.dumps({
                    "created_at_utc": now.isoformat(), "gt_reference": args.gt_csv, "model_configs": [cfg],
                }, ensure_ascii=False, indent=2))
                (run_dir / "gt_context.json").write_text(json.dumps(gt_context, ensure_ascii=False))
                (run_dir / "status.json").write_text(json.dumps({"status": "running", "pid": os.getpid()}, ensure_ascii=False))

                _run_one(cfg, gt_map, df, run_dir)

                s = compute_summary_from_details(pd.read_csv(run_dir / "details.csv"), gt_context)
                is_pool = cfg.get("pipeline_type") == "pool"
                rr = " +period-rerank" if cfg.get("period_rerank") else ""
                tag = f"M{i} {'pool' if is_pool else cfg.get('model_type')} j={cfg.get('j')} k={cfg.get('k')}{rr}"
                r = round(float(s["Recall M1"].mean()), 3)
                p = round(float(s["Precision M1"].mean()), 3)
                nd = round(float(s["NDCG M1"].mean()), 3)
                leaderboard.append((tag, r, p, nd))
                print(f"  {tag} -> R={r} P={p} NDCG={nd} · {run_dir}", flush=True)

            print("\n[paper] LEADERBOARD (Recall / Precision / NDCG):")
            for tag, r, p, nd in sorted(leaderboard, key=lambda x: -x[1]):
                print(f"  {tag:28} R={r:.3f} P={p:.3f} NDCG={nd:.3f}")

    #elif args.command == "rag":
        #rag = RAG(es=es)
        #out = rag.run(
        #    index_name=args.index_name,
        #    query=args.query,
        #    top_k=args.top_k,
        #    embedder=args.embedder,
        #    llm=args.llm,
        #)
        #print(out)


if __name__ == "__main__":
    main()


# usage:
# python main.py index --name_prefix myidx --data_source data --dataset_type periodi --embedders text-embedding-3-large
# python main.py index --name_prefix comboidx --data_source data --dataset_type combo --embedders text-embedding-3-large
# python main.py rag --index_name myidx_ab12 --query "cos'è la rubrica?" --top_k 10 --embedder e5 --llm gpt-4o-mini
