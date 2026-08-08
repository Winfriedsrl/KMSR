import argparse
from vectorizer import Vectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Elasticsearch indexing and RAG inference",
        allow_abbrev=False,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ---------- INDEX COMMAND ----------
    index_parser = subparsers.add_parser(
        "index",
        help="build an Elasticsearch index",
    )
    index_parser.add_argument(
        "--name_prefix",
        required=True,
        help="prefix for the index name",
    )
    index_parser.add_argument(
        "--data_source",
        required=True,
        help="path to the dataset directory",
    )
    index_parser.add_argument(
        "--dataset_type",
        choices=["periodi", "periodi_flat", "combo", "reati", "reati_circostanziati_full", "reati_circostanziati_light"],
        default="periodi",
        help="dataset schema to index",
    )
    index_parser.add_argument(
        "--source_file",
        default=None,
        help="override del nome file sorgente nella cartella (per ora usato da periodi_flat)",
    )
    index_parser.add_argument(
        "--embedders",
        nargs="+",
        default=[],
        help="embedders for dense indexing",
    )

    # ---------- RESUME COMMAND ----------
    resume_parser = subparsers.add_parser(
        "resume",
        help="resume embedding indexing on an existing index",
    )
    resume_parser.add_argument(
        "--index_name",
        required=True,
        help="name of the existing Elasticsearch index to resume",
    )
    resume_parser.add_argument(
        "--data_source",
        required=True,
        help="path to the dataset directory (must be the same used during original indexing)",
    )
    resume_parser.add_argument(
        "--dataset_type",
        choices=["periodi", "periodi_flat", "combo", "reati", "reati_circostanziati_full", "reati_circostanziati_light"],
        required=True,
    )
    resume_parser.add_argument(
        "--embedders",
        nargs="+",
        required=True,
        help="embedders to resume",
    )

    # ---------- EVALUATE COMMAND ----------
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="run model evaluation on a GT csv",
    )
    eval_parser.add_argument(
        "--gt_csv",
        required=True,
        help="path to gt csv (wf_training_set format)",
    )
    eval_parser.add_argument(
        "--models_config",
        required=True,
        help="path to model config file (.json/.yaml/.yml)",
    )

    # ---------- EVALUATE-PIPELINE COMMAND ----------
    eval_pipeline_parser = subparsers.add_parser(
        "evaluate-pipeline",
        help="run reati pipeline evaluation on a GT csv (one run per model config)",
    )
    eval_pipeline_parser.add_argument(
        "--gt_csv",
        default=None,
        help="path to gt csv (wf_ground_truth format); required unless --resume is used",
    )
    eval_pipeline_parser.add_argument(
        "--models_config",
        default=None,
        help="path to JSON file with list of pipeline model configs; required unless --resume is used",
    )
    eval_pipeline_parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="max number of queries to evaluate (0 = all)",
    )
    eval_pipeline_parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_DIR",
        help="resume an interrupted run: path to existing run directory (single config only)",
    )

    # ---------- EVALUATE-PAPER COMMAND ----------
    eval_paper_parser = subparsers.add_parser(
        "evaluate-paper",
        help="run paper evaluation (baseline B e/o pool M) on the paper GT, period-level",
    )
    eval_paper_parser.add_argument(
        "--gt_csv",
        default="data/final/gt/wf_ground_truth_vecchio_formato.csv",
        help="path to paper GT csv (columns: query, reati_estratti)",
    )
    eval_paper_parser.add_argument(
        "--models_config",
        default=None,
        help="path to JSON file with list of paper model configs (es. configs/paper_baselines.json); required unless --resume",
    )
    eval_paper_parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="max number of distinct queries to evaluate (0 = all)",
    )
    eval_paper_parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_DIR",
        help="resume an interrupted run: path to existing run directory (single config only)",
    )
    eval_paper_parser.add_argument(
        "--hard",
        action="store_true",
        help="GT 'hard' legacy (colonne query, result, type): metriche di blocco + tipi, salva in runs/paper_hard",
    )

    args = parser.parse_args()

    # ---------- VALIDATION ----------
    if args.command in ("index", "resume"):
        invalid = [
            e for e in args.embedders
            if e not in Vectorizer.AVAILABLE_MODELS
        ]
        if invalid:
            parser.error(
                "Invalid embedders. Choose from: "
                + str(list(Vectorizer.AVAILABLE_MODELS.keys()))
            )

    return args
