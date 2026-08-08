
import pandas as pd
from pathlib import Path


class Periodo:
    def __init__(self, index, periodo, fonte_normativa, rubrica, testo):
        self.index = index
        self.periodo = periodo
        self.fonte_normativa = fonte_normativa
        self.rubrica = rubrica
        self.testo = testo


class PeriodoFlat:
    def __init__(self, index, periodo, periodo_content):
        self.index = index
        self.periodo = periodo
        self.periodo_content = periodo_content


class Query:
    def __init__(self, query: str):
        self.query = query


class ComboPeriodo:
    def __init__(self, index, rubriche_con_periodi, rubrica_imputativa):
        self.index = index
        self.rubriche_con_periodi = rubriche_con_periodi
        self.rubrica_imputativa = rubrica_imputativa


class Reato:
    def __init__(self, index, combo_reati_content, combo_reati):
        self.index = index
        self.combo_reati_content = combo_reati_content
        self.combo_reati = combo_reati


class ReatoCircostanziato:
    def __init__(self, index, combo_reati_circostanziati_content, combo_reati_circostanziati):
        self.index = index
        self.combo_reati_circostanziati_content = combo_reati_circostanziati_content
        self.combo_reati_circostanziati = combo_reati_circostanziati


class Dataset:
    def __init__(self, folder, dataset_type="periodi", source_file=None):
        folder = Path(folder)
        self.dataset_type = dataset_type

        if dataset_type == "periodi":
            df_periodi = pd.read_csv(folder / "wf_periodi.csv")
            self.periodi = [
                Periodo(
                    row["index"],
                    row["periodo"],
                    row["fonte_normativa"],
                    None if pd.isna(row["rubrica"]) else row["rubrica"],
                    row["testo"],
                )
                for _, row in df_periodi.iterrows()
            ]
            self.records = self.periodi
        elif dataset_type == "periodi_flat":
            df = pd.read_csv(
                folder / (source_file or "wf_reati_circostanze_flat.csv"),
                keep_default_na=False,
                na_filter=False,
            )
            self.records = [
                PeriodoFlat(
                    row["index"],
                    _none_if_empty(row["periodo"]),
                    _none_if_empty(row["periodo_content"]),
                )
                for _, row in df.iterrows()
            ]
        elif dataset_type == "combo":
            df_combo = pd.read_csv(
                folder / "wf_combo_periodi.csv",
                keep_default_na=False,
                na_filter=False,
            )
            self.combo_periodi = [
                ComboPeriodo(
                    f"combo_{i}",
                    _none_if_empty(row["rubriche_con_periodi"]),
                    _none_if_empty(row["rubrica_imputativa"]),
                )
                for i, (_, row) in enumerate(df_combo.iterrows())
            ]
            self.records = self.combo_periodi
            self.periodi = []
        elif dataset_type == "reati":
            df = pd.read_csv(folder / "wf_reati.csv", keep_default_na=False, na_filter=False)
            self.records = [
                Reato(f"reato_{i}", row["combo_reati_content"], row["combo_reati"])
                for i, (_, row) in enumerate(df.iterrows())
            ]

        elif dataset_type == "reati_circostanziati_full":
            df = pd.read_csv(folder / "wf_reati_circostanziati_full.csv", keep_default_na=False, na_filter=False)
            self.records = [
                ReatoCircostanziato(f"reato_circ_{i}", row["combo_reati_circostanziati_content"], row["combo_reati_circostanziati"])
                for i, (_, row) in enumerate(df.iterrows())
            ]

        elif dataset_type == "reati_circostanziati_light":
            df = pd.read_csv(folder / "wf_reati_circostanziati_light.csv", keep_default_na=False, na_filter=False)
            self.records = [
                ReatoCircostanziato(f"reato_circ_{i}", row["combo_reati_circostanziati_content"], row["combo_reati_circostanziati"])
                for i, (_, row) in enumerate(df.iterrows())
            ]

        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        queries_path = folder / "wf_queries.csv"
        if queries_path.exists():
            df_queries = pd.read_csv(
                queries_path,
                keep_default_na=False,
                na_filter=False,
            )
            df_queries = df_queries.replace({"nan": None, "NaN": None, "": None}).astype(object)
            self.queries = [
                Query(row["query"])
                for _, row in df_queries.iterrows()
            ]
        else:
            self.queries = []


def _none_if_empty(value):
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value
