"""Metriche di retrieval: funzioni pure che prendono liste di chiavi e tornano un float.

Usate sia da evaluation.py (pipeline "periodi") sia da evaluation_pipeline.py
(pipeline "reati"). Nota: in recall/precision l'ordine dei predetti non conta;
in ndcg/block_ap conta (i parametri si chiamano `ranked`).
"""
import math


def recall(targets: list[str], predicted: list[str]) -> float:
    """Frazione delle chiavi corrette che sono state recuperate."""
    gt = set(targets)
    if not gt:
        return 0.0
    return len(gt & set(predicted)) / len(gt)


def precision(targets: list[str], predicted: list[str]) -> float:
    """Frazione delle chiavi recuperate che sono corrette.

    Il denominatore è il numero di predetti (duplicati inclusi), com'era prima.
    """
    if not predicted:
        return 0.0
    return len(set(targets) & set(predicted)) / len(predicted)


def ndcg(targets: list[str], ranked: list[str]) -> float:
    """NDCG binario: premia le chiavi corrette che compaiono in alto nel ranking."""
    gt = set(targets)
    if not gt:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, key in enumerate(ranked, 1) if key in gt)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, len(gt) + 1))
    return dcg / idcg


def block_recall(blocks: list[list[str]], predicted: list[str]) -> float:
    """Frazione di blocchi recuperati per intero. Un blocco parziale vale zero."""
    if not blocks:
        return 0.0
    pred = set(predicted)
    completi = sum(1 for block in blocks if set(block).issubset(pred))
    return completi / len(blocks)


def block_ap(blocks: list[list[str]], ranked: list[str]) -> float:
    """Block Average Precision normalizzata in [0, 1].

    Premia il completamento dei blocchi in alto nel ranking, normalizzando
    rispetto all'ordine ideale (blocchi più piccoli prima).
    """
    if not blocks:
        return 0.0

    key_to_rank = {key: rank for rank, key in enumerate(ranked, 1)}

    # Per ogni blocco recuperato per intero, il rank in cui si completa.
    completion_ranks = []
    for block in blocks:
        ranks = [key_to_rank[k] for k in block if k in key_to_rank]
        if len(ranks) == len(block):
            completion_ranks.append(max(ranks))
    completion_ranks.sort()
    raw_ap = sum(i / r for i, r in enumerate(completion_ranks, 1)) / len(blocks)

    # Caso ideale: blocchi ordinati dal più piccolo, completati il prima possibile.
    cumulative = 0
    ideal_sum = 0.0
    for j, size in enumerate(sorted(len(b) for b in blocks), 1):
        cumulative += size
        ideal_sum += j / cumulative
    ideal_ap = ideal_sum / len(blocks)

    if ideal_ap == 0:
        return 0.0
    return raw_ap / ideal_ap
