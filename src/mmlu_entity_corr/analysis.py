"""Correlation analysis and reporting."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .config import get_path
from .io_utils import read_jsonl, write_csv, write_json
from .matching import best_one_to_one_matches, build_similarity_matrix

try:
    from scipy.stats import pearsonr as scipy_pearsonr  # type: ignore
except ImportError:  # pragma: no cover - exercised in environments without scipy
    scipy_pearsonr = None


def _pearson_fallback(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None, None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None, None
    r_value = numerator / (denom_x * denom_y)
    if len(xs) <= 3 or abs(r_value) >= 1:
        return r_value, 0.0
    fisher_z = 0.5 * math.log((1 + r_value) / (1 - r_value))
    z_score = abs(fisher_z) * math.sqrt(len(xs) - 3)
    p_value = 2 * (1 - NormalDist().cdf(z_score))
    return r_value, p_value


def pearsonr(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if scipy_pearsonr is not None:
        result = scipy_pearsonr(xs, ys)
        return float(result.statistic), float(result.pvalue)
    return _pearson_fallback(xs, ys)


def bootstrap_ci(
    xs: list[float], ys: list[float], *, samples: int, seed: int
) -> tuple[float | None, float | None]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    indices = list(range(len(xs)))
    for _ in range(samples):
        sampled_indices = [rng.choice(indices) for _ in indices]
        sample_xs = [xs[index] for index in sampled_indices]
        sample_ys = [ys[index] for index in sampled_indices]
        r_value, _ = pearsonr(sample_xs, sample_ys)
        if r_value is not None and not math.isnan(r_value):
            estimates.append(r_value)
    if not estimates:
        return None, None
    estimates.sort()
    lower_index = max(0, int(0.025 * (len(estimates) - 1)))
    upper_index = min(len(estimates) - 1, int(0.975 * (len(estimates) - 1)))
    return estimates[lower_index], estimates[upper_index]


def _recompute_recall(record: dict[str, Any], threshold: float) -> float | None:
    gold_entities = record.get("gold_entity_norm", [])
    pred_entities = record.get("pred_entity_norm", [])
    if not gold_entities:
        return None
    similarity_matrix = build_similarity_matrix(gold_entities, pred_entities)
    matches = best_one_to_one_matches(similarity_matrix, threshold)
    return len(matches) / len(gold_entities)


def analyze_correlation(config: dict[str, Any], *, input_path: str | Path | None = None) -> dict[str, Any]:
    source = Path(input_path) if input_path else get_path(config, "entity_scores")
    score_records = read_jsonl(source)
    included = [record for record in score_records if not record.get("excluded", True)]
    excluded = [record for record in score_records if record.get("excluded", True)]
    recall_values = [float(record["recall"]) for record in included if record.get("recall") is not None]
    acc_values = [float(record["acc"]) for record in included if record.get("recall") is not None]
    r_value, p_value = pearsonr(recall_values, acc_values)
    bootstrap_low, bootstrap_high = bootstrap_ci(
        recall_values,
        acc_values,
        samples=int(config["analysis"]["bootstrap_samples"]),
        seed=int(config["analysis"]["seed"]),
    )
    summary = {
        "n": len(included),
        "r": r_value,
        "p_value": p_value,
        "mean_recall": (sum(recall_values) / len(recall_values)) if recall_values else None,
        "mean_acc": (sum(acc_values) / len(acc_values)) if acc_values else None,
        "bootstrap_ci_95": [bootstrap_low, bootstrap_high],
        "excluded_counts": dict(Counter(record.get("exclude_reason") for record in excluded)),
        "threshold": float(config["matching"]["threshold"]),
    }
    thresholds = config["matching"]["threshold_sweep"]
    sweep_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        sweep_recalls: list[float] = []
        sweep_accs: list[float] = []
        for record in included:
            recall = _recompute_recall(record, float(threshold))
            if recall is None:
                continue
            sweep_recalls.append(recall)
            sweep_accs.append(float(record["acc"]))
        sweep_r, sweep_p = pearsonr(sweep_recalls, sweep_accs)
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "n": len(sweep_recalls),
                "r": sweep_r,
                "p_value": sweep_p,
                "mean_recall": (sum(sweep_recalls) / len(sweep_recalls)) if sweep_recalls else None,
                "mean_acc": (sum(sweep_accs) / len(sweep_accs)) if sweep_accs else None,
            }
        )
    write_json(get_path(config, "correlation_summary"), summary, pretty=True)
    write_csv(get_path(config, "included_samples"), included)
    write_csv(get_path(config, "excluded_samples"), excluded)
    write_csv(get_path(config, "threshold_sweep"), sweep_rows)
    return summary

