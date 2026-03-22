"""Entity similarity, one-to-one matching, and recall scoring."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import get_path
from .io_utils import read_jsonl, write_jsonl
from .normalize import critical_token_signature, normalize_entities

try:
    from rapidfuzz import fuzz  # type: ignore
except ImportError:  # pragma: no cover - exercised in environments without rapidfuzz
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def token_sort_ratio(left: str, right: str) -> float:
            left_tokens = " ".join(sorted(left.split()))
            right_tokens = " ".join(sorted(right.split()))
            return SequenceMatcher(None, left_tokens, right_tokens).ratio() * 100

    fuzz = _FallbackFuzz()

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except ImportError:  # pragma: no cover - exercised in environments without scipy
    linear_sum_assignment = None


def entity_similarity(gold_entity: str, pred_entity: str) -> float:
    normalized_gold = normalize_entities([gold_entity])
    normalized_pred = normalize_entities([pred_entity])
    if not normalized_gold or not normalized_pred:
        return 0.0
    gold_value = normalized_gold[0]
    pred_value = normalized_pred[0]
    if critical_token_signature(gold_value) != critical_token_signature(pred_value):
        return 0.0
    return fuzz.token_sort_ratio(gold_value, pred_value) / 100.0


def build_similarity_matrix(gold_entities: list[str], pred_entities: list[str]) -> list[list[float]]:
    return [
        [entity_similarity(gold_entity, pred_entity) for pred_entity in pred_entities]
        for gold_entity in gold_entities
    ]


def _best_matching_fallback(similarity_matrix: list[list[float]], threshold: float) -> list[tuple[int, int, float]]:
    if not similarity_matrix or not similarity_matrix[0]:
        return []
    row_count = len(similarity_matrix)
    col_count = len(similarity_matrix[0])

    @lru_cache(maxsize=None)
    def solve(row_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int, float], ...]]:
        if row_index == row_count:
            return 0.0, ()
        best_score, best_pairs = solve(row_index + 1, used_mask)
        for col_index in range(col_count):
            score = similarity_matrix[row_index][col_index]
            if score < threshold or (used_mask & (1 << col_index)):
                continue
            downstream_score, downstream_pairs = solve(row_index + 1, used_mask | (1 << col_index))
            candidate_score = score + downstream_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_pairs = ((row_index, col_index, score),) + downstream_pairs
        return best_score, best_pairs

    return list(solve(0, 0)[1])


def _best_matching_scipy(similarity_matrix: list[list[float]], threshold: float) -> list[tuple[int, int, float]]:
    if not similarity_matrix or not similarity_matrix[0] or linear_sum_assignment is None:
        return []
    score_rows = len(similarity_matrix)
    score_cols = len(similarity_matrix[0])
    padded_size = max(score_rows, score_cols)
    padded_matrix = [[0.0 for _ in range(padded_size)] for _ in range(padded_size)]
    for row_index in range(score_rows):
        for col_index in range(score_cols):
            score = similarity_matrix[row_index][col_index]
            padded_matrix[row_index][col_index] = score if score >= threshold else 0.0
    cost_matrix = [[1.0 - value for value in row] for row in padded_matrix]
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int, float]] = []
    for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
        if row_index >= score_rows or col_index >= score_cols:
            continue
        score = padded_matrix[row_index][col_index]
        if score >= threshold:
            matches.append((row_index, col_index, score))
    return matches


def best_one_to_one_matches(similarity_matrix: list[list[float]], threshold: float) -> list[tuple[int, int, float]]:
    if linear_sum_assignment is not None:
        return _best_matching_scipy(similarity_matrix, threshold)
    return _best_matching_fallback(similarity_matrix, threshold)


def _success_status(record: dict[str, Any] | None) -> bool:
    return record is not None and record.get("parse_status") in {"ok", "empty"}


def score_dataset_records(
    dataset_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    gold_map = {record["question_uid"]: record for record in gold_records}
    pred_map = {record["question_uid"]: record for record in pred_records}
    scored: list[dict[str, Any]] = []
    for dataset_record in dataset_records:
        question_uid = dataset_record["question_uid"]
        gold_record = gold_map.get(question_uid)
        pred_record = pred_map.get(question_uid)
        output: dict[str, Any] = {
            "question_uid": question_uid,
            "subject": dataset_record.get("subject", ""),
            "question": dataset_record.get("question", ""),
            "gold_entity_raw": gold_record.get("entities_raw", []) if gold_record else [],
            "pred_entity_raw": pred_record.get("entities_raw", []) if pred_record else [],
            "gold_entity_norm": [],
            "pred_entity_norm": [],
            "similarity_matrix": [],
            "matched_pairs": [],
            "matched_gold_count": 0,
            "gold_count": 0,
            "recall": None,
            "acc": dataset_record.get("acc"),
            "excluded": True,
            "exclude_reason": None,
        }
        if dataset_record.get("acc") is None:
            output["exclude_reason"] = "missing_acc"
            scored.append(output)
            continue
        if not _success_status(gold_record):
            output["exclude_reason"] = "gold_extraction_failed"
            scored.append(output)
            continue
        if not _success_status(pred_record):
            output["exclude_reason"] = "pred_extraction_failed"
            scored.append(output)
            continue
        gold_entities_norm = normalize_entities(gold_record.get("entities_raw", []))
        pred_entities_norm = normalize_entities(pred_record.get("entities_raw", []))
        output["gold_entity_norm"] = gold_entities_norm
        output["pred_entity_norm"] = pred_entities_norm
        if not gold_entities_norm:
            output["exclude_reason"] = "empty_gold_entity"
            scored.append(output)
            continue
        similarity_matrix = build_similarity_matrix(gold_entities_norm, pred_entities_norm)
        matches = best_one_to_one_matches(similarity_matrix, threshold)
        output["similarity_matrix"] = similarity_matrix
        output["matched_pairs"] = [
            {
                "gold": gold_entities_norm[row_index],
                "pred": pred_entities_norm[col_index],
                "score": score,
            }
            for row_index, col_index, score in matches
        ]
        output["matched_gold_count"] = len(matches)
        output["gold_count"] = len(gold_entities_norm)
        output["recall"] = len(matches) / len(gold_entities_norm)
        output["excluded"] = False
        scored.append(output)
    return scored


def score_recall(
    config: dict[str, Any],
    *,
    dataset_path: str | Path | None = None,
    gold_path: str | Path | None = None,
    pred_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_records = read_jsonl(Path(dataset_path) if dataset_path else get_path(config, "dataset_with_acc"))
    gold_records = read_jsonl(Path(gold_path) if gold_path else get_path(config, "gold_entities"))
    pred_records = read_jsonl(Path(pred_path) if pred_path else get_path(config, "pred_entities"))
    scored = score_dataset_records(
        dataset_records,
        gold_records,
        pred_records,
        threshold=float(config["matching"]["threshold"]),
    )
    output_path = get_path(config, "entity_scores")
    write_jsonl(output_path, scored)
    excluded_counter = Counter(record["exclude_reason"] for record in scored if record["excluded"])
    return {
        "output": str(output_path),
        "row_count": len(scored),
        "excluded_counts": dict(excluded_counter),
    }
