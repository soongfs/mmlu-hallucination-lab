"""Dataset preparation and lm-eval alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import get_path
from .io_utils import write_jsonl
from .lm_eval import load_lm_eval_records
from .normalize import compute_question_uid


def load_hf_mmlu_records(dataset_path: str, dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing at runtime
        raise RuntimeError("datasets is required to load HuggingFace MMLU data.") from exc
    dataset = load_dataset(dataset_path, dataset_name, split=split)
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(dataset):
        question = row.get("question", "")
        choices = list(row.get("choices", []))
        subject = row.get("subject", "")
        answer = row.get("answer")
        question_uid = compute_question_uid(subject, question, choices)
        records.append(
            {
                "question_uid": question_uid,
                "subject": str(subject),
                "question": str(question),
                "choices": [str(choice) for choice in choices],
                "answer": answer,
                "_source_index": row_index,
            }
        )
    return records


def _build_group_map(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = record["question_uid"]
        result.setdefault(key, []).append(record)
    return result


def align_records(
    hf_records: list[dict[str, Any]], lm_eval_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hf_map = _build_group_map(hf_records)
    lm_map = _build_group_map(lm_eval_records)
    matched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for question_uid in sorted(set(hf_map) | set(lm_map)):
        hf_group = sorted(hf_map.get(question_uid, []), key=lambda record: record.get("_source_index", 0))
        lm_group = sorted(lm_map.get(question_uid, []), key=lambda record: record.get("_source_index", 0))
        pair_count = min(len(hf_group), len(lm_group))
        for duplicate_index in range(pair_count):
            hf_record = hf_group[duplicate_index]
            lm_record = lm_group[duplicate_index]
            matched.append(
                {
                    **hf_record,
                    "base_question_uid": question_uid,
                    "question_uid": f"{question_uid}:{duplicate_index}",
                    "duplicate_index": duplicate_index,
                    "prediction": lm_record.get("prediction"),
                    "target": lm_record.get("target"),
                    "acc": lm_record.get("acc"),
                }
            )
        for duplicate_index, hf_record in enumerate(hf_group[pair_count:], start=pair_count):
            failures.append(
                {
                    "question_uid": f"{question_uid}:{duplicate_index}",
                    "base_question_uid": question_uid,
                    "duplicate_index": duplicate_index,
                    "source": "huggingface",
                    "reason": "missing_lm_eval_record",
                    "subject": hf_record["subject"],
                    "question": hf_record["question"],
                    "choices": hf_record["choices"],
                }
            )
        for duplicate_index, lm_record in enumerate(lm_group[pair_count:], start=pair_count):
            failures.append(
                {
                    "question_uid": f"{question_uid}:{duplicate_index}",
                    "base_question_uid": question_uid,
                    "duplicate_index": duplicate_index,
                    "source": "lm_eval",
                    "reason": "missing_hf_record",
                    "subject": lm_record.get("subject", ""),
                    "question": lm_record.get("question", ""),
                    "choices": lm_record.get("choices", []),
                }
            )
    return matched, failures


def prepare_dataset(config: dict[str, Any], *, lm_eval_path: str | Path) -> dict[str, Any]:
    hf_records = load_hf_mmlu_records(
        config["dataset"]["path"],
        config["dataset"]["name"],
        config["dataset"]["split"],
    )
    lm_eval_records = load_lm_eval_records(lm_eval_path)
    matched, failures = align_records(hf_records, lm_eval_records)
    dataset_output = get_path(config, "dataset_with_acc")
    failures_output = get_path(config, "alignment_failures")
    write_jsonl(dataset_output, matched)
    write_jsonl(failures_output, failures)
    return {
        "dataset_output": str(dataset_output),
        "alignment_failures_output": str(failures_output),
        "matched_count": len(matched),
        "failure_count": len(failures),
    }
