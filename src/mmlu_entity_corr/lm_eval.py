"""Helpers for loading and normalizing lm-eval per-question outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_json, read_jsonl
from .normalize import compute_question_uid, normalize_text


def _nested_get(record: dict[str, Any], path: str) -> Any:
    cursor: Any = record
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _first_present(record: dict[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = _nested_get(record, path)
        if value is not None:
            return value
    return None


def _flatten_singletons(value: Any) -> Any:
    cursor = value
    while isinstance(cursor, list) and len(cursor) == 1:
        cursor = cursor[0]
    if isinstance(cursor, dict):
        for key in ("text", "value", "label", "prediction", "answer"):
            if key in cursor:
                return _flatten_singletons(cursor[key])
    return cursor


def _coerce_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(item) for _, item in sorted(value.items())]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_scalar(value: Any) -> Any:
    value = _flatten_singletons(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return value


def _task_to_subject(task_name: Any) -> str | None:
    if not isinstance(task_name, str):
        return None
    if task_name.startswith("mmlu_"):
        return task_name[len("mmlu_") :]
    return task_name


def _label_variants(value: Any, choices: list[str]) -> set[str]:
    variants: set[str] = set()
    scalar = _coerce_scalar(value)
    if scalar is None:
        return variants
    if isinstance(scalar, bool):
        variants.add(str(int(scalar)))
        return variants
    if isinstance(scalar, int):
        variants.add(str(scalar))
        if 0 <= scalar < 26:
            variants.add(chr(ord("A") + scalar))
        if 0 <= scalar < len(choices):
            variants.add(normalize_text(choices[scalar]))
        return variants
    if isinstance(scalar, str):
        variants.add(normalize_text(scalar))
        if len(scalar) == 1 and scalar.upper().isalpha():
            index = ord(scalar.upper()) - ord("A")
            variants.add(str(index))
            if 0 <= index < len(choices):
                variants.add(normalize_text(choices[index]))
        for index, choice in enumerate(choices):
            if normalize_text(choice) == normalize_text(scalar):
                variants.add(str(index))
                variants.add(chr(ord("A") + index))
        return variants
    variants.add(normalize_text(str(scalar)))
    return variants


def compute_acc(prediction: Any, target: Any, choices: list[str]) -> int | None:
    pred_variants = _label_variants(prediction, choices)
    target_variants = _label_variants(target, choices)
    if not pred_variants or not target_variants:
        return None
    return int(bool(pred_variants.intersection(target_variants)))


def iter_lm_eval_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported lm-eval payload type: {type(payload)!r}")
    if "samples" in payload:
        samples = payload["samples"]
        if isinstance(samples, list):
            return [item for item in samples if isinstance(item, dict)]
        if isinstance(samples, dict):
            collected: list[dict[str, Any]] = []
            for task_name, task_samples in samples.items():
                if isinstance(task_samples, list):
                    for item in task_samples:
                        if isinstance(item, dict):
                            merged = dict(item)
                            merged.setdefault("task_name", task_name)
                            collected.append(merged)
            return collected
    if "records" in payload and isinstance(payload["records"], list):
        return [item for item in payload["records"] if isinstance(item, dict)]
    raise ValueError("Could not locate per-question records in lm-eval payload.")


def _load_single_lm_eval_source(source: Path) -> list[dict[str, Any]]:
    if source.suffix == ".jsonl":
        return read_jsonl(source)
    return iter_lm_eval_payload(read_json(source))


def _expand_lm_eval_sources(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted(candidate for candidate in path.rglob("samples_mmlu_*.jsonl") if candidate.is_file())
    if candidates:
        return candidates
    jsonl_candidates = sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file())
    if jsonl_candidates:
        return jsonl_candidates
    json_candidates = sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file())
    if json_candidates:
        return json_candidates
    raise FileNotFoundError(f"No lm-eval sample files found under {path}")


def load_lm_eval_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    sources = _expand_lm_eval_sources(source)
    raw_records: list[dict[str, Any]] = []
    for item in sources:
        raw_records.extend(_load_single_lm_eval_source(item))
    records: list[dict[str, Any]] = []
    for record_index, raw in enumerate(raw_records):
        doc = raw.get("doc", {}) if isinstance(raw.get("doc"), dict) else {}
        question = _first_present(raw, ("question", "query", "prompt", "doc.question", "doc.query"))
        choices = _first_present(raw, ("choices", "options", "doc.choices", "doc.options"))
        answer = _first_present(raw, ("answer", "label", "doc.answer", "doc.label", "target"))
        subject = _first_present(raw, ("subject", "doc.subject", "category")) or _task_to_subject(
            _first_present(raw, ("task_name", "task", "task_alias"))
        )
        prediction = _first_present(
            raw,
            (
                "prediction",
                "pred",
                "model_choice",
                "response",
                "filtered_resps",
                "resps",
                "model_output",
            ),
        )
        target = _first_present(raw, ("target", "gold", "expected", "answer", "doc.answer"))
        acc = _first_present(raw, ("acc", "exact_match", "is_correct"))
        parsed_choices = _coerce_choices(choices)
        if acc is None:
            acc = compute_acc(prediction, target if target is not None else answer, parsed_choices)
        record = {
            "subject": str(subject) if subject is not None else "",
            "question": str(question) if question is not None else "",
            "choices": parsed_choices,
            "prediction": _coerce_scalar(prediction),
            "target": _coerce_scalar(target if target is not None else answer),
            "acc": None if acc is None else int(acc),
            "raw_record": raw,
            "doc": doc,
            "_source_index": record_index,
        }
        record["question_uid"] = compute_question_uid(record["subject"], record["question"], record["choices"])
        records.append(record)
    return records
