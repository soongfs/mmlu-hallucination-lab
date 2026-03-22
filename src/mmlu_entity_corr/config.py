"""Configuration loading and path resolution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/experiment.yaml")


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",
    },
    "paths": {
        "artifacts_dir": "artifacts",
        "dataset_with_acc": "artifacts/dataset_with_acc.jsonl",
        "alignment_failures": "artifacts/debug/alignment_failures.jsonl",
        "gold_entities": "artifacts/gold_entities.jsonl",
        "pred_entities": "artifacts/pred_entities.jsonl",
        "entity_scores": "artifacts/entity_scores.jsonl",
        "correlation_summary": "artifacts/reports/correlation_summary.json",
        "included_samples": "artifacts/reports/included_samples.csv",
        "excluded_samples": "artifacts/reports/excluded_samples.csv",
        "threshold_sweep": "artifacts/reports/threshold_sweep.csv",
        "cache_dir": "artifacts/cache",
    },
    "models": {
        "gold": {
            "provider": "openai_responses",
            "model": "gpt-5.4",
            "base_url": "https://sub2api.jntm.us/v1",
            "base_url_env": "OPENAI_BASE_URL",
            "api_key_env": "OPENAI_API_KEY",
            "supports_seed": False,
            "default_headers": {
                "User-Agent": "codex-cli/1.0",
            },
        },
        "pred": {
            "provider": "huggingface_chat",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "api_key_env": "HF_TOKEN",
            "supports_seed": True,
        },
    },
    "extraction": {
        "prompt_version": "v1",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 256,
        "seed": 42,
        "workers": 4,
    },
    "matching": {
        "threshold": 0.9,
        "threshold_sweep": [0.85, 0.88, 0.9, 0.92, 0.95],
    },
    "analysis": {
        "bootstrap_samples": 10000,
        "seed": 42,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    payload = {}
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - exercised when non-JSON YAML is used without PyYAML
                raise RuntimeError(
                    f"Configuration at {config_path} is not JSON-compatible YAML; install PyYAML or convert it."
                ) from exc
            payload = yaml.safe_load(text) or {}
    config = _deep_merge(DEFAULT_CONFIG, payload)
    config["_meta"] = {
        "config_path": str(config_path),
        "base_dir": str(Path.cwd().resolve()),
    }
    return config


def get_path(config: dict[str, Any], key: str) -> Path:
    raw = config["paths"][key]
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    base_dir = Path(config["_meta"]["base_dir"])
    return (base_dir / candidate).resolve()
