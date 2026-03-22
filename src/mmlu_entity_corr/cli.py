"""Command line interface for the MMLU entity correlation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .analysis import analyze_correlation
from .config import DEFAULT_CONFIG_PATH, load_config
from .extraction import extract_entities
from .json_utils import dumps
from .matching import score_recall
from .prepare import prepare_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmlu-entity-corr")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the experiment config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-dataset", help="Load HF MMLU and align with lm-eval details.")
    prepare_parser.add_argument("--lm-eval", required=True, help="Path to per-question lm-eval JSON or JSONL.")

    extract_parser = subparsers.add_parser("extract-entities", help="Run entity extraction for gold/pred models.")
    extract_parser.add_argument(
        "--target",
        default="both",
        choices=("gold", "pred", "both"),
        help="Which extraction target to run.",
    )
    extract_parser.add_argument("--input", help="Optional path to dataset_with_acc.jsonl.")
    extract_parser.add_argument("--workers", type=int, help="Number of parallel API workers to use for extraction.")
    extract_parser.add_argument("--force", action="store_true", help="Ignore cached extraction outputs.")

    score_parser = subparsers.add_parser("score-recall", help="Compute per-question entity recall.")
    score_parser.add_argument("--dataset", help="Optional path to dataset_with_acc.jsonl.")
    score_parser.add_argument("--gold", help="Optional path to gold_entities.jsonl.")
    score_parser.add_argument("--pred", help="Optional path to pred_entities.jsonl.")

    analyze_parser = subparsers.add_parser("analyze-correlation", help="Run Pearson correlation analysis.")
    analyze_parser.add_argument("--input", help="Optional path to entity_scores.jsonl.")
    return parser


def _run_command(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.command == "prepare-dataset":
        return prepare_dataset(config, lm_eval_path=args.lm_eval)
    if args.command == "extract-entities":
        return extract_entities(
            config,
            target=args.target,
            input_path=args.input,
            workers=args.workers,
            force=args.force,
        )
    if args.command == "score-recall":
        return score_recall(config, dataset_path=args.dataset, gold_path=args.gold, pred_path=args.pred)
    if args.command == "analyze-correlation":
        return analyze_correlation(config, input_path=args.input)
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    result = _run_command(args, config)
    print(dumps(result, pretty=True, sort_keys=True))
    return 0
