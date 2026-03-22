from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mmlu_entity_corr.analysis import analyze_correlation
from mmlu_entity_corr.config import load_config
from mmlu_entity_corr.io_utils import read_json, read_jsonl, write_jsonl
from mmlu_entity_corr.matching import score_recall


class PipelineIntegrationTests(unittest.TestCase):
    def test_score_and_analyze_pipeline_outputs_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            config = load_config()
            config["_meta"]["base_dir"] = str(base)
            dataset_rows = [
                {
                    "question_uid": "q1",
                    "subject": "history",
                    "question": "Who led the French Revolution?",
                    "choices": ["Napoleon Bonaparte", "Louis XVI"],
                    "answer": 0,
                    "acc": 1,
                },
                {
                    "question_uid": "q2",
                    "subject": "medicine",
                    "question": "Which virus causes H1N1 influenza?",
                    "choices": ["H1N1 influenza", "SARS-CoV-2"],
                    "answer": 0,
                    "acc": 0,
                },
                {
                    "question_uid": "q3",
                    "subject": "geography",
                    "question": "What is the capital city?",
                    "choices": ["Paris", "Rome"],
                    "answer": 0,
                    "acc": 1,
                },
            ]
            gold_rows = [
                {"question_uid": "q1", "entities_raw": ["French Revolution", "Napoleon Bonaparte"], "parse_status": "ok"},
                {"question_uid": "q2", "entities_raw": ["H1N1 influenza"], "parse_status": "ok"},
                {"question_uid": "q3", "entities_raw": [], "parse_status": "empty"},
            ]
            pred_rows = [
                {"question_uid": "q1", "entities_raw": ["the French Revolution", "Napoleon Bonaparte"], "parse_status": "ok"},
                {"question_uid": "q2", "entities_raw": ["influenza H1N1"], "parse_status": "ok"},
                {"question_uid": "q3", "entities_raw": ["capital city"], "parse_status": "ok"},
            ]
            write_jsonl(base / "artifacts/dataset_with_acc.jsonl", dataset_rows)
            write_jsonl(base / "artifacts/gold_entities.jsonl", gold_rows)
            write_jsonl(base / "artifacts/pred_entities.jsonl", pred_rows)

            score_recall(config)
            summary = analyze_correlation(config)

            scores = read_jsonl(base / "artifacts/entity_scores.jsonl")
            self.assertEqual(len(scores), 3)
            self.assertEqual(sum(not row["excluded"] for row in scores), 2)
            self.assertEqual(summary["n"], 2)
            self.assertTrue((base / "artifacts/reports/correlation_summary.json").exists())
            self.assertTrue((base / "artifacts/reports/threshold_sweep.csv").exists())
            parsed_summary = read_json(base / "artifacts/reports/correlation_summary.json")
            self.assertIn("bootstrap_ci_95", parsed_summary)


if __name__ == "__main__":
    unittest.main()
