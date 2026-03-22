from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mmlu_entity_corr.lm_eval import load_lm_eval_records
from mmlu_entity_corr.prepare import align_records


class PrepareTests(unittest.TestCase):
    def test_alignment_matches_by_question_uid(self) -> None:
        hf_records = [
            {
                "question_uid": "q1",
                "subject": "history",
                "question": "Who led the French Revolution?",
                "choices": ["Napoleon", "Louis XVI"],
                "answer": 1,
            }
        ]
        lm_eval_records = [
            {
                "question_uid": "q1",
                "subject": "history",
                "question": "Who led the French Revolution?",
                "choices": ["Napoleon", "Louis XVI"],
                "prediction": "B",
                "target": "B",
                "acc": 1,
            }
        ]
        matched, failures = align_records(hf_records, lm_eval_records)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(failures), 0)
        self.assertEqual(matched[0]["acc"], 1)

    def test_duplicate_uid_raises(self) -> None:
        hf_records = [
            {"question_uid": "q1", "subject": "history", "question": "Q1", "choices": [], "answer": 0, "_source_index": 0},
            {"question_uid": "q1", "subject": "history", "question": "Q1", "choices": [], "answer": 0, "_source_index": 1},
        ]
        lm_eval_records = [
            {"question_uid": "q1", "subject": "history", "question": "Q1", "choices": [], "target": 0, "acc": 1, "_source_index": 0},
            {"question_uid": "q1", "subject": "history", "question": "Q1", "choices": [], "target": 0, "acc": 1, "_source_index": 1},
        ]
        matched, failures = align_records(hf_records, lm_eval_records)
        self.assertEqual(len(matched), 2)
        self.assertEqual({row["question_uid"] for row in matched}, {"q1:0", "q1:1"})
        self.assertEqual(len(failures), 0)

    def test_load_lm_eval_records_accepts_directory_of_mmlu_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            sample_dir = base / "results" / "MMLU-Llama" / "model"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / "samples_mmlu_abstract_algebra_fake.jsonl"
            sample_path.write_text(
                '{"doc": {"question": "Q?", "subject": "abstract_algebra", "choices": ["A", "B"], "answer": 1}, "target": "1", "acc": 1.0}\n',
                encoding="utf-8",
            )
            records = load_lm_eval_records(base / "results")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["subject"], "abstract_algebra")


if __name__ == "__main__":
    unittest.main()
