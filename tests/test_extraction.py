from __future__ import annotations

import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmlu_entity_corr.config import load_config
from mmlu_entity_corr.extraction import extract_entities, extract_entities_for_records, parse_entity_response


class ExtractionTests(unittest.TestCase):
    def test_parse_direct_json_array(self) -> None:
        self.assertEqual(parse_entity_response('["French Revolution", "Napoleon"]'), ["French Revolution", "Napoleon"])

    def test_parse_json_embedded_in_text(self) -> None:
        payload = "Here is the result:\n```json\n[\"French Revolution\"]\n```"
        self.assertEqual(parse_entity_response(payload), ["French Revolution"])

    def test_parse_invalid_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_entity_response("not json")

    def test_cached_extraction_skips_request(self) -> None:
        records = [{"question_uid": "q1", "question": "Who led the French Revolution?"}]
        model_config = {"model": "gpt-5.4"}
        extraction_config = {"prompt_version": "v1", "temperature": 0, "top_p": 1, "max_tokens": 32, "seed": 42}
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            with patch("mmlu_entity_corr.extraction._request_entities", return_value=('["Napoleon"]', ["Napoleon"])) as mocked:
                first = extract_entities_for_records(
                    records,
                    model_config=model_config,
                    extraction_config=extraction_config,
                    cache_dir=cache_dir,
                )
                second = extract_entities_for_records(
                    records,
                    model_config=model_config,
                    extraction_config=extraction_config,
                    cache_dir=cache_dir,
                )
        self.assertEqual(first[0]["entities_raw"], ["Napoleon"])
        self.assertEqual(second[0]["entities_raw"], ["Napoleon"])
        self.assertEqual(mocked.call_count, 1)

    def test_parallel_extraction_preserves_input_order(self) -> None:
        records = [
            {"question_uid": "q1", "question": "Q1"},
            {"question_uid": "q2", "question": "Q2"},
            {"question_uid": "q3", "question": "Q3"},
        ]
        model_config = {"model": "gpt-5.4", "provider": "openai_responses"}
        extraction_config = {"prompt_version": "v1", "temperature": 0, "top_p": 1, "max_tokens": 32, "seed": 42}

        def fake_request(question: str, _model_config: dict[str, object], _extraction_config: dict[str, object]):
            delays = {"Q1": 0.03, "Q2": 0.01, "Q3": 0.02}
            time.sleep(delays[question])
            return f'["{question}"]', [question]

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            with patch("mmlu_entity_corr.extraction._request_entities", side_effect=fake_request):
                outputs = extract_entities_for_records(
                    records,
                    model_config=model_config,
                    extraction_config=extraction_config,
                    cache_dir=cache_dir,
                    workers=3,
                )
        self.assertEqual([row["question_uid"] for row in outputs], ["q1", "q2", "q3"])
        self.assertEqual([row["entities_raw"] for row in outputs], [["Q1"], ["Q2"], ["Q3"]])

    def test_extract_entities_rejects_non_positive_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "artifacts/dataset_with_acc.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text('{"question_uid":"q1","question":"Q1"}\n', encoding="utf-8")
            config = load_config()
            config["_meta"]["base_dir"] = tmp_dir
            with self.assertRaises(ValueError):
                extract_entities(config, target="gold", workers=0)

    def test_extract_entities_requires_prepared_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_config()
            config["_meta"]["base_dir"] = tmp_dir
            with self.assertRaises(FileNotFoundError):
                extract_entities(config, target="gold")

    def test_extract_entities_rejects_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_path = base / "artifacts/dataset_with_acc.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("", encoding="utf-8")
            config = load_config()
            config["_meta"]["base_dir"] = tmp_dir
            with self.assertRaises(ValueError):
                extract_entities(config, target="gold")


if __name__ == "__main__":
    unittest.main()
