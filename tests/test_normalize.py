from __future__ import annotations

import unittest

from mmlu_entity_corr.normalize import (
    compute_question_uid,
    critical_token_signature,
    normalize_entities,
    normalize_text_for_uid,
)


class NormalizeTests(unittest.TestCase):
    def test_normalize_entities_drops_article(self) -> None:
        self.assertEqual(normalize_entities(["the French Revolution"]), ["french revolution"])

    def test_critical_token_signature_tracks_roman_numerals(self) -> None:
        self.assertEqual(critical_token_signature("World War II"), ["ii"])

    def test_question_uid_is_stable(self) -> None:
        uid_a = compute_question_uid("history", "Who led the French Revolution?", ["Napoleon", "Louis XVI"])
        uid_b = compute_question_uid("History", "Who led the French Revolution?", ["Napoleon", "Louis XVI"])
        self.assertEqual(uid_a, uid_b)

    def test_uid_normalization_preserves_semantic_punctuation(self) -> None:
        uid_a = compute_question_uid(
            "abstract_algebra",
            "Find the degree for Q(sqrt(2) + sqrt(3)) over Q.",
            ["0", "4", "2", "6"],
        )
        uid_b = compute_question_uid(
            "abstract_algebra",
            "Find the degree for Q(sqrt(2), sqrt(3)) over Q.",
            ["0", "4", "2", "6"],
        )
        self.assertNotEqual(uid_a, uid_b)

    def test_normalize_text_for_uid_keeps_punctuation(self) -> None:
        self.assertEqual(normalize_text_for_uid("0.6c"), "0.6c")


if __name__ == "__main__":
    unittest.main()
