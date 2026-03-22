from __future__ import annotations

import unittest

from mmlu_entity_corr.matching import (
    best_one_to_one_matches,
    build_similarity_matrix,
    entity_similarity,
    score_dataset_records,
)


class MatchingTests(unittest.TestCase):
    def test_french_revolution_matches_with_article(self) -> None:
        self.assertGreaterEqual(entity_similarity("french revolution", "the french revolution"), 0.9)

    def test_world_war_i_not_match_world_war_ii(self) -> None:
        self.assertEqual(entity_similarity("world war i", "world war ii"), 0.0)

    def test_h1n1_matches_reordered(self) -> None:
        self.assertGreaterEqual(entity_similarity("h1n1 influenza", "influenza h1n1"), 0.9)

    def test_united_states_does_not_match_longer_name(self) -> None:
        self.assertLess(entity_similarity("united states", "united states of america"), 0.9)

    def test_one_to_one_match_chooses_best_pairs(self) -> None:
        matrix = build_similarity_matrix(["french revolution", "napoleon bonaparte"], ["napoleon bonaparte", "french revolution"])
        matches = best_one_to_one_matches(matrix, 0.9)
        self.assertEqual(len(matches), 2)

    def test_empty_gold_entity_is_excluded(self) -> None:
        scored = score_dataset_records(
            [{"question_uid": "q1", "subject": "history", "question": "Q", "acc": 1}],
            [{"question_uid": "q1", "entities_raw": [], "parse_status": "empty"}],
            [{"question_uid": "q1", "entities_raw": ["Napoleon"], "parse_status": "ok"}],
            threshold=0.9,
        )
        self.assertTrue(scored[0]["excluded"])
        self.assertEqual(scored[0]["exclude_reason"], "empty_gold_entity")


if __name__ == "__main__":
    unittest.main()

