"""Normalization helpers for question IDs and entity matching."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable


ROMAN_NUMERAL_TOKENS = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
}

PUNCT_RE = re.compile(r"[^\w\s]")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(text: object, *, drop_leading_article: bool = False) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = normalized.replace("-", " ").replace("/", " ")
    normalized = PUNCT_RE.sub(" ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized).strip()
    if drop_leading_article:
        tokens = normalized.split()
        if len(tokens) > 1 and tokens[0] in {"a", "an", "the"}:
            normalized = " ".join(tokens[1:])
    return normalized


def normalize_text_for_uid(text: object) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return MULTISPACE_RE.sub(" ", normalized)


def normalize_choice_list(choices: Iterable[object]) -> list[str]:
    return [normalize_text(choice) for choice in choices]


def normalize_choice_list_for_uid(choices: Iterable[object]) -> list[str]:
    return [normalize_text_for_uid(choice) for choice in choices]


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def normalize_entities(entities: Iterable[object]) -> list[str]:
    normalized = [normalize_text(entity, drop_leading_article=True) for entity in entities]
    filtered = [entity for entity in normalized if entity]
    return dedupe_preserve_order(filtered)


def critical_token_signature(text: str) -> list[str]:
    signature: list[str] = []
    for token in normalize_text(text).split():
        if any(character.isdigit() for character in token) or token in ROMAN_NUMERAL_TOKENS:
            signature.append(token)
    return signature


def compute_question_uid(subject: object, question: object, choices: Iterable[object]) -> str:
    subject_norm = normalize_text_for_uid(subject)
    question_norm = normalize_text_for_uid(question)
    choices_norm = "\n".join(normalize_choice_list_for_uid(choices))
    payload = "\n".join((subject_norm, question_norm, choices_norm))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
