from __future__ import annotations

from typing import Sequence


def levenshtein_distance(source: Sequence[str], target: Sequence[str]) -> int:
    if source == target:
        return 0
    if len(source) == 0:
        return len(target)
    if len(target) == 0:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current_row = [i]
        for j, target_item in enumerate(target, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (0 if source_item == target_item else 1)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    edits = levenshtein_distance(list(reference), list(hypothesis))
    return edits / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    edits = levenshtein_distance(reference_words, hypothesis_words)
    return edits / len(reference_words)
