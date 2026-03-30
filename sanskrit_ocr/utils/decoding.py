from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from sanskrit_ocr.data.charset import SanskritCharset


def collapse_repeats(indices: List[int], blank_index: int) -> List[int]:
    collapsed: List[int] = []
    previous = None
    for idx in indices:
        if idx == blank_index:
            previous = None
            continue
        if idx != previous:
            collapsed.append(idx)
        previous = idx
    return collapsed


def ctc_greedy_decode(log_probs: Tensor, charset: SanskritCharset) -> List[Tuple[str, float]]:
    probabilities = log_probs.exp()
    max_probs, max_indices = probabilities.max(dim=2)
    results: List[Tuple[str, float]] = []
    for batch_idx in range(log_probs.shape[1]):
        decoded_indices = collapse_repeats(max_indices[:, batch_idx].tolist(), charset.blank_index)
        text = charset.decode(decoded_indices)
        confidence = max_probs[:, batch_idx].mean().item()
        results.append((text, confidence))
    return results


def _log_sum_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_beam_search_decode(log_probs: Tensor, charset: SanskritCharset, beam_width: int = 5) -> List[Tuple[str, float]]:
    probs = log_probs.cpu()
    time_steps, batch_size, num_classes = probs.shape
    results: List[Tuple[str, float]] = []

    for batch_idx in range(batch_size):
        beam: Dict[Tuple[int, ...], Tuple[float, float]] = {tuple(): (0.0, -math.inf)}
        for timestep in range(time_steps):
            next_beam: Dict[Tuple[int, ...], Tuple[float, float]] = defaultdict(lambda: (-math.inf, -math.inf))
            for prefix, (prob_blank, prob_non_blank) in beam.items():
                for char_idx in range(num_classes):
                    log_p = probs[timestep, batch_idx, char_idx].item()
                    if char_idx == charset.blank_index:
                        blank_score, non_blank_score = next_beam[prefix]
                        next_beam[prefix] = (
                            _log_sum_exp(blank_score, _log_sum_exp(prob_blank + log_p, prob_non_blank + log_p)),
                            non_blank_score,
                        )
                        continue

                    last_char = prefix[-1] if prefix else None
                    extended = prefix + (char_idx,)
                    if char_idx == last_char:
                        blank_score, non_blank_score = next_beam[prefix]
                        next_beam[prefix] = (
                            blank_score,
                            _log_sum_exp(non_blank_score, prob_non_blank + log_p),
                        )
                        blank_score, non_blank_score = next_beam[extended]
                        next_beam[extended] = (
                            blank_score,
                            _log_sum_exp(non_blank_score, prob_blank + log_p),
                        )
                    else:
                        blank_score, non_blank_score = next_beam[extended]
                        next_beam[extended] = (
                            blank_score,
                            _log_sum_exp(non_blank_score, _log_sum_exp(prob_blank + log_p, prob_non_blank + log_p)),
                        )

            beam = dict(
                sorted(
                    next_beam.items(),
                    key=lambda item: _log_sum_exp(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam_width]
            )

        best_prefix, (prob_blank, prob_non_blank) = max(
            beam.items(),
            key=lambda item: _log_sum_exp(item[1][0], item[1][1]),
        )
        confidence = math.exp(_log_sum_exp(prob_blank, prob_non_blank))
        results.append((charset.decode(list(best_prefix)), confidence))

    return results
