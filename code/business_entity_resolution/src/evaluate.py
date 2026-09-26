"""Metric and blocking-quality evaluation functions for entity resolution."""

from typing import Mapping, Iterable, Any


def f05(pred: set[str], truth: set[str]) -> float:
    """
    Calculate F0.5 score for a single record.

    F0.5 = 1.25 * P * R / (0.25 * P + R)
    where P = precision, R = recall

    Special cases:
    - truth empty & pred empty → 1.0
    - truth empty & pred non-empty → 0.0
    - truth non-empty & pred empty → 0.0
    - truth non-empty & no overlap → 0.0
    """
    # Special case: both empty
    if not truth and not pred:
        return 1.0

    # Special case: truth empty but pred non-empty
    if not truth and pred:
        return 0.0

    # Special case: truth non-empty but pred empty
    if truth and not pred:
        return 0.0

    # Both non-empty: calculate F0.5
    overlap = len(pred & truth)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(truth) if truth else 0.0

    # If no overlap, F0.5 = 0
    if overlap == 0:
        return 0.0

    # F0.5 = 1.25 * P * R / (0.25 * P + R)
    numerator = 1.25 * precision * recall
    denominator = 0.25 * precision + recall

    if denominator == 0:
        return 0.0

    return numerator / denominator


def macro_f05(
    pred: Mapping[str, set[str]],
    truth: Mapping[str, set[str]],
    s1_ids: Iterable[str]
) -> float:
    """
    Calculate macro F0.5 score over all given S1 ids.

    Missing keys are treated as empty sets.
    Returns the mean F0.5 score across all S1 ids.
    """
    scores = []
    for s1_id in s1_ids:
        pred_set = pred.get(s1_id, set())
        truth_set = truth.get(s1_id, set())
        score = f05(pred_set, truth_set)
        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def pair_recall(
    cands: Mapping[str, set[str]],
    truth: Mapping[str, set[str]]
) -> float:
    """
    Calculate pair recall over all true pairs.

    pair_recall = (true pairs present in candidates) / (all true pairs)

    If there are zero true pairs, return 1.0.
    """
    # Count all true pairs
    total_true_pairs = 0
    for s1_id, true_set in truth.items():
        total_true_pairs += len(true_set)

    # If no true pairs, return 1.0
    if total_true_pairs == 0:
        return 1.0

    # Count how many true pairs are in candidates
    found_true_pairs = 0
    for s1_id, true_set in truth.items():
        cand_set = cands.get(s1_id, set())
        # Count pairs from true_set that are in cand_set
        found_true_pairs += len(true_set & cand_set)

    return found_true_pairs / total_true_pairs


def reduction_ratio(n_candidate_pairs: int, n_s1: int, n_s23: int) -> float:
    """
    Calculate reduction ratio.

    reduction_ratio = 1 - n_candidate_pairs / (n_s1 * n_s23)
    """
    max_pairs = n_s1 * n_s23
    if max_pairs == 0:
        return 1.0
    return 1.0 - (n_candidate_pairs / max_pairs)


def report(
    pred: Mapping[str, set[str]],
    truth: Mapping[str, set[str]],
    s1_country: Mapping[str, str]
) -> dict[str, Any]:
    """
    Generate a comprehensive evaluation report.

    Returns a dict with keys:
    - overall: macro F0.5 over all S1 ids
    - by_country: dict of macro F0.5 per country
    - singletons: macro F0.5 for S1 records with empty truth (None if none exist)
    - pair_precision: precision for pairs (pooled over all records)
    - pair_recall: recall for pairs (pooled over all records)
    """
    s1_ids = list(s1_country.keys())

    # Calculate overall macro F0.5
    overall = macro_f05(pred, truth, s1_ids)

    # Calculate by_country
    by_country_data = {}  # country -> list of scores
    for s1_id in s1_ids:
        country = s1_country[s1_id]
        if country not in by_country_data:
            by_country_data[country] = []

        pred_set = pred.get(s1_id, set())
        truth_set = truth.get(s1_id, set())
        score = f05(pred_set, truth_set)
        by_country_data[country].append(score)

    by_country = {country: sum(scores) / len(scores) for country, scores in by_country_data.items()}

    # Calculate singletons (S1 records with empty truth)
    singleton_scores = []
    for s1_id in s1_ids:
        truth_set = truth.get(s1_id, set())
        if not truth_set:  # Empty truth set
            pred_set = pred.get(s1_id, set())
            score = f05(pred_set, truth_set)
            singleton_scores.append(score)

    singletons = sum(singleton_scores) / len(singleton_scores) if singleton_scores else None

    # Calculate pair precision and recall (restricted to s1_country's keys)
    total_pred_pairs = 0
    total_true_pairs = 0

    for s1_id in s1_ids:
        pred_set = pred.get(s1_id, set())
        truth_set = truth.get(s1_id, set())
        total_pred_pairs += len(pred_set)
        total_true_pairs += len(truth_set)

    # Count overlap pairs (restricted to s1_country's keys)
    overlap_pairs = 0
    for s1_id in s1_ids:
        pred_set = pred.get(s1_id, set())
        truth_set = truth.get(s1_id, set())
        overlap_pairs += len(pred_set & truth_set)

    pair_precision = overlap_pairs / total_pred_pairs if total_pred_pairs > 0 else 0.0

    # pair_recall restricted to s1_country's keys
    if total_true_pairs == 0:
        pair_recall_val = 1.0
    else:
        pair_recall_val = overlap_pairs / total_true_pairs

    return {
        "overall": overall,
        "by_country": by_country,
        "singletons": singletons,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall_val,
    }
