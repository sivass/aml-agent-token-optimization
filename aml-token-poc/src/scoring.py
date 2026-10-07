"""Weighted AML factor score (0-100) and risk band.

Weights and thresholds come from config.py. They are POC assumptions:
AML/business users must validate them before production use.
"""
from __future__ import annotations

import config


def factor_credit(value) -> float | None:
    """Turn a factor result into 0.0-1.0 credit. None = cannot be compared."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return max(0.0, min(1.0, float(value)))


def score_breakdown(factors: dict, weights: dict | None = None,
                    missing_credit: float | None = None) -> dict:
    """Points contributed by each weighted factor, scaled so the max is 100."""
    weights = weights or config.FACTOR_WEIGHTS
    if missing_credit is None:
        missing_credit = config.MISSING_FACTOR_CREDIT
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("FACTOR_WEIGHTS must sum to more than zero")
    breakdown = {}
    for name, weight in weights.items():
        credit = factor_credit(factors.get(name))
        if credit is None:
            credit = missing_credit
        breakdown[name] = round(100 * weight * credit / total_weight, 2)
    return breakdown


def calculate_weighted_score(factors: dict, weights: dict | None = None,
                             missing_credit: float | None = None) -> int:
    points = sum(score_breakdown(factors, weights, missing_credit).values())
    return int(points + 0.5)


def risk_band(score: float, bands: list | None = None) -> str:
    for minimum, band in sorted(bands or config.RISK_BANDS, reverse=True):
        if score >= minimum:
            return band
    return sorted(bands or config.RISK_BANDS)[0][1]
