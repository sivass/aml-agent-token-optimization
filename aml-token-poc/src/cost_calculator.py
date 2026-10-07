"""Token -> cost, and the monthly Finance projection."""
from __future__ import annotations

import config


def calculate_cost(input_tokens: float, output_tokens: float,
                   input_price: float | None = None,
                   output_price: float | None = None) -> dict:
    """Prices are USD per 1M tokens."""
    input_price = config.INPUT_PRICE_PER_1M_TOKENS if input_price is None else input_price
    output_price = config.OUTPUT_PRICE_PER_1M_TOKENS if output_price is None else output_price
    input_cost = input_tokens / 1_000_000 * input_price
    output_cost = output_tokens / 1_000_000 * output_price
    return {"input_cost": input_cost, "output_cost": output_cost,
            "total_cost": input_cost + output_cost}


def average_hits(distribution: dict) -> float:
    """Average hits per alert for a {hits: share_of_alerts} distribution."""
    total_share = sum(distribution.values())
    return sum(hits * share for hits, share in distribution.items()) / total_share


def build_finance_projection(avg_input_per_hit: float, avg_output_per_hit: float,
                             observed_avg_hits: float,
                             scenarios: dict | None = None, volumes: list | None = None,
                             input_price: float | None = None,
                             output_price: float | None = None) -> list[dict]:
    """One row per scenario x monthly alert volume.

    tokens per alert = average hits per alert x average tokens per hit,
    which is why the number of watchlist hits drives the cost.
    """
    scenarios = config.FINANCE_SCENARIOS if scenarios is None else scenarios
    volumes = sorted(set(volumes or config.PROJECTION_VOLUMES + [config.MONTHLY_ALERTS]))
    hit_rates = {"observed_sample": observed_avg_hits}
    hit_rates.update({name: average_hits(dist) for name, dist in scenarios.items()})

    rows = []
    for scenario, hits_per_alert in hit_rates.items():
        input_per_alert = hits_per_alert * avg_input_per_hit
        output_per_alert = hits_per_alert * avg_output_per_hit
        for alerts in volumes:
            monthly_input = input_per_alert * alerts
            monthly_output = output_per_alert * alerts
            cost = calculate_cost(monthly_input, monthly_output, input_price, output_price)
            rows.append({
                "scenario": scenario,
                "monthly_alerts": alerts,
                "average_hits_per_alert": round(hits_per_alert, 2),
                "average_tokens_per_alert": round(input_per_alert + output_per_alert, 1),
                "monthly_input_tokens": round(monthly_input),
                "monthly_output_tokens": round(monthly_output),
                "monthly_total_tokens": round(monthly_input + monthly_output),
                "estimated_monthly_cost": round(cost["total_cost"], 2),
            })
    return rows
