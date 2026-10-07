"""Alert-level aggregation, CSV reports and the console summary."""
from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import config
from src.cost_calculator import calculate_cost
from src.factor_matching import FACTOR_NAMES

HIT_COLUMNS = [
    "alert_id", "party_id", "entry_id", "weighted_factor_score", "risk_band",
    "input_tokens", "output_tokens", "total_tokens", "estimated_cost",
    "sent_to_ai", "token_source",
]
ALERT_COLUMNS = [
    "alert_id", "party_id", "watchlist_hits", "total_input_tokens",
    "total_output_tokens", "total_tokens", "average_tokens_per_hit",
    "estimated_cost", "max_factor_score", "hits_sent_to_ai",
]
PROJECTION_COLUMNS = [
    "scenario", "monthly_alerts", "average_hits_per_alert", "average_tokens_per_alert",
    "monthly_input_tokens", "monthly_output_tokens", "monthly_total_tokens",
    "estimated_monthly_cost",
]
VALIDATION_COLUMNS = [
    "alert_id", "entry_id", "simulated_input_tokens", "actual_input_tokens",
    "input_delta_pct", "simulated_output_tokens", "actual_output_tokens",
    "output_delta_pct",
]


def aggregate_alerts(hits: list[dict]) -> list[dict]:
    """Alert token usage = sum of its hit-level AI processing."""
    groups: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for hit in hits:
        groups.setdefault((hit["alert_id"], hit["party_id"]), []).append(hit)
    alerts = []
    for (alert_id, party_id), group in groups.items():
        total_in = sum(h["input_tokens"] for h in group)
        total_out = sum(h["output_tokens"] for h in group)
        alerts.append({
            "alert_id": alert_id,
            "party_id": party_id,
            "watchlist_hits": len(group),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "average_tokens_per_hit": round((total_in + total_out) / len(group), 1),
            "estimated_cost": round(sum(h["estimated_cost"] for h in group), 6),
            "max_factor_score": max(h["weighted_factor_score"] for h in group),
            "hits_sent_to_ai": sum(1 for h in group if h["sent_to_ai"]),
        })
    return alerts


def _pct(simulated: float, actual: float) -> float | str:
    return round((simulated - actual) / actual * 100, 1) if actual else ""


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir: Path, hits: list[dict], alerts: list[dict],
                  projection: list[dict]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    path = output_dir / "aml_hit_token_results.csv"
    _write_csv(path, HIT_COLUMNS, hits)
    written.append(path)

    path = output_dir / "aml_alert_token_results.csv"
    _write_csv(path, ALERT_COLUMNS, alerts)
    written.append(path)

    path = output_dir / "aml_finance_projection.csv"
    _write_csv(path, PROJECTION_COLUMNS, projection)
    written.append(path)

    # Full audit trail per hit: factors, evidence, prompt and AI response.
    path = output_dir / "aml_hit_details.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for hit in hits:
            handle.write(json.dumps({
                key: hit[key] for key in (
                    "alert_id", "party_id", "entry_id", "factor_results", "score_breakdown",
                    "weighted_factor_score", "risk_band", "sent_to_ai", "evidence",
                    "messages", "ai_response", "ai_response_source", "final_decision",
                )
            }) + "\n")
    written.append(path)

    actual = [h for h in hits if h.get("actual_input_tokens") is not None]
    if actual:
        path = output_dir / "aml_token_validation.csv"
        _write_csv(path, VALIDATION_COLUMNS, [{
            "alert_id": h["alert_id"],
            "entry_id": h["entry_id"],
            "simulated_input_tokens": h["simulated_input_tokens"],
            "actual_input_tokens": h["actual_input_tokens"],
            "input_delta_pct": _pct(h["simulated_input_tokens"], h["actual_input_tokens"]),
            "simulated_output_tokens": h["simulated_output_tokens"],
            "actual_output_tokens": h["actual_output_tokens"],
            "output_delta_pct": _pct(h["simulated_output_tokens"], h["actual_output_tokens"]),
        } for h in actual])
        written.append(path)
    return written


def summarise(hits: list[dict], alerts: list[dict]) -> dict:
    n_hits, n_alerts = len(hits), len(alerts)
    total_in = sum(h["input_tokens"] for h in hits)
    total_out = sum(h["output_tokens"] for h in hits)
    total_cost = sum(h["estimated_cost"] for h in hits)
    per_alert = sorted(a["total_tokens"] for a in alerts)
    by_hit_count: dict[int, list[int]] = {}
    for alert in alerts:
        by_hit_count.setdefault(alert["watchlist_hits"], []).append(alert["total_tokens"])
    return {
        "alerts": n_alerts,
        "hits": n_hits,
        "hits_sent_to_ai": sum(1 for h in hits if h["sent_to_ai"]),
        "avg_hits_per_alert": n_hits / n_alerts,
        "factors_per_hit": len(FACTOR_NAMES),
        "avg_input_per_hit": total_in / n_hits,
        "avg_output_per_hit": total_out / n_hits,
        "avg_total_per_hit": (total_in + total_out) / n_hits,
        "avg_tokens_per_alert": (total_in + total_out) / n_alerts,
        "min_tokens_per_alert": per_alert[0],
        "max_tokens_per_alert": per_alert[-1],
        "avg_cost_per_hit": total_cost / n_hits,
        "avg_cost_per_alert": total_cost / n_alerts,
        "tokens_by_hit_count": {k: sum(v) / len(v) for k, v in sorted(by_hit_count.items())},
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost": total_cost,
    }


def _millions(tokens: float) -> str:
    return f"{tokens / 1_000_000:,.2f}M"


def format_console(summary: dict, projection: list[dict], hits: list[dict],
                   tokenizer_method: str, used_azure: bool) -> str:
    s = summary
    line = "-" * 56
    out = [
        "=" * 56, "AML AI TOKEN USAGE POC", "=" * 56, "",
        f"Mode:                         {'Azure OpenAI (actual usage)' if used_azure else 'Offline simulation'}",
        f"Tokenizer:                    {tokenizer_method}",
        f"Output tokens:                {config.OUTPUT_TOKEN_MODE}"
        + (f" ({config.EXPECTED_OUTPUT_TOKENS}/hit)" if config.OUTPUT_TOKEN_MODE == "fixed" else ""),
        f"Price / 1M tokens (USD):      input {config.INPUT_PRICE_PER_1M_TOKENS:.2f}, "
        f"output {config.OUTPUT_PRICE_PER_1M_TOKENS:.2f}",
        "",
        f"Alerts analysed:              {s['alerts']}",
        f"Watchlist hits:               {s['hits']}",
        f"Hits sent to AI:              {s['hits_sent_to_ai']}",
        f"Average hits / alert:         {s['avg_hits_per_alert']:.2f}",
        f"Factors evaluated / hit:      {s['factors_per_hit']}",
        "",
        f"Average input tokens / hit:   {s['avg_input_per_hit']:,.0f}",
        f"Average output tokens / hit:  {s['avg_output_per_hit']:,.0f}",
        f"Average total tokens / hit:   {s['avg_total_per_hit']:,.0f}",
        "",
        f"Average tokens / alert:       {s['avg_tokens_per_alert']:,.0f}",
        f"Min / max tokens per alert:   {s['min_tokens_per_alert']:,} / {s['max_tokens_per_alert']:,}",
        f"Average cost / hit:           USD {s['avg_cost_per_hit']:.6f}",
        f"Average cost / alert:         USD {s['avg_cost_per_alert']:.6f}",
        "", line, "EFFECT OF MULTIPLE HITS (average tokens per alert)", line,
    ]
    for hit_count, tokens in s["tokens_by_hit_count"].items():
        out.append(f"{hit_count} hit(s) / alert:             {tokens:,.0f}")

    out += ["", line, "FINANCE PROJECTION (observed sample mix)", line]
    for row in projection:
        if row["scenario"] == "observed_sample":
            out += [
                f"{row['monthly_alerts']:,} alerts/month",
                f"Estimated tokens:             {_millions(row['monthly_total_tokens'])}",
                f"Estimated cost:               USD {row['estimated_monthly_cost']:,.2f}",
                "",
            ]

    out += [line, f"SCENARIOS at {config.MONTHLY_ALERTS:,} alerts/month", line,
            f"{'scenario':<17}{'hits/alert':>10}{'tokens/alert':>14}{'tokens':>10}{'USD/month':>12}"]
    for row in projection:
        if row["monthly_alerts"] == config.MONTHLY_ALERTS:
            out.append(
                f"{row['scenario']:<17}{row['average_hits_per_alert']:>10.2f}"
                f"{row['average_tokens_per_alert']:>14,.0f}"
                f"{_millions(row['monthly_total_tokens']):>10}"
                f"{row['estimated_monthly_cost']:>12,.2f}"
            )

    if not used_azure and config.OUTPUT_TOKEN_MODE != "fixed":
        ceiling = calculate_cost(s["avg_input_per_hit"], config.EXPECTED_OUTPUT_TOKENS)
        out += ["", line, "OUTPUT-SIZE SENSITIVITY", line,
                f"If every response used {config.EXPECTED_OUTPUT_TOKENS} output tokens:",
                f"Tokens / hit:                 {s['avg_input_per_hit'] + config.EXPECTED_OUTPUT_TOKENS:,.0f}",
                f"Cost / alert:                 USD {ceiling['total_cost'] * s['avg_hits_per_alert']:.6f}"]

    actual = [h for h in hits if h.get("actual_input_tokens") is not None]
    if actual:
        act_in = sum(h["actual_input_tokens"] for h in actual)
        act_out = sum(h["actual_output_tokens"] for h in actual)
        sim_in = sum(h["simulated_input_tokens"] for h in actual)
        sim_out = sum(h["simulated_output_tokens"] for h in actual)
        cost = calculate_cost(act_in, act_out)["total_cost"]
        out += ["", line, "Actual Azure OpenAI Usage", line,
                f"Prompt tokens:                {act_in:,}",
                f"Completion tokens:            {act_out:,}",
                f"Total tokens:                 {act_in + act_out:,}",
                f"Estimated cost:               USD {cost:.4f}",
                "",
                f"Offline estimate vs actual:   input {_pct(sim_in, act_in):+}%, "
                f"output {_pct(sim_out, act_out):+}%"]
    return "\n".join(out)
