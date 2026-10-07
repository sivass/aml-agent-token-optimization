"""AML AI token usage POC - entry point.

    python main.py                    # offline simulation, no Azure key needed
    python main.py --regenerate-data  # rewrite the synthetic CSV first

Pipeline per hit (party_id + entry_id):
    CSV -> factor comparison -> weighted score -> evidence -> prompt
        -> token count -> simulated (or Azure) AI response -> cost
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from src import azure_openai, reporting
from src.ai_prompt import build_evidence, build_messages, simulate_ai_response
from src.cost_calculator import build_finance_projection, calculate_cost
from src.csv_loader import load_hits
from src.factor_matching import compare_hit
from src.generate_dataset import write_dataset
from src.scoring import calculate_weighted_score, risk_band, score_breakdown
from src.token_counter import TokenCounter


def process_hit(row: dict, counter: TokenCounter, azure_client=None) -> dict:
    """Deterministic screening first, then the AI assessment and token count."""
    factors = compare_hit(row)
    score = calculate_weighted_score(factors)
    band = risk_band(score)
    evidence = build_evidence(row, factors, score, band)
    messages = build_messages(evidence)
    sent_to_ai = score >= config.AI_MIN_FACTOR_SCORE

    simulated_response = simulate_ai_response(evidence) if sent_to_ai else None
    simulated_input = simulated_output = 0
    if sent_to_ai:
        simulated_input = counter.count_messages(messages)
        if config.OUTPUT_TOKEN_MODE == "fixed":
            simulated_output = config.EXPECTED_OUTPUT_TOKENS
        else:
            simulated_output = counter.count_text(json.dumps(simulated_response, indent=2))

    actual = azure_openai.assess(azure_client, messages) if (sent_to_ai and azure_client) else None
    input_tokens = actual["input_tokens"] if actual else simulated_input
    output_tokens = actual["output_tokens"] if actual else simulated_output
    cost = calculate_cost(input_tokens, output_tokens)

    return {
        "alert_id": row["alert_id"],
        "party_id": row["party_id"],
        "entry_id": row["entry_id"],
        "factor_results": factors,
        "score_breakdown": score_breakdown(factors),
        "weighted_factor_score": score,
        "risk_band": band,
        "sent_to_ai": sent_to_ai,
        "evidence": evidence,
        "messages": messages if sent_to_ai else None,
        "ai_response": (actual["response"] or actual["response_text"]) if actual else simulated_response,
        "ai_response_source": "azure_openai" if actual else ("simulated" if sent_to_ai else "not_sent"),
        # The AI only recommends. The decision is always the analyst's.
        "final_decision": config.PENDING_DECISION,
        "simulated_input_tokens": simulated_input,
        "simulated_output_tokens": simulated_output,
        "actual_input_tokens": actual["input_tokens"] if actual else None,
        "actual_output_tokens": actual["output_tokens"] if actual else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": round(cost["total_cost"], 6),
        "token_source": "azure_actual" if actual else ("offline_estimate" if sent_to_ai else "not_sent"),
    }


def run(input_path: Path | None = None, output_dir: Path | None = None,
        counter: TokenCounter | None = None, azure_client=None) -> dict:
    input_path = Path(input_path or config.DATA_FILE)
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    counter = counter or TokenCounter()
    if azure_client is None and config.USE_AZURE_OPENAI:
        azure_client = azure_openai.create_client()

    rows = load_hits(input_path)
    hits = [process_hit(row, counter, azure_client) for row in rows]
    alerts = reporting.aggregate_alerts(hits)
    summary = reporting.summarise(hits, alerts)
    projection = build_finance_projection(
        summary["avg_input_per_hit"], summary["avg_output_per_hit"],
        summary["avg_hits_per_alert"],
    )
    files = reporting.write_reports(output_dir, hits, alerts, projection)
    console = reporting.format_console(summary, projection, hits, counter.method,
                                       used_azure=azure_client is not None)
    return {"hits": hits, "alerts": alerts, "summary": summary,
            "projection": projection, "files": files, "console": console}


def main() -> None:
    parser = argparse.ArgumentParser(description="AML AI token usage POC")
    parser.add_argument("--input", type=Path, default=config.DATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--regenerate-data", action="store_true",
                        help="rewrite the synthetic CSV before running")
    args = parser.parse_args()

    if args.regenerate_data or (args.input == config.DATA_FILE and not args.input.exists()):
        count = write_dataset(config.DATA_FILE)
        print(f"Synthetic dataset written: {config.DATA_FILE} ({count} hits)\n")

    result = run(args.input, args.output_dir)
    print(result["console"])
    print("\nReports written:")
    for path in result["files"]:
        print(f"  {path}")
    (args.output_dir / "aml_console_summary.txt").write_text(result["console"] + "\n",
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
