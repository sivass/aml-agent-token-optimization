
import os
import re
import json
import math
from pathlib import Path

import pandas as pd

# Offline tokenizer. For a production measurement, use the tokenizer
# appropriate to the selected Azure OpenAI model.
try:
    import tiktoken
    enc = tiktoken.encoding_for_model(os.getenv("TOKENIZER_MODEL", "gpt-4o"))
except Exception:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

INPUT_CSV = os.getenv("INPUT_CSV", "aml_party_watchlist_factors.csv")
MONTHLY_ALERTS = int(os.getenv("MONTHLY_ALERTS", "10000"))

# Example weights only. These are POC assumptions and should be agreed
# with the AML/business team before production.
WEIGHTS = {
    "name": 30,
    "dob": 20,
    "nationality": 10,
    "gender": 5,
    "country": 10,
    "name_exact": 10,
    "dob_exact": 5,
    "country_exact": 5,
}
# Total = 95; normalize to 100 below.

def normalize(value):
    if value is None:
        return ""
    value = str(value).upper().strip()
    return re.sub(r"[^A-Z0-9]", "", value)

def token_count(text):
    return len(enc.encode(text))

def similarity(a, b):
    # Lightweight deterministic character/token similarity for POC.
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Jaccard over bigrams
    def grams(s):
        return {s[i:i+2] for i in range(len(s)-1)} or {s}
    x, y = grams(a), grams(b)
    return len(x & y) / max(1, len(x | y))

def compare(row):
    name_sim = similarity(row.party_name, row.watchlist_name)
    dob_match = normalize(row.party_dob) == normalize(row.watchlist_dob)
    nationality_match = normalize(row.party_nationality) == normalize(row.watchlist_nationality)
    gender_match = normalize(row.party_gender) == normalize(row.watchlist_gender)
    country_match = normalize(row.party_country) == normalize(row.watchlist_country)
    name_exact = normalize(row.party_name) == normalize(row.watchlist_name)
    dob_exact = dob_match
    country_exact = country_match

    # Weighted score. Name similarity contributes proportionally.
    raw = (
        WEIGHTS["name"] * name_sim +
        WEIGHTS["dob"] * int(dob_match) +
        WEIGHTS["nationality"] * int(nationality_match) +
        WEIGHTS["gender"] * int(gender_match) +
        WEIGHTS["country"] * int(country_match) +
        WEIGHTS["name_exact"] * int(name_exact) +
        WEIGHTS["dob_exact"] * int(dob_exact) +
        WEIGHTS["country_exact"] * int(country_exact)
    )
    max_score = sum(WEIGHTS.values())
    score = round(raw / max_score * 100, 2)

    if score >= 80:
        band = "HIGH_POTENTIAL_MATCH"
    elif score >= 55:
        band = "REVIEW"
    else:
        band = "LOW_SIMILARITY"

    return {
        "name_similarity": round(name_sim, 3),
        "dob_match": dob_match,
        "nationality_match": nationality_match,
        "gender_match": gender_match,
        "country_match": country_match,
        "name_exact": name_exact,
        "weighted_score": score,
        "risk_band": band,
    }

def build_ai_evidence(row, result):
    # This is deliberately compact: AI gets the evidence, not the raw CSV.
    return {
        "alert_id": row.alert_id,
        "party_id": row.party_id,
        "entry_id": row.entry_id,
        "party": {
            "name": row.party_name,
            "dob": row.party_dob,
            "nationality": row.party_nationality,
            "gender": row.party_gender,
            "country": row.party_country,
        },
        "watchlist": {
            "name": row.watchlist_name,
            "dob": row.watchlist_dob,
            "nationality": row.watchlist_nationality,
            "gender": row.watchlist_gender,
            "country": row.watchlist_country,
        },
        "deterministic_factors": result,
    }

def build_prompt(evidence):
    return f"""
You are an AML screening assessment assistant.

The deterministic screening engine has compared a NICE Actimize party
against one Dow Jones Risk & Compliance watchlist entry.

Do not make the final AML decision. Assess the evidence only.

Evidence:
{json.dumps(evidence, separators=(",", ":"), ensure_ascii=False)}

Return JSON:
{{
  "ai_score": 0-100,
  "recommendation": "LIKELY_FALSE_POSITIVE | POTENTIAL_MATCH | REQUIRES_FURTHER_INVESTIGATION | INSUFFICIENT_EVIDENCE",
  "summary": "concise assessment",
  "evidence": [
    "specific supporting or contradicting factor"
  ],
  "confidence": 0-1
}}

Explain why the score was assigned using only the supplied evidence.
"""

def estimate_output_tokens():
    # Conservative POC assumption. Replace with measured Azure output
    # after running the same prompt against Azure OpenAI.
    sample = """
    {
      "ai_score": 72,
      "recommendation": "POTENTIAL_MATCH",
      "summary": "Name, date of birth and nationality align, while country differs.",
      "evidence": ["Exact DOB match", "High name similarity", "Country mismatch"],
      "confidence": 0.84
    }
    """
    return token_count(sample)

def main():
    df = pd.read_csv(INPUT_CSV)
    hit_results = []

    simulated_output = estimate_output_tokens()

    for _, row in df.iterrows():
        factor = compare(row)
        evidence = build_ai_evidence(row, factor)
        prompt = build_prompt(evidence)

        input_tokens = token_count(prompt)
        output_tokens = simulated_output

        hit_results.append({
            "alert_id": row.alert_id,
            "party_id": row.party_id,
            "entry_id": row.entry_id,
            "weighted_factor_score": factor["weighted_score"],
            "risk_band": factor["risk_band"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        })

    result = pd.DataFrame(hit_results)

    # Alert-level rollup: one alert may have multiple watchlist hits.
    alert_rollup = (
        result.groupby(["alert_id", "party_id"], as_index=False)
        .agg(
            watchlist_hits=("entry_id", "count"),
            total_input_tokens=("input_tokens", "sum"),
            total_output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            max_factor_score=("weighted_factor_score", "max"),
        )
    )

    result.to_csv("aml_hit_token_results.csv", index=False)
    alert_rollup.to_csv("aml_alert_token_results.csv", index=False)

    avg_input = alert_rollup.total_input_tokens.mean()
    avg_output = alert_rollup.total_output_tokens.mean()
    avg_total = alert_rollup.total_tokens.mean()
    avg_hits = alert_rollup.watchlist_hits.mean()

    print("\n=== AML TOKEN POC ===")
    print(f"Alerts tested: {len(alert_rollup)}")
    print(f"Hits tested: {len(result)}")
    print(f"Average hits per alert: {avg_hits:.2f}")
    print(f"Average input tokens / alert: {avg_input:.0f}")
    print(f"Average output tokens / alert: {avg_output:.0f}")
    print(f"Average total tokens / alert: {avg_total:.0f}")
    print(f"Projected total tokens for {MONTHLY_ALERTS:,} alerts/month: "
          f"{avg_total * MONTHLY_ALERTS:,.0f}")

    print("\nHit-level results:")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
