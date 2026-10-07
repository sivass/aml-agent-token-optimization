"""Evidence object, controlled prompt and the offline simulated AI response."""
from __future__ import annotations

import json

import config

SYSTEM_PROMPT = """You are an AML screening assessment assistant.

Your responsibility is to assess the supplied screening evidence.

The deterministic screening engine has already compared the customer/party against the watchlist entry.

Do not perform basic matching again.
Do not make the final AML compliance decision. A human analyst decides.

Assess the evidence and explain whether the available evidence supports a potential match, likely false positive, or requires further investigation.

Return structured JSON containing exactly these fields:
- ai_score: integer 0-100
- recommendation: one of LIKELY_FALSE_POSITIVE, POTENTIAL_MATCH, REQUIRES_FURTHER_INVESTIGATION, INSUFFICIENT_EVIDENCE
- confidence: number 0-1
- summary: 2-3 sentences
- supporting_evidence: list of short strings
- contradicting_evidence: list of short strings
- requires_further_investigation: boolean

Use only the evidence supplied in the request.
Do not invent information.
Do not expose chain-of-thought or hidden reasoning.
Provide concise, auditable explanations."""


def build_evidence(row: dict, factors: dict, score: int, band: str) -> dict:
    """Compact evidence for one hit: only what the AI needs, nothing else."""
    screening = {k: v for k, v in factors.items() if k != "dob_match"}
    screening["weighted_score"] = score
    screening["risk_band"] = band
    return {
        "alert_id": row["alert_id"],
        "party_id": row["party_id"],
        "entry_id": row["entry_id"],
        "party": {f: row[f"party_{f}"] or None
                  for f in ("name", "dob", "nationality", "gender", "country")},
        "watchlist": {f: row[f"watchlist_{f}"] or None
                      for f in ("name", "dob", "nationality", "gender", "country")},
        "deterministic_screening": screening,
    }


def evidence_to_json(evidence: dict, compact: bool | None = None) -> str:
    compact = config.COMPACT_EVIDENCE_JSON if compact is None else compact
    if compact:
        return json.dumps(evidence, separators=(",", ":"))
    return json.dumps(evidence, indent=2)


def build_messages(evidence: dict, compact: bool | None = None) -> list[dict]:
    """The exact chat messages sent to Azure OpenAI (and counted offline)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Screening evidence:\n" + evidence_to_json(evidence, compact)},
    ]


def simulate_ai_response(evidence: dict) -> dict:
    """Rule-based stand-in for the AI response, used when no Azure key exists.

    Its only job is to be a representative SIZE and SHAPE so output tokens
    can be estimated. It says nothing about real model quality.
    """
    s = evidence["deterministic_screening"]
    score = s["weighted_score"]
    supporting, contradicting = [], []

    if s["exact_name"]:
        supporting.append("Exact name match")
    elif s["alias_name_variation"]:
        supporting.append(f"Name is a recognised variation (similarity {s['name_similarity']})")
    elif s["name_similarity"] >= config.NAME_MATCH_THRESHOLD:
        supporting.append(f"High name similarity ({s['name_similarity']})")
    else:
        contradicting.append(f"Name similarity is only {s['name_similarity']}")

    dob_text = {
        "EXACT": (supporting, "Exact date of birth match"),
        "DAY_MONTH_TRANSPOSED": (supporting, "Date of birth matches if day and month are transposed"),
        "YEAR_ONLY": (supporting, "Year of birth matches; full date not confirmed"),
        "MISMATCH": (contradicting, "Date of birth mismatch"),
    }.get(s["dob_comparison"])
    if dob_text:
        dob_text[0].append(dob_text[1])

    for key, label in (("nationality_match", "Nationality"), ("gender_match", "Gender"),
                       ("country_match", "Country")):
        if s[key] is True:
            supporting.append(f"{label} match")
        elif s[key] is False:
            contradicting.append(f"{label} mismatch")
    if s["country_match"] and s["exact_country"] is False:
        contradicting.append("Same country but different recorded location")

    missing = s["missing_fields"]
    if len(missing) >= 2:
        recommendation = "INSUFFICIENT_EVIDENCE"
    elif score >= 70 and not contradicting:
        recommendation = "POTENTIAL_MATCH"
    elif score >= 40:
        recommendation = "REQUIRES_FURTHER_INVESTIGATION"
    else:
        recommendation = "LIKELY_FALSE_POSITIVE"

    confidence = min(0.95, 0.5 + abs(score - 55) / 100) - 0.1 * len(missing)
    closing = {
        "POTENTIAL_MATCH": "The evidence supports a potential match and should be reviewed by an analyst.",
        "REQUIRES_FURTHER_INVESTIGATION": "The evidence is mixed, so further investigation is needed before a decision.",
        "LIKELY_FALSE_POSITIVE": "The contradicting factors outweigh the name similarity, which points to a likely false positive.",
        "INSUFFICIENT_EVIDENCE": "Key identifiers are missing, so the available evidence is not sufficient to assess the hit.",
    }[recommendation]
    summary = (
        f"The party and watchlist entry have a name similarity of {s['name_similarity']} "
        f"and a deterministic screening score of {score} ({s['risk_band']}). "
        f"{len(supporting)} factor(s) support the match and {len(contradicting)} contradict it"
        + (f", with {', '.join(missing)} unavailable for comparison. " if missing else ". ")
        + closing
    )
    return {
        "ai_score": score,
        "recommendation": recommendation,
        "confidence": round(max(0.3, confidence), 2),
        "summary": summary,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "requires_further_investigation": recommendation != "LIKELY_FALSE_POSITIVE",
    }
