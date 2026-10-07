"""Central configuration for the AML AI token usage POC.

Everything a business user, AML SME or Finance analyst may want to change
lives here or in the .env file. Nothing below is hard-coded elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (no extra dependency). Real env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    return int(float(_env(name, str(default))))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_FILE = BASE_DIR / "data" / "aml_party_watchlist_factors.csv"
OUTPUT_DIR = BASE_DIR / "output"

# --------------------------------------------------------------------------
# Azure OpenAI (optional). Never put credentials in this file.
# --------------------------------------------------------------------------
USE_AZURE_OPENAI = _env_bool("USE_AZURE_OPENAI", False)
AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT")
# Sent as-is when set. Leave blank for deployments that reject a custom
# temperature (some reasoning models do).
AZURE_OPENAI_TEMPERATURE = _env("AZURE_OPENAI_TEMPERATURE", "0")

# --------------------------------------------------------------------------
# Token simulation
# --------------------------------------------------------------------------
TOKENIZER_MODEL = _env("TOKENIZER_MODEL", "gpt-4o")
# simulated = count a representative synthetic AI response per hit
# fixed     = assume EXPECTED_OUTPUT_TOKENS for every hit
OUTPUT_TOKEN_MODE = _env("OUTPUT_TOKEN_MODE", "simulated").lower()
EXPECTED_OUTPUT_TOKENS = _env_int("EXPECTED_OUTPUT_TOKENS", 350)
# Used only if tiktoken cannot load its vocabulary (e.g. blocked network).
# Calibrated against o200k_base on this POC's prompts and responses.
HEURISTIC_CHARS_PER_TOKEN = _env_float("HEURISTIC_CHARS_PER_TOKEN", 4.0)
# Compact JSON (no whitespace) is cheaper than pretty-printed JSON.
COMPACT_EVIDENCE_JSON = _env_bool("COMPACT_EVIDENCE_JSON", True)
# Hits scoring below this are NOT sent to AI (0 = every hit is assessed).
AI_MIN_FACTOR_SCORE = _env_int("AI_MIN_FACTOR_SCORE", 0)

# --------------------------------------------------------------------------
# Pricing (USD per 1M tokens). Replace with your contracted Azure rate.
# Defaults = GPT-4o Global Standard list price at the time of writing.
# --------------------------------------------------------------------------
INPUT_PRICE_PER_1M_TOKENS = _env_float("INPUT_PRICE_PER_1M_TOKENS", 2.50)
OUTPUT_PRICE_PER_1M_TOKENS = _env_float("OUTPUT_PRICE_PER_1M_TOKENS", 10.00)

# --------------------------------------------------------------------------
# Finance projection
# --------------------------------------------------------------------------
MONTHLY_ALERTS = _env_int("MONTHLY_ALERTS", 10000)
PROJECTION_VOLUMES = [1_000, 5_000, 10_000, 50_000, 100_000]

# Share of alerts by number of watchlist hits. Assumptions - replace with the
# real hit distribution from NICE once it is known.
FINANCE_SCENARIOS = {
    "low": {1: 0.60, 2: 0.40},                       # 1-2 hits / alert
    "expected": {1: 0.40, 2: 0.35, 3: 0.25},         # 1-3 hits / alert
    "high": {3: 0.40, 4: 0.30, 5: 0.20, 6: 0.10},    # 3-5+ hits / alert
}

# --------------------------------------------------------------------------
# AML scoring. POC ASSUMPTIONS ONLY.
# AML/business users must validate the factor weights and thresholds
# before production use.
# --------------------------------------------------------------------------
FACTOR_WEIGHTS = {
    "name_similarity": 30,
    "dob_match": 20,
    "nationality_match": 10,
    "gender_match": 5,
    "country_match": 10,
    "exact_name": 10,
    "exact_dob": 5,
    "exact_country": 5,
    "data_consistency": 5,
}

# (minimum score, band) - evaluated top down.
RISK_BANDS = [
    (85, "HIGH_POTENTIAL_MATCH"),
    (70, "POTENTIAL_MATCH"),
    (40, "REVIEW"),
    (0, "LOW_SIMILARITY"),
]

# Credit given to a weighted factor that cannot be compared because a value
# is missing. 0.5 = neutral: missing data neither confirms nor contradicts.
MISSING_FACTOR_CREDIT = 0.5

# Name similarity at or above this counts as "names agree" for consistency.
NAME_MATCH_THRESHOLD = 0.85

# Partial credit for imperfect date-of-birth agreement.
DOB_PARTIAL_CREDIT = {
    "EXACT": 1.0,
    "DAY_MONTH_TRANSPOSED": 0.7,
    "YEAR_ONLY": 0.5,
    "MISMATCH": 0.0,
}

# --------------------------------------------------------------------------
# Decision boundaries: AI recommends, the analyst decides.
# --------------------------------------------------------------------------
AI_RECOMMENDATIONS = [
    "LIKELY_FALSE_POSITIVE",
    "POTENTIAL_MATCH",
    "REQUIRES_FURTHER_INVESTIGATION",
    "INSUFFICIENT_EVIDENCE",
]
ANALYST_DECISIONS = [
    "CONFIRMED_MATCH",
    "FALSE_POSITIVE",
    "ESCALATE",
    "REQUEST_MORE_INFORMATION",
]
PENDING_DECISION = "PENDING_ANALYST_REVIEW"
