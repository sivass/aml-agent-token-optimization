"""Deterministic factor comparison: NICE party vs Dow Jones watchlist entry.

No AI is involved here. Every function is a plain, explainable rule so an
analyst or auditor can reproduce the result by hand.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import config

# Illustrative only - a production list would come from the AML team.
NAME_VARIANTS = {
    "MOHD": "MUHAMMAD", "MOHAMED": "MUHAMMAD", "MOHAMMAD": "MUHAMMAD",
    "MOHAMAD": "MUHAMMAD", "MUHAMAD": "MUHAMMAD", "MD": "MUHAMMAD",
    "ABD": "ABDUL", "BT": "BINTI", "BTE": "BINTI", "B": "BIN",
}
# Connectors that carry no identity information on their own.
NAME_CONNECTORS = {"BIN", "BINTI", "A", "L", "P", "AL", "AP", "HAJI", "HJ"}

NATIONALITY_CODES = {
    "MALAYSIA": "MY", "MALAYSIAN": "MY", "SINGAPORE": "SG", "SINGAPOREAN": "SG",
    "BRUNEI": "BN", "BRUNEIAN": "BN", "INDONESIA": "ID", "INDONESIAN": "ID",
    "INDIA": "IN", "INDIAN": "IN", "CHINA": "CN", "CHINESE": "CN",
}

# Free-text location -> ISO country, so a city can be compared to a country.
LOCATION_COUNTRY = {
    "MY": "MY", "MALAYSIA": "MY", "KUALA LUMPUR": "MY", "SELANGOR": "MY",
    "SHAH ALAM": "MY", "KLANG": "MY", "PENANG": "MY", "GEORGE TOWN": "MY",
    "JOHOR BAHRU": "MY", "IPOH": "MY", "KUCHING": "MY", "KOTA KINABALU": "MY",
    "KOTA BHARU": "MY", "KUALA TERENGGANU": "MY",
    "SG": "SG", "SINGAPORE": "SG",
    "BN": "BN", "BRUNEI": "BN", "BANDAR SERI BEGAWAN": "BN",
    "ID": "ID", "INDONESIA": "ID", "JAKARTA": "ID",
    "IN": "IN", "INDIA": "IN", "CHENNAI": "IN", "MUMBAI": "IN",
    "LK": "LK", "SRI LANKA": "LK", "COLOMBO": "LK",
    "CN": "CN", "CHINA": "CN", "SHANGHAI": "CN",
    "HK": "HK", "HONG KONG": "HK",
    "TW": "TW", "TAIWAN": "TW", "TAIPEI": "TW",
    "SA": "SA", "SAUDI ARABIA": "SA", "RIYADH": "SA",
}

FACTOR_NAMES = [
    "name_similarity", "exact_name", "alias_name_variation",
    "dob_match", "exact_dob",
    "nationality_match", "gender_match",
    "country_match", "exact_country", "location_similarity",
    "missing_fields", "data_consistency",
]


def _clean(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]", " ", text)).strip().upper()


def normalize_name(name: str | None) -> str:
    """Upper-case, strip accents and punctuation, collapse whitespace."""
    return _clean(name)


def _canonical_tokens(name: str) -> list[str]:
    return [NAME_VARIANTS.get(tok, tok) for tok in normalize_name(name).split()]


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def calculate_name_similarity(party_name: str, watchlist_name: str) -> float:
    """0.0-1.0. Best of: as written, word order ignored, known variants mapped."""
    a, b = normalize_name(party_name), normalize_name(watchlist_name)
    if not a or not b:
        return 0.0
    ca, cb = _canonical_tokens(party_name), _canonical_tokens(watchlist_name)
    best = max(
        _ratio(a, b),
        _ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split()))),
        _ratio(" ".join(sorted(ca)), " ".join(sorted(cb))),
    )
    return round(best, 2)


def is_exact_name(party_name: str, watchlist_name: str) -> bool:
    a = normalize_name(party_name)
    return bool(a) and a == normalize_name(watchlist_name)


def is_alias_variation(party_name: str, watchlist_name: str) -> bool:
    """Same person-name written differently: reordered, shortened or a known
    spelling variant (e.g. MOHD / MUHAMMAD). Exact names are not aliases."""
    if is_exact_name(party_name, watchlist_name):
        return False
    a = set(_canonical_tokens(party_name)) - NAME_CONNECTORS
    b = set(_canonical_tokens(watchlist_name)) - NAME_CONNECTORS
    if not a or not b:
        return False
    smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
    return len(smaller) >= 2 and smaller <= larger


def _parse_dob(value: str | None):
    """Return (year, month, day); month/day are None for a year-only DOB."""
    text = (value or "").strip()
    if re.fullmatch(r"\d{4}", text):
        return int(text), None, None
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return tuple(int(part) for part in match.groups())
    return None


def compare_dob(party_dob: str, watchlist_dob: str) -> str:
    """EXACT | DAY_MONTH_TRANSPOSED | YEAR_ONLY | MISMATCH | MISSING."""
    a, b = _parse_dob(party_dob), _parse_dob(watchlist_dob)
    if a is None or b is None:
        return "MISSING"
    if a[0] != b[0]:
        return "MISMATCH"
    if a[1] is None or b[1] is None:
        return "YEAR_ONLY"
    if a == b:
        return "EXACT"
    if a[1] == b[2] and a[2] == b[1]:
        return "DAY_MONTH_TRANSPOSED"
    return "MISMATCH"


def _normalize_nationality(value: str | None) -> str:
    text = _clean(value)
    return NATIONALITY_CODES.get(text, text)


def compare_nationality(party: str, watchlist: str) -> bool | None:
    a, b = _normalize_nationality(party), _normalize_nationality(watchlist)
    return None if not a or not b else a == b


def _normalize_gender(value: str | None) -> str:
    text = _clean(value)
    return {"MALE": "M", "FEMALE": "F"}.get(text, text if text in {"M", "F"} else "")


def compare_gender(party: str, watchlist: str) -> bool | None:
    a, b = _normalize_gender(party), _normalize_gender(watchlist)
    return None if not a or not b else a == b


def resolve_country(location: str | None) -> str:
    text = _clean(location)
    return LOCATION_COUNTRY.get(text, text)


def compare_country(party: str, watchlist: str) -> bool | None:
    """True when both locations resolve to the same country."""
    a, b = resolve_country(party), resolve_country(watchlist)
    return None if not a or not b else a == b


def is_exact_country(party: str, watchlist: str) -> bool | None:
    """True only when the location text itself is identical."""
    a, b = _clean(party), _clean(watchlist)
    return None if not a or not b else a == b


def location_similarity(party: str, watchlist: str) -> float | None:
    a, b = _clean(party), _clean(watchlist)
    return None if not a or not b else round(_ratio(a, b), 2)


def compare_hit(row: dict) -> dict:
    """Run every factor for one party_id + entry_id pair."""
    name_similarity = calculate_name_similarity(row["party_name"], row["watchlist_name"])
    dob_comparison = compare_dob(row["party_dob"], row["watchlist_dob"])
    dob_match = config.DOB_PARTIAL_CREDIT.get(dob_comparison)  # None when MISSING
    nationality = compare_nationality(row["party_nationality"], row["watchlist_nationality"])
    gender = compare_gender(row["party_gender"], row["watchlist_gender"])
    country = compare_country(row["party_country"], row["watchlist_country"])

    missing_fields = [
        field for field in ("dob", "nationality", "gender", "country")
        if not row[f"party_{field}"] or not row[f"watchlist_{field}"]
    ]

    # Overall identity consistency: of the fields that can be compared,
    # how many agree?
    checks = [name_similarity >= config.NAME_MATCH_THRESHOLD]
    if dob_match is not None:
        checks.append(dob_match >= 0.5)
    checks += [value for value in (nationality, gender, country) if value is not None]
    data_consistency = round(sum(checks) / len(checks), 2)

    return {
        "name_similarity": name_similarity,
        "exact_name": is_exact_name(row["party_name"], row["watchlist_name"]),
        "alias_name_variation": is_alias_variation(row["party_name"], row["watchlist_name"]),
        "dob_match": dob_match,
        "dob_comparison": dob_comparison,
        "exact_dob": None if dob_comparison == "MISSING" else dob_comparison == "EXACT",
        "nationality_match": nationality,
        "gender_match": gender,
        "country_match": country,
        "exact_country": is_exact_country(row["party_country"], row["watchlist_country"]),
        "location_similarity": location_similarity(row["party_country"], row["watchlist_country"]),
        "missing_fields": missing_fields,
        "data_consistency": data_consistency,
    }
