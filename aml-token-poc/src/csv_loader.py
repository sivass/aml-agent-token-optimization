"""Load and validate the raw screening CSV, and group hits into alerts."""
from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

REQUIRED_COLUMNS = [
    "alert_id", "party_id", "entry_id",
    "party_name", "party_dob", "party_nationality", "party_gender", "party_country",
    "watchlist_name", "watchlist_dob", "watchlist_nationality", "watchlist_gender",
    "watchlist_country",
]


def load_hits(path: Path) -> list[dict]:
    """Return one dict per watchlist hit (party_id + entry_id)."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        rows = [{c: (row.get(c) or "").strip() for c in REQUIRED_COLUMNS} for row in reader]

    seen = set()
    for line, row in enumerate(rows, start=2):
        for key in ("alert_id", "party_id", "entry_id"):
            if not row[key]:
                raise ValueError(f"Line {line}: {key} is empty")
        hit_key = (row["alert_id"], row["party_id"], row["entry_id"])
        if hit_key in seen:
            raise ValueError(f"Line {line}: duplicate hit {hit_key}")
        seen.add(hit_key)
    return rows


def group_by_alert(rows: list[dict]) -> "OrderedDict[tuple[str, str], list[dict]]":
    """Group hits by (alert_id, party_id), keeping file order."""
    groups: "OrderedDict[tuple[str, str], list[dict]]" = OrderedDict()
    for row in rows:
        groups.setdefault((row["alert_id"], row["party_id"]), []).append(row)
    return groups
