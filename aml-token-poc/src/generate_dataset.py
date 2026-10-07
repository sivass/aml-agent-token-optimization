"""Synthetic NICE party vs Dow Jones watchlist dataset.

All people here are invented. The rows are written by hand (not random) so
every scenario in the brief is covered and each run is reproducible.

The CSV holds raw factors only: no scores, no AI output, no token counts.
"""
from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = [
    "alert_id", "party_id", "entry_id",
    "party_name", "party_dob", "party_nationality", "party_gender", "party_country",
    "watchlist_name", "watchlist_dob", "watchlist_nationality", "watchlist_gender",
    "watchlist_country",
]

# (name, dob, nationality, gender, country/location)
ALERTS = [
    # ALT-0001 - Scenario A: strong match, plus a near-identical second hit
    (("AARON LIM", "1982-04-17", "MY", "M", "Kuala Lumpur"), [
        ("AARON LIM", "1982-04-17", "MY", "M", "Kuala Lumpur"),
        ("AARON LAM", "1982-04-17", "MY", "M", "Kuala Lumpur"),
    ]),
    # ALT-0002 - Scenario B: similar name, DOB mismatch, nationality match
    (("SITI AMINAH BINTI YUSOF", "1975-11-02", "MY", "F", "Johor Bahru"), [
        ("SITI AMINAH BINTI YUSUF", "1979-03-21", "MY", "F", "Johor Bahru"),
    ]),
    # ALT-0003 - Scenario C: exact name, different DOB
    (("TAN WEI MING", "1990-06-25", "SG", "M", "Singapore"), [
        ("TAN WEI MING", "1968-01-09", "SG", "M", "Singapore"),
    ]),
    # ALT-0004 - Scenario D: alias-like name, one hit with a year-only DOB
    (("MOHD FAIZAL BIN HASSAN", "1984-09-13", "MY", "M", "Kuala Lumpur"), [
        ("MUHAMMAD FAIZAL HASSAN", "1984-09-13", "MY", "M", "Kuala Lumpur"),
        ("FAIZAL HASSAN", "1984", "MY", "M", "Selangor"),
    ]),
    # ALT-0005 - Scenario E: different nationality
    (("DAVID WONG KOK LEONG", "1979-02-28", "MY", "M", "Penang"), [
        ("DAVID WONG KOK LEONG", "1979-02-28", "CN", "M", "Penang"),
    ]),
    # ALT-0006 - Scenario F: different country
    (("NURUL HUDA BINTI ISMAIL", "1992-12-05", "BN", "F", "Bandar Seri Begawan"), [
        ("NURUL HUDA BINTI ISMAIL", "1992-12-05", "BN", "F", "Jakarta"),
    ]),
    # ALT-0007 - Scenario G: four hits, only one strong
    (("RAJESH KUMAR A/L SUBRAMANIAM", "1971-07-30", "MY", "M", "Kuala Lumpur"), [
        ("RAJESH KUMAR A/L SUBRAMANIAM", "1971-07-30", "MY", "M", "Kuala Lumpur"),
        ("RAJESH KUMAR", "1985-03-14", "IN", "M", "Chennai"),
        ("RAJESH KUMAR SUBRAMANIAN", "1966-10-02", "IN", "M", "Mumbai"),
        ("RAJEEV KUMAR", "1971-01-19", "LK", "M", "Colombo"),
    ]),
    # ALT-0008 - Scenario H: three hits, all weak
    (("LEE MEI LING", "1988-03-08", "SG", "F", "Singapore"), [
        ("LI MEI LIN", "1961-05-20", "CN", "F", "Shanghai"),
        ("LEE MEI LENG", "1975-09-11", "MY", "F", "Ipoh"),
        ("LEE MING LI", "1990-12-30", "TW", "M", "Taipei"),
    ]),
    # ALT-0009 - Scenario I: missing factors on the watchlist side
    (("AHMAD ZAKI BIN OSMAN", "1969-01-22", "MY", "M", "Kuching"), [
        ("AHMAD ZAKI BIN OSMAN", "", "", "M", "Kuching"),
        ("AHMAD ZAKY OSMAN", "1969-01-22", "MY", "", ""),
    ]),
    # ALT-0010 - Scenario J: strong name match, multiple contradicting factors
    (("CHONG KAH YEE", "1995-08-19", "MY", "F", "Kuala Lumpur"), [
        ("CHONG KAH YEE", "1958-02-03", "CN", "M", "Hong Kong"),
        ("CHONG KA YEE", "1995-08-19", "MY", "F", "Kuala Lumpur"),
        ("CHONG KAR YEE", "1983-11-27", "MY", "F", "Kota Kinabalu"),
    ]),
    # ALT-0011 - Common name, five hits (4+), incl. transposed and year-only DOB
    (("ABDUL RAHMAN BIN HAJI OMAR", "1963-05-06", "BN", "M", "Bandar Seri Begawan"), [
        ("ABDUL RAHMAN BIN OMAR", "1963-06-05", "BN", "M", "Bandar Seri Begawan"),
        ("ABD RAHMAN OMAR", "1963", "MY", "M", "Kota Bharu"),
        ("ABDUL RAHMAN BIN HAJI OMAR", "1950-12-12", "ID", "M", "Jakarta"),
        ("ABDUL RAHIM BIN OMAR", "1971-08-08", "MY", "M", "Kuala Terengganu"),
        ("ABDULRAHMAN OMAR", "1980-01-01", "SA", "M", "Riyadh"),
    ]),
    # ALT-0012 - Scenario A: strong match, single hit
    (("GRACE TEO HUI MIN", "1986-10-14", "SG", "F", "Singapore"), [
        ("GRACE TEO HUI MIN", "1986-10-14", "SG", "F", "Singapore"),
    ]),
    # ALT-0013 - Alias-like strong hit plus one weak hit
    (("KAVITHA A/P MUNIANDY", "1993-04-01", "MY", "F", "Klang"), [
        ("KAVITHA MUNIANDY", "1993-04-01", "MY", "F", "Klang"),
        ("KAVITA MUNIANDI", "1987-07-23", "IN", "F", "Chennai"),
    ]),
    # ALT-0014 - Same country, different city; reordered name
    (("LIM CHEE KEONG", "1977-12-21", "MY", "M", "Penang"), [
        ("LIM CHEE KEONG", "1977-12-21", "MY", "M", "George Town"),
        ("LIM CHEE KIONG", "1977-12-21", "SG", "M", "Singapore"),
        ("KEONG LIM CHEE", "1978-12-21", "MY", "M", "Penang"),
    ]),
    # ALT-0015 - Scenario D: alias-like strong match, single hit
    (("MUHAMMAD HAFIZ BIN ROSLAN", "1998-02-11", "MY", "M", "Shah Alam"), [
        ("MOHD HAFIZ ROSLAN", "1998-02-11", "MY", "M", "Shah Alam"),
    ]),
]


def build_rows() -> list[dict]:
    rows, entry_number = [], 101
    for index, (party, hits) in enumerate(ALERTS, start=1):
        for hit in hits:
            values = [f"ALT-{index:04d}", f"PTY-{index:04d}", f"DJ-{entry_number:06d}",
                      *party, *hit]
            rows.append(dict(zip(COLUMNS, values)))
            entry_number += 1
    return rows


def write_dataset(path: Path) -> int:
    rows = build_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
