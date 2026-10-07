"""Basic tests: matching, scoring, multi-hit aggregation, tokens and cost."""
from types import SimpleNamespace

import pytest

import config
import main
from src import azure_openai, reporting
from src.ai_prompt import build_evidence, build_messages, simulate_ai_response
from src.cost_calculator import average_hits, build_finance_projection, calculate_cost
from src.csv_loader import REQUIRED_COLUMNS, group_by_alert, load_hits
from src.factor_matching import (
    FACTOR_NAMES, calculate_name_similarity, compare_country, compare_dob,
    compare_gender, compare_hit, compare_nationality, is_alias_variation,
    is_exact_country, normalize_name,
)
from src.generate_dataset import build_rows, write_dataset
from src.scoring import calculate_weighted_score, risk_band
from src.token_counter import TokenCounter


def make_row(**overrides):
    row = {
        "alert_id": "ALT-T", "party_id": "PTY-T", "entry_id": "DJ-T",
        "party_name": "AARON LIM", "party_dob": "1982-04-17", "party_nationality": "MY",
        "party_gender": "M", "party_country": "Kuala Lumpur",
        "watchlist_name": "AARON LIM", "watchlist_dob": "1982-04-17",
        "watchlist_nationality": "MY", "watchlist_gender": "M",
        "watchlist_country": "Kuala Lumpur",
    }
    row.update(overrides)
    return row


@pytest.fixture
def counter():
    # Heuristic counter: tests must not depend on a tokenizer download.
    return TokenCounter(allow_tiktoken=False)


# ---- name -----------------------------------------------------------------
def test_exact_name_is_1():
    assert calculate_name_similarity("AARON LIM", "AARON LIM") == 1.0


def test_similar_name_is_high_but_not_exact():
    assert 0.85 <= calculate_name_similarity("AARON LIM", "AARON LAM") < 1.0


def test_unrelated_name_is_low():
    assert calculate_name_similarity("AARON LIM", "SITI NURHALIZA") < 0.5


def test_normalize_name():
    assert normalize_name("  Rajesh  Kumar a/l Subramaniam ") == "RAJESH KUMAR A L SUBRAMANIAM"


def test_alias_variation():
    assert is_alias_variation("MOHD FAIZAL BIN HASSAN", "MUHAMMAD FAIZAL HASSAN")
    assert is_alias_variation("LIM CHEE KEONG", "KEONG LIM CHEE")
    assert not is_alias_variation("AARON LIM", "AARON LIM")   # exact, not alias
    assert not is_alias_variation("AARON LIM", "AARON LAM")


# ---- other factors --------------------------------------------------------
def test_dob():
    assert compare_dob("1982-04-17", "1982-04-17") == "EXACT"
    assert compare_dob("1982-04-17", "1983-04-17") == "MISMATCH"
    assert compare_dob("1963-05-06", "1963-06-05") == "DAY_MONTH_TRANSPOSED"
    assert compare_dob("1984-09-13", "1984") == "YEAR_ONLY"
    assert compare_dob("1984-09-13", "") == "MISSING"


def test_nationality():
    assert compare_nationality("MY", "my") is True
    assert compare_nationality("MY", "Malaysian") is True
    assert compare_nationality("MY", "SG") is False
    assert compare_nationality("MY", "") is None


def test_gender():
    assert compare_gender("M", "Male") is True
    assert compare_gender("M", "F") is False
    assert compare_gender("M", "") is None


def test_country():
    assert compare_country("Kuala Lumpur", "Penang") is True      # both Malaysia
    assert is_exact_country("Kuala Lumpur", "Penang") is False
    assert compare_country("Bandar Seri Begawan", "Jakarta") is False
    assert compare_country("Kuching", "") is None


def test_twelve_factors_per_hit():
    factors = compare_hit(make_row())
    assert len(FACTOR_NAMES) == 12
    assert all(name in factors for name in FACTOR_NAMES)


# ---- scoring --------------------------------------------------------------
def test_weights_total_100():
    assert sum(config.FACTOR_WEIGHTS.values()) == 100


def test_strong_match_scores_100():
    score = calculate_weighted_score(compare_hit(make_row()))
    assert score == 100
    assert risk_band(score) == "HIGH_POTENTIAL_MATCH"


def test_dob_mismatch_lowers_score():
    full = calculate_weighted_score(compare_hit(make_row()))
    mismatch = calculate_weighted_score(compare_hit(make_row(watchlist_dob="1961-01-01")))
    assert mismatch < full
    # 20 (dob) + 5 (exact dob) + 1 (consistency 4/5) points lost
    assert mismatch == 74


def test_weights_are_configurable():
    factors = compare_hit(make_row(watchlist_dob="1961-01-01"))
    assert calculate_weighted_score(factors, weights={"name_similarity": 1}) == 100
    assert calculate_weighted_score(factors, weights={"dob_match": 1}) == 0


def test_missing_factor_gets_neutral_credit():
    factors = compare_hit(make_row(watchlist_dob=""))
    assert factors["missing_fields"] == ["dob"]
    assert calculate_weighted_score(factors, weights={"dob_match": 1}) == 50


def test_risk_bands():
    assert [risk_band(s) for s in (0, 39, 40, 69, 70, 84, 85, 100)] == [
        "LOW_SIMILARITY", "LOW_SIMILARITY", "REVIEW", "REVIEW",
        "POTENTIAL_MATCH", "POTENTIAL_MATCH", "HIGH_POTENTIAL_MATCH", "HIGH_POTENTIAL_MATCH",
    ]


# ---- dataset --------------------------------------------------------------
def test_dataset_is_raw_factors_only(tmp_path):
    path = tmp_path / "data.csv"
    write_dataset(path)
    rows = load_hits(path)
    assert list(rows[0].keys()) == REQUIRED_COLUMNS
    assert len({r["entry_id"] for r in rows}) == len(rows)       # unique entry_id
    hit_counts = {len(g) for g in group_by_alert(rows).values()}
    assert {1, 2, 3} <= hit_counts and max(hit_counts) >= 4
    assert 10 <= len(group_by_alert(rows)) <= 20 and 20 <= len(rows) <= 40


def test_loader_rejects_duplicate_hit(tmp_path):
    path = tmp_path / "dup.csv"
    row = make_row()
    path.write_text(",".join(REQUIRED_COLUMNS) + "\n"
                    + (",".join(row[c] for c in REQUIRED_COLUMNS) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate hit"):
        load_hits(path)


# ---- AI prompt and decision boundary --------------------------------------
def test_simulated_response_shape():
    row = make_row()
    factors = compare_hit(row)
    response = simulate_ai_response(build_evidence(row, factors, 100, "HIGH_POTENTIAL_MATCH"))
    assert set(response) == {
        "ai_score", "recommendation", "confidence", "summary",
        "supporting_evidence", "contradicting_evidence", "requires_further_investigation",
    }
    assert response["recommendation"] in config.AI_RECOMMENDATIONS


def test_ai_never_makes_the_final_decision(counter):
    assert not set(config.AI_RECOMMENDATIONS) & set(config.ANALYST_DECISIONS)
    hit = main.process_hit(make_row(), counter)
    assert hit["final_decision"] == config.PENDING_DECISION


def test_compact_evidence_uses_fewer_tokens(counter):
    row = make_row()
    evidence = build_evidence(row, compare_hit(row), 100, "HIGH_POTENTIAL_MATCH")
    assert (counter.count_messages(build_messages(evidence, compact=True))
            < counter.count_messages(build_messages(evidence, compact=False)))


# ---- multiple hits --------------------------------------------------------
def test_three_hits_give_three_calculations_and_one_alert(counter):
    rows = [make_row(entry_id=f"DJ-{i}", watchlist_name=name)
            for i, name in enumerate(["AARON LIM", "AARON LAM", "ARUN LIM"], start=1)]
    hits = [main.process_hit(row, counter) for row in rows]
    assert len(hits) == 3
    alerts = reporting.aggregate_alerts(hits)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["watchlist_hits"] == 3
    assert alert["total_input_tokens"] == sum(h["input_tokens"] for h in hits)
    assert alert["total_output_tokens"] == sum(h["output_tokens"] for h in hits)
    assert alert["total_tokens"] == sum(h["total_tokens"] for h in hits)
    assert alert["max_factor_score"] == 100


def test_spec_aggregation_example():
    hits = [
        {"alert_id": "ALT-0001", "party_id": "PTY-0001", "entry_id": e,
         "input_tokens": i, "output_tokens": o, "total_tokens": i + o,
         "estimated_cost": 0.0, "weighted_factor_score": 50, "sent_to_ai": True}
        for e, i, o in [("DJ-000101", 850, 320), ("DJ-000102", 790, 300), ("DJ-000103", 700, 280)]
    ]
    assert reporting.aggregate_alerts(hits)[0]["total_tokens"] == 3240


def test_score_threshold_skips_low_hits(counter, monkeypatch):
    monkeypatch.setattr(config, "AI_MIN_FACTOR_SCORE", 40)
    weak = make_row(watchlist_name="RAJEEV KUMAR", watchlist_dob="1971-01-19",
                    watchlist_nationality="LK", watchlist_country="Colombo")
    hit = main.process_hit(weak, counter)
    assert hit["sent_to_ai"] is False and hit["total_tokens"] == 0


def test_fixed_output_mode(counter, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_TOKEN_MODE", "fixed")
    monkeypatch.setattr(config, "EXPECTED_OUTPUT_TOKENS", 350)
    assert main.process_hit(make_row(), counter)["output_tokens"] == 350


# ---- cost -----------------------------------------------------------------
def test_cost_formula():
    cost = calculate_cost(1_000_000, 500_000, input_price=2.5, output_price=10)
    assert cost == {"input_cost": 2.5, "output_cost": 5.0, "total_cost": 7.5}


def test_finance_projection_scales_with_hits_and_volume():
    rows = build_finance_projection(
        800, 200, observed_avg_hits=2.0,
        scenarios={"low": {1: 0.5, 2: 0.5}, "high": {4: 1.0}},
        volumes=[1_000, 10_000], input_price=2.5, output_price=10,
    )
    by_key = {(r["scenario"], r["monthly_alerts"]): r for r in rows}
    assert average_hits({1: 0.5, 2: 0.5}) == 1.5
    assert by_key[("high", 1_000)]["average_tokens_per_alert"] == 4000
    assert by_key[("high", 1_000)]["monthly_total_tokens"] == 4_000_000
    # 3.2M input x 2.5 + 0.8M output x 10 = 8 + 8
    assert by_key[("high", 1_000)]["estimated_monthly_cost"] == 16.0
    assert (by_key[("high", 10_000)]["estimated_monthly_cost"]
            == 10 * by_key[("high", 1_000)]["estimated_monthly_cost"])
    assert (by_key[("high", 1_000)]["monthly_total_tokens"]
            > by_key[("low", 1_000)]["monthly_total_tokens"])


# ---- end to end -----------------------------------------------------------
def test_offline_run_needs_no_credentials(tmp_path, counter):
    data = tmp_path / "data.csv"
    write_dataset(data)
    result = main.run(data, tmp_path / "out", counter=counter)
    assert len(result["hits"]) == len(build_rows())
    assert all(h["token_source"] == "offline_estimate" for h in result["hits"])
    for name in ("aml_hit_token_results.csv", "aml_alert_token_results.csv",
                 "aml_finance_projection.csv"):
        assert (tmp_path / "out" / name).exists()
    total = sum(a["total_tokens"] for a in result["alerts"])
    assert total == sum(h["total_tokens"] for h in result["hits"])


def test_azure_mode_records_actual_usage(tmp_path, counter):
    """Azure mode with a stand-in client: actual usage replaces the estimate."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"ai_score": 90, "recommendation": "POTENTIAL_MATCH"}'))],
            usage=SimpleNamespace(prompt_tokens=777, completion_tokens=111, total_tokens=888),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    hit = main.process_hit(make_row(), counter, azure_client=client)
    assert (hit["input_tokens"], hit["output_tokens"], hit["total_tokens"]) == (777, 111, 888)
    assert hit["token_source"] == "azure_actual"
    assert hit["simulated_input_tokens"] > 0            # kept for comparison
    assert calls[0]["messages"] == hit["messages"]      # same prompt as counted
    assert hit["final_decision"] == config.PENDING_DECISION


def test_azure_mode_requires_settings(monkeypatch):
    monkeypatch.setattr(config, "AZURE_OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
        azure_openai.create_client()
