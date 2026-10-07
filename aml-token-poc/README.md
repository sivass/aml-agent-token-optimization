# AML Screening AI Token Usage & Assessment — POC

> This POC uses synthetic data and illustrative AML scoring weights. It is intended to estimate AI token consumption and demonstrate the screening/assessment workflow. It is not a production AML decision engine.

## 1. Business problem

An insurer receives AML screening alerts from NICE Actimize. Each alert is a customer (party) that may match one or more Dow Jones Risk & Compliance watchlist entries. Before putting an AI assessment step into that workflow, Finance needs to know what it will cost: how many tokens one alert consumes, how that grows with the number of watchlist hits, and what the monthly Azure OpenAI bill looks like at different alert volumes.

## 2. POC objective

Measure, on a small synthetic dataset, the tokens and cost of an AI assessment per hit and per alert, and project that to monthly volumes. It runs offline with no API key, and can optionally call Azure OpenAI to check the offline estimate against real usage.

## 3. Architecture

One Python process, no infrastructure.

```text
data/aml_party_watchlist_factors.csv
        |
  csv_loader        load + validate, one row = one hit
        |
  factor_matching   12 deterministic factors            } Python only,
  scoring           weighted score 0-100 + risk band    } no AI
        |
  ai_prompt         compact evidence JSON + controlled prompt
        |
  token_counter     offline estimate      azure_openai   actual usage (optional)
        |
  cost_calculator   cost per hit / alert, monthly projection
  reporting         CSV reports + console summary
```

Three responsibilities are kept apart:

| Step | Who | Does |
|---|---|---|
| Deterministic screening | Python | normalisation, similarity, factor matching, weighted score |
| AI assessment | Azure OpenAI | interprets the evidence, explains, flags contradictions, recommends |
| Decision | Analyst | `CONFIRMED_MATCH`, `FALSE_POSITIVE`, `ESCALATE`, `REQUEST_MORE_INFORMATION` |

The AI never makes the final AML decision. Every hit leaves the POC as `PENDING_ANALYST_REVIEW`.

## 4. CSV structure

One file, raw factors only. No scores, AI output or token counts.

```text
alert_id, party_id, entry_id,
party_name, party_dob, party_nationality, party_gender, party_country,
watchlist_name, watchlist_dob, watchlist_nationality, watchlist_gender, watchlist_country
```

`*_country` is free text (a city or a country). `*_dob` is `YYYY-MM-DD`, or `YYYY` when only the year is known. Blank means unknown.

The synthetic file has 15 alerts and 32 hits and covers every scenario in the brief:

| Alert | Hits | Scenario |
|---|---|---|
| ALT-0001 | 2 | A: strong match, plus a near-identical name |
| ALT-0002 | 1 | B: similar name, DOB mismatch |
| ALT-0003 | 1 | C: exact name, different DOB |
| ALT-0004 | 2 | D: alias-like name; year-only DOB |
| ALT-0005 | 1 | E: different nationality |
| ALT-0006 | 1 | F: different country |
| ALT-0007 | 4 | G: only one of four hits is strong |
| ALT-0008 | 3 | H: all hits weak |
| ALT-0009 | 2 | I: missing factors |
| ALT-0010 | 3 | J: exact name, several contradicting factors |
| ALT-0011 | 5 | common name, 4+ hits, transposed DOB |
| ALT-0012 | 1 | A: strong match |
| ALT-0013 | 2 | alias-like strong hit plus a weak hit |
| ALT-0014 | 3 | same country, different city; reordered name |
| ALT-0015 | 1 | D: alias-like strong match |

## 5. NICE party vs Dow Jones entry

```text
NICE alert (alert_id)
  └─ party (party_id)
       ├─ Dow Jones entry (entry_id)   ← one hit
       ├─ Dow Jones entry (entry_id)   ← one hit
       └─ ...
```

A hit is `party_id + entry_id`. One alert is not one AI request.

## 6. Multiple-hit handling

Each hit is screened and assessed on its own. Alert usage is the sum of its hits:

```text
ALT-0001 = DJ-000101 + DJ-000102 + ...
```

Because each hit carries the full prompt, alert cost grows linearly with hits (see section 17).

## 7. Deterministic factor comparison

Twelve factors per hit, all plain Python rules in `src/factor_matching.py`:

| # | Factor | Result |
|---|---|---|
| 1 | `name_similarity` | 0–1, best of: as written, word order ignored, known variants mapped (`difflib`) |
| 2 | `exact_name` | true / false after normalisation |
| 3 | `alias_name_variation` | reordered, shortened or known spelling variant (MOHD / MUHAMMAD) |
| 4 | `dob_match` | `EXACT` 1.0, `DAY_MONTH_TRANSPOSED` 0.7, `YEAR_ONLY` 0.5, `MISMATCH` 0 |
| 5 | `exact_dob` | true / false |
| 6 | `nationality_match` | true / false |
| 7 | `gender_match` | true / false |
| 8 | `country_match` | both locations resolve to the same country |
| 9 | `exact_country` | location text is identical |
| 10 | `location_similarity` | 0–1 text similarity of the two locations |
| 11 | `missing_fields` | which fields could not be compared |
| 12 | `data_consistency` | share of comparable identity fields that agree |

A factor is `null` when either side is blank.

## 8. AML weighted scoring

Weights live in `config.py` (`FACTOR_WEIGHTS`) and are normalised, so they can be changed freely:

| Factor | Weight |
|---|---|
| name_similarity | 30 |
| dob_match | 20 |
| nationality_match | 10 |
| country_match | 10 |
| exact_name | 10 |
| gender_match | 5 |
| exact_dob | 5 |
| exact_country | 5 |
| data_consistency | 5 |

A missing factor gets neutral credit (`MISSING_FACTOR_CREDIT = 0.5`): missing data neither confirms nor contradicts.

| Score | Band |
|---|---|
| 0–39 | `LOW_SIMILARITY` |
| 40–69 | `REVIEW` |
| 70–84 | `POTENTIAL_MATCH` |
| 85–100 | `HIGH_POTENTIAL_MATCH` |

**AML/business users must validate the factor weights and thresholds before production use.** The same applies to the DOB partial credit, the missing-data credit, and the small name-variant and city-to-country tables, which are illustrative.

## 9. AI assessment

Only the compact evidence for one hit is sent: IDs, the five party factors, the five watchlist factors, the factor results, the score and the band. The prompt (`src/ai_prompt.py`) tells the model not to redo matching, not to decide, to use only the supplied evidence, and to return JSON:

```json
{
  "ai_score": 87,
  "recommendation": "POTENTIAL_MATCH",
  "confidence": 0.82,
  "summary": "...",
  "supporting_evidence": ["High name similarity (0.89)", "Exact date of birth match"],
  "contradicting_evidence": [],
  "requires_further_investigation": true
}
```

Allowed recommendations: `LIKELY_FALSE_POSITIVE`, `POTENTIAL_MATCH`, `REQUIRES_FURTHER_INVESTIGATION`, `INSUFFICIENT_EVIDENCE`. The AI score is an assessment score, not a decision.

## 10. Token measurement

Per hit: input tokens (system prompt + evidence + chat framing), output tokens, total, cost. Per alert: the sums, average per hit, highest factor score.

About half of the input is the fixed system prompt (212 of ~419 tokens), repeated for every hit.

## 11. Offline mode (default)

`USE_AZURE_OPENAI=false`. No key needed.

- Input tokens: the real prompt is counted with `tiktoken` for `TOKENIZER_MODEL`.
- Output tokens: `OUTPUT_TOKEN_MODE=simulated` counts a rule-based stand-in response of realistic shape; `OUTPUT_TOKEN_MODE=fixed` assumes `EXPECTED_OUTPUT_TOKENS` per hit.

> Offline tokenization is an estimate. Exact production token usage must be validated using the actual Azure OpenAI deployment.

`tiktoken` downloads its vocabulary the first time it runs. If the network blocks that download, the POC falls back to a characters-per-token estimate and prints `Tokenizer: heuristic` so you know. On this dataset the fallback lands within about 2% of the real tokenizer. To use the real tokenizer on a restricted machine, copy a populated tiktoken cache folder across and set `TIKTOKEN_CACHE_DIR`.

## 12. Azure OpenAI mode (optional)

Set `USE_AZURE_OPENAI=true` and the four `AZURE_OPENAI_*` variables. The POC then sends the same messages it counted offline, records `prompt_tokens`, `completion_tokens` and `total_tokens` from the API, stores the real responses in `output/aml_hit_details.jsonl`, and writes `output/aml_token_validation.csv` with the offline estimate next to the actual usage for every hit. Reports and costs use the actual figures.

A full run makes one call per hit (32 calls on the sample data). Credentials are read from the environment only.

## 13. Cost calculation

```text
input_cost  = input_tokens  / 1,000,000 × INPUT_PRICE_PER_1M_TOKENS
output_cost = output_tokens / 1,000,000 × OUTPUT_PRICE_PER_1M_TOKENS
total_cost  = input_cost + output_cost
```

Defaults are USD 2.50 input and USD 10.00 output per 1M tokens, the GPT-4o Global Standard list price at the time of writing. Regional and data-zone deployments are listed about 10% higher. Replace both with the contracted rate for the deployment you will actually use.

## 14. Finance scenarios

`tokens per alert = average hits per alert × average tokens per hit`

| Scenario | Hit mix (share of alerts) | Avg hits / alert |
|---|---|---|
| low | 1 hit 60%, 2 hits 40% | 1.40 |
| expected | 1 hit 40%, 2 hits 35%, 3 hits 25% | 1.85 |
| high | 3 hits 40%, 4 hits 30%, 5 hits 20%, 6 hits 10% | 4.00 |
| observed_sample | the synthetic dataset | 2.13 |

The mixes are assumptions in `config.FINANCE_SCENARIOS`. Replace them with the real hit distribution from NICE.

## 15. How to run

```bash
pip install -r requirements.txt
python main.py                     # or: python aml_token_poc.py
python main.py --regenerate-data   # rewrite the synthetic CSV first
python main.py --input my.csv --output-dir out
python -m pytest                   # 30 tests
```

Python 3.10+.

## 16. Environment variables

Copy `.env.example` to `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `USE_AZURE_OPENAI` | `false` | `true` calls Azure OpenAI and records actual usage |
| `AZURE_OPENAI_ENDPOINT` / `_API_KEY` / `_API_VERSION` / `_DEPLOYMENT` | – | Azure mode only |
| `AZURE_OPENAI_TEMPERATURE` | `0` | blank = do not send |
| `TOKENIZER_MODEL` | `gpt-4o` | tokenizer for the offline estimate |
| `OUTPUT_TOKEN_MODE` | `simulated` | `simulated` or `fixed` |
| `EXPECTED_OUTPUT_TOKENS` | `350` | output tokens per hit in `fixed` mode |
| `INPUT_PRICE_PER_1M_TOKENS` | `2.50` | USD |
| `OUTPUT_PRICE_PER_1M_TOKENS` | `10.00` | USD |
| `MONTHLY_ALERTS` | `10000` | volume used for the scenario table |
| `AI_MIN_FACTOR_SCORE` | `0` | hits scoring below this are not sent to AI |
| `COMPACT_EVIDENCE_JSON` | `true` | `false` sends pretty-printed JSON (more tokens) |

## 17. Sample output

Offline run, `tiktoken` (o200k_base), simulated output, default prices:

```text
Alerts analysed:              15
Watchlist hits:               32
Average hits / alert:         2.13
Factors evaluated / hit:      12

Average input tokens / hit:   419
Average output tokens / hit:  162
Average total tokens / hit:   581

Average tokens / alert:       1,239
Min / max tokens per alert:   568 / 2,943
Average cost / hit:           USD 0.002664
Average cost / alert:         USD 0.005683

EFFECT OF MULTIPLE HITS (average tokens per alert)
1 hit(s) / alert:             579
2 hit(s) / alert:             1,160
3 hit(s) / alert:             1,728
4 hit(s) / alert:             2,347
5 hit(s) / alert:             2,943

SCENARIOS at 10,000 alerts/month
scenario         hits/alert  tokens/alert    tokens   USD/month
observed_sample        2.13         1,239    12.39M       56.83
low                    1.40           813     8.13M       37.29
expected               1.85         1,074    10.74M       49.28
high                   4.00         2,323    23.23M      106.56

OUTPUT-SIZE SENSITIVITY
If every response used 350 output tokens:
Tokens / hit:                 769
Cost / alert:                 USD 0.009702
```

Monthly cost, observed sample mix:

| Alerts / month | Tokens | Estimated cost (USD) |
|---|---|---|
| 1,000 | 1.24M | 5.68 |
| 5,000 | 6.20M | 28.41 |
| 10,000 | 12.39M | 56.83 |
| 50,000 | 61.95M | 284.15 |
| 100,000 | 123.90M | 568.30 |

Files written to `output/`:

| File | Content |
|---|---|
| `aml_hit_token_results.csv` | one row per hit: score, band, tokens, cost |
| `aml_alert_token_results.csv` | one row per alert: hits, token totals, cost, max score |
| `aml_finance_projection.csv` | scenario × monthly volume |
| `aml_hit_details.jsonl` | per hit: factors, score breakdown, evidence, prompt, AI response |
| `aml_token_validation.csv` | Azure mode only: offline estimate vs actual |
| `aml_console_summary.txt` | the console summary |

## 18. Limitations

- Synthetic data. Real names, hit counts and missing-data rates will differ.
- The output-token figure is the weakest number. The simulated response is short and tidy; a real model may write more. Output costs four times as much as input at list price, so treat 162 tokens per hit as a lower estimate and 350 as a cautious one until Azure mode has been run.
- Offline input counts follow OpenAI's published chat-format overhead. Expect small differences from the API.
- Azure mode has been tested with a stand-in client, not against a live deployment.
- Reasoning models bill hidden reasoning tokens as output, which this estimate does not cover.
- Prices, model versions and retirement dates change. Confirm them for the deployment you will use.
- Weights, thresholds, name variants and the city-to-country table are illustrative.
- Real Dow Jones entries carry more data (aliases, several DOBs, list type, notes). Sending that will raise input tokens.

## 19. Production considerations

- **Cost levers to test:** skip low-scoring hits (`AI_MIN_FACTOR_SCORE`), one request per alert instead of per hit (the system prompt is then paid once), a smaller model for clear-cut hits, batch pricing for non-urgent alerts, and a cap on output length.
- **Volume:** take the real hits-per-alert distribution from NICE; it drives cost more than anything else here.
- **Controls:** validate the AI's JSON against a schema, log prompt, response and model version for every hit, and keep the analyst as the only decision maker.
- **Data:** customer data must stay inside the approved Azure tenancy and region; confirm data-residency and retention terms before real data is used.
- **Model risk:** the scoring weights and the AI recommendations both need back-testing against analyst decisions before they influence workflow.
