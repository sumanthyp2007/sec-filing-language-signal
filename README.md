# Earnings-Call and Filing NLP Signal Engine

Turns SEC filings, earnings releases and (your own) earnings-call transcripts into a tradable signal,
then tests honestly whether it predicts abnormal returns.

```
EDGAR / transcripts -> clean text -> LLM or lexicon features -> SQLite cache
      -> QoQ change, cross-sectional z-scores, composite signal
      -> event study (market-model CARs at 1/5/20 days)
      -> quintile spreads, IC, clustered regression, out-of-sample test -> charts + results.md
```

## Status: what is and is not verified

- **Verified offline (10 passing tests + synthetic demo):** event-timing rules, market-model CARs,
  QoQ differencing, quintile/IC/regression/OOS logic, LLM-response parsing, the full pipeline. On synthetic data with a
  planted effect it recovers the effect (20d IC ~0.12, OOS holds); with effect=0 it finds nothing.
- **Written but NOT run against live services** (the build environment had no access to sec.gov, Yahoo Finance
  or the Anthropic API): EDGAR ingestion, yfinance download, the live LLM call. Expect to fix small things on first
  contact (e.g. an 8-K exhibit that the filename heuristic misses). Start with `-n 5`.
- **No real-market results are included.** `examples/synthetic_demo/` is synthetic and labelled as such.

## Setup (Python 3.10+)

```bash
pip install -r requirements.txt
cp .env.example .env     # then export the variables, or set them in your shell
export SEC_USER_AGENT="Your Name your@email.com"     # required by the SEC
export ANTHROPIC_API_KEY=...                         # only for --extractor llm
python -m pytest -q tests                            # sanity check
python -m nlp_signal.cli demo                        # offline synthetic run -> outputs/demo/
```

## Real run

```bash
# 1. Download 10-K/10-Q MD&A and 8-K Item 2.02 earnings releases (start small!)
python -m nlp_signal.cli ingest -n 10 --start 2019-01-01

# 2. Optional: your own transcripts as data/transcripts/AAPL_2024-02-01_1630.txt  (ET time optional)
python -m nlp_signal.cli transcripts

# 3. Features. Start with the free lexicon baseline, then the LLM; both are cached in SQLite
python -m nlp_signal.cli extract --extractor lexicon
python -m nlp_signal.cli extract --extractor llm --limit 50     # check quality/cost on a few first

# 4. Analyse (conservative: skip the reaction day)
python -m nlp_signal.cli analyze --extractor llm --offset 1 --split 2022-01-01 --cost-bps 10
```

Open `notebooks/results.ipynb` for charts and tables. Fill in `docs/WRITEUP_TEMPLATE.md` from a real run.

**LLM cost:** the full ~100 tickers x 2018-2025 is roughly 6,000 documents, i.e. tens of millions of input tokens.
Use `-n`, a shorter `--start`, `--sources`, or a smaller `MAX_CHARS` while developing, and check current pricing.

## Design decisions worth defending in an interview

| Choice | Why |
|---|---|
| Signs of each feature fixed in `config.py` before seeing returns; equal-weight prior composite | No fitted weights means nothing to overfit; an OLS-fitted version is evaluated only out of sample |
| Features differenced vs the same ticker's previous document of the *same source* | Compares like with like (10-Q to 10-Q, release to release) |
| z-scores within (source, calendar quarter) | Removes market-wide language drift (e.g. everyone mentions FX in a strong-dollar quarter) |
| Day 0 = first tradable session after SEC acceptance time (ET); `--offset 1` skips it | Avoids trading on information already in the price |
| Market model estimated on [-250,-30] | Keeps the event window out of the estimation window |
| t-stats computed across quarters; regression SEs clustered by quarter | Events in the same quarter share market shocks, so they are not independent |
| Lexicon baseline alongside the LLM | The LLM must beat word counts to justify its cost |
| Every LLM output cached by (document, extractor, prompt version) | Reproducible, resumable, cheap to re-analyse |

## Known limitations (also list these in your write-up)

1. **Survivorship bias:** the default universe is today's large caps.
2. **LLM look-ahead / memorisation:** a model may have seen these filings *and* the subsequent price moves in training.
   The prompt forbids outside knowledge but cannot guarantee it. Test periods after the model's training cutoff.
3. **Timing of 10-K/10-Q:** they often arrive days to weeks after the earnings release, so the "event" is not the
   earnings surprise. Releases (8-K 2.02) and transcripts are the cleaner event; compare with `--sources`.
4. **Transcripts:** no free, complete source exists; bring your own licensed text.
5. **8-K exhibit finding is a filename heuristic** (`ex99*.htm`), best effort.
6. **Multiple testing:** 3 horizons x 6 features x several sources. Report how many you looked at.
7. **Costs:** a flat bps assumption; no borrow cost or capacity analysis.
8. MD&A extraction is regex-based and can fail on unusual filings (it falls back to the full text).

## Layout

```
nlp_signal/  config, db, universe, edgar, textproc, extract, prices, signals,
             event_study, analysis, report, pipeline, demo, cli
tests/       offline unit + end-to-end tests
notebooks/   results.ipynb
docs/        WRITEUP_TEMPLATE.md
examples/    synthetic_demo/  (synthetic, for pipeline validation only)
```
