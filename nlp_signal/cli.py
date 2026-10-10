"""Command line interface:  python -m nlp_signal.cli <command> --help"""
import argparse

from . import config, db, edgar, extract, pipeline, prices, universe


def cmd_ingest(a):
    config.ensure_dirs()
    con = db.connect()
    client = edgar.EdgarClient()
    tickers = universe.load_universe(a.universe, a.n)
    total = 0
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}")
        total += edgar.ingest_ticker(con, client, t, a.start, a.end, with_releases=not a.no_releases)
    print(f"Added {total} documents")


def cmd_transcripts(a):
    config.ensure_dirs()
    con = db.connect()
    print(f"Loaded {edgar.ingest_transcripts(con, a.dir)} transcripts")


def cmd_extract(a):
    config.ensure_dirs()
    con = db.connect()
    n = extract.run_extraction(con, extract.get_extractor(a.extractor), a.limit)
    print(f"Extracted {n} documents with '{a.extractor}'")


def cmd_analyze(a):
    config.ensure_dirs()
    con = db.connect()
    tickers = sorted({r["ticker"] for r in con.execute("SELECT DISTINCT ticker FROM filings")})
    returns = prices.get_returns(tickers, a.price_start, a.price_end)
    sources = a.sources.split(",") if a.sources else None
    res = pipeline.run_analysis(con, returns, a.extractor, a.out, a.split, a.offset, a.cost_bps,
                                label=f"extractor={a.extractor}, offset={a.offset}", sources=sources)
    print((a.out / "results.md").read_text() if hasattr(a.out, "read_text") else open(f"{a.out}/results.md").read())


def cmd_demo(a):
    from . import demo
    con = db.connect(":memory:")
    returns = demo.make_synthetic(con, effect=a.effect)
    label = (f"SYNTHETIC DEMO DATA (planted effect={a.effect}). Validates the pipeline only; "
             "says nothing about real markets.")
    pipeline.run_analysis(con, returns, "demo", a.out, "2019-01-01", offset=0, cost_bps=10, label=label)
    print(open(f"{a.out}/results.md").read())


def main(argv=None):
    from pathlib import Path
    p = argparse.ArgumentParser(prog="nlp_signal")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="download 10-K/10-Q (+8-K earnings releases) from EDGAR")
    s.add_argument("--universe", help="text file of tickers (default: built-in ~100 large caps)")
    s.add_argument("-n", type=int, help="use only first N tickers")
    s.add_argument("--start", default="2018-01-01"); s.add_argument("--end", default="2025-12-31")
    s.add_argument("--no-releases", action="store_true", help="skip 8-K earnings releases")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("transcripts", help="load your own transcripts: TICKER_YYYY-MM-DD[_HHMM].txt")
    s.add_argument("--dir", type=Path, default=config.TRANSCRIPT_DIR); s.set_defaults(fn=cmd_transcripts)

    s = sub.add_parser("extract", help="run feature extraction (cached in SQLite)")
    s.add_argument("--extractor", choices=["llm", "lexicon"], default="lexicon")
    s.add_argument("--limit", type=int); s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("analyze", help="event study + signal tests + report")
    s.add_argument("--extractor", choices=["llm", "lexicon"], default="lexicon")
    s.add_argument("--split", default="2022-01-01", help="OOS split date (YYYY-MM-DD)")
    s.add_argument("--offset", type=int, default=0, help="1 = skip reaction day (conservative)")
    s.add_argument("--cost-bps", type=float, default=10.0)
    s.add_argument("--sources", help="comma list of periodic,release,transcript")
    s.add_argument("--price-start", default="2016-06-01"); s.add_argument("--price-end", default="2026-01-01")
    s.add_argument("--out", type=Path, default=config.OUT_DIR); s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("demo", help="offline synthetic end-to-end run")
    s.add_argument("--effect", type=float, default=0.02)
    s.add_argument("--out", type=Path, default=config.OUT_DIR / "demo"); s.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
