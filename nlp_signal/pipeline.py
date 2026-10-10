"""Glue: DB -> signal frame -> event study -> analysis report."""
from pathlib import Path

from . import config, db, event_study, report, signals


def run_analysis(con, returns, extractor, out_dir, split_date, offset=0, cost_bps=10.0,
                 label="", sources=None, horizons=config.HORIZONS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filings = db.filings_df(con)
    extr = db.extractions_df(con, extractor, config.PROMPT_VERSION)
    if extr.empty:
        raise SystemExit(f"No extractions for extractor '{extractor}'. Run the extract step first.")
    sig = signals.build_signal_frame(extr, filings)
    if sources:
        sig = sig[sig["source"].isin(sources)]
    sig = sig.dropna(subset=["signal_prior"]).reset_index(drop=True)
    events, paths = event_study.run_event_study(
        sig[["accession", "ticker", "filing_date", "acceptance_utc"]], returns,
        offset=offset, horizons=horizons)
    sig.to_csv(out_dir / "signal_frame.csv", index=False)
    events.to_csv(out_dir / "events.csv", index=False)
    print(f"{len(sig)} signals, {len(events)} events with computable abnormal returns")
    return report.run_all(sig, events, paths, out_dir, split_date, cost_bps, label, horizons)
