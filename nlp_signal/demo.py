"""Synthetic end-to-end demo that needs no network and no API key.

Generates fake prices and fake extractor output with a *planted* relationship between the
text-feature changes and subsequent abnormal returns. It exists to prove the pipeline is
wired correctly (it should recover the planted effect, and recover ~0 with effect=0).
Results from it say NOTHING about real markets.
"""
import numpy as np
import pandas as pd

from . import config, db


def make_synthetic(con, n_tickers=80, start="2012-01-02", end="2023-12-29", effect=0.02, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    nd = len(days)
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    rm = rng.normal(0.0004, 0.01, nd)
    betas = rng.uniform(0.6, 1.4, n_tickers)
    rets = betas * rm[:, None] + rng.normal(0, 0.015, (nd, n_tickers))

    feats = config.FEATURES
    scale = {"hedging_score": 1.0, "tone_score": 0.1, "risk_supply_chain": 1.0,
             "risk_fx": 1.0, "risk_demand": 1.0}
    qends = pd.date_range("2012-12-31", "2023-06-30", freq="QE")
    for j, tk in enumerate(tickers):
        level = {f: rng.normal(10, 3) for f in feats}
        for qe in qends:
            fd = qe + pd.Timedelta(days=int(rng.integers(20, 50)))
            pos = days.searchsorted(fd)
            if pos >= nd - 25:
                continue
            fdate = days[pos]
            pre = rng.random() < 0.5
            hour = 7 if pre else 17
            acc = f"DEMO-{tk}-{fdate.date()}"
            et = pd.Timestamp(f"{fdate.date()} {hour:02d}:30", tz="America/New_York").tz_convert("UTC")
            innov = {f: rng.normal() for f in feats}
            comp = np.mean([config.FEATURE_SIGNS[f] * innov[f] for f in feats])
            day0 = pos if pre else pos + 1
            if day0 + 20 <= nd:
                rets[day0:day0 + 20, j] += effect * comp / 20.0
            obs = {}
            for f in feats:
                if f == "guidance_direction":
                    g = innov[f]
                    obs[f] = 1.0 if g > 0.7 else (-1.0 if g < -0.7 else 0.0)
                else:
                    level[f] += innov[f] * scale[f]
                    obs[f] = level[f] + rng.normal(0, 0.2 * scale[f])
            db.upsert_filing(con, dict(
                accession=acc, ticker=tk, cik=None, source="periodic", form="10-Q",
                filing_date=str(fdate.date()), acceptance_utc=et.strftime("%Y-%m-%dT%H:%M:%SZ"),
                report_date=str(qe.date()), url=None, text_path="synthetic", n_chars=0))
            db.save_extraction(con, acc, "demo", config.PROMPT_VERSION, obs)
    returns = pd.DataFrame(rets, index=days, columns=tickers)
    returns[config.BENCHMARK] = rm
    return returns
