"""Event study: market-model abnormal returns after each filing.

Timing convention
-----------------
* Day 0 is the first trading day on which the market could react: the filing day if it was
  accepted before 16:00 ET on a trading day, otherwise the next trading day.
* Windows are [day0+offset, day0+offset+h-1] for h in HORIZONS (so h=1 is just day 0).
* The market model (alpha, beta vs benchmark) is estimated on trading days
  [-250, -30] relative to day 0, which stays clear of the event window.
* offset=0 assumes you could have traded at the prior close (optimistic for pre-market
  filings). offset=1 skips the first reaction day and is the conservative, tradable choice.
"""
import numpy as np
import pandas as pd

from . import config

PATH_PRE, PATH_POST = 5, 20  # event-time path for charts: day -5 .. +20


def reaction_pos(acceptance_utc, filing_date, trading_days: pd.DatetimeIndex):
    """Index of day 0 within trading_days, or None."""
    after_close = True  # conservative default when time is unknown
    if acceptance_utc:
        ts = pd.Timestamp(acceptance_utc)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        et = ts.tz_convert("America/New_York")
        d = pd.Timestamp(et.date())
        after_close = et.hour >= 16
    else:
        d = pd.Timestamp(filing_date)
    pos = trading_days.searchsorted(d)  # first trading day >= d
    if pos >= len(trading_days):
        return None
    if trading_days[pos] == d and after_close:
        pos += 1
    return pos if pos < len(trading_days) else None


def market_model(ri, rm, min_obs=120):
    ok = ~(np.isnan(ri) | np.isnan(rm))
    if ok.sum() < min_obs:
        return None
    x, y = rm[ok], ri[ok]
    var = x.var(ddof=1)
    if var == 0:
        return None
    beta = np.cov(x, y, ddof=1)[0, 1] / var
    return y.mean() - beta * x.mean(), beta


def event_metrics(ri, rm, pos, horizons=config.HORIZONS, offset=0, est=(-250, -30), min_obs=120):
    """Return (dict of CARs, event-time AR path) or None if not computable."""
    lo, hi = pos + est[0], pos + est[1]
    lo = max(lo, 0)
    if hi <= lo:
        return None
    mm = market_model(ri[lo:hi], rm[lo:hi], min_obs)
    if mm is None:
        return None
    a, b = mm
    ar = ri - (a + b * rm)
    s = pos + offset
    out = {"beta": b}
    for h in horizons:
        e = s + h
        if e > len(ri) or np.isnan(ar[s:e]).any():
            out[f"car_{h}"] = np.nan
        else:
            out[f"car_{h}"] = float(ar[s:e].sum())
    p0, p1 = pos - PATH_PRE, pos + PATH_POST + 1
    path = np.full(PATH_PRE + PATH_POST + 1, np.nan)
    if p0 >= 0 and p1 <= len(ar):
        path = ar[p0:p1]
    return out, path


def run_event_study(filings: pd.DataFrame, returns: pd.DataFrame, bench=None,
                    horizons=config.HORIZONS, offset=0):
    """filings needs: accession, ticker, filing_date, acceptance_utc.

    Returns (events DataFrame, paths DataFrame indexed by accession, columns -5..+20).
    """
    bench = bench or config.BENCHMARK
    days = returns.index
    rm = returns[bench].to_numpy(dtype=float)
    rows, paths = [], {}
    for f in filings.itertuples(index=False):
        if f.ticker not in returns.columns:
            continue
        pos = reaction_pos(getattr(f, "acceptance_utc", None), f.filing_date, days)
        if pos is None:
            continue
        res = event_metrics(returns[f.ticker].to_numpy(dtype=float), rm, pos, horizons, offset)
        if res is None:
            continue
        m, path = res
        m.update(accession=f.accession, day0=days[pos])
        rows.append(m)
        paths[f.accession] = path
    events = pd.DataFrame(rows)
    path_df = pd.DataFrame.from_dict(paths, orient="index",
                                     columns=list(range(-PATH_PRE, PATH_POST + 1)))
    return events, path_df
