"""Does the signal predict abnormal returns? Quantile portfolios, IC, regression, OOS."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config


def make_panel(sig: pd.DataFrame, events: pd.DataFrame, h: int) -> pd.DataFrame:
    cols = ["accession", f"car_{h}", "beta", "day0"]
    p = sig.merge(events[cols], on="accession", how="inner").rename(columns={f"car_{h}": "car"})
    return p.dropna(subset=["car"]).reset_index(drop=True)


def assign_quintiles(panel, col, n=5, min_per_period=20) -> pd.Series:
    lab = pd.Series(np.nan, index=panel.index)
    for _, g in panel.dropna(subset=[col]).groupby("period"):
        if len(g) < min_per_period:
            continue
        lab.loc[g.index] = pd.qcut(g[col].rank(method="first"), n, labels=False) + 1
    return lab


def _tstat(x: pd.Series):
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def quintile_returns(panel, col="signal_prior", n=5, min_per_period=20, cost_bps=0.0):
    """Mean CAR per quantile per formation period, plus the long-short (Q_n - Q_1) spread.

    cost_bps is a per-leg round-trip cost; long-short pays it on both legs.
    """
    lab = assign_quintiles(panel, col, n, min_per_period)
    d = panel.assign(q=lab).dropna(subset=["q"])
    if d.empty:
        return dict(table=pd.DataFrame(), ls=pd.Series(dtype=float), stats={})
    tbl = d.groupby(["period", "q"])["car"].mean().unstack("q")
    tbl.columns = [int(c) for c in tbl.columns]
    ls = (tbl[n] - tbl[1]).dropna()
    ls_net = ls - 2 * cost_bps / 1e4
    stats = dict(
        n_periods=int(len(ls)), n_events=int(len(d)),
        mean_by_quantile={int(k): float(v) for k, v in d.groupby("q")["car"].mean().items()},
        ls_mean=float(ls.mean()), ls_t=_tstat(ls),
        ls_net_mean=float(ls_net.mean()), ls_net_t=_tstat(ls_net),
        ls_hit_rate=float((ls > 0).mean()),
    )
    return dict(table=tbl, ls=ls, stats=stats)


def ic_series(panel, col="signal_prior", min_per_period=10) -> pd.Series:
    out = {}
    for per, g in panel.dropna(subset=[col, "car"]).groupby("period"):
        if len(g) >= min_per_period:
            out[per] = g[col].rank().corr(g["car"].rank())
    return pd.Series(out, dtype=float)


def ic_stats(ic: pd.Series) -> dict:
    return dict(mean_ic=float(ic.mean()) if len(ic) else np.nan, ic_t=_tstat(ic),
                ic_hit=float((ic > 0).mean()) if len(ic) else np.nan, ic_periods=int(len(ic)))


def regress(panel, cols, add_controls=False) -> pd.DataFrame:
    """CAR ~ signal(s) with standard errors clustered by filing quarter."""
    d = panel.dropna(subset=cols + ["car"])
    X = sm.add_constant(d[cols])
    groups = d["period"].astype("category").cat.codes
    if groups.nunique() >= 5:
        res = sm.OLS(d["car"], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = sm.OLS(d["car"], X).fit(cov_type="HC1")
    ci = res.conf_int()
    return pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues,
                         "ci_lo": ci[0], "ci_hi": ci[1], "n": int(res.nobs)})


def oos_evaluation(panel, split_date, n=5, min_per_period=20, cost_bps=0.0):
    """Fit feature weights on data BEFORE split_date only; evaluate on data on/after it.

    Compares (a) the pre-registered equal-weight prior signal and (b) an OLS-fitted signal.
    """
    zcols = [f"z_{f}" for f in config.FEATURES]
    split = pd.Timestamp(split_date)
    tr = panel[panel["filing_date"] < split].dropna(subset=zcols + ["car"])
    te = panel[panel["filing_date"] >= split].dropna(subset=zcols).copy()
    out = dict(split=str(split.date()), n_train=int(len(tr)), n_test=int(len(te)))
    if len(tr) < 100 or len(te) < 50:
        out["note"] = "too few observations on one side of the split"
        return out
    res = sm.OLS(tr["car"], sm.add_constant(tr[zcols])).fit()
    beta = res.params.drop("const")
    te["signal_fit"] = te[zcols].to_numpy() @ beta.to_numpy()
    out["fitted_weights"] = {k: float(v) for k, v in beta.items()}
    for name in ("signal_prior", "signal_fit"):
        q = quintile_returns(te, name, n, min_per_period, cost_bps)
        out[name] = dict(**ic_stats(ic_series(te, name)), **q.get("stats", {}))
    return out
