"""Turn raw extractor output into cross-sectionally standardised signals.

* Level features (hedging, tone, risk themes) are differenced vs the same ticker's previous
  document of the same source (periodic / release / transcript) -> quarter-over-quarter change.
* guidance_direction is already a change, so it is used as-is.
* Every feature is z-scored within (source, calendar quarter) and winsorised at +/-3.
* signal_prior = equal-weighted sum of sign-adjusted z-scores, with signs fixed *before*
  looking at returns (config.FEATURE_SIGNS). Nothing in it is fitted to returns.
"""
import numpy as np
import pandas as pd

from . import config


def _zscore_group(s: pd.Series, min_n=5):
    if s.notna().sum() < min_n:
        return pd.Series(np.nan, index=s.index)
    sd = s.std(ddof=1)
    if not sd or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return ((s - s.mean()) / sd).clip(-3, 3)


def build_signal_frame(extr: pd.DataFrame, filings: pd.DataFrame,
                       features=config.FEATURES, min_feats=3) -> pd.DataFrame:
    df = filings.merge(extr, on="accession", how="inner")
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.sort_values(["ticker", "source", "filing_date"]).reset_index(drop=True)
    for f in features:
        if f in config.DIRECT_FEATURES:
            df[f"chg_{f}"] = df[f]
        else:
            df[f"chg_{f}"] = df.groupby(["ticker", "source"])[f].diff()
    df["period"] = df["filing_date"].dt.to_period("Q").astype(str)
    zcols = []
    for f in features:
        c = f"z_{f}"
        df[c] = df.groupby(["source", "period"], group_keys=False)[f"chg_{f}"].apply(_zscore_group)
        zcols.append(c)
    signed = pd.DataFrame({c: df[c] * config.FEATURE_SIGNS[c[2:]] for c in zcols})
    df["signal_prior"] = signed.mean(axis=1, skipna=True)
    df.loc[signed.notna().sum(axis=1) < min_feats, "signal_prior"] = np.nan
    return df
