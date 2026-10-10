"""Daily adjusted-close download (yfinance) with on-disk CSV cache."""
import pandas as pd

from . import config


def get_returns(tickers, start, end, benchmark=None, cache_dir=None, refresh=False):
    """Return a DataFrame of daily simple returns (columns = tickers + benchmark)."""
    import yfinance as yf
    benchmark = benchmark or config.BENCHMARK
    cache_dir = cache_dir or config.PRICE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cols = {}
    for t in list(dict.fromkeys(list(tickers) + [benchmark])):
        f = cache_dir / f"{t}.csv"
        if f.exists() and not refresh:
            px = pd.read_csv(f, index_col=0, parse_dates=True)["close"]
        else:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                print(f"  [warn] no prices for {t}")
                continue
            close = df["Close"]
            px = close.iloc[:, 0] if hasattr(close, "columns") else close
            px.name = "close"
            px.to_frame("close").to_csv(f)
        cols[t] = px
    prices = pd.DataFrame(cols).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.pct_change(fill_method=None).iloc[1:]
