"""Default universe: ~100 large-cap US stocks (current S&P 500 members).

NOTE: using *current* constituents introduces survivorship bias. See README.
"""
DEFAULT_TICKERS = """
AAPL MSFT AMZN GOOGL META NVDA TSLA BRK-B JPM JNJ V PG UNH HD MA XOM CVX LLY ABBV PFE KO PEP MRK
AVGO COST WMT BAC CSCO TMO ABT MCD ACN CRM DHR NKE LIN TXN NEE ADBE VZ CMCSA ORCL AMD QCOM HON UPS
PM INTC LOW IBM CAT GS MS BA AMGN SBUX RTX DE INTU BLK ELV MDT GE AXP T BKNG ISRG PLD SPGI MDLZ ADP
GILD CVS TJX SYK LMT MMC ZTS C MO CB SO DUK USB PNC BDX CL EOG SLB NOC ITW FDX EMR GM F MMM HUM
""".split()


def load_universe(path=None, n=None):
    if path:
        with open(path) as f:
            tickers = [t.strip().upper() for t in f.read().replace(",", "\n").split() if t.strip()]
    else:
        tickers = list(DEFAULT_TICKERS)
    return tickers[:n] if n else tickers
