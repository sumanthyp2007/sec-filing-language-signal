"""Minimal SEC EDGAR client (submissions API + document download).

SEC fair-access policy: <=10 requests/sec and a descriptive User-Agent with contact
info. Set SEC_USER_AGENT="Your Name your@email.com".
"""
import time
from datetime import date

import requests

from . import config, db, textproc

SEC_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
INDEX_JSON = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json"

_MIN_INTERVAL = 0.15
_last = [0.0]


class EdgarClient:
    def __init__(self, user_agent=None):
        ua = user_agent or config.SEC_USER_AGENT
        if not ua:
            raise RuntimeError('Set SEC_USER_AGENT="Your Name your@email.com" (required by the SEC).')
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self._tickers = None

    def _get(self, url, retries=4):
        for i in range(retries):
            wait = _MIN_INTERVAL - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
            r = self.s.get(url, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 500, 502, 503):
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
        r.raise_for_status()

    def cik_for(self, ticker: str):
        if self._tickers is None:
            data = self._get(SEC_TICKER_MAP).json()
            self._tickers = {v["ticker"].upper(): str(v["cik_str"]) for v in data.values()}
        t = ticker.upper()
        return self._tickers.get(t) or self._tickers.get(t.replace("-", "."))

    def submissions(self, cik):
        return self._get(SUBMISSIONS.format(cik=int(cik))).json()

    @staticmethod
    def _rows(block):
        keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form",
                "primaryDocument", "reportDate", "items"]
        n = len(block.get("accessionNumber", []))
        for i in range(n):
            yield {k: (block[k][i] if k in block and i < len(block[k]) else None) for k in keys}

    def list_filings(self, cik, forms, start: str, end: str):
        sub = self.submissions(cik)
        rows = list(self._rows(sub["filings"]["recent"]))
        earliest = min((r["filingDate"] for r in rows), default="9999")
        if start < earliest:  # older filings live in paginated files
            for f in sub["filings"].get("files", []):
                if f["filingTo"] >= start:
                    older = self._get(f"https://data.sec.gov/submissions/{f['name']}").json()
                    rows += list(self._rows(older))
        out = []
        for r in rows:
            if r["form"] in forms and start <= r["filingDate"] <= end:
                out.append(r)
        return out

    def fetch_html(self, cik, accession, doc):
        url = ARCHIVE.format(cik=int(cik), acc_nodash=accession.replace("-", ""), doc=doc)
        return url, self._get(url).text

    def earnings_release_doc(self, cik, accession):
        """Best effort: find the EX-99.x document of an 8-K (press release)."""
        idx = self._get(INDEX_JSON.format(cik=int(cik), acc_nodash=accession.replace("-", ""))).json()
        names = [it["name"] for it in idx.get("directory", {}).get("item", [])]
        cands = [n for n in names if n.lower().endswith((".htm", ".html"))
                 and ("ex99" in n.lower().replace("-", "").replace("_", "")
                      or "ex-99" in n.lower())]
        return sorted(cands)[0] if cands else None


def ingest_ticker(con, client: EdgarClient, ticker, start, end, with_releases=True):
    """Download + clean + store filings for one ticker. Returns number added."""
    config.ensure_dirs()
    cik = client.cik_for(ticker)
    if not cik:
        print(f"  [skip] no CIK for {ticker}")
        return 0
    forms = {"10-K", "10-Q"} | ({"8-K"} if with_releases else set())
    added = 0
    for r in client.list_filings(cik, forms, start, end):
        acc = r["accessionNumber"]
        if db.filing_exists(con, acc):
            continue
        form = r["form"]
        if form == "8-K":
            if "2.02" not in (r.get("items") or ""):
                continue
            doc = client.earnings_release_doc(cik, acc)
            source = "release"
        else:
            doc, source = r["primaryDocument"], "periodic"
        if not doc:
            continue
        try:
            url, raw = client.fetch_html(cik, acc, doc)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {ticker} {acc}: {e}")
            continue
        text = textproc.html_to_text(raw)
        if source == "periodic":
            text = textproc.extract_mdna(text, form)
        path = config.TEXT_DIR / f"{acc}.txt"
        path.write_text(text, encoding="utf-8")
        acceptance = r.get("acceptanceDateTime")
        if acceptance and not acceptance.endswith("Z"):
            acceptance += "Z"
        db.upsert_filing(con, dict(
            accession=acc, ticker=ticker, cik=cik, source=source, form=form,
            filing_date=r["filingDate"], acceptance_utc=acceptance,
            report_date=r.get("reportDate"), url=url, text_path=str(path), n_chars=len(text)))
        added += 1
    return added


def ingest_transcripts(con, folder=None):
    """Load user-supplied transcripts named TICKER_YYYY-MM-DD[_HHMM].txt.

    Free, complete earnings-call transcripts are not available from a free API, so
    bring your own (e.g. exported from a data vendor you have a licence for).
    Time is assumed to be *after market close* unless _HHMM (ET, 24h) is in the name.
    """
    import re
    from zoneinfo import ZoneInfo

    import pandas as pd
    folder = folder or config.TRANSCRIPT_DIR
    n = 0
    for p in sorted(folder.glob("*.txt")):
        m = re.match(r"([A-Za-z.\-]+)_(\d{4}-\d{2}-\d{2})(?:_(\d{4}))?", p.stem)
        if not m:
            print(f"  [skip] bad name {p.name}")
            continue
        tk, d, hhmm = m.group(1).upper(), m.group(2), m.group(3) or "1630"
        ts = pd.Timestamp(f"{d} {hhmm[:2]}:{hhmm[2:]}", tz=ZoneInfo("America/New_York")).tz_convert("UTC")
        text = p.read_text(encoding="utf-8", errors="ignore")
        db.upsert_filing(con, dict(
            accession=f"TRANSCRIPT-{tk}-{d}", ticker=tk, cik=None, source="transcript",
            form="TRANSCRIPT", filing_date=d, acceptance_utc=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            report_date=None, url=str(p), text_path=str(p), n_chars=len(text)))
        n += 1
    return n
