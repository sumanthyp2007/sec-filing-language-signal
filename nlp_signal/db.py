"""SQLite storage: filings metadata + cached extractor outputs."""
import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    cik TEXT,
    source TEXT NOT NULL,          -- periodic | release | transcript
    form TEXT,
    filing_date TEXT NOT NULL,     -- YYYY-MM-DD
    acceptance_utc TEXT,           -- ISO timestamp (UTC) when available
    report_date TEXT,
    url TEXT,
    text_path TEXT,
    n_chars INTEGER
);
CREATE INDEX IF NOT EXISTS ix_filings_ticker ON filings(ticker, source, filing_date);
CREATE TABLE IF NOT EXISTS extractions (
    accession TEXT NOT NULL,
    extractor TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    features_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (accession, extractor, prompt_version)
);
"""


def connect(path=None):
    path = path or config.DB_PATH
    if str(path) != ":memory:":
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert_filing(con, rec: dict):
    cols = ["accession", "ticker", "cik", "source", "form", "filing_date",
            "acceptance_utc", "report_date", "url", "text_path", "n_chars"]
    vals = [rec.get(c) for c in cols]
    con.execute(
        f"INSERT OR REPLACE INTO filings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        vals,
    )
    con.commit()


def filing_exists(con, accession: str) -> bool:
    return con.execute("SELECT 1 FROM filings WHERE accession=?", (accession,)).fetchone() is not None


def filings_df(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM filings", con)


def pending_extractions(con, extractor: str, prompt_version: str, limit=None):
    q = """SELECT f.* FROM filings f
           LEFT JOIN extractions e ON e.accession=f.accession
                AND e.extractor=? AND e.prompt_version=?
           WHERE e.accession IS NULL AND f.text_path IS NOT NULL
           ORDER BY f.filing_date"""
    if limit:
        q += f" LIMIT {int(limit)}"
    return con.execute(q, (extractor, prompt_version)).fetchall()


def save_extraction(con, accession, extractor, prompt_version, features: dict):
    con.execute(
        "INSERT OR REPLACE INTO extractions VALUES (?,?,?,?,?)",
        (accession, extractor, prompt_version, json.dumps(features),
         datetime.now(timezone.utc).isoformat()),
    )
    con.commit()


def extractions_df(con, extractor: str, prompt_version: str) -> pd.DataFrame:
    rows = con.execute(
        "SELECT accession, features_json FROM extractions WHERE extractor=? AND prompt_version=?",
        (extractor, prompt_version),
    ).fetchall()
    recs = []
    for r in rows:
        d = json.loads(r["features_json"])
        d["accession"] = r["accession"]
        recs.append(d)
    return pd.DataFrame(recs)
