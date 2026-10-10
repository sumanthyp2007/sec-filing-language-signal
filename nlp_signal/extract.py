"""Feature extractors. Both return the same feature dict (see config.FEATURES).

* LLMExtractor     - Anthropic API, structured JSON output, cached in SQLite.
* LexiconExtractor - transparent word-list baseline; needs no API key. Also serves as
                     the benchmark the LLM must beat to justify its cost.
"""
import json
import math
import re
import time

import numpy as np

from . import config, db, textproc

# ---------------------------------------------------------------- lexicon baseline
HEDGE = set("""may might could possibly perhaps approximately uncertain uncertainty uncertainties
depend depends dependent believe believes appear appears suggest suggests potential potentially
likely unlikely assume assumes assumption assumptions contingent volatile volatility fluctuate
fluctuations cautious caution unpredictable""".split())
POS = set("""strong growth growing improved improvement improving record gains gain increased exceeded
exceed outperform momentum robust confident opportunity opportunities positive favorable
successful success expanded expansion accelerate accelerated""".split())
NEG = set("""decline declined declining weak weakness decreased decrease loss losses impairment
challenging challenges headwind headwinds adverse unfavorable pressure pressures deteriorate
deteriorated shortfall missed disappointing negative lower reduced""".split())
RISK_WORDS = {
    "risk_supply_chain": ["supply chain", "supply constraints", "shortage", "shortages", "logistics",
                          "freight", "component availability", "supplier", "suppliers", "inventory constraints"],
    "risk_fx": ["foreign currency", "foreign exchange", "exchange rate", "exchange rates",
                "currency headwind", "currency headwinds", "fx", "stronger dollar", "strengthening dollar"],
    "risk_demand": ["demand", "softness", "slowdown", "consumer spending", "customer spending",
                    "order cancellations", "weak demand", "demand environment"],
}
_UP = re.compile(r"(rais|increas|upgrad|higher|improv)\w*\s+([\w-]+\s+){0,4}(guidance|outlook|forecast)", re.I)
_DOWN = re.compile(r"(lower|reduc|cut|decreas|withdr[ae]w)\w*\s+([\w-]+\s+){0,4}(guidance|outlook|forecast)", re.I)
_UP2 = re.compile(r"(guidance|outlook|forecast)\s+([\w-]+\s+){0,3}(raised|increased|higher|improved)", re.I)
_DOWN2 = re.compile(r"(guidance|outlook|forecast)\s+([\w-]+\s+){0,3}(lowered|reduced|cut|withdrawn|lower)", re.I)
_WORD = re.compile(r"[a-z']+")


class LexiconExtractor:
    name = "lexicon"

    def extract(self, text: str, meta: dict | None = None) -> dict:
        low = text.lower()
        words = _WORD.findall(low)
        n = max(len(words), 1)
        per_k = 1000.0 / n
        hedge = sum(w in HEDGE for w in words) * per_k
        pos = sum(w in POS for w in words)
        neg = sum(w in NEG for w in words)
        tone = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
        up = len(_UP.findall(text)) + len(_UP2.findall(text))
        down = len(_DOWN.findall(text)) + len(_DOWN2.findall(text))
        out = {
            "guidance_direction": float(np.sign(up - down)),
            "hedging_score": hedge,
            "tone_score": tone,
        }
        for k, kws in RISK_WORDS.items():
            out[k] = sum(low.count(w) for w in kws) * per_k
        return out


# ---------------------------------------------------------------------- LLM
SYSTEM_PROMPT = """You are a careful equity-research analyst. You read one company document \
(SEC filing section, earnings press release or earnings-call transcript) and score it. \
Base every score ONLY on the text provided. Do not use outside knowledge about the company, \
its stock price, or what happened afterwards. Reply with a single JSON object and nothing else."""

USER_TEMPLATE = """Document type: {form}
Company ticker: {ticker}

Score the document and return JSON with exactly these keys:
- "guidance_provided": true/false - does management give forward-looking numeric or directional guidance?
- "guidance_direction": -1 if guidance was lowered/withdrawn/weaker than before, 0 if none or unchanged/maintained, +1 if raised/improved
- "hedging_score": integer 0-10. 0 = direct, committed language; 10 = heavily hedged, qualified, uncertain
- "tone_score": number -1.0 to +1.0. Management tone about the business (-1 very negative, +1 very positive)
- "risk_supply_chain": integer 0-10, emphasis on supply-chain problems (shortages, logistics, supplier issues)
- "risk_fx": integer 0-10, emphasis on foreign-exchange headwinds/exposure
- "risk_demand": integer 0-10, emphasis on weak/uncertain customer demand
- "rationale": one sentence (max 30 words) justifying the scores

DOCUMENT:
\"\"\"
{text}
\"\"\""""

_RANGES = {"guidance_direction": (-1, 1), "hedging_score": (0, 10), "tone_score": (-1, 1),
           "risk_supply_chain": (0, 10), "risk_fx": (0, 10), "risk_demand": (0, 10)}


def parse_llm_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON in response: {raw[:200]!r}")
    d = json.loads(m.group(0))
    out = {}
    for k, (lo, hi) in _RANGES.items():
        v = d.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise ValueError(f"missing {k}")
        out[k] = float(min(max(float(v), lo), hi))
    if not d.get("guidance_provided", True):
        out["guidance_direction"] = 0.0
    out["rationale"] = str(d.get("rationale", ""))[:300]
    return out


class LLMExtractor:
    name = "llm"

    def __init__(self, model=None, client=None):
        import anthropic
        self.model = model or config.ANTHROPIC_MODEL
        self.client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def extract(self, text: str, meta: dict | None = None) -> dict:
        meta = meta or {}
        body = textproc.truncate(text, config.MAX_CHARS)
        prompt = USER_TEMPLATE.format(form=meta.get("form", "document"),
                                      ticker=meta.get("ticker", "?"), text=body)
        last = None
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=500, temperature=0,
                    system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}])
                raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                return parse_llm_json(raw)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM extraction failed: {last}")


def get_extractor(name: str):
    if name == "lexicon":
        return LexiconExtractor()
    if name == "llm":
        return LLMExtractor()
    raise ValueError(name)


def run_extraction(con, extractor, limit=None, verbose=True):
    """Extract features for every filing that has no cached result. Returns count."""
    rows = db.pending_extractions(con, extractor.name, config.PROMPT_VERSION, limit)
    done = 0
    for i, r in enumerate(rows, 1):
        try:
            text = open(r["text_path"], encoding="utf-8", errors="ignore").read()
            feats = extractor.extract(text, dict(form=r["form"], ticker=r["ticker"]))
            db.save_extraction(con, r["accession"], extractor.name, config.PROMPT_VERSION, feats)
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {r['accession']}: {e}")
        if verbose and i % 25 == 0:
            print(f"  extracted {i}/{len(rows)}")
    return done
