"""HTML -> clean text, and best-effort extraction of the MD&A section."""
import re

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")


def html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.lower() in ("ix:header",)):
        tag.decompose()
    return _WS.sub(" ", soup.get_text(" ")).strip()


_PATTERNS = {
    "10-K": (r"item\s*7\s*\.?\s*management.{0,3}s\s+discussion",
             r"item\s*7a\s*\.?|item\s*8\s*\.?\s*financial\s+statements"),
    "10-Q": (r"item\s*2\s*\.?\s*management.{0,3}s\s+discussion",
             r"item\s*3\s*\.?\s*quantitative|item\s*4\s*\.?\s*controls"),
}


def extract_mdna(text: str, form: str, min_chars: int = 3000) -> str:
    """Return the longest span between an MD&A heading and the next item heading.

    The table of contents also matches the heading, so we take the longest span.
    Falls back to the full text if nothing sensible is found.
    """
    key = "10-K" if form and form.upper().startswith("10-K") else "10-Q"
    if not form or not form.upper().startswith(("10-K", "10-Q")):
        return text
    start_pat, end_pat = (re.compile(p, re.I) for p in _PATTERNS[key])
    starts = [m.start() for m in start_pat.finditer(text)]
    ends = [m.start() for m in end_pat.finditer(text)]
    best = ""
    for s in starts:
        nxt = next((e for e in ends if e > s + 200), None)
        span = text[s:nxt] if nxt else text[s:]
        if len(span) > len(best):
            best = span
    return best if len(best) >= min_chars else text


def truncate(text: str, max_chars: int) -> str:
    """Keep head and tail (guidance language often sits near the end)."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    return text[:head] + "\n[...truncated...]\n" + text[-(max_chars - head):]
