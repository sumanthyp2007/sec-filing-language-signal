"""Central configuration. Override paths/keys with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.getenv("NLP_SIGNAL_HOME", Path.cwd()))
DATA = ROOT / "data"
DB_PATH = Path(os.getenv("NLP_SIGNAL_DB", DATA / "signals.db"))
TEXT_DIR = DATA / "text"
PRICE_DIR = DATA / "prices"
TRANSCRIPT_DIR = DATA / "transcripts"
OUT_DIR = ROOT / "outputs"

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_CHARS = int(os.getenv("MAX_CHARS", "60000"))
PROMPT_VERSION = "v1"

BENCHMARK = "SPY"
HORIZONS = (1, 5, 20)  # trading days, window starts on the reaction day

# Features produced by every extractor
FEATURES = [
    "guidance_direction",  # -1 lowered, 0 none/maintained, +1 raised
    "hedging_score",       # higher = more hedged / uncertain language
    "tone_score",          # higher = more positive tone
    "risk_supply_chain",   # higher = more emphasis
    "risk_fx",
    "risk_demand",
]
# Features that are already a "change" and are NOT differenced quarter-over-quarter
DIRECT_FEATURES = {"guidance_direction"}
# Prior (pre-registered) sign of each feature's expected relation to returns
FEATURE_SIGNS = {
    "guidance_direction": +1,
    "hedging_score": -1,
    "tone_score": +1,
    "risk_supply_chain": -1,
    "risk_fx": -1,
    "risk_demand": -1,
}


def ensure_dirs():
    for p in (DATA, TEXT_DIR, PRICE_DIR, TRANSCRIPT_DIR, OUT_DIR):
        p.mkdir(parents=True, exist_ok=True)
