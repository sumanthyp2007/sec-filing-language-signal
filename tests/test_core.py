import numpy as np
import pandas as pd
import pytest

from nlp_signal import analysis, config, db, demo, event_study, extract, pipeline, signals, textproc


def test_reaction_day_rules():
    days = pd.bdate_range("2024-01-01", "2024-01-31")
    # Wed 2024-01-10 08:00 ET (=13:00 UTC) pre-market -> same day
    p = event_study.reaction_pos("2024-01-10T13:00:00Z", "2024-01-10", days)
    assert days[p] == pd.Timestamp("2024-01-10")
    # Wed 2024-01-10 17:00 ET (=22:00 UTC) after close -> next trading day
    p = event_study.reaction_pos("2024-01-10T22:00:00Z", "2024-01-10", days)
    assert days[p] == pd.Timestamp("2024-01-11")
    # Saturday filing -> Monday
    p = event_study.reaction_pos("2024-01-13T15:00:00Z", "2024-01-13", days)
    assert days[p] == pd.Timestamp("2024-01-15")
    # Unknown time -> conservative (next day)
    p = event_study.reaction_pos(None, "2024-01-10", days)
    assert days[p] == pd.Timestamp("2024-01-11")


def test_event_study_recovers_planted_car():
    rng = np.random.default_rng(0)
    n = 600
    rm = rng.normal(0, 0.01, n)
    ri = 1.2 * rm + rng.normal(0, 0.005, n)
    pos = 400
    ri[pos:pos + 20] += 0.01 / 20 * 20 / 20  # +0.0005/day -> +1% over 20d
    out, path = event_study.event_metrics(ri, rm, pos)
    assert out["car_20"] == pytest.approx(0.01, abs=0.01)
    assert out["beta"] == pytest.approx(1.2, abs=0.15)
    assert out["car_1"] < out["car_20"] + 0.02
    assert len(path) == 26 and not np.isnan(path).any()


def test_event_window_excluded_from_estimation_and_offset():
    rng = np.random.default_rng(1)
    rm = rng.normal(0, 0.01, 600)
    ri = rm + rng.normal(0, 0.004, 600)
    ri[400] += 0.10  # one-day jump on day 0
    a, _ = event_study.event_metrics(ri, rm, 400, offset=0)
    b, _ = event_study.event_metrics(ri, rm, 400, offset=1)
    assert a["car_1"] > 0.08          # captured on day 0
    assert abs(b["car_1"]) < 0.03     # skipped with offset=1


def test_mdna_extraction_picks_body_not_toc():
    toc = "Item 7. Management's Discussion and Analysis 25 Item 7A. Quantitative 40 "
    body = "Item 7. Management's Discussion and Analysis " + ("Revenue grew. " * 600) + " Item 7A. Quantitative"
    out = textproc.extract_mdna(toc + body, "10-K")
    assert "Revenue grew" in out and len(out) > 3000


def test_lexicon_extractor_directional():
    L = extract.LexiconExtractor()
    good = L.extract("We raised our full-year outlook on strong growth and record demand.")
    bad = L.extract("We lowered our guidance amid weak demand, decline and challenging headwinds.")
    assert good["guidance_direction"] == 1 and bad["guidance_direction"] == -1
    assert good["tone_score"] > bad["tone_score"]


def test_llm_json_parser_validates_and_clamps():
    d = extract.parse_llm_json('```json\n{"guidance_provided": true, "guidance_direction": 3, "hedging_score": 4, '
                               '"tone_score": -2, "risk_supply_chain": 1, "risk_fx": 2, "risk_demand": 3, "rationale": "x"}```')
    assert d["guidance_direction"] == 1 and d["tone_score"] == -1
    with pytest.raises(ValueError):
        extract.parse_llm_json('{"guidance_direction": 1}')
    d2 = extract.parse_llm_json('{"guidance_provided": false, "guidance_direction": 1, "hedging_score": 1, '
                                '"tone_score": 0, "risk_supply_chain": 0, "risk_fx": 0, "risk_demand": 0}')
    assert d2["guidance_direction"] == 0


def test_llm_extractor_uses_injected_client():
    class R:  # fake anthropic response
        content = [type("B", (), {"type": "text", "text": '{"guidance_provided": true, "guidance_direction": 1, '
                                  '"hedging_score": 2, "tone_score": 0.5, "risk_supply_chain": 0, "risk_fx": 1, '
                                  '"risk_demand": 2, "rationale": "ok"}'})()]
    calls = {}

    class C:
        class messages:
            @staticmethod
            def create(**kw):
                calls.update(kw)
                return R()
    out = extract.LLMExtractor(model="m", client=C()).extract("text", {"form": "10-Q", "ticker": "X"})
    assert out["tone_score"] == 0.5 and calls["temperature"] == 0 and "X" in calls["messages"][0]["content"]


def test_signals_qoq_diff_and_no_lookahead():
    f = pd.DataFrame({
        "accession": list("abcd"), "ticker": ["A"] * 4, "source": ["periodic"] * 4,
        "form": ["10-Q"] * 4, "filing_date": ["2020-02-01", "2020-05-01", "2020-08-01", "2020-11-01"],
        "acceptance_utc": [None] * 4})
    e = pd.DataFrame({"accession": list("abcd"), "guidance_direction": [0, 1, 0, -1.0],
                      "hedging_score": [1, 2, 4, 4.0], "tone_score": [0, 0, 0, 0.0],
                      "risk_supply_chain": [1, 1, 1, 1.0], "risk_fx": [0, 0, 0, 0.0], "risk_demand": [5, 5, 5, 5.0]})
    s = signals.build_signal_frame(e, f)
    assert s["chg_hedging_score"].isna().iloc[0]
    assert list(s["chg_hedging_score"].iloc[1:]) == [1, 2, 0]
    assert list(s["chg_guidance_direction"]) == [0, 1, 0, -1]


def test_cost_and_quintile_logic():
    rng = np.random.default_rng(3)
    n = 600
    panel = pd.DataFrame({"period": np.repeat([f"P{i}" for i in range(12)], 50),
                          "signal_prior": rng.normal(size=n)})
    panel["car"] = 0.02 * panel["signal_prior"] + rng.normal(0, 0.01, n)
    q = analysis.quintile_returns(panel, cost_bps=10)
    assert q["stats"]["ls_mean"] > 0.03
    assert q["stats"]["ls_net_mean"] == pytest.approx(q["stats"]["ls_mean"] - 0.002)
    assert analysis.ic_stats(analysis.ic_series(panel))["mean_ic"] > 0.8


def _run(effect, tmp_path, seed=7):
    con = db.connect(":memory:")
    returns = demo.make_synthetic(con, n_tickers=60, effect=effect, seed=seed)
    return pipeline.run_analysis(con, returns, "demo", tmp_path, "2019-01-01", label="test")


def test_pipeline_recovers_planted_effect_and_null(tmp_path):
    pos = _run(0.02, tmp_path / "pos")
    h = pos["horizons"][20]
    assert h["ic"]["mean_ic"] > 0.05 and h["quantiles"]["ls_t"] > 3
    assert h["oos"]["signal_prior"]["mean_ic"] > 0.03          # holds out of sample
    assert (tmp_path / "pos" / "quintile_car.png").exists()
    null = _run(0.0, tmp_path / "null")
    assert abs(null["horizons"][20]["quantiles"]["ls_t"]) < 3   # no false positive
