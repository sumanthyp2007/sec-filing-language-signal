"""Run the full analysis for all horizons and write charts + results.json + results.md."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import analysis, config  # noqa: E402


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run_all(sig, events, paths, out_dir, split_date, cost_bps=10.0, label="", horizons=config.HORIZONS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = dict(label=label, split_date=str(split_date), cost_bps=cost_bps, horizons={})
    zcols = [f"z_{f}" for f in config.FEATURES]
    main_h = max(horizons)
    for h in horizons:
        panel = analysis.make_panel(sig, events, h)
        r = dict(n_events=int(len(panel)))
        if len(panel) < 30:
            results["horizons"][h] = {**r, "note": "too few events"}
            continue
        q = analysis.quintile_returns(panel, "signal_prior", cost_bps=cost_bps)
        r["quantiles"] = q.get("stats", {})
        r["ic"] = analysis.ic_stats(analysis.ic_series(panel, "signal_prior"))
        r["regression_prior"] = analysis.regress(panel, ["signal_prior"]).round(6).to_dict("index")
        r["regression_features"] = analysis.regress(panel, zcols).round(6).to_dict("index")
        r["oos"] = analysis.oos_evaluation(panel, split_date, cost_bps=cost_bps)
        results["horizons"][h] = r
        if h == main_h:
            _charts(panel, q, paths, out_dir, h, zcols)
    (out_dir / "results.json").write_text(json.dumps(_clean(results), indent=2))
    (out_dir / "results.md").write_text(_markdown(results))
    return results


def _charts(panel, q, paths, out_dir, h, zcols):
    if q.get("stats"):
        m = q["stats"]["mean_by_quantile"]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.bar(list(m.keys()), [v * 100 for v in m.values()], color="#2c5d8f")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Signal quintile (1 = most negative, 5 = most positive)")
        ax.set_ylabel(f"Mean {h}-day CAR (%)")
        ax.set_title("Abnormal return by signal quintile")
        fig.tight_layout(); fig.savefig(out_dir / "quintile_car.png", dpi=150); plt.close(fig)

        lab = analysis.assign_quintiles(panel, "signal_prior")
        d = panel.assign(q=lab).dropna(subset=["q"]).set_index("accession")
        pp = paths.reindex(d.index).dropna()
        if len(pp):
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            for k, col in ((5, "#1b7f3b"), (1, "#b22222")):
                sel = pp.loc[d.loc[pp.index, "q"] == k]
                if len(sel):
                    ax.plot(pp.columns, sel.mean().cumsum() * 100, label=f"Q{k}", color=col)
            ax.axvline(0, color="grey", ls="--", lw=0.8)
            ax.set_xlabel("Trading days from reaction day"); ax.set_ylabel("Cumulative AR (%)")
            ax.set_title("Event-time path: top vs bottom quintile"); ax.legend()
            fig.tight_layout(); fig.savefig(out_dir / "event_path.png", dpi=150); plt.close(fig)

    ic = analysis.ic_series(panel, "signal_prior")
    if len(ic):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(range(len(ic)), ic.values, color=np.where(ic.values >= 0, "#2c5d8f", "#b22222"))
        ax.set_xticks(range(len(ic))); ax.set_xticklabels(ic.index, rotation=90, fontsize=6)
        ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Spearman IC")
        ax.set_title(f"Information coefficient by quarter ({h}-day CAR)")
        fig.tight_layout(); fig.savefig(out_dir / "ic_by_period.png", dpi=150); plt.close(fig)

    reg = analysis.regress(panel, zcols)
    reg = reg.drop(index="const")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ypos = np.arange(len(reg))
    ax.errorbar(reg["coef"] * 100, ypos, xerr=[(reg["coef"] - reg["ci_lo"]) * 100, (reg["ci_hi"] - reg["coef"]) * 100],
                fmt="o", color="#2c5d8f", capsize=3)
    ax.set_yticks(ypos); ax.set_yticklabels([i[2:] for i in reg.index])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(f"CAR change per 1 s.d. of feature (%), {h}-day")
    ax.set_title("Feature coefficients (95% CI, clustered by quarter)")
    fig.tight_layout(); fig.savefig(out_dir / "feature_coefs.png", dpi=150); plt.close(fig)


def _f(x, pct=False, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.{d}f}%" if pct else f"{x:.{d}f}"


def _markdown(res) -> str:
    L = ["# Results", ""]
    if res.get("label"):
        L += [f"> **{res['label']}**", ""]
    L += [f"Out-of-sample split: {res['split_date']} | cost assumption: {res['cost_bps']} bps per leg round trip", ""]
    L += ["| Horizon | Events | IC (mean) | IC t | LS gross | LS t | LS net | OOS prior IC | OOS fitted IC |",
          "|---|---|---|---|---|---|---|---|---|"]
    for h, r in res["horizons"].items():
        if "quantiles" not in r:
            L.append(f"| {h}d | {r['n_events']} | too few events | | | | | | |")
            continue
        q, ic, o = r["quantiles"], r["ic"], r.get("oos", {})
        L.append(f"| {h}d | {r['n_events']} | {_f(ic.get('mean_ic'))} | {_f(ic.get('ic_t'))} | "
                 f"{_f(q.get('ls_mean'), True)} | {_f(q.get('ls_t'))} | {_f(q.get('ls_net_mean'), True)} | "
                 f"{_f((o.get('signal_prior') or {}).get('mean_ic'))} | {_f((o.get('signal_fit') or {}).get('mean_ic'))} |")
    L += ["", "LS = Q5 minus Q1 mean cumulative abnormal return per formation quarter. t-stats are across quarters.",
          "IC = Spearman rank correlation between signal and abnormal return, per quarter.", ""]
    return "\n".join(L)
