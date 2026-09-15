"""
reporting/validation_pack.py
==============================
Reliability diagrams + scenario replay harness for the Part C crisis
episodes.

UPDATE (this pass): the underlying price series is now REAL Nifty 50
history (see data/loader.py) spanning 2007-2026, so all four case
studies below now replay the model against the ACTUAL historical price
action for each episode -- not a synthetic stand-in. The historical
facts (dates, real-world magnitude) were already accurate; now the
"model response" and P&L columns are computed on real data too.
"""
import sys, os
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.loader import load_market_data
from features.engineering import build_features, FEATURE_COLUMNS
from models.hmm_frequentist import run_frequentist_hmm
from models.conformal import reliability_diagram, expected_calibration_error
from models.sequential_inference import bayesian_online_changepoint_detection

REGIME_NAMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]

# Real, independently-documented facts about each episode. Model response
# and P&L are now computed on real Nifty 50 price history for the same window.
CASE_STUDIES = [
    dict(name="2008 Global Financial Crisis", window=("2008-09-01", "2009-03-31"),
         fact="Nifty 50 fell roughly 60% peak-to-trough between Jan 2008 and Oct-Nov 2008 "
              "as the Lehman collapse triggered a synchronized global equity selloff and "
              "sharp FII outflows from Indian markets."),
    dict(name="2013 Taper Tantrum", window=("2013-05-01", "2013-09-30"),
         fact="Fed taper signalling in May 2013 triggered sharp INR depreciation (breaching "
              "68/USD) and heavy FII debt/equity outflows from India, alongside a spike in "
              "the 10Y Gilt yield above 9%."),
    dict(name="2020 Covid Crash", window=("2020-02-01", "2020-04-30"),
         fact="Nifty 50 fell about 38% from its Jan 2020 high to the Mar 23 2020 low in "
              "roughly seven weeks, the fastest bear-market decline in the index's history, "
              "followed by an unusually sharp V-shaped recovery."),
    dict(name="2024 Election/Budget Volatility", window=("2024-05-15", "2024-07-31"),
         fact="India VIX spiked sharply around the June 2024 general-election result as "
              "exit-poll expectations diverged from the actual outcome, followed by a rapid "
              "post-Budget stabilisation."),
]


def make_reliability_plots(outdir="outputs/validation_pack"):
    os.makedirs(outdir, exist_ok=True)
    df = load_market_data()
    feat = build_features(df)
    hmm_out = run_frequentist_hmm(feat)

    y_map = {n: i for i, n in enumerate(REGIME_NAMES)}
    Xy = feat[FEATURE_COLUMNS].join(hmm_out["result"]["regime"]).dropna()
    y = Xy["regime"].map(y_map).values
    probs = hmm_out["result"][[f"prob_{r}" for r in REGIME_NAMES]].reindex(Xy.index).values

    conf, acc, count = reliability_diagram(probs, y, n_bins=10)
    ece = expected_calibration_error(probs, y, n_bins=10)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    valid = ~np.isnan(conf)
    ax.bar(conf[valid], acc[valid], width=0.08, alpha=0.7, color="#2fbf71", edgecolor="black",
           label="HMM regime classifier")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability Diagram — HMM Ensemble\nECE = {ece:.4f}")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{outdir}/reliability_diagram_hmm.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {outdir}/reliability_diagram_hmm.png  (ECE={ece:.4f})")
    return ece


def scenario_replay(outdir="outputs/validation_pack"):
    os.makedirs(outdir, exist_ok=True)
    df = load_market_data()
    feat = build_features(df)
    hmm_out = run_frequentist_hmm(feat)
    result = hmm_out["result"]

    rows = []
    for cs in CASE_STUDIES:
        start, end = cs["window"]
        window = result.loc[start:end]
        if len(window) == 0:
            rows.append(dict(episode=cs["name"], **{"model_response": "no data in window (synthetic panel gap)"}))
            continue
        dominant_regime = window["regime"].mode().iloc[0]
        pct_days_stress = (window["regime"].isin(["Post-Shock", "Risk-Off"])).mean()
        worst_day_ret = feat.loc[window.index, "ret_1d"].min()

        # illustrative portfolio P&L: 100 Nifty-linked units through the window
        window_ret = feat.loc[window.index, "ret_1d"].fillna(0)
        pnl_pct = (1 + window_ret).prod() - 1

        # BOCPD firing check within window
        ret_series = feat["ret_21d"].reindex(result.index).ffill()
        window_returns = ret_series.loc[start:end].dropna().values
        cp_prob = bayesian_online_changepoint_detection(window_returns, hazard=1 / 250) if len(window_returns) > 5 else np.array([0])

        rows.append(dict(
            episode=cs["name"], window=f"{start} to {end}",
            documented_fact=cs["fact"],
            dominant_model_regime_call=dominant_regime,
            pct_days_called_stress=round(float(pct_days_stress), 3),
            worst_single_day_return=round(float(worst_day_ret), 4) if pd.notna(worst_day_ret) else None,
            real_window_pnl=round(float(pnl_pct), 4),
            bocpd_max_changepoint_prob=round(float(cp_prob.max()), 4),
        ))

    out = pd.DataFrame(rows)
    out.to_csv(f"{outdir}/scenario_replay.csv", index=False)
    print(f"Wrote {outdir}/scenario_replay.csv")
    return out


if __name__ == "__main__":
    ece = make_reliability_plots()
    replay = scenario_replay()
    print("\n=== Scenario replay summary ===")
    print(replay[["episode", "dominant_model_regime_call", "pct_days_called_stress", "real_window_pnl"]]
          .to_string(index=False))
