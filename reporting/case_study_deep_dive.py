"""
reporting/case_study_deep_dive.py
====================================
Deeper analytics for the standalone case-study report (Day 12
deliverable): for each episode, find the REAL peak-to-trough (not just
the fixed narrative window), measure how many trading days after the
peak the model's regime call flipped to Post-Shock/Risk-Off ("detection
lag"), and measure the recovery time back to the pre-crisis peak.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from data.loader import load_market_data
from features.engineering import build_features
from models.hmm_frequentist import run_frequentist_hmm

REGIME_NAMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
STRESS_REGIMES = {"Post-Shock", "Risk-Off"}

EPISODES = [
    dict(name="2008 Global Financial Crisis", search_start="2007-12-01", search_end="2009-06-30"),
    dict(name="2013 Taper Tantrum", search_start="2013-04-01", search_end="2013-12-31"),
    dict(name="2020 Covid Crash", search_start="2020-01-01", search_end="2020-12-31"),
    dict(name="2024 Election/Budget Volatility", search_start="2024-04-01", search_end="2024-12-31"),
]


def analyze_episode(price: pd.Series, regime: pd.Series, ep):
    window = price.loc[ep["search_start"]:ep["search_end"]]
    if len(window) == 0:
        return dict(name=ep["name"], status="no_data")

    running_max = window.cummax()
    dd = window / running_max - 1
    trough_date = dd.idxmin()
    peak_date = window.loc[:trough_date].idxmax()
    peak_price = window.loc[peak_date]
    trough_price = window.loc[trough_date]
    pct_decline = trough_price / peak_price - 1

    # Trading days peak-to-trough: computed from the PRICE index directly
    # (always available), not the regime index (which starts later due to
    # the 252-day feature warmup -- see caveat below).
    price_span = price.loc[peak_date:trough_date]
    trading_days_peak_to_trough = len(price_span) - 1

    # Detection lag: only meaningful over the window where regime data
    # actually exists. If the regime series doesn't cover the peak date
    # (feature warmup), this is reported explicitly rather than silently
    # producing a misleadingly short "lag".
    regime_start = regime.index.min()
    warmup_gap_days = None
    if peak_date < regime_start:
        warmup_gap_days = int((price.loc[peak_date:regime_start].shape[0]) - 1)

    post_peak_regime = regime.loc[max(peak_date, regime_start):trough_date]
    stress_days = post_peak_regime[post_peak_regime.isin(STRESS_REGIMES)]
    detection_lag = None
    if len(stress_days) > 0:
        first_stress_date = stress_days.index[0]
        # lag measured from the peak in PRICE trading days, not from
        # wherever the regime series happens to start
        detection_lag = int(price.loc[peak_date:first_stress_date].shape[0] - 1)

    post_trough = price.loc[trough_date:]
    recovered = post_trough[post_trough >= peak_price]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None
    recovery_days = (len(post_trough.loc[:recovery_date]) - 1) if recovery_date is not None else None

    return dict(
        name=ep["name"], status="ok",
        peak_date=str(peak_date.date()), trough_date=str(trough_date.date()),
        peak_price=round(float(peak_price), 1), trough_price=round(float(trough_price), 1),
        pct_decline=round(float(pct_decline), 4),
        trading_days_peak_to_trough=trading_days_peak_to_trough,
        regime_data_covers_peak=bool(peak_date >= regime_start),
        warmup_gap_trading_days=warmup_gap_days,
        detection_lag_days=detection_lag,
        recovery_date=str(recovery_date.date()) if recovery_date is not None else "not yet recovered in data",
        recovery_days_from_trough=recovery_days,
    )


if __name__ == "__main__":
    df = load_market_data()
    feat = build_features(df)
    hmm_out = run_frequentist_hmm(feat)
    price = df.set_index("date")["nifty_close"]
    regime = hmm_out["result"]["regime"]

    results = []
    for ep in EPISODES:
        r = analyze_episode(price, regime, ep)
        results.append(r)
        print(r)

    pd.DataFrame(results).to_csv("outputs/validation_pack/case_study_deep_dive.csv", index=False)
    print("\nWrote outputs/validation_pack/case_study_deep_dive.csv")
