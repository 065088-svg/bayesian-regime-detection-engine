"""
reporting/case_study_charts.py
=================================
Price + regime-overlay charts for each of the 4 case-study episodes,
embedded in the standalone case-study report.
"""
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from data.loader import load_market_data
from features.engineering import build_features
from models.hmm_frequentist import run_frequentist_hmm

REGIME_COLORS = {
    "Risk-On": "#2fbf71", "Late-Cycle": "#7ec8e3", "Transitional": "#e3b23c",
    "Post-Shock": "#e35d5d", "Risk-Off": "#8a4fff",
}

CHARTS = [
    dict(name="2008 Global Financial Crisis", plot_start="2007-10-01", plot_end="2009-12-31",
         peak="2008-01-08", trough="2008-10-27", fname="chart_2008.png"),
    dict(name="2013 Taper Tantrum", plot_start="2013-03-01", plot_end="2013-12-31",
         peak="2013-05-17", trough="2013-08-28", fname="chart_2013.png"),
    dict(name="2020 Covid Crash", plot_start="2019-11-01", plot_end="2020-12-31",
         peak="2020-01-14", trough="2020-03-23", fname="chart_2020.png"),
    dict(name="2024 Q4 Correction", plot_start="2024-07-01", plot_end="2025-06-30",
         peak="2024-09-26", trough="2024-11-21", fname="chart_2024.png"),
]


def make_chart(price, regime, cfg, outdir):
    window_price = price.loc[cfg["plot_start"]:cfg["plot_end"]]
    window_regime = regime.reindex(window_price.index)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    # shade regime backgrounds
    prev_date = window_regime.index[0]
    prev_r = window_regime.iloc[0]
    for date, r in window_regime.items():
        if pd.isna(r):
            r = "no data"
        if r != prev_r:
            if prev_r in REGIME_COLORS:
                ax.axvspan(prev_date, date, color=REGIME_COLORS[prev_r], alpha=0.18, lw=0)
            prev_date, prev_r = date, r
    if prev_r in REGIME_COLORS:
        ax.axvspan(prev_date, window_regime.index[-1], color=REGIME_COLORS[prev_r], alpha=0.18, lw=0)

    ax.plot(window_price.index, window_price.values, color="#1E2761", linewidth=1.4)
    ax.axvline(pd.Timestamp(cfg["peak"]), color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(pd.Timestamp(cfg["trough"]), color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate("Peak", (pd.Timestamp(cfg["peak"]), window_price.max()),
                textcoords="offset points", xytext=(2, 0), fontsize=8)
    ax.annotate("Trough", (pd.Timestamp(cfg["trough"]), window_price.max()),
                textcoords="offset points", xytext=(2, 0), fontsize=8)

    ax.set_title(f"{cfg['name']}: Nifty 50 price with model regime overlay", fontsize=10)
    ax.set_ylabel("Nifty 50 close", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{cfg['fname']}", dpi=140)
    plt.close(fig)
    print(f"Wrote {outdir}/{cfg['fname']}")


if __name__ == "__main__":
    import os
    outdir = "reporting"
    os.makedirs(outdir, exist_ok=True)

    df = load_market_data()
    feat = build_features(df)
    hmm_out = run_frequentist_hmm(feat)
    price = df.set_index("date")["nifty_close"]
    regime = hmm_out["result"]["regime"]

    for cfg in CHARTS:
        make_chart(price, regime, cfg, outdir)
