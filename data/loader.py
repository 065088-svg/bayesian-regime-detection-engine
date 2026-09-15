"""
data/loader.py
================
Data ingestion layer for the Bayesian Regime Detection Engine.

MAJOR UPDATE (this pass): the core price series is now REAL data.

`data/real/nifty50_real_daily.csv` is real Nifty 50 daily OHLC, sourced
from a public GitHub dataset (kalilurrahman/NIFTY_50_STOCK_DATA,
NIFTY50_stock_history.csv) and spot-checked against the documented
historical record — e.g. the 2020 Covid-crash closing low in this file
is 7,610.25 on 2020-03-23, which matches the widely-reported real Nifty
50 low for that crash. Coverage: 2007-09-17 to 2026-04-13 (4,554 trading
days, no gaps beyond weekends/holidays), spanning all four crisis
episodes used in the case-study section (2008 GFC, 2013 taper tantrum,
2020 Covid crash, 2024 election volatility).

WHAT IS STILL SYNTHETIC: India VIX, USD/INR, 10Y Gilt yield, FII/DII net
flows, SIP totals, and the Midcap/Smallcap proxies have NO real public
source found in this sandbox (no network path to NSE/AMFI/RBI, and no
GitHub-hosted dataset located for these series specifically). These
remain regime-conditioned synthetic series, generated in a way that is
now *derived from the real Nifty regime path* (real drawdowns and real
realized volatility drive the synthetic auxiliary series' regime
switching) rather than fully independently simulated as in the prior
pass — this makes the auxiliary series internally consistent with the
real price history, but they are still not real historical data and
must not be reported as such.

Net effect: every number that depends on price, returns, or realized
volatility alone (HMM features `ret_*`, `rvol_*`, `ma*_dist`, the
regime-taxonomy classification itself, the case-study replay, the
backtest) is now computed on real market history. Every number that
depends on VIX, FII/DII, INR, or Gilt levels is still a demonstration on
synthetic data.
"""
import os
import numpy as np
import pandas as pd

REAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "real", "nifty50_real_daily.csv")

SCHEMA_COLUMNS = [
    "date", "nifty_close", "midcap_close", "smallcap_close",
    "india_vix", "usdinr", "gilt10y", "fii_net_cr", "dii_net_cr",
    "sip_total_cr", "regime_true",
]

REGIME_NAMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]

# Auxiliary-series parameters, now keyed by a REALIZED (real, not fitted)
# drawdown/vol regime rather than a simulated latent state -- see
# `_classify_realized_regime` below.
AUX_PARAMS = {
    "Risk-On":      dict(vix=13.5, fii=350, dii=150),
    "Late-Cycle":   dict(vix=16.5, fii=100, dii=300),
    "Transitional": dict(vix=20.0, fii=-150, dii=400),
    "Post-Shock":   dict(vix=28.0, fii=-900, dii=900),
    "Risk-Off":     dict(vix=24.0, fii=-500, dii=700),
}

# REGIME_PARAMS: kept for backward compatibility with models/monte_carlo.py's
# regime_conditioned_monte_carlo(), which needs a (mu, sigma) per regime
# index to simulate forward paths. Prior pass hard-coded these; this pass
# ESTIMATES them from the real Nifty return series, conditioned on the
# realized-regime classification below -- i.e. these are now real,
# empirically-measured regime statistics, not assumed parameters.
# Computed at the bottom of this module, once the needed functions exist.


def load_real_nifty_prices() -> pd.DataFrame:
    """Real Nifty 50 daily close, 2007-09-17 to 2026-04-13. See module
    docstring for source and validation notes."""
    df = pd.read_csv(REAL_DATA_PATH, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "Close": "nifty_close"})
    df = df[["date", "nifty_close"]].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def _classify_realized_regime(price: pd.Series) -> pd.Series:
    """A simple, transparent realized-regime classifier used ONLY to drive
    the synthetic auxiliary series' parameters -- NOT the model's own
    regime call (that's `models/ground_truth.py` / the HMM). Based on
    trailing 60-day return and drawdown from a 60-day peak, on the REAL
    price series."""
    trailing_60d = price.pct_change(60)
    roll_peak = price.rolling(60).max()
    dd = price / roll_peak - 1
    post_shock = (dd <= -0.15).rolling(15, min_periods=1).max().astype(bool)

    regime = pd.Series("Transitional", index=price.index)
    regime[trailing_60d >= 0.08] = "Late-Cycle"
    regime[(trailing_60d >= 0.08) & (trailing_60d.rolling(21).std() <= trailing_60d.rolling(252).std())] = "Risk-On"
    regime[trailing_60d <= -0.08] = "Risk-Off"
    regime[post_shock] = "Post-Shock"
    return regime


def _fit_regime_params_from_real_data():
    """Real, empirically-measured (mean, std) of daily Nifty returns
    within each realized regime -- replaces the prior pass's assumed
    REGIME_PARAMS with parameters actually estimated from market history."""
    real = load_real_nifty_prices()
    ret = real["nifty_close"].pct_change()
    regime = _classify_realized_regime(real["nifty_close"])
    params = {}
    for i, name in enumerate(REGIME_NAMES):
        mask = regime == name
        r = ret[mask].dropna()
        if len(r) > 5:
            params[i] = dict(mu=float(r.mean()), sigma=float(r.std()))
        else:
            params[i] = dict(mu=0.0, sigma=0.015)
    return params


REGIME_PARAMS = _fit_regime_params_from_real_data()


def generate_market_panel(seed=42) -> pd.DataFrame:
    """Real Nifty price/returns + regime-consistent synthetic auxiliary
    series. See module docstring for exactly what is real vs. synthetic."""
    rng = np.random.default_rng(seed)
    real = load_real_nifty_prices()
    n = len(real)

    realized_regime = _classify_realized_regime(real["nifty_close"])
    regime_idx = realized_regime.map({n: i for i, n in enumerate(REGIME_NAMES)}).values

    nifty_ret = real["nifty_close"].pct_change().fillna(0).values
    mid_idio = np.array([rng.normal(0, 0.006 if realized_regime.iloc[i] in ("Post-Shock", "Risk-Off") else 0.004)
                          for i in range(n)])
    small_idio = np.array([rng.normal(0, 0.009 if realized_regime.iloc[i] in ("Post-Shock", "Risk-Off") else 0.006)
                            for i in range(n)])
    mid_ret = nifty_ret * 1.1 + mid_idio
    small_ret = nifty_ret * 1.2 + small_idio
    midcap_close = 8000 * np.exp(np.cumsum(mid_ret))
    smallcap_close = 3200 * np.exp(np.cumsum(small_ret))

    vix = np.array([max(9.0, rng.normal(AUX_PARAMS[realized_regime.iloc[i]]["vix"], 2.0)) for i in range(n)])
    fii = np.array([rng.normal(AUX_PARAMS[realized_regime.iloc[i]]["fii"], 400) for i in range(n)])
    dii = np.array([rng.normal(AUX_PARAMS[realized_regime.iloc[i]]["dii"], 300) for i in range(n)])

    usdinr = 45 + np.cumsum(rng.normal(0.0006, 0.003, n))
    gilt10y = np.clip(7.2 + 1.4 * np.sin(np.linspace(0, 9, n)) + rng.normal(0, 0.05, n).cumsum() * 0.02, 5.5, 9.5)
    base_sip = np.linspace(3000, 26000, n)
    sip_dampen = np.where(np.isin(regime_idx, [3, 4]), 0.93, 1.0)
    sip_total = base_sip * sip_dampen * (1 + rng.normal(0, 0.02, n))

    df = pd.DataFrame({
        "date": real["date"],
        "nifty_close": real["nifty_close"].values,          # REAL
        "midcap_close": midcap_close,                         # synthetic, real-factor-driven
        "smallcap_close": smallcap_close,                     # synthetic, real-factor-driven
        "india_vix": vix,                                     # synthetic
        "usdinr": usdinr,                                     # synthetic
        "gilt10y": gilt10y,                                   # synthetic
        "fii_net_cr": fii,                                    # synthetic
        "dii_net_cr": dii,                                    # synthetic
        "sip_total_cr": sip_total,                            # synthetic
        "regime_true": regime_idx,                            # derived from REAL price (realized-regime proxy, not the model's own call)
    })
    return df


# Backwards-compatible alias -- older modules import `generate_synthetic_market`
def generate_synthetic_market(start=None, end=None, seed=42):
    return generate_market_panel(seed=seed)


def load_market_data(cache_path="/home/claude/regime_engine/data/market_panel.parquet",
                      force_regen=False):
    if not force_regen and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = generate_market_panel()
    df.to_parquet(cache_path, index=False)
    return df


if __name__ == "__main__":
    df = load_market_data(force_regen=True)
    print(df.head())
    print(df.tail())
    print(df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print(pd.Series(df["regime_true"]).map(dict(enumerate(REGIME_NAMES))).value_counts())
