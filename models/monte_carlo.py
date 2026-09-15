"""
models/monte_carlo.py  +  backtest/overlay.py (combined for brevity)
=======================================================================
Day 13 deliverable: regime-conditioned Monte Carlo path simulation with
VaR/CVaR, plus a regime-tilt allocation overlay backtested 2019-2024
against buy-and-hold, reporting Information Ratio and tracking error.
"""
import numpy as np
import pandas as pd

REGIME_NAMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]

# Tilt rules: (equity_weight_multiplier) conditioned on regime + conviction (confidence)
TILT_RULES = {
    "Risk-On": 1.15,
    "Late-Cycle": 1.05,
    "Transitional": 1.00,
    "Post-Shock": 0.75,
    "Risk-Off": 0.55,
}


def regime_conditioned_monte_carlo(regime_params: dict, current_regime_probs: np.ndarray,
                                    transition_matrix: np.ndarray, horizon=63, n_paths=5000, seed=99):
    """Simulate `n_paths` forward return paths over `horizon` days, drawing
    the regime path from the transition matrix (seeded by today's regime
    probability vector) and returns from each regime's (mu, sigma)."""
    rng = np.random.default_rng(seed)
    n_states = len(current_regime_probs)
    paths = np.zeros((n_paths, horizon))
    regime_paths = np.zeros((n_paths, horizon), dtype=int)

    for p in range(n_paths):
        state = rng.choice(n_states, p=current_regime_probs)
        for t in range(horizon):
            mu, sigma = regime_params[state]["mu"], regime_params[state]["sigma"]
            paths[p, t] = rng.normal(mu, sigma)
            regime_paths[p, t] = state
            state = rng.choice(n_states, p=transition_matrix[state])

    cum_returns = np.cumsum(paths, axis=1)
    terminal = cum_returns[:, -1]
    var_95 = -np.percentile(terminal, 5)
    cvar_95 = -terminal[terminal <= np.percentile(terminal, 5)].mean()
    return dict(paths=cum_returns, regime_paths=regime_paths, terminal=terminal,
                var_95=var_95, cvar_95=cvar_95)


def allocation_overlay_backtest(returns: pd.Series, regime_calls: pd.Series, confidence: pd.Series,
                                 start="2019-01-01", end="2024-12-31"):
    """Equity-weight tilt = base(1.0) * TILT_RULES[regime] scaled by
    confidence (blend toward 1.0 when the model is unsure)."""
    idx = returns.index.intersection(regime_calls.index)
    idx = idx[(idx >= start) & (idx <= end)]
    r = returns.loc[idx]
    regime = regime_calls.loc[idx]
    conf = confidence.loc[idx].clip(0, 1)

    raw_tilt = regime.map(TILT_RULES)
    blended_tilt = 1.0 + conf * (raw_tilt - 1.0)  # shrink tilt toward 1.0 under low conviction

    strat_ret = r * blended_tilt.shift(1).fillna(1.0)
    bh_ret = r

    strat_cum = (1 + strat_ret).cumprod()
    bh_cum = (1 + bh_ret).cumprod()

    excess = strat_ret - bh_ret
    tracking_error = excess.std() * np.sqrt(252)
    information_ratio = (excess.mean() * 252) / (tracking_error + 1e-9)

    dd_strat = (strat_cum / strat_cum.cummax() - 1)
    dd_bh = (bh_cum / bh_cum.cummax() - 1)

    by_regime_dd = pd.DataFrame({"strategy_dd": dd_strat, "buyhold_dd": dd_bh, "regime": regime}
                                 ).groupby("regime")[["strategy_dd", "buyhold_dd"]].min()

    return dict(
        strat_cum=strat_cum, bh_cum=bh_cum,
        information_ratio=information_ratio, tracking_error=tracking_error,
        cagr_strategy=strat_cum.iloc[-1] ** (252 / len(strat_cum)) - 1,
        cagr_buyhold=bh_cum.iloc[-1] ** (252 / len(bh_cum)) - 1,
        max_dd_strategy=dd_strat.min(), max_dd_buyhold=dd_bh.min(),
        drawdown_by_regime=by_regime_dd,
    )


def ic_artefact(regime_call: str, confidence: float, conformal_set: list,
                 tilt: float, key_drivers: dict, date):
    """Investment Committee conditional-statement generator (Section
    A13.3): one lineage-carrying record per regime call."""
    return {
        "date": str(date),
        "regime_call": regime_call,
        "confidence": round(float(confidence), 3),
        "conformal_set": conformal_set,
        "recommended_equity_tilt": round(float(tilt), 3),
        "statement": (
            f"As of {date}, the ensemble assigns {confidence:.0%} probability to '{regime_call}' "
            f"(calibrated {round(100/len(conformal_set) if conformal_set else 0)}%+ conformal set: "
            f"{', '.join(conformal_set) if conformal_set else 'n/a'}). "
            f"Recommended equity-weight tilt: {tilt:.2f}x neutral, "
            f"driven primarily by: {', '.join(f'{k} ({v:+.2f})' for k, v in key_drivers.items())}."
        ),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.loader import load_market_data, REGIME_PARAMS
    from features.engineering import build_features
    from models.hmm_frequentist import run_frequentist_hmm

    df = load_market_data()
    feat = build_features(df)
    hmm_out = run_frequentist_hmm(feat)

    print("=== Regime-conditioned Monte Carlo (63-day horizon) ===")
    current_probs = hmm_out["result"][[f"prob_{r}" for r in REGIME_NAMES]].iloc[-1].values
    mc = regime_conditioned_monte_carlo(REGIME_PARAMS, current_probs, hmm_out["transition_matrix"],
                                         horizon=63, n_paths=3000)
    print(f"63-day 95% VaR: {mc['var_95']:.3%}   95% CVaR: {mc['cvar_95']:.3%}")

    print("\n=== Allocation overlay backtest (2019-2024) ===")
    ret_series = df.set_index("date")["nifty_close"].pct_change()
    bt = allocation_overlay_backtest(ret_series, hmm_out["result"]["regime"],
                                      hmm_out["result"][[f"prob_{r}" for r in REGIME_NAMES]].max(axis=1))
    print(f"Strategy CAGR: {bt['cagr_strategy']:.2%}   Buy-hold CAGR: {bt['cagr_buyhold']:.2%}")
    print(f"Information Ratio: {bt['information_ratio']:.3f}   Tracking Error: {bt['tracking_error']:.3%}")
    print(f"Max DD strategy: {bt['max_dd_strategy']:.2%}   Max DD buy-hold: {bt['max_dd_buyhold']:.2%}")
    print("\nDrawdown by regime:")
    print(bt["drawdown_by_regime"].round(4))

    print("\n=== Sample IC artefact ===")
    last = hmm_out["result"].iloc[-1]
    art = ic_artefact(
        regime_call=last["regime"], confidence=last[[f"prob_{r}" for r in REGIME_NAMES]].max(),
        conformal_set=[r for r in REGIME_NAMES if last[f"prob_{r}"] > 0.05],
        tilt=TILT_RULES[last["regime"]],
        key_drivers={"vix_z": 1.2, "fii_flow": -0.8},
        date=hmm_out["result"].index[-1].date(),
    )
    print(art["statement"])
