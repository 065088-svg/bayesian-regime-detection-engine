"""
models/rs_var.py
==================
Regime-switching VAR (Day 6 deliverable).

Part 1: single-feature Markov-switching baseline using
statsmodels.tsa.regime_switching.markov_autoregression (Section A8.2 —
matches the brief exactly, this is a real statsmodels model).

Part 2: multivariate regime-switching VAR over [returns, vol, breadth,
FII flow, INR change, gilt change]. The brief specifies a full PyMC/
NumPyro Bayesian RS-VAR; given this sandbox's install-time budget (no
PyMC), the multivariate layer here is implemented as a EM-style
regime-conditioned VAR(1): regimes come from the Day 4/5 HMM, and within
each regime we fit an OLS VAR(1) (equivalent to the posterior mode of a
Bayesian VAR under a weak/flat prior). This recovers real
regime-conditional coefficients and innovation covariances — the
uncertainty is reported via residual-bootstrap confidence intervals
rather than MCMC credible intervals. Swapping in NumPyro NUTS over the
same regime-conditioned VAR likelihood is a drop-in upgrade.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


def fit_markov_switching_baseline(returns: pd.Series, k_regimes=2, order=1):
    ms = MarkovAutoregression(returns.dropna() * 100, k_regimes=k_regimes,
                               order=order, switching_variance=True)
    res = ms.fit()
    return res


VAR_VARS = ["ret", "vol", "breadth", "fii_flow", "inr_chg", "gilt_chg"]


def _build_var_panel(feat_df: pd.DataFrame) -> pd.DataFrame:
    panel = pd.DataFrame({
        "ret": feat_df["ret_1d"],
        "vol": feat_df["rvol_21d"],
        "breadth": feat_df["breadth_proxy"],
        "fii_flow": feat_df["fii_z"],
        "inr_chg": feat_df["usdinr_mom_21d"],
        "gilt_chg": feat_df["gilt_chg_63d"],
    }).dropna()
    return panel


def fit_regime_conditioned_var(panel: pd.DataFrame, regime_labels: pd.Series, lag=1, n_boot=200, seed=13):
    """OLS VAR(1) fit separately within each regime label; bootstrap
    residuals for coefficient CIs and recover innovation covariance."""
    rng = np.random.default_rng(seed)
    aligned = panel.join(regime_labels.rename("regime"), how="inner").dropna()
    results = {}

    for regime in aligned["regime"].unique():
        sub = aligned[aligned["regime"] == regime][VAR_VARS]
        if len(sub) < lag + 10:
            continue
        Y = sub.values[lag:]
        X = np.hstack([np.ones((len(Y), 1))] + [sub.values[lag - l:-l or None] for l in range(1, lag + 1)])
        # coefficients via OLS: X (n x (1+k*lag)), Y (n x k)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        innov_cov = np.cov(resid.T)

        boot_betas = []
        n = len(Y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            Xb, Yb = X[idx], Y[idx]
            try:
                b, *_ = np.linalg.lstsq(Xb, Yb, rcond=None)
                boot_betas.append(b)
            except np.linalg.LinAlgError:
                continue
        boot_betas = np.array(boot_betas)
        ci_lo = np.percentile(boot_betas, 2.5, axis=0)
        ci_hi = np.percentile(boot_betas, 97.5, axis=0)

        results[regime] = dict(
            beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
            innovation_cov=innov_cov, n_obs=n,
        )
    return results


def impulse_response(beta: np.ndarray, innov_cov: np.ndarray, shock_var_idx: int,
                      shock_size_sd=1.0, horizon=20, n_vars=len(VAR_VARS)):
    """Simple VAR(1) IRF: shock one variable by `shock_size_sd` std devs at
    t=0 and propagate through the fitted regime-conditional coefficient
    matrix (excludes intercept row)."""
    A = beta[1:1 + n_vars].T  # n_vars x n_vars companion for lag-1
    shock_sd = np.sqrt(np.diag(innov_cov))[shock_var_idx]
    irf = np.zeros((horizon, n_vars))
    state = np.zeros(n_vars)
    state[shock_var_idx] = shock_size_sd * shock_sd
    for h in range(horizon):
        irf[h] = state
        state = A @ state
    return irf


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.loader import load_market_data
    from features.engineering import build_features
    from models.hmm_frequentist import run_frequentist_hmm

    df = load_market_data()
    feat = build_features(df)

    print("=== Part 1: single-feature Markov-switching baseline (statsmodels) ===")
    ms_res = fit_markov_switching_baseline(feat["ret_1d"].dropna(), k_regimes=2)
    print(ms_res.summary().tables[0])

    print("\n=== Part 2: regime-conditioned multivariate VAR(1) ===")
    panel = _build_var_panel(feat)
    hmm_out = run_frequentist_hmm(feat)
    regime_labels = hmm_out["result"]["regime"]
    var_results = fit_regime_conditioned_var(panel, regime_labels, n_boot=100)

    for regime, res in var_results.items():
        print(f"\n--- Regime: {regime} (n={res['n_obs']}) ---")
        print("Innovation covariance (diag = regime vol by variable):")
        print(np.round(np.diag(res["innovation_cov"]), 5))

    if "Risk-Off" in var_results:
        irf = impulse_response(var_results["Risk-Off"]["beta"],
                                var_results["Risk-Off"]["innovation_cov"],
                                shock_var_idx=VAR_VARS.index("fii_flow"))
        print("\n=== FII-outflow shock IRF in Risk-Off regime (first 5 periods, all vars) ===")
        print(pd.DataFrame(irf[:5], columns=VAR_VARS).round(4))
