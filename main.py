"""
main.py
========
Runs the full pipeline end-to-end (Day 14 "integration" deliverable) and
dumps the artifacts the dashboard and reports read:
  outputs/regime_timeline.csv     - date, regime probs, ensemble call, confidence
  outputs/transition_matrix.json
  outputs/model_comparison.csv
  outputs/backtest_summary.json
  outputs/mc_var.json
  outputs/calibration.json
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from data.loader import load_market_data, REGIME_PARAMS
from features.engineering import build_features, FEATURE_COLUMNS
from models.hmm_frequentist import run_frequentist_hmm
from models.hmm_bayesian import run_bayesian_hmm
from models.ground_truth import rules_based_regime_labels
from models.bayesian_dl import MCDropoutMLP, DeepEnsemble
from models.ensemble import bayesian_model_averaging, constrained_stacking, oof_log_likelihood, waic_style_comparison, combined_output_contract
from models.conformal import split_conformal_sets, expected_calibration_error, rolling_conformal_coverage
from models.monte_carlo import regime_conditioned_monte_carlo, allocation_overlay_backtest, TILT_RULES

REGIME_NAMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]


def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    print("[1/9] Loading market data...")
    df = load_market_data()
    price = df.set_index("date")["nifty_close"]

    print("[2/9] Building features...")
    feat = build_features(df)

    print("[3/9] Fitting frequentist HMM...")
    hmm_out = run_frequentist_hmm(feat)

    print("[4/9] Fitting Bayesian HMM (conjugate posterior)...")
    bayes_out = run_bayesian_hmm(feat, n_draws=300)

    print("[5/9] Building independent (rules-based) regime ground-truth...")
    rb_labels = rules_based_regime_labels(feat, price)
    hmm_vs_rb_agreement = float((hmm_out["result"]["regime"].reindex(rb_labels.index) == rb_labels).mean())

    print("[6/9] Training Bayesian deep learning members against the INDEPENDENT label...")
    y_map = {n: i for i, n in enumerate(REGIME_NAMES)}
    Xy = feat[FEATURE_COLUMNS].join(rb_labels.rename("regime")).dropna()
    X = Xy[FEATURE_COLUMNS].values
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = Xy["regime"].map(y_map).values
    n = len(Xn); split = int(n * 0.7)
    Xtr, ytr, Xte, yte = Xn[:split], y[:split], Xn[split:], y[split:]

    mc_model = MCDropoutMLP(Xn.shape[1]).fit(Xtr, ytr, epochs=200)
    mc_pred = mc_model.predict_mc(Xte, T=30)
    ens_model = DeepEnsemble(Xn.shape[1], M=6).fit(Xtr, ytr, epochs=120)
    ens_pred = ens_model.predict(Xte)

    print("[7/9] Ensembling (BMA + constrained stacking) against the INDEPENDENT label...")
    hmm_probs_aligned = hmm_out["result"][[f"prob_{r}" for r in REGIME_NAMES]].reindex(Xy.index).ffill().bfill()
    member_probs = {
        "HMM": hmm_probs_aligned.values[split:],
        "MC-Dropout": mc_pred["mean"],
        "DeepEnsemble": ens_pred["mean"],
    }
    member_oof_ll = {k: oof_log_likelihood(v, yte) for k, v in member_probs.items()}
    bma_probs, bma_w = bayesian_model_averaging(member_probs, member_oof_ll)
    stack_probs, stack_w = constrained_stacking(member_probs, yte)
    comparison = waic_style_comparison(
        {**member_probs, "BMA": bma_probs, "Stack": stack_probs}, yte,
        n_params=dict(HMM=30, **{"MC-Dropout": 500, "DeepEnsemble": 3000, "BMA": 0, "Stack": 3}))
    best_name = comparison.iloc[0]["model"]
    best_probs = {"BMA": bma_probs, "Stack": stack_probs}.get(best_name, bma_probs)

    print("[8/9] Conformal calibration + Monte Carlo + backtest...")
    n_te = len(yte)
    cal_n = n_te // 2
    p_cal, y_cal = best_probs[:cal_n], yte[:cal_n]
    p_eval, y_eval = best_probs[cal_n:], yte[cal_n:]
    sets, qhat = split_conformal_sets(p_cal, y_cal, p_eval, alpha=0.1)
    ece = expected_calibration_error(p_eval, y_eval)

    current_probs = hmm_out["result"][[f"prob_{r}" for r in REGIME_NAMES]].iloc[-1].values
    mc_sim = regime_conditioned_monte_carlo(REGIME_PARAMS, current_probs, hmm_out["transition_matrix"],
                                             horizon=63, n_paths=3000)

    ret_series = df.set_index("date")["nifty_close"].pct_change()
    bt = allocation_overlay_backtest(ret_series, hmm_out["result"]["regime"],
                                      hmm_out["result"][[f"prob_{r}" for r in REGIME_NAMES]].max(axis=1))

    print("[9/9] Writing artifacts...")
    contract = combined_output_contract(Xy.index[split:][cal_n:], p_eval, REGIME_NAMES)
    contract.to_csv("outputs/regime_timeline.csv")

    hmm_out["result"].to_csv("outputs/full_hmm_regime_history.csv")
    rb_labels.to_frame("rules_based_regime").to_csv("outputs/rules_based_labels.csv")

    with open("outputs/transition_matrix.json", "w") as f:
        json.dump(dict(
            matrix=hmm_out["transition_matrix"].tolist(),
            regime_names=REGIME_NAMES,
            durations=hmm_out["durations"],
            bayesian_ci_width_mean=float((bayes_out["transition_ci_hi"] - bayes_out["transition_ci_lo"]).mean()),
        ), f, indent=2)

    comparison.to_csv("outputs/model_comparison.csv", index=False)

    with open("outputs/backtest_summary.json", "w") as f:
        json.dump(dict(
            information_ratio=float(bt["information_ratio"]),
            tracking_error=float(bt["tracking_error"]),
            cagr_strategy=float(bt["cagr_strategy"]),
            cagr_buyhold=float(bt["cagr_buyhold"]),
            max_dd_strategy=float(bt["max_dd_strategy"]),
            max_dd_buyhold=float(bt["max_dd_buyhold"]),
            tilt_rules=TILT_RULES,
        ), f, indent=2)

    with open("outputs/mc_var.json", "w") as f:
        json.dump(dict(var_95=float(mc_sim["var_95"]), cvar_95=float(mc_sim["cvar_95"]),
                        horizon_days=63, n_paths=3000), f, indent=2)

    with open("outputs/calibration.json", "w") as f:
        json.dump(dict(
            ensemble_used=best_name, qhat=float(qhat),
            avg_conformal_set_size=float(sets.sum(1).mean()),
            empirical_coverage=float(np.mean([sets[i, y_eval[i]] for i in range(len(y_eval))])),
            ece=float(ece),
            bma_weights={k: float(v) for k, v in bma_w.items()},
            stacking_weights={k: float(v) for k, v in stack_w.items()},
            ground_truth="rules_based_independent",
            hmm_vs_rules_based_agreement=hmm_vs_rb_agreement,
        ), f, indent=2)

    (bt["strat_cum"].rename("strategy").to_frame()
     .join(bt["bh_cum"].rename("buyhold"))).to_csv("outputs/backtest_curves.csv")

    print("\nDone. Artifacts written to outputs/")
    print(f"Best ensemble: {best_name} | IR={bt['information_ratio']:.3f} | ECE={ece:.4f} | "
          f"Coverage={np.mean([sets[i, y_eval[i]] for i in range(len(y_eval))]):.3f}")


if __name__ == "__main__":
    main()
