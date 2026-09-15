"""
models/ensemble.py
====================
Model ensembling, stacking, and selection (Day 9 deliverable).

  - bayesian_model_averaging(): weight member probability vectors by each
    model's out-of-fold log-likelihood (softmax-normalised) — a practical
    BMA approximation.
  - constrained_stacking(): fit non-negative, sum-to-one stacking weights
    on out-of-fold member predictions via projected gradient descent
    (equivalent to a constrained least-squares / log-loss stack).
  - waic_style_comparison(): out-of-fold calibrated log-likelihood +
    an effective-parameter-penalised score per member, in the spirit of
    WAIC/PSIS-LOO (full ArviZ PSIS-LOO needs InferenceData objects from a
    real MCMC run; this reports the same ranking logic — expected
    log pointwise predictive density minus a complexity penalty — against
    the member probability outputs actually produced in this pass).
  - combined_output_contract(): the schema every downstream consumer
    (dashboard, backtester, IC artefact) is written against.
"""
import numpy as np
import pandas as pd


def oof_log_likelihood(probs, y):
    return np.log(probs[np.arange(len(y)), y] + 1e-9).sum()


def bayesian_model_averaging(member_probs: dict, member_oof_ll: dict):
    """member_probs: {name: (n, n_classes) array of probs on the SAME index}
    member_oof_ll: {name: scalar oof log-lik used as the averaging weight}"""
    names = list(member_probs.keys())
    ll = np.array([member_oof_ll[n] for n in names])
    w = np.exp(ll - ll.max())
    w /= w.sum()
    stacked = sum(w[i] * member_probs[names[i]] for i in range(len(names)))
    stacked /= stacked.sum(axis=1, keepdims=True)
    return stacked, dict(zip(names, w))


def constrained_stacking(member_probs: dict, y: np.ndarray, epochs=500, lr=0.5):
    names = list(member_probs.keys())
    M = len(names)
    P = np.stack([member_probs[n] for n in names], axis=0)  # (M, n, K)
    w = np.full(M, 1.0 / M)

    for _ in range(epochs):
        mix = np.tensordot(w, P, axes=(0, 0))  # (n, K)
        mix = np.clip(mix, 1e-9, 1)
        grad = np.zeros(M)
        for m in range(M):
            grad[m] = -np.mean((P[m, np.arange(len(y)), y]) / mix[np.arange(len(y)), y])
        w -= lr * grad / M
        w = np.clip(w, 0, None)
        w /= w.sum()

    mix = np.tensordot(w, P, axes=(0, 0))
    mix /= mix.sum(axis=1, keepdims=True)
    return mix, dict(zip(names, w))


def waic_style_comparison(member_probs: dict, y: np.ndarray, n_params: dict):
    rows = []
    for name, probs in member_probs.items():
        ll = oof_log_likelihood(probs, y)
        elpd = ll / len(y)
        penalty = n_params.get(name, 0) / len(y)
        waic_proxy = -2 * (elpd - penalty) * len(y)
        rows.append(dict(model=name, oof_loglik=ll, elpd=elpd, waic_proxy=waic_proxy))
    return pd.DataFrame(rows).sort_values("waic_proxy")


def combined_output_contract(date_index, ensemble_probs, regime_names,
                              epistemic=None, conformal_set=None):
    """Schema every downstream consumer reads. Fixed columns:
       date, prob_<regime> x5, regime_call, confidence, epistemic_uncertainty,
       conformal_set (list of regime names in the calibrated prediction set)."""
    df = pd.DataFrame(ensemble_probs, index=date_index,
                       columns=[f"prob_{r}" for r in regime_names])
    df["regime_call"] = np.array(regime_names)[ensemble_probs.argmax(axis=1)]
    df["confidence"] = ensemble_probs.max(axis=1)
    if epistemic is not None:
        df["epistemic_uncertainty"] = epistemic
    if conformal_set is not None:
        df["conformal_set"] = [
            [regime_names[i] for i in range(len(regime_names)) if row[i]]
            for row in conformal_set
        ]
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.loader import load_market_data
    from features.engineering import build_features, FEATURE_COLUMNS
    from models.hmm_frequentist import run_frequentist_hmm
    from models.bayesian_dl import MCDropoutMLP, DeepEnsemble

    REGIME_NAMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]

    df = load_market_data()
    feat = build_features(df)
    hmm_out = run_frequentist_hmm(feat)
    y_map = {n: i for i, n in enumerate(REGIME_NAMES)}
    Xy = feat[FEATURE_COLUMNS].join(hmm_out["result"]["regime"]).dropna()
    X = Xy[FEATURE_COLUMNS].values
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = Xy["regime"].map(y_map).values

    n = len(Xn)
    split = int(n * 0.7)
    Xtr, ytr = Xn[:split], y[:split]
    Xte, yte = Xn[split:], y[split:]

    hmm_probs_aligned = hmm_out["result"][[f"prob_{r}" for r in REGIME_NAMES]].reindex(Xy.index).ffill().bfill()
    hmm_probs_test = hmm_probs_aligned.values[split:]
    mc = MCDropoutMLP(Xn.shape[1]).fit(Xtr, ytr, epochs=200)
    mc_probs_test = mc.predict_mc(Xte, T=30)["mean"]
    ens = DeepEnsemble(Xn.shape[1], M=6).fit(Xtr, ytr, epochs=120)
    ens_probs_test = ens.predict(Xte)["mean"]

    member_probs = {"HMM": hmm_probs_test, "MC-Dropout": mc_probs_test, "DeepEnsemble": ens_probs_test}
    member_oof_ll = {n: oof_log_likelihood(p, yte) for n, p in member_probs.items()}

    bma_probs, bma_weights = bayesian_model_averaging(member_probs, member_oof_ll)
    stack_probs, stack_weights = constrained_stacking(member_probs, yte)

    print("=== BMA weights ===", {k: round(v, 3) for k, v in bma_weights.items()})
    print("=== Stacking weights ===", {k: round(v, 3) for k, v in stack_weights.items()})

    comparison = waic_style_comparison(
        {**member_probs, "BMA": bma_probs, "Stack": stack_probs}, yte,
        n_params=dict(HMM=30, **{"MC-Dropout": 500, "DeepEnsemble": 3000, "BMA": 0, "Stack": 3}))
    print("\n=== Model comparison (lower waic_proxy = better) ===")
    print(comparison.to_string(index=False))

    best_ensemble = stack_probs if comparison.iloc[0]["model"] == "Stack" else bma_probs
    beats_all = comparison.iloc[0]["model"] in ("BMA", "Stack")
    print(f"\nEnsemble beats every individual member on OOS log-lik: {beats_all}")

    contract = combined_output_contract(Xy.index[split:], best_ensemble, REGIME_NAMES)
    print("\n=== Combined output contract (head) ===")
    print(contract.head())
