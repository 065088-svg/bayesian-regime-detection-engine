"""
models/hmm_frequentist.py
==========================
Frequentist Gaussian HMM regime classifier (Day 4 deliverable).

Fits a Gaussian HMM to a small return/vol feature block, does post-hoc
regime labelling by sorting fitted states on mean return (so the label
order is stable across refits), and reports transition matrix, regime
durations/stickiness, and BIC across 3/5/7-state specifications.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REGIME_NAMES_BY_RANK = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]


def fit_hmm(X: np.ndarray, n_states=5, n_iter=200, seed=7):
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                         n_iter=n_iter, random_state=seed)
    model.fit(X)
    return model


def label_states_by_return(model: GaussianHMM, return_col_idx=0):
    """Rank hidden states by fitted mean return (desc) -> stable labels."""
    means = model.means_[:, return_col_idx]
    order = np.argsort(-means)  # highest return first
    state_to_rank = {state: rank for rank, state in enumerate(order)}
    return state_to_rank


def transition_matrix(model):
    return model.transmat_


def regime_durations(state_seq: np.ndarray):
    """Average run-length per state (stickiness proxy)."""
    durations = {}
    run_state, run_len = state_seq[0], 1
    runs = []
    for s in state_seq[1:]:
        if s == run_state:
            run_len += 1
        else:
            runs.append((run_state, run_len))
            run_state, run_len = s, 1
    runs.append((run_state, run_len))
    for s in np.unique(state_seq):
        lens = [l for st, l in runs if st == s]
        durations[int(s)] = float(np.mean(lens)) if lens else 0.0
    return durations


def bic_model_selection(X: np.ndarray, states=(3, 5, 7), seed=7):
    results = {}
    for k in states:
        m = fit_hmm(X, n_states=k, seed=seed)
        logL = m.score(X)
        n_params = k * k - k + k * X.shape[1] * 2  # transitions + means/vars (diag)
        bic = -2 * logL + n_params * np.log(len(X))
        results[k] = dict(loglik=logL, bic=bic, n_params=n_params)
    return results


def run_frequentist_hmm(feat_df: pd.DataFrame, feature_cols=("ret_21d", "rvol_21d", "vix_z")):
    sub = feat_df[list(feature_cols)].dropna()
    X = sub.values

    bic_table = bic_model_selection(X, states=(3, 5, 7))

    model = fit_hmm(X, n_states=5)
    state_seq = model.predict(X)
    post_probs = model.predict_proba(X)

    rank_map = label_states_by_return(model, return_col_idx=0)
    labeled_seq = np.array([rank_map[s] for s in state_seq])
    label_names = np.array(REGIME_NAMES_BY_RANK)[labeled_seq]

    durations = regime_durations(labeled_seq)
    durations_named = {REGIME_NAMES_BY_RANK[k]: v for k, v in durations.items()}

    result = pd.DataFrame({
        "date": sub.index,
        "state": labeled_seq,
        "regime": label_names,
    }).set_index("date")
    for rank in range(5):
        orig_state = [s for s, r in rank_map.items() if r == rank][0]
        result[f"prob_{REGIME_NAMES_BY_RANK[rank]}"] = post_probs[:, orig_state]

    return dict(
        model=model,
        bic_table=bic_table,
        transition_matrix=transition_matrix(model),
        rank_map=rank_map,
        durations=durations_named,
        result=result,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.loader import load_market_data
    from features.engineering import build_features

    df = load_market_data()
    feat = build_features(df)
    out = run_frequentist_hmm(feat)

    print("=== BIC by state count ===")
    for k, v in out["bic_table"].items():
        print(f"  k={k}: loglik={v['loglik']:.1f}  BIC={v['bic']:.1f}")
    print("\n=== Regime durations (avg trading days) ===")
    for name, d in out["durations"].items():
        print(f"  {name}: {d:.1f}")
    print("\n=== Transition matrix (rows=from, in fitted-state order) ===")
    print(np.round(out["transition_matrix"], 3))
    print("\n=== Sample output ===")
    print(out["result"].tail(10))
