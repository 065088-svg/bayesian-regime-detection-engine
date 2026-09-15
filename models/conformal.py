"""
models/conformal.py
=====================
Conformal prediction and calibration (Day 11 deliverable).

  - split_conformal_sets():   classic split-conformal classifier (A6.2)
  - adaptive_prediction_sets(): APS (A6.3) — sets built from cumulative
    sorted softmax mass, giving tighter sets in confident regions
  - rolling_conformal_coverage(): tracks empirical coverage on a rolling
    252-day window (distribution-shift-robust re-calibration akin to
    Adaptive Conformal Inference (A6.5) — the miscoverage-driven alpha
    update from Gibbs & Candès 2021 is implemented directly)
  - reliability_diagram() / expected_calibration_error(): standard
    calibration diagnostics
"""
import numpy as np


def split_conformal_sets(probs_calib, y_calib, probs_test, alpha=0.1):
    """Score = 1 - p(true class) on calibration set; threshold = the
    (1-alpha) empirical quantile; test sets = all classes with
    1 - p(class) <= threshold."""
    n = len(y_calib)
    scores = 1 - probs_calib[np.arange(n), y_calib]
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores, q_level)
    sets = (1 - probs_test) <= qhat
    return sets, qhat


def adaptive_prediction_sets(probs_calib, y_calib, probs_test, alpha=0.1):
    """APS: cumulative sorted-probability mass until the true label is
    covered defines the calibration score; test sets built the same way."""
    def score_fn(p, y_idx):
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        rank = np.where(order == y_idx)[0][0]
        return cum[rank]

    n = len(y_calib)
    scores = np.array([score_fn(probs_calib[i], y_calib[i]) for i in range(n)])
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(scores, q_level)

    sets = np.zeros_like(probs_test, dtype=bool)
    for i, p in enumerate(probs_test):
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        k = np.searchsorted(cum, qhat) + 1
        sets[i, order[:k]] = True
    return sets, qhat


def adaptive_conformal_inference(probs_stream, y_stream, alpha_target=0.1, gamma=0.02):
    """Online ACI (Gibbs & Candès 2021): alpha_t adapts based on whether the
    previous prediction set covered the true label, giving distribution-
    shift-robust coverage without refitting. Returns per-step set sizes,
    coverage indicator, and the alpha path."""
    n = len(y_stream)
    alpha_t = alpha_target
    alphas, covered, set_sizes = [], [], []
    for t in range(n):
        p = probs_stream[t]
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        k = np.searchsorted(cum, 1 - alpha_t) + 1
        k = max(1, min(k, len(p)))
        pred_set = set(order[:k])
        hit = y_stream[t] in pred_set
        covered.append(hit)
        set_sizes.append(k)
        alphas.append(alpha_t)
        # ACI update
        err = 0 if hit else 1
        alpha_t = alpha_t + gamma * (alpha_target - err)
        alpha_t = np.clip(alpha_t, 0.01, 0.5)
    return dict(alpha_path=np.array(alphas), covered=np.array(covered), set_size=np.array(set_sizes))


def rolling_conformal_coverage(probs, y, window=252, alpha=0.1):
    n = len(y)
    coverage = np.full(n, np.nan)
    for t in range(window, n):
        calib_p, calib_y = probs[t - window:t], y[t - window:t]
        sets, _ = split_conformal_sets(calib_p, calib_y, probs[t:t + 1], alpha=alpha)
        coverage[t] = sets[0, y[t]]
    return coverage


def reliability_diagram(probs, y, n_bins=10):
    """Top-label reliability: bin by max predicted prob, compare to
    empirical accuracy in each bin."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(conf[mask].mean())
            bin_count.append(mask.sum())
        else:
            bin_acc.append(np.nan); bin_conf.append(np.nan); bin_count.append(0)
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


def expected_calibration_error(probs, y, n_bins=10):
    bin_conf, bin_acc, bin_count = reliability_diagram(probs, y, n_bins)
    total = sum(bin_count)
    ece = sum(c * abs(a - conf) for a, conf, c in zip(bin_acc, bin_conf, bin_count) if c > 0) / total
    return ece


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.loader import load_market_data
    from features.engineering import build_features, FEATURE_COLUMNS
    from models.hmm_frequentist import run_frequentist_hmm
    from models.bayesian_dl import MCDropoutMLP

    df = load_market_data()
    feat = build_features(df)
    hmm_out = run_frequentist_hmm(feat)
    y_map = {n: i for i, n in enumerate(["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"])}
    Xy = feat[FEATURE_COLUMNS].join(hmm_out["result"]["regime"]).dropna()
    X = Xy[FEATURE_COLUMNS].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = Xy["regime"].map(y_map).values

    n = len(X)
    split1, split2 = int(n * 0.6), int(n * 0.8)
    Xtr, ytr = X[:split1], y[:split1]
    Xcal, ycal = X[split1:split2], y[split1:split2]
    Xte, yte = X[split2:], y[split2:]

    mc = MCDropoutMLP(X.shape[1]).fit(Xtr, ytr, epochs=200)
    p_cal = mc.predict_mc(Xcal, T=30)["mean"]
    p_te = mc.predict_mc(Xte, T=30)["mean"]

    sets_sc, qhat_sc = split_conformal_sets(p_cal, ycal, p_te, alpha=0.1)
    sets_aps, qhat_aps = adaptive_prediction_sets(p_cal, ycal, p_te, alpha=0.1)
    print(f"Split-conformal qhat={qhat_sc:.3f}, avg set size={sets_sc.sum(1).mean():.2f}, "
          f"empirical coverage={np.mean([sets_sc[i, yte[i]] for i in range(len(yte))]):.3f}")
    print(f"APS qhat={qhat_aps:.3f}, avg set size={sets_aps.sum(1).mean():.2f}, "
          f"empirical coverage={np.mean([sets_aps[i, yte[i]] for i in range(len(yte))]):.3f}")

    aci = adaptive_conformal_inference(p_te, yte, alpha_target=0.1)
    print(f"ACI online coverage={aci['covered'].mean():.3f}, avg set size={aci['set_size'].mean():.2f}")

    ece = expected_calibration_error(p_te, yte)
    print(f"Expected Calibration Error (test): {ece:.4f}")
