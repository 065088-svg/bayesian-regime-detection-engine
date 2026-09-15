"""
models/ground_truth.py
========================
Independent regime ground-truth (fixes the circularity flagged repeatedly:
the ensemble was trained/evaluated against the HMM's own Viterbi output,
so BMA collapsed onto the HMM trivially).

This is a RULES-BASED labeler — deterministic thresholds on drawdown,
trailing return, and realized volatility — independent of any HMM/ML
fitting. It is still applied to the same synthetic panel (no real market
data available in this sandbox), so it does not fix the "synthetic data"
limitation, but it DOES fix the "ensemble predicts itself" problem: the
HMM and the rules-based labeler are now two independent methodologies
that can genuinely agree or disagree.

Rule definitions (applied to the daily panel):
  - Post-Shock:    drawdown from trailing 60d peak <= -15%, within the
                   most recent 15 trading days of that drawdown
  - Risk-Off:      trailing 60d return <= -8% and NOT Post-Shock
  - Risk-On:       trailing 60d return >= +8% and trailing 21d realized
                   vol <= its 3-year median
  - Late-Cycle:    trailing 60d return >= +8% and trailing 21d realized
                   vol > its 3-year median (i.e. still rising, but choppier)
  - Transitional:  everything else (the default / residual bucket)

This mirrors the qualitative definitions in the brief's regime taxonomy
without using any of the HMM's fitted parameters.
"""
import numpy as np
import pandas as pd

REGIME_NAMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]


def rules_based_regime_labels(feat_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    price = price.reindex(feat_df.index)
    trailing_60d_ret = price.pct_change(60)
    roll_peak_60d = price.rolling(60).max()
    drawdown = price / roll_peak_60d - 1
    vol_21d = feat_df["rvol_21d"]
    vol_median_3y = vol_21d.rolling(756, min_periods=252).median()

    in_post_shock_dd = drawdown <= -0.15
    # "within the most recent 15 days of a >=15% drawdown": forward-fill a
    # flag for 15 days after any day the drawdown condition is first true
    post_shock_flag = in_post_shock_dd.rolling(15, min_periods=1).max().astype(bool)

    labels = pd.Series(REGIME_NAMES[2], index=feat_df.index)  # default Transitional
    labels[trailing_60d_ret >= 0.08] = "Late-Cycle"
    labels[(trailing_60d_ret >= 0.08) & (vol_21d <= vol_median_3y)] = "Risk-On"
    labels[trailing_60d_ret <= -0.08] = "Risk-Off"
    labels[post_shock_flag] = "Post-Shock"  # highest priority, applied last

    return labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.loader import load_market_data
    from features.engineering import build_features
    from models.hmm_frequentist import run_frequentist_hmm

    df = load_market_data()
    feat = build_features(df)
    price = df.set_index("date")["nifty_close"]

    rb_labels = rules_based_regime_labels(feat, price)
    print("=== Rules-based label distribution ===")
    print(rb_labels.value_counts())

    hmm_out = run_frequentist_hmm(feat)
    hmm_labels = hmm_out["result"]["regime"].reindex(rb_labels.index)

    common = rb_labels.dropna().index.intersection(hmm_labels.dropna().index)
    agree = (rb_labels.loc[common] == hmm_labels.loc[common]).mean()
    print(f"\nAgreement between rules-based labels and HMM labels: {agree:.3f}")
    print("(Genuine independent cross-check -- not expected to be 1.0; "
          "if it were, the two methodologies wouldn't be adding anything.)")
    print("\nCross-tab:")
    print(pd.crosstab(rb_labels.loc[common], hmm_labels.loc[common]))
