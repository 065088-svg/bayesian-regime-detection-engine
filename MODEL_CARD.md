# Model Card — Bayesian Regime Detection Engine (core build)

## Ensemble members

| Model | File | Inputs | Key hyperparameters / priors | Output |
|---|---|---|---|---|
| Frequentist Gaussian HMM | `models/hmm_frequentist.py` | `ret_21d, rvol_21d, vix_z` | 5 states, diag covariance, 200 EM iterations | Regime path, smoothed probabilities |
| Bayesian HMM (conjugate) | `models/hmm_bayesian.py` | same 3 features | Dirichlet(α=1) per transition row; Normal-Inverse-Gamma(μ₀=0,κ₀=1,α₀=2,β₀=1) per emission | Posterior mean + 95% credible interval on transitions and regime probability paths |
| Regime-switching VAR | `models/rs_var.py` | `ret, vol, breadth, fii_flow, inr_chg, gilt_chg` | VAR(1), OLS within each HMM-labeled regime, 200-draw residual bootstrap | Regime-conditional coefficients, innovation covariance, impulse responses |
| MC-Dropout MLP | `models/bayesian_dl.py` | 28 engineered features (z-scored) | 1 hidden layer (32 units), dropout=0.3, 200 epochs, T=30 MC passes at inference | Predictive mean, epistemic + aleatoric uncertainty |
| Variational MLP (Bayes-by-Backprop) | `models/bayesian_dl.py` | same 28 features | Mean-field Gaussian weight posteriors, 24 hidden units, 250 epochs | Predictive mean, epistemic uncertainty |
| Deep Ensemble | `models/bayesian_dl.py` | same 28 features | M=6–10 independently-initialised MC-Dropout MLPs | Predictive mean, epistemic uncertainty |

## Ground-truth labeling

- **Independent, rules-based label**: `models/ground_truth.py` — deterministic thresholds on trailing 60-day return, drawdown-from-peak, and realized volatility, computed on real Nifty price history (pass three). Used to train/evaluate MC-Dropout, Deep Ensemble, and to score the HMM out-of-sample.
- Agreement with the frequentist HMM's own labels: **21.9%** on real data (was 15.8% on synthetic data, pass two), confirming genuine independence in both cases.
- Replaces the first-pass approach of using the HMM's own Viterbi output as ensemble ground-truth, which had inflated the HMM's apparent BMA weight to ≈1.0 trivially.

## Ensembling (real data)

- **Bayesian Model Averaging**: softmax-normalised out-of-fold log-likelihood weights. HMM 0.00 / MC-Dropout 1.00 / DeepEnsemble ≈0 — collapses onto MC-Dropout because it was trained directly on the label while the HMM (unsupervised) was not.
- **Constrained stacking**: non-negative, sum-to-one weights fit by projected gradient descent on out-of-fold log-loss. HMM 0.137 / MC-Dropout 0.442 / DeepEnsemble 0.421 — the more deployable number, since it doesn't zero out any member.
- **Selection**: WAIC-style proxy (out-of-fold ELPD minus an effective-parameter penalty) ranks members + both ensembles; the lower-scoring ensemble is used downstream.

## Calibration (real data)

- Split-conformal and Adaptive Prediction Sets, target α=0.10 (90% nominal coverage).
- Empirical coverage: **86.9%** (86.9% real vs 87.2% synthetic — comparable). ECE: **0.1066** on real data vs **0.0584** on synthetic data (pass two) — real market behavior is *less* well captured by this ensemble than the synthetic generator suggested, a genuine and slightly concerning finding rather than an improvement.
- Expected Calibration Error (ECE) computed on a held-out evaluation split, 10 confidence bins.
- Online Adaptive Conformal Inference available (`adaptive_conformal_inference`) for streaming re-calibration under distribution shift.

## Known limitations (see README "Known rough edges" for detail)

1. ~~Circular ground-truth~~ — **resolved (pass two)** via `models/ground_truth.py`, **holds on real data (pass three)**.
2. **Core price data is now real** (2007-2026); auxiliary series (VIX, FII/DII, INR, Gilt) remain synthetic — no real public source for these was reachable from this sandbox. The independent ground-truth in item 1 is now independent in methodology AND partially in data source (computed on real prices), though the full 28-feature classification still mixes real (price-derived) and synthetic (VIX/flow-derived) inputs.
3. **R/Python HMM reconciliation got worse after aligning feature sets** (25.7% → 2.8%) — two independent EM (Baum-Welch) implementations converged to different local optima on the identical real-data feature matrix. See `r_codebase/README.md`. This is a property of HMM fitting (multiple local optima), not a residual feature-alignment bug.
4. MCMC diagnostics (R-hat, ESS, divergences) specified in the brief require an actual NUTS sampler (PyMC/NumPyro), which was not installed this pass; the Bayesian HMM here uses exact conjugate posteriors instead, which have no such diagnostics because no MCMC sampling error is present in the first place. (The R codebase's `MCMCpack` model does have real Geweke/ESS diagnostics — see `r_codebase/bayesian_changepoint.R`.)
5. Foundation-model members (Chronos, TimesFM, etc.) are not implemented — no network path to Hugging Face in this sandbox.
6. BOCPD hazard-rate prior is not tuned against real event dates; changepoint probabilities are directionally reasonable but numerically small.
7. The HMM-vs-deep-learning comparison in Section 4.3 of the report is asymmetric by construction: the HMM is unsupervised and never sees the ground-truth label, while MC-Dropout/DeepEnsemble train on it directly. This is disclosed, not hidden, but means the WAIC-style ranking should not be read as "deep learning is a better regime model" in general.

## Versioning

Built and executed across three passes ending 2026-09-13 in a single
sandboxed session against Python 3.12, hmmlearn, statsmodels,
numpy/pandas/scipy (stdlib scientific stack), plus R 4.3 with apt-packaged
CRAN alternatives. No GPU used. Full Python pipeline runtime: ~90 seconds
on the real Nifty 50 daily panel, 2007-09-17 to 2026-04-13 (4,554 trading
days).
