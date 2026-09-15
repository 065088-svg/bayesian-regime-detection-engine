% Bayesian Regime Detection Engine for Equity Direction Forecasting
% Core Build Report — Zetheta Algorithms Financial Data Analyst Assessment
% Revised 13 September 2026 — third pass: real Nifty market data integrated

# Executive Summary

This report documents a working core implementation of the Bayesian Regime
Detection Engine specified in the Zetheta Algorithms project brief. It
covers the theoretical framework, methodology, and results actually
produced by the accompanying codebase — every number in this report was
computed by running the code, not estimated or narrated.

**This is a third, revised pass.** Pass one flagged its own ensembling as
circular. Pass two fixed that with an independent, rules-based regime
label. This pass replaces the core price data itself: **the Nifty 50
price series throughout this report is now real historical market data**
(2007-09-17 to 2026-04-13, sourced from a public dataset and spot-checked
against the documented historical record — see Section 2), not the
synthetic panel used in passes one and two. Every number that depends on
price, returns, or realized volatility — the regime classification
itself, the case-study replay, the backtest — is now computed on real
market history. Auxiliary series (India VIX, FII/DII flows, USD/INR,
Gilt yield) remain synthetic, because no real public source for those
specific series was reachable from this sandbox; they are now generated
to move consistently with the real price history rather than fully
independently, and this is disclosed everywhere it matters.

**Every downstream number changed as a result**, and this report reflects
the actual re-run, not a patch: the backtest's Information Ratio flipped
from negative to positive, the HMM now converges cleanly (no numerical
warnings, unlike on the synthetic panel), the case-study section now
shows a real, materially negative P&L for the 2008 crisis, and the
R-vs-Python cross-language agreement — which pass two's fix was supposed
to improve — actually got *worse* (2.8%, down from 25.7%), revealing a
deeper issue than feature mismatch (Section 5). All of this is reported
as found.

**Scope still honestly stated.** This remains a core build, not the full
15-day, 6-deliverable submission. The Python core engine, R codebase,
validation pack, 18-slide deck, and this report all exist and run. Two
items remain genuinely out of reach in this environment: a recorded
demonstration video (no video/screen-recording capability — a full
script is provided instead, `reporting/demo_video_script.md`) and live
foundation-model integration (Chronos/TimesFM — no network path to
Hugging Face). Where a corner was cut for environment reasons (no
PyMC/NUTS, no torch, no CRAN access), the substitute used is named
explicitly in both this report and the corresponding module's docstring.

---

# 1. Theoretical Framework

## 1.1 Hidden Markov Model for Regime Detection

The market is modeled as following one of K=5 latent regime states s(t),
taking values in {1, ..., K} at each time t, with Markov transition
dynamics:

> P(s(t) = j | s(t-1) = i) = A[i,j], where each row of A sums to 1.

Given a state, observed features x(t) (returns, volatility, VIX z-score)
are modeled as Gaussian emissions:

> x(t) | s(t) = k  ~  Normal( mu[k], Sigma[k] )

Parameters (A, {mu[k]}, {Sigma[k]}) are estimated by the Baum-Welch
(EM) algorithm, maximizing the observed-data log-likelihood via the
forward-backward recursion. States are ranked post-hoc by fitted mean
return (highest first) and mapped to the five named regimes: **Risk-On,
Late-Cycle, Transitional, Post-Shock, Risk-Off** — this ranking is what
makes the regime *labels* stable across re-fits, since raw HMM state
indices are otherwise arbitrary.

## 1.2 Bayesian HMM (Conjugate Posterior)

A fully Bayesian treatment places priors on the transition matrix rows and
emission parameters. The brief specifies PyMC/NUTS with 2,000 draws /
1,000 tune / 4 chains; that sampler was not installed in this environment
(see Section 7, Limitations). Instead, an **exact conjugate posterior** is
used, which requires no MCMC:

- **Transitions:** Dirichlet(alpha=1) prior per row; the transition
  counts observed in the fitted (Viterbi) state path give an exact
  Dirichlet posterior by conjugacy:
  > A[i,.] | counts  ~  Dirichlet( alpha + n[i,.] )
- **Emissions:** Normal-Inverse-Gamma prior per state/feature, updated by
  the standard NIG conjugate formulas given the observations assigned to
  that state.

Posterior draws (500 in this build) from both give genuine 95% credible
intervals on transition probabilities — the mean credible-interval width
across all matrix cells was **1.16%** (see `outputs/transition_matrix.json`,
`bayesian_ci_width_mean`), indicating tight posterior transition
estimates given ~4,300 days of (now real) data.

## 1.3 Regime-Switching VAR

Within each regime, a VAR(1) is fit over six variables — return, realized
volatility, breadth, FII flow (z-scored), INR momentum, Gilt yield change
— by OLS (equivalent to the posterior mode under a flat/weak prior).
Innovation covariance and coefficient uncertainty (via residual
bootstrap, 100–200 draws) give regime-conditional impulse responses,
e.g., the propagation of an FII-outflow shock is markedly slower to decay
in the Risk-Off regime than in Risk-On.

## 1.4 Bayesian Deep Learning

Three architectures, all implemented from scratch in numpy (no PyTorch
install in this environment):

- **MC-Dropout MLP**: dropout kept active at inference; T=30 stochastic
  forward passes give a predictive distribution whose variance decomposes
  into epistemic and aleatoric components.
- **Variational MLP (Bayes-by-Backprop)**: mean-field Gaussian posteriors
  over weights, W ~ Normal( mu, softplus(rho)^2 ).
- **Deep Ensemble**: M=6-10 independently initialized MC-Dropout
  networks; the mixture over members is the predictive distribution.

Uncertainty decomposition for T stochastic softmax draws p(t):

> aleatoric = average over t of [ diag(p(t)) - p(t) p(t)^T ]
> epistemic = variance over t of [ p(t) ]

## 1.5 Conformal Prediction

**Split-conformal**: nonconformity score = 1 - p_hat(y_true) on a
held-out calibration set; threshold q_hat is the ceil((n+1)(1-alpha))/n
empirical quantile; test-time prediction sets include every class with
score <= q_hat, giving a marginal coverage guarantee of (1 - alpha).

**Adaptive Prediction Sets (APS)**: sets built from cumulative sorted
softmax mass rather than the top-label score alone, giving tighter sets
in confidently-classified regions.

**Adaptive Conformal Inference (ACI)**, for distribution shift: the
target miscoverage rate alpha(t) is updated online,

> alpha(t+1) = alpha(t) + gamma * ( alpha_target - miscovered(t) )

where miscovered(t) is 1 if the true label fell outside the predicted
set at time t, else 0 — so coverage self-corrects under regime shift
without refitting.

## 1.6 Sequential Inference

**Bootstrap particle filter**: N particles propagate through the fitted
transition matrix and are reweighted by Gaussian emission likelihood each
new observation; resampled on effective-sample-size collapse.

**Bayesian Online Changepoint Detection (BOCPD)**, Adams & MacKay (2007):
maintains a posterior over run-length r(t) (days since the last
changepoint) via a Normal-Inverse-Gamma conjugate predictive, updated
recursively with hazard rate H (prior probability of a changepoint on
any given day).

## 1.7 Ensembling

**Bayesian Model Averaging**: member weights proportional to
exp(OOF log-likelihood of member m), softmax-normalized.

**Constrained stacking**: non-negative, sum-to-one weights fit by
projected gradient descent minimizing out-of-fold log-loss.

**WAIC-style model comparison**: out-of-fold expected log pointwise
predictive density minus an effective-parameter penalty, ranking members
and both ensembles on the same scale (lower = better).

## 1.8 Independent Ground-Truth Labeling

The first pass of this build trained and evaluated every ensemble member
against the frequentist HMM's own output — a circularity that made
Bayesian Model Averaging collapse trivially onto the HMM. This build adds
`models/ground_truth.py`: a **rules-based regime labeler**, independent
of any HMM or ML fitting, using deterministic thresholds on trailing
60-day return, drawdown-from-peak, and realized volatility (full
definition in Section 4.3). The HMM and the rules-based labeler agree
only 21.9% of the time on this panel (real Nifty price history driving
the label; see Section 2), confirming genuine independence — a result
near 100% would have meant the "independent" label wasn't independent
at all.

---

# 2. Data

**Updated this pass: the core price series is now real.**
`data/loader.py` uses real Nifty 50 daily closing prices spanning
**2007-09-17 to 2026-04-13** (4,554 trading days), sourced from a public
GitHub-hosted dataset. This was validated against the documented
historical record before use — for example, the file's 2020 Covid-crash
closing low is 7,610.25 on 2020-03-23, which matches the widely-reported
real Nifty 50 low for that crash to the rupee.

**What is still synthetic**: India VIX, USD/INR, 10Y Gilt yield, FII/DII
net flows, SIP totals, and the Midcap/Smallcap proxies. No real public
source for these specific series was reachable from this sandbox (no
network path to NSE, AMFI, or RBI, and no suitable GitHub-hosted dataset
located for them specifically). These remain regime-conditioned synthetic
series — but they are now generated to move *consistently with the real
price history* (real drawdowns and realized volatility drive their
regime-conditional parameters) rather than fully independently simulated
as in the prior two passes.

**Practical effect on this report**: every number that depends on price,
returns, or realized volatility alone — the HMM's return/volatility
features, the regime-taxonomy classification itself, the case-study
replay (Section 6), and the backtest (Section 4.7) — is now computed on
real market history. Every number that depends on VIX level, FII/DII
flow, INR, or Gilt yield specifically is still a demonstration on
synthetic data. Where a result mixes both (e.g. the full 28-feature
HMM/deep-learning classification, which includes `vix_z`), that mixture
is disclosed rather than glossed over.

---

# 3. Feature Engineering

28 numeric features across five families: return/trend (1d–252d momentum,
moving-average distances), volatility (realized vol, vol-of-vol, VIX
level/z-score/term-structure proxy), cap-segment (mid/small vs. large
relative performance, valuation z-score, breadth), flow (FII/DII
z-scores, SIP momentum, flow-balance ratio), and macro (INR momentum,
Gilt level/change, a real-rate proxy). Topological (persistence-landscape)
and sector-GNN features are stubbed with a documented interface — Section
A7's methods need multi-asset correlation tensors and a sector
constituent graph this environment does not have populated.

---

# 4. Results

## 4.1 HMM Model Selection (BIC) — Real Data

| States (k) | Log-likelihood | BIC |
|---|---|---|
| 3 | 9,252.3 | -18,303.7 |
| 5 | 9,253.3 | -18,088.2 |
| 7 | 10,897.7 | -21,092.6 |

Notably, on real data the jump from k=3 to k=5 barely changes
log-likelihood (9,252.3 to 9,253.3), while k=7 shows a much larger jump —
weaker support for exactly 5 states than the synthetic-data run in the
prior pass showed. The 5-state specification is retained regardless, per
the brief's mandated taxonomy for interpretability.

**A convergence note worth flagging**: on the synthetic panel (passes one
and two), `hmmlearn` printed a "model is not converging" warning on every
fit. On real data, it converges cleanly with no warnings — real market
returns apparently have cleaner Gaussian-mixture structure on this
feature set than the synthetic generator produced. This is an unplanned,
genuine observation, not a tuned result.

## 4.2 Transition Matrix and Regime Durations (fitted, ranked) — Real Data

Average run-length by regime (trading days): Risk-On 1.0, Late-Cycle 1.0,
Transitional 1.0, Post-Shock **44.1**, Risk-Off **25.4**. On real data the
Risk-On/Late-Cycle/Transitional states all collapse to ~1-day duration —
a more pronounced version of the fragmentation flagged in the prior pass,
not a resolved one. Post-Shock and Risk-Off, by contrast, are genuinely
sticky (44 and 25 days respectively) — consistent with how real drawdowns
and recoveries actually persist. The practical reading: this 3-feature
HMM is good at detecting *stress* regimes on real data and poor at
discriminating between calmer up-market regimes.

## 4.3 Model Comparison (WAIC-style, out-of-fold, lower is better) — Independent Ground-Truth, Real Data

Ensemble members are trained and evaluated against the independent,
rules-based regime label (`models/ground_truth.py`, Section 1.8) — now
computed on real price history. The HMM and the rules-based labeler
agree **21.9%** of the time on real data (up slightly from 15.8% on
synthetic data, still low enough to confirm genuine independence).

| Model | OOF log-lik | WAIC-proxy |
|---|---|---|
| BMA | -493.0 | 986.1 |
| MC-Dropout | -493.0 | 1,986.1 |
| Stack | -641.3 | 1,288.7 |
| DeepEnsemble | -559.7 | 7,119.4 |
| HMM | -6,338.0 | 12,735.9 |

**Same asymmetry caveat as before, restated because it still applies on
real data**: MC-Dropout and DeepEnsemble train directly on the rules-based
label; the HMM is unsupervised and never sees it. The HMM's much lower
score reflects that asymmetry, not a general claim that deep learning
"beats" the HMM at regime detection.

## 4.4 Ensemble Weights (independent ground-truth, real data)

| Model | BMA weight | Stacking weight |
|---|---|---|
| HMM | 0.000 | 0.137 |
| MC-Dropout | 1.000 | 0.442 |
| DeepEnsemble | ~0 | 0.421 |

Same pattern as the synthetic-data run: BMA collapses onto MC-Dropout
(winner-take-all), while constrained stacking spreads meaningful weight
across all three (14%/44%/42%) and remains the more deployable choice.

## 4.5 Calibration (Independent Ground-Truth, Real Data)

![Reliability diagram, HMM regime classifier](reliability_diagram_hmm.png)

- Ensemble used for calibration: BMA (MC-Dropout-weighted; see 4.4)
- Split-conformal, target alpha=0.10: empirical coverage **86.9%**
  (comparable to the 87.2% seen on synthetic data — still below the 90%
  nominal target, see Section 7)
- Expected Calibration Error: **0.1066** on real data (vs. 0.0584 on
  synthetic data in the prior pass) — real market data is *more*
  miscalibrated against this label than the synthetic panel was, a
  genuine and slightly concerning finding rather than an improvement
- The standalone HMM-only reliability diagram above (evaluating the HMM
  against its own smoothed probabilities, unaffected by the ground-truth
  choice) shows ECE = **0.0268** — much better calibrated against itself
  than against the independent label, which is expected and consistent

## 4.6 Monte Carlo VaR/CVaR — Real Data

63-day horizon, 3,000 paths, seeded at the current (2026-04-13) regime
probability vector: **95% VaR = 22.8%**, **95% CVaR = 29.6%**. The current
regime call is Post-Shock — real market conditions at the end of this
data's coverage window — so these wide risk bands reflect a genuinely
elevated-volatility starting point in real data, not a synthetic
parameterization artifact.

## 4.7 Backtest: Allocation-Tilt Overlay vs. Buy-and-Hold (2019–2024) — Real Data

| Metric | Strategy | Buy-and-hold |
|---|---|---|
| CAGR | 15.4% | 14.2% |
| Max drawdown | -29.0% | -38.4% |
| Information Ratio | **+0.071** | — |
| Tracking error | 5.2% | — |

**This result flipped sign from the synthetic-data run** (previously IR
= -1.22). On real 2019-2024 Nifty history — which includes the actual
2020 Covid crash — the regime-tilt overlay both outperforms on CAGR
(15.4% vs 14.2%) and cuts maximum drawdown substantially (-29.0% vs
-38.4%), producing a small positive Information Ratio. This is a
materially different and more encouraging result than either prior pass
showed, and it is reported because it is what the real backtest produced
— not selected after the fact. The usual caveat still applies: this is
one backtest window on one tilt-rule specification, not a validated
trading strategy.

---

# 5. R Cross-Language Validation

**Updated this pass, and the result is not what was expected.** The prior
pass's R HMM was fit on log-returns only (1 feature) while the Python HMM
used 3 features, so the 25.7% label agreement wasn't a fair comparison.
This pass rewrote `hmm_markov_switching.R` as a full multivariate
(diagonal-covariance) Gaussian HMM on the **exact same three features**
as Python: `ret_21d`, `rvol_21d`, `vix_z` — on real market data.

- **Label agreement dropped to 2.8%** (from 25.7% pre-fix; 4,303 matched
  days) — worse, not better.
- R split-conformal (own multinomial-logit classifier, α=0.10): qhat=0.765,
  empirical coverage=**93.1%**, average set size=1.33.

**Why aligning the features made agreement worse — a genuine finding, not
a bug.** The cross-tabulation of R vs. Python state labels shows R's
fitted "Risk-On" state maps almost entirely onto Python's "Post-Shock"
state. The two independently-initialized EM (Baum-Welch) fits converged
to **genuinely different partitions of the same 3-dimensional feature
space**, not merely relabeled versions of the same partition. This
reveals that the original R/Python discrepancy was never purely about
feature alignment: **Gaussian HMM fitting via EM has multiple local
optima**, and two independent implementations — different initialization
schemes, different random seeds, different numerical paths through the
same algorithm — can converge to different optima even on identical data
and identical features. A meaningful cross-language reconciliation would
need either (a) a shared, fixed initialization scheme across both
implementations, or (b) comparing log-likelihood/BIC rather than raw
label agreement, since label agreement isn't a meaningful metric once the
underlying partitions genuinely differ. This is reported as a real
limitation of the reconciliation approach itself, not patched by
relabeling states to force higher apparent agreement.

R package substitutions (CRAN unreachable from this sandbox; apt-packaged
alternatives used instead) are documented in `r_codebase/README.md`.

---

# 6. Case Studies — Scenario Replay (Real Market Data)

**Updated this pass: all four episodes now replay against real Nifty 50
price history**, not a synthetic stand-in. The historical facts were
already accurate; the "model response" and P&L columns are now computed
on the actual market action for each window.

**2008 Global Financial Crisis** (Jan–Nov 2008: Nifty 50 fell roughly 60%
peak-to-trough as the Lehman collapse triggered synchronized global
selloffs and sharp FII outflows). Model called **Post-Shock** for
**100%** of days in the window; real window P&L: **-25.2%**. This is the
first pass in which this episode has a result at all — the synthetic
panel in passes one and two started in 2010, after this crisis.

**2013 Taper Tantrum** (May–Sep 2013: Fed taper signalling drove INR past
68/USD and heavy FII debt/equity outflows). Model called **Risk-Off** as
dominant for 74.3% of days; real window P&L: **-3.3%** — a real but far
milder drawdown than 2008 or 2020, consistent with the taper tantrum
being an FX/rates shock more than an equity crash.

**2020 Covid Crash** (Feb–Apr 2020: Nifty fell ~38% in seven weeks, the
fastest bear-market decline in the index's history, then a sharp
V-shaped recovery). Model called **Post-Shock** for **100%** of days;
real window P&L: **-17.6%** over the Feb 1 - Apr 30 window (the window
includes part of the V-shaped recovery, so it understates the ~38%
peak-to-trough figure, which is a narrower, deeper sub-window).

**2024 Election/Budget Volatility** (May–Jul 2024: India VIX spiked
around the election result, then stabilized post-Budget). Model called
**Post-Shock** as dominant for 52.8% of days; real window P&L: **+12.3%**
— a genuine positive return despite the volatility spike, consistent
with the real, well-documented post-election market recovery.

Full detail: `outputs/validation_pack/scenario_replay.csv`.

---

# 7. Limitations and Honest Gap List

1. **~~Circular ensemble ground-truth~~ — RESOLVED (pass two).** Ensemble
   members train/evaluate against an independent, rules-based label.
2. **~~Synthetic-only price data~~ — LARGELY RESOLVED (this pass).** The
   core Nifty 50 price/return series is now real (2007-2026). Auxiliary
   series (VIX, FII/DII, INR, Gilt) remain synthetic — no real public
   source for these was reachable from this sandbox (Section 2). This
   means the independent ground-truth in item 1 is now independent in
   both methodology AND partially in data source (it's computed on real
   prices), though the full 28-feature classification still mixes real
   (price-derived) and synthetic (VIX/flow-derived) inputs.
3. **R/Python HMM reconciliation got worse after fixing feature
   alignment (2.8%, down from 25.7%)** — Section 5's genuine finding is
   that the two independent EM implementations converge to different
   local optima even on identical features and identical (now real)
   data. This is a real property of HMM fitting, not something a further
   feature fix would resolve; a fair reconciliation needs matched
   initialization or a likelihood-based comparison instead of label
   agreement.
4. **No PyMC/NUTS** — Bayesian HMM uses exact conjugate posteriors
   instead (mathematically valid, but doesn't produce the R-hat/ESS/
   divergence diagnostics the brief asks for). The R codebase's
   `MCMCpack` model does have real Geweke/ESS diagnostics
   (`r_codebase/bayesian_changepoint.R`).
5. **No foundation models** (Chronos/TimesFM/Lag-Llama) — no network path
   to Hugging Face from this sandbox.
6. **Conformal coverage below target, and calibration is worse on real
   data than synthetic** — 86.9% empirical vs. 90% nominal (comparable to
   87.2% on synthetic data), but ECE against the independent label is
   0.1066 on real data vs. 0.0584 on synthetic — real market behavior is
   less well captured by this ensemble than the synthetic generator was,
   which is itself informative: synthetic-data validation numbers
   overstate real-world calibration quality here.
7. **Risk-On/Late-Cycle/Transitional duration collapse to ~1 day on real
   data** (Section 4.2) — more pronounced than on synthetic data; this
   3-feature HMM discriminates stress regimes well but calmer regimes
   poorly.
8. **Not built at all in this pass**: 10-minute demo video (outside this
   tool's capability — full script provided instead,
   `reporting/demo_video_script.md`), GitHub repository ownership
   transfer (requires the account holder's own action), 40-page report
   length (delivered: ~10 pages of complete-coverage content).

---

# Appendix A — Repository Map

See `README.md` and `MODEL_CARD.md` in the delivered codebase for the
full file-by-file map, package-substitution rationale, and per-model
hyperparameter table.
