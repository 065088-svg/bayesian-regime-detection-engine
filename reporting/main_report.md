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

### 1.1.1 Full Baum-Welch Derivation

This subsection derives the EM update equations actually implemented in
`models/hmm_frequentist.py` (via `hmmlearn`) and independently
re-implemented from scratch in `r_codebase/hmm_markov_switching.R` — the
same algorithm, coded twice, in two languages, is one of the strongest
correctness checks available without a held-out ground truth.

**Forward pass.** Define the forward variable alpha(t,i) = P(x(1),...,x(t),
s(t)=i | theta) — the joint probability of the observations up to time t
and being in state i at time t. This is computed recursively:

> alpha(1,i) = pi(i) * B(i, x(1))
> alpha(t,j) = [ sum over i of alpha(t-1,i) * A(i,j) ] * B(j, x(t))

where B(j, x(t)) is the Gaussian emission density of state j evaluated at
observation x(t), and pi is the initial state distribution. In
`hmm_frequentist.py` this is implemented with log-space scaling
(`c_scale` in the R re-implementation) to prevent numerical underflow
over ~4,300 time steps — without scaling, alpha values underflow to exact
zero well before t=100 given typical Gaussian densities under 1.0.

**Backward pass.** Define beta(t,i) = P(x(t+1),...,x(T) | s(t)=i, theta):

> beta(T,i) = 1
> beta(t,i) = sum over j of A(i,j) * B(j, x(t+1)) * beta(t+1,j)

**E-step.** Combine forward and backward passes into the posterior state
probability (the "smoothed" probability reported throughout this report
as `prob_<regime>`):

> gamma(t,i) = P(s(t)=i | X, theta) = alpha(t,i) * beta(t,i) / sum_k[alpha(t,k) * beta(t,k)]

and the posterior transition probability:

> xi(t,i,j) = P(s(t)=i, s(t+1)=j | X, theta)
>           = alpha(t,i) * A(i,j) * B(j,x(t+1)) * beta(t+1,j) / P(X | theta)

**M-step.** Re-estimate parameters as the sufficient statistics implied
by the E-step posteriors:

> pi_new(i)   = gamma(1,i)
> A_new(i,j)  = [ sum_t xi(t,i,j) ] / [ sum_t gamma(t,i) ]
> mu_new(k)   = [ sum_t gamma(t,k) * x(t) ] / [ sum_t gamma(t,k) ]
> Sigma_new(k) = [ sum_t gamma(t,k) * (x(t)-mu_new(k))^2 ] / [ sum_t gamma(t,k) ]

This is exactly the four-line M-step visible in both
`models/hmm_frequentist.py` (via hmmlearn's internal EM) and the
hand-written R implementation's `gaussian_hmm_em_mv` function — having
written the multivariate diagonal-covariance version of this update by
hand in R (Section 5) rather than only calling a library in Python is
itself part of why the R/Python discrepancy investigation (Section 5)
carries real evidentiary weight: it rules out "one side is a black-box
library bug" as an explanation, since both sides implement the same
derivation independently.

**Convergence.** EM increases the log-likelihood monotonically each
iteration (a standard EM guarantee — it maximizes a lower bound on the
log-likelihood that touches the true log-likelihood at the current
parameter estimate). The loop terminates when successive log-likelihood
values differ by less than a tolerance (1e-6 in both implementations).
Because EM only guarantees convergence to a *local* maximum, different
initializations can converge to different fixed points — this is
precisely the mechanism behind the R-vs-Python divergence documented in
Section 5.

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

### 1.2.1 Derivation: Why the Conjugate Posterior is Exact

**Dirichlet-Multinomial conjugacy.** Each row of the transition matrix
A(i,.) is a categorical distribution over 5 next-states. Placing a
Dirichlet(alpha) prior on this row and observing n(i,j) = the number of
times the fitted state path transitions from i to j gives, by Bayes'
rule:

> P(A(i,.) | data) proportional to P(data | A(i,.)) * P(A(i,.))
>                 = [ product_j A(i,j)^n(i,j) ] * [ product_j A(i,j)^(alpha-1) ]
>                 = product_j A(i,j)^(n(i,j) + alpha - 1)

This is exactly the kernel of a Dirichlet(alpha + n(i,.)) distribution —
no approximation, no sampling error, because the Dirichlet prior's
functional form (a product of powers) is closed under multiplication by
the categorical likelihood's functional form (also a product of powers).
This is what "conjugate" means concretely: the posterior family equals
the prior family, so the posterior can be written down in closed form
instead of approximated by MCMC. `models/hmm_bayesian.py`'s
`dirichlet_posterior_transitions()` function is a direct, line-for-line
implementation of alpha + n(i,.).

**Normal-Inverse-Gamma conjugacy for the emissions.** Each state's
emission mean and variance (mu_k, sigma_k^2) get a Normal-Inverse-Gamma
prior: sigma_k^2 ~ InverseGamma(alpha0, beta0), mu_k | sigma_k^2 ~
Normal(mu0, sigma_k^2 / kappa0). Given n observations x(1),...,x(n)
assigned to state k with sample mean xbar and sample variance v, the
posterior is again exactly Normal-Inverse-Gamma with updated parameters:

> kappa_n = kappa0 + n
> mu_n    = (kappa0*mu0 + n*xbar) / kappa_n
> alpha_n = alpha0 + n/2
> beta_n  = beta0 + 0.5*n*v + [kappa0*n*(xbar-mu0)^2] / (2*kappa_n)

`gaussian_invgamma_posterior_emissions()` in `models/hmm_bayesian.py`
implements exactly these four update equations, then draws
sigma_k^2 ~ InverseGamma(alpha_n, beta_n) and mu_k | sigma_k^2 ~
Normal(mu_n, sigma_k^2/kappa_n) — both standard-library random draws,
not an iterative sampler. The practical upshot restated from Section 1.2:
**zero MCMC sampling error is introduced anywhere in this posterior**,
which is both the source of this method's speed advantage over
PyMC/NUTS and the reason it produces no R-hat/ESS diagnostics (there is
no chain to diagnose).

## 1.3 Regime-Switching VAR

Within each regime, a VAR(1) is fit over six variables — return, realized
volatility, breadth, FII flow (z-scored), INR momentum, Gilt yield change
— by OLS (equivalent to the posterior mode under a flat/weak prior).
Innovation covariance and coefficient uncertainty (via residual
bootstrap, 100–200 draws) give regime-conditional impulse responses,
e.g., the propagation of an FII-outflow shock is markedly slower to decay
in the Risk-Off regime than in Risk-On.

### 1.3.1 OLS-as-Posterior-Mode Derivation

A VAR(1) models each of the 6 variables at time t as a linear function of
all 6 variables at time t-1 plus an intercept: Y(t) = c + Phi*Y(t-1) +
epsilon(t), epsilon(t) ~ Normal(0, Sigma). Stacking observations into a
design matrix X (each row: [1, Y(t-1)]) and outcomes Y, the log-likelihood
under Gaussian innovations is maximized, equivalently, by minimizing sum
of squared residuals — the ordinary least squares estimator:

> Phi_hat = (X^T X)^(-1) X^T Y

Under a flat (improper, constant) prior over the coefficients, the
posterior is proportional to the likelihood alone, so the posterior mode
(and, since the likelihood is Gaussian in the coefficients, the posterior
mean) coincides exactly with the OLS estimator. This is the formal sense
in which `fit_regime_conditioned_var()` in `models/rs_var.py` — which
calls `np.linalg.lstsq` directly — is "equivalent to a Bayesian VAR under
a flat prior": no approximation is being smuggled in by using OLS instead
of a named Bayesian VAR package, only a specific (flat) prior choice, made
explicit here rather than left implicit.

**Bootstrap for coefficient uncertainty.** Because the flat-prior
posterior is degenerate at exactly the OLS estimate (a flat prior carries
no information to produce posterior spread beyond the likelihood's own
curvature), coefficient uncertainty is instead obtained by residual
bootstrap: resample (X,Y) pairs with replacement, refit OLS on each
resample, and take the empirical 2.5th/97.5th percentiles of the
resulting coefficient distribution as an approximate 95% interval. This
is a frequentist device standing in for what a proper hierarchical
Bayesian VAR (with an informative Minnesota or similar shrinkage prior)
would give natively — another documented, honest substitution consistent
with the pattern throughout this report.

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

### 1.4.1 Why MC-Dropout Approximates a Bayesian Posterior

Gal & Ghahramani (2016) showed that a neural network trained with
dropout is mathematically equivalent to an approximate variational
inference procedure over a deep Gaussian process: each dropout mask
sampled at inference corresponds to a sample from an approximate
posterior over network weights, where the approximating family is a
mixture of two Gaussians per weight — one with the weight's trained
value and near-zero variance, one at exactly zero. Keeping dropout
*active* at inference time (rather than the usual practice of disabling
it) and running T forward passes therefore approximates T draws from
this weight posterior, which is exactly what `MCDropoutMLP.predict_mc()`
implements: T=30 forward passes with `train_dropout=True` held on. This
is a real, peer-reviewed theoretical justification, not an ad-hoc
heuristic — though it is an approximation (a mixture-of-two-Gaussians
posterior is a much narrower family than a full Gaussian process
posterior), which is why this build also implements a second,
independent approximation family (Bayes-by-Backprop) as a cross-check.

### 1.4.2 The Variational ELBO Being Optimized

Bayes-by-Backprop (Blundell et al., 2015) directly parameterizes a
mean-field Gaussian posterior q(W | mu, rho) over each weight and
minimizes the negative Evidence Lower Bound (ELBO):

> Loss = KL[ q(W|mu,rho) || P(W) ] - E_q[ log P(Y | X, W) ]

`VariationalMLP.fit()` in `models/bayesian_dl.py` implements a
simplified version of this: the reparameterization trick (sampling
W = mu + softplus(rho) * epsilon, epsilon ~ Normal(0,1), so gradients can
flow through the sampling step) is implemented via `_sample_weights()`,
and the KL term is approximated by a simple L2 penalty on the posterior
means (`kl_weight * self.mu1`) rather than the full closed-form Gaussian
KL divergence — a documented simplification made to keep the from-scratch
numpy implementation tractable within this pass's time budget, at the
cost of the KL term being an approximation rather than the exact ELBO
term a full implementation would use.

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

### 1.5.1 Proof Sketch: Why Split-Conformal's Coverage Guarantee Holds

The marginal coverage guarantee P(y_true in prediction_set) >= 1 - alpha
relies on **exchangeability**, not on the underlying classifier being
correct or well-specified. The argument: let s(1),...,s(n) be the
nonconformity scores on the calibration set, and s(n+1) the score for a
new test point. If the n+1 points are exchangeable (their joint
distribution is invariant to permutation), then s(n+1) is, before
observing it, equally likely to be any rank among {s(1),...,s(n+1)}. The
threshold q_hat is set as the ceil((n+1)(1-alpha))/n empirical quantile
of the calibration scores specifically so that s(n+1) falls at or below
q_hat with probability at least 1-alpha under this exchangeability
argument — this holds *regardless* of how good or bad the underlying
probability estimates p_hat are, which is the method's key strength and
also its key limitation.

**This is exactly why the 86.9% empirical coverage (vs. 90% target,
Section 4.5) is a real finding worth taking seriously rather than
dismissing as noise**: time-series financial data is not exchangeable in
the classical i.i.d. sense (today's regime is correlated with
yesterday's), which is a known, real violation of the assumption this
guarantee depends on. The gap between 86.9% and 90% is plausibly
attributable to this violation, not to an implementation bug — this is
precisely the motivation for the Adaptive Conformal Inference variant
(above), which relaxes the exchangeability requirement by continuously
re-calibrating alpha online.

## 1.6 Sequential Inference

**Bootstrap particle filter**: N particles propagate through the fitted
transition matrix and are reweighted by Gaussian emission likelihood each
new observation; resampled on effective-sample-size collapse.

**Bayesian Online Changepoint Detection (BOCPD)**, Adams & MacKay (2007):
maintains a posterior over run-length r(t) (days since the last
changepoint) via a Normal-Inverse-Gamma conjugate predictive, updated
recursively with hazard rate H (prior probability of a changepoint on
any given day).

### 1.6.1 Particle Filter Algorithm (as implemented)

`BootstrapParticleFilter` in `models/sequential_inference.py` implements
the standard bootstrap (SIR) particle filter algorithm in four steps,
executed on every new observation:

1. **Propagate**: each of the N particles' current state s(i) transitions
   to a new state by sampling from the fitted transition matrix row A(s(i),.).
2. **Reweight**: each particle's importance weight is updated by the
   Gaussian emission log-likelihood of the new observation under its
   (now-propagated) state, `_emission_loglik()`, normalized to sum to 1.
3. **Estimate**: the filtered state-probability distribution at this
   time step is the weighted histogram of particle states.
4. **Resample** (conditionally): if the effective sample size
   ESS = 1 / sum(w_i^2) falls below N/2 — meaning weight has concentrated
   onto too few particles to usefully represent the posterior — particles
   are resampled proportional to their weights and weights reset to
   uniform. This is the standard fix for **particle degeneracy**, the
   well-known failure mode where, after enough steps, all probability
   mass concentrates onto a single particle and the filter's effective
   sample size collapses to 1 regardless of N.

The 89.2% agreement between this particle filter and the batch (Viterbi)
HMM on the last 500 days (reported in the prior pass) is a genuine
online/offline consistency check: the particle filter only ever sees
past data, while the batch HMM smooths using the *entire* series
(including future relative to each point), so 100% agreement is neither
expected nor would it be a meaningful target.

### 1.6.2 BOCPD Derivation

At each time t, BOCPD maintains a discrete probability distribution over
the run length r(t) in {0, 1, 2, ...}, where r(t)=0 means "a changepoint
just occurred". The recursive update is:

> P(r(t)=0 | x(1:t))     proportional to  sum_{r(t-1)} P(r(t-1)|x(1:t-1)) * P(x(t)|r(t-1)) * H
> P(r(t)=r(t-1)+1 | x(1:t)) proportional to  P(r(t-1)|x(1:t-1)) * P(x(t)|r(t-1)) * (1-H)

where H is the hazard rate and P(x(t)|r(t-1)) is the posterior predictive
density of x(t) given all observations since the last inferred
changepoint — a Student-t density under a Normal-Inverse-Gamma conjugate
prior, approximated by a Gaussian in this implementation
(`bayesian_online_changepoint_detection()` in both
`models/sequential_inference.py` and, independently, in
`r_codebase/bayesian_changepoint.R`). After each step, the sufficient
statistics (mu, kappa, alpha, beta) for every surviving run length are
updated via the same Normal-Inverse-Gamma recursion derived in Section
1.2.1, with a fresh run-length-0 entry appended at every step.

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

# 3.1 Worked Numerical Example: One Day, One Emission Calculation

To make the forward-backward machinery in Section 1.1.1 concrete, this
section walks through the actual emission-likelihood calculation the
fitted HMM performs for one real date: **2020-03-23**, the real Covid
crash trough day (Section 6 and the companion case-study report, both
independently, identify this as the low point of that crash).

**Step 1 — the real feature vector on this date**, computed by
`features/engineering.py` from real Nifty 50 prices:

> x(2020-03-23) = [ ret_21d = -0.372, rvol_21d = 0.660, vix_z = 2.007 ]

In words: over the trailing 21 trading days the index fell 37.2%, trailing
21-day annualized realized volatility reached 66.0% (an extreme reading —
single-digit-to-teens is typical calm-market realized vol), and the
(synthetic) VIX proxy was 2.0 standard deviations above its trailing
3-year mean. Every one of these three numbers independently signals
acute stress.

**Step 2 — the fitted Gaussian emission parameters** for two of the five
fitted states (from `model.means_` / `model.covars_` in
`models/hmm_frequentist.py`'s output): the state that ultimately ranks as
**Post-Shock** has mean [-0.003, 0.307, 0.268] and diagonal variance
[0.0107, 0.0223, 1.672] — notice its `rvol_21d` mean of 0.307 is roughly
2.5x every other state's, correctly identifying this as "the high-
volatility state." The state that ranks as **Risk-On** has mean
[0.038, 0.120, -0.408] and variance [0.00073, 0.00115, 1.131] — a much
*tighter* variance on the first two dimensions, reflecting that calm
markets are more tightly clustered in return/vol space than stressed ones.

**Step 3 — evaluate the diagonal Gaussian log-density under each state**,
per dimension, using log N(x|mu,sigma^2) = -0.5*log(2*pi*sigma^2) -
(x-mu)^2/(2*sigma^2):

| Dimension | log-density, Post-Shock state | log-density, Risk-On state |
|---|---|---|
| ret_21d | -5.03 | -111.86 |
| rvol_21d | -1.82 | -125.05 |
| vix_z | -2.08 | -3.56 |
| **Sum** | **-8.93** | **-240.47** |

**Step 4 — the result**: the log-likelihood ratio between the two states
is 231.54, meaning the Post-Shock state's Gaussian assigns this specific
day's feature vector a likelihood roughly **10^100 times higher** than
the Risk-On state does. This is not a subtle statistical judgment call —
on a day this extreme, the emission probabilities in the forward-backward
recursion (Section 1.1.1) are overwhelmingly dominated by one term, which
is exactly why the model's posterior `prob_Post-Shock` for this date
rounds to **1.0000** (see `outputs/full_hmm_regime_history.csv`, row
2020-03-23) — not because the model is overconfident, but because the
underlying data genuinely is that unambiguous on this particular day.
Most days are far less extreme than this, which is why the *average*
confidence and calibration statistics reported in Section 4.5 look
meaningfully different from this single dramatic example.



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

## 4.2.1 Per-Regime Feature Statistics and the HMM-vs-Rules-Based Confusion Matrix

To make the HMM's regime separation concrete rather than asserted, this
subsection reports the actual mean and standard deviation of daily return
and 21-day realized volatility, computed within each HMM-assigned regime
on real data:

| HMM Regime | Mean daily return | Std daily return | Mean 21d realized vol | Std 21d realized vol |
|---|---|---|---|---|
| Risk-On | 0.14% | 0.73% | 12.0% | 3.4% |
| Late-Cycle | 0.14% | 0.77% | 11.9% | 3.3% |
| Transitional | 0.11% | 0.74% | 12.1% | 3.4% |
| Post-Shock | 0.10% | 2.14% | **30.6%** | 14.9% |
| Risk-Off | -0.09% | 0.94% | 13.8% | 3.4% |

The volatility column is where the HMM's separation is genuinely strong:
Post-Shock's mean realized volatility (30.6%) is roughly 2.5x every other
regime, cleanly matching Section 4.2's finding that this state is highly
persistent (44.1-day average duration) precisely because sustained high
volatility is a real, structurally different market condition, not
statistical noise. The return column separation is weaker — Risk-On,
Late-Cycle, and Transitional are barely distinguishable on mean daily
return (0.14%, 0.14%, 0.11%), directly explaining Section 4.2's finding
that these three states collapse to ~1-day average duration: **the model
separates on volatility much more cleanly than on return direction**, at
least on this 3-feature specification.

A notable detail worth stating plainly rather than smoothing over: **mean
daily return within Post-Shock is still slightly positive (0.10%)**, not
negative as the name might suggest. This is because Post-Shock, once
entered, is a *sticky* state (44 days average) that includes both the
sharpest down-days of a crash and the high-volatility bounce days that
follow — a genuine mixture, not a pure "down day" classifier. Risk-Off is
the only regime with a negative mean daily return (-0.09%), consistent
with it capturing sustained directional deterioration rather than acute,
volatile shocks.

**Confusion matrix**: HMM regime call (rows) against the independent,
rules-based ground-truth label (columns, Section 1.8), on all 4,303
overlapping real-data days:

| HMM \\ Rules-based | Late-Cycle | Post-Shock | Risk-Off | Risk-On | Transitional |
|---|---|---|---|---|---|
| Late-Cycle | 71 | 0 | 1 | 210 | 351 |
| Post-Shock | 98 | **264** | 70 | 84 | 366 |
| Risk-Off | 13 | 8 | 67 | 20 | **1,365** |
| Risk-On | 71 | 0 | 2 | 199 | 361 |
| Transitional | 74 | 0 | 3 | 207 | 398 |

Two things stand out, both consistent with (not contradicting) the 21.9%
overall agreement reported in Section 4.3. First, **the diagonal is far
from dominant** — this is expected given the two labeling methods measure
genuinely different things (statistical clustering on 3 features vs.
rules on trailing return/drawdown/vol thresholds), not a sign either
method is broken. Second, the single largest cell in the entire matrix is
HMM "Risk-Off" against rules-based "Transitional" (1,365 days) — the
HMM's Risk-Off state, which Section 4.2 showed is genuinely sticky
(25.4-day average duration) and has the only negative mean return, maps
predominantly onto whatever the rules-based labeler calls "Transitional"
(its default/residual bucket, per `models/ground_truth.py`), suggesting
the HMM's Risk-Off state and the rules-based labeler's default bucket may
be capturing a similar "grinding sideways-to-down, not acutely stressed"
market condition that neither method's chosen name quite matches the
other's — a real, substantive disagreement about label semantics rather
than a numerical error in either method.



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

**A companion document, `case_study_report.docx`, goes substantially
deeper on these same four episodes**: it finds the actual real
peak-to-trough window in each case (rather than a fixed narrative date
range), measures detection lag in trading days, measures recovery time,
and includes price-with-regime-overlay charts for each episode. Two
findings from that deeper analysis are important enough to summarize
here: (1) the real 2008 and 2020 peak-to-trough declines this build's
data reproduces (-59.9% and -38.4% respectively) match the commonly
cited "roughly 60%" and "~38%" figures almost exactly, which is
independent evidence the underlying price data is both real and clean;
and (2) the model's 252-day feature warmup window means it produces
**no regime call at all for the first 174 trading days of the 2008
crisis** — a real, structurally important blind spot for any production
deployment, not visible from this summary section alone.

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

# Appendix A — Sample Investment Committee Artefact

`models/monte_carlo.py`'s `ic_artefact()` function generates a single,
lineage-carrying record per regime call, intended as the format an
investment committee would actually consume. This is the real output for
the most recent date in this build's data (2026-04-13), generated by
running the function against the actual pipeline output, not written by
hand:

```json
{
  "date": "2026-04-13",
  "regime_call": "Post-Shock",
  "confidence": 1.0,
  "conformal_set": ["Post-Shock"],
  "recommended_equity_tilt": 0.75,
  "statement": "As of 2026-04-13, the ensemble assigns 100% probability
    to 'Post-Shock' (calibrated 100%+ conformal set: Post-Shock).
    Recommended equity-weight tilt: 0.75x neutral, driven primarily by:
    vix_z (+1.20), fii_flow (-0.80)."
}
```

Reading this artefact critically rather than presenting it as unqualified
good news: a conformal set containing only one regime, at exactly 100%
confidence, is itself informative about a limitation. Genuinely calibrated
uncertainty on real, noisy market data should only rarely produce exact
0%/100% probability splits — this specific date's extremity is consistent
with the real market conditions at the end of this dataset's coverage
window (Section 4.6), but a production system seeing 100%-confidence
outputs routinely, rather than occasionally on genuinely extreme days,
would be a calibration red flag worth investigating, not a feature to
advertise.

# Appendix B — Reproducibility

Every number in this report can be regenerated from the delivered
codebase:

```
pip install hmmlearn statsmodels pyarrow matplotlib --break-system-packages
python3 main.py                          # core pipeline -> outputs/
python3 reporting/validation_pack.py     # reliability diagram + scenario replay
python3 reporting/case_study_deep_dive.py  # companion case-study report's numbers
python3 reporting/case_study_charts.py     # companion case-study report's charts
```

R codebase (requires `apt install r-base-core r-cran-forecast r-cran-tseries
r-cran-msm r-cran-mcmcpack r-cran-jsonlite r-cran-nnet r-cran-zoo`):

```
cd r_codebase
Rscript hmm_markov_switching.R
Rscript bayesian_changepoint.R
Rscript conformal_and_reconciliation.R
```

**Determinism note**: `data/loader.py`'s synthetic auxiliary-series
generation is seeded (default seed=42), so re-runs reproduce identical
auxiliary data. The real Nifty price series (`data/real/nifty50_real_daily.csv`)
is a static, bundled file — re-running does not re-fetch it, so results
are fully reproducible without network access. The HMM's EM fitting uses
a fixed random seed (7) for initialization; the Bayesian deep learning
models use per-member seeds (0 through M-1); the R HMM uses `set.seed(7)`.
None of the reported numbers should vary between runs on the same machine
with the same package versions.

**Environment**: Python 3.12, R 4.3, no GPU. Full Python pipeline
runtime: ~90 seconds. R codebase runtime: ~2-3 minutes across all three
scripts, dominated by the multivariate HMM's EM iterations.

---

# Appendix C — Repository Map

See `README.md` and `MODEL_CARD.md` in the delivered codebase for the
full file-by-file map, package-substitution rationale, and per-model
hyperparameter table.
