# Bayesian Regime Detection Engine — Working Core Build

Project brief: Zetheta Algorithms Private Limited — "Bayesian Regime Detection
Engine for Equity Direction Forecasting" (15-day Financial Data Analyst
assessment).

## Ready to submit

This directory is a git repository with the full build committed
(`git log` shows one commit). To submit: `git remote add origin
<your-github-url>` then `git push -u origin main`. That push is the one
step only you can do — everything else in this checklist is done.

## What this is

A working implementation covering 5 of the brief's 6 deliverables, built
and executed across four passes in one extended sandboxed session. Every
number anywhere in this repo — code, report, deck, R output — was
produced by actually running something. Nothing is narrated or faked.

**Pass three integrated real market data.** The core Nifty 50 price
series is real history (2007-09-17 to 2026-04-13), sourced from a public
dataset and spot-checked against the documented record (its 2020
Covid-crash closing low of 7,610.25 on 2020-03-23 matches the real,
widely-reported figure). Auxiliary series (VIX, FII/DII, INR, Gilt)
remain synthetic — no real public source for those specific series was
reachable from this sandbox — but are now generated to move consistently
with the real price path.

**Pass four (current) added a standalone case-study report and expanded
the main report.** `reporting/case_study_report.docx` (7 pages) is a new,
deeper deliverable: real peak-to-trough dates found by search (not
narrative date ranges), detection-lag-in-trading-days, recovery time, and
a price+regime-overlay chart per episode. It found that this build's
252-day feature warmup means the model produces **no regime call for the
first 174 trading days of the 2008 crisis** — a real, structurally
important finding, visible directly in the chart. The main report grew
from 10 to 18 pages with genuine new content: full mathematical
derivations (Baum-Welch, Dirichlet/NIG conjugacy, conformal coverage
proof, BOCPD), a worked numerical example on the real 2020-03-23
Covid-trough date, a real Investment Committee artefact example, and a
reproducibility appendix.

| Deliverable | Status |
|---|---|
| 1. Main report | `reporting/main_report.docx` — 18 pages (up from 10), full derivations and worked examples added this pass. Still short of the 40-page target but every required section is present in depth. |
| 2. Python codebase | Built and runs end-to-end (`python3 main.py`, ~90s) on real Nifty price data. Only gap: foundation-model integration (Chronos/TimesFM) — no Hugging Face access from this sandbox. |
| 3. R codebase | Built and runs (`r_codebase/`) on the same real data — HMM (multivariate, matched to Python's feature set), Bayesian regression w/ MCMC diagnostics, BOCPD, conformal, cross-language reconciliation. CRAN unreachable; apt-packaged substitutes used and documented. |
| 4. Backtesting & simulation | Built and re-run on real data — regime-tilt backtester (Information Ratio flipped from -1.22 to **+0.071**), Monte Carlo VaR/CVaR, IC artefact generator, scenario replay harness with real P&L for all 4 crisis episodes, **plus** the new standalone case-study deep-dive (`reporting/case_study_report.docx`). |
| 5. Model card & validation pack | Built — `MODEL_CARD.md`, reliability diagrams, ECE, online/batch reconciliation, cross-language check — all re-run on real data. No MCMC diagnostics on the Python side (conjugate posteriors instead of NUTS — see below). |
| 6. Presentation | 18-slide deck (`reporting/deck/regime_engine_deck.pptx`), fully updated with real-data numbers, validated and visually QA'd. **Demo video not recorded — outside this tool's capability.** Full script/storyboard provided instead: `reporting/demo_video_script.md`. |

**Two things this build genuinely cannot do**, stated plainly rather than
worked around: recording a video, and pushing/transferring the GitHub
repository itself (the repo is committed and ready — only the `git push`
requires your credentials).

**Real data changed real conclusions, not just cosmetics**: the backtest's
Information Ratio flipped sign, the HMM converges cleanly on real data
(it didn't on the synthetic panel), fixing the R/Python HMM feature
mismatch — expected to *improve* cross-language agreement — instead made
it worse (25.7% → 2.8%, revealing EM local-optima instability as the real
issue), and the case-study deep-dive found the 2008 crisis has a 174-day
model blind spot invisible in the earlier summary-level case-study
section. All of this is reported in the relevant report sections rather
than smoothed over.

The **40-page report length target was still not hit** (delivered: 18
pages across the main report, plus 7 more in the companion case-study
report — 25 pages combined). Content coverage is complete across both
documents.

Where a corner was cut for sandbox-time or environment reasons (no
PyMC/NUTS, no torch, no CRAN access, no Hugging Face access), the
substitute used is named explicitly in the relevant module's docstring,

in `MODEL_CARD.md`, and in the report's Limitations section — plus what a
production upgrade looks like.

## Architecture

```
data/loader.py              REAL Nifty 50 daily prices (2007-2026, see
                             data/real/nifty50_real_daily.csv) + synthetic
                             auxiliary series (VIX, Midcap/Smallcap, USDINR,
                             Gilt, FII/DII) regime-conditioned on the real
                             price path.
features/engineering.py     28 numeric features: returns/trend, vol, cap-
                             segment, flow, macro. TDA + GNN features stubbed
                             with a documented interface (Section A7 needs
                             data this sandbox doesn't have).
models/hmm_frequentist.py   5-state Gaussian HMM (hmmlearn), BIC model
                             selection across k=3/5/7, transition matrix,
                             regime durations.
models/hmm_bayesian.py      Bayesian HMM via exact Dirichlet / Normal-
                             Inverse-Gamma conjugate posteriors (not NUTS —
                             see docstring) -> real credible intervals.
models/rs_var.py            Markov-switching baseline (statsmodels) +
                             regime-conditioned multivariate VAR(1) with
                             bootstrap CIs + impulse responses.
models/bayesian_dl.py       MC-Dropout MLP, mean-field variational MLP
                             (Bayes-by-Backprop), deep ensemble (M=6-10) —
                             all hand-rolled in numpy (no torch install).
                             Uncertainty decomposition (epistemic/aleatoric).
models/conformal.py         Split-conformal, Adaptive Prediction Sets,
                             online Adaptive Conformal Inference, reliability
                             diagrams, ECE.
models/sequential_inference.py  Bootstrap particle filter + Bayesian Online
                             Changepoint Detection + batch/online reconciliation.
models/ensemble.py          Bayesian Model Averaging, constrained stacking,
                             WAIC-style model comparison, combined output
                             contract schema.
models/monte_carlo.py       Regime-conditioned Monte Carlo (VaR/CVaR) +
                             allocation-tilt backtester (2019-2024, IR,
                             tracking error, regime-conditioned drawdowns) +
                             Investment Committee artefact generator.
dashboard/dashboard.html    Self-contained (data inlined) monitoring
                             dashboard: regime timeline, transition heatmap,
                             backtest curves, model comparison, calibration.
main.py                     Orchestrates the full pipeline, writes outputs/.
reporting/validation_pack.py  Reliability diagrams + scenario replay harness
                             for 4 named crisis episodes (2008/2013/2020/2024).
reporting/main_report.md/.docx  The main report (Deliverable 1), 18 pages.
reporting/case_study_report.md/.docx  Standalone case-study deep-dive
                             (Day 12 deliverable): real peak-to-trough
                             search, detection lag, recovery time, and a
                             regime-overlay chart per episode.
reporting/case_study_deep_dive.py  Computes the case-study report's numbers.
reporting/case_study_charts.py  Generates the case-study report's charts.
reporting/deck/build_deck.js + regime_engine_deck.pptx  18-slide presentation
                             (Deliverable 6, minus the demo video).
reporting/demo_video_script.md  Full 10-minute script/storyboard, since
                             this tool cannot record actual video.
r_codebase/                 R implementation (Deliverable 3) — HMM,
                             Bayesian regression + MCMC diagnostics, BOCPD,
                             conformal prediction, cross-language check.
                             See r_codebase/README.md for package
                             substitution notes.
```

Run the Python pipeline:
```
pip install hmmlearn statsmodels pyarrow matplotlib --break-system-packages
python3 main.py
python3 reporting/validation_pack.py
python3 reporting/case_study_deep_dive.py
python3 reporting/case_study_charts.py
```

Run the R codebase (needs `apt install r-base-core r-cran-forecast
r-cran-tseries r-cran-msm r-cran-mcmcpack r-cran-jsonlite r-cran-nnet
r-cran-zoo`):
```
cd r_codebase
Rscript hmm_markov_switching.R
Rscript bayesian_changepoint.R
Rscript conformal_and_reconciliation.R
```

## Foundation models (Day 8) — not integrated

Chronos / TimesFM / Lag-Llama require downloading pretrained weights from
Hugging Face at runtime; this sandbox's network allowlist doesn't include
huggingface.co, so no foundation-model embeddings were generated. The
hybrid pattern (`foundation embedding -> Bayesian classification head`) is
architecturally identical to what `models/bayesian_dl.py`'s heads already
do on top of the hand-engineered features — swapping in a real Chronos
embedding as the head's input is a data-layer change, not a modeling one.

## What's genuinely not here

- **10-minute demo video**: outside this tool's capability (no video or
  screen-recording). A script/storyboard can be written on request.
- **GitHub repo transfer to @ZethetaIntern**: requires your own GitHub
  account/auth — only you can do this step.
- **Foundation models** (Chronos/TimesFM/Lag-Llama): no network path to
  Hugging Face from this sandbox. Architecturally slotted into
  `models/bayesian_dl.py` as an additional input, not wired in.
- **PyMC/NUTS**: not installed (sandbox time budget). Bayesian HMM uses
  exact conjugate posteriors instead — see `models/hmm_bayesian.py`
  docstring. This means no R-hat/ESS/divergence diagnostics on the Python
  side specifically (the R side, via MCMCpack, does have real Geweke/ESS
  diagnostics — see `r_codebase/bayesian_changepoint.R`).
- **TDA (gtda) / GNN (torch_geometric) features**: stubbed with a
  documented interface in `features/engineering.py` — need data
  (multi-asset correlation tensors, a sector graph) this build doesn't have.
- **Real market data**: the core Nifty 50 price series is now real
  (2007-2026, see `data/loader.py` docstring for source and validation).
  Auxiliary series (VIX, FII/DII, INR, Gilt) remain synthetic — no real
  public source for these specific series was reachable from this
  sandbox. No number derived from VIX/FII/DII/INR/Gilt levels should be
  quoted as real; numbers derived from price/returns/volatility now can be.
- **Report length**: 10 pages delivered against a 40-page target. Content
  coverage is complete (every required section exists with real numbers);
  depth/length is the gap.

## Known rough edges in this pass

- **~~Circular ensembling~~ — fixed (pass two), holds on real data
  (pass three).** `models/ground_truth.py` provides an independent,
  rules-based regime label (21.9% agreement with the HMM on real data,
  confirming independence). Conformal coverage: 86.9% (vs 90% target);
  ECE: 0.1066 on real data (vs 0.0584 on the earlier synthetic-data run —
  real market behavior is *less* well captured by this ensemble than
  synthetic data suggested, a genuine and informative finding).
- **R/Python HMM reconciliation got worse after fixing feature
  alignment** (25.7% → 2.8%) — see `r_codebase/README.md` and report
  Section 5. This revealed EM local-optima instability as the real
  underlying issue, not feature mismatch.
- BOCPD changepoint probabilities are very small in absolute terms in this
  run (hazard-rate scaling) — the run-length posterior itself is sound,
  but the hazard prior likely needs tuning against real event dates before
  the "fires on 2018/2020, not on noise" validation in the brief would pass.
- A **production-grade independent label** would use real expert-annotated
  history rather than rules-on-real-price-data — the current label is
  independent in methodology and now partially in data source (real
  prices), but still not human-annotated.
