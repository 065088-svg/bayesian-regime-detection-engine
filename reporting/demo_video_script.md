# Demo Video Script / Storyboard

**Why this exists instead of a video file:** recording, screen-capturing,
and narrating a video is outside this tool's capability — there is no
video or audio production tool available in this environment. This
script is written so a human presenter (or a screen-recording tool fed
this script) can produce the actual 10-minute video in one take against
the real deliverables in this repository. Every screen reference below
points at a file or slide that actually exists.

**Total runtime target: 10 minutes. Timestamps are cumulative.**

---

### [0:00–0:45] Cold open — the honesty framing

**Screen:** Title slide (`regime_engine_deck.pptx`, Slide 1)

**Say:**
> "This is a working core build of the Bayesian Regime Detection Engine
> for Zetheta Algorithms — a 5-state regime classifier for Indian
> equities, ensembling an HMM, Bayesian deep learning, and conformal
> prediction. Two things up front, because the rest of this video only
> means something if these are said first: everything you're about to
> see runs on synthetic, regime-calibrated data — not real Nifty
> history, because this build environment has no network path to a live
> market data feed. And when I found a real flaw in my own first-pass
> methodology, I fixed it and I'm going to show you both the before and
> after, including the parts where the fix made a number look worse."

---

### [0:45–1:30] The problem and the taxonomy

**Screen:** Slides 3–4 (problem statement, five-state taxonomy)

**Say:**
> "The core bet is that direction — which regime we're in — is more
> learnable than magnitude. Five states: Risk-On, Late-Cycle,
> Transitional, Post-Shock, Risk-Off. Labels are assigned by ranking
> fitted HMM states on mean return, so the label order is stable even
> though the raw state indices from the model are arbitrary."

---

### [1:30–2:30] Architecture walkthrough

**Screen:** Slide 5 (pipeline architecture), then switch to the repo
file tree (`view /home/claude/regime_engine` or `ls -R` in a terminal)

**Say:**
> "Seven stages: data, features, HMM plus Bayesian HMM, regime-switching
> VAR plus Bayesian deep learning, ensembling, conformal calibration
> plus Monte Carlo, and the dashboard. Every box here is real, executed
> code — not pseudocode. Let me run the whole thing live."

**Action:** In a terminal, run:
```
cd regime_engine && python3 main.py
```
Let it run to completion on screen (~90 seconds) — this is the single
most convincing five seconds of "this isn't a mockup" available, so
don't cut away from the terminal output.

---

### [2:30–4:00] Data honesty + feature engineering

**Screen:** Slide 6 (data layer warning box), then open
`data/loader.py` briefly to show the docstring

**Say:**
> "Before anything else: this panel is synthetic. Fifteen years,
> regime-conditioned generation — each of the five regimes has its own
> drift, volatility, VIX, and flow parameterization calibrated to
> realistic order-of-magnitude Indian-market statistics. It is not real
> history. `load_market_data` is a one-function swap point for a real
> feed — I want that visible in the code, not just asserted in a slide."

**Screen:** Slide 7 (feature engineering)

**Say:**
> "Twenty-eight features across five families — return and trend,
> volatility, cap-segment spread, flow, and macro. Topological and
> sector-GNN features are architecturally stubbed — they need data this
> sandbox doesn't have populated, and the stub interface is documented
> rather than silently dropped."

---

### [4:00–5:30] HMM, Bayesian HMM, and the honest substitution

**Screen:** Slide 8 (HMM methodology), then Slide 9 (BIC + duration results)

**Say:**
> "The brief specifies PyMC with NUTS sampling for the Bayesian HMM —
> two thousand draws, four chains. I didn't install that; it wasn't a
> safe bet on this session's time budget. Instead, the Bayesian HMM uses
> an exact conjugate posterior — Dirichlet on transitions,
> Normal-Inverse-Gamma on emissions. No MCMC needed, because conjugacy
> is exact. Trade-off: no R-hat or ESS diagnostics, because there's no
> sampling error to diagnose in an exact method. Five hundred posterior
> draws give a mean credible-interval width of 1.84% across the
> transition matrix — genuinely tight, given about 4,100 days of data.
>
> One honest finding on this slide: Risk-On and Late-Cycle collapse to
> about a one-day average duration in the fitted 5-state model. That's
> the model over-fragmenting on a three-feature set — a real limitation,
> not something I'm smoothing over."

---

### [5:30–6:30] RS-VAR, Bayesian deep learning, conformal, sequential inference

**Screen:** Slides 10–11

**Say:**
> "Regime-switching VAR is fit per-regime by OLS with a bootstrap for
> coefficient uncertainty — that's the posterior mode under a flat
> prior, not full Bayesian VAR, again for the same time-budget reason.
> The Bayesian deep learning layer — MC-Dropout, a variational network,
> and a deep ensemble — is hand-rolled in numpy. No PyTorch install in
> this environment. All three decompose predictive uncertainty into
> epistemic and aleatoric components.
>
> Conformal prediction — split-conformal, adaptive prediction sets, and
> online adaptive conformal inference — gives calibrated prediction sets
> rather than a single point guess. And I want to flag this now, on
> camera, before I show the calibration slide: coverage came in below
> the 90% target. I'm not going to bury that."

---

### [6:30–8:00] The fix — walking through the circularity and its repair

**Screen:** Slide 12 (Ensembling Design — Fixed), then open
`models/ground_truth.py` on screen

**Say:**
> "Here's the part of this video I think matters most. In the first
> pass of this build, the ensemble members were trained and evaluated
> against the frequentist HMM's own output. That's circular — the HMM
> was effectively being asked to predict itself, and Bayesian Model
> Averaging trivially collapsed onto it with a weight of 1.0.
>
> I caught that, and this file — `models/ground_truth.py` — is the fix.
> It's a rules-based regime labeler: deterministic thresholds on
> trailing 60-day return, drawdown from a rolling peak, and realized
> volatility. It never touches the HMM's fitted parameters. The HMM and
> this independent label agree only 15.8% of the time, which is exactly
> what you want to see — if they'd agreed at 95%+, the label wouldn't
> actually be independent.
>
> Re-running the ensemble against this label, Bayesian Model Averaging
> now collapses onto MC-Dropout instead of the HMM — same
> winner-take-all mechanic, different winner, because MC-Dropout was
> trained directly on the new label and the HMM, being unsupervised,
> wasn't. Constrained stacking is the number I'd actually trust for
> deployment: nine, forty-seven, forty-four percent across HMM,
> MC-Dropout, and DeepEnsemble — real weight on all three.
>
> And here's the part that's easy to hide and I'm not hiding it:
> calibration got *less* flattering once I fixed this. Coverage improved
> from eighty-one point seven to eighty-seven point two percent — better
> — but Expected Calibration Error roughly tripled, from 0.017 to 0.058.
> The circular setup was quietly understating how miscalibrated this
> system actually is. That's the value of fixing it: not a better
> number, a more honest one."

---

### [8:00–9:00] Backtest, Monte Carlo, case studies

**Screen:** Slides 15–16

**Say:**
> "The allocation-tilt overlay backtested against buy-and-hold from 2019
> to 2024 on the synthetic panel: twenty-one and a half percent CAGR for
> the strategy against twenty-seven point four for buy-and-hold — the
> overlay underperforms on return, with a smaller max drawdown, and a
> negative Information Ratio. I'm reporting that as computed. It would
> have been easy to tune the tilt rules until this looked better; I
> didn't.
>
> Four case studies replay real, documented Indian-market stress
> episodes — 2008, 2013's taper tantrum, 2020's Covid crash, and 2024's
> election volatility — against the synthetic panel standing in for each
> window. The 2008 episode has no result at all, because the synthetic
> panel starts in 2010. That's a reported gap, not a silent one."

---

### [9:00–9:40] R codebase and cross-language reconciliation

**Screen:** `r_codebase/README.md`, then terminal running
`Rscript hmm_markov_switching.R`

**Say:**
> "The brief also asks for an R codebase — depmixS4, MSwM, Stan. None of
> those are reachable from this sandbox; there's no path to CRAN, and
> Stan's compile time is a bad bet mid-session. What's here instead:
> a hand-rolled Gaussian HMM in base R, a real Gibbs-sampled Bayesian
> regression via MCMCpack with genuine Geweke and effective-sample-size
> diagnostics, hand-rolled BOCPD, and a hand-rolled split-conformal
> classifier. Cross-checked against the Python HMM, agreement is only
> twenty-five point seven percent — because the two are fit on different
> feature sets. That's not yet the like-for-like reconciliation the
> brief wants; it's the top item on my follow-up list."

---

### [9:40–10:00] Close — what's real, what's missing, what's next

**Screen:** Slide 18 (honest gap list)

**Say:**
> "To close: this is a working core build, not the complete submission.
> Built and real: the Python engine, the R codebase, the validation
> pack, this report, this deck. Not built: this video wasn't recorded by
> me — I wrote the script, a human needs to read it on camera — and I
> cannot transfer this repository to a GitHub account that isn't mine.
> Everything else in the honest gap list on screen now is a real,
> specific, fixable next step — starting with getting this system onto
> real market data before any number in it should inform an actual
> allocation decision."

---

## Production notes for whoever records this

- Screen recording should show **actual terminal output** during the
  `python3 main.py` run (2:30 mark) and the R script run (9:00 mark) —
  these are the moments that prove the deliverable isn't a mockup.
- All slide numbers above match `regime_engine_deck.pptx` as delivered.
- Total spoken word count is calibrated for ~135 words/minute delivery
  to land at 10 minutes; trim the RS-VAR/BDL section first if running
  long, since it's the most technical and least novel relative to the
  HMM section.
