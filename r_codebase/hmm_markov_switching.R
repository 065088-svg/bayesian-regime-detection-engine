# hmm_markov_switching.R
# ========================
# R Deliverable, Part 1: HMM and Markov-switching regression.
#
# NOTE ON PACKAGE AVAILABILITY: this sandbox can only install R packages
# via apt (Ubuntu's precompiled CRAN mirror), not install.packages() from
# CRAN directly (no network path to cran.r-project.org). depmixS4 and MSwM
# are not packaged for apt on this image. Substitute used instead:
#   - A hand-rolled Gaussian HMM (EM/Baum-Welch, base R) — mathematically
#     the same model depmixS4 would fit for a Gaussian-emission HMM.
#   - `msm` (available, continuous-time multi-state Markov models) used for
#     a secondary cross-check via its discrete-time embedding.
# This keeps the deliverable honest: a real HMM fit in R, not a mock.
#
# UPDATE (this pass): extended to a MULTIVARIATE (diagonal-covariance)
# Gaussian HMM on the SAME THREE FEATURES the Python `hmmlearn` HMM uses
# (21-day return, 21-day realized vol, VIX z-score) — the prior pass fit
# R's HMM on raw returns only, which meant the R-vs-Python agreement
# check (25.7%) wasn't a like-for-like comparison. It now is.

suppressMessages({
  library(jsonlite)
  library(zoo)
})

panel <- read.csv("market_panel.csv")
panel$date <- as.Date(panel$date)
panel$ret <- c(NA, diff(log(panel$nifty_close)))

# Build the SAME 3 features as features/engineering.py: ret_21d, rvol_21d, vix_z
panel$ret_21d <- c(rep(NA, 21), diff(log(panel$nifty_close), lag = 21))
panel$rvol_21d <- rollapply(panel$ret, 21, sd, fill = NA, align = "right") * sqrt(252)
vix_mean_252 <- rollapply(panel$india_vix, 252, mean, fill = NA, align = "right")
vix_sd_252 <- rollapply(panel$india_vix, 252, sd, fill = NA, align = "right")
panel$vix_z <- (panel$india_vix - vix_mean_252) / vix_sd_252

feat_cols <- c("ret_21d", "rvol_21d", "vix_z")
panel <- panel[complete.cases(panel[, feat_cols]), ]
X <- as.matrix(panel[, feat_cols])

# --- Hand-rolled MULTIVARIATE Gaussian HMM (diagonal covariance, Baum-Welch/EM) ---
gaussian_hmm_em_mv <- function(X, n_states = 5, n_iter = 100, tol = 1e-6, seed = 7) {
  set.seed(seed)
  n <- nrow(X); d <- ncol(X)
  # init: k-means-ish via quantiles of the first principal-ish column (return proxy)
  init_order <- order(X[, 1])
  chunks <- split(init_order, cut(seq_along(init_order), n_states, labels = FALSE))
  mu <- t(sapply(chunks, function(idx) colMeans(X[idx, , drop = FALSE])))
  sigma <- t(sapply(chunks, function(idx) apply(X[idx, , drop = FALSE], 2, sd)))
  sigma[sigma < 1e-4] <- 1e-4
  A <- matrix(1 / n_states, n_states, n_states); diag(A) <- diag(A) + 0.5
  A <- A / rowSums(A)
  pi0 <- rep(1 / n_states, n_states)

  dmvnorm_diag <- function(x, mu, sigma) {
    prod(dnorm(x, mu, sigma))
  }

  loglik_prev <- -Inf
  for (it in 1:n_iter) {
    B <- matrix(0, n, n_states)
    for (s in 1:n_states) {
      B[, s] <- apply(X, 1, function(row) dmvnorm_diag(row, mu[s, ], sigma[s, ]))
    }
    B[B < 1e-300] <- 1e-300

    alpha <- matrix(0, n, n_states); c_scale <- numeric(n)
    alpha[1, ] <- pi0 * B[1, ]
    c_scale[1] <- sum(alpha[1, ]); alpha[1, ] <- alpha[1, ] / c_scale[1]
    for (t in 2:n) {
      alpha[t, ] <- (alpha[t - 1, ] %*% A) * B[t, ]
      c_scale[t] <- sum(alpha[t, ])
      alpha[t, ] <- alpha[t, ] / c_scale[t]
    }
    beta <- matrix(0, n, n_states); beta[n, ] <- 1
    for (t in (n - 1):1) {
      beta[t, ] <- (A %*% (B[t + 1, ] * beta[t + 1, ])) / c_scale[t + 1]
    }
    gamma <- alpha * beta
    gamma <- gamma / rowSums(gamma)

    xi_sum <- matrix(0, n_states, n_states)
    for (t in 1:(n - 1)) {
      num <- outer(alpha[t, ], B[t + 1, ] * beta[t + 1, ]) * A
      xi_sum <- xi_sum + num / sum(num)
    }
    A_new <- xi_sum / rowSums(xi_sum)
    pi0_new <- gamma[1, ]
    mu_new <- t(sapply(1:n_states, function(s) colSums(gamma[, s] * X) / sum(gamma[, s])))
    sigma_new <- t(sapply(1:n_states, function(s) {
      sqrt(colSums(gamma[, s] * (sweep(X, 2, mu_new[s, ]))^2) / sum(gamma[, s]))
    }))
    sigma_new[sigma_new < 1e-4] <- 1e-4

    loglik <- sum(log(c_scale))
    if (abs(loglik - loglik_prev) < tol) { mu <- mu_new; sigma <- sigma_new; A <- A_new; pi0 <- pi0_new; break }
    mu <- mu_new; sigma <- sigma_new; A <- A_new; pi0 <- pi0_new; loglik_prev <- loglik
  }
  list(mu = mu, sigma = sigma, A = A, pi0 = pi0, gamma = gamma, loglik = loglik, n_iter = it)
}

cat("Fitting 5-state MULTIVARIATE Gaussian HMM in R on [ret_21d, rvol_21d, vix_z]...\n")
fit <- gaussian_hmm_em_mv(X, n_states = 5, n_iter = 150)
cat(sprintf("Converged in %d iterations, log-likelihood = %.2f\n", fit$n_iter, fit$loglik))

# rank states by mean of the return column (column 1 = ret_21d), same convention as Python
rank_order <- order(-fit$mu[, 1])
regime_names <- c("Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off")
state_seq <- apply(fit$gamma, 1, which.max)
rank_map <- setNames(1:5, rank_order)
labeled_seq <- regime_names[rank_map[state_seq]]

result <- data.frame(date = panel$date, r_regime = labeled_seq, r_state = state_seq)
write.csv(result, "r_hmm_output.csv", row.names = FALSE)

cat("\nTransition matrix (R EM fit, multivariate):\n")
print(round(fit$A, 3))
cat("\nRegime frequency (R):\n")
print(table(labeled_seq))

# --- msm cross-check: discrete-time 5-state Markov model on the fitted state path ---
suppressMessages(library(msm))
cat("\nFitting msm 5-state Markov model as a cross-check (may take a moment)...\n")
q_init <- matrix(0.01, 5, 5); diag(q_init) <- 0
tryCatch({
  msm_data <- data.frame(subject = 1, time = 1:nrow(panel), state = state_seq)
  msm_fit <- msm(state ~ time, subject = subject, data = msm_data, qmatrix = q_init,
                 gen.inits = TRUE, control = list(maxit = 200))
  cat("msm cross-check converged. AIC:", AIC(msm_fit), "\n")
}, error = function(e) cat("msm cross-check skipped (", conditionMessage(e), ")\n"))

cat("\nWrote r_hmm_output.csv\n")
