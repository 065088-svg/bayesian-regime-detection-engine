"""
models/sequential_inference.py
================================
Sequential / online regime inference (Day 10 deliverable).

  - BootstrapParticleFilter: propagates N particles through the fitted
    HMM transition matrix, reweights by the Gaussian emission likelihood
    of each new observation, resamples on effective-sample-size collapse.
  - bayesian_online_changepoint_detection(): BOCPD (Adams & MacKay 2007)
    on a 1-D statistic (here: 21-day return) — maintains a run-length
    posterior and reports P(changepoint) per day.
  - online_batch_reconciliation(): compares the particle filter's
    streaming regime call against the smoothed batch HMM call, flags days
    where they disagree by more than a threshold.
"""
import numpy as np


class BootstrapParticleFilter:
    def __init__(self, transition_matrix, emission_means, emission_stds, n_particles=2000, seed=21):
        self.T = transition_matrix
        self.means = emission_means   # (n_states, n_feat)
        self.stds = emission_stds     # (n_states, n_feat)
        self.n_states = transition_matrix.shape[0]
        self.n_particles = n_particles
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.integers(0, self.n_states, n_particles)
        self.weights = np.full(n_particles, 1.0 / n_particles)

    def _emission_loglik(self, state, obs):
        mu, sd = self.means[state], self.stds[state]
        return -0.5 * np.sum(((obs - mu) / sd) ** 2 + 2 * np.log(sd) + np.log(2 * np.pi))

    def step(self, obs):
        # propagate
        new_particles = np.array([
            self.rng.choice(self.n_states, p=self.T[s]) for s in self.particles
        ])
        # reweight
        logliks = np.array([self._emission_loglik(s, obs) for s in new_particles])
        logliks -= logliks.max()
        w = np.exp(logliks) * self.weights
        w /= w.sum()

        self.particles = new_particles
        self.weights = w

        ess = 1.0 / np.sum(w ** 2)
        if ess < self.n_particles / 2:
            idx = self.rng.choice(self.n_particles, self.n_particles, p=w)
            self.particles = self.particles[idx]
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

        state_probs = np.bincount(self.particles, weights=self.weights, minlength=self.n_states)
        return state_probs

    def run(self, obs_sequence):
        return np.array([self.step(o) for o in obs_sequence])


def bayesian_online_changepoint_detection(x: np.ndarray, hazard=1 / 250, mu0=0.0, kappa0=1.0,
                                           alpha0=1.0, beta0=1.0):
    """BOCPD with a Normal-Inverse-Gamma conjugate predictive on x.
    Returns P(changepoint at t) = posterior mass on run-length 0."""
    n = len(x)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    mu = np.array([mu0]); kappa = np.array([kappa0]); alpha = np.array([alpha0]); beta = np.array([beta0])
    cp_prob = np.zeros(n)

    for t in range(n):
        xt = x[t]
        # predictive prob under each run length (Student-t predictive, approximated Gaussian)
        pred_var = beta * (kappa + 1) / (alpha * kappa)
        pred_std = np.sqrt(pred_var)
        pred_prob = np.exp(-0.5 * ((xt - mu) / pred_std) ** 2) / (pred_std * np.sqrt(2 * np.pi))

        growth = R[t, :t + 1] * pred_prob * (1 - hazard)
        cp = np.sum(R[t, :t + 1] * pred_prob * hazard)

        R[t + 1, 1:t + 2] = growth
        R[t + 1, 0] = cp
        R[t + 1, :t + 2] /= R[t + 1, :t + 2].sum() + 1e-12

        cp_prob[t] = R[t + 1, 0]

        # update sufficient stats (append a new run-length-0 set, update existing)
        new_kappa = np.concatenate(([kappa0], kappa + 1))
        new_mu = np.concatenate(([mu0], (kappa * mu + xt) / (kappa + 1)))
        new_alpha = np.concatenate(([alpha0], alpha + 0.5))
        new_beta = np.concatenate(([beta0], beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))))
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

    return cp_prob


def online_batch_reconciliation(streaming_state: np.ndarray, batch_state: np.ndarray):
    agree = streaming_state == batch_state
    return dict(
        agreement_rate=float(agree.mean()),
        disagreement_idx=np.where(~agree)[0],
        n_disagreements=int((~agree).sum()),
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.loader import load_market_data
    from features.engineering import build_features
    from models.hmm_frequentist import run_frequentist_hmm, fit_hmm

    df = load_market_data()
    feat = build_features(df)
    feature_cols = ["ret_21d", "rvol_21d", "vix_z"]
    sub = feat[feature_cols].dropna()
    X = sub.values

    hmm_out = run_frequentist_hmm(feat, feature_cols=feature_cols)
    model = hmm_out["model"]
    rank_map = hmm_out["rank_map"]
    n_states = 5
    means_ranked = np.zeros((n_states, X.shape[1]))
    stds_ranked = np.zeros((n_states, X.shape[1]))
    for orig_s, rank in rank_map.items():
        means_ranked[rank] = model.means_[orig_s]
        stds_ranked[rank] = np.sqrt(model.covars_[orig_s]) if model.covars_[orig_s].ndim == 1 else np.sqrt(np.diag(model.covars_[orig_s]))
    T_ranked = np.zeros((n_states, n_states))
    inv_rank = {v: k for k, v in rank_map.items()}
    for r1 in range(n_states):
        for r2 in range(n_states):
            T_ranked[r1, r2] = model.transmat_[inv_rank[r1], inv_rank[r2]]
    T_ranked = T_ranked / T_ranked.sum(axis=1, keepdims=True)

    pf = BootstrapParticleFilter(T_ranked, means_ranked, stds_ranked, n_particles=1000)
    pf_probs = pf.run(X[-500:])
    pf_state = pf_probs.argmax(axis=1)

    batch_state = hmm_out["result"]["state"].values[-500:]
    recon = online_batch_reconciliation(pf_state, batch_state)
    print("=== Particle filter vs batch HMM reconciliation (last 500 days) ===")
    print(f"Agreement rate: {recon['agreement_rate']:.3f}  n_disagreements: {recon['n_disagreements']}")

    print("\n=== BOCPD on 21-day returns ===")
    cp_prob = bayesian_online_changepoint_detection(sub["ret_21d"].values[-1000:], hazard=1 / 250)
    top_cp_days = np.argsort(-cp_prob)[:5]
    print("Top-5 highest-probability changepoint days (index into last 1000):", sorted(top_cp_days.tolist()))
    print("Max P(changepoint):", round(cp_prob.max(), 4), " Mean P(changepoint):", round(cp_prob.mean(), 4))
