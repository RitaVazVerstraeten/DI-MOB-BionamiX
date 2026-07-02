// Reduced (non-DLNM) prototype for ICAR-structured AR(1) innovations. Same
// distributed-lag structure as hierarchical_state_space_AR_perCMF.stan, but the
// AR(1) innovation at each time slice is spatially smoothed via ICAR instead of
// being i.i.d. across CMFs (see hierarchical_state_space_AR_perCMF_ICARinnov_DLNM_ix.stan
// for the full write-up of why and the DLNM+interaction version of this model).
//
// Use this smaller model to sanity-check that the ICAR-on-AR1-innovations
// construction actually samples well (rho and the ICAR innovation scale separate
// cleanly, no divergences) before paying for a full DLNM+ix fit.

data {
  int<lower=1> N;
  array[N] int<lower=0> y;
  array[N] int<lower=0> n_bt;
  int<lower=1> K;
  int<lower=1> Lp1;
  matrix[N, K*Lp1] X_lag_flat;
  int<lower=1> Ku;
  matrix[N, Ku] X_unlagged;
  int<lower=1> B;
  int<lower=1> T;
  array[N] int<lower=1,upper=B> block;
  array[N] int<lower=1,upper=T> time;
  array[N] int<lower=0> C_bt;

  // ICAR neighbourhood graph (shared across all T time slices)
  int<lower=0> N_edges;
  array[N_edges] int<lower=1,upper=B> node1;
  array[N_edges] int<lower=1,upper=B> node2;
  // Geometric mean of the ICAR precision pseudoinverse diagonal (Riebler et al. 2016).
  real<lower=0> scaling_factor;

  int<lower=0,upper=1> fix_phi;    // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;          // value used when fix_phi = 1 (ignored otherwise)
  real<lower=0> kappa;
}

parameters {
  real alpha;
  matrix[K, Lp1] w;
  vector[Ku] w_unlagged;
  matrix[B, T] v_raw;              // per-time-slice ICAR-structured innovations (non-centred)
  real<lower=0> tau;               // marginal stationary SD of the AR(1) process
  real<lower=-1,upper=1> rho;      // shared AR(1) coefficient
  real<lower=0> delta1;
  real<lower=0> phi_raw;           // used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  // Derived innovation SD: near rho=1 sigma_v->0 (smooth process), no blow-up.
  real<lower=0> sigma_v = tau * sqrt(fmax(1e-8, 1.0 - rho * rho));

  // Rescale each time slice's raw ICAR field to unit marginal variance before it
  // enters the AR(1) recursion below.
  matrix[B, T] v_raw_scaled = v_raw / sqrt(scaling_factor);

  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  matrix[B, T] v;                  // per-CMF AR(1) states, spatially-correlated shocks
  vector[N] x_effect;

  // 1. Per-CMF AR(1): stationary initialisation using tau (marginal SD).
  for (b in 1:B) {
    v[b, 1] = tau * v_raw_scaled[b, 1];
    for (t in 2:T)
      v[b, t] = rho * v[b, t-1] + sigma_v * v_raw_scaled[b, t];
  }

  // 2. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 3. Linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N)
    eta[i] = alpha + x_effect[i] + v[block[i], time[i]];
  p_bt = inv_logit(eta);

  // 4. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * log(C_bt[i]));
    } else {
      p_R[i] = p_bt[i];
    }
  }

  // 5. Effective observation probability
  for (i in 1:N) {
    if (n_bt[i] == 0) {
      omega[i] = 0;
      pi[i] = 0;
    } else if (C_bt[i] > 0) {
      omega[i] = fmin(1.0, (kappa * C_bt[i]) / n_bt[i]);
      pi[i] = (1 - omega[i]) * p_bt[i] + omega[i] * p_R[i];
    } else {
      omega[i] = 0;
      pi[i] = p_bt[i];
    }
  }
}

model {
  alpha ~ normal(-7.0, 1.5);

  to_vector(w) ~ normal(0, 1.0);
  w_unlagged   ~ normal(0, 0.5);

  // ICAR smoothness prior applied independently at each time slice (see the
  // DLNM+ix version of this file for the full rationale).
  for (t in 1:T) {
    vector[B] v_raw_t = col(v_raw, t);
    target += -0.5 * dot_self(v_raw_t[node1] - v_raw_t[node2]);
    sum(v_raw_t) ~ normal(0, 0.001 * B);
  }

  tau          ~ normal(0, 1.0);
  rho          ~ normal(0.4, 0.1);
  delta1       ~ normal(0, 0.5);
  if (fix_phi == 0) phi_raw ~ gamma(13, 0.1);

  for (i in 1:N)
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
}

generated quantities {
  real tau_out     = tau;
  real sigma_v_out = sigma_v;
  vector[N] p_bt_out   = p_bt;
  vector[N] p_R_out    = p_R;
  matrix[B, T] v_cmf_out = v;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
