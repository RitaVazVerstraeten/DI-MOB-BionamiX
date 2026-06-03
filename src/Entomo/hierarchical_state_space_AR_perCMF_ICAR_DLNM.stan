// Per-CMF AR(1) + ICAR spatial random effects + DLNM cross-basis environmental predictors.
//
// AR(1) uses the (tau, rho) parameterisation from hierarchical_state_space_AR_perCMF_blockRE_DLNM.stan:
//   tau  = marginal stationary SD of the AR(1) process (directly identified by trajectory spread)
//   rho  = AR(1) autocorrelation coefficient
//   sigma_v = tau * sqrt(1 - rho^2)  (derived; stable near rho=1, no 1/sqrt(1-rho^2) blow-up)
//
// Spatial structure: plain ICAR (no BYM2 mixing parameter) for faster convergence.
//   u_icar[b] ~ ICAR(sigma_icar)  -- static, time-constant offset per block
//
// Build X_cb in R with build_dlnm_stan_data(); pass P_cb = ncol(X_cb).
// Build ICAR edges with build_icar_edges(); pass N_edges, node1, node2.

data {
  int<lower=1> N;
  array[N] int<lower=0> y;
  array[N] int<lower=0> n_bt;

  // DLNM cross-basis
  int<lower=1> P_cb;
  matrix[N, P_cb] X_cb;

  int<lower=1> Ku;
  matrix[N, Ku] X_unlagged;

  int<lower=1> B;
  int<lower=1> T;
  array[N] int<lower=1,upper=B> block;
  array[N] int<lower=1,upper=T> time;
  array[N] int<lower=0> C_bt;

  // ICAR neighbourhood graph
  int<lower=0> N_edges;
  array[N_edges] int<lower=1,upper=B> node1;
  array[N_edges] int<lower=1,upper=B> node2;

  int<lower=0,upper=1> fix_phi;
  real<lower=0> phi_data;
  real<lower=0> kappa;
}

parameters {
  real alpha;
  vector[P_cb] w_cb;
  vector[Ku] w_unlagged;

  // Per-CMF AR(1) — (tau, rho) parameterisation
  matrix[B, T] v_raw;
  real<lower=0> tau;
  real<lower=-1,upper=1> rho;

  // ICAR spatial random effects (non-centred)
  vector[B] u_icar_raw;
  real<lower=0> sigma_icar;

  real<lower=0> delta1;
  real<lower=0> phi_raw;
}

transformed parameters {
  real<lower=0> phi    = fix_phi ? phi_data : phi_raw;
  real<lower=0> sigma_v = tau * sqrt(fmax(1e-8, 1.0 - rho * rho));

  vector[B] u_icar = sigma_icar * u_icar_raw;

  matrix[B, T] v;
  for (b in 1:B) {
    v[b, 1] = tau * v_raw[b, 1];           // stationary initialisation: v[b,1] ~ N(0, tau^2)
    for (t in 2:T)
      v[b, t] = rho * v[b, t-1] + sigma_v * v_raw[b, t];
  }

  vector[N] x_effect = X_cb * w_cb + X_unlagged * w_unlagged;

  vector[N] eta;
  for (i in 1:N)
    eta[i] = alpha + x_effect[i] + v[block[i], time[i]] + u_icar[block[i]];

  vector[N] p_bt = inv_logit(eta);

  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * log(C_bt[i]));
    } else {
      p_R[i] = p_bt[i];
    }

    if (n_bt[i] == 0) {
      omega[i] = 0;
      pi[i]    = 0;
    } else if (C_bt[i] > 0) {
      omega[i] = fmin(1.0, (kappa * C_bt[i]) / n_bt[i]);
      pi[i]    = (1 - omega[i]) * p_bt[i] + omega[i] * p_R[i];
    } else {
      omega[i] = 0;
      pi[i]    = p_bt[i];
    }
  }
}

model {
  alpha      ~ normal(-7.0, 1.5);
  w_cb       ~ normal(0, 1.0);
  w_unlagged ~ normal(0, 0.5);

  to_vector(v_raw) ~ normal(0, 1);
  tau  ~ normal(0, 1.0);
  rho  ~ normal(0.4, 0.1);

  // ICAR prior: penalises spatial discontinuity between neighbours
  target += -0.5 * dot_self(u_icar_raw[node1] - u_icar_raw[node2]);
  // Soft sum-to-zero: identifies intercept separately from spatial field
  sum(u_icar_raw) ~ normal(0, 0.001 * B);
  sigma_icar ~ normal(0, 0.3);

  delta1 ~ normal(0, 0.5);
  if (fix_phi == 0) phi_raw ~ gamma(13, 0.1);

  for (i in 1:N)
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
}

generated quantities {
  real tau_out     = tau;
  real sigma_v_out = sigma_v;
  vector[N] p_bt_out      = p_bt;
  vector[N] p_R_out       = p_R;
  matrix[B, T] v_cmf_out  = v;
  vector[B] u_icar_out    = u_icar;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;
  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
