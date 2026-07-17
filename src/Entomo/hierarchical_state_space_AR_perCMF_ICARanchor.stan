// Reduced (non-DLNM) prototype for an ICAR-structured AR(1) *anchor*. Same
// distributed-lag structure as hierarchical_state_space_AR_perCMF.stan, but
// instead of a zero-mean AR(1) plus a separately-additive block/ICAR term (two
// competing per-block levels -> non-identifiability), the AR(1) reverts to a
// single spatially-structured anchor u_block[b]:
//
//   v[b,t] = u_block[b] + rho * (v[b,t-1] - u_block[b]) + sigma_v * innovation
//
// u_block carries the ICAR smoothness prior directly (not the innovations, and
// not a second additive term). See hierarchical_state_space_AR_perCMF_ICARanchor_DLNM_ix.stan
// for the full write-up of the rationale and the DLNM+interaction version.
//
// Use this smaller model to sanity-check that the anchored-AR(1) construction
// actually samples well (rho, tau and sigma_icar separate cleanly, no
// divergences/funnels) before paying for a full DLNM+ix fit.

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

  // ICAR neighbourhood graph (static -- the anchor doesn't vary over time)
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
  matrix[B, T] v_raw;              // AR(1) innovations around the anchor (non-centred, plain iid)
  real<lower=0> tau;                // marginal stationary SD of the AR(1) deviation from u_block
  real<lower=-1,upper=1> rho;       // shared AR(1) coefficient
  vector[B] u_block_raw;            // ICAR-structured anchor (raw, unscaled)
  real<lower=0> sigma_icar;         // marginal SD of the spatial anchor
  real<lower=0> delta1;
  real<lower=0> phi_raw;            // used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  // Derived innovation SD: near rho=1 sigma_v->0 (smooth process), no blow-up.
  real<lower=0> sigma_v = tau * sqrt(fmax(1e-8, 1.0 - rho * rho));

  // ICAR-structured anchor, rescaled to unit marginal variance before sigma_icar
  // sets its magnitude (same Riebler et al. 2016 convention used for BYM2 and
  // for the ICAR-on-AR1-innovations variant elsewhere in this codebase).
  vector[B] u_block = sigma_icar * (u_block_raw / sqrt(scaling_factor));

  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  matrix[B, T] v;                  // per-CMF AR(1) states, reverting to u_block
  vector[N] x_effect;

  // 1. Per-CMF AR(1), reverting to the block-specific spatial anchor u_block[b]
  //    (non-centred: v is built directly from the anchor + raw innovations,
  //    rather than tracking a separate deviation vector).
  for (b in 1:B) {
    v[b, 1] = u_block[b] + tau * v_raw[b, 1];
    for (t in 2:T)
      v[b, t] = u_block[b] + rho * (v[b, t-1] - u_block[b]) + sigma_v * v_raw[b, t];
  }

  // 2. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 3. Linear predictor (u_block enters only through v -- no separate additive term)
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

  // Plain iid AR(1) innovations -- spatial structure lives in u_block, not here.
  to_vector(v_raw) ~ normal(0, 1);

  // ICAR smoothness prior on the (single, static) spatial anchor.
  target += -0.5 * dot_self(u_block_raw[node1] - u_block_raw[node2]);
  sum(u_block_raw) ~ normal(0, 0.001 * B);
  sigma_icar   ~ normal(0, 0.3);

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
  vector[B] u_block_out = u_block;
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
