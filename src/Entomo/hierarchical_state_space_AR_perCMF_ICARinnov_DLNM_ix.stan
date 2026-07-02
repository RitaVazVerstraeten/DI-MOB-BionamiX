// Per-CMF AR(1) with ICAR-structured innovations + DLNM cross-basis + interaction
// cross-bases. Separable space-time structure (Knorr-Held "type IV" interaction):
// at every time step t, the AR(1) innovation vector across all B blocks is spatially
// smoothed via an ICAR prior, instead of being i.i.d. across blocks. The spatially
// correlated shock then propagates forward through the AR(1) recursion exactly as
// in hierarchical_state_space_AR_perCMF_blockRE_DLNM_ix.stan.
//
// This REPLACES the separate additive block-level term (u_block or a static u_icar)
// used in the blockRE/ICAR variants: spatial structure and temporal persistence are
// now the same latent process rather than two additive terms competing to explain
// "why is this block chronically different" — the source of the AR1-vs-block-level
// non-identifiability seen when combining a per-CMF AR(1) with a time-constant
// spatial/block random effect.
//
// Consequence: there is no longer a literal "permanent, unchanging neighbourhood
// risk" term. Spatial clustering now has a memory governed by rho — a shock that
// hits a cluster of CMFs together fades at the same rate as everything else in the
// AR(1), rather than persisting forever. If a genuinely static spatial trend needs
// to be captured too, extend this file with an additional static ICAR mean-reversion
// field (see hierarchical_state_space_AR_perCMF_ICAR_DLNM.stan for that pattern).
//
// AR(1) re-parameterised as (tau, rho): tau = marginal stationary SD of the AR(1)
// process; sigma_v = tau * sqrt(1 - rho^2) is derived (avoids the 1/sqrt(1-rho^2)
// blow-up near rho=1 seen in the (sigma_v, rho) parameterisation).
//
// Build X_ix in R via build_dlnm_stan_data() with cfg$dlnm_ix_vars set. Build the
// ICAR edges with build_icar_edges() and the scaling factor with
// compute_bym2_scaling() (same graph and scaling used for every time slice, since
// the CMF adjacency structure doesn't change over time).

data {
  int<lower=1> N;
  array[N] int<lower=0> y;
  array[N] int<lower=0> n_bt;
  int<lower=1> P_cb;             // total DLNM cross-basis columns
  matrix[N, P_cb] X_cb;          // pre-built cross-basis matrix
  int<lower=0> P_ix;             // interaction cross-basis columns; 0 = no interactions
  matrix[N, P_ix] X_ix;          // binary modifier * DLNM sub-basis; empty when P_ix = 0
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
  // Rescales each time slice's raw ICAR field to unit marginal variance so that tau/
  // sigma_v keep the same interpretation as in the i.i.d.-innovation AR(1) model.
  real<lower=0> scaling_factor;

  int<lower=0,upper=1> fix_phi;
  real<lower=0> phi_data;
  real<lower=0> kappa;
}

parameters {
  real alpha;
  vector[P_cb] w_cb;
  vector[P_ix] w_ix;             // interaction modifiers; empty vector when P_ix = 0
  vector[Ku] w_unlagged;
  matrix[B, T] v_raw;            // per-time-slice ICAR-structured innovations (non-centred)
  real<lower=0> tau;             // marginal stationary SD of the AR(1) process
  real<lower=-1,upper=1> rho;    // shared AR(1) coefficient
  real<lower=0> delta1;
  real<lower=0> phi_raw;
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  // Derived innovation SD: near rho=1 sigma_v->0 (smooth process), no blow-up.
  real<lower=0> sigma_v = tau * sqrt(fmax(1e-8, 1.0 - rho * rho));

  // Rescale each time slice's raw ICAR field to unit marginal variance before it
  // enters the AR(1) recursion below (same convention as BYM2's u_icar_raw scaling).
  matrix[B, T] v_raw_scaled = v_raw / sqrt(scaling_factor);

  matrix[B, T] v;                // per-CMF AR(1) states, spatially-correlated shocks
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  // Baseline DLNM + unlagged effects + interaction modifier (X_ix * w_ix = 0 when P_ix = 0)
  vector[N] x_effect = X_cb * w_cb + X_unlagged * w_unlagged + X_ix * w_ix;

  // 1. Per-CMF AR(1): stationary initialisation using tau (marginal SD).
  for (b in 1:B) {
    v[b, 1] = tau * v_raw_scaled[b, 1];
    for (t in 2:T)
      v[b, t] = rho * v[b, t-1] + sigma_v * v_raw_scaled[b, t];
  }

  // 2. Linear predictor (no separate additive block/spatial term — it's in v)
  vector[N] eta;
  for (i in 1:N)
    eta[i] = alpha + x_effect[i] + v[block[i], time[i]];
  p_bt = inv_logit(eta);

  // 3. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * log(C_bt[i]));
    } else {
      p_R[i] = p_bt[i];
    }
  }

  // 4. Effective observation probability
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
  alpha        ~ normal(-7.0, 1.5);
  w_cb         ~ normal(0, 1.0);
  w_ix         ~ normal(0, 0.5);  // tighter than w_cb: interactions expected smaller
  w_unlagged   ~ normal(0, 0.5);

  // ICAR smoothness prior applied independently at each time slice: shocks that hit
  // at month t are spatially smoothed across neighbouring CMFs (pairwise-difference
  // penalty; improper on its own), then propagate forward through the AR(1)
  // recursion above. A soft sum-to-zero per slice keeps each month's spatial field
  // from drifting the intercept (mirrors the static-ICAR convention, applied here
  // once per time slice instead of once globally).
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
  vector[N] p_bt_out     = p_bt;
  vector[N] p_R_out      = p_R;
  matrix[B, T] v_cmf_out = v;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
