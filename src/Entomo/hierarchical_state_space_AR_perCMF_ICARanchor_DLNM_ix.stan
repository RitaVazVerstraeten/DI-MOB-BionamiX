// Per-CMF AR(1) that reverts to an ICAR-structured spatial *anchor* + DLNM
// cross-basis environmental predictors + interaction cross-bases.
//
// This replaces two earlier, less identifiable constructions:
//   1. hierarchical_state_space_AR_perCMF_ICAR_DLNM.stan: a zero-mean AR(1)
//      *plus* a separately-additive static ICAR term u_icar[b]. Two free
//      per-block "levels" (u_icar[b], and the AR(1)'s own persistent
//      component under high rho) compete to explain "why is this block
//      chronically different" -> non-identifiability. Confirmed by the ICAR
//      run: it swapped the iid block RE for a spatially-smoothed one but kept
//      the same two-competing-levels structure, so it inherited the same
//      ridge in the posterior geometry -- the problem is having two additive
//      per-block terms at all, not whether the second one is iid or ICAR.
//   2. hierarchical_state_space_AR_perCMF_ICARinnov_DLNM_ix.stan: ICAR
//      smoothing applied to the AR(1) *innovations* at every time slice
//      instead. This avoids the two-levels problem (no second additive term)
//      but also removes any literal "permanent, unchanging neighbourhood
//      risk" -- spatial clustering only persists as long as rho lets it.
//
// Here there is exactly one per-block level object, u_block[b], and it is
// spatially structured (ICAR). The AR(1) is nested as fluctuation *around*
// it, not alongside it:
//
//   v[b,t] = u_block[b] + rho * (v[b,t-1] - u_block[b]) + sigma_v * innovation
//
// u_block is the equilibrium each block's trajectory decays toward; the
// AR(1) only has to explain deviations from it. Because it's one object
// instead of two, there's good reason to expect this avoids the
// non-identifiability ridge seen in (1) rather than just describing it after
// the fact, the way (2) does.
//
// AR(1) re-parameterised as (tau, rho): tau = marginal stationary SD of the
// deviation from the anchor; sigma_v = tau * sqrt(1 - rho^2) is derived
// (avoids the 1/sqrt(1-rho^2) blow-up near rho=1). Both the anchor
// (u_block_raw, sigma_icar) and the AR(1) deviation (v_raw, non-centred) are
// parameterised non-centred -- this "AR around a hierarchical/spatial mean"
// structure is prone to funnels otherwise.
//
// Expect some soft tension between rho, sigma_icar and sigma_v to remain in
// the posterior (correlated, not necessarily bad Rhat/ESS) -- normal for this
// model class. Weakly informative priors on the variance components help.
//
// Build X_ix in R via build_dlnm_stan_data() with cfg$dlnm_ix_vars set. Build
// the ICAR edges with build_icar_edges() and the scaling factor with
// compute_bym2_scaling() (same graph and scaling used for the static anchor).

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

  // ICAR neighbourhood graph (static -- the anchor doesn't vary over time)
  int<lower=0> N_edges;
  array[N_edges] int<lower=1,upper=B> node1;
  array[N_edges] int<lower=1,upper=B> node2;
  // Geometric mean of the ICAR precision pseudoinverse diagonal (Riebler et al. 2016).
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
  matrix[B, T] v_raw;            // AR(1) innovations around the anchor (non-centred, plain iid)
  real<lower=0> tau;              // marginal stationary SD of the AR(1) deviation from u_block
  real<lower=-1,upper=1> rho;     // shared AR(1) coefficient
  vector[B] u_block_raw;          // ICAR-structured anchor (raw, unscaled)
  real<lower=0> sigma_icar;       // marginal SD of the spatial anchor
  real<lower=0> delta1;
  real<lower=0> phi_raw;
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  // Derived innovation SD: near rho=1 sigma_v->0 (smooth process), no blow-up.
  real<lower=0> sigma_v = tau * sqrt(fmax(1e-8, 1.0 - rho * rho));

  // ICAR-structured anchor, rescaled to unit marginal variance before sigma_icar
  // sets its magnitude (same convention as BYM2's u_icar_raw scaling).
  vector[B] u_block = sigma_icar * (u_block_raw / sqrt(scaling_factor));

  matrix[B, T] v;                // per-CMF AR(1) states, reverting to u_block
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  // Baseline DLNM + unlagged effects + interaction modifier (X_ix * w_ix = 0 when P_ix = 0)
  vector[N] x_effect = X_cb * w_cb + X_unlagged * w_unlagged + X_ix * w_ix;

  // 1. Per-CMF AR(1), reverting to the block-specific spatial anchor u_block[b]
  //    (non-centred: v is built directly from the anchor + raw innovations,
  //    rather than tracking a separate deviation vector).
  for (b in 1:B) {
    v[b, 1] = u_block[b] + tau * v_raw[b, 1];
    for (t in 2:T)
      v[b, t] = u_block[b] + rho * (v[b, t-1] - u_block[b]) + sigma_v * v_raw[b, t];
  }

  // 2. Linear predictor (u_block enters only through v -- no separate additive term)
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

  // Plain iid AR(1) innovations -- spatial structure lives in u_block, not here.
  to_vector(v_raw) ~ normal(0, 1);

  // ICAR smoothness prior on the (single, static) spatial anchor: pairwise-
  // difference penalty (improper on its own) plus a soft sum-to-zero so the
  // spatial field doesn't drift the intercept alpha.
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
  vector[B] u_block_out  = u_block;
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
