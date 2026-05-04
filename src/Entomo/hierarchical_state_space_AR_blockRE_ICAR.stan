data {
  int<lower=1> N;                 // number of observations (block-time combinations)
  array[N] int<lower=0> y;        // observed mosquito-positive inspection events (y_bt)
  array[N] int<lower=0> n_bt;     // total number of inspection events (= N_HH + kappa*C_bt)
  int<lower=1> K;               // number of lagged environmental covariates
  int<lower=1> Lp1;             // number of lags (L + 1, including lag 0)
  matrix[N, K*Lp1] X_lag_flat;  // flattened lagged covariates [N, K*Lp1]
  int<lower=1> Ku;              // number of unlagged block-level covariates
  matrix[N, Ku] X_unlagged;     // unlagged covariates (is_urban, has_aljibes, etc.)
  int<lower=1> B;               // number of blocks
  int<lower=1> T;               // number of time periods
  array[N] int<lower=1,upper=B> block;  // block index for each observation
  array[N] int<lower=1,upper=T> time;   // time index for each observation
  array[N] int<lower=0> C_bt;    // number of dengue cases per block-time

  // ----- ICAR neighbourhood graph -----
  int<lower=0> N_edges;                          // number of undirected edges
  array[N_edges] int<lower=1,upper=B> node1;     // first node of each edge (node1 < node2)
  array[N_edges] int<lower=1,upper=B> node2;     // second node of each edge

  int<lower=0,upper=1> fix_phi;    // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;          // value used when fix_phi = 1 (ignored otherwise)
}

transformed data {
  real kappa = 2.0;  // fixed scaling factor for reactive inspections
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights for environmental covariates (free estimation)
  vector[Ku] w_unlagged;   // weights for unlagged block-level covariates
  vector[T] v_global_raw;          // global AR(1) trend (non-centered)
  vector[B] v_block_dev_raw;       // per-block deviation from global trend (non-centered)
  real<lower=0> sigma_v;           // SD of global temporal trend
  real<lower=-1,upper=1> rho;      // AR(1) coefficient
  real<lower=0> sigma_block_dev;   // SD of per-block deviations
  real<lower=0> delta1;    // linear increase in detection probability with dengue cases

  // ----- ICAR spatial random effects -----
  // u_icar has an improper ICAR prior (pairwise differences);
  // soft sum-to-zero constraint pins the level.
  vector[B] u_icar_raw;            // raw ICAR spatial effects (unscaled) one per block
  real<lower=0> sigma_icar;        // marginal SD of the spatial RE - overall magnitude of spatial variation
  real<lower=0> phi_raw;     // beta-binomial concentration; used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  vector[N] p_bt;          // latent ecological probability (true mosquito presence)
  vector[N] p_R;           // reactive surveillance probability (biased upward)
  vector[N] omega;         // fraction of inspections that are reactive (omega_bt = kappa*C_bt / n_bt)
  vector[N] pi;            // effective observation probability (mixture of p_bt and p_R)
  vector[T] v_global;      // global AR(1) temporal trend
  vector[B] v_block_dev;   // per-block deviation from global trend
  vector[N] x_effect;      // linear predictor for environmental effects
  vector[B] u_icar;        // scaled ICAR spatial effects

  // 1. ICAR spatial random effects (scaled)
  u_icar = sigma_icar * u_icar_raw; // one spatial offset per block in log-odds units 
  // u_icar_raw is unit-less, sigma_icar then converts it to the log-odds units 

  // 2. Global AR(1) trend + per-block deviations
  v_global[1] = sigma_v * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T)
    v_global[t] = rho * v_global[t-1] + sigma_v * v_global_raw[t];
  // Centre v_global so its mean is exactly zero.
  // The sum-to-zero constraint belongs on v_global itself, but Stan only allows
  // priors in the model block. Explicit centering here achieves the same effect
  // and is exact rather than approximate.
  v_global = v_global - mean(v_global);

  v_block_dev = sigma_block_dev * v_block_dev_raw;

  // 3. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 4. Linear predictor and latent ecological probability
  vector[N] eta; // log-odds 
  for (i in 1:N) {
    eta[i] = alpha + x_effect[i] + u_icar[block[i]] + v_global[time[i]] + v_block_dev[block[i]]; 
    // baseline  + evironmental + spatial ICAR + global AR time + block-specific deviation 
  }
  p_bt = inv_logit(eta);

  // 5. Reactive surveillance probability
  // Linear effect of case count on the log-odds of detection during reactive visits.
  // When C_bt = 0, p_R = p_bt (no reactive bias).
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * C_bt[i]);
    } else {
      p_R[i] = p_bt[i];
    }
  }

  // 6. Effective observation probability
  for (i in 1:N) {
    if (n_bt[i] == 0) {
      omega[i] = 0;
      pi[i] = 0;
    } else if (C_bt[i] > 0) {
      omega[i] = fmin(1.0, (kappa * C_bt[i]) / n_bt[i]); // cap omega to max 1
      pi[i] = (1 - omega[i]) * p_bt[i] + omega[i] * p_R[i];
    } else {
      omega[i] = 0;
      pi[i] = p_bt[i];
    }
  }
}

model {
  // Priors
  alpha ~ normal(-7.0, 1.5);

  // Free lag weights: independent normal prior on each w[k,l]
  to_vector(w) ~ normal(0, 0.5);
  w_unlagged   ~ normal(0, 0.5);
  v_global_raw    ~ normal(0, 1);
  v_block_dev_raw ~ normal(0, 1);
  sigma_v         ~ exponential(1);
  sigma_block_dev ~ exponential(2);
  rho             ~ normal(0.4, 0.2);
  // v_global is centred exactly in transformed parameters (mean subtracted).
  // v_block_dev is a direct scaling of v_block_dev_raw, so constraining the
  // sum of v_block_dev_raw is equivalent to constraining sum(v_block_dev).
  sum(v_block_dev_raw) ~ normal(0, 0.001 * B);
  delta1      ~ normal(0, 0.1);  // half-normal (lower=0): positive bias expected

  // ICAR prior: pairwise differences penalise spatial discontinuity
  // it is equivalent to a proper intrinsic CAR with all weights = 1.
  
  target += -0.5 * dot_self(u_icar_raw[node1] - u_icar_raw[node2]); // dot_self is sum of squared elements 
  
  //Nearby blocks are pulled towards each other. Blocks with many neighbours are more strongly pulled. 
  // This is the ICAR pairwise-difference likelihood — it's an improper prior (doesn't integrate to a finite number) because it only constrains differences, not the absolute level.

  // Soft sum-to-zero: pins the overall level (mean ≈ 0)
  sum(u_icar_raw) ~ normal(0, 0.001 * B);

  // Marginal SD: half-normal weakly regularising prior
  sigma_icar ~ normal(0, 1); // controlls magnitude of spatial variation sigma_icar = 0.3 means
  // neighbouring blocks differ by 0.3 log-odds on average
  if (fix_phi == 0) phi_raw ~ gamma(2, 0.1);

  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}

generated quantities {
  vector[N] p_bt_out       = p_bt;
  vector[N] p_R_out        = p_R;
  vector[B] u_icar_out     = u_icar;
  vector[T] v_global_out   = v_global;
  vector[B] v_block_dev_out = v_block_dev;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
