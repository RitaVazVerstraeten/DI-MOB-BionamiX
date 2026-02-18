data {
  int<lower=1> N;          // number of observations (block-time combinations)
  array[N] int<lower=0> y;       // observed mosquito-positive inspection events (y_bt)
  array[N] int<lower=1> N_HH;    // baseline household inspections (systematic surveillance)
  int<lower=1> K;          // number of lagged environmental covariates
  int<lower=1> Lp1;        // number of lags (L + 1, including lag 0)
  matrix[N, K*Lp1] X_lag_flat;  // flattened lagged covariates [N, K*Lp1]
  int<lower=1> Ku;         // number of unlagged block-level covariates
  matrix[N, Ku] X_unlagged;  // unlagged covariates (is_urban, has_aljibes, etc.)
  int<lower=1> B;          // number of blocks
  int<lower=1> T;          // number of time periods
  array[N] int<lower=1,upper=B> block;  // block index for each observation
  array[N] int<lower=1,upper=T> time;   // time index for each observation
  array[N] int<lower=0> C_bt;    // number of dengue cases per block-time
}

transformed data {
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights for environmental covariates
  vector<lower=0>[K] sigma_w;  // random walk SD for each covariate's lag structure
  vector[Ku] w_unlagged;   // weights for unlagged block-level covariates
  vector[B] u_block_raw;   // spatial random effects (non-centered)
  vector[T] v_time_raw;    // temporal random effects (non-centered)
  real<lower=0> sigma_u;   // SD of spatial random effects
  real<lower=0> sigma_v;   // SD of temporal random effects
  real<lower=-1,upper=1> rho;  // temporal autoregression parameter
  real<lower=0> kappa;     // scaling factor for reactive inspections
  real delta0;             // baseline targeting bias (reactive surveillance)
  real delta1;             // log-linear increase with outbreak intensity
}

transformed parameters {
  vector[N] p_bt;          // latent ecological probability
  vector[N] p_R;           // reactive surveillance probability
  vector[N] lambda;        // Poisson rate: expected mosquito findings
  vector[B] u_block;       // spatial random effects (centered)
  vector[T] v_time;        // temporal random effects with AR(1)
  vector[N] x_effect;      // linear predictor for environmental effects
  
  // 1. Center spatial random effects
  u_block = sigma_u * u_block_raw;
  
  // 2. AR(1) temporal structure: v_t = rho * v_{t-1} + epsilon_t
  v_time[1] = sigma_v * v_time_raw[1] / sqrt(1 - rho^2);  // stationary initialization
  for (t in 2:T) {
    v_time[t] = rho * v_time[t-1] + sigma_v * v_time_raw[t];
  }
  
  // 3. Calculate environmental effects (matrix multiplication)
  // Flatten w to [K*Lp1] and multiply with X_lag_flat[N, K*Lp1]
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;
  
  // 4. Calculate linear predictor and latent ecological probability
  vector[N] eta = alpha + x_effect + u_block[block] + v_time[time];
  p_bt = inv_logit(eta);
  
  // 5. Reactive surveillance probability (loop-based conditional)
  // Work on linear predictor scale to avoid numerical issues
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta0 + delta1 * log(C_bt[i]));
    } else {
      p_R[i] = p_bt[i];  // no reactive bias when no cases
    }
  }
  
  // 6. Vectorized Poisson rate calculation
  lambda = to_vector(N_HH) .* p_bt + kappa * to_vector(C_bt) .* p_R;
}

model {
  // Priors
  alpha ~ normal(-4.5, 1.2);                    // Shifted way down for rare events
  
  // Random walk prior on lag weights: enforces smoothness across lags
  for (k in 1:K) {
    w[k, 1] ~ normal(0, 0.5);  // initial lag-0 weight
    for (l in 2:Lp1) {
      w[k, l] ~ normal(w[k, l-1], sigma_w[k]);  // random walk
    }
  }
  sigma_w ~ exponential(2);  // shrink toward smooth lag structure
  
  w_unlagged ~ normal(0, 0.5);    // unlagged covariate weights
  u_block_raw ~ normal(0, 1);     // non-centered parameterization
  v_time_raw ~ normal(0, 1);      // non-centered parameterization
  sigma_u ~ exponential(2);        // spatial scale: tighter (prevents extreme variance during init)
  sigma_v ~ exponential(3);        // temporal scale: stricter (AR(1) structure)
  rho ~ normal(0, 0.2);         // tighter prior on AR(1) parameter
  kappa ~ lognormal(log(2), 0.35); // scaling factor for reactive inspections (centered at 2)
  delta0 ~ normal(0.3, 0.4);      // slightly reduced baseline targeting bias
  delta1 ~ normal(0, 0.2);        // reduced log-linear increase to stabilize init

  // Observation model: y_bt ~ Poisson(lambda)
  // Expected value: lambda = N_HH * p_bt + kappa * C_bt * p_R
  // This combines baseline inspections with ecological probability and reactive inspections with reactive probability
  y ~ poisson(lambda);
}

generated quantities {
  // Save probabilities and random effects for posterior analysis
  vector[N] p_bt_out = p_bt;
  vector[N] p_R_out = p_R;
  vector[B] u_block_out = u_block;
  vector[T] v_time_out = v_time;
  
  // Posterior predictive checks
  array[N] int<lower=0> y_pred;
  vector[N] log_lik;
  
  for (i in 1:N) {
    y_pred[i] = poisson_rng(lambda[i]);
    log_lik[i] = poisson_lpmf(y[i] | lambda[i]);
  }
}
