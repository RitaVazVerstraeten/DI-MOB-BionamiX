// Base model: covariates + reactive surveillance only
// No temporal AR1, no spatial ICAR, no per-block random effect
data {
  int<lower=1> N;                  // number of observations (block-time combinations)
  array[N] int<lower=0> y;         // observed mosquito-positive inspection events (y_bt)
  array[N] int<lower=0> n_bt;      // total number of inspection events (= N_HH + kappa*C_bt)
  int<lower=1> K;                  // number of lagged environmental covariates
  int<lower=1> Lp1;                // number of lags (L + 1, including lag 0)
  matrix[N, K*Lp1] X_lag_flat;    // flattened lagged covariates [N, K*Lp1]
  int<lower=1> Ku;                 // number of unlagged block-level covariates
  matrix[N, Ku] X_unlagged;       // unlagged covariates (is_urban, has_aljibes, etc.)
  int<lower=1> B;                  // number of blocks
  int<lower=1> T;                  // number of time periods
  array[N] int<lower=1,upper=B> block;  // block index for each observation
  array[N] int<lower=1,upper=T> time;   // time index for each observation
  array[N] int<lower=0> C_bt;      // number of dengue cases per block-time

  int<lower=0,upper=1> fix_phi;    // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;          // value used when fix_phi = 1 (ignored otherwise)
}

transformed data {
  real kappa = 2.0;  // fixed scaling factor for reactive inspections
}

parameters {
  real alpha;                // baseline intercept
  matrix[K, Lp1] w;          // distributed lag weights for environmental covariates
  vector[Ku] w_unlagged;     // weights for unlagged block-level covariates
  real<lower=0> delta1;      // linear increase in detection probability with dengue cases
  real<lower=0> phi_raw;     // beta-binomial concentration; used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;  // active concentration
  vector[N] p_bt;    // latent ecological probability (true mosquito presence)
  vector[N] p_R;     // reactive surveillance probability (biased upward)
  vector[N] omega;   // fraction of inspections that are reactive
  vector[N] pi;      // effective observation probability
  vector[N] x_effect;

  // 1. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 2. Linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N)
    eta[i] = alpha + x_effect[i];
  p_bt = inv_logit(eta);

  // 3. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0)
      p_R[i] = inv_logit(eta[i] + delta1 * C_bt[i]);
    else
      p_R[i] = p_bt[i];
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
  alpha ~ normal(-7.0, 1.5);

  // Free lag weights: independent normal prior on each w[k,l]
  to_vector(w) ~ normal(0, 1.0);  // on log-odds scale -> max plausible odds-ratio becomes 2.7x

  w_unlagged ~ normal(0, 1.0);

  // dengue case correction
  delta1 ~ normal(0, 0.5);  // half-normal (lower=0): positive bias expected

  if (fix_phi == 0) phi_raw ~ gamma(2, 0.1);  // prior when estimating: mean=20, weakly regularising

  for (i in 1:N)
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
}

generated quantities {
  vector[N] p_bt_out = p_bt;
  vector[N] p_R_out  = p_R;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
