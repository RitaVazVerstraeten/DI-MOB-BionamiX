data {
  int<lower=1> N;          // number of observations (block-time combinations)
  array[N] int<lower=0> y;       // observed mosquito-positive inspection events (y_bt)
  array[N] int<lower=1> N_HH;    // baseline household inspections (systematic surveillance)
  array[N] int<lower=0> n_bt;    // total number of inspection events (from R)
  int<lower=1> K;          // number of lagged environmental covariates
  int<lower=1> Lp1;        // number of lags (L + 1, including lag 0)
  matrix[N, K*Lp1] X_lag_flat;  // flattened lagged covariates [N, K*Lp1]
  int<lower=1> Ku;         // number of unlagged block-level covariates
  matrix[N, Ku] X_unlagged;  // unlagged covariates (is_urban, has_aljibes, etc.)
  int<lower=1> B;          // number of blocks
  int<lower=1> T;          // number of time periods (kept for data compatibility)
  array[N] int<lower=1,upper=B> block;  // block index for each observation
  array[N] int<lower=1,upper=T> time;   // time index for each observation (unused here)
  array[N] int<lower=0> C_bt;    // number of dengue cases per block-time
}

transformed data {
  real kappa = 2.0;  // fixed scaling factor for reactive inspections
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights for environmental covariates
  vector<lower=0>[K] sigma_w;  // random walk SD for each covariate's lag structure
  vector[Ku] w_unlagged;   // weights for unlagged block-level covariates
  vector[B] u_block_raw;   // spatial random effects (non-centered)
  real<lower=0> sigma_u;   // SD of spatial random effects
  real delta0;             // baseline targeting bias (reactive surveillance)
  real delta1;             // log-linear increase with outbreak intensity
}

transformed parameters {
  vector[N] p_bt;          // latent ecological probability (true mosquito presence)
  vector[N] p_R;           // reactive surveillance probability (biased upward)
  vector[N] omega;         // fraction of inspections that are reactive (omega_bt = kappa*C_bt / n_bt)
  vector[N] pi;            // effective observation probability (mixture of p_bt and p_R)
  vector[B] u_block;       // spatial random effects (centered)
  vector[N] x_effect;      // linear predictor for environmental effects

  // 1. Center spatial random effects
  u_block = sigma_u * u_block_raw;

  // 2. Calculate environmental effects (matrix multiplication)
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 3. Calculate linear predictor and latent ecological probability
  vector[N] eta = alpha + x_effect + u_block[block];
  p_bt = inv_logit(eta);

  // 4. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta0 + delta1 * log(C_bt[i]));
    } else {
      p_R[i] = p_bt[i];
    }
  }

  // 5. Fraction of reactive inspections and mixture probability
  for (i in 1:N) {
    if (n_bt[i] == 0) {
      omega[i] = 0;
      pi[i] = 0;
    } else if (C_bt[i] > 0) {
      omega[i] = (kappa * C_bt[i]) / n_bt[i];
      pi[i] = (1 - omega[i]) * p_bt[i] + omega[i] * p_R[i];
    } else {
      omega[i] = 0;
      pi[i] = p_bt[i];
    }
  }
}

model {
  // Priors
  alpha ~ normal(-4.5, 1.2);

  for (k in 1:K) {
    w[k, 1] ~ normal(0, 0.5);
    for (l in 2:Lp1) {
      w[k, l] ~ normal(w[k, l-1], sigma_w[k]);
    }
  }
  sigma_w ~ exponential(2);

  w_unlagged ~ normal(0, 0.5);
  u_block_raw ~ normal(0, 1);
  sigma_u ~ exponential(2);
  delta0 ~ normal(0.3, 0.4);
  delta1 ~ normal(0, 0.2);

  for (i in 1:N) {
    y[i] ~ binomial(n_bt[i], pi[i]);
  }
}

generated quantities {
  vector[N] p_bt_out = p_bt;
  vector[N] p_R_out = p_R;
  vector[B] u_block_out = u_block;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i] = binomial_rng(n_bt[i], pi[i]);
    log_lik[i] = binomial_lpmf(y[i] | n_bt[i], pi[i]);
  }
}
