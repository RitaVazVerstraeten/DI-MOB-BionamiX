data {
  int<lower=1> N;          // number of observations (block-time combinations)
  int<lower=0> y[N];       // observed mosquito-positive inspection events (y_bt)
  int<lower=1> N_HH[N];    // baseline household inspections (systematic surveillance)
  int<lower=1> K;          // number of lagged environmental covariates
  int<lower=1> Lp1;        // number of lags (L + 1, including lag 0)
  array[N, K, Lp1] real X_lag;  // lagged covariates X_{k,b,t-l}
  int<lower=1> Ku;         // number of unlagged block-level covariates
  matrix[N, Ku] X_unlagged;  // unlagged covariates (is_urban, has_aljibes, etc.)
  int<lower=1> B;          // number of blocks
  int<lower=1> T;          // number of time periods
  int<lower=1,upper=B> block[N];  // block index for each observation
  int<lower=1,upper=T> time[N];   // time index for each observation
  int<lower=0> C_bt[N];    // number of dengue cases per block-time
}

transformed data {
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights for environmental covariates
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
  
  // 1. Center spatial random effects
  u_block = sigma_u * u_block_raw;
  
  // 2. AR(1) temporal structure: v_t = rho * v_{t-1} + epsilon_t
  v_time[1] = sigma_v * v_time_raw[1] / sqrt(1 - rho^2);  // stationary initialization
  for (t in 2:T) {
    v_time[t] = rho * v_time[t-1] + sigma_v * v_time_raw[t];
  }
  
  for (i in 1:N) {
    // 3. Latent ecological process: logit(p_bt) = alpha + X*beta + u_b + v_t
    real x_effect = 0.0;
    // Lagged environmental covariates
    for (k in 1:K) {
      for (l in 1:Lp1) {
        x_effect += w[k, l] * X_lag[i, k, l];
      }
    }
    // Unlagged block-level covariates
    for (j in 1:Ku) {
      x_effect += w_unlagged[j] * X_unlagged[i, j];
    }
    p_bt[i] = inv_logit(alpha + x_effect + u_block[block[i]] + v_time[time[i]]);
    
    // 4. Reactive surveillance targeting bias (only when C_bt > 0)
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(logit(p_bt[i]) + delta0 + delta1 * log(C_bt[i]));
    } else {
      p_R[i] = p_bt[i];  // no reactive bias when no cases
    }
    
    // 5. Poisson rate: baseline inspections with ecological probability + reactive inspections with reactive probability
    lambda[i] = N_HH[i] * p_bt[i] + kappa * C_bt[i] * p_R[i];
  }
}

model {
  // Priors
  alpha ~ normal(-2, 1.5);        // baseline mosquito presence (logit scale)
  to_vector(w) ~ normal(0, 0.7);  // lag weights
  w_unlagged ~ normal(0, 0.5);    // unlagged covariate weights
  u_block_raw ~ normal(0, 1);     // non-centered parameterization
  v_time_raw ~ normal(0, 1);      // non-centered parameterization
  sigma_u ~ exponential(1);        // spatial scale: prevents extreme variance
  sigma_v ~ exponential(2);        // temporal scale: stricter (AR(1) structure)
  rho ~ normal(0.3, 0.3);         // truncated to [-1, 1]: AR(1) parameter
  kappa ~ lognormal(log(2), 0.4); // scaling factor for reactive inspections (centered at 2)
  delta0 ~ normal(0.5, 0.5);      // baseline targeting bias
  delta1 ~ normal(0, 0.3);        // log-linear increase with outbreak intensity

  // Observation model: y_bt ~ Poisson(lambda)
  // Expected value: lambda = N_HH * p_bt + kappa * C_bt * p_R
  // This combines baseline inspections with ecological probability and reactive inspections with reactive probability
  y ~ poisson(lambda);
}

generated quantities {
  // Posterior predictive checks
  int<lower=0> y_pred[N];
  vector[N] log_lik;
  
  for (i in 1:N) {
    y_pred[i] = poisson_rng(lambda[i]);
    log_lik[i] = poisson_lpmf(y[i] | lambda[i]);
  }
}
