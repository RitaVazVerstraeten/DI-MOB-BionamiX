data {
  int<lower=1> N;          // number of observations (block-time combinations)
  int<lower=0> y[N];       // observed mosquito-positive inspection events (y_bt)
  int<lower=1> n_bt[N];    // total number of inspection events
  int<lower=1> K;          // number of environmental covariates
  int<lower=1> Lp1;        // number of lags (L + 1, including lag 0)
  array[N, K, Lp1] real X_lag;  // lagged covariates X_{k,b,t-l}
  int<lower=1> B;          // number of blocks
  int<lower=1> T;          // number of time periods
  int<lower=1,upper=B> block[N];  // block index for each observation
  int<lower=1,upper=T> time[N];   // time index for each observation
  int<lower=0> C_bt[N];    // number of dengue cases per block-time
  real<lower=0> kappa;     // scaling factor for reactive inspections (fixed at 1)
}

transformed data {
  vector<lower=0,upper=1>[N] omega;  // fraction of reactive inspections
  vector[N] N_HH;                    // number of households per block
  
  // Compute omega_bt and N_HH from n_bt and C_bt
  for (i in 1:N) {
    N_HH[i] = n_bt[i] - kappa * C_bt[i];
    if (C_bt[i] > 0) {
      omega[i] = (kappa * C_bt[i]) / n_bt[i];
    } else {
      omega[i] = 0.0;
    }
  }
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights
  vector[B] u_block_raw;   // spatial random effects (non-centered)
  vector[T] v_time_raw;    // temporal random effects (non-centered)
  real<lower=0> sigma_u;   // SD of spatial random effects
  real<lower=0> sigma_v;   // SD of temporal random effects
  real<lower=0> rho;       // temporal autoregression parameter
}

transformed parameters {
  vector[N] p_bt;          // latent ecological probability
  vector[N] p_R;           // reactive surveillance probability
  vector[N] pi_bt;         // effective observation probability
  vector[B] u_block;       // spatial random effects (centered)
  vector[T] v_time;        // temporal random effects with AR(1)
  
  // Center spatial random effects
  u_block = sigma_u * u_block_raw;
  
  // AR(1) temporal structure: v_t = rho * v_{t-1} + epsilon_t
  v_time[1] = sigma_v * v_time_raw[1] / sqrt(1 - rho^2);  // stationary initialization
  for (t in 2:T) {
    v_time[t] = rho * v_time[t-1] + sigma_v * v_time_raw[t];
  }

  // Fixed values for reactive surveillance bias (can be estimated if desired)
  real delta0 = 0.5;  // baseline targeting bias
  real delta1 = 0.2;  // log-linear increase with outbreak intensity
  
  for (i in 1:N) {
    // Latent ecological process: logit(p_bt) = alpha + X*beta + u_b + v_t
    real x_effect = 0.0;
    for (k in 1:K) {
      for (l in 1:Lp1) {
        x_effect += w[k, l] * X_lag[i, k, l];
      }
    }
    p_bt[i] = inv_logit(alpha + x_effect + u_block[block[i]] + v_time[time[i]]);
    
    // Reactive surveillance targeting bias (only when C_bt > 0)
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(logit(p_bt[i]) + delta0 + delta1 * log(C_bt[i]));
    } else {
      p_R[i] = p_bt[i];  // no reactive bias when no cases
    }
    
    // Effective observation probability: mixture of systematic and reactive
    pi_bt[i] = (1 - omega[i]) * p_bt[i] + omega[i] * p_R[i];
  }
}

model {
  // Priors
  alpha ~ normal(-2, 2);         // baseline mosquito presence (logit scale)
  to_vector(w) ~ normal(0, 1);   // lag weights
  u_block_raw ~ normal(0, 1);    // non-centered parameterization
  v_time_raw ~ normal(0, 1);
  sigma_u ~ normal(0, 0.5);      // spatial variation
  sigma_v ~ normal(0, 0.5);      // temporal variation
  rho ~ beta(8, 2);              // AR parameter (centered near 0.8)

  // Observation model: y_bt ~ Binomial(n_bt, pi_bt)
  y ~ binomial(n_bt, pi_bt);
}

generated quantities {
  // Posterior predictive checks
  int<lower=0> y_pred[N];
  vector[N] log_lik;
  
  for (i in 1:N) {
    y_pred[i] = binomial_rng(n_bt[i], pi_bt[i]);
    log_lik[i] = binomial_lpmf(y[i] | n_bt[i], pi_bt[i]);
  }
}
