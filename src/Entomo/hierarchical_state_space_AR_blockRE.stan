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

  int<lower=0,upper=1> fix_phi;    // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;          // value used when fix_phi = 1 (ignored otherwise)
}

transformed data {
  real kappa = 2.0;
}

parameters {
  real alpha;
  matrix[K, Lp1] w;
  vector[Ku] w_unlagged;
  vector[T] v_global_raw;
  real<lower=0> sigma_v;
  real<lower=-1,upper=1> rho;
  vector[B] v_block_dev_raw;
  real<lower=0> sigma_block_dev;
  real<lower=0> delta1;
  real<lower=0> phi_raw;     // beta-binomial concentration; used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  vector[T] v_global;
  vector[B] v_block_dev;
  vector[N] x_effect;

  // 1. Global AR(1) trend + per-block deviations (no spatial GP)
  v_global[1] = sigma_v * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T) {
    v_global[t] = rho * v_global[t-1] + sigma_v * v_global_raw[t];
  }
  v_block_dev = sigma_block_dev * v_block_dev_raw;

  // 2. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 3. Linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N) {
    eta[i] = alpha + x_effect[i] + v_global[time[i]] + v_block_dev[block[i]];
  }
  p_bt = inv_logit(eta);

  // 4. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * C_bt[i]);
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

  to_vector(w)    ~ normal(0, 1.0);
  w_unlagged      ~ normal(0, 0.5);
  v_global_raw    ~ normal(0, 1);
  sigma_v         ~ exponential(1);
  rho             ~ normal(0.4, 0.2);
  v_block_dev_raw ~ normal(0, 1);
  sigma_block_dev ~ exponential(2);
  delta1          ~ normal(0, 0.5);
  if (fix_phi == 0) phi_raw ~ gamma(2, 0.1);

  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}

generated quantities {
  vector[N] p_bt_out       = p_bt;
  vector[N] p_R_out        = p_R;
  vector[T] v_global_out   = v_global;
  vector[B] v_block_dev_out = v_block_dev;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
