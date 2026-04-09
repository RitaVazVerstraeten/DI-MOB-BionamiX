// Base model: covariates + reactive surveillance only
// No temporal AR1, no spatial GP, no per-block random effect
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
  real<lower=0> phi;
}

transformed data {
  real kappa = 2.0;
}

parameters {
  real alpha;
  matrix[K, Lp1] w;
  vector<lower=0>[K] sigma_w;
  vector[Ku] w_unlagged;
  real delta0;
  real delta1;
}

transformed parameters {
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  vector[N] x_effect;

  // 1. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 2. Linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N) {
    eta[i] = alpha + x_effect[i];
  }
  p_bt = inv_logit(eta);

  // 3. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta0 + delta1 * log(C_bt[i]));
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
  alpha ~ normal(-7.0, 1.5);

  for (k in 1:K) {
    w[k, 1] ~ normal(0, 0.5);
    for (l in 2:Lp1) {
      w[k, l] ~ normal(w[k, l-1], sigma_w[k]);
    }
  }
  sigma_w    ~ exponential(2);
  w_unlagged ~ normal(0, 0.5);
  delta0     ~ normal(0.3, 0.4);
  delta1     ~ normal(0, 0.2);

  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
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
