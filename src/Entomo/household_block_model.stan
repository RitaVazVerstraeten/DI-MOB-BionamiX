data {
  int<lower=1> B;                 // number of blocks
  int<lower=1> T;                 // number of time periods
  int<lower=1> K;                 // number of environmental covariates
  array[B] int<lower=1> N_b;      // households per block
  array[B, T] int<lower=0> Y;     // observed positives per block-time
  array[B, T] int<lower=0> C;     // dengue cases per block-time
  array[B, T, K] real X;          // environmental covariates per block-time
}

parameters {
  // Visit process
  real alpha0;                    // baseline visit log-odds
  real alpha1;                    // effect of dengue cases on visit probability
  real alpha2;                    // revisit effect

  // Ecological process
  real alpha_gamma;               // colonization intercept
  real alpha_phi;                 // persistence intercept
  vector[K] beta_env;             // shared environmental effects
  real<lower=0> theta;            // control effectiveness

  // Random effects
  vector[B] u_block_raw;          // spatial random effects (non-centered)
  vector[T] v_time_raw;           // temporal random effects (non-centered)
  real<lower=0> sigma_u;           // SD of spatial random effects
  real<lower=0> sigma_v;           // SD of temporal innovations
  real<lower=-1,upper=1> rho;      // AR(1) temporal persistence

  // Initial states
  vector[B] p0_raw;               // initial p_bt (logit scale)
  vector[B] q0_raw;               // initial q_bt (logit scale)
}

transformed parameters {
  vector[B] u_block;
  vector[T] v_time;
  matrix[B, T] p_bt;              // expected larvae prevalence
  matrix[B, T] q_bt;              // expected visit probability
  matrix[B, T] r_bt;              // expected observed positives (p*q)

  // Center spatial random effects
  u_block = sigma_u * u_block_raw;

  // AR(1) temporal random effects
  v_time[1] = sigma_v * v_time_raw[1] / sqrt(1 - rho^2);
  for (t in 2:T) {
    v_time[t] = rho * v_time[t - 1] + sigma_v * v_time_raw[t];
  }

  // Initial conditions
  for (b in 1:B) {
    p_bt[b, 1] = inv_logit(p0_raw[b]);
    q_bt[b, 1] = inv_logit(q0_raw[b]);
    r_bt[b, 1] = p_bt[b, 1] * q_bt[b, 1];
  }

  // Dynamics
  for (b in 1:B) {
    for (t in 2:T) {
      real gamma_bt = alpha_gamma + dot_product(to_vector(X[b, t]), beta_env) + u_block[b] + v_time[t];
      real phi_bt   = alpha_phi   + dot_product(to_vector(X[b, t]), beta_env) + u_block[b] + v_time[t];

      real visit_new = inv_logit(alpha0 + alpha1 * C[b, t]);
      real visit_re  = inv_logit(alpha0 + alpha1 * C[b, t] + alpha2);

      q_bt[b, t] = (1 - q_bt[b, t - 1]) * visit_new + q_bt[b, t - 1] * visit_re;

      real colonize     = inv_logit(gamma_bt);
      real persist_no   = inv_logit(phi_bt);
      real persist_ctrl = inv_logit(phi_bt - theta);

      p_bt[b, t] = colonize * (1 - p_bt[b, t - 1])
                + p_bt[b, t - 1] * ((1 - q_bt[b, t - 1]) * persist_no + q_bt[b, t - 1] * persist_ctrl);

      r_bt[b, t] = p_bt[b, t] * q_bt[b, t];
    }
  }
}

model {
  // Priors
  alpha0 ~ normal(-1, 1.5);
  alpha1 ~ normal(0, 0.5);
  alpha2 ~ normal(0, 0.5);

  alpha_gamma ~ normal(-2, 1.5);
  alpha_phi ~ normal(-1.5, 1.5);
  beta_env ~ normal(0, 0.7);
  theta ~ normal(0, 1);

  u_block_raw ~ normal(0, 1);
  v_time_raw ~ normal(0, 1);
  sigma_u ~ exponential(1);
  sigma_v ~ exponential(2);
  rho ~ normal(0.3, 0.3);

  p0_raw ~ normal(0, 1);
  q0_raw ~ normal(0, 1);

  // Observation model
  for (b in 1:B) {
    for (t in 1:T) {
      Y[b, t] ~ binomial(N_b[b], r_bt[b, t]);
    }
  }
}

generated quantities {
  array[B, T] int<lower=0> Y_pred;
  array[B, T] real log_lik;

  for (b in 1:B) {
    for (t in 1:T) {
      Y_pred[b, t] = binomial_rng(N_b[b], r_bt[b, t]);
      log_lik[b, t] = binomial_lpmf(Y[b, t] | N_b[b], r_bt[b, t]);
    }
  }
}
