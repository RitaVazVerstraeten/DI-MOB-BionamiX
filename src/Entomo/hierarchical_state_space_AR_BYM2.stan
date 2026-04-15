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

  // ICAR neighbourhood graph
  int<lower=0> N_edges;
  array[N_edges] int<lower=1,upper=B> node1;
  array[N_edges] int<lower=1,upper=B> node2;

  // BYM2 scaling factor: geometric mean of the diagonal of the ICAR
  // precision matrix pseudoinverse — scales u_icar_raw to unit marginal variance.
  // Computed in R from the graph structure before fitting.
  real<lower=0> scaling_factor;

  real<lower=0> phi;  // beta-binomial concentration (fixed)
}

transformed data {
  real kappa = 2.0;
}

parameters {
  real alpha;
  matrix[K, Lp1] w;
  vector<lower=0>[K] sigma_w;
  vector[Ku] w_unlagged;
  vector[T] v_global_raw;
  real<lower=0> sigma_v;
  real<lower=-1,upper=1> rho;
  real<lower=0> delta1;

  // BYM2 spatial random effects
  vector[B] u_icar_raw;            // ICAR (structured) component — improper prior
  vector[B] u_het_raw;             // unstructured (iid) component
  real<lower=0> sigma_spatial;     // total spatial SD (log-odds scale)
  real<lower=0,upper=1> phi_mix;   // mixing: 1 = pure ICAR, 0 = pure iid
}

transformed parameters {
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  vector[T] v_global;
  vector[N] x_effect;
  vector[B] u_spatial;  // combined BYM2 spatial RE (one value per block)

  // 1. BYM2 spatial random effect
  //    u_icar_raw is scaled to unit marginal variance via scaling_factor.
  //    phi_mix blends structured and unstructured components.
  //    sigma_spatial sets the total magnitude on the log-odds scale.
  u_spatial = sigma_spatial * (
    sqrt(phi_mix)     * u_icar_raw / sqrt(scaling_factor) +
    sqrt(1 - phi_mix) * u_het_raw
  );

  // 2. Global AR(1) temporal trend (non-centred), then centred exactly
  v_global[1] = sigma_v * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T)
    v_global[t] = rho * v_global[t-1] + sigma_v * v_global_raw[t];
  v_global = v_global - mean(v_global);

  // 3. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 4. Linear predictor → latent ecological probability
  vector[N] eta;
  for (i in 1:N)
    eta[i] = alpha + x_effect[i] + u_spatial[block[i]] + v_global[time[i]];
  p_bt = inv_logit(eta);

  // 5. Reactive surveillance probability (linear case effect on log-odds)
  for (i in 1:N) {
    if (C_bt[i] > 0)
      p_R[i] = inv_logit(eta[i] + delta1 * C_bt[i]);
    else
      p_R[i] = p_bt[i];
  }

  // 6. Effective observation probability (mixture of systematic and reactive)
  for (i in 1:N) {
    if (n_bt[i] == 0) {
      omega[i] = 0;
      pi[i]    = 0;
    } else if (C_bt[i] > 0) {
      omega[i] = fmin(1.0, (kappa * C_bt[i]) / n_bt[i]);
      pi[i]    = (1 - omega[i]) * p_bt[i] + omega[i] * p_R[i];
    } else {
      omega[i] = 0;
      pi[i]    = p_bt[i];
    }
  }
}

model {
  // Fixed-effect priors
  alpha      ~ normal(-7.0, 1.5);
  w_unlagged ~ normal(0, 0.5);
  delta1     ~ normal(0, 0.1);  // half-normal (lower=0)

  // Distributed lag weights: random walk smoothness prior
  for (k in 1:K) {
    w[k, 1] ~ normal(0, 0.5);
    for (l in 2:Lp1)
      w[k, l] ~ normal(w[k, l-1], sigma_w[k]);
  }
  sigma_w ~ exponential(2);

  // Temporal AR(1)
  v_global_raw ~ normal(0, 1);
  sigma_v      ~ exponential(1);
  rho          ~ normal(0.4, 0.2);
  // v_global is centred exactly in transformed parameters (mean subtracted).

  // BYM2 spatial priors
  // ICAR smoothness prior (pairwise differences; improper)
  target += -0.5 * dot_self(u_icar_raw[node1] - u_icar_raw[node2]);
  // Soft sum-to-zero: spatial level stays in alpha
  sum(u_icar_raw) ~ normal(0, 0.001 * B);
  // Unstructured component
  u_het_raw ~ normal(0, 1);
  // Interpretable hyperpriors
  sigma_spatial ~ normal(0, 1);       // total spatial SD; half-normal
  phi_mix       ~ beta(0.5, 0.5);     // weakly informative; allows any mix

  // Likelihood
  for (i in 1:N)
    y[i] ~ beta_binomial(n_bt[i],
                         fmax(pi[i] * phi, 1e-6),
                         fmax((1 - pi[i]) * phi, 1e-6));
}

generated quantities {
  vector[N] p_bt_out      = p_bt;
  vector[N] p_R_out       = p_R;
  vector[B] u_spatial_out = u_spatial;   // combined BYM2 spatial RE per block
  vector[T] v_global_out  = v_global;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i],
                                   fmax(pi[i] * phi, 1e-6),
                                   fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i],
                                    fmax(pi[i] * phi, 1e-6),
                                    fmax((1 - pi[i]) * phi, 1e-6));
  }
}
