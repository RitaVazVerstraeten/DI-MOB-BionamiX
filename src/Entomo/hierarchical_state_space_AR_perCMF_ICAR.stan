// Per-CMF AR(1) + ICAR spatial random effects
// Each CMF follows its own AR(1) trajectory (shared rho, sigma_v).
// A static ICAR spatial offset captures persistent spatial structure across blocks.
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

  // ----- ICAR neighbourhood graph -----
  int<lower=0> N_edges;
  array[N_edges] int<lower=1,upper=B> node1;    // node1 < node2 for each undirected edge
  array[N_edges] int<lower=1,upper=B> node2;

  int<lower=0,upper=1> fix_phi;    // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;          // value used when fix_phi = 1 (ignored otherwise)
  real<lower=0> kappa;
}

parameters {
  real alpha;
  matrix[K, Lp1] w;
  vector[Ku] w_unlagged;
  matrix[B, T] v_raw;              // per-CMF AR(1) innovations (non-centered)
  real<lower=0> sigma_v;           // shared innovation SD (partial pooling across CMFs)
  real<lower=-1,upper=1> rho;      // shared AR(1) coefficient
  vector[B] u_icar_raw;            // raw ICAR spatial effects (unscaled)
  real<lower=0> sigma_icar;        // marginal SD of the spatial RE
  real<lower=0> delta1;
  real<lower=0> phi_raw;           // used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  matrix[B, T] v;                  // per-CMF AR(1) states
  vector[B] u_icar;                // scaled ICAR spatial effects
  vector[N] x_effect;

  // 1. ICAR spatial random effects (scaled)
  u_icar = sigma_icar * u_icar_raw;

  // 2. Per-CMF AR(1): each CMF follows its own trajectory, sharing rho and sigma_v
  for (b in 1:B) {
    v[b, 1] = sigma_v * v_raw[b, 1] / sqrt(fmax(1e-6, 1 - rho^2));
    for (t in 2:T)
      v[b, t] = rho * v[b, t-1] + sigma_v * v_raw[b, t];
  }

  // 3. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 4. Linear predictor: per-CMF AR(1) trajectory + static ICAR spatial offset
  vector[N] eta;
  for (i in 1:N)
    eta[i] = alpha + x_effect[i] + v[block[i], time[i]] + u_icar[block[i]];
  p_bt = inv_logit(eta);

  // 5. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * log(C_bt[i]));
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

  to_vector(w) ~ normal(0, 1.0);
  w_unlagged   ~ normal(0, 0.5);
  to_vector(v_raw) ~ normal(0, 1);
  sigma_v      ~ normal(0, 0.3);  // half-normal via constraint <lower=0>
  rho          ~ normal(0.35, 0.1);
  delta1       ~ normal(0, 0.5);

  // ICAR prior: pairwise differences penalise spatial discontinuity
  target += -0.5 * dot_self(u_icar_raw[node1] - u_icar_raw[node2]);
  // Soft sum-to-zero: pins the overall level
  sum(u_icar_raw) ~ normal(0, 0.001 * B);
  sigma_icar ~ normal(0, 0.3);

  if (fix_phi == 0) phi_raw ~ gamma(13, 0.1);

  for (i in 1:N)
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
}

generated quantities {
  vector[N] p_bt_out      = p_bt;
  vector[N] p_R_out       = p_R;
  matrix[B, T] v_cmf_out  = v;
  vector[B] u_icar_out    = u_icar;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
