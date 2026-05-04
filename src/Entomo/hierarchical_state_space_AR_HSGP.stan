data {
  int<lower=1> N;                 // number of observations (block-time combinations)
  array[N] int<lower=0> y;        // observed mosquito-positive inspection events (y_bt)
  array[N] int<lower=0> n_bt;     // total number of inspection events (= N_HH + kappa*C_bt)
  int<lower=1> K;               // number of lagged environmental covariates
  int<lower=1> Lp1;             // number of lags (L + 1, including lag 0)
  matrix[N, K*Lp1] X_lag_flat;  // flattened lagged covariates [N, K*Lp1]
  int<lower=1> Ku;              // number of unlagged block-level covariates
  matrix[N, Ku] X_unlagged;     // unlagged covariates (is_urban, has_aljibes, etc.)
  int<lower=1> B;               // number of blocks
  int<lower=1> T;               // number of time periods
  array[N] int<lower=1,upper=B> block;  // block index for each observation
  array[N] int<lower=1,upper=T> time;   // time index for each observation
  array[N] int<lower=0> C_bt;    // number of dengue cases per block-time
  matrix[B, 2] coords_block;     // block centroid coordinates in metres (projected CRS)
  int<lower=1> M;                // HSGP basis functions per dimension (e.g. 20)
  real<lower=1> c_boundary;      // HSGP boundary factor (e.g. 1.5)
  int<lower=0,upper=1> fix_phi;    // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;          // value used when fix_phi = 1 (ignored otherwise)
}

transformed data {
  real kappa = 2.0;

  // --- HSGP precomputation (done once, not every HMC step) ---
  int M_total = M * M;

  real x1_mean = mean(coords_block[, 1]);
  real x2_mean = mean(coords_block[, 2]);
  vector[B] x1 = to_vector(coords_block[, 1]) - x1_mean;
  vector[B] x2 = to_vector(coords_block[, 2]) - x2_mean;
  real L1 = c_boundary * (max(x1) - min(x1)) / 2.0;
  real L2 = c_boundary * (max(x2) - min(x2)) / 2.0;

  vector[M_total] lambda;
  matrix[B, M_total] PHI;
  {
    int idx = 1;
    vector[B] phi1;
    vector[B] phi2;
    for (m1 in 1:M) {
      phi1 = sin(m1 * pi() * (x1 + L1) / (2.0 * L1)) / sqrt(L1);
      for (m2 in 1:M) {
        phi2 = sin(m2 * pi() * (x2 + L2) / (2.0 * L2)) / sqrt(L2);
        lambda[idx] = square(m1 * pi() / (2.0 * L1)) + square(m2 * pi() / (2.0 * L2));
        PHI[, idx] = phi1 .* phi2;
        idx = idx + 1;
      }
    }
  }
}

parameters {
  real alpha;
  matrix[K, Lp1] w;
  vector[Ku] w_unlagged;
  vector[T] v_global_raw;
  real<lower=0> sigma_v;
  real<lower=-1,upper=1> rho;
  real<lower=0> delta1;
  real<lower=0> sigma_gp;
  real<lower=0> rho_gp;
  vector[M_total] beta_gp;
  real<lower=0> phi_raw;     // beta-binomial concentration; used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] omega;
  vector[N] pi;
  vector[T] v_global;
  vector[N] x_effect;
  vector[B] u_gp;

  // 1. HSGP spatial random effects
  {
    vector[M_total] diag_SPD;
    real inv_rho2 = 1.0 / rho_gp^2;
    for (m in 1:M_total) {
      diag_SPD[m] = sigma_gp * sqrt(2.0 * pi() / rho_gp) * (inv_rho2 + lambda[m])^(-0.75);
    }
    u_gp = PHI * (diag_SPD .* beta_gp);
  }

  // 2. Global AR(1) trend (no per-block deviation)
  v_global[1] = sigma_v * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T) {
    v_global[t] = rho * v_global[t-1] + sigma_v * v_global_raw[t];
  }

  // 3. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 4. Linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N) {
    eta[i] = alpha + x_effect[i] + u_gp[block[i]] + v_global[time[i]];
  }
  p_bt = inv_logit(eta);

  // 5. Reactive surveillance probability
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * C_bt[i]);
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

  to_vector(w)    ~ normal(0, 1.0);
  w_unlagged      ~ normal(0, 0.5);
  v_global_raw    ~ normal(0, 1);
  sigma_v         ~ exponential(1);
  rho             ~ normal(0.4, 0.2);
  delta1          ~ normal(0, 0.5);
  beta_gp         ~ normal(0, 1);
  sigma_gp        ~ normal(0, 1);
  rho_gp          ~ inv_gamma(3, 150);
  if (fix_phi == 0) phi_raw ~ gamma(2, 0.1);

  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}

generated quantities {
  vector[N] p_bt_out      = p_bt;
  vector[N] p_R_out       = p_R;
  vector[B] u_gp_out      = u_gp;
  vector[T] v_global_out  = v_global;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
