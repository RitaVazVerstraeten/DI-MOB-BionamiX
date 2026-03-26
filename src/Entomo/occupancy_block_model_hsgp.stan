// ============================================================
// Mosquito occupancy model with reactive surveillance — HSGP variant
// Combines:
//   - Explicit colonisation/persistence dynamics (HMM on p_bt)
//   - Separate environmental effects for gamma vs phi
//   - HSGP spatial random effects (Hilbert Space GP approximation)
//   - Global AR(1) + per-block deviations for temporal structure
//   - Reactive surveillance mixture (omega / delta0 / delta1)
//   - Beta-binomial observation model with fixed phi
//   - Distributed lags (random walk prior)
// HSGP replaces Cholesky GP: O(B*M^2) vs O(B^3) per HMC step
// ============================================================

data {
  int<lower=1> N;                   // total observations (block-time pairs with n_bt > 0)
  int<lower=1> B;                   // number of blocks
  int<lower=1> T;                   // number of time periods
  int<lower=1> K;                   // number of lagged environmental covariates
  int<lower=1> Lp1;                 // number of lags + 1 (lag 0 .. lag L)
  int<lower=1> Ku;                  // number of unlagged covariates

  array[N] int<lower=0> y;          // observed positives per block-time
  array[N] int<lower=0> n_bt;       // total inspections (N_HH + kappa * C_bt)
  array[N] int<lower=0> C_bt;       // dengue cases per block-time
  array[N] int<lower=1,upper=B> block;  // block index
  array[N] int<lower=1,upper=T> time;   // time index

  matrix[N, K*Lp1] X_lag_flat;     // flattened lagged covariates [N, K*(L+1)]
  matrix[N, Ku]    X_unlagged;      // unlagged covariates (is_urban, is_WUI, ...)

  matrix[B, 2] coords_block;        // block centroid coordinates in metres (projected CRS)
  int<lower=1> M;                   // HSGP basis functions per dimension (e.g. 20)
  real<lower=1> c_boundary;         // HSGP boundary factor (e.g. 1.5)

  real<lower=0> phi;                // beta-binomial concentration (fixed)
                                    // estimated from zero-case cells: phi ~ 23 for 100-block subset
}

transformed data {
  real kappa = 2.0;                 // reactive inspection multiplier (fixed)

  // --- HSGP precomputation (done once, not every HMC step) ---
  int M_total = M * M;

  // Centre coordinates and compute domain half-widths
  real x1_mean = mean(coords_block[, 1]);
  real x2_mean = mean(coords_block[, 2]);
  vector[B] x1 = to_vector(coords_block[, 1]) - x1_mean;
  vector[B] x2 = to_vector(coords_block[, 2]) - x2_mean;
  real L1 = c_boundary * (max(x1) - min(x1)) / 2.0;
  real L2 = c_boundary * (max(x2) - min(x2)) / 2.0;

  // Eigenvalues and eigenfunctions PHI evaluated at block locations
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
  // --- Intercepts ---
  real alpha_gamma;                 // colonisation baseline (log-odds)
  real alpha_phi;                   // persistence baseline (log-odds)

  // --- Distributed lag weights ---
  matrix[K, Lp1] w_gamma;          // lag weights for colonisation
  matrix[K, Lp1] w_phi;            // lag weights for persistence
  vector<lower=0>[K] sigma_w_gamma; // random walk SD per covariate (colonisation)
  vector<lower=0>[K] sigma_w_phi;   // random walk SD per covariate (persistence)

  // --- Unlagged covariate weights ---
  vector[Ku] w_unlagged_gamma;      // unlagged effects on colonisation
  vector[Ku] w_unlagged_phi;        // unlagged effects on persistence

  // --- Control effectiveness ---
  real<lower=0> theta;              // log-odds reduction in persistence after inspection

  // --- Spatial GP (HSGP) ---
  real<lower=0> sigma_gp;           // GP marginal SD
  real<lower=0> rho_gp;             // GP length-scale (metres)
  vector[M_total] beta_gp;          // HSGP basis coefficients ~ normal(0,1)

  // --- Global AR(1) temporal trend ---
  vector[T] v_global_raw;           // non-centred global temporal innovations
  real<lower=0> sigma_global;       // SD of global temporal innovations
  real<lower=-1,upper=1> rho;       // AR(1) coefficient (shared)

  // --- Per-block temporal deviations ---
  matrix[B, T] v_block_raw;         // non-centred per-block deviations
  real<lower=0> sigma_block;        // SD of per-block deviations

  // --- Reactive surveillance ---
  real delta0;                      // baseline log-odds shift for reactive inspections
  real delta1;                      // log-linear scaling with case count
}

transformed parameters {
  // --- Spatial GP (HSGP) ---
  vector[B] u_gp;

  {
    // Spectral density of Matern 1/2 (exponential) kernel in 2D:
    //   S(omega) = 2*pi * sigma^2 / rho * (1/rho^2 + omega^2)^(-3/2)
    //   diag_SPD[m] = sigma_gp * sqrt(2*pi/rho_gp) * (1/rho_gp^2 + lambda[m])^(-3/4)
    vector[M_total] diag_SPD;
    real inv_rho2 = 1.0 / rho_gp^2;
    for (m in 1:M_total) {
      diag_SPD[m] = sigma_gp * sqrt(2.0 * pi() / rho_gp) * (inv_rho2 + lambda[m])^(-0.75);
    }
    u_gp = PHI * (diag_SPD .* beta_gp);
  }

  // --- Global AR(1) ---
  vector[T] v_global;
  v_global[1] = sigma_global * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T) {
    v_global[t] = rho * v_global[t-1] + sigma_global * v_global_raw[t];
  }

  // --- Per-block AR(1) deviations ---
  matrix[B, T] v_block;
  for (b in 1:B) {
    v_block[b, 1] = sigma_block * v_block_raw[b, 1] / sqrt(fmax(1e-6, 1 - rho^2));
    for (t in 2:T) {
      v_block[b, t] = rho * v_block[b, t-1] + sigma_block * v_block_raw[b, t];
    }
  }

  // --- Environmental effects per observation ---
  vector[N] xeff_gamma = X_lag_flat * to_vector(w_gamma) + X_unlagged * w_unlagged_gamma;
  vector[N] xeff_phi   = X_lag_flat * to_vector(w_phi)   + X_unlagged * w_unlagged_phi;

  // --- Occupancy dynamics ---
  matrix[B, T] p_bt;
  matrix[B, T] eta_gamma_mat;
  matrix[B, T] eta_phi_mat;

  for (i in 1:N) {
    int b = block[i];
    int t = time[i];
    real shared_re = u_gp[b] + v_global[t] + v_block[b, t];
    eta_gamma_mat[b, t] = alpha_gamma + xeff_gamma[i] + shared_re;
    eta_phi_mat[b, t]   = alpha_phi   + xeff_phi[i]   + shared_re;
  }

  // Initial condition: stationary occupancy probability
  for (b in 1:B) {
    real g0 = inv_logit(eta_gamma_mat[b, 1]);
    real f0 = inv_logit(eta_phi_mat[b, 1]);
    real denom = fmax(1e-6, g0 + (1 - f0));
    p_bt[b, 1] = g0 / denom;
  }

  // Forward pass: occupancy dynamics t=2..T
  for (b in 1:B) {
    for (t in 2:T) {
      real colonize   = inv_logit(eta_gamma_mat[b, t]);
      real persist_no = inv_logit(eta_phi_mat[b, t]);
      p_bt[b, t] = colonize * (1 - p_bt[b, t-1])
                 + p_bt[b, t-1] * persist_no;
    }
  }

  // --- Observation model quantities ---
  vector[N] pi;
  vector[N] omega;
  vector[N] p_obs;
  vector[N] p_R;

  for (i in 1:N) {
    int b = block[i];
    int t = time[i];

    p_obs[i] = p_bt[b, t];

    if (C_bt[i] > 0) {
      p_R[i]   = inv_logit(eta_gamma_mat[b, t] + delta0 + delta1 * log(C_bt[i]));
      omega[i] = fmin(1.0, (kappa * C_bt[i]) / n_bt[i]);
      pi[i]    = (1 - omega[i]) * p_obs[i] + omega[i] * p_R[i];
    } else {
      p_R[i]   = p_obs[i];
      omega[i] = 0.0;
      pi[i]    = p_obs[i];
    }

    pi[i] = fmin(fmax(pi[i], 1e-6), 1 - 1e-6);
  }
}

model {
  // ---- Priors ----

  alpha_gamma ~ normal(-5.0, 1.5);
  alpha_phi   ~ normal(-1.5, 0.5);
  theta       ~ normal(1.5, 0.5);

  for (k in 1:K) {
    w_gamma[k, 1] ~ normal(0, 0.5);
    w_phi[k, 1]   ~ normal(0, 0.5);
    for (l in 2:Lp1) {
      w_gamma[k, l] ~ normal(w_gamma[k, l-1], sigma_w_gamma[k]);
      w_phi[k, l]   ~ normal(w_phi[k, l-1],   sigma_w_phi[k]);
    }
  }
  sigma_w_gamma ~ exponential(2);
  sigma_w_phi   ~ exponential(2);

  w_unlagged_gamma ~ normal(0, 0.5);
  w_unlagged_phi   ~ normal(0, 0.5);

  // Spatial GP (HSGP)
  beta_gp  ~ normal(0, 1);
  sigma_gp ~ normal(0, 0.5);                   // half-normal
  rho_gp   ~ inv_gamma(3, 150);              // mode ~75m

  // Global temporal AR(1)
  v_global_raw ~ normal(0, 1);
  sigma_global ~ exponential(3);
  rho          ~ normal(0.3, 0.2);

  // Per-block temporal deviations
  to_vector(v_block_raw) ~ normal(0, 1);
  sigma_block ~ exponential(3);

  // Reactive surveillance
  delta0 ~ normal(2.0, 0.5);
  delta1 ~ normal(0, 0.4);

  // ---- Likelihood ----
  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], pi[i] * phi, (1 - pi[i]) * phi);
  }
}

generated quantities {
  vector[N] p_bt_out;
  vector[N] p_R_out;
  vector[N] omega_out;
  vector[B] u_gp_out    = u_gp;
  vector[T] v_global_out = v_global;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    p_bt_out[i]  = p_bt[block[i], time[i]];
    p_R_out[i]   = p_R[i];
    omega_out[i] = omega[i];
    y_pred[i]    = beta_binomial_rng(n_bt[i], pi[i] * phi, (1 - pi[i]) * phi);
    log_lik[i]   = beta_binomial_lpmf(y[i] | n_bt[i], pi[i] * phi, (1 - pi[i]) * phi);
  }
}
