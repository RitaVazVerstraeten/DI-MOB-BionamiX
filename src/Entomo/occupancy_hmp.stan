// ============================================================
// Mean-Field Occupancy Model with Visit Markov Chain
//
// Two latent block-level states, both evolved as mean-field
// approximations of household-level Markov chains:
//
//   p_bt = E[z_ibt] = P(larvae present in a random household)
//   q_bt = E[v_ibt] = P(a random household is visited)
//
// Ecological state z_ibt transitions:
//   P(z=1 | z_{t-1}=0) = gamma_bt          (colonisation)
//   P(z=1 | z_{t-1}=1) = phi_bt            (persistence, unvisited)
//                      = phi_eff_bt         (persistence, after visit)
//     phi_eff = inv_logit(logit(phi_bt) - theta)
//
// Mean-field closure for p_bt:
//   p_bt = (1 - p_{bt-1}) * gamma_bt
//          + p_{bt-1} * [(1 - q_{bt-1}) * phi_bt + q_{bt-1} * phi_eff_bt]
//
// Visit state v_ibt transitions:
//   P(v=1 | v_{t-1}=0) = inv_logit(alpha0 + alpha1 * C_bt)   (new visit)
//   P(v=1 | v_{t-1}=1) = inv_logit(alpha0 + alpha1 * C_bt + alpha2)  (revisit)
//   where C_bt = actual dengue cases in block b at time t.
//
// Mean-field closure for q_bt :
//   q_bt = inv_logit(alpha0 + alpha1*C_bt) * (1 - q_{bt-1})
//          + inv_logit(alpha0 + alpha1*C_bt + alpha2) * q_{bt-1}
//
// Observation :
//   r_bt = p_bt * q_bt       (joint: present AND visited)
//   Y_bt ~ Binomial(N_HH[b], r_bt)
//
// Assumptions :
//   - Households are interchangeable within a block
//   - Detection given visit is perfect (z and v independent within month)
//   - Each house visited at most once per month (Binomial thinning)
//
// Note on identifiability:
//   p_bt and q_bt enter only as a product r_bt = p_bt * q_bt.
//   External forcing via dengue time series (alpha1) and ecological
//   covariates helps separate the two processes, but weak data may
//   still yield correlated posteriors.
//
// Spatial RE  : HSGP (Hilbert-Space GP approximation)
// Temporal RE : Global AR(1) + per-block time-invariant offset
// ============================================================

data {
  // ----- Dimensions -----
  int<lower=1> N;             // observed (b,t) pairs with n_bt > 0
  int<lower=1> B;             // blocks
  int<lower=1> T;             // time periods
  int<lower=1> K;             // lagged environmental covariates
  int<lower=1> Lp1;           // lags + 1 (lag 0..L)
  int<lower=1> Ku;            // unlagged covariates

  // ----- Observation-indexed data (N rows, n_bt > 0 only) -----
  array[N] int<lower=0>         y;        // positives per block-time
  array[N] int<lower=0>         n_bt;     // total inspections (baseline + kappa*C)
  array[N] int<lower=0>         C_bt;     // dengue cases
  array[N] int<lower=1,upper=B> block;    // block index for row i
  array[N] int<lower=1,upper=T> time;     // time index for row i
  matrix[N, K*Lp1] X_lag_flat;            // flattened lagged covariates
  matrix[N, Ku]    X_unlagged;            // unlagged covariates

  // ----- Complete B×T grid -----
  // n_mat[b,t] = 0 for months with no inspections
  array[B, T] int<lower=0> y_mat;         // positives (0 if unobserved)
  array[B, T] int<lower=0> n_mat;         // inspections (0 if unobserved)
  array[B, T] int<lower=0> C_mat;         // dengue cases per block-time

  // ----- Spatial GP (HSGP) -----
  matrix[B, 2] coords_block;              // block centroids in metres (projected CRS)
  int<lower=1> M;                         // basis functions per spatial dimension
  real<lower=1> c_boundary;              // boundary expansion factor
}

transformed data {
  // --- HSGP precomputation (once, not every HMC step) ---
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
    for (m1 in 1:M) {
      vector[B] phi1 = sin(m1 * pi() * (x1 + L1) / (2.0 * L1)) / sqrt(L1);
      for (m2 in 1:M) {
        vector[B] phi2 = sin(m2 * pi() * (x2 + L2) / (2.0 * L2)) / sqrt(L2);
        lambda[idx] = square(m1 * pi() / (2.0 * L1)) + square(m2 * pi() / (2.0 * L2));
        PHI[, idx]  = phi1 .* phi2;
        idx = idx + 1;
      }
    }
  }
}

parameters {
  // --- Ecological process (z_bt: mosquito presence) ---
  real alpha_gamma;                        // colonisation baseline (logit scale)
  real alpha_phi;                          // persistence baseline (logit scale)
  matrix[K, Lp1] w_gamma;                 // distributed lag weights for colonisation
  matrix[K, Lp1] w_phi;                   // distributed lag weights for persistence
  vector<lower=0>[K] sigma_w_gamma;        // random walk SD per covariate (colonisation)
  vector<lower=0>[K] sigma_w_phi;          // random walk SD per covariate (persistence)
  vector[Ku] w_unlagged_gamma;             // unlagged covariate effects on colonisation
  vector[Ku] w_unlagged_phi;              // unlagged covariate effects on persistence

  // --- Intervention effect on persistence ---
  // phi_eff = inv_logit(logit(phi_bt) - theta) when a household was visited AND found
  // positive (systematic or reactive). Under perfect detection, visited + occupied = removed.
  // Applied in p_mat recursion as: persist = (1 - q_prev)*phi + q_prev*phi_eff
  real<lower=0> theta;

  // --- Visit Markov chain (v_bt: household visit probability) ---
  real alpha0;    // baseline visit log-odds (intercept)
  real alpha1;    // log-odds increase per dengue case C_bt
  real alpha2;    // additional log-odds for revisiting (v_{t-1}=1) vs new visit

  // --- Spatial GP (HSGP) ---
  real<lower=0> sigma_gp;                  // GP marginal SD
  real<lower=0> rho_gp;                    // GP length-scale (metres)
  vector[M_total] beta_gp;                 // HSGP basis coefficients ~ normal(0,1)

  // --- Global AR(1) temporal trend ---
  vector[T] v_global_raw;                  // non-centred innovations
  real<lower=0> sigma_global;              // innovation SD
  real<lower=-1,upper=1> rho;             // AR(1) coefficient

  // --- Per-block time-invariant RE ---
  vector[B] u_block_raw;                   // non-centred block offsets ~ normal(0,1)
  real<lower=0> sigma_block;              // block RE SD
}

transformed parameters {
  // --- HSGP spatial RE ---
  vector[B] u_gp;
  {
    vector[M_total] diag_SPD;
    real inv_rho2 = 1.0 / rho_gp^2;
    for (m in 1:M_total)
      diag_SPD[m] = sigma_gp * sqrt(2.0 * pi() / rho_gp) * (inv_rho2 + lambda[m])^(-0.75);
    u_gp = PHI * (diag_SPD .* beta_gp);
  }

  // --- Global AR(1) ---
  vector[T] v_global;
  v_global[1] = sigma_global * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T)
    v_global[t] = rho * v_global[t-1] + sigma_global * v_global_raw[t];

  // --- Per-block RE ---
  vector[B] u_block = sigma_block * u_block_raw;

  // --- Environmental effects (N-indexed, observed cells only) ---
  vector[N] xeff_gamma = X_lag_flat * to_vector(w_gamma) + X_unlagged * w_unlagged_gamma;
  vector[N] xeff_phi   = X_lag_flat * to_vector(w_phi)   + X_unlagged * w_unlagged_phi;

  // --- B×T linear predictors for colonisation and persistence ---
  // Unobserved cells retain baseline RE; observed cells get full covariate effects.
  matrix[B, T] eta_gamma_mat;
  matrix[B, T] eta_phi_mat;
  for (b in 1:B) {
    for (t in 1:T) {
      real shared_re = u_gp[b] + v_global[t] + u_block[b];
      eta_gamma_mat[b, t] = alpha_gamma + shared_re;
      eta_phi_mat[b, t]   = alpha_phi   + shared_re;
    }
  }
  for (i in 1:N) {
    int b = block[i];
    int t = time[i];
    real shared_re = u_gp[b] + v_global[t] + u_block[b];
    eta_gamma_mat[b, t] = alpha_gamma + xeff_gamma[i] + shared_re;
    eta_phi_mat[b, t]   = alpha_phi   + xeff_phi[i]   + shared_re;
  }

  // --- Mean-field visit probability q_bt ---
  // Forward recursion from t=1, initialised at stationary distribution:
  //   q* = v_new / (v_new + (1 - v_pers))
  matrix[B, T] q_mat;
  for (b in 1:B) {
    real C_b1    = C_mat[b, 1] * 1.0;
    real v_new1  = inv_logit(alpha0 + alpha1 * C_b1);
    real v_pers1 = inv_logit(alpha0 + alpha1 * C_b1 + alpha2);
    q_mat[b, 1]  = v_new1 / fmax(1e-8, v_new1 + (1.0 - v_pers1));

    for (t in 2:T) {
      real C_bt_r = C_mat[b, t] * 1.0;
      real v_new  = inv_logit(alpha0 + alpha1 * C_bt_r);
      real v_pers = inv_logit(alpha0 + alpha1 * C_bt_r + alpha2);
      q_mat[b, t] = v_new * (1.0 - q_mat[b, t-1]) + v_pers * q_mat[b, t-1];
    }
  }

  // --- Mean-field mosquito presence probability p_bt ---
  // Forward recursion from t=1, initialised at stationary distribution:
  //   p* = gamma / (gamma + (1 - phi))  [no control at t=0]
  // Transition uses q_{t-1} to split persistence between controlled/uncontrolled:
  //   persist_t = (1 - q_{t-1}) * phi_t + q_{t-1} * phi_eff_t
  matrix[B, T] p_mat;
  for (b in 1:B) {
    real g0    = inv_logit(eta_gamma_mat[b, 1]);
    real f0    = inv_logit(eta_phi_mat[b, 1]);
    p_mat[b, 1] = g0 / fmax(1e-8, g0 + (1.0 - f0));

    for (t in 2:T) {
      real gamma_t   = inv_logit(eta_gamma_mat[b, t]);
      real phi_t     = inv_logit(eta_phi_mat[b, t]);
      real phi_eff_t = inv_logit(eta_phi_mat[b, t] - theta);
      real q_prev    = q_mat[b, t-1];
      real persist_t = (1.0 - q_prev) * phi_t + q_prev * phi_eff_t;
      p_mat[b, t]    = (1.0 - p_mat[b, t-1]) * gamma_t + p_mat[b, t-1] * persist_t;
    }
  }

  // --- Observation probability r_bt = p_bt * q_bt ---
  matrix[B, T] r_mat;
  for (b in 1:B)
    for (t in 1:T)
      r_mat[b, t] = p_mat[b, t] * q_mat[b, t];
}

model {
  // ==== Priors ====

  // Ecological intercepts
  alpha_gamma ~ normal(-5.0, 1.5);   // ~1% colonisation: logit(0.01) ≈ -4.6
  alpha_phi   ~ normal(-1.0, 1.5);   // ~30% persistence:  logit(0.3)  ≈ -0.85

  // Distributed lag weights (random walk prior for smoothness across lags)
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

  // Intervention: positive = reduces persistence
  theta ~ normal(1.5, 1.0);

  // Visit Markov chain
  // alpha0: baseline ~85% monthly visit prob 
  alpha0 ~ normal(logit(0.85), 0.3);
  // alpha1: positive — more dengue cases → more visits
  alpha1 ~ normal(1.0, 1.0);
  // alpha2: centered at zero, wide enough for data to decide
  alpha2 ~ normal(0, 1.0);

  // Spatial GP
  beta_gp  ~ normal(0, 1);
  sigma_gp ~ normal(0, 1);
  rho_gp   ~ inv_gamma(3, 150);        // mode ~75m

  // Temporal
  v_global_raw ~ normal(0, 1);
  sigma_global ~ exponential(1);
  rho          ~ normal(0.4, 0.2);

  // Block RE
  u_block_raw ~ normal(0, 1);
  sigma_block ~ exponential(3);

  // ==== Observation model  ====
  // Y_bt ~ Binomial(n_mat[b,t], r_bt)  where r_bt = p_bt * q_bt
  // Trials = actual inspection events (baseline + kappa*C_bt reactive),
  // which is always >= y_mat. r_bt is the per-inspection detection rate.
  for (b in 1:B) {
    for (t in 1:T) {
      if (n_mat[b, t] > 0) {
        y_mat[b, t] ~ binomial(n_mat[b, t], r_mat[b, t]);
      }
    }
  }
}

generated quantities {
  // ---- Latent state summaries ----
  matrix[B, T] p_out = p_mat;   // E[z_ibt]: block-level mosquito presence probability
  matrix[B, T] q_out = q_mat;   // E[v_ibt]: block-level visit probability
  matrix[B, T] r_out = r_mat;   // r_bt = p_bt * q_bt: detection probability

  // ---- Posterior predictive samples (B×T grid) ----
  array[B, T] int y_pred_mat;
  for (b in 1:B) {
    for (t in 1:T) {
      if (n_mat[b, t] > 0) {
        y_pred_mat[b, t] = binomial_rng(n_mat[b, t], r_mat[b, t]);
      } else {
        y_pred_mat[b, t] = 0;
      }
    }
  }

  // ---- Per-observation log-likelihood for LOO-CV ----
  vector[N] log_lik;
  for (i in 1:N) {
    int b = block[i];
    int t = time[i];
    log_lik[i] = binomial_lpmf(y[i] | n_bt[i], r_mat[b, t]);
  }
}
