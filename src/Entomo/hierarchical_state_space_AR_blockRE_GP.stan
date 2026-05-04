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
  matrix[B, B] dist_block;       // pairwise distances between block centroids (metres)

  int<lower=0,upper=1> fix_phi;    // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;          // value used when fix_phi = 1 (ignored otherwise)
}

transformed data {
  real kappa = 2.0;  // fixed scaling factor for reactive inspections
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights for environmental covariates
  vector[Ku] w_unlagged;   // weights for unlagged block-level covariates
  vector[T] v_global_raw;          // global AR(1) trend (non-centered)
  vector[B] v_block_dev_raw;       // per-block deviation from global trend (non-centered)
  real<lower=0> sigma_v;           // SD of global temporal trend
  real<lower=-1,upper=1> rho;      // AR(1) coefficient
  real<lower=0> sigma_block_dev;   // SD of per-block deviations
  real<lower=0> delta1;            // linear increase in detection probability with dengue cases
  real<lower=0> sigma_gp;  // GP marginal SD (spatial)
  real<lower=0> rho_gp;    // GP length scale (metres); exp kernel: corr = exp(-d/rho_gp)
  vector[B] z_gp;          // non-centred GP weights ~ normal(0,1)
  real<lower=0> phi_raw;   // beta-binomial concentration; used only when fix_phi = 0
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;
  vector[N] p_bt;          // latent ecological probability (true mosquito presence)
  vector[N] p_R;           // reactive surveillance probability (biased upward)
  vector[N] omega;         // fraction of inspections that are reactive (omega_bt = kappa*C_bt / n_bt)
  vector[N] pi;            // effective observation probability (mixture of p_bt and p_R)
  vector[T] v_global;      // global AR(1) temporal trend
  vector[B] v_block_dev;  // per-block deviation from global trend
  vector[N] x_effect;      // linear predictor for environmental effects
  vector[B] u_gp;          // spatial random effects: GP with exponential kernel

  // 1. Gaussian process with exponential kernel (no distance cap)
  //     K[b1,b2] = sigma_gp^2 * exp(-dist/rho_gp)
  //     Cholesky parameterisation: u_gp = L_gp * z_gp
  //     NOTE: Cholesky of a B×B matrix is computed every HMC step → O(B³).
  //     For large B consider an approximate GP (HSGP) instead.
  {
    matrix[B, B] K_gp;
    for (b1 in 1:B) {
      for (b2 in b1:B) {
        real k = sigma_gp^2 * exp(-dist_block[b1, b2] / rho_gp);
        K_gp[b1, b2] = k;
        K_gp[b2, b1] = k;
      }
      K_gp[b1, b1] = sigma_gp^2 * 1.05;  // nugget = 5% of marginal variance (scale-consistent)
    }
    u_gp = cholesky_decompose(K_gp) * z_gp;
  }

  // 2. Global AR(1) trend + per-block deviations
  //    v_global[t] = rho * v_global[t-1] + sigma_v * eps_t  (shared across blocks)
  //    v_block_dev[b] ~ N(0, sigma_block_dev)                (block-specific offset)
  v_global[1] = sigma_v * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T) {
    v_global[t] = rho * v_global[t-1] + sigma_v * v_global_raw[t];
  }
  v_block_dev = sigma_block_dev * v_block_dev_raw;

  // 3. Calculate environmental effects (matrix multiplication)
  // Flatten w to [K*Lp1] and multiply with X_lag_flat[N, K*Lp1]
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 4. Calculate linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N) {
    eta[i] = alpha + x_effect[i] + u_gp[block[i]] + v_global[time[i]] + v_block_dev[block[i]];
  }
  p_bt = inv_logit(eta);

  // 5. Reactive surveillance probability (loop-based conditional)
  // Work on linear predictor scale to avoid numerical issues
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta1 * C_bt[i]);
    } else {
      p_R[i] = p_bt[i];  // no reactive bias when no cases
    }
  }

  // 6. Fraction of reactive inspections: omega_bt = kappa * C_bt / n_bt
  // 7. Calculation of p_i depending on C_bt[i]
  for (i in 1:N) {
    if (n_bt[i] == 0) {
      omega[i] = 0;
      pi[i] = 0; // or set to p_bt[i], depending on your model logic
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
  // Priors
  alpha ~ normal(-7.0, 1.5);                    // 0.5–5% prevalence → logit ≈ −5.3 to −2.9; centred on rarer events
  
  to_vector(w)    ~ normal(0, 1.0);
  w_unlagged      ~ normal(0, 0.5);
  v_global_raw    ~ normal(0, 1);
  v_block_dev_raw ~ normal(0, 1);
  sigma_v         ~ exponential(1);
  sigma_block_dev ~ exponential(2);
  rho             ~ normal(0.4, 0.2);
  delta1          ~ normal(0, 0.5);
  z_gp            ~ normal(0, 1);       // non-centred GP weights
  sigma_gp  ~ normal(0, 1);       // GP marginal SD (half-normal)
  rho_gp    ~ inv_gamma(3, 150);  // mode at 75m, matching observed residual spatial peak
  if (fix_phi == 0) phi_raw ~ gamma(2, 0.1);

  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}

generated quantities {
  // Save probabilities and random effects for posterior analysis
  vector[N] p_bt_out = p_bt;
  vector[N] p_R_out = p_R;
  vector[B] u_gp_out          = u_gp;
  vector[T] v_global_out      = v_global;
  vector[B] v_block_dev_out   = v_block_dev;

  // Posterior predictive checks
  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i] = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6));
  }
}
