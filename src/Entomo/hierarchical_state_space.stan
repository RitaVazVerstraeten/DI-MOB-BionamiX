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
}

transformed data {
  real kappa = 2.0;  // fixed scaling factor for reactive inspections
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights for environmental covariates
  vector<lower=0>[K] sigma_w;  // random walk SD for each covariate's lag structure
  vector[Ku] w_unlagged;   // weights for unlagged block-level covariates
  matrix[B, T] v_time_raw;    // temporal random effects (non-centered), per block
  real<lower=0> sigma_v;      // SD of temporal random effects (shared)
  real<lower=-1,upper=1> rho; // temporal autoregression parameter (shared)
  real delta0;             // baseline targeting bias (reactive surveillance)
  real delta1;             // log-linear increase with outbreak intensity
  real<lower=0> phi;       // beta-binomial concentration (phi→∞ = binomial)
  real<lower=0> sigma_gp;  // GP marginal SD (spatial)
  real<lower=0> rho_gp;    // GP length scale (metres); exp kernel: corr = exp(-d/rho_gp)
  vector[B] z_gp;          // non-centred GP weights ~ normal(0,1)
}

transformed parameters {
  vector[N] p_bt;          // latent ecological probability (true mosquito presence)
  vector[N] p_R;           // reactive surveillance probability (biased upward)
  vector[N] omega;         // fraction of inspections that are reactive (omega_bt = kappa*C_bt / n_bt)
  vector[N] pi;            // effective observation probability (mixture of p_bt and p_R)
  matrix[B, T] v_time;     // temporal random effects with AR(1), per block
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

  // 2. AR(1) temporal structure per block: v_t = rho * v_{t-1} + epsilon_t
  for (b in 1:B) {
    v_time[b, 1] = sigma_v * v_time_raw[b, 1] / sqrt(fmax(1e-6, 1 - rho^2));
    for (t in 2:T) {
      v_time[b, t] = rho * v_time[b, t-1] + sigma_v * v_time_raw[b, t];
    }
  }

  // 3. Calculate environmental effects (matrix multiplication)
  // Flatten w to [K*Lp1] and multiply with X_lag_flat[N, K*Lp1]
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 4. Calculate linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N) {
    eta[i] = alpha + x_effect[i] + u_gp[block[i]] + v_time[block[i], time[i]];
  }
  p_bt = inv_logit(eta);

  // 5. Reactive surveillance probability (loop-based conditional)
  // Work on linear predictor scale to avoid numerical issues
  for (i in 1:N) {
    if (C_bt[i] > 0) {
      p_R[i] = inv_logit(eta[i] + delta0 + delta1 * log(C_bt[i]));
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
  
  // Random walk prior on lag weights: enforces smoothness across lags
  for (k in 1:K) {
    w[k, 1] ~ normal(0, 0.5);  // initial lag-0 weight
    for (l in 2:Lp1) {
      w[k, l] ~ normal(w[k, l-1], sigma_w[k]);  // random walk
    }
  }
  sigma_w ~ exponential(2);  // shrink toward smooth lag structure
  
  w_unlagged ~ normal(0, 0.5);    // unlagged covariate weights
  to_vector(v_time_raw) ~ normal(0, 1);      // non-centered parameterization for all blocks and times
  sigma_v ~ exponential(1);        // temporal scale: loosened (mean = 1.0 vs 0.33)
  rho ~ normal(0.4, 0.2);          // recentred toward positive autocorrelation
  delta0 ~ normal(0.3, 0.4);      // slightly reduced baseline targeting bias
  delta1 ~ normal(0, 0.2);        // reduced log-linear increase to stabilize init
  phi    ~ gamma(2, 0.1);         // concentration: mean=20, allows moderate overdispersion
                                   // phi→∞ recovers binomial; small phi = heavy overdispersion
  z_gp      ~ normal(0, 1);       // non-centred GP weights
  sigma_gp  ~ normal(0, 1);       // GP marginal SD (half-normal)
  rho_gp    ~ inv_gamma(3, 150);  // mode at 75m, matching observed residual spatial peak

  // Observation model: y_bt ~ BetaBinomial(n_bt, pi*phi, (1-pi)*phi)
  // Beta-binomial relaxes the binomial variance assumption, allowing overdispersion.
  // Mean is identical to binomial (n * pi); variance = n*pi*(1-pi)*(n+phi)/(1+phi).
  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], pi[i] * phi, (1 - pi[i]) * phi);
  }
}

generated quantities {
  // Save probabilities and random effects for posterior analysis
  vector[N] p_bt_out = p_bt;
  vector[N] p_R_out = p_R;
  vector[B] u_gp_out    = u_gp;
  matrix[B, T] v_time_out  = v_time;
  
  // Posterior predictive checks
  array[N] int<lower=0> y_pred;
  vector[N] log_lik;
  
  for (i in 1:N) {
    y_pred[i] = beta_binomial_rng(n_bt[i], pi[i] * phi, (1 - pi[i]) * phi);
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], pi[i] * phi, (1 - pi[i]) * phi);
  }
}
