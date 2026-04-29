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

  // ----- ICAR neighbourhood graph -----
  int<lower=0> N_edges;                          // number of undirected edges
  array[N_edges] int<lower=1,upper=B> node1;     // first node of each edge (node1 < node2)
  array[N_edges] int<lower=1,upper=B> node2;     // second node of each edge

  int<lower=0,upper=1> fix_phi;  // 1 = phi fixed at phi_data; 0 = phi estimated
  real<lower=0> phi_data;        // value used when fix_phi = 1 (ignored otherwise)
}

transformed data {
  real kappa = 2.0;  // fixed scaling factor for reactive inspections
}

parameters {
  real alpha;              // baseline intercept
  matrix[K, Lp1] w;        // distributed lag weights for environmental covariates (free estimation)
  vector[Ku] w_unlagged;   // weights for unlagged block-level covariates
  vector[T] v_global_raw;          // global AR(1) trend (non-centered)
  real<lower=0> sigma_v;           // SD of global temporal trend
  real<lower=-1,upper=1> rho;      // AR(1) coefficient
  real<lower=0> delta1;    // linear increase in detection probability with dengue cases
  real<lower=0> phi_raw;   // beta-binomial concentration; used only when fix_phi = 0

  // ----- ICAR spatial random effects -----
  vector[B] u_icar_raw;            // raw ICAR spatial effects (unscaled) one per block
  real<lower=0> sigma_icar;        // marginal SD of the spatial RE
}

transformed parameters {
  real<lower=0> phi = fix_phi ? phi_data : phi_raw;  // active concentration
  vector[N] p_bt;          // latent ecological probability (true mosquito presence)
  vector[N] p_R;           // reactive surveillance probability (biased upward)
  vector[N] omega;         // fraction of inspections that are reactive (omega_bt = kappa*C_bt / n_bt)
  vector[N] pi;            // effective observation probability (mixture of p_bt and p_R)
  vector[T] v_global;      // global AR(1) temporal trend
  vector[N] x_effect;      // linear predictor for environmental effects
  vector[B] u_icar;        // scaled ICAR spatial effects

  // 1. ICAR spatial random effects (scaled) - hard centering
  u_icar = sigma_icar * u_icar_raw;
  u_icar = u_icar - mean(u_icar);

  // 2. Global AR(1) trend (no per-block deviation in this variant)
  v_global[1] = sigma_v * v_global_raw[1] / sqrt(fmax(1e-6, 1 - rho^2));
  for (t in 2:T)
    v_global[t] = rho * v_global[t-1] + sigma_v * v_global_raw[t];
  // Centre v_global so its mean is exactly zero.
  v_global = v_global - mean(v_global);

  // 3. Environmental effects
  x_effect = X_lag_flat * to_vector(w) + X_unlagged * w_unlagged;

  // 4. Linear predictor and latent ecological probability
  vector[N] eta;
  for (i in 1:N) {
    eta[i] = alpha + x_effect[i] + u_icar[block[i]] + v_global[time[i]];
    // baseline + environmental + spatial ICAR + global AR time
    // (no per-block temporal deviation)
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
  // Priors
  alpha ~ normal(-7.0, 1.5); //baseline intercept (log-odds of mosq presence) can go from 0.01% to 2% on probability scale 

  // Free lag weights: independent normal prior on each w[k,l]
  to_vector(w) ~ normal(0, 1.0); //on log-odds scale -> max plausible odds-ratio becomes 2.7x

  w_unlagged   ~ normal(0, 1.0);

  // dengue case correction:
  delta1      ~ normal(0, 0.5);  // half-normal (lower=0): positive bias expected
  if (fix_phi == 0) phi_raw ~ gamma(2, 0.1);  // prior when estimating: mean=20, weakly regularising
  
  // Temporal auto-correlation 
  v_global_raw    ~ normal(0, 1); //technical trick from Claude instead of putting prior on v_global which is harder for the sampler to explore
  
  sigma_v         ~ exponential(1); // controls variation in municipality-wide temporal trend
  
  rho             ~ normal(0.4, 0.2); //autocorrelation constrained ot -1, 1
  // v_global is centred exactly in transformed parameters (mean subtracted).

  // ICAR prior: pairwise differences penalise spatial discontinuity
  target += -0.5 * dot_self(u_icar_raw[node1] - u_icar_raw[node2]);

  sigma_icar ~ normal(0, 1); // Marginal SD: half-normal determines how much neighbouring blocks can differ 

  for (i in 1:N) {
    y[i] ~ beta_binomial(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6)); // observed y - output log-prob contribution - used for parameter estimation 
  }
}

generated quantities {
  vector[N] p_bt_out       = p_bt;
  vector[N] p_R_out        = p_R;
  vector[B] u_icar_out     = u_icar;
  vector[T] v_global_out   = v_global;

  array[N] int<lower=0> y_pred;
  vector[N] log_lik;

  for (i in 1:N) {
    y_pred[i]  = beta_binomial_rng(n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6)); // generates a random draw from the distribution
    log_lik[i] = beta_binomial_lpmf(y[i] | n_bt[i], fmax(pi[i] * phi, 1e-6), fmax((1 - pi[i]) * phi, 1e-6)); // log probability mass function 
  }
}
