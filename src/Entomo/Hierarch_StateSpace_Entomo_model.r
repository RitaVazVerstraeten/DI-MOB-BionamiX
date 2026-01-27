# =====================================================
# Mosquito hierarchical model with reactive surveillance
# Stan + R implementation
# =====================================================

# Initialize in your project folder
# renv::init()  # creates renv/ folder and lockfile
# install.packages("rstan")
# install.packages("dplyr")
# install.packages(c("rmarkdown", "knitr", "yaml", "jsonlite", "xfun"))
renv::snapshot()
# --- 0. Load libraries ---
library(rstan)
library(dplyr)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# --- 1. Example dataset (simulate for demonstration) ---
set.seed(123)
B <- 10  # number of blocks
T <- 20  # number of time periods
df <- expand.grid(block = 1:B, time = 1:T)
df$N_HH <- sample(50:70, nrow(df), replace = TRUE)
df$C_bt <- rpois(nrow(df), lambda = 2)   # dengue cases
df$X1 <- rnorm(nrow(df))
df$X2 <- rnorm(nrow(df))

# True latent mosquito probability
alpha <- -1
beta1 <- 0.5
beta2 <- -0.3
u_block <- rnorm(B, 0, 0.3)
v_time <- rnorm(T, 0, 0.2)
logit_p_bt <- alpha + beta1*df$X1 + beta2*df$X2 +
              u_block[df$block] + v_time[df$time]
df$p_bt <- plogis(logit_p_bt)

# Reactive bias parameters
delta0 <- 1
delta1 <- 0.5
kappa <- 1

# Mixture weights
df$omega <- ifelse(df$C_bt > 0,
                   kappa * df$C_bt / (df$N_HH + kappa*df$C_bt),
                   0)

# Reactive probability
df$p_R <- plogis(qlogis(df$p_bt) + delta0 + delta1*log(df$C_bt + 1))

# Total inspection events
df$n_bt <- df$N_HH + kappa*df$C_bt

# Observed positives
df$y_bt <- rbinom(nrow(df), size = df$n_bt, prob = (1-df$omega)*df$p_bt + df$omega*df$p_R)

# --- 2. Prepare data for Stan ---
stan_data <- list(
  N = nrow(df),
  y = df$y_bt,
  n_bt = df$n_bt,
  X1 = df$X1,
  X2 = df$X2,
  B = B,
  T = T,
  block = df$block,
  time = df$time,
  C_bt = df$C_bt,
  omega = df$omega
)

# --- 3. Stan model code ---
stan_code <- "
data {
  int<lower=1> N;          // number of observations
  int<lower=0> y[N];       // observed positives
  int<lower=1> n_bt[N];    // number of inspection events
  vector[N] X1;
  vector[N] X2;
  int<lower=1> B;          // blocks
  int<lower=1> T;          // time periods
  int<lower=1,upper=B> block[N];
  int<lower=1,upper=T> time[N];
  vector[N] C_bt;
  vector[N] omega;
}

parameters {
  real alpha;
  real beta1;
  real beta2;
  vector[B] u_block;
  vector[T] v_time;
  real delta0;
  real delta1;
  real<lower=0> sigma_u;
  real<lower=0> sigma_v;
}

transformed parameters {
  vector[N] p_bt;
  vector[N] p_R;
  vector[N] pi_bt;

  for (i in 1:N) {
    // latent mosquito probability
    p_bt[i] = inv_logit(alpha + beta1*X1[i] + beta2*X2[i] + u_block[block[i]] + v_time[time[i]]);
    // reactive surveillance probability
    p_R[i] = inv_logit(logit(p_bt[i]) + delta0 + delta1 * log(C_bt[i] + 1));
    // mixture probability
    pi_bt[i] = (1 - omega[i]) * p_bt[i] + omega[i] * p_R[i];
  }
}

model {
  // Priors
  alpha ~ normal(0,5);
  beta1 ~ normal(0,2);
  beta2 ~ normal(0,2);
  u_block ~ normal(0,sigma_u);
  v_time ~ normal(0,sigma_v);
  sigma_u ~ normal(0,2);
  sigma_v ~ normal(0,2);
  delta0 ~ normal(0,1);
  delta1 ~ normal(0,1);

  // Likelihood
  y ~ binomial(n_bt, pi_bt);
}
"

# --- 4. Fit Stan model ---
fit <- stan(model_code = stan_code, data = stan_data, chains = 4, iter = 2000, warmup = 1000)

# --- 5. Summarize results ---
print(fit, pars = c("alpha","beta1","beta2","delta0","delta1","sigma_u","sigma_v"))

# --- 6. Extract fitted latent mosquito probabilities ---
post <- extract(fit)
fitted_p_bt <- apply(post$p_bt, 2, mean)  # posterior mean for each observation
df$fitted_p_bt <- fitted_p_bt

# --- 7. Plot fitted vs observed proportions ---
ggplot(df, aes(x = y_bt / n_bt, y = fitted_p_bt)) +
  geom_point(alpha=0.5) +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(x="Observed mosquito proportion", y="Fitted p_bt",
       title="Observed vs Fitted Mosquito Probability") +
  theme_minimal()

# --- 8. Optional: plot block-level random effects ---
u_post <- apply(post$u_block, 2, mean)
v_post <- apply(post$v_time, 2, mean)
plot(u_post, type='b', main="Block random effects (u_b)")
plot(v_post, type='b', main="Time random effects (v_t)")
