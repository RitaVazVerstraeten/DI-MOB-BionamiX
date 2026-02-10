# =====================================================
# Mosquito hierarchical model with reactive surveillance
# Stan + R implementation
# =====================================================

# Initialize in your project folder
# renv::init()  # creates renv/ folder and lockfile
install.packages("rstan")
install.packages("dplyr")
install.packages(c("rmarkdown", "knitr", "yaml", "jsonlite", "xfun"))
renv::snapshot()
# --- 0. Load libraries ---
library(rstan)
library(dplyr)
library(ggplot2)

rstan_options(auto_write = TRUE)
# options(mc.cores = parallel::detectCores())

# --- 1. Example dataset (simulate for demonstration) ---
set.seed(123)
B <- 10  # number of blocks
T <- 20  # number of time periods
df <- expand.grid(block = 1:B, time = 1:T)
df <- df[order(df$block, df$time), ]
df$N_HH <- sample(50:70, nrow(df), replace = TRUE)
df$C_bt <- rpois(nrow(df), lambda = 2)   # dengue cases
df$X1 <- rnorm(nrow(df))
df$X2 <- rnorm(nrow(df))

# Distributed lag setup
lag_vars <- c("X1", "X2")
K <- length(lag_vars)
L <- 2  # maximum lag
Lp1 <- L + 1
X_lag <- array(0, dim = c(nrow(df), K, Lp1))

for (b in 1:B) {
     idx <- which(df$block == b)
     for (k in seq_len(K)) {
          x <- df[idx, lag_vars[k]]
          for (l in 0:L) {
               if (l == 0) {
                    lagged <- x
               } else {
                    lagged <- c(rep(NA_real_, l), x[1:(length(x) - l)])
               }
               X_lag[idx, k, l + 1] <- lagged
          }
     }
}
X_lag[is.na(X_lag)] <- 0

# True latent mosquito probability
alpha <- -1
w_true <- matrix(c(0.6, 0.3, 0.1,
                                              0.5, 0.3, 0.2),
                                         nrow = K, byrow = TRUE)
u_block <- rnorm(B, 0, 0.3)
v_time <- rnorm(T, 0, 0.2)
X_effect <- numeric(nrow(df))
for (i in seq_len(nrow(df))) {
     X_effect[i] <- sum(X_lag[i, , ] * w_true)
}
logit_p_bt <- alpha + X_effect + u_block[df$block] + v_time[df$time]
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
     K = K,
     Lp1 = Lp1,
     X_lag = X_lag,
  B = B,
  T = T,
  block = df$block,
  time = df$time,
  C_bt = df$C_bt,
  kappa = kappa  # Fixed scaling factor
)

# --- 3. Fit Stan model from external file ---
fit <- stan(file = "src/Entomo/hierarchical_state_space.stan", 
            data = stan_data, 
            chains = 4, 
            iter = 2000, 
            warmup = 1000)

# --- 4. Summarize results ---
print(fit, pars = c("alpha","sigma_u","sigma_v","rho","w"))

# --- 5. Extract fitted latent mosquito probabilities ---
post <- extract(fit)
fitted_p_bt <- apply(post$p_bt, 2, mean)  # posterior mean for each observation
fitted_pi_bt <- apply(post$pi_bt, 2, mean)  # effective observation probability
df$fitted_p_bt <- fitted_p_bt
df$fitted_pi_bt <- fitted_pi_bt

# --- 6. Plot fitted vs observed proportions ---
# Plot latent ecological probability
ggplot(df, aes(x = p_bt, y = fitted_p_bt)) +
  geom_point(alpha=0.5, color="blue") +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(x="True p_bt (latent)", y="Fitted p_bt (posterior mean)",
       title="True vs Fitted Latent Mosquito Probability") +
  theme_minimal()

# Plot observation model fit
ggplot(df, aes(x = y_bt / n_bt, y = fitted_pi_bt)) +
  geom_point(alpha=0.5, color="darkgreen") +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(x="Observed mosquito proportion (y_bt/n_bt)", y="Fitted pi_bt",
       title="Observed vs Fitted Observation Probability") +
  theme_minimal()

# --- 7. Plot block-level and temporal random effects ---
u_post <- apply(post$u_block, 2, mean)
v_post <- apply(post$v_time, 2, mean)

par(mfrow=c(2,1))
plot(u_post, type='b', main="Spatial random effects (u_b)", 
     xlab="Block", ylab="Effect", col="blue", pch=19)
abline(h=0, lty=2, col="gray")

plot(v_post, type='b', main="Temporal random effects (v_t) with AR(1)", 
     xlab="Time", ylab="Effect", col="red", pch=19)
abline(h=0, lty=2, col="gray")
par(mfrow=c(1,1))

# --- 8. Posterior predictive check ---
y_pred <- apply(post$y_pred, 2, mean)
ggplot(data.frame(observed = df$y_bt, predicted = y_pred), 
       aes(x = observed, y = predicted)) +
  geom_point(alpha=0.5) +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(x="Observed y_bt", y="Predicted y_bt (posterior mean)",
       title="Posterior Predictive Check") +
  theme_minimal()
