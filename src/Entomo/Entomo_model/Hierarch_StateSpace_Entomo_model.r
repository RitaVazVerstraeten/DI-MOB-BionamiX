# =====================================================
# Mosquito hierarchical model with reactive surveillance
# Stan + R implementation
# =====================================================

# Initialize in your project folder
# renv::init()  # creates renv/ folder and lockfile
if (!require("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}
# install.packages("dplyr")
# install.packages(c("rmarkdown", "knitr", "yaml", "jsonlite", "xfun"))
renv::restore()  # Restore from renv cache (binary if available)
# --- 0. Load libraries ---
library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)

# --- Detect hostname and set cores and data path accordingly ---
hostname <- Sys.info()["nodename"]
if (hostname == "frietjes") {
  options(mc.cores = 6)  # Use 6 cores in parallel on frietjes
  data_dir <- "~/data/Entomo"
} else {
  options(mc.cores = 2)  # Conservative: 2 chains sequentially on local machine (15GB RAM system)
  # Default to local path
  data_dir <- "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/"
}

data_file <- file.path(data_dir, "env_epi_entomo_data_per_manzana_2016_04_to_2019_12.csv")
cat("Using hostname:", hostname, "\n")
cat("Data directory:", data_dir, "\n")

# --- Create output directory ---
output_dir <- file.path("/home/rita/PyProjects/DI-MOB-BionamiX", "results", "Entomo", "fitting")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
date_suffix <- format(Sys.Date(), "%Y%m%d")  # e.g., "20260216"

input_data <- read_csv(data_file)

# # --- 1. Example dataset (simulate for demonstration) ---
# set.seed(123)
# B <- 10  # number of blocks
# T <- 20  # number of time periods
# df <- expand.grid(block = 1:B, time = 1:T)
# df <- df[order(df$block, df$time), ]
# df$N_HH <- sample(50:70, nrow(df), replace = TRUE)
# df$C_bt <- rpois(nrow(df), lambda = 2)   # dengue cases
# df$X1 <- rnorm(nrow(df))
# df$X2 <- rnorm(nrow(df))

# # True latent mosquito probability
# alpha <- -1
# w_true <- matrix(c(0.6, 0.3, 0.1, 0.5, 0.3, 0.2),
#                                          nrow = K, byrow = TRUE)
# u_block <- rnorm(B, 0, 0.3)
# # Temporal random effects with AR(1) structure: v_t = ρ_v * v_{t-1} + ε_t
# rho_v <- 0.5
# v_time <- numeric(T)
# v_time[1] <- rnorm(1, 0, 0.2 / sqrt(1 - rho_v^2))
# for (t in 2:T) {
#   v_time[t] <- rho_v * v_time[t-1] + rnorm(1, 0, 0.2)
# }

# X_effect <- numeric(nrow(df))
# for (i in seq_len(nrow(df))) {
#      X_effect[i] <- sum(X_lag[i, , ] * w_true)
# }
# logit_p_bt <- alpha + X_effect + u_block[df$block] + v_time[df$time]
# df$p_bt <- plogis(logit_p_bt)

# # Reactive bias parameters (used for synthetic data generation; will be estimated in Stan)
# delta0 <- 0.5  # Aligned with prior N(0.5, 0.5)
# delta1 <- 0.2  # Aligned with prior N(0, 0.3)
# kappa <- 2  # Aligned with LogNormal(log(2), 0.4) centered at 2

# # Mixture weights
# df$omega <- ifelse(df$C_bt > 0,
#                    kappa * df$C_bt / (df$N_HH + kappa*df$C_bt),
#                    0)

# # Reactive probability
# df$p_R <- plogis(qlogis(df$p_bt) + delta0 + delta1*log(df$C_bt + 1))

# # Total inspection events
# df$n_bt <- df$N_HH + kappa*df$C_bt

# # Observed positives
# df$y_bt <- rbinom(nrow(df), size = df$n_bt, prob = (1-df$omega)*df$p_bt + df$omega*df$p_R)


# --- 1. Set-up input data variables ---
# Ensure year_month is ordered and create block/time indices
input_data <- input_data %>%
     mutate(year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d")) %>%
     relocate(year_month_date, .after = year_month) %>%
     select(!c(CMF, CP, AREA))
 
block_levels <- sort(unique(input_data$manzana)) # 1337
time_levels <- sort(unique(input_data$year_month_date)) # 45

# creates integer IDs for each manzana and month, then sorts the data by those IDs.
df <- input_data %>%
     mutate(
          block = match(manzana, block_levels),
          time = match(year_month_date, time_levels)
     ) %>%
     arrange(block, time)

B <- length(block_levels)
T <- length(time_levels)

df <- df %>%
     rename(
          N_HH = Inspected_houses,
          C_bt = cases, 
          y_bt = Houses_pos_IS, 
     ) %>%
     mutate(
          N_HH = as.integer(N_HH),
          C_bt = as.integer(C_bt),
          y_bt = as.integer(y_bt)
     )

# Covariates for distributed lags
lag_vars <- c("avg_temp", "rel_hum", "total_precip", "WS2M", "mean_ndmi", "mean_ndwi", "mean_ndvi")
df <- df %>%
     mutate(across(all_of(lag_vars), ~coalesce(., 0)))

# Distributed lag setup
K <- length(lag_vars)
L <- 2  # maximum lag
Lp1 <- L + 1
X_lag <- array(0, dim = c(nrow(df), K, Lp1))

for (b in 1:B) {
     idx <- which(df$block == b)        # for each block
     for (k in seq_len(K)) {
          x <- df[[lag_vars[k]]][idx]   # extract as vector for this block
          for (l in 0:L) {              # for each lag time 
               if (l == 0) {
                    lagged <- x
               } else {
                    lagged <- c(rep(NA_real_, l), x[1:(length(x) - l)]) # Shifts the values back by l positions, with leading NAs 
               }
               X_lag[idx, k, l + 1] <- lagged
          }
     }
}

# Remove early rows without sufficient lag history (time > L for each block)
rows_to_keep <- df$time > L
df <- df[rows_to_keep, ]
X_lag <- X_lag[rows_to_keep, , ]

# Flatten X_lag to 2D matrix for efficient Stan computation
# X_lag[N, K, Lp1] becomes X_lag_flat[N, K*Lp1]
X_lag_flat <- matrix(0, nrow = nrow(df), ncol = K * Lp1)
for (i in seq_len(nrow(df))) {
  X_lag_flat[i, ] <- as.vector(X_lag[i, , ])
}

# Unlagged covariates (block-level characteristics)
unlagged_vars <- c("is_urban", "has_aljibes", "nr_aljibes", "is_WI", "is_WUI")
X_unlagged <- as.matrix(df[, unlagged_vars])
Ku <- ncol(X_unlagged)

# --- 2. Prepare data for Stan ---
stan_data <- list(
  N = nrow(df),
  y = df$y_bt,           # mosquito findings
  N_HH = df$N_HH,        # universe
  K = K,                 # number of lagged env covariates
  Lp1 = Lp1,             # total number of lag terms including lag 0 (3 for max_lag=2)
  X_lag_flat = X_lag_flat,  # flattened lagged covariates matrix [N, K*Lp1]
  Ku = Ku,               # number of unlagged block-level covariates
  X_unlagged = X_unlagged,  # unlagged covariates matrix [N, Ku]
  B = B,                 # number of manzanas
  T = T,                 # number of time steps
  block = df$block,      # numeric block indices
  time = df$time,        # numeric time indices
  C_bt = df$C_bt         # dengue cases
  # kappa will be estimated as a parameter in Stan
)

# # --- DATA VALIDATION ---
# cat("\n=== DATA STRUCTURE VALIDATION ===\n")
# cat("Observations (N):", stan_data$N, "\n")
# cat("Blocks (B):", stan_data$B, "\n")
# cat("Time periods (T):", stan_data$T, "\n")
# cat("Lagged covariates (K):", stan_data$K, "\n")
# cat("Lag terms (Lp1):", stan_data$Lp1, "\n")
# cat("Unlagged covariates (Ku):", stan_data$Ku, "\n\n")

# cat("Data types:\n")
# cat("  y (observations):", class(stan_data$y), "length:", length(stan_data$y), "\n")
# cat("  N_HH (denominator):", class(stan_data$N_HH), "length:", length(stan_data$N_HH), "\n")
# cat("  C_bt (cases):", class(stan_data$C_bt), "length:", length(stan_data$C_bt), "\n")
# cat("  X_lag (lagged covariates):", class(stan_data$X_lag), "dim:", paste(dim(stan_data$X_lag), collapse=" x "), "\n")
# cat("  X_unlagged (static covariates):", class(stan_data$X_unlagged), "dim:", paste(dim(stan_data$X_unlagged), collapse=" x "), "\n")
# cat("  block:", class(stan_data$block), "length:", length(stan_data$block), "\n")
# cat("  time:", class(stan_data$time), "length:", length(stan_data$time), "\n\n")

# cat("Value ranges:\n")
# cat("  y_bt:", "min =", min(stan_data$y), ", max =", max(stan_data$y), ", mean =", round(mean(stan_data$y), 3), "\n")
# cat("  N_HH:", "min =", min(stan_data$N_HH), ", max =", max(stan_data$N_HH), ", mean =", round(mean(stan_data$N_HH), 1), "\n")
# cat("  C_bt:", "min =", min(stan_data$C_bt), ", max =", max(stan_data$C_bt), ", mean =", round(mean(stan_data$C_bt), 3), "\n")
# cat("  block:", "min =", min(stan_data$block), ", max =", max(stan_data$block), "\n")
# cat("  time:", "min =", min(stan_data$time), ", max =", max(stan_data$time), "\n\n")

# cat("Missing values (NAs):\n")
# cat("  y:", sum(is.na(stan_data$y)), "\n")
# cat("  N_HH:", sum(is.na(stan_data$N_HH)), "\n")
# cat("  C_bt:", sum(is.na(stan_data$C_bt)), "\n")
# cat("  X_lag:", sum(is.na(stan_data$X_lag)), "\n")
# cat("  X_unlagged:", sum(is.na(stan_data$X_unlagged)), "\n")
# cat("=================================\n\n")

# # --- TEST POISSON ASSUMPTION ---
# # small probabilities
# test_for_poisson <- df$y_bt / df$N_HH
# cat("\n1. Poisson Assumption Test Rare Event Condition -> p_bt is small: ")
# cat("\nsummary y_bt/N_HH \n")
# print(summary(test_for_poisson))

# # overdispersion
# mean_y <- mean(df$y_bt)
# var_y  <- var(df$y_bt)
# var_y / mean_y
# print("\n 2. Poisson assumption, no overdispersion; ")
# cat("\n var_y / mean_y if >> 1 -> overdispersion \n")
# print(var_y / mean_y)

# --- 3. Fit Stan model from external file ---
stan_file <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/hierarchical_state_space.stan"

# Provide reasonable initial values to help both chains start properly
init_fun <- function() {
  list(
    alpha = rnorm(1, -4.5, 0.4),              # Tighter SD to avoid extreme p_bt
    w = matrix(rnorm(stan_data$K * stan_data$Lp1, 0, 0.08), stan_data$K, stan_data$Lp1),  # Slightly reduced SD
    sigma_w = runif(stan_data$K, 0.1, 0.3),   # Random walk SD for lag smoothing
    w_unlagged = rnorm(stan_data$Ku, 0, 0.1),
    u_block_raw = rnorm(stan_data$B, 0, 0.5),
    v_time_raw = rnorm(stan_data$T, 0, 0.5),
    sigma_u = runif(1, 0.1, 0.5),
    sigma_v = runif(1, 0.1, 0.5),
    rho = rnorm(1, 0, 0.25),  # Allow negative autocorrelation
    delta0 = rnorm(1, 0.3, 0.2),
    delta1 = rnorm(1, 0, 0.1)
  )
}

# Compile and fit with cmdstanr
mod <- cmdstan_model(stan_file)

# fit <- mod$sample(
#   data = stan_data,
#   chains = 2,
#   iter_warmup = 1000,
#   iter_sampling = 1000,
#   thin = 2,  # Keep every 2nd sample (reduces memory)
#   init = init_fun,
#   adapt_delta = 0.90,
#   max_treedepth = 10,
#   parallel_chains = 1  # Sequential execution for memory safety
# )

fit <- mod$sample(
  data = stan_data,
  chains = 2,
  iter_warmup = 100,
  iter_sampling = 100,
#   thin = 2,  # Keep every 2nd sample (reduces memory)
  init = init_fun,
  adapt_delta = 0.95,
  max_treedepth = 12,
  parallel_chains = if (hostname == "frietjes") 2 else 1  # Parallel on frietjes, sequential on local for memory safety
)


# --- 4. Summarize results ---
summary_output <- capture.output(print(fit$summary(variables = c("alpha","sigma_u","sigma_v","rho","delta0","delta1","w"))))
cat(summary_output, sep = "\n")
writeLines(summary_output, file.path(output_dir, paste0("model_summary_", date_suffix, ".txt")))

# --- 5. Extract fitted latent mosquito probabilities ---
post_array <- fit$draws()  # Extract as array
fitted_p_bt <- apply(post_array[, , "p_bt_out"], 2, mean)  # posterior mean for each observation
fitted_p_R <- apply(post_array[, , "p_R_out"], 2, mean)  # reactive surveillance probability
df$fitted_p_bt <- fitted_p_bt
df$fitted_p_R <- fitted_p_R

# --- 6. Plot fitted vs observed proportions ---
# Use observed proportion (y_bt / N_HH) since true latent p_bt is not in the data
df$observed_prop <- df$y_bt / df$N_HH
p1 <- ggplot(df, aes(x = observed_prop, y = fitted_p_bt)) +
     geom_point(alpha=0.5, color="blue") +
     geom_abline(slope=1, intercept=0, color='red') +
     labs(x="Observed proportion (y_bt / N_HH)", y="Fitted p_bt (posterior mean)",
                title="Observed vs Fitted Mosquito Probability") +
     theme_minimal()
ggsave(file.path(output_dir, paste0("observed_vs_fitted_p_bt_", date_suffix, ".png")), 
                p1, width = 8, height = 6)
print(p1)

# Plot reactive surveillance probability
p2 <- ggplot(df, aes(x = fitted_p_bt, y = fitted_p_R)) +
  geom_point(alpha=0.5, color="darkgreen") +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(x="Fitted ecological probability (p_bt)", y="Fitted reactive probability (p_R)",
       title="Ecological vs Reactive Surveillance Probability") +
  theme_minimal()
ggsave(file.path(output_dir, paste0("observed_vs_fitted_pi_bt_", date_suffix, ".png")), 
       p2, width = 8, height = 6)
print(p2)

# --- 7. Plot block-level and temporal random effects ---
u_post <- apply(post_array[, , "u_block_out"], 2, mean)
v_post <- apply(post_array[, , "v_time_out"], 2, mean)

png(file.path(output_dir, paste0("random_effects_", date_suffix, ".png")), 
    width = 1000, height = 800)
par(mfrow=c(2,2))

# Spatial effects: histogram instead of line plot (too many blocks)
hist(u_post, breaks = 50, main = "Distribution of Spatial Random Effects (u_b)", 
     xlab = "Effect", col = "lightblue", border = "white")
abline(v = 0, lty = 2, col = "red", lwd = 2)
text(x = min(u_post), y = par("usr")[4] * 0.9, 
     labels = sprintf("n = %d blocks", length(u_post)), pos = 4, cex = 0.9)

# Spatial effects: quantile plot
qqnorm(u_post, main = "Q-Q Plot: Spatial Effects", pch = 19, cex = 0.5, col = "blue")
qqline(u_post, col = "red", lwd = 2)

# Temporal effects: line plot
plot(v_post, type = 'b', main = "Temporal Random Effects (v_t) with AR(1)", 
     xlab = "Time", ylab = "Effect", col = "red", pch = 19)
abline(h = 0, lty = 2, col = "gray")

# Temporal effects: ACF plot
acf(v_post, main = "ACF of Temporal Effects", col = "darkred")

par(mfrow=c(1,1))
dev.off()

# --- 8. Posterior predictive check ---
y_pred <- apply(post_array[, , "y_pred"], 2, mean)
p3 <- ggplot(data.frame(observed = df$y_bt, predicted = y_pred), 
       aes(x = observed, y = predicted)) +
  geom_point(alpha=0.5) +
  geom_abline(slope=1, intercept=0, color='red') +
  labs(x="Observed y_bt", y="Predicted y_bt (posterior mean)",
       title="Posterior Predictive Check") +
  theme_minimal()
ggsave(file.path(output_dir, paste0("posterior_predictive_check_", date_suffix, ".png")), 
       p3, width = 8, height = 6)
print(p3)


# --- 9. Trace plots for MCMC chains ---
if (requireNamespace("bayesplot", quietly = TRUE)) {
     library(bayesplot)
     # Extract draws as array for bayesplot
     draws_array <- as_draws_array(fit$draws())
     trace_params <- c("alpha", "sigma_u", "sigma_v", "rho", "delta0", "delta1")
     trace_plot <- mcmc_trace(draws_array, pars = trace_params)
     trace_file <- file.path(output_dir, paste0("traceplot_params_", date_suffix, ".png"))
     ggsave(trace_file, trace_plot, width = 10, height = 8)
     cat("Trace plot saved to:", trace_file, "\n")
} else {
     cat("bayesplot package not installed; skipping trace plot.\n")
}

cat("\nAll outputs saved to:", output_dir, "\n")
