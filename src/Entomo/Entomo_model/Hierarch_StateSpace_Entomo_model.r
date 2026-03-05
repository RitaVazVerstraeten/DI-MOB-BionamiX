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

# --- Run settings (quick toggles) ---
use_temporal_re <- FALSE  # TRUE: AR(1) temporal RE model, FALSE: no temporal RE model
model_tag <- ifelse(use_temporal_re, "withTimeRE", "noTimeRE")
run_suffix <- paste0(date_suffix, "_stand_", model_tag)

input_data <- read_csv(data_file)


# --- 1. Set-up input data variables ---
# Ensure year_month is ordered and create block/time indices
input_data <- input_data %>%
     mutate(year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d")) %>%
     relocate(year_month_date, .after = year_month) %>%
     select(!c(CMF, CP, AREA))
 
block_levels <- sort(unique(input_data$manzana)) # 1337
time_levels <- sort(unique(input_data$year_month_date)) # 45

df <- input_data %>%
     mutate(
          block = match(manzana, block_levels),
          time = match(year_month_date, time_levels)
     ) %>%
     arrange(block, time) %>%
     rename(
          N_HH = Inspected_houses,
          C_bt = cases, 
          y_bt = Houses_pos_IS
     ) %>%
     mutate(
          N_HH = as.integer(N_HH),
          C_bt = as.integer(C_bt),
          y_bt = as.integer(y_bt)
     )

# --- Subset to first 100 unique blocks ---
selected_blocks <- sort(unique(df$block))[1:100]
df <- df %>% filter(block %in% selected_blocks)
B <- length(selected_blocks)

# Recalculate time_levels and time indices for the subset
time_levels <- sort(unique(df$year_month_date))
T <- length(time_levels)
df <- df %>%
    mutate(time = match(year_month_date, time_levels))

# Covariates for distributed lags
# lag_vars <- c("avg_temp", "rel_hum", "total_precip", "WS2M", "mean_ndmi", "mean_ndwi", "mean_ndvi")
lag_vars <- c("avg_temp", "rel_hum", "total_precip", "mean_ndvi") # removed NDMI, NDWI and WS

df <- df %>%
     mutate(across(all_of(lag_vars), ~coalesce(., 0)))

# Distributed lag setup
K <- length(lag_vars)
L <- 1  # maximum lag
Lp1 <- L + 1
X_lag <- array(0, dim = c(nrow(df), K, Lp1))
kappa <- 2

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
unlagged_vars <- c("is_urban", "has_aljibes", "is_WI", "is_WUI", "water_shortage", "WS2M")
binary_unlagged_vars <- c("is_urban", "has_aljibes", "is_WI", "is_WUI", "water_shortage")
continuous_unlagged_vars <- setdiff(unlagged_vars, binary_unlagged_vars)
df <- df %>%
  mutate(across(all_of(unlagged_vars), ~coalesce(., 0)))
X_unlagged <- as.matrix(df[, unlagged_vars])
Ku <- ncol(X_unlagged)


# --- Standardize numeric covariates ---
# Store means and SDs for later back-transformation
X_lag_flat_means <- colMeans(X_lag_flat)
X_lag_flat_sds <- apply(X_lag_flat, 2, sd)
# X_lag_flat_sds[X_lag_flat_sds == 0 | is.na(X_lag_flat_sds)] <- 1
X_lag_flat_std <- scale(X_lag_flat, center = X_lag_flat_means, scale = X_lag_flat_sds)

X_unlagged_means <- colMeans(X_unlagged)
X_unlagged_sds <- apply(X_unlagged, 2, sd)
# X_unlagged_sds[X_unlagged_sds == 0 | is.na(X_unlagged_sds)] <- 1
X_unlagged_std <- X_unlagged

# Keep binary indicators on their original 0/1 scale; standardize only continuous columns
if (length(continuous_unlagged_vars) > 0) {
  cont_idx <- match(continuous_unlagged_vars, colnames(X_unlagged))
  X_unlagged_std[, cont_idx] <- scale(
    X_unlagged[, cont_idx, drop = FALSE],
    center = X_unlagged_means[cont_idx],
    scale = X_unlagged_sds[cont_idx]
  )
}

# # Safety check: replace any remaining non-finite values after scaling
# X_lag_flat_std[!is.finite(X_lag_flat_std)] <- 0
# X_unlagged_std[!is.finite(X_unlagged_std)] <- 0

# Pass standardized versions to Stan
stan_data <- list(
     N = nrow(df),
     y = df$y_bt,
     N_HH = df$N_HH,
     K = K,
     Lp1 = Lp1,
     X_lag_flat = X_lag_flat_std,        # ← Use standardized
     Ku = Ku,
     X_unlagged = X_unlagged_std,        # ← Use standardized
     B = B,
     T = T,
     block = df$block,
     time = df$time,
     C_bt = df$C_bt,
     n_bt = as.integer(df$N_HH + kappa * df$C_bt)
)

# Save means/SDs for interpretation later
cat("Covariate means and SDs saved for back-transformation\n")



# # --- 2. Prepare data for Stan ---
# stan_data <- list(
#      N = nrow(df),
#      y = df$y_bt,           # mosquito findings
#      N_HH = df$N_HH,        # universe
#      K = K,                 # number of lagged env covariates
#      Lp1 = Lp1,             # total number of lag terms including lag 0 (3 for max_lag=2)
#      X_lag_flat = X_lag_flat,  # flattened lagged covariates matrix [N, K*Lp1]
#      Ku = Ku,               # number of unlagged block-level covariates
#      X_unlagged = X_unlagged,  # unlagged covariates matrix [N, Ku]
#      B = B,                 # number of manzanas (now 100)
#      T = T,                 # number of time steps
#      block = df$block,      # numeric block indices
#      time = df$time,        # numeric time indices
#      C_bt = df$C_bt,        # dengue cases
#      n_bt = as.integer(df$N_HH + kappa * df$C_bt)  # total inspections, kappa fixed
#      # kappa will be estimated as a parameter in Stan
# )

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
stan_file <- if (use_temporal_re) {
  "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/Entomo_model/hierarchical_state_space.stan"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/Entomo_model/hierarchical_state_space_no_time_re.stan"
}
cat("Model variant:", ifelse(use_temporal_re, "with temporal RE", "without temporal RE"), "\n")

# Provide reasonable initial values to help both chains start properly
init_fun <- function() {
  init_vals <- list(
    alpha = rnorm(1, -4.5, 0.4),              # Tighter SD to avoid extreme p_bt
    w = matrix(rnorm(stan_data$K * stan_data$Lp1, 0, 0.08), stan_data$K, stan_data$Lp1),  # Slightly reduced SD
    sigma_w = runif(stan_data$K, 0.1, 0.3),   # Random walk SD for lag smoothing
    w_unlagged = rnorm(stan_data$Ku, 0, 0.1),
    u_block_raw = rnorm(stan_data$B, 0, 0.5),
    sigma_u = runif(1, 0.1, 0.5),
    delta0 = rnorm(1, 0.3, 0.2),
    delta1 = rnorm(1, 0, 0.1)
  )

  if (use_temporal_re) {
    init_vals$v_time_raw <- rnorm(stan_data$T, 0, 0.5)
    init_vals$sigma_v <- runif(1, 0.1, 0.5)
    init_vals$rho <- rnorm(1, 0, 0.25)
  }

  init_vals
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
  iter_warmup = 1000,
  iter_sampling = 1000,
  thin = 2,  # Keep every 2nd sample (reduces memory)
  init = init_fun,
  adapt_delta = 0.95,
  max_treedepth = 12,
  parallel_chains = if (hostname == "frietjes") 2 else 1  # Parallel on frietjes, sequential on local for memory safety
)

# Save fit object with today's date and run suffix (use save_object for cmdstanr)
fit$save_object(file.path(output_dir, paste0("fit_", run_suffix, ".rds")))

# --- 4. Summarize results ---
summary_vars <- c("alpha", "sigma_u", "delta0", "delta1", "w")
if (use_temporal_re) {
  summary_vars <- c(summary_vars, "sigma_v", "rho")
}
summary_output <- capture.output(print(fit$summary(variables = summary_vars)))
cat(summary_output, sep = "\n")
writeLines(summary_output, file.path(output_dir, paste0("model_summary_", run_suffix, ".txt")))

# --- 5. Extract fitted latent mosquito probabilities ---
# Note: Generated quantities are stored as indexed variables (e.g., "p_bt_out[1]", "p_bt_out[2]", ...)
# Extract as matrix to access indexed variables directly
post_matrix <- fit$draws(format = "matrix")  # [total_draws, variables]

cat("\nDimensions of posterior matrix (total_draws, variables):", dim(post_matrix), "\n")
var_names <- colnames(post_matrix)

# Extract fitted probabilities using regex matching for indexed variables
p_bt_cols <- grep("^p_bt_out\\[", var_names)
p_R_cols <- grep("^p_R_out\\[", var_names)
u_block_cols <- grep("^u_block_out\\[", var_names)
v_time_cols <- grep("^v_time_out\\[", var_names)
y_pred_cols <- grep("^y_pred\\[", var_names)

if (length(p_bt_cols) > 0) {
  fitted_p_bt <- colMeans(post_matrix[, p_bt_cols])
  cat("\n✓ Successfully extracted p_bt_out (N =", length(fitted_p_bt), ")\n")
} else {
  cat("\n✗ Warning: p_bt_out not found in posterior\n")
  fitted_p_bt <- rep(NA, nrow(df))
}

if (length(p_R_cols) > 0) {
  fitted_p_R <- colMeans(post_matrix[, p_R_cols])
  cat("✓ Successfully extracted p_R_out (N =", length(fitted_p_R), ")\n")
} else {
  cat("✗ Warning: p_R_out not found in posterior\n")
  fitted_p_R <- rep(NA, nrow(df))
}

if (length(u_block_cols) > 0) {
  u_post <- colMeans(post_matrix[, u_block_cols])
  cat("✓ Successfully extracted u_block_out (n =", length(u_post), " blocks)\n")
} else {
  cat("✗ Warning: u_block_out not found in posterior\n")
  u_post <- NA
}

if (length(v_time_cols) > 0) {
  v_post <- colMeans(post_matrix[, v_time_cols])
  cat("✓ Successfully extracted v_time_out (T =", length(v_post), " time periods)\n")
} else {
  cat("✗ Warning: v_time_out not found in posterior\n")
  v_post <- NA
}

if (length(y_pred_cols) > 0) {
  y_pred <- colMeans(post_matrix[, y_pred_cols])
  cat("✓ Successfully extracted y_pred (N =", length(y_pred), ")\n")
} else {
  cat("✗ Warning: y_pred not found in posterior\n")
  y_pred <- NA
}
# df$fitted_p_bt <- fitted_p_bt
# df$fitted_p_R <- fitted_p_R

# --- 6-8. Plots (temporarily disabled until variable names are verified) ---
# Uncomment after verifying Stan output variable names
# 
# # --- 6. Plot fitted vs observed proportions ---
# df$observed_prop <- df$y_bt / df$N_HH
# p1 <- ggplot(df, aes(x = observed_prop, y = fitted_p_bt)) + ...

# --- 7-9. Plots and diagnostics ---
# These will only execute if posterior extraction was successful

if (!all(is.na(fitted_p_bt))) {
  
  png(file.path(output_dir, paste0("random_effects_", run_suffix, ".png")), 
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
  if (!all(is.na(v_post))) {
    plot(v_post, type = 'b', main = "Temporal Random Effects (v_t) with AR(1)", 
         xlab = "Time", ylab = "Effect", col = "red", pch = 19)
    abline(h = 0, lty = 2, col = "gray")
    
    # Temporal effects: ACF plot
    acf(v_post, main = "ACF of Temporal Effects", col = "darkred")
  } else {
    plot.new()
    text(0.5, 0.5, "Temporal RE disabled\n(no v_time_out in model)", cex = 1.0)
    plot.new()
    text(0.5, 0.5, "ACF unavailable\n(temporal RE disabled)", cex = 1.0)
  }
  
  par(mfrow=c(1,1))
  dev.off()
  cat("Random effects plot saved.\n")
  
  # --- Posterior predictive check ---
  if (!all(is.na(y_pred))) {
    p3 <- ggplot(data.frame(observed = df$y_bt, predicted = y_pred), 
           aes(x = observed, y = predicted)) +
      geom_point(alpha=0.5) +
      geom_abline(slope=1, intercept=0, color='red') +
      labs(x="Observed y_bt", y="Predicted y_bt (posterior mean)",
           title="Posterior Predictive Check") +
      theme_minimal()
    ggsave(file.path(output_dir, paste0("posterior_predictive_check_", run_suffix, ".png")), 
           p3, width = 8, height = 6)
    print(p3)
    cat("Posterior predictive check plot saved.\n")
  }
  
  # --- Trace plots for MCMC chains ---
  if (requireNamespace("bayesplot", quietly = TRUE)) {
       library(bayesplot)
       # Extract draws as array for bayesplot
       draws_array <- fit$draws(format = "array")
       
       # Trace plot 1: Main hyperparameters
       trace_params <- c("alpha", "sigma_u", "delta0", "delta1")
       if (use_temporal_re) {
         trace_params <- c(trace_params, "sigma_v", "rho")
       }
       trace_plot <- mcmc_trace(draws_array, pars = trace_params)
       trace_file <- file.path(output_dir, paste0("traceplot_params_", run_suffix, ".png"))
       ggsave(trace_file, trace_plot, width = 10, height = 8)
       cat("Trace plot saved to:", trace_file, "\n")
       
       # Trace plot 2: Lagged covariate weights (w)
       w_params <- grep("^w\\[", dimnames(draws_array)[[3]], value = TRUE)
       if (length(w_params) > 0) {
         trace_plot_w <- mcmc_trace(draws_array, pars = w_params)
         trace_file_w <- file.path(output_dir, paste0("traceplot_weights_w_", run_suffix, ".png"))
         ggsave(trace_file_w, trace_plot_w, width = 12, height = 10)
         cat("Weight trace plot saved to:", trace_file_w, "\n")
       }
       
       # Trace plot 3: Unlagged covariate weights (w_unlagged)
       w_unlagged_params <- grep("^w_unlagged\\[", dimnames(draws_array)[[3]], value = TRUE)
       if (length(w_unlagged_params) > 0) {
         trace_plot_wu <- mcmc_trace(draws_array, pars = w_unlagged_params)
         trace_file_wu <- file.path(output_dir, paste0("traceplot_weights_unlagged_", run_suffix, ".png"))
         ggsave(trace_file_wu, trace_plot_wu, width = 12, height = 8)
         cat("Unlagged weight trace plot saved to:", trace_file_wu, "\n")
       }
  } else {
       cat("bayesplot package not installed; skipping trace plot.\n")
  }
  
} else {
  cat("\nNote: Posterior extraction did not complete; skipping plots.\n")
}

cat("\nAll outputs saved to:", output_dir, "\n")

