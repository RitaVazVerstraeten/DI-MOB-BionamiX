# =====================================================
# Mosquito occupancy model with reactive surveillance
# Corrected R script — integrates lessons from:
#   - GLMM diagnostics (priors, covariates, reactive correction)
#   - Hierarchical state-space model (GP, AR(1), overdispersion)
#   - Block-level time series analysis (intermittent occupancy structure)
# =====================================================

library(cmdstanr)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(sf)
library(lubridate)
library(posterior)
library(bayesplot)

# =========================
# 1) SETTINGS
# =========================
hostname <- Sys.info()["nodename"]

cfg <- list(
  # Paths
  data_dir  = if (hostname == "frietjes") "~/data/Entomo"
              else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/",
  data_file_name = "env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv",
  output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan_occupancy",
  stan_file  = NULL,   # set below after use_hsgp is resolved
  shapefile_path = if (hostname == "frietjes")
    "/home/rita/data/Entomo/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
  else
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp",
  sf_block_col = "CODIGO_",

  # Data prep
  n_blocks  = 100,            # NULL = all blocks; use 100 for testing
  lag_vars  = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
  max_lag   = 1,              # lags 0 and 1 → Lp1 = 2
  kappa     = 2,
  unlagged_vars = c("is_urban", "is_WUI"),
  numeric_vars  = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),

  # phi: fixed from empirical overdispersion estimate
  # Estimated from zero-case cells: phi_pooled = (n_rep - disp_ratio) / (disp_ratio - 1)
  # 100-block subset gave phi ~ 23; re-estimate when changing block set
  phi_fixed = 23.0,

  # GP approximation
  use_hsgp   = TRUE,   # TRUE = HSGP (O(B*M^2)); FALSE = exact Cholesky (O(B^3))
  hsgp_m     = 20,     # basis functions per spatial dimension
  hsgp_c     = 1.5,    # boundary expansion factor

  # MCMC
  chains          = 2,
  iter_warmup     = 500,
  iter_sampling   = 500,
  adapt_delta     = 0.95,
  max_treedepth   = 12,
  parallel_chains = if (hostname == "frietjes") 2 else 1
)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)
stan_dir <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo"
cfg$stan_file <- if (isTRUE(cfg$use_hsgp)) {
  file.path(stan_dir, "occupancy_block_model_hsgp.stan")
} else {
  file.path(stan_dir, "occupancy_block_model.stan")
}

gp_type_suffix <- ifelse(isTRUE(cfg$use_hsgp), "HSGP", "GP")
cat(sprintf("GP type: %s | Stan: %s\n",
            gp_type_suffix, basename(cfg$stan_file)))

date_suffix <- format(Sys.Date(), "%Y%m%d")
run_suffix  <- paste0(date_suffix, "_occupancy")

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 2) LOAD AND PREPARE DATA
# =========================
df_raw <- read_csv(cfg$data_file, show_col_types = FALSE) %>%
  mutate(
    year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d"),
    block           = factor(manzana),
    n_bt            = Inspected_houses + cfg$kappa * cases,
    y_bt            = Houses_pos_IS,
    C_bt            = cases
  )

# Optionally subset blocks for testing
if (!is.null(cfg$n_blocks)) {
  set.seed(42)
  selected_blocks <- sample(unique(df_raw$manzana), cfg$n_blocks)
  df_raw <- df_raw %>% filter(manzana %in% selected_blocks)
  cat("Subset to", cfg$n_blocks, "blocks\n")
}

# Standardise numeric covariates (z-score)
for (var in cfg$numeric_vars) {
  if (var %in% names(df_raw) && is.numeric(df_raw[[var]])) {
    m <- mean(df_raw[[var]], na.rm = TRUE)
    s <- sd(df_raw[[var]], na.rm = TRUE)
    if (!is.na(s) && s > 0) df_raw[[var]] <- (df_raw[[var]] - m) / s
  }
}

# Create lagged covariates
L <- cfg$max_lag
for (var in cfg$lag_vars) {
  for (l in 0:L) {
    lag_col <- paste0(var, "_lag", l)
    df_raw <- df_raw %>%
      group_by(manzana) %>%
      arrange(year_month_date, .by_group = TRUE) %>%
      mutate(!!lag_col := lag(.data[[var]], n = l, default = NA_real_)) %>%
      ungroup()
  }
}

# Drop rows with NA in any lag column (first L rows per block)
lagged_cols   <- unlist(lapply(cfg$lag_vars, function(v) paste0(v, "_lag", 0:L)))
df <- df_raw %>%
  filter(if_all(all_of(lagged_cols), ~!is.na(.))) %>%
  filter(n_bt > 0)   # keep only observed block-months

cat("Rows after lag creation and n_bt > 0 filter:", nrow(df), "\n")

# Block and time indices
block_levels <- sort(unique(as.character(df$manzana)))
time_levels  <- sort(unique(df$year_month_date))
B <- length(block_levels)
T <- length(time_levels)

df <- df %>%
  mutate(
    block_idx = match(as.character(manzana), block_levels),
    time_idx  = match(year_month_date, time_levels)
  )

cat("B =", B, "| T =", T, "| N =", nrow(df), "\n")

# =========================
# 3) BUILD COVARIATE MATRICES
# =========================
Lp1 <- L + 1
K   <- length(cfg$lag_vars)
Ku  <- length(cfg$unlagged_vars)

# Flattened lag matrix [N, K*Lp1] — column order must match Stan's to_vector(w[K, Lp1])
# Stan's to_vector on matrix[K, Lp1] goes column-major: lag0_cov1, lag0_cov2, ..., lag1_cov1, ...
# So we order columns as: for each lag l, all K covariates
X_lag_list <- vector("list", Lp1)
for (l in 0:L) {
  cols <- paste0(cfg$lag_vars, "_lag", l)
  X_lag_list[[l + 1]] <- as.matrix(df[, cols])
}
# Bind in lag-major order to match column-major to_vector(w[K, Lp1])
# w[k, l] maps to column (l-1)*K + k in the flat matrix
X_lag_flat <- do.call(cbind, X_lag_list)
stopifnot(ncol(X_lag_flat) == K * Lp1)

# Unlagged covariate matrix [N, Ku]
X_unlagged <- as.matrix(df[, cfg$unlagged_vars])
# Convert logical columns to numeric
for (j in seq_len(ncol(X_unlagged))) {
  if (is.logical(X_unlagged[, j])) X_unlagged[, j] <- as.numeric(X_unlagged[, j])
}

# =========================
# 4) SPATIAL DISTANCE MATRIX
# =========================
sf_blocks <- st_read(cfg$shapefile_path, quiet = TRUE)
pts       <- suppressWarnings(st_point_on_surface(sf_blocks))
if (st_is_longlat(pts)) pts <- st_transform(pts, 3857)

coords_df <- sf_blocks %>%
  st_drop_geometry() %>%
  mutate(
    block_chr = as.character(.data[[cfg$sf_block_col]]),
    x = st_coordinates(pts)[, 1],
    y = st_coordinates(pts)[, 2]
  ) %>%
  filter(block_chr %in% block_levels) %>%
  arrange(match(block_chr, block_levels)) %>%
  distinct(block_chr, .keep_all = TRUE)

# Confirm alignment
stopifnot(all(coords_df$block_chr == block_levels))

dist_mat <- as.matrix(dist(coords_df[, c("x", "y")]))
cat(sprintf("Distance matrix: %d × %d blocks\n", nrow(dist_mat), ncol(dist_mat)))

# =========================
# 5) ESTIMATE phi FROM DATA
# =========================
# Empirical overdispersion from zero-case cells (baseline ecological overdispersion)
# Re-estimate each time block set changes
zero_case <- df %>% filter(C_bt == 0, n_bt > 0) %>%
  mutate(y_rate = y_bt / n_bt)

p_bar_disp  <- mean(zero_case$y_rate)
var_obs_disp <- var(zero_case$y_rate)
n_rep_disp  <- median(zero_case$n_bt)
var_binom_disp <- p_bar_disp * (1 - p_bar_disp) / n_rep_disp
disp_ratio  <- var_obs_disp / var_binom_disp
phi_empirical <- (n_rep_disp - disp_ratio) / (disp_ratio - 1)

cat(sprintf(
  "Overdispersion diagnostics:\n  n zero-case cells: %d\n  median n_bt: %.0f\n  dispersion ratio: %.2f\n  implied phi: %.1f\n",
  nrow(zero_case), n_rep_disp, disp_ratio, phi_empirical
))

# Use empirical phi if reasonable; otherwise fall back to cfg$phi_fixed
phi_use <- if (is.finite(phi_empirical) && phi_empirical > 0 && phi_empirical < 500) {
  round(phi_empirical, 1)
} else {
  cfg$phi_fixed
}
cat("Using phi =", phi_use, "\n")

# =========================
# 6) ASSEMBLE STAN DATA
# =========================
stan_data <- list(
  N          = nrow(df),
  B          = B,
  T          = T,
  K          = K,
  Lp1        = Lp1,
  Ku         = Ku,
  y          = as.integer(df$y_bt),
  n_bt       = as.integer(df$n_bt),
  C_bt       = as.integer(df$C_bt),
  block      = as.integer(df$block_idx),
  time       = as.integer(df$time_idx),
  X_lag_flat = X_lag_flat,
  X_unlagged = X_unlagged,
  phi        = phi_use
)

if (isTRUE(cfg$use_hsgp)) {
  coords_mat <- as.matrix(coords_df[, c("x", "y")])
  stan_data$coords_block <- coords_mat
  stan_data$M            <- cfg$hsgp_m
  stan_data$c_boundary   <- cfg$hsgp_c
  cat(sprintf("HSGP: M=%d, c=%.1f, M_total=%d basis functions\n",
              cfg$hsgp_m, cfg$hsgp_c, cfg$hsgp_m^2))
} else {
  stan_data$dist_block <- dist_mat
  cat(sprintf("Exact GP: %d x %d distance matrix\n", B, B))
}

# Sanity checks
stopifnot(all(stan_data$y <= stan_data$n_bt))
stopifnot(all(stan_data$block >= 1 & stan_data$block <= B))
stopifnot(all(stan_data$time  >= 1 & stan_data$time  <= T))
cat("Stan data assembled and validated\n")

# =========================
# 7) INITIAL VALUES
# =========================
# Informed initialisations reduce warmup time substantially
# Values based on GLMM estimates and previous Stan posteriors
make_init <- function(stan_data, use_hsgp = FALSE) {
  B   <- stan_data$B
  T   <- stan_data$T
  K   <- stan_data$K
  Lp1 <- stan_data$Lp1
  Ku  <- stan_data$Ku

  function() {
    init_vals <- list(
      # Intercepts: target ~1% equilibrium occupancy
      alpha_gamma = -5.0,
      alpha_phi   = -1.0,
      theta       = 1.5,

      # Lag weights near GLMM estimates for lag-0
      w_gamma      = matrix(c(-0.15, -0.15,  0.10, -0.15,
                               0.10,  0.15, -0.05, -0.10),
                             nrow = K, ncol = Lp1),
      w_phi        = matrix(0, nrow = K, ncol = Lp1),
      sigma_w_gamma = rep(0.2, K),
      sigma_w_phi   = rep(0.2, K),

      w_unlagged_gamma = rep(0, Ku),
      w_unlagged_phi   = rep(0, Ku),

      # Spatial GP
      sigma_gp = 0.28,
      rho_gp   = 200,

      # Global AR(1)
      v_global_raw = rep(0, T),
      sigma_global = 0.3,
      rho          = 0.4,

      # Per-block deviations
      v_block_raw  = matrix(0, nrow = B, ncol = T),
      sigma_block  = 0.1,

      # Reactive surveillance
      delta0 = 2.0,
      delta1 = 0.0
    )

    if (isTRUE(use_hsgp)) {
      init_vals$beta_gp <- rnorm(stan_data$M^2, 0, 0.1)
    } else {
      init_vals$z_gp <- rep(0, B)
    }

    init_vals
  }
}

# =========================
# 8) FIT MODEL
# =========================
mod <- cmdstan_model(cfg$stan_file)

fit <- mod$sample(
  data            = stan_data,
  chains          = cfg$chains,
  iter_warmup     = cfg$iter_warmup,
  iter_sampling   = cfg$iter_sampling,
  init            = make_init(stan_data, use_hsgp = isTRUE(cfg$use_hsgp)),
  adapt_delta     = cfg$adapt_delta,
  max_treedepth   = cfg$max_treedepth,
  parallel_chains = cfg$parallel_chains
)

fit$save_object(file.path(cfg$output_dir, paste0("fit_", run_suffix, ".rds")))

# =========================
# 9) SUMMARISE
# =========================
summary_vars <- c(
  "alpha_gamma", "alpha_phi", "theta",
  "sigma_gp", "rho_gp",
  "sigma_global", "sigma_block", "rho",
  "delta0", "delta1"
)

fit_summary <- fit$summary(variables = summary_vars)
print(fit_summary)
writeLines(
  capture.output(print(fit_summary)),
  file.path(cfg$output_dir, paste0("model_summary_", run_suffix, ".txt"))
)

# Flag convergence issues
bad_rhat <- fit_summary %>% filter(rhat > 1.05)
if (nrow(bad_rhat) > 0) {
  cat("\nWARNING: R-hat > 1.05 for:", paste(bad_rhat$variable, collapse = ", "), "\n")
} else {
  cat("\nAll key parameters converged (R-hat <= 1.05)\n")
}

low_ess <- fit_summary %>% filter(ess_bulk < 100)
if (nrow(low_ess) > 0) {
  cat("WARNING: ESS < 100 for:", paste(low_ess$variable, collapse = ", "), "\n")
}

# Divergences
diag <- fit$diagnostic_summary()
cat("Divergences:", sum(diag$num_divergent), "\n")
cat("Max treedepth hits:", sum(diag$num_max_treedepth), "\n")

# =========================
# 10) EXTRACT POSTERIORS
# =========================
draws       <- fit$draws(format = "df")
p_bt_draws  <- fit$draws("p_bt_out",  format = "matrix")
y_pred_draws <- fit$draws("y_pred",   format = "matrix")

df$p_bt_mean     <- colMeans(p_bt_draws)
df$y_pred_mean   <- colMeans(y_pred_draws)
df$p_bt_q05      <- apply(p_bt_draws, 2, quantile, 0.05)
df$p_bt_q95      <- apply(p_bt_draws, 2, quantile, 0.95)
df$observed_rate <- df$y_bt / df$n_bt

# =========================
# 11) PLOTS
# =========================

# --- Time series: aggregate ---
ts_agg <- df %>%
  group_by(year_month_date) %>%
  summarise(
    obs_mean    = mean(observed_rate, na.rm = TRUE),
    fit_mean    = mean(p_bt_mean,     na.rm = TRUE),
    fit_q05     = mean(p_bt_q05,      na.rm = TRUE),
    fit_q95     = mean(p_bt_q95,      na.rm = TRUE),
    .groups = "drop"
  )

p_ts_agg <- ggplot(ts_agg, aes(x = year_month_date)) +
  geom_ribbon(aes(ymin = fit_q05, ymax = fit_q95), fill = "#378ADD", alpha = 0.25) +
  geom_line(aes(y = fit_mean,  colour = "Fitted p_bt"),    linewidth = 0.9) +
  geom_point(aes(y = fit_mean, colour = "Fitted p_bt"),    size = 1.5) +
  geom_line(aes(y = obs_mean,  colour = "Observed rate"),  linewidth = 0.7) +
  geom_point(aes(y = obs_mean, colour = "Observed rate"),  size = 1.5) +
  scale_colour_manual(values = c("Fitted p_bt" = "#185FA5", "Observed rate" = "#D85A30")) +
  labs(
    title    = "Time series: observed vs fitted occupancy probability (mean across blocks)",
    subtitle = paste0("phi = ", phi_use, " | B = ", B, " | ", cfg$iter_sampling, " post-warmup samples"),
    x = "Time", y = "Probability", colour = NULL
  ) +
  theme_minimal()

ggsave(
  file.path(cfg$output_dir, paste0("timeseries_aggregate_", run_suffix, ".png")),
  p_ts_agg, width = 12, height = 5, dpi = 150
)

# --- Time series: by block (random sample) ---
sample_blocks <- sample(unique(df$block_idx), min(9, B))

p_ts_block <- df %>%
  filter(block_idx %in% sample_blocks) %>%
  ggplot(aes(x = year_month_date)) +
  geom_ribbon(aes(ymin = p_bt_q05, ymax = p_bt_q95), fill = "#378ADD", alpha = 0.2) +
  geom_line(aes(y = p_bt_mean,     colour = "Fitted"),   linewidth = 0.7) +
  geom_line(aes(y = observed_rate, colour = "Observed"), linewidth = 0.6) +
  geom_point(aes(y = observed_rate, colour = "Observed"), size = 1) +
  scale_colour_manual(values = c("Fitted" = "#185FA5", "Observed" = "#D85A30")) +
  facet_wrap(~block_idx, scales = "free_y") +
  labs(
    title = "Time series by block: observed vs fitted occupancy probability",
    x = "Time", y = "Probability", colour = NULL
  ) +
  theme_minimal(base_size = 10)

ggsave(
  file.path(cfg$output_dir, paste0("timeseries_by_block_", run_suffix, ".png")),
  p_ts_block, width = 14, height = 8, dpi = 150
)

# --- Posterior predictive check ---
obs_rate  <- df$observed_rate
pred_rate <- df$y_pred_mean / df$n_bt

p_ppc <- ggplot(data.frame(obs = obs_rate, pred = pred_rate), aes(x = pred, y = obs)) +
  geom_point(alpha = 0.3, size = 0.8, colour = "#185FA5") +
  geom_abline(slope = 1, intercept = 0, colour = "red", linetype = "dashed") +
  scale_x_continuous(limits = c(0, max(obs_rate, pred_rate, na.rm = TRUE))) +
  scale_y_continuous(limits = c(0, max(obs_rate, pred_rate, na.rm = TRUE))) +
  labs(
    title    = "Posterior predictive check",
    subtitle = "Points should scatter symmetrically around the red 1:1 line",
    x = "Predicted rate (y_pred / n_bt)", y = "Observed rate (y_bt / n_bt)"
  ) +
  theme_minimal()

ggsave(
  file.path(cfg$output_dir, paste0("ppc_", run_suffix, ".png")),
  p_ppc, width = 7, height = 6, dpi = 150
)

# --- Occupancy diagnostics: zero vs positive months ---
p_zero <- df %>%
  mutate(positive = y_bt > 0) %>%
  group_by(year_month_date) %>%
  summarise(
    pct_positive_obs  = mean(positive),
    mean_p_bt_fitted  = mean(p_bt_mean),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = year_month_date)) +
  geom_line(aes(y = pct_positive_obs,  colour = "Fraction blocks with y>0"), linewidth = 0.8) +
  geom_line(aes(y = mean_p_bt_fitted,  colour = "Mean fitted p_bt"),          linewidth = 0.8) +
  scale_colour_manual(values = c(
    "Fraction blocks with y>0" = "#D85A30",
    "Mean fitted p_bt"         = "#185FA5"
  )) +
  labs(
    title    = "Occupancy: fraction of blocks with observed positives vs mean fitted p_bt",
    subtitle = "If model is well-specified, lines should track each other",
    x = "Time", y = "Proportion / Probability", colour = NULL
  ) +
  theme_minimal()

ggsave(
  file.path(cfg$output_dir, paste0("occupancy_diagnostic_", run_suffix, ".png")),
  p_zero, width = 12, height = 5, dpi = 150
)

# --- Traceplots ---
draws_array <- fit$draws(variables = summary_vars, format = "array")
p_trace <- mcmc_trace(draws_array, pars = summary_vars)
ggsave(
  file.path(cfg$output_dir, paste0("traceplots_", run_suffix, ".png")),
  p_trace, width = 14, height = 10, dpi = 150
)

cat("\nAll outputs saved to:", cfg$output_dir, "\n")
