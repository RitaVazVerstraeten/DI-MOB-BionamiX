# =====================================================
# Fully Hidden Markov Process (HMP) Occupancy Model
#
# Two hidden layers:
#   Z_bt  — ecological colonisation state (HMM, Markov chain)
#   D_hbt — detection of larvae per visited household (Binomial given Z)
#
# Intervention tracking:
#   theta reduces persistence when a household was visited AND found positive
#   (both systematic and reactive visits count — under perfect detection, visited + occupied = removed)
#   phi_eff = inv_logit(logit(phi_bt) - theta), weighted by q_{b,t-1} in the recursion
#
# Key differences vs occupancy_block_model_hsgp.stan:
#   - Proper HMM forward algorithm (marginalises over Z_{b,1:T})
#   - Separate detection probability p_det (not conflated with occupancy)
#   - Binomial likelihood (overdispersion handled by latent state)
#   - Complete B×T grid required for forward algorithm
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
library(patchwork)

# =========================
# 1) SETTINGS
# =========================
hostname <- Sys.info()["nodename"]

cfg <- list(
  # Paths
  data_dir  = if (hostname == "frietjes") "~/data/Entomo"
              else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/",
  data_file_name = "env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv",
  base_output_dir = if (hostname == "frietjes")
    "~/data/Entomo/fitting/stan_occupancy_hmp"
  else
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan_occupancy_hmp",
  # Random effects configuration
  # use_hsgp = TRUE  → HSGP spatial GP + block RE + global AR(1)  [occupancy_hmp.stan]
  # use_hsgp = FALSE → block RE + global AR(1) only               [occupancy_ARglobal_REblock.stan]
  use_hsgp  = TRUE,

  shapefile_path = if (hostname == "frietjes")
    "/home/rita/data/Entomo/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
  else
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp",
  sf_block_col = "CODIGO_",

  # Data prep
  n_blocks  = 100,           # NULL = all blocks; use 100 for testing
  lag_vars  = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
  max_lag   = 1,             # lags 0 and 1 → Lp1 = 2
  kappa     = 4,
  unlagged_vars = c("is_urban", "is_WUI"),
  numeric_vars  = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),

  # Fixed false-positive rate for the emission model
  epsilon = 0.001,

  # HSGP
  hsgp_m      = 20,          # basis functions per spatial dimension
  hsgp_c      = 1.5,         # boundary expansion factor

  # MCMC
  chains          = 2,
  iter_warmup     = 200,
  iter_sampling   = 200,
  adapt_delta     = 0.95,
  max_treedepth   = 12,
  parallel_chains = if (hostname == "frietjes") 2 else 1
)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)

src_dir <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo"
cfg$stan_file <- ifelse(isTRUE(cfg$use_hsgp),
  file.path(src_dir, "occupancy_hmp.stan"),
  file.path(src_dir, "occupancy_ARglobal_REblock.stan"))

b_label     <- if (is.null(cfg$n_blocks)) "B_All" else paste0("B", cfg$n_blocks)
date_suffix <- format(Sys.Date(), "%Y%m%d")
re_label    <- if (isTRUE(cfg$use_hsgp)) paste0("HSGP_M", cfg$hsgp_m) else "REblock"
run_label   <- paste0("MeanField_", re_label, "_", b_label)
run_suffix  <- paste0(date_suffix, "_", run_label)
cfg$output_dir <- file.path(cfg$base_output_dir, run_label)
dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("Stan: %s\n", basename(cfg$stan_file)))
cat(sprintf("Output dir: %s\n", cfg$output_dir))

# =========================
# 2) LOAD AND PREPARE DATA
# =========================
df_raw <- read_csv(cfg$data_file, show_col_types = FALSE) %>%
  mutate(
    year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d"),
    block           = factor(manzana),
    # n_bt includes the kappa multiplier on reactive inspections
    n_bt            = Inspected_houses + cfg$kappa * cases,
    y_bt            = Houses_pos_IS,
    C_bt            = cases
  )

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

# Create lagged covariates per block
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

lagged_cols <- unlist(lapply(cfg$lag_vars, function(v) paste0(v, "_lag", 0:L)))

# Block and time indices (computed on the FULL data, before any n_bt filter)
block_levels <- sort(unique(as.character(df_raw$manzana)))
time_levels  <- sort(unique(df_raw$year_month_date))
B <- length(block_levels)
T <- length(time_levels)

df_raw <- df_raw %>%
  mutate(
    block_idx = match(as.character(manzana), block_levels),
    time_idx  = match(year_month_date, time_levels)
  )

# Drop rows where any lag column is NA (first L rows per block)
# These rows are excluded from the covariate matrices (N-indexed)
df_obs <- df_raw %>%
  filter(if_all(all_of(lagged_cols), ~!is.na(.))) %>%
  filter(n_bt > 0)

cat("Observed cells (n_bt > 0, lags available):", nrow(df_obs), "\n")
cat("B =", B, "| T =", T, "| N =", nrow(df_obs), "\n")

# =========================
# 3) BUILD COVARIATE MATRICES (N-indexed, observed cells only)
# =========================
Lp1 <- L + 1
K   <- length(cfg$lag_vars)
Ku  <- length(cfg$unlagged_vars)

X_lag_list <- vector("list", Lp1)
for (l in 0:L) {
  cols <- paste0(cfg$lag_vars, "_lag", l)
  X_lag_list[[l + 1]] <- as.matrix(df_obs[, cols])
}
X_lag_flat <- do.call(cbind, X_lag_list)
stopifnot(ncol(X_lag_flat) == K * Lp1)

X_unlagged <- as.matrix(df_obs[, cfg$unlagged_vars])
for (j in seq_len(ncol(X_unlagged))) {
  if (is.logical(X_unlagged[, j])) X_unlagged[, j] <- as.numeric(X_unlagged[, j])
}

# =========================
# 4) BUILD COMPLETE B×T GRID (for HMM forward algorithm)
# =========================
# All (block, time) combinations. For unobserved cells: y=0, n=0, C=0.
all_bt <- expand.grid(block_idx = 1:B, time_idx = 1:T)

df_grid <- df_raw %>%
  select(block_idx, time_idx, y_bt, n_bt, C_bt) %>%
  # Aggregate in case of duplicates
  group_by(block_idx, time_idx) %>%
  summarise(y_bt = sum(y_bt, na.rm = TRUE),
            n_bt = sum(n_bt, na.rm = TRUE),
            C_bt = sum(C_bt, na.rm = TRUE),
            .groups = "drop") %>%
  right_join(all_bt, by = c("block_idx", "time_idx")) %>%
  mutate(
    y_bt = replace_na(as.integer(y_bt), 0L),
    n_bt = replace_na(as.integer(n_bt), 0L),
    C_bt = replace_na(as.integer(C_bt), 0L)
  ) %>%
  arrange(block_idx, time_idx)

# Build B×T matrices (row = block, col = time)
y_mat <- matrix(df_grid$y_bt, nrow = B, ncol = T, byrow = FALSE)
n_mat <- matrix(df_grid$n_bt, nrow = B, ncol = T, byrow = FALSE)
C_mat <- matrix(df_grid$C_bt, nrow = B, ncol = T, byrow = FALSE)

cat(sprintf("B×T grid: %d cells | unobserved (n=0): %d\n",
            B * T, sum(n_mat == 0)))
stopifnot(all(y_mat <= n_mat))

# =========================
# 5) SPATIAL COORDINATES (HSGP only)
# =========================
if (isTRUE(cfg$use_hsgp)) {
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

  stopifnot(all(coords_df$block_chr == block_levels))
  coords_mat <- as.matrix(coords_df[, c("x", "y")])
  cat(sprintf("Coordinates: %d blocks\n", nrow(coords_mat)))
}

# =========================
# 6) ASSEMBLE STAN DATA
# =========================
stan_data <- list(
  # Dimensions
  N    = nrow(df_obs),
  B    = B,
  T    = T,
  K    = K,
  Lp1  = Lp1,
  Ku   = Ku,

  # N-indexed observed cells (for environmental covariate computation)
  y          = as.integer(df_obs$y_bt),
  n_bt       = as.integer(df_obs$n_bt),
  C_bt       = as.integer(df_obs$C_bt),
  block      = as.integer(df_obs$block_idx),
  time       = as.integer(df_obs$time_idx),
  X_lag_flat = X_lag_flat,
  X_unlagged = X_unlagged,

  # Complete B×T grid for mean-field forward recursion
  y_mat = y_mat,
  n_mat = n_mat,
  C_mat = C_mat
)

if (isTRUE(cfg$use_hsgp)) {
  stan_data$coords_block <- coords_mat
  stan_data$M            <- cfg$hsgp_m
  stan_data$c_boundary   <- cfg$hsgp_c
}

# Sanity checks
stopifnot(all(stan_data$y <= stan_data$n_bt))
stopifnot(all(stan_data$block >= 1 & stan_data$block <= B))
stopifnot(all(stan_data$time  >= 1 & stan_data$time  <= T))
cat("Stan data assembled and validated\n")

# =========================
# 7) INITIAL VALUES
# =========================
make_init <- function(stan_data, use_hsgp = TRUE) {
  B   <- stan_data$B
  T   <- stan_data$T
  K   <- stan_data$K
  Lp1 <- stan_data$Lp1
  Ku  <- stan_data$Ku

  function() {
    init <- list(
      alpha_gamma      = -5.0,
      alpha_phi        = -1.0,
      theta            = 1.5,

      alpha0           = -3.0,
      alpha1           =  2.0,
      alpha2           =  0.5,

      w_gamma          = matrix(c(-0.15, -0.15,  0.10, -0.15,
                                   0.10,  0.15, -0.05, -0.10),
                                nrow = K, ncol = Lp1),
      w_phi            = matrix(0, nrow = K, ncol = Lp1),
      sigma_w_gamma    = rep(0.2, K),
      sigma_w_phi      = rep(0.2, K),
      w_unlagged_gamma = rep(0, Ku),
      w_unlagged_phi   = rep(0, Ku),

      # Temporal
      v_global_raw = rep(0, T),
      sigma_global = 0.3,
      rho          = 0.4,

      # Block RE
      u_block_raw = rep(0, B),
      sigma_block = 0.1
    )
    if (isTRUE(use_hsgp)) {
      M <- stan_data$M
      init$beta_gp  <- rnorm(M^2, 0, 0.1)
      init$sigma_gp <- 0.28
      init$rho_gp   <- 200
    }
    init
  }
}

# =========================
# 8) MODEL SPEC + FIT
# =========================
cat(paste(rep("=", 55), collapse = ""), "\n")
cat("MODEL SPECIFICATION — Mean-Field Occupancy (slides 15-22)\n")
cat(paste(rep("=", 55), collapse = ""), "\n")
cat(sprintf("  Blocks (B):          %d\n", B))
cat(sprintf("  Time periods (T):    %d\n", T))
cat(sprintf("  Observations (N):    %d\n", nrow(df_obs)))
cat(sprintf("  Unobserved cells:    %d / %d\n", sum(n_mat == 0), B * T))
cat(sprintf("  Lag covariates (K):  %d  (L=%d)\n", K, L))
cat(sprintf("  Unlagged covars:     %d\n", Ku))
if (isTRUE(cfg$use_hsgp))
  cat(sprintf("  Spatial RE:          HSGP (M=%d per dim, M_total=%d, c=%.1f) + block offset\n",
              cfg$hsgp_m, cfg$hsgp_m^2, cfg$hsgp_c)) else
  cat("  Spatial RE:          per-block iid offset (no GP)\n")
cat("  Temporal structure:  Global AR(1) + per-block offset\n")
cat("  Latent states:       p_bt (occupancy), q_bt (visit prob), r_bt = p*q\n")
cat("  Visit chain:         alpha0 + alpha1*C_bt [+ alpha2 if revisit]\n")
cat("  Intervention:        theta (log-odds persistence reduction after visit)\n")
cat(sprintf("  Likelihood:          Binomial(n_mat[b,t], r_bt)\n"))
cat(sprintf("  MCMC:                %d chains x %d samples (warmup %d)\n",
            cfg$chains, cfg$iter_sampling, cfg$iter_warmup))
cat(sprintf("  Output dir:          %s\n", cfg$output_dir))
cat(paste(rep("=", 55), collapse = ""), "\n")

mod <- cmdstan_model(cfg$stan_file)

fit <- mod$sample(
  data            = stan_data,
  chains          = cfg$chains,
  iter_warmup     = cfg$iter_warmup,
  iter_sampling   = cfg$iter_sampling,
  init            = make_init(stan_data, use_hsgp = cfg$use_hsgp),
  adapt_delta     = cfg$adapt_delta,
  max_treedepth   = cfg$max_treedepth,
  parallel_chains = cfg$parallel_chains,
  refresh         = 10
)

fit$save_object(file.path(cfg$output_dir, paste0("fit_", run_suffix, ".rds")))

# =========================
# 9) SUMMARISE
# =========================
summary_vars <- c(
  "alpha_gamma", "alpha_phi", "theta",
  "alpha0", "alpha1", "alpha2",
  "sigma_gp", "rho_gp",
  "sigma_global", "sigma_block", "rho"
)

fit_summary <- fit$summary(variables = summary_vars)
print(fit_summary)
writeLines(
  capture.output(print(fit_summary)),
  file.path(cfg$output_dir, paste0("model_summary_", run_suffix, ".txt"))
)

bad_rhat <- fit_summary %>% filter(rhat > 1.05)
if (nrow(bad_rhat) > 0) {
  cat("\nWARNING: R-hat > 1.05 for:", paste(bad_rhat$variable, collapse = ", "), "\n")
} else {
  cat("\nAll key parameters converged (R-hat <= 1.05)\n")
}

low_ess <- fit_summary %>% filter(ess_bulk < 100)
if (nrow(low_ess) > 0)
  cat("WARNING: ESS < 100 for:", paste(low_ess$variable, collapse = ", "), "\n")

diag <- fit$diagnostic_summary()
cat("Divergences:", sum(diag$num_divergent), "\n")
cat("Max treedepth hits:", sum(diag$num_max_treedepth), "\n")

# =========================
# 10) EXTRACT POSTERIORS
# =========================
# p_out[b,t]: E[z_ibt] — block-level mosquito presence probability (p_bt)
# q_out[b,t]: E[v_ibt] — block-level visit probability (q_bt)
# r_out[b,t]: r_bt = p_bt * q_bt — detection probability (used for likelihood)
p_occ_draws  <- fit$draws("p_out",     format = "matrix")
q_occ_draws  <- fit$draws("q_out",     format = "matrix")
y_pred_draws <- fit$draws("y_pred_mat", format = "matrix")

# Flatten B×T → index mapping: Stan uses column-major (b varies fastest for matrix[B,T])
bt_idx <- function(b, t, B) (t - 1) * B + b

df_obs$p_occ_mean <- NA_real_
df_obs$p_occ_q05  <- NA_real_
df_obs$p_occ_q95  <- NA_real_
df_obs$q_occ_mean <- NA_real_
df_obs$y_pred_mean <- NA_real_
df_obs$observed_rate <- df_obs$y_bt / df_obs$n_bt

for (i in seq_len(nrow(df_obs))) {
  b <- df_obs$block_idx[i]
  t <- df_obs$time_idx[i]
  col_p    <- paste0("p_out[", b, ",", t, "]")
  col_q    <- paste0("q_out[", b, ",", t, "]")
  col_pred <- paste0("y_pred_mat[", b, ",", t, "]")
  if (col_p %in% colnames(p_occ_draws)) {
    draws_p <- p_occ_draws[, col_p]
    df_obs$p_occ_mean[i] <- mean(draws_p)
    df_obs$p_occ_q05[i]  <- quantile(draws_p, 0.05)
    df_obs$p_occ_q95[i]  <- quantile(draws_p, 0.95)
  }
  if (col_q %in% colnames(q_occ_draws)) {
    df_obs$q_occ_mean[i] <- mean(q_occ_draws[, col_q])
  }
  if (col_pred %in% colnames(y_pred_draws)) {
    df_obs$y_pred_mean[i] <- mean(y_pred_draws[, col_pred])
  }
}

# =========================
# 11) PLOTS
# =========================

# --- Aggregate time series ---
ts_agg <- df_obs %>%
  group_by(year_month_date) %>%
  summarise(
    obs_mean   = mean(observed_rate, na.rm = TRUE),
    p_occ_mean = mean(p_occ_mean,    na.rm = TRUE),
    p_occ_q05  = mean(p_occ_q05,     na.rm = TRUE),
    p_occ_q95  = mean(p_occ_q95,     na.rm = TRUE),
    q_occ_mean = mean(q_occ_mean,    na.rm = TRUE),
    .groups = "drop"
  )

p_ts_agg <- ggplot(ts_agg, aes(x = year_month_date)) +
  geom_ribbon(aes(ymin = p_occ_q05, ymax = p_occ_q95), fill = "#378ADD", alpha = 0.25) +
  geom_line(aes(y = p_occ_mean, colour = "p_bt (mosquito presence)"), linewidth = 0.9) +
  geom_line(aes(y = q_occ_mean, colour = "q_bt (visit prob)"),        linewidth = 0.9, linetype = "dashed") +
  geom_line(aes(y = obs_mean,   colour = "Observed rate"),            linewidth = 0.7) +
  geom_point(aes(y = obs_mean,  colour = "Observed rate"),            size = 1.5) +
  scale_colour_manual(values = c(
    "p_bt (mosquito presence)" = "#185FA5",
    "q_bt (visit prob)"        = "#2CA02C",
    "Observed rate"            = "#D85A30"
  )) +
  labs(
    title    = "Mean-field occupancy: p_bt, q_bt and observed detection rate",
    subtitle = paste0("B = ", B, " | r_bt = p_bt × q_bt | ", cfg$iter_sampling, " post-warmup samples"),
    x = "Time", y = "Probability", colour = NULL
  ) +
  theme_minimal()

ggsave(
  file.path(cfg$output_dir, paste0("timeseries_aggregate_", run_suffix, ".png")),
  p_ts_agg, width = 12, height = 5, dpi = 150
)

# --- By-block time series (random sample) ---
sample_blocks <- sample(unique(df_obs$block_idx), min(9, B))

p_ts_block <- df_obs %>%
  filter(block_idx %in% sample_blocks) %>%
  ggplot(aes(x = year_month_date)) +
  geom_ribbon(aes(ymin = p_occ_q05, ymax = p_occ_q95), fill = "#378ADD", alpha = 0.2) +
  geom_line(aes(y = p_occ_mean,    colour = "Fitted"),   linewidth = 0.7) +
  geom_line(aes(y = observed_rate, colour = "Observed"), linewidth = 0.6) +
  geom_point(aes(y = observed_rate, colour = "Observed"), size = 1.0) +
  scale_colour_manual(values = c("Fitted" = "#185FA5", "Observed" = "#D85A30")) +
  facet_wrap(~block_idx, scales = "free_y") +
  labs(
    title  = "HMP occupancy by block: observed rate vs P(occupied | data)",
    x = "Time", y = "Probability", colour = NULL
  ) +
  theme_minimal(base_size = 10)

ggsave(
  file.path(cfg$output_dir, paste0("timeseries_by_block_", run_suffix, ".png")),
  p_ts_block, width = 14, height = 8, dpi = 150
)

# --- Posterior predictive check (3-panel, same layout as hierarchical model) ---
# Subset y_pred_mat draws to observed cells (N-indexed), reusing y_pred_draws from section 10
n_draws_ppc <- 50
y_pred_obs_draws <- matrix(NA_real_, nrow = nrow(y_pred_draws), ncol = nrow(df_obs))
for (i in seq_len(nrow(df_obs))) {
  col_pred <- paste0("y_pred_mat[", df_obs$block_idx[i], ",", df_obs$time_idx[i], "]")
  if (col_pred %in% colnames(y_pred_draws))
    y_pred_obs_draws[, i] <- y_pred_draws[, col_pred]
}

y_obs_ppc    <- df_obs$y_bt
post_mean_ppc <- colMeans(y_pred_obs_draws, na.rm = TRUE)

# Panel 1: proportion of zeros
prop_zero_rep <- rowMeans(y_pred_obs_draws == 0, na.rm = TRUE)
prop_zero_obs <- mean(y_obs_ppc == 0)

p_ppc1 <- ggplot(data.frame(prop_zero = prop_zero_rep), aes(x = prop_zero)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = prop_zero_obs, colour = "red", linewidth = 1) +
  annotate("text", x = prop_zero_obs, y = Inf,
           label = sprintf("observed\n%.2f", prop_zero_obs),
           colour = "red", hjust = -0.1, vjust = 1.5, size = 3) +
  labs(title = "Proportion of zeros",
       subtitle = "Histogram = replicated datasets; red = observed",
       x = "Proportion of zeros", y = "Count") +
  theme_minimal()

# Panel 2: non-zero count distribution
nonzero_obs  <- y_obs_ppc[y_obs_ppc > 0]
draw_idx_ppc <- sample(nrow(y_pred_obs_draws), min(n_draws_ppc, nrow(y_pred_obs_draws)))

rep_nonzero_df <- do.call(rbind, lapply(draw_idx_ppc, function(i) {
  vals <- y_pred_obs_draws[i, ][y_pred_obs_draws[i, ] > 0]
  if (length(vals) == 0) return(NULL)
  data.frame(count = vals, draw = i)
}))

obs_counts       <- as.data.frame(table(count = nonzero_obs))
obs_counts$count <- as.integer(as.character(obs_counts$count))

p_ppc2 <- ggplot() +
  geom_histogram(
    data = rep_nonzero_df,
    aes(x = count, group = draw),
    binwidth = 1, center = 1, fill = "steelblue", alpha = 0.05, position = "identity"
  ) +
  geom_point(data = obs_counts, aes(x = count, y = Freq), colour = "red", size = 1.5) +
  geom_line( data = obs_counts, aes(x = count, y = Freq), colour = "red", linewidth = 0.6) +
  labs(title = "Distribution of non-zero counts",
       subtitle = sprintf("Blue = %d replicated datasets; red = observed", n_draws_ppc),
       x = "y_bt (non-zero only)", y = "Count") +
  scale_x_continuous(
    breaks      = function(x) seq(ceiling(x[1]), floor(x[2]), by = 1),
    labels      = function(x) ifelse(x %% 5 == 0, x, ""),
    minor_breaks = NULL
  ) +
  coord_cartesian(xlim = c(1, NA)) +
  theme_minimal() +
  theme(panel.grid.major.x = element_line(colour = "grey85", linewidth = 0.3))

# Panel 3: fitted vs observed (posterior mean)
p_ppc3 <- ggplot(
  data.frame(observed = y_obs_ppc, predicted = post_mean_ppc),
  aes(observed, predicted)
) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_abline(slope = 1, intercept = 0, colour = "red") +
  labs(title = "Fitted vs observed (posterior mean)",
       x = "Observed y_bt", y = "Posterior mean y_pred") +
  theme_minimal()

p_ppc <- p_ppc1 + p_ppc2 + p_ppc3 + patchwork::plot_layout(ncol = 3)

ggsave(
  file.path(cfg$output_dir, paste0("posterior_predictive_check_", run_suffix, ".png")),
  p_ppc, width = 15, height = 5, dpi = 150
)

# --- Traceplots ---
draws_array <- fit$draws(variables = summary_vars, format = "array")
p_trace <- mcmc_trace(draws_array, pars = summary_vars)
ggsave(
  file.path(cfg$output_dir, paste0("traceplots_", run_suffix, ".png")),
  p_trace, width = 14, height = 10, dpi = 150
)

cat("\nAll outputs saved to:", cfg$output_dir, "\n")
