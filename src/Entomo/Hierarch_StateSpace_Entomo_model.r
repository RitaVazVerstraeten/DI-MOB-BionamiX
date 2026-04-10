# =====================================================
# Mosquito hierarchical model with reactive surveillance
# Clean calibration script (function-based)
# =====================================================

if (!require("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}
renv::restore(prompt = FALSE)

library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)
library(sf)
library(lubridate)
library(spdep)

# =========================
# 0) LOAD HELPER FUNCTIONS
# =========================
script_dir <- tryCatch({
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    if (rstudioapi::isAvailable()) {
      dirname(rstudioapi::getActiveDocumentContext()$path)
    } else { 
      getwd()
    }
  } else {
    getwd()
  }
}, error = function(e) getwd())

source(file.path(script_dir, "helper_functions.r"))
source(file.path(script_dir, "plot_functions.r"))

# =========================
# 1) SETTINGS
# =========================
hostname <- Sys.info()["nodename"]


# ========== Output structure and config =============
cfg <- list(
  data_dir = if (hostname == "frietjes") "~/data/Entomo" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/",
  data_file_name = "env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv",
  output_dir = if (hostname == "frietjes") "/home/rita/data/Entomo/fitting/stan" else "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan",

  # model variant
  use_time_RE     = FALSE,  # TRUE = iid time RE + iid block RE (no AR1, no GP); overrides others
  use_temporal_AR = TRUE,   # (ignored if use_time_RE = TRUE) TRUE = global AR1 trend
  use_spatial_AC  = TRUE,    # (ignored if use_time_RE = TRUE) TRUE = spatial AC
  use_hsgp        = FALSE,   # (only if use_spatial_AC = TRUE and use_icar = FALSE) TRUE = HSGP; FALSE = exact GP
  use_icar        = TRUE,    # (only if use_spatial_AC = TRUE) TRUE = ICAR neighbour-based; overrides use_hsgp
  hsgp_m          = 20,     # basis functions per dimension (20 → 400 total)
  hsgp_c          = 1.5,    # boundary factor (domain = c * data range)
  use_block_dev   = TRUE,   # (ignored if use_time_RE = TRUE) TRUE = per-block deviation

  # spatial
  shapefile_path = if (hostname == "frietjes")
    "/home/rita/data/Entomo"
  else
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders",
  sf_block_col = "CODIGO_",

  # data prep
  n_blocks = 100, # set NULL for all blocks
  lag_vars = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
  max_lag = 2,
  kappa = 4,
  unlagged_vars = c("is_urban", "is_WUI"),
  numeric_vars = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"), 

  # MCMC
  chains = 2,
  iter_warmup = 500,
  iter_sampling = 500,
  # thin = 2,
  adapt_delta = 0.95,
  max_treedepth = 12,
  parallel_chains = if (hostname == "frietjes") 2 else 1,

  # phi: set fix_phi = TRUE to pass phi as data (fixed); FALSE to estimate it
  fix_phi = TRUE,
  phi_fixed = 25,   # beta-binomial concentration -> later replace with gamma(2, 0.25)
  # prior predictive check (set TRUE before first real fit)
  run_prior_predictive = FALSE,

  # outputs (individual plot toggles)
  plot_traceplots = TRUE,
  plot_random_effects = TRUE,
  plot_ppc = TRUE,
  plot_timeseries = TRUE,
  n_blocks_facet = 9
)

# ========== Output directory structure =============
date_suffix <- format(Sys.Date(), "%Y%m%d")
model_spec <- if (isTRUE(cfg$use_time_RE)) {
  paste0("timeRE_blockRE_lag", cfg$max_lag, "_k", cfg$kappa)
} else {
  ar1_suffix <- ifelse(isTRUE(cfg$use_temporal_AR), "AR1", "noAR1")
  gp_suffix  <- if (!isTRUE(cfg$use_spatial_AC))  "noGP"
                else if (isTRUE(cfg$use_icar))    "ICAR"
                else if (isTRUE(cfg$use_hsgp))    "HSGP"
                else                              "GP"
  re_suffix  <- ifelse(isTRUE(cfg$use_block_dev), "blockRE", "noBlockRE")
  paste0(ar1_suffix, "_", gp_suffix, "_", re_suffix,
         "_lag", cfg$max_lag, "_k", cfg$kappa)
}
# model_tag <- ifelse(isTRUE(cfg$use_time_RE), "timeRE_blockRE",
#              ifelse(isTRUE(cfg$use_temporal_AR), "withAR1", "noAR1"))
predictor_spec <- paste0(
  "lag-", paste(cfg$lag_vars, collapse = "-"),
  "_unlag-", paste(cfg$unlagged_vars, collapse = "-")
)
# run_suffix <- paste0(date_suffix)

model_output_dir  <- file.path(cfg$output_dir, predictor_spec, model_spec)
run_output_dir    <- file.path(model_output_dir, date_suffix)
plots_output_dir  <- file.path(run_output_dir, "plots")
resid_output_dir  <- file.path(run_output_dir, "residuals_check")
dir.create(run_output_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(plots_output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(resid_output_dir, recursive = TRUE, showWarnings = FALSE)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)

# ================= selection stan file ============================
stan_dir <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo"
cfg$stan_file <- if (isTRUE(cfg$use_time_RE)) {
  # iid time RE + iid block RE (no AR1, no GP)
  file.path(stan_dir, "hierarchical_state_space_timeRE_blockRE.stan")
} else if (!isTRUE(cfg$use_temporal_AR) && !isTRUE(cfg$use_spatial_AC)) {
  # Base: no AR, no GP, no blockRE
  file.path(stan_dir, "hierarchical_state_space.stan")
} else if (!isTRUE(cfg$use_spatial_AC)) {
  # AR only variants (no GP)
  if (isTRUE(cfg$use_block_dev)) {
    file.path(stan_dir, "hierarchical_state_space_AR_blockRE.stan")
  } else {
    file.path(stan_dir, "hierarchical_state_space_AR.stan")
  }
} else if (isTRUE(cfg$use_icar)) {
  # ICAR neighbour-based spatial RE
  file.path(stan_dir, "hierarchical_state_space_AR_blockRE_ICAR.stan")
} else if (isTRUE(cfg$use_hsgp)) {
  # HSGP variants
  if (isTRUE(cfg$use_block_dev)) {
    file.path(stan_dir, "hierarchical_state_space_AR_blockRE_HSGP.stan")
  } else {
    file.path(stan_dir, "hierarchical_state_space_AR_HSGP.stan")
  }
} else {
  # Exact GP variant
  file.path(stan_dir, "hierarchical_state_space_AR_blockRE_GP.stan")
}
cat("Stan file:", cfg$stan_file, "\n")
# cfg$stan_file <- if (cfg$use_temporal_AR) {
#   "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/hierarchical_state_space.stan"
# } else {
#   "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/hierarchical_state_space_no_time_re.stan"
# }


options(mc.cores = if (hostname == "frietjes") 6 else 2)
dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# information on host, data source and model variant
# =========================
cat("Using hostname:", hostname, "\n")
cat("Data directory:", cfg$data_dir, "\n")
cat("Model variant:" , "Predictors: ", predictor_spec, "\n","RE and AR: ", model_spec, "\n")


# =========================
# STANDARDIZE NUMERIC COVARIATES - done in helper functions
# =========================

prep <- build_stan_data(cfg)
stan_data <- prep$stan_data
df <- prep$df


# =========================
# SPATIAL DISTANCE MATRIX
# =========================
sf_blocks  <- st_read(file.path(cfg$shapefile_path, "Manzanas_cleaned_05032026", "Mz_CMF_Correcto_2022026.shp"), quiet = TRUE)
pts        <- suppressWarnings(st_point_on_surface(sf_blocks))
if (st_is_longlat(pts)) pts <- st_transform(pts, 3857)

block_ids  <- sort(unique(as.character(df$manzana)))  # ordered to match block index
coords_sf  <- sf_blocks %>%
  st_drop_geometry() %>%
  mutate(
    block_chr = as.character(.data[[cfg$sf_block_col]]),
    x = st_coordinates(pts)[, 1],
    y = st_coordinates(pts)[, 2]
  ) %>%
  filter(block_chr %in% block_ids) %>%
  arrange(match(block_chr, block_ids)) %>%
  select(block_chr, x, y) %>%
  distinct(block_chr, .keep_all = TRUE)

# ======================== spatial data prep ====================================
if (isTRUE(cfg$use_spatial_AC)) {
  if (isTRUE(cfg$use_icar)) {
    icar_edges <- build_icar_edges(sf_blocks, block_ids, cfg$sf_block_col, snap_m = 100)
    stan_data$N_edges <- icar_edges$N_edges
    stan_data$node1   <- icar_edges$node1
    stan_data$node2   <- icar_edges$node2
    cat(sprintf("ICAR: %d blocks, %d unique edges\n", length(block_ids), stan_data$N_edges))

  } else if (isTRUE(cfg$use_hsgp)) {
    stan_data$coords_block <- as.matrix(coords_sf[, c("x", "y")])
    stan_data$M            <- cfg$hsgp_m
    stan_data$c_boundary   <- cfg$hsgp_c
    cat(sprintf(
      "HSGP: %d blocks, M=%d per dim (%d basis functions total), c=%.1f\n",
      nrow(stan_data$coords_block), cfg$hsgp_m, cfg$hsgp_m^2, cfg$hsgp_c
    ))
  } else {
    dist_mat <- as.matrix(dist(coords_sf[, c("x", "y")]))
    stan_data$dist_block <- dist_mat
    cat(sprintf(
      "Exact GP: distance matrix %d x %d blocks\n",
      nrow(dist_mat), ncol(dist_mat)
    ))
  }
} else {
  cat("No spatial AC: skipping coordinate/distance/neighbour setup.\n")
}


#####################################
# Checking for overdispersion in my data
#####################################
disp_df <- df %>%
  mutate(
    n_bt   = stan_data$n_bt,
    y_bt   = stan_data$y
  ) %>%
  filter(n_bt > 0) %>%
  mutate(
    y_rate  = y_bt / n_bt
  )

phi_grouped <- disp_df %>%
  filter(n_bt > 0) %>% 
  mutate(y_rate = y_bt / n_bt) %>%
  group_by(n_bt) %>%
  filter(n() >= 30)  %>%  # only bins with enough cells to estimate variance (at least 30 measures though time with n_bt number of households - can be repeats of same block or different blocks)
  summarise(
    n_cells      = n(),
    p_bar        = mean(y_rate),
    var_observed = var(y_rate),
    var_binomial = p_bar * (1 - p_bar) / first(n_bt),  # expected under binomial
    dispersion_ratio = var_observed / var_binomial,
    phi_implied      = (first(n_bt) - dispersion_ratio) / (dispersion_ratio - 1)
  ) %>%
  filter(dispersion_ratio > 1, phi_implied > 0) %>%  # only valid estimates
  summarise(
    phi_median   = median(phi_implied),
    phi_mean     = mean(phi_implied),
    phi_weighted = weighted.mean(phi_implied, w = n_cells)  # weight by cell count
  )
cat("implied phi (grouped median across n_bt bins):", round(phi_grouped$phi_median, 2), "\n")
print(phi_grouped)

disp_df %>%
  filter(n_bt > 0) %>%
  mutate(y_rate = y_bt / n_bt) %>%
  group_by(n_bt) %>%
  # filter(n() >= 30) %>%
  summarise(
    p_bar        = mean(y_rate),
    var_observed = var(y_rate),
    var_binomial = p_bar * (1 - p_bar) / first(n_bt)
  ) %>%
  ggplot(aes(x = n_bt)) +
  geom_point(aes(y = var_observed), colour = "steelblue") +
  geom_line(aes(y = var_binomial), colour = "red") +
  scale_y_log10() +
  labs(x = "n_bt", y = "variance of y/n_bt (log scale)",
       title = "Observed variance vs binomial expectation",
       subtitle = "Blue dots above red line = overdispersion")

       
# Pass phi as fixed data when fix_phi = TRUE
if (isTRUE(cfg$fix_phi)) {
  phi_grouped_median <- phi_grouped$phi_median
  phi_use <- if (is.finite(phi_grouped_median) && phi_grouped_median > 0 && phi_grouped_median < 500) {
    round(phi_grouped_median, 1)
  } else {
    cfg$phi_fixed
  }
  stan_data$phi <- phi_use
  cat(sprintf(
    "phi fixed at %.1f (grouped median across n_bt bins; cfg fallback = %.1f)\n",
    phi_use, cfg$phi_fixed
  ))
  cfg$phi_used <- phi_use
}

# ================== compile stan model ==========================
mod <- cmdstan_model(cfg$stan_file,
  force_recompile = hostname == "frietjes")

# =========================
# PRIOR PREDICTIVE CHECK
# =========================
# Run with fixed_param = TRUE to sample from the prior only (no likelihood).
# Check that implied prevalence is plausible (target ~0.5–5% positivity).
# Set cfg$run_prior_predictive = TRUE to enable; keep FALSE for real fits.
if (isTRUE(cfg$run_prior_predictive)) {
  cat("\nRunning prior predictive check (fixed_param = TRUE)...\n")
  fit_prior <- mod$sample(
    data            = stan_data,
    chains          = 2,
    iter_warmup     = 200,
    iter_sampling   = 200,
    fixed_param     = TRUE,
    parallel_chains = cfg$parallel_chains
  )
  ppc_draws    <- fit_prior$draws("y_pred", format = "matrix")
  implied_prev <- ppc_draws /
    matrix(stan_data$n_bt,
           nrow = nrow(ppc_draws), ncol = length(stan_data$n_bt), byrow = TRUE)
  cat(sprintf(
    "Prior predictive prevalence: median=%.3f  5th=%.3f  95th=%.3f\n",
    median(implied_prev, na.rm = TRUE),
    quantile(implied_prev, 0.05, na.rm = TRUE),
    quantile(implied_prev, 0.95, na.rm = TRUE)
  ))
  p_prior <- ggplot(data.frame(prev = as.vector(implied_prev)), aes(x = prev)) +
    geom_histogram(bins = 80, fill = "steelblue") +
    geom_vline(xintercept = c(0.005, 0.05),
               linetype = "dashed", colour = "red") +
    scale_x_continuous(limits = c(0, 0.3)) +
    labs(title    = "Prior predictive prevalence",
         subtitle = "Dashed red = 0.5% and 5% target range",
         x = "Implied positivity rate", y = "Count") +
    theme_minimal()
  ggsave(
    file.path(cfg$output_dir,
              paste0("prior_predictive_", run_suffix, ".png")),
    p_prior, width = 8, height = 5, dpi = 150
  )
  cat("Prior predictive plot saved to:", cfg$output_dir, "\n")
  stop("Prior predictive check complete. Set run_prior_predictive = FALSE to proceed with the real fit.")
}

# ========================= MCMC sampling =======================================
fit <- mod$sample(
  data = stan_data,
  chains = cfg$chains,
  iter_warmup = cfg$iter_warmup,
  iter_sampling = cfg$iter_sampling,
  thin = if (!is.null(cfg$thin)) cfg$thin else 1,
  init = make_init_fun(
    stan_data, cfg$use_temporal_AR,
    use_hsgp       = isTRUE(cfg$use_hsgp) && !isTRUE(cfg$use_icar),
    use_icar       = isTRUE(cfg$use_icar),
    use_time_RE    = isTRUE(cfg$use_time_RE),
    use_spatial_AC = isTRUE(cfg$use_spatial_AC)
  ),
  adapt_delta = cfg$adapt_delta,
  max_treedepth = cfg$max_treedepth,
  parallel_chains = cfg$parallel_chains,
  output_dir = run_output_dir   # write chain CSVs here instead of /tmp
)


# CSV chain files are already in run_output_dir; skip .rds to save disk space
# (re-load later with: fit <- as_cmdstan_fit(list.files(run_output_dir, "*.csv", full.names=TRUE)))
# Remove CmdStan auxiliary files (config/metric JSONs) — not needed for post-processing
invisible(file.remove(list.files(run_output_dir, pattern = "_(config|metric)\\.json$", full.names = TRUE)))

# ======================= make model summary ============================
summary_vars <- c("alpha", "delta0", "delta1", "w", "sigma_w", "w_unlagged")
if (isTRUE(cfg$use_time_RE)) {
  summary_vars <- c(summary_vars, "sigma_time", "sigma_block")
} else {
  if (isTRUE(cfg$use_spatial_AC)) {
    if (isTRUE(cfg$use_icar))   summary_vars <- c(summary_vars, "sigma_icar")
    else                         summary_vars <- c(summary_vars, "sigma_gp", "rho_gp")
  }
  if (isTRUE(cfg$use_temporal_AR)) summary_vars <- c(summary_vars, "sigma_v", "rho")
  if (isTRUE(cfg$use_block_dev))   summary_vars <- c(summary_vars, "sigma_block_dev")
}
if (!isTRUE(cfg$fix_phi)) summary_vars <- c(summary_vars, "phi")

summary_output <- capture.output(print(fit$summary(variables = summary_vars), n = Inf))
writeLines(summary_output, file.path(run_output_dir, paste0("model_summary_", run_suffix, ".txt")))

# =========================== posterior draws for plotting ===========================

post <- extract_means(fit, nrow(df))

# Prepare data for plotting
df$n_bt              <- stan_data$n_bt
df$fitted_p_bt       <- post$p_bt
df$observed_p_bt     <- df$y_bt / df$n_bt
df$y_pred_rate       <- colMeans(fit$draws("y_pred", format = "matrix")) / df$n_bt
y_pred_draws_mat     <- fit$draws("y_pred", format = "matrix")
df$y_pred_rate_q05   <- apply(y_pred_draws_mat, 2, quantile, probs = 0.05) / df$n_bt
df$y_pred_rate_q95   <- apply(y_pred_draws_mat, 2, quantile, probs = 0.95) / df$n_bt
rm(y_pred_draws_mat)


# =========================
# 3) GENERATE PLOTS
# =========================

# Robust random effects plotting for spatial GP and/or temporal AR
if (cfg$plot_random_effects) {
  if (!all(is.na(post$u)) && length(post$u) > 0) {
    cat("\nGenerating random effects plot (spatial + temporal)...\n")
    save_random_effects(post$u, post$v, plots_output_dir, run_suffix)
  } else if (!all(is.na(post$v)) && length(post$v) > 0) {
    cat("\nNo spatial random effects found. Plotting only temporal AR effects...\n")
    # Plot only temporal AR effects
    png(file.path(plots_output_dir, paste0("random_effects_temporal_only_", run_suffix, ".png")), width = 800, height = 600)
    par(mfrow = c(1, 2))
    plot(post$v, type = "b", main = "Temporal Random Effects (v_t)",
         xlab = "Time", ylab = "Effect", col = "red", pch = 19)
    abline(h = 0, lty = 2, col = "gray")
    acf(post$v, main = "ACF of Temporal Effects", col = "darkred")
    par(mfrow = c(1, 1))
    dev.off()
  } else {
    cat("\nNo random effects found to plot (neither spatial nor temporal).\n")
  }
}


if (cfg$plot_ppc) {
  cat("Generating posterior predictive check plot...\n")
  save_ppc(df, fit, plots_output_dir, run_suffix)
}

# Robust traceplot generation: only plot parameters that exist in the fit object
if (cfg$plot_traceplots) {
  cat("Generating trace plots...\n")
  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    cat("bayesplot package not installed; skipping trace plots.\n")
  } else {
    library(bayesplot)

    available_params <- fit$summary()$variable

    # Whitelist: scalar model parameters worth tracing
    scalar_include <- c("alpha", "sigma_gp", "rho_gp", "sigma_icar",
                        "delta0", "delta1",
                        "sigma_v", "rho", "sigma_block_dev",
                        "sigma_time", "sigma_block",
                        "phi")
    scalar_params <- available_params[
      available_params %in% scalar_include |
      grepl("^sigma_w\\[", available_params) |  # K elements, one per covariate
      grepl("^v_global\\[", available_params)   # T=12 global AR(1) trend values
    ]

    w_params  <- available_params[grepl("^w\\[", available_params)]
    wu_params <- available_params[grepl("^w_unlagged\\[", available_params)]

    # Helper: save chunked traceplots (avoids huge single ggplot)
    save_trace_chunks <- function(vars, draws_arr, file_prefix, chunk_size = 12, w, h) {
      chunks <- split(vars, ceiling(seq_along(vars) / chunk_size))
      for (i in seq_along(chunks)) {
        ggsave(
          file.path(plots_output_dir, paste0(file_prefix, "_part", i, "_", run_suffix, ".png")),
          mcmc_trace(draws_arr, pars = chunks[[i]]), width = w, height = h
        )
      }
    }

    # Scalar params
    if (length(scalar_params) > 0) {
      draws_scalar <- fit$draws(variables = scalar_params, format = "array")
      save_trace_chunks(scalar_params, draws_scalar, "traceplot_params", chunk_size = 12, w = 10, h = 8)
    } else {
      cat("No scalar parameters found for traceplot.\n")
    }

    # Lagged weights
    if (length(w_params) > 0) {
      draws_w <- fit$draws(variables = w_params, format = "array")
      save_trace_chunks(w_params, draws_w, "traceplot_weights_w", chunk_size = 12, w = 12, h = 10)
    }

    # Unlagged weights
    if (length(wu_params) > 0) {
      draws_wu <- fit$draws(variables = wu_params, format = "array")
      save_trace_chunks(wu_params, draws_wu, "traceplot_weights_unlagged", chunk_size = 12, w = 12, h = 8)
    }
  }
}


if (cfg$plot_timeseries) {
  cat("Generating time series plots...\n")
  save_timeseries_plots(df, plots_output_dir, run_suffix, cfg$n_blocks_facet)
}


# glmmTMB-specific diagnostics (residuals, QQ plots) are not applicable to CmdStan fits; skipped

cat("\nAll outputs saved to:", run_output_dir, "\n")

# =========================
# 4) MORAN'S I ON STAN POSTERIOR RESIDUALS
# =========================
# Computes Pearson residuals from posterior mean y_pred, then runs the same
# 50m-annuli correlogram as GLMM.r section 15g, stratified by year.
# Key question: does 2016 spatial autocorrelation disappear after Stan's
# reactive mixture correction?

if (requireNamespace("spdep", quietly = TRUE)) {

  post_y_pred  <- fit$draws("y_pred", format = "matrix")
  mean_y_pred  <- colMeans(post_y_pred)
  df$n_bt      <- stan_data$n_bt

  df$pearson_resid_stan <- (df$y_bt - mean_y_pred) /
    sqrt(pmax(mean_y_pred * (1 - mean_y_pred / df$n_bt), 1e-6))
  df$year <- lubridate::year(df$year_month_date)

  df_spatial <- df %>%
    mutate(block_chr = as.character(manzana)) %>%
    left_join(coords_sf %>% select(block_chr, x, y), by = "block_chr")

  distance_breaks <- seq(0, 2000, by = 50)
  years_stan      <- sort(unique(df_spatial$year))

  moran_stan_list <- vector("list", length(years_stan))

  for (yr in years_stan) {
    block_resid_yr <- df_spatial %>%
      filter(year == yr) %>%
      group_by(block_chr) %>%
      summarise(mean_resid = mean(pearson_resid_stan, na.rm = TRUE),
                x = first(x), y = first(y), .groups = "drop") %>%
      filter(!is.na(x), !is.na(y), !is.na(mean_resid), is.finite(mean_resid))

    if (nrow(block_resid_yr) < 10) next

    coords_yr  <- as.matrix(block_resid_yr[, c("x", "y")])
    dist_yr    <- as.matrix(dist(coords_yr))
    diag(dist_yr) <- NA_real_

    band_list <- vector("list", length(distance_breaks) - 1)
    for (i in seq_len(length(distance_breaks) - 1)) {
      d_low  <- distance_breaks[i]
      d_high <- distance_breaks[i + 1]
      w <- matrix(0, nrow = nrow(dist_yr), ncol = ncol(dist_yr))
      w[!is.na(dist_yr) & dist_yr > d_low & dist_yr <= d_high] <- 1
      if (sum(w) == 0) {
        band_list[[i]] <- data.frame(
          year = yr, d_low = d_low, d_high = d_high,
          d_mid = (d_low + d_high) / 2,
          morans_I = NA_real_, p_value = NA_real_, significant = NA)
        next
      }
      lw <- spdep::mat2listw(w, style = "W", zero.policy = TRUE)
      mt <- tryCatch(
        spdep::moran.test(block_resid_yr$mean_resid, lw, zero.policy = TRUE),
        error   = function(e) NULL,
        warning = function(w) suppressWarnings(
          spdep::moran.test(block_resid_yr$mean_resid, lw, zero.policy = TRUE))
      )
      if (is.null(mt)) {
        band_list[[i]] <- data.frame(
          year = yr, d_low = d_low, d_high = d_high,
          d_mid = (d_low + d_high) / 2,
          morans_I = NA_real_, p_value = NA_real_, significant = NA)
        next
      }
      band_list[[i]] <- data.frame(
        year        = yr,
        d_low       = d_low, d_high = d_high,
        d_mid       = (d_low + d_high) / 2,
        morans_I    = unname(mt$estimate[["Moran I statistic"]]),
        p_value     = mt$p.value,
        significant = mt$p.value < 0.05)
    }
    moran_stan_list[[which(years_stan == yr)]] <- do.call(rbind, band_list)
  }

  moran_stan_df <- do.call(rbind, moran_stan_list) %>%
    filter(!is.na(morans_I)) %>%
    mutate(year = factor(year))

  p_moran_stan <- ggplot(
    moran_stan_df,
    aes(x = d_mid, y = morans_I, colour = year, group = year)
  ) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "gray50") +
    geom_line(linewidth = 0.8) +
    geom_point(aes(shape = significant), size = 2) +
    scale_shape_manual(values  = c("TRUE" = 16, "FALSE" = 1),
                       labels  = c("TRUE" = "p < 0.05", "FALSE" = "p >= 0.05"),
                       na.value = 1) +
    scale_x_continuous(breaks = seq(0, 2000, by = 200)) +
    labs(
      title    = "Moran's I on Stan posterior Pearson residuals - by year",
      subtitle = "Remaining autocorrelation after reactive-mixture correction",
      x = "Distance band midpoint (m)", y = "Moran's I",
      colour = "Year", shape = NULL
    ) +
    theme_minimal()

  ggsave(
    file.path(plots_output_dir,
              paste0("moransI_stan_by_year_", run_suffix, ".png")),
    p_moran_stan, width = 11, height = 5, dpi = 150
  )

  write.csv(
    moran_stan_df,
    file.path(plots_output_dir,
              paste0("moransI_stan_by_year_", run_suffix, ".csv")),
    row.names = FALSE
  )
  cat("Stan Moran's I correlogram saved to:", plots_output_dir, "\n")

} else {
  cat("Skipping Stan Moran's I: package 'spdep' not installed.\n")
}
