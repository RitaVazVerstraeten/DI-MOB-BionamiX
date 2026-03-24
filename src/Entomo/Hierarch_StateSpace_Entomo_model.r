# =====================================================
# Mosquito hierarchical model with reactive surveillance
# Clean calibration script (function-based)
# =====================================================

if (!require("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}
renv::restore()

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

cfg <- list(
  data_dir = if (hostname == "frietjes") "~/data/Entomo" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/",
  data_file_name = "env_epi_entomo_data_per_manzana_2016_04_to_2019_12.csv",
  output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan",

  # model variant
  use_temporal_re = FALSE,

  # spatial
  shapefile_path = if (hostname == "frietjes")
    "/home/rita/data/Entomo/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
  else
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp",
  sf_block_col = "CODIGO_",

  # data prep
  n_blocks = 100, # set NULL for all blocks
  lag_vars = c("avg_temp", "rel_hum", "total_precip", "mean_ndvi"),
  max_lag = 1,
  kappa = 2,
  unlagged_vars = c("is_urban", "has_aljibes", "is_WI", "is_WUI", "water_shortage", "WS2M"),
  binary_unlagged_vars = c("is_urban", "has_aljibes", "is_WI", "is_WUI", "water_shortage"),

  # MCMC
  chains = 2,
  iter_warmup = 1000,
  iter_sampling = 1000,
  # thin = 2,
  adapt_delta = 0.95,
  max_treedepth = 12,
  parallel_chains = if (hostname == "frietjes") 2 else 1,

  # prior predictive check (set TRUE before first real fit)
  run_prior_predictive = FALSE,

  # outputs (individual plot toggles)
  plot_traceplots = TRUE,
  plot_random_effects = TRUE,
  plot_ppc = TRUE,
  plot_timeseries = TRUE,
  n_blocks_facet = 9
)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)
cfg$stan_file <- if (cfg$use_temporal_re) {
  "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/Entomo_model/hierarchical_state_space.stan"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/Entomo_model/hierarchical_state_space_no_time_re.stan"
}

date_suffix <- format(Sys.Date(), "%Y%m%d")
model_tag <- ifelse(cfg$use_temporal_re, "withTimeRE", "noTimeRE")
run_suffix <- paste0(date_suffix, "_stand_", model_tag)

options(mc.cores = if (hostname == "frietjes") 6 else 2)
dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 2) MAIN
# =========================
cat("Using hostname:", hostname, "\n")
cat("Data directory:", cfg$data_dir, "\n")
cat("Model variant:", ifelse(cfg$use_temporal_re, "with temporal RE", "without temporal RE"), "\n")

prep <- build_stan_data(cfg)
stan_data <- prep$stan_data
df <- prep$df

# =========================
# 2b) SPATIAL DISTANCE MATRIX
# =========================
sf_blocks  <- st_read(cfg$shapefile_path, quiet = TRUE)
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

dist_mat <- as.matrix(dist(coords_sf[, c("x", "y")]))
stan_data$dist_block <- dist_mat
cat(sprintf("Distance matrix: %d × %d blocks\n", nrow(dist_mat), ncol(dist_mat)))

mod <- cmdstan_model(cfg$stan_file)

# =========================
# 2c) PRIOR PREDICTIVE CHECK
# =========================
# Run with fixed_param = TRUE to sample from the prior only (no likelihood).
# Check that implied prevalence is plausible (target ~0.5–5% positivity).
# Set cfg$run_prior_predictive = TRUE to enable; keep FALSE for real fits.
if (isTRUE(cfg$run_prior_predictive)) {
  cat("\nRunning prior predictive check (fixed_param = TRUE)...\n")
  fit_prior <- mod$sample(
    data            = stan_data,
    chains          = 2,
    iter_warmup     = 0,
    iter_sampling   = 500,
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

fit <- mod$sample(
  data = stan_data,
  chains = cfg$chains,
  iter_warmup = cfg$iter_warmup,
  iter_sampling = cfg$iter_sampling,
  thin = cfg$thin,
  init = make_init_fun(stan_data, cfg$use_temporal_re),
  adapt_delta = cfg$adapt_delta,
  max_treedepth = cfg$max_treedepth,
  parallel_chains = cfg$parallel_chains
)

fit$save_object(file.path(cfg$output_dir, paste0("fit_", run_suffix, ".rds")))

summary_vars <- c("alpha", "sigma_gp", "rho_gp", "phi", "delta0", "delta1", "w")
if (cfg$use_temporal_re) summary_vars <- c(summary_vars, "sigma_v", "rho")
summary_output <- capture.output(print(fit$summary(variables = summary_vars)))
cat(summary_output, sep = "\n")
writeLines(summary_output, file.path(cfg$output_dir, paste0("model_summary_", run_suffix, ".txt")))

post <- extract_means(fit, nrow(df))

# Prepare data for plotting
df$fitted_p_bt <- post$p_bt
df$observed_p_bt <- df$y_bt / df$N_HH

# =========================
# 3) GENERATE PLOTS
# =========================

if (cfg$plot_random_effects && !all(is.na(post$u))) {
  cat("\nGenerating random effects plot...\n")
  save_random_effects(post$u, post$v, cfg$output_dir, run_suffix)
}

if (cfg$plot_ppc) {
  cat("Generating posterior predictive check plot...\n")
  save_ppc(df, post$y_pred, cfg$output_dir, run_suffix)
}

if (cfg$plot_traceplots) {
  cat("Generating trace plots...\n")
  save_trace_plots(fit, cfg$output_dir, run_suffix, cfg$use_temporal_re)
}

if (cfg$plot_timeseries) {
  cat("Generating time series plots...\n")
  save_timeseries_plots(df, cfg$output_dir, run_suffix, cfg$n_blocks_facet)
}

cat("\nAll outputs saved to:", cfg$output_dir, "\n")

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
      filter(!is.na(x), !is.na(y), !is.na(mean_resid))

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
      mt <- spdep::moran.test(block_resid_yr$mean_resid, lw, zero.policy = TRUE)
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
                       labels  = c("TRUE" = "p < 0.05", "FALSE" = "p ≥ 0.05"),
                       na.value = 1) +
    scale_x_continuous(breaks = seq(0, 2000, by = 200)) +
    labs(
      title    = "Moran's I on Stan posterior Pearson residuals — by year",
      subtitle = "Remaining autocorrelation after reactive-mixture correction",
      x = "Distance band midpoint (m)", y = "Moran's I",
      colour = "Year", shape = NULL
    ) +
    theme_minimal()

  ggsave(
    file.path(cfg$output_dir,
              paste0("moransI_stan_by_year_", run_suffix, ".png")),
    p_moran_stan, width = 11, height = 5, dpi = 150
  )

  write.csv(
    moran_stan_df,
    file.path(cfg$output_dir,
              paste0("moransI_stan_by_year_", run_suffix, ".csv")),
    row.names = FALSE
  )
  cat("Stan Moran's I correlogram saved to:", cfg$output_dir, "\n")

} else {
  cat("Skipping Stan Moran's I: package 'spdep' not installed.\n")
}
