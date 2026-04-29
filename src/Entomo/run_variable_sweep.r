# =============================================================
# Variable sweep: run ICAR model with incrementally added predictors
# =============================================================
# Edit the `combinations` list below to define which variable sets to fit.
# All other settings are shared across runs (same model structure, same MCMC).
# The Stan model is compiled once; only the design matrices change per run.
# =============================================================

if (!require("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}

library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)
library(sf)
library(lubridate)
library(spdep)

# ── Script directory detection ────────────────────────────────────────────────
script_dir <- tryCatch({
  p <- rstudioapi::getActiveDocumentContext()$path
  if (nzchar(p)) dirname(p) else stop("empty path")
}, error = function(e) tryCatch({
  frames <- sys.frames()
  for (f in rev(frames)) {
    if (!is.null(f$ofile) && nzchar(f$ofile))
      return(dirname(normalizePath(f$ofile, mustWork = FALSE)))
  }
  args <- commandArgs(trailingOnly = FALSE)
  fa   <- grep("--file=", args, value = TRUE)
  if (length(fa)) dirname(normalizePath(sub("--file=", "", fa[1]), mustWork = FALSE))
  else stop("no path")
}, error = function(e2) {
  candidate <- file.path(getwd(), "src", "Entomo")
  if (file.exists(file.path(candidate, "helper_functions.r"))) candidate else getwd()
}))

renv::restore(project = script_dir, prompt = FALSE)
source(file.path(script_dir, "helper_functions.r"))
source(file.path(script_dir, "plot_functions.r"))

# ── Define your variable combinations here ───────────────────────────────────
# Each list entry = one model run. Add, remove, or reorder rows freely.
# numeric_vars = lag_vars minus any ending in _cat (those are treated as categorical).
combinations <- list(
  # original variables from GLMM + extra's 
  list(lag = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI")),
  list(lag = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI", "is_WI")),
  list(lag = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI", "is_WI", "has_aljibes")),
  list(lag = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI", "is_WI", "has_aljibes", "water_shortage")),
  list(lag = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI", "is_WI", "has_aljibes", "water_containers")),
  list(lag = c("total_rainy_days", "temp_cat", "avg_VPD", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI", "is_WI", "has_aljibes", "water_containers")),
  
  # variations in rain indicators: 
  list(lag = c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban")),
  list(lag = c("total_precip", "avg_VPD", "precip_max_day_resid", "mean_ndvi"),
        unlag = c("is_urban")),
  list(lag = c("consec_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
        unlag = c("is_urban")),
  list(lag = c("precip_cat", "avg_VPD", "precip_max_day", "mean_ndvi"),
        unlag = c("is_urban")),     
  # addition of temperature indicators 
  list(lag = c("avg_temp", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban")),
  list(lag = c("avg_temp", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI")),
  list(lag = c("avg_temp", "precip_max_day", "mean_ndvi"),
       unlag = c("is_urban", "is_WUI", "is_WI")),
  list(lag = c("temp_cat", "total_precip", "mean_ndvi"),
        unlag = c("is_urban")),
  list(lag = c("temp_cat", "total_rainy_days", "mean_ndvi"),
        unlag = c("is_urban")),
  list(lag = c("consec_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi"),
        unlag = c("is_urban"))
)
# ─────────────────────────────────────────────────────────────────────────────

hostname      <- Sys.info()["nodename"]
spatial_level <- "CMF"
stan_dir      <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo"
date_suffix   <- format(Sys.Date(), "%Y%m%d")

# ── Base configuration (shared across all runs) ───────────────────────────────
cfg <- list(
  data_dir = if (hostname == "frietjes") "~/data/Entomo"
             else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/",
  data_file_name = if (spatial_level == "CMF")
    "env_epi_entomo_data_per_CMF_2016_01_to_2019_12_noColinnearity.csv"
  else
    "env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv",
  output_dir = if (hostname == "frietjes")
    "/home/rita/data/Entomo/fitting/stan"
  else
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan",

  use_time_RE     = FALSE,
  use_temporal_AR = FALSE,
  use_spatial_AC  = FALSE,
  use_hsgp        = FALSE,
  use_icar        = FALSE,
  use_bym2        = FALSE,
  hsgp_m          = 20,
  hsgp_c          = 1.5,
  use_block_dev   = FALSE,

  shapefile_path = if (hostname == "frietjes")
    "/home/rita/data/Entomo"
  else
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders",
  sf_block_col = if (spatial_level == "CMF") "Area_CMF" else "CODIGO_",
  block_col    = if (spatial_level == "CMF") "cmf"      else "manzana",

  n_blocks  = 5,
  max_lag   = 2,
  kappa     = 4,

  chains          = 2,
  iter_warmup     = 500,
  iter_sampling   = 500,
  adapt_delta     = 0.95,
  max_treedepth   = 12,
  parallel_chains = if (hostname == "frietjes") 4 else 1,

  fix_phi              = FALSE,
  phi_fixed            = 25,
  run_prior_predictive = FALSE,

  plot_traceplots     = TRUE,
  plot_random_effects = TRUE,
  plot_ppc            = TRUE,
  plot_timeseries     = TRUE,
  n_blocks_facet      = 9
)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)

# ── Stan file selection ───────────────────────────────────────────────────────
cfg$stan_file <- if (isTRUE(cfg$use_time_RE)) {
  file.path(stan_dir, "hierarchical_state_space_timeRE_blockRE.stan")
} else if (!isTRUE(cfg$use_temporal_AR) && !isTRUE(cfg$use_spatial_AC)) {
  file.path(stan_dir, "hierarchical_state_space.stan")
} else if (!isTRUE(cfg$use_spatial_AC)) {
  if (isTRUE(cfg$use_block_dev))
    file.path(stan_dir, "hierarchical_state_space_AR_blockRE.stan")
  else
    file.path(stan_dir, "hierarchical_state_space_AR.stan")
} else if (isTRUE(cfg$use_bym2)) {
  file.path(stan_dir, "hierarchical_state_space_AR_BYM2.stan")
} else if (isTRUE(cfg$use_icar)) {
  if (isTRUE(cfg$use_block_dev))
    file.path(stan_dir, "hierarchical_state_space_AR_blockRE_ICAR.stan")
  else
    file.path(stan_dir, "hierarchical_state_space_AR_ICAR_noBlockDev.stan")
} else if (isTRUE(cfg$use_hsgp)) {
  if (isTRUE(cfg$use_block_dev))
    file.path(stan_dir, "hierarchical_state_space_AR_blockRE_HSGP.stan")
  else
    file.path(stan_dir, "hierarchical_state_space_AR_HSGP.stan")
} else {
  file.path(stan_dir, "hierarchical_state_space_AR_blockRE_GP.stan")
}
cat("Stan file:", cfg$stan_file, "\n")

# ── model_spec: encodes model structure (same for all combos) ─────────────────
model_spec <- if (isTRUE(cfg$use_time_RE)) {
  paste0("timeRE_blockRE_lag", cfg$max_lag, "_k", cfg$kappa)
} else {
  ar1_suffix     <- ifelse(isTRUE(cfg$use_temporal_AR), "AR1", "noAR1")
  gp_suffix      <- if (!isTRUE(cfg$use_spatial_AC))  "noGP"
                    else if (isTRUE(cfg$use_bym2))    "BYM2"
                    else if (isTRUE(cfg$use_icar))    "ICAR"
                    else if (isTRUE(cfg$use_hsgp))    "HSGP"
                    else                              "GP"
  re_suffix      <- ifelse(isTRUE(cfg$use_block_dev), "blockRE", "noBlockRE")
  n_block_suffix <- paste0(ifelse(is.null(cfg$n_blocks), "All", cfg$n_blocks), "Blocks")
  paste0(ar1_suffix, "_", gp_suffix, "_", re_suffix,
         "_lag", cfg$max_lag, "_k", cfg$kappa, "_", n_block_suffix)
}
model_spec <- paste0(spatial_level, "_", model_spec)

options(mc.cores = if (hostname == "frietjes") 6 else 2)
dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

# ── One-time spatial setup (outside the loop) ─────────────────────────────────
# Use the last (fullest) combination to get block_ids — they don't depend on
# which variables are included, only on the data file and spatial resolution.
cfg$lag_vars      <- combinations[[length(combinations)]]$lag
cfg$unlagged_vars <- combinations[[length(combinations)]]$unlag
cfg$numeric_vars  <- cfg$lag_vars[!grepl("_cat$", cfg$lag_vars)]
prep_init <- build_stan_data(cfg)
block_ids <- sort(unique(as.character(prep_init$df[[cfg$block_col]])))

sf_blocks <- if (spatial_level == "CMF") {
  st_read(file.path(cfg$shapefile_path, "CMF",
                    "Poligonos CMF Cienfuegos_28032025.shp"), quiet = TRUE) %>%
    mutate(Area_CMF = paste(AS, CMF, sep = "_"))
} else {
  st_read(file.path(cfg$shapefile_path, "Manzanas_cleaned_05032026",
                    "Mz_CMF_Correcto_2022026.shp"), quiet = TRUE)
}
pts <- suppressWarnings(st_point_on_surface(sf_blocks))
if (st_is_longlat(pts)) pts <- st_transform(pts, 3857)
coords_sf <- sf_blocks %>%
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

icar_edges <- build_icar_edges(sf_blocks, block_ids, cfg$sf_block_col, snap_m = 100)
cat(sprintf("ICAR: %d blocks, %d unique edges\n", length(block_ids), icar_edges$N_edges))

# ── Compile Stan model once ───────────────────────────────────────────────────
# K and Ku are passed as data, not hard-coded in the Stan file, so the same
# compiled binary is valid for all variable combinations.
mod <- cmdstan_model(cfg$stan_file, force_recompile = hostname == "frietjes")
cat("Stan model compiled.\n")

# ── Sweep log initialisation ──────────────────────────────────────────────────
sweep_log <- vector("list", length(combinations))

# ── Main sweep loop ───────────────────────────────────────────────────────────
for (combo_i in seq_along(combinations)) {
  combo <- combinations[[combo_i]]
  cat(sprintf(
    "\n========== Run %d / %d ==========\n  lag:   %s\n  unlag: %s\n",
    combo_i, length(combinations),
    paste(combo$lag, collapse = ", "),
    if (length(combo$unlag) == 0) "(none)" else paste(combo$unlag, collapse = ", ")
  ))

  cfg$lag_vars      <- combo$lag
  cfg$unlagged_vars <- combo$unlag
  cfg$numeric_vars  <- combo$lag[!grepl("_cat$", combo$lag)]

  # predictor_spec encodes the variable set → unique output dir per combo
  predictor_spec   <- paste0(
    "lag-",    paste(cfg$lag_vars,      collapse = "-"),
    "_unlag-", paste(cfg$unlagged_vars, collapse = "-")
  )
  run_output_dir   <- file.path(cfg$output_dir, predictor_spec, model_spec, date_suffix)
  plots_output_dir <- file.path(run_output_dir, "plots")
  dir.create(run_output_dir,   recursive = TRUE, showWarnings = FALSE)
  dir.create(plots_output_dir, recursive = TRUE, showWarnings = FALSE)

  t_start <- proc.time()["elapsed"]

  result <- tryCatch({

    # Rebuild design matrices for this combo (fast; spatial setup already done)
    prep      <- build_stan_data(cfg)
    stan_data <- prep$stan_data
    df        <- prep$df

    # Inject pre-computed ICAR edges
    stan_data$N_edges <- icar_edges$N_edges
    stan_data$node1   <- icar_edges$node1
    stan_data$node2   <- icar_edges$node2

    # phi setup
    stan_data$fix_phi <- as.integer(isTRUE(cfg$fix_phi))
    if (isTRUE(cfg$fix_phi)) {
      disp_df <- df %>%
        mutate(n_bt = stan_data$n_bt, y_bt = stan_data$y) %>%
        filter(n_bt > 0) %>%
        mutate(y_rate = y_bt / n_bt)
      phi_grouped <- disp_df %>%
        group_by(n_bt) %>%
        filter(n() >= 30) %>%
        summarise(
          p_bar            = mean(y_rate),
          var_observed     = var(y_rate),
          var_binomial     = p_bar * (1 - p_bar) / first(n_bt),
          dispersion_ratio = var_observed / var_binomial,
          phi_implied      = (first(n_bt) - dispersion_ratio) / (dispersion_ratio - 1),
          .groups          = "drop"
        ) %>%
        filter(dispersion_ratio > 1, phi_implied > 0) %>%
        summarise(phi_median = median(phi_implied))
      phi_grouped_median <- phi_grouped$phi_median
      phi_use <- if (is.finite(phi_grouped_median) && phi_grouped_median > 0 &&
                     phi_grouped_median < 500) {
        round(phi_grouped_median, 1)
      } else {
        cfg$phi_fixed
      }
      stan_data$phi_data <- phi_use
      cat(sprintf("phi fixed at %.1f\n", phi_use))
    } else {
      stan_data$phi_data <- 1.0
      cat("phi will be estimated\n")
    }

    # MCMC
    fit <- mod$sample(
      data            = stan_data,
      chains          = cfg$chains,
      iter_warmup     = cfg$iter_warmup,
      iter_sampling   = cfg$iter_sampling,
      thin            = if (!is.null(cfg$thin)) cfg$thin else 1,
      init            = make_init_fun(
        stan_data,      cfg$use_temporal_AR,
        use_hsgp        = isTRUE(cfg$use_hsgp) && !isTRUE(cfg$use_icar) && !isTRUE(cfg$use_bym2),
        use_icar        = isTRUE(cfg$use_icar) && !isTRUE(cfg$use_bym2),
        use_bym2        = isTRUE(cfg$use_bym2),
        use_time_RE     = isTRUE(cfg$use_time_RE),
        use_spatial_AC  = isTRUE(cfg$use_spatial_AC),
        use_block_dev   = isTRUE(cfg$use_block_dev)
      ),
      adapt_delta     = cfg$adapt_delta,
      max_treedepth   = cfg$max_treedepth,
      parallel_chains = cfg$parallel_chains,
      output_dir      = run_output_dir
    )
    invisible(file.remove(list.files(run_output_dir,
                                     pattern = "_(config|metric)\\.json$",
                                     full.names = TRUE)))

    # Model summary
    summary_vars <- c("alpha", "delta1", "w", "w_unlagged")
    if (isTRUE(cfg$use_time_RE)) {
      summary_vars <- c(summary_vars, "sigma_time", "sigma_block")
    } else {
      if (isTRUE(cfg$use_spatial_AC)) {
        if (isTRUE(cfg$use_bym2))      summary_vars <- c(summary_vars, "sigma_spatial", "phi_mix")
        else if (isTRUE(cfg$use_icar)) summary_vars <- c(summary_vars, "sigma_icar")
        else                            summary_vars <- c(summary_vars, "sigma_gp", "rho_gp")
      }
      if (isTRUE(cfg$use_temporal_AR)) summary_vars <- c(summary_vars, "sigma_v", "rho")
      if (!isTRUE(cfg$use_bym2) && isTRUE(cfg$use_block_dev))
        summary_vars <- c(summary_vars, "sigma_block_dev")
    }
    if (!isTRUE(cfg$fix_phi)) summary_vars <- c(summary_vars, "phi")
    summary_output <- capture.output(print(fit$summary(variables = summary_vars), n = Inf))
    writeLines(summary_output,
               file.path(run_output_dir, paste0("model_summary_", model_spec, ".txt")))

    # Posterior draws for plotting
    post <- extract_means(fit, nrow(df))
    df$n_bt            <- stan_data$n_bt
    df$fitted_p_bt     <- post$p_bt
    df$observed_p_bt   <- df$y_bt / df$n_bt
    y_pred_mat         <- fit$draws("y_pred", format = "matrix")
    df$y_pred_rate     <- colMeans(y_pred_mat) / df$n_bt
    df$y_pred_rate_q05 <- apply(y_pred_mat, 2, quantile, probs = 0.05) / df$n_bt
    df$y_pred_rate_q95 <- apply(y_pred_mat, 2, quantile, probs = 0.95) / df$n_bt
    rm(y_pred_mat)

    # Random effects plot
    if (cfg$plot_random_effects) {
      if (!all(is.na(post$u)) && length(post$u) > 0) {
        save_random_effects(post$u, post$v, plots_output_dir, model_spec)
      } else if (!all(is.na(post$v)) && length(post$v) > 0) {
        png(file.path(plots_output_dir,
                      paste0("random_effects_temporal_only_", model_spec, ".png")),
            width = 800, height = 600)
        par(mfrow = c(1, 2))
        plot(post$v, type = "b", main = "Temporal Random Effects",
             xlab = "Time", ylab = "Effect", col = "red", pch = 19)
        abline(h = 0, lty = 2, col = "gray")
        acf(post$v, main = "ACF of Temporal Effects", col = "darkred")
        par(mfrow = c(1, 1))
        dev.off()
      }
    }
    if (cfg$plot_ppc)        save_ppc(df, fit, plots_output_dir, model_spec)
    if (cfg$plot_timeseries) save_timeseries_plots(df, plots_output_dir, model_spec,
                                                    cfg$n_blocks_facet)

    # Traceplots
    if (cfg$plot_traceplots && requireNamespace("bayesplot", quietly = TRUE)) {
      library(bayesplot)
      model_vars     <- fit$metadata()$stan_variables
      scalar_include <- c("alpha", "sigma_gp", "rho_gp", "sigma_icar",
                          "sigma_spatial", "phi_mix", "delta1",
                          "sigma_v", "rho", "sigma_block_dev",
                          "sigma_time", "sigma_block", "phi")
      scalar_vars    <- intersect(scalar_include, model_vars)
      save_trace_chunks <- function(vars, draws_arr, file_prefix, chunk_size = 12, w, h) {
        chunks <- split(vars, ceiling(seq_along(vars) / chunk_size))
        for (i in seq_along(chunks))
          ggsave(
            file.path(plots_output_dir,
                      paste0(file_prefix, "_part", i, "_", model_spec, ".png")),
            mcmc_trace(draws_arr, pars = chunks[[i]]), width = w, height = h
          )
      }
      if (length(scalar_vars) > 0) {
        draws_scalar  <- fit$draws(variables = scalar_vars, format = "array")
        scalar_params <- dimnames(draws_scalar)[[3]]
        save_trace_chunks(scalar_params, draws_scalar, "traceplot_params", w = 10, h = 8)
      }
      if ("w" %in% model_vars) {
        draws_w <- fit$draws(variables = "w", format = "array")
        save_trace_chunks(dimnames(draws_w)[[3]], draws_w,
                          "traceplot_weights_w", w = 12, h = 10)
      }
      if ("w_unlagged" %in% model_vars) {
        draws_wu <- fit$draws(variables = "w_unlagged", format = "array")
        save_trace_chunks(dimnames(draws_wu)[[3]], draws_wu,
                          "traceplot_weights_unlagged", w = 12, h = 8)
      }
    }

    # Delete chain CSVs — plots and summary already saved, raw draws not needed
    invisible(file.remove(list.files(run_output_dir,
                                     pattern = "\\.csv$",
                                     full.names = TRUE)))

    cat("Run", combo_i, "complete. Output:", run_output_dir, "\n")
    "success"

  }, error = function(e) {
    cat("ERROR in run", combo_i, ":", conditionMessage(e), "\n")
    "error"
  })

  elapsed_min <- round((proc.time()["elapsed"] - t_start) / 60, 1)
  sweep_log[[combo_i]] <- data.frame(
    run         = combo_i,
    lag_vars    = paste(combo$lag,   collapse = "|"),
    unlag_vars  = paste(combo$unlag, collapse = "|"),
    n_lag       = length(combo$lag),
    n_unlag     = length(combo$unlag),
    status      = result,
    elapsed_min = elapsed_min
  )
}

# ── Write sweep log ───────────────────────────────────────────────────────────
sweep_log_df <- do.call(rbind, sweep_log)
log_path     <- file.path(cfg$output_dir,
                           paste0("variable_sweep_log_", date_suffix, ".csv"))
write.csv(sweep_log_df, log_path, row.names = FALSE)
cat("\nSweep complete. Log written to:", log_path, "\n")
print(sweep_log_df)
