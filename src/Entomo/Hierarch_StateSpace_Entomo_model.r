# =====================================================
# Mosquito hierarchical model with reactive surveillance
# Clean calibration script (function-based)
# =====================================================

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

# =========================
# 0) LOAD HELPER FUNCTIONS
# =========================
script_dir <- tryCatch({
  # RStudio / Positron via rstudioapi
  p <- rstudioapi::getActiveDocumentContext()$path
  if (nzchar(p)) dirname(p) else stop("empty path")
}, error = function(e) tryCatch({
  # When run via source() or Rscript --file=
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
  # Last resort: find helper_functions.r relative to the working directory
  candidate <- file.path(getwd(), "src", "Entomo")
  if (file.exists(file.path(candidate, "helper_functions.r"))) candidate else getwd()
}))

renv::restore(project = script_dir, prompt = FALSE)

source(file.path(script_dir, "helper_functions.r"))
source(file.path(script_dir, "plot_functions.r"))

# =========================
# 1) SETTINGS
# =========================
hostname <- Sys.info()["nodename"]


# ========== Spatial resolution =============
# Set to "CMF" or "manzana" — all level-specific paths and column names derive from this.
spatial_level <- "CMF" 

# ========== Output structure and config =============
cfg <- list(
  data_dir = if (hostname == "frietjes") "~/data/Entomo" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data",
  # Extended-lag dataset: env 2015-2019 (NDVI/NDMI/NDWI observed 2016-2019, climatology-backfilled for 2015), ento-epi 2016-2019.
  # 2015 rows serve as lag lead-in; response_start below marks the observation period.
  data_file_name = if (spatial_level == "CMF")"env_epi_entomo_data_per_CMF_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv" else "env_epi_entomo_data_per_manzana_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv",
  output_dir = if (hostname == "frietjes") "/home/rita/data/Entomo/fitting/stan" else "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan",

  # model variant
  use_time_RE          = FALSE,  # TRUE = iid time RE + iid block RE (no AR1, no GP); overrides others
  use_temporal_AR      = TRUE,  # (ignored if use_time_RE = TRUE) TRUE = single global AR1 trend
  use_temporal_AR_perCMF = TRUE, # (ignored if use_time_RE = TRUE) TRUE = independent AR1 per CMF
  use_spatial_AC  = FALSE,    # (ignored if use_time_RE = TRUE) TRUE = spatial AC
  use_hsgp        = FALSE,   # (only if use_spatial_AC = TRUE and use_icar/bym2 = FALSE) TRUE = HSGP
  use_icar        = FALSE,   # (only if use_spatial_AC = TRUE) TRUE = plain ICAR
  use_bym2        = FALSE,    # (only if use_spatial_AC = TRUE) TRUE = BYM2 (structured+unstructured); overrides use_icar
  hsgp_m          = 20,     # basis functions per dimension (20 → 400 total)
  hsgp_c          = 1.5,    # boundary factor (domain = c * data range)
  use_block_dev   = TRUE,   # (ignored if use_time_RE = TRUE) TRUE = per-block deviation
  use_dlnm        = TRUE,  # TRUE = replace lag flat matrix with DLNM cross-basis (blockRE only for now)

  # spatial
  shapefile_path = if (hostname == "frietjes")
    "/home/rita/data/Entomo"
  else
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders",
  sf_block_col = if (spatial_level == "CMF") "Area_CMF" else "CODIGO_",
  block_col    = if (spatial_level == "CMF") "cmf"      else "manzana",

  # data prep
  # response_start: marks the start of the ento response period. 2015 rows are
  # used purely as lag lead-in and are never passed to Stan as observations.
  response_start = "2016_01",
  n_blocks = NULL, # set NULL for all blocks/CMFs

  lag_vars = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp"),
  # lag_vars = c("total_rainy_days", "avg_VPD"),

  max_lag = 6,
  kappa = 4,

  # unlagged_vars = c("is_urban", "is_WUI", "is_WI", "has_aljibes", "water_containers", "water_shortage"),
  unlagged_vars = c("HFP_urbanization", "is_WUI", "is_WI", "water_containers", "mean_ndvi"),

  numeric_vars = c("total_precip",  "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "mean_ndvi", "HFP_urbanization"),

  # DLNM settings (only used when use_dlnm = TRUE)
  dlnm_vars   = c("total_precip",  "avg_VPD", "precip_max_day_resid_on_tp"),

  dlnm_argvar = list(
    total_precip                = list(fun = "ns", df = 3),
    avg_VPD                     = list(fun = "ns", df = 3),
    precip_max_day_resid_on_tp = list(fun = "ns", df = 3)
    # avg_temp                     = list(fun = "ns", df = 3)
  
  ),
  dlnm_arglag = list(fun = "ns", df = 3),  # shared lag basis across all DLNM vars

  # Interaction cross-bases: each entry is (binary_var, active_level, dlnm_var,
  # label) - a 0/1 indicator modifier, active where binary_var == active_level
  # (e.g. is_urban coded 1=urban (reference), 0=non-urban -> active_level=0
  # for the non-urban modifier). Continuous modifiers are not supported (see
  # helper_functions.r::build_dlnm_stan_data() - a linear-in-modifier tilt
  # evaluated only at mean/+1 SD is hard to interpret as anything but an
  # arbitrary two-point probe of a continuous effect-modification surface).
  # Set dlnm_ix_vars = NULL to run the base DLNM model without interactions.
  dlnm_ix_vars = list(
    list(binary_var = "is_urban", active_level = 0, dlnm_var = "total_precip", label = "nonurban_x_tp")
  ),
  # dlnm_ix_vars = NULL,

  # MCMC
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1500,
  adapt_delta = 0.97, # target average acceptance probability for the NUTS sampler in stan
  max_treedepth = 10, # caps how many steps the NUTS sampler can take in a single iteration.
  parallel_chains = if (hostname == "frietjes") 4 else 1,

  # delta1: set fix_delta1 = TRUE to fix the reactive detection boost at delta1_fixed value.
  # Use delta1_fixed = 0 to disable reactive detection enhancement entirely.
  # Only applies to the DLNM + perCMF + blockRE variant (selects the _delta1fixed Stan file).
  fix_delta1  = FALSE,
  delta1_fixed = 0,

  # phi: set fix_phi = TRUE to pass phi as data (fixed); FALSE to estimate it
  fix_phi = FALSE,
  phi_fixed = 25,   # beta-binomial concentration -> later replace with gamma(2, 0.25)
  # prior predictive check (set TRUE before first real fit)
  run_prior_predictive = FALSE,

  # outputs (individual plot toggles)
  plot_traceplots = TRUE,
  plot_random_effects = TRUE,
  plot_ppc = TRUE,
  plot_timeseries = TRUE,
  plot_morans_I = TRUE,   # FALSE to skip (memory-heavy: loads full y_pred matrix)
  n_blocks_facet = 9
)

# Allow a calling script to inject cfg overrides before the model runs.
# Set .hierarch_cfg_override <- list(...) before source()-ing this script.
if (exists(".hierarch_cfg_override") && is.list(.hierarch_cfg_override))
  cfg <- modifyList(cfg, .hierarch_cfg_override)

# ========== Output directory structure =============
date_suffix <- format(Sys.Date(), "%Y%m%d")
model_spec <- if (isTRUE(cfg$use_time_RE)) {
  paste0("timeRE_blockRE_lag", cfg$max_lag, "_k", cfg$kappa)
} else if (isTRUE(cfg$use_dlnm)) {
  ar1_suffix     <- if (isTRUE(cfg$use_temporal_AR_perCMF)) "AR1perCMF"
                    else if (isTRUE(cfg$use_temporal_AR))   "AR1global"
                    else                                     "noAR1"
  sp_suffix      <- if (isTRUE(cfg$use_icar)) "ICAR"
                    else if (isTRUE(cfg$use_bym2)) "BYM2"
                    else "noGP"
  re_suffix      <- if (isTRUE(cfg$use_icar)) "ICAR"
                    else if (isTRUE(cfg$use_block_dev)) "blockRE"
                    else "noBlockRE"
  n_block_suffix <- paste0(ifelse(is.null(cfg$n_blocks), "All", cfg$n_blocks), "Blocks")
  paste0("DLNM_", ar1_suffix, "_", sp_suffix, "_", re_suffix,
         "_lag", cfg$max_lag, "_k", cfg$kappa, "_", n_block_suffix)
} else {
  ar1_suffix <- if (isTRUE(cfg$use_temporal_AR_perCMF)) "AR1perCMF"
               else if (isTRUE(cfg$use_temporal_AR))      "AR1global"
               else                                        "noAR1"
  gp_suffix  <- if (!isTRUE(cfg$use_spatial_AC))  "noGP"
                else if (isTRUE(cfg$use_bym2))    "BYM2"
                else if (isTRUE(cfg$use_icar))    "ICAR"
                else if (isTRUE(cfg$use_hsgp))    "HSGP"
                else                              "GP"
  re_suffix  <- if (isTRUE(cfg$use_temporal_AR_perCMF) && isTRUE(cfg$use_block_dev)) "blockRE"
               else if (isTRUE(cfg$use_temporal_AR_perCMF)) "noBlockRE"
               else ifelse(isTRUE(cfg$use_block_dev), "blockRE", "noBlockRE")
  n_block_suffix  <- ifelse(is.null(cfg$n_blocks), "All", cfg$n_blocks)
  n_block_suffix  <- paste0(n_block_suffix, "Blocks")
  paste0(ar1_suffix, "_", gp_suffix, "_", re_suffix,
         "_lag", cfg$max_lag, "_k", cfg$kappa, "_", n_block_suffix)
}
model_spec <- paste0(spatial_level, "_", model_spec)
if (isTRUE(cfg$fix_delta1)) {
  model_spec <- paste0(model_spec, "_delta1fix", cfg$delta1_fixed)
}
predictor_spec <- if (isTRUE(cfg$use_dlnm)) {
  base_spec <- paste0("dlnm-", paste(cfg$dlnm_vars, collapse = "-"),
                      "_unlag-", paste(cfg$unlagged_vars, collapse = "-"))
  if (!is.null(cfg$dlnm_ix_vars) && length(cfg$dlnm_ix_vars) > 0) {
    ix_labels <- sapply(cfg$dlnm_ix_vars, function(ix) ix$label)
    paste0(base_spec, "_ix-", paste(ix_labels, collapse = "-"))
  } else {
    base_spec
  }
} else {
  paste0("lag-", paste(cfg$lag_vars, collapse = "-"),
         "_unlag-", paste(cfg$unlagged_vars, collapse = "-"))
}
run_suffix <- paste0(date_suffix)
if (exists(".hierarch_run_suffix")) run_suffix <- .hierarch_run_suffix

model_output_dir  <- file.path(cfg$output_dir, predictor_spec, model_spec)
run_output_dir    <- file.path(model_output_dir, run_suffix)
plots_output_dir  <- file.path(run_output_dir, "plots")
dir.create(run_output_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(plots_output_dir, recursive = TRUE, showWarnings = FALSE)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)

# ================= selection stan file ============================
stan_dir <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo"
cfg$stan_file <- if (isTRUE(cfg$use_time_RE)) {
  # iid time RE + iid block RE (no AR1, no GP)
  file.path(stan_dir, "hierarchical_state_space_timeRE_blockRE.stan")
} else if (isTRUE(cfg$use_dlnm)) {
  # DLNM cross-basis: per-CMF AR(1) variants
  if (isTRUE(cfg$use_temporal_AR_perCMF) && isTRUE(cfg$use_icar)) {
    file.path(stan_dir, "hierarchical_state_space_AR_perCMF_ICAR_DLNM.stan")
  } else if (isTRUE(cfg$use_temporal_AR_perCMF) && isTRUE(cfg$use_block_dev)) {
    if (isTRUE(cfg$fix_delta1))
      file.path(stan_dir, "hierarchical_state_space_AR_perCMF_blockRE_DLNM_delta1fixed.stan")
    else if (!is.null(cfg$dlnm_ix_vars) && length(cfg$dlnm_ix_vars) > 0)
      file.path(stan_dir, "hierarchical_state_space_AR_perCMF_blockRE_DLNM_ix.stan")
    else
      file.path(stan_dir, "hierarchical_state_space_AR_perCMF_blockRE_DLNM.stan")
  } else {
    file.path(stan_dir, "hierarchical_state_space_blockRE_DLNM.stan")
  }
} else if (!isTRUE(cfg$use_temporal_AR) && !isTRUE(cfg$use_temporal_AR_perCMF) && !isTRUE(cfg$use_spatial_AC)) {
  # No AR, no GP: base or blockRE-only
  if (isTRUE(cfg$use_block_dev)) {
    file.path(stan_dir, "hierarchical_state_space_blockRE.stan")
  } else {
    file.path(stan_dir, "hierarchical_state_space.stan")
  }
} else if (!isTRUE(cfg$use_spatial_AC)) {
  # AR only variants (no GP)
  if (isTRUE(cfg$use_temporal_AR_perCMF) && isTRUE(cfg$use_block_dev)) {
    file.path(stan_dir, "hierarchical_state_space_AR_perCMF_blockRE.stan")
  } else if (isTRUE(cfg$use_temporal_AR_perCMF)) {
    file.path(stan_dir, "hierarchical_state_space_AR_perCMF.stan")
  } else if (isTRUE(cfg$use_block_dev)) {
    file.path(stan_dir, "hierarchical_state_space_AR_blockRE.stan")
  } else {
    file.path(stan_dir, "hierarchical_state_space_AR.stan")
  }
} else if (isTRUE(cfg$use_bym2)) {
  # BYM2: structured (ICAR) + unstructured spatial RE, single sigma_spatial
  file.path(stan_dir, "hierarchical_state_space_AR_BYM2.stan")
} else if (isTRUE(cfg$use_icar)) {
  # Plain ICAR: per-CMF AR, or global AR with/without per-block temporal deviation
  if (isTRUE(cfg$use_temporal_AR_perCMF)) {
    file.path(stan_dir, "hierarchical_state_space_AR_perCMF_ICAR.stan")
  } else if (isTRUE(cfg$use_block_dev)) {
    file.path(stan_dir, "hierarchical_state_space_AR_blockRE_ICAR.stan")
  } else {
    file.path(stan_dir, "hierarchical_state_space_AR_ICAR_noBlockDev.stan")
  }
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
cat("Model variant:" , "Predictors: ", predictor_spec, "\n","level, RE and AR: ", model_spec, "\n")


# =========================
# STANDARDIZE NUMERIC COVARIATES - done in helper functions
# =========================

prep <- if (isTRUE(cfg$use_dlnm)) build_dlnm_stan_data(cfg) else build_stan_data(cfg)
stan_data <- prep$stan_data
df <- prep$df


# =========================
# SPATIAL DISTANCE MATRIX
# =========================
sf_blocks <- if (spatial_level == "CMF") {
  st_read(file.path(cfg$shapefile_path, "CMF", "Poligonos CMF Cienfuegos_28032025.shp"), quiet = TRUE) %>%
    mutate(Area_CMF = paste(AS, CMF, sep = "_"))
} else {
  st_read(file.path(cfg$shapefile_path, "Manzanas_cleaned_05032026", "Mz_CMF_Correcto_2022026.shp"), quiet = TRUE)
}
# transform to meters 
pts        <- suppressWarnings(st_point_on_surface(sf_blocks))
if (st_is_longlat(pts)) pts <- st_transform(pts, 3857)

block_ids  <- sort(unique(as.character(df[[cfg$block_col]])))  # ordered to match block index
coords_sf  <- sf_blocks %>%
  st_drop_geometry() %>%
  mutate(
    block_chr = as.character(.data[[cfg$sf_block_col]]),
    x = st_coordinates(pts)[, 1],
    y = st_coordinates(pts)[, 2]
  ) %>%
  filter(block_chr %in% block_ids) %>% # if sampling for instance 100 blocks then needed 
  arrange(match(block_chr, block_ids)) %>%
  select(block_chr, x, y) %>%
  distinct(block_chr, .keep_all = TRUE)

# ======================== spatial data prep ====================================
# use_icar can also be set without use_spatial_AC when combined with use_dlnm
if (isTRUE(cfg$use_spatial_AC) || isTRUE(cfg$use_icar)) {
  if (isTRUE(cfg$use_bym2) || isTRUE(cfg$use_icar)) {
    icar_edges <- build_icar_edges(sf_blocks, block_ids, cfg$sf_block_col, snap_m = 100)
    stan_data$N_edges <- icar_edges$N_edges
    stan_data$node1   <- icar_edges$node1
    stan_data$node2   <- icar_edges$node2
    cat(sprintf("%s: %d blocks, %d unique edges\n",
                ifelse(isTRUE(cfg$use_bym2), "BYM2", "ICAR"),
                length(block_ids), stan_data$N_edges))
    if (isTRUE(cfg$use_bym2)) {
      stan_data$scaling_factor <- compute_bym2_scaling(
        icar_edges$node1, icar_edges$node2, stan_data$B
      )
    }

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
  cat("No spatial AC or ICAR: skipping coordinate/distance/neighbour setup.\n")
}

# #================= check adjacency matrix (optional)==============
# # Build adjacency matrix from ICAR edges
# B     <- stan_data$B
# node1 <- stan_data$node1
# node2 <- stan_data$node2

# adj <- matrix(0L, nrow = B, ncol = B)
# for (k in seq_along(node1)) {
#   adj[node1[k], node2[k]] <- 1L
#   adj[node2[k], node1[k]] <- 1L
# }

# # Label rows/cols with block IDs
# dimnames(adj) <- list(block_ids, block_ids)

# # Print a corner (first 10 blocks) to check
# print(adj[1:min(10, B), 1:min(10, B)])

# # Summary stats
# cat("Blocks:", B, "\n")
# cat("Edges:", length(node1), "\n")
# cat("Neighbours per block (min/median/max):",
#     min(rowSums(adj)), median(rowSums(adj)), max(rowSums(adj)), "\n")
# cat("Islands (0 neighbours):", sum(rowSums(adj) == 0), "\n")

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

if (isTRUE(cfg$fix_phi)) {
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
}

       
# Pass delta1 as data when fix_delta1 = TRUE (uses the _delta1fixed Stan file)
if (isTRUE(cfg$fix_delta1)) {
  stan_data$delta1 <- cfg$delta1_fixed
  cat(sprintf("delta1 fixed at %.4f (reactive detection boost disabled)\n", cfg$delta1_fixed))
}

# Always pass fix_phi flag; pass phi_data (used only when fix_phi = TRUE)
stan_data$fix_phi <- as.integer(isTRUE(cfg$fix_phi))
if (isTRUE(cfg$fix_phi)) {
  phi_grouped_median <- phi_grouped$phi_median
  phi_use <- if (is.finite(phi_grouped_median) && phi_grouped_median > 0 && phi_grouped_median < 500) {
    round(phi_grouped_median, 1)
  } else {
    cfg$phi_fixed
  }
  stan_data$phi_data <- phi_use
  cat(sprintf(
    "phi fixed at %.1f (grouped median across n_bt bins; cfg fallback = %.1f)\n",
    phi_use, cfg$phi_fixed
  ))
  cfg$phi_used <- phi_use
} else {
  stan_data$phi_data <- 1.0  # dummy; Stan uses phi_raw (estimated) instead
  cat("phi will be estimated from data (fix_phi = FALSE)\n")
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
              paste0("prior_predictive_", model_spec, ".png")),
    p_prior, width = 8, height = 5, dpi = 150
  )
  cat("Prior predictive plot saved to:", cfg$output_dir, "\n")
  stop("Prior predictive check complete. Set run_prior_predictive = FALSE to proceed with the real fit.")
}

# ========================= MCMC sampling =======================================
# If a previous run already wrote complete chain CSVs to run_output_dir (e.g. you're
# re-sourcing this script only to regenerate post-fit plots), reload them instead of
# re-sampling. Set .hierarch_run_suffix before source()-ing to point at that run's
# date folder. NB: this only checks the file count, not chain health -- if a prior
# run crashed mid-sampling, delete its partial CSVs first to force a fresh fit.
existing_csv <- list.files(
  run_output_dir,
  pattern = paste0("^", tools::file_path_sans_ext(basename(cfg$stan_file)), "-.*\\.csv$"),
  full.names = TRUE
)

if (length(existing_csv) >= cfg$chains) {
  cat(sprintf("Found %d existing chain CSV(s) in %s; reloading fit instead of re-sampling.\n",
              length(existing_csv), run_output_dir))
  fit <- as_cmdstan_fit(existing_csv)
} else {
  fit <- mod$sample(
    data = stan_data,
    chains = cfg$chains,
    iter_warmup = cfg$iter_warmup,
    iter_sampling = cfg$iter_sampling,
    thin = if (!is.null(cfg$thin)) cfg$thin else 1,
    init = make_init_fun(
      stan_data, cfg$use_temporal_AR,
      use_hsgp              = isTRUE(cfg$use_hsgp) && !isTRUE(cfg$use_icar) && !isTRUE(cfg$use_bym2),
      use_icar              = isTRUE(cfg$use_icar) && !isTRUE(cfg$use_bym2),
      use_bym2              = isTRUE(cfg$use_bym2),
      use_time_RE           = isTRUE(cfg$use_time_RE),
      use_spatial_AC        = isTRUE(cfg$use_spatial_AC),
      use_block_dev         = isTRUE(cfg$use_block_dev),
      use_temporal_AR_perCMF = isTRUE(cfg$use_temporal_AR_perCMF),
      use_dlnm              = isTRUE(cfg$use_dlnm)
    ),
    adapt_delta = cfg$adapt_delta,
    max_treedepth = cfg$max_treedepth,
    parallel_chains = cfg$parallel_chains,
    output_dir = run_output_dir   # write chain CSVs here instead of /tmp
  )
}


# CSV chain files are already in run_output_dir; skip .rds to save disk space
# (re-load later with: fit <- as_cmdstan_fit(list.files(run_output_dir, "*.csv", full.names=TRUE)))
# Remove CmdStan auxiliary files (config/metric JSONs) — not needed for post-processing
invisible(file.remove(list.files(run_output_dir, pattern = "_(config|metric)\\.json$", full.names = TRUE)))

# ======================= make model summary ============================
summary_vars <- c("alpha", if (!isTRUE(cfg$fix_delta1)) "delta1", if (isTRUE(cfg$use_dlnm)) "w_cb" else "w", "w_unlagged",
                  if (isTRUE(cfg$use_dlnm) && !is.null(cfg$dlnm_ix_vars) && length(cfg$dlnm_ix_vars) > 0) "w_ix")
if (isTRUE(cfg$use_time_RE)) {
  summary_vars <- c(summary_vars, "sigma_time", "sigma_block")
} else {
  if (isTRUE(cfg$use_spatial_AC)) {
    if (isTRUE(cfg$use_bym2))    summary_vars <- c(summary_vars, "sigma_spatial", "phi_mix")
    else if (isTRUE(cfg$use_icar)) summary_vars <- c(summary_vars, "sigma_icar")
    else                           summary_vars <- c(summary_vars, "sigma_gp", "rho_gp")
  }
  if (isTRUE(cfg$use_temporal_AR) || isTRUE(cfg$use_temporal_AR_perCMF))
    summary_vars <- c(summary_vars, "tau", "sigma_v", "rho")
  # DLNM+ICAR: sigma_icar not captured by use_spatial_AC branch above
  if (isTRUE(cfg$use_dlnm) && isTRUE(cfg$use_icar))
    summary_vars <- c(summary_vars, "sigma_icar")
  if (!isTRUE(cfg$use_bym2) && !isTRUE(cfg$use_icar) && isTRUE(cfg$use_block_dev)) {
    if (isTRUE(cfg$use_temporal_AR) && !isTRUE(cfg$use_temporal_AR_perCMF))
      summary_vars <- c(summary_vars, "sigma_block_dev")
    else
      summary_vars <- c(summary_vars, "sigma_block")
  }
}
if (!isTRUE(cfg$fix_phi)) summary_vars <- c(summary_vars, "phi")

# ======================= model selection criteria (LOO / WAIC / log-lik) ====
# Runs before the summary capture.output block so loo_result is always saved
# to disk (and available in globalenv) even if the summary section errors.
if (requireNamespace("loo", quietly = TRUE)) {
  log_lik_draws <- fit$draws("log_lik", format = "array")
  log_lik_mat   <- fit$draws("log_lik", format = "matrix")

  r_eff      <- loo::relative_eff(exp(log_lik_draws))
  loo_result <- loo::loo(log_lik_draws, r_eff = r_eff)
  waic_result <- loo::waic(log_lik_mat)

  draw_llik   <- rowSums(log_lik_mat)
  llik_summary <- c(
    mean   = mean(draw_llik),
    sd     = sd(draw_llik),
    q5     = quantile(draw_llik, 0.05),
    median = median(draw_llik),
    q95    = quantile(draw_llik, 0.95)
  )

  criteria_output <- capture.output({
    cat("Model:", model_spec, "\n")
    cat("Run:  ", run_suffix,  "\n\n")

    cat("=== LOO-CV ===\n")
    print(loo_result)

    cat("\n=== WAIC ===\n")
    print(waic_result)

    cat("\n=== Total log-likelihood (sum over observations, posterior draws) ===\n")
    cat(sprintf("  Mean   : %.2f\n", llik_summary["mean"]))
    cat(sprintf("  SD     : %.2f\n", llik_summary["sd"]))
    cat(sprintf("  5%%     : %.2f\n", llik_summary["q5.5%"]))
    cat(sprintf("  Median : %.2f\n", llik_summary["median"]))
    cat(sprintf("  95%%    : %.2f\n", llik_summary["q95.95%"]))
  })

  cat(paste(criteria_output, collapse = "\n"), "\n")

  crit_file <- file.path(run_output_dir,
                         paste0("model_selection_criteria_", model_spec, ".txt"))
  writeLines(criteria_output, crit_file)
  saveRDS(loo_result,  file.path(run_output_dir, paste0("loo_",  model_spec, ".rds")))
  saveRDS(waic_result, file.path(run_output_dir, paste0("waic_", model_spec, ".rds")))
  cat("Model selection criteria saved to:", run_output_dir, "\n")
} else {
  cat("Package 'loo' not installed; skipping model selection criteria.\n")
}

model_sum      <- rename_w_in_summary(
  fit$summary(variables = summary_vars),
  lag_vars_expanded = if (isTRUE(cfg$use_dlnm)) NULL else prep$lag_vars_expanded,
  unlagged_vars     = prep$unlagged_vars
)
summary_output <- capture.output({
  cat("=== MODEL CONFIGURATION ===\n")
  cat("Run suffix  :", run_suffix, "\n")
  cat("Stan file   :", cfg$stan_file, "\n")
  cat("Spatial level:", cfg$spatial_level, "\n\n")

  cat("--- Predictors ---\n")
  cat("Lag vars    :", paste(cfg$lag_vars, collapse = ", "), "\n")
  cat("Unlagged    :", paste(cfg$unlagged_vars, collapse = ", "), "\n")
  cat("Max lag     :", cfg$max_lag, "\n")
  if (isTRUE(cfg$use_dlnm)) {
    argspec_str <- function(s) {
      if (!is.list(s) || is.null(s$fun)) return("?")
      if (s$fun == "lin")    return("lin")
      if (s$fun == "strata") return(paste0("strata(", s$breaks, ")"))
      paste0(s$fun, "(df=", s$df, ")")
    }
    cat("DLNM argvar :", paste(names(cfg$dlnm_argvar), sapply(cfg$dlnm_argvar, argspec_str), sep = "=", collapse = ", "), "\n")
    arglag_is_per_var <- !is.null(names(cfg$dlnm_arglag)) && any(names(cfg$dlnm_arglag) %in% cfg$dlnm_vars)
    if (arglag_is_per_var) {
      cat("DLNM arglag : per-variable —", paste(names(cfg$dlnm_arglag), sapply(cfg$dlnm_arglag, argspec_str), sep = "=", collapse = ", "), "\n")
    } else {
      cat("DLNM arglag :", argspec_str(cfg$dlnm_arglag), "\n")
    }
  }
  if (!is.null(cfg$ix_vars) && length(cfg$ix_vars) > 0)
    cat("Interactions:", paste(sapply(cfg$ix_vars, paste, collapse = " x "), collapse = ", "), "\n")

  cat("\n--- Random effects ---\n")
  cat("Block RE            :", isTRUE(cfg$use_block_RE) || isTRUE(cfg$use_block_dev), "\n")
  cat("Temporal AR(1)/CMF  :", isTRUE(cfg$use_temporal_AR_perCMF), "\n")
  cat("Temporal AR(1) global:", isTRUE(cfg$use_temporal_AR) && !isTRUE(cfg$use_temporal_AR_perCMF), "\n")
  cat("Spatial ICAR        :", isTRUE(cfg$use_icar), "\n")
  cat("Spatial BYM2        :", isTRUE(cfg$use_bym2), "\n")
  cat("Reactive shift (delta1):", !isTRUE(cfg$fix_delta1), "\n")
  cat("phi fixed           :", isTRUE(cfg$fix_phi), "\n")

  cat("\n=== PARAMETER ESTIMATES ===\n")
  old_width <- options(width = 10000)
  print(as.data.frame(model_sum), digits = 3, row.names = FALSE)
  options(old_width)
})
writeLines(summary_output, file.path(run_output_dir, paste0("model_summary_", model_spec, ".txt")))

# ── Pareto k diagnostics ─────────────────────────────────────────────────────
if (exists("loo_result")) {
  
  k_values <- loo_result$diagnostics$pareto_k
  
  cat("\n--- Pareto k Summary ---\n")
  cat("Good    (k < 0.5) :", sum(k_values < 0.5),  
      sprintf("(%.1f%%)\n", 100 * mean(k_values < 0.5)))
  cat("OK      (0.5-0.7) :", sum(k_values >= 0.5 & k_values < 0.7),
      sprintf("(%.1f%%)\n", 100 * mean(k_values >= 0.5 & k_values < 0.7)))
  cat("Bad     (0.7-1.0) :", sum(k_values >= 0.7 & k_values < 1.0),
      sprintf("(%.1f%%)\n", 100 * mean(k_values >= 0.7 & k_values < 1.0)))
  cat("Very bad (k > 1.0):", sum(k_values >= 1.0),
      sprintf("(%.1f%%)\n", 100 * mean(k_values >= 1.0)))
  
  # Save flagged observations for inspection
  bad_obs <- which(k_values > 0.7)
  if (length(bad_obs) > 0) {
    bad_df <- data.frame(
      obs_idx   = bad_obs,
      pareto_k  = k_values[bad_obs],
      block     = stan_data$block[bad_obs],
      time      = stan_data$time[bad_obs],
      y         = stan_data$y[bad_obs],
      n_bt      = stan_data$n_bt[bad_obs],
      C_bt      = stan_data$C_bt[bad_obs]
    )
    bad_df <- bad_df[order(-bad_df$pareto_k), ]
    flagged_base <- file.path(run_output_dir, paste0("pareto_k_flagged_", model_spec))
    if (requireNamespace("writexl", quietly = TRUE)) {
      writexl::write_xlsx(bad_df, paste0(flagged_base, ".xlsx"))
    } else {
      write.csv(bad_df, paste0(flagged_base, ".csv"), row.names = FALSE)
    }
    cat("Flagged observations saved to:", run_output_dir, "\n")
  }
  
  # Pareto k plot
  png(file.path(run_output_dir, paste0("pareto_k_plot_", model_spec, ".png")),
      width = 800, height = 400)
  plot(loo_result, main = paste("Pareto k —", model_spec))
  abline(h = 0.7, col = "orange", lty = 2)
  abline(h = 1.0, col = "red",    lty = 2)
  dev.off()
}

# =========================== posterior draws for quantive checks ===========================


# 1. PPC statistics (quantitative complement to your visual PPC)
if (cfg$plot_ppc) {
  cat("\n--- PPC Summary Statistics ---\n")
  
  y_pred_draws_mat <- fit$draws("y_pred", format = "matrix")
  y_obs            <- stan_data$y
  
  # Bayesian p-values for key statistics
  ppc_stats <- data.frame(
    statistic = c("mean", "sd", "prop_zero", "q95", "max"),
    observed  = c(
      mean(y_obs),
      sd(y_obs),
      mean(y_obs == 0),
      quantile(y_obs, 0.95),
      max(y_obs)
    ),
    pred_mean = c(
      mean(apply(y_pred_draws_mat, 1, mean)),
      mean(apply(y_pred_draws_mat, 1, sd)),
      mean(apply(y_pred_draws_mat, 1, function(x) mean(x == 0))),
      mean(apply(y_pred_draws_mat, 1, quantile, 0.95)),
      mean(apply(y_pred_draws_mat, 1, max))
    ),
    bayes_p   = c(
      mean(apply(y_pred_draws_mat, 1, mean)    > mean(y_obs)),
      mean(apply(y_pred_draws_mat, 1, sd)      > sd(y_obs)),
      mean(apply(y_pred_draws_mat, 1, function(x) mean(x == 0)) > mean(y_obs == 0)),
      mean(apply(y_pred_draws_mat, 1, quantile, 0.95) > quantile(y_obs, 0.95)),
      mean(apply(y_pred_draws_mat, 1, max)     > max(y_obs))
    )
  )
  
  print(ppc_stats)
  write.csv(ppc_stats,
            file.path(run_output_dir, paste0("ppc_stats_", model_spec, ".csv")),
            row.names = FALSE)
  rm(y_pred_draws_mat); gc()
}

# 2. Credible interval coverage
cat("\n--- Credible Interval Coverage ---\n")

y_pred_draws_mat <- fit$draws("y_pred", format = "matrix")
y_obs            <- stan_data$y

coverage_df <- do.call(rbind, lapply(c(0.50, 0.80, 0.90, 0.95), function(prob) {
  lo    <- apply(y_pred_draws_mat, 2, quantile, (1 - prob) / 2)
  hi    <- apply(y_pred_draws_mat, 2, quantile, 1 - (1 - prob) / 2)
  data.frame(
    nominal  = prob,
    observed = mean(y_obs >= lo & y_obs <= hi),
    diff     = mean(y_obs >= lo & y_obs <= hi) - prob
  )
}))

print(coverage_df)
write.csv(coverage_df,
          file.path(run_output_dir, paste0("ci_coverage_", model_spec, ".csv")),
          row.names = FALSE)
rm(y_pred_draws_mat); gc()

# # 3. Convergence summary (named parameters only)
# cat("\n--- Convergence Summary ---\n")

# model_sum  <- fit$summary()
# key_params <- model_sum[!grepl("^v\\[|^log_lik|^y_pred|^p_bt|^p_R|^omega|^pi\\[",
#                                 model_sum$variable), ]

# rhat_vals  <- key_params$rhat[!is.na(key_params$rhat)]
# ess_vals   <- key_params$ess_bulk[!is.na(key_params$ess_bulk)]

# cat("R-hat > 1.01:", sum(rhat_vals > 1.01), "of", length(rhat_vals), "\n")
# cat("R-hat > 1.05:", sum(rhat_vals > 1.05), "\n")
# cat("Max R-hat:   ", round(max(rhat_vals), 4), "\n")
# cat("ESS < 100:   ", sum(ess_vals < 100),  "of", length(ess_vals), "\n")
# cat("ESS < 400:   ", sum(ess_vals < 400), "\n")
# cat("Min ESS:     ", round(min(ess_vals)), "\n")

# # Flag bad parameters explicitly
# bad_params <- key_params[!is.na(key_params$rhat) & key_params$rhat > 1.05,
#                           c("variable", "rhat", "ess_bulk")]
# if (nrow(bad_params) > 0) {
#   cat("\nParameters with R-hat > 1.05:\n")
#   print(bad_params)
# }

# write.csv(key_params,
#           file.path(run_output_dir, paste0("convergence_", model_spec, ".csv")),
#           row.names = FALSE)

# 4. Per-CMF predictive performance
cat("\n--- Per-CMF Predictive Performance ---\n")

y_pred_mean <- colMeans(fit$draws("y_pred", format = "matrix"))
y_obs       <- stan_data$y

cor_by_block <- tapply(seq_along(y_obs), stan_data$block, function(idx) {
  if (length(idx) < 3) return(NA)
  cor(y_obs[idx], y_pred_mean[idx], method = "spearman")
})

cat("Median per-CMF Spearman r:", 
    round(median(cor_by_block, na.rm = TRUE), 3), "\n")
cat("CMFs with r < 0.3:        ", 
    sum(cor_by_block < 0.3, na.rm = TRUE), "\n")
cat("CMFs with r < 0.0:        ", 
    sum(cor_by_block < 0.0, na.rm = TRUE), "\n")

block_perf <- data.frame(
  block      = as.integer(names(cor_by_block)),
  spearman_r = round(cor_by_block, 3),
  n_obs      = as.integer(table(stan_data$block))
)
block_perf <- block_perf[order(block_perf$spearman_r), ]

write.csv(block_perf,
          file.path(run_output_dir, 
                    paste0("cmf_performance_", model_spec, ".csv")),
          row.names = FALSE)

cat("Per-CMF performance saved.\n")

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
    save_random_effects(post$u, post$v, plots_output_dir, model_spec)
  } else if (!all(is.na(post$v)) && length(post$v) > 0) {
    cat("\nNo spatial random effects found. Plotting only temporal AR effects...\n")
    # Plot only temporal AR effects
    png(file.path(plots_output_dir, paste0("random_effects_temporal_only_", model_spec, ".png")), width = 800, height = 600)
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


# v_bt spaghetti plot and u_block dot plot (present in AR-perCMF+blockRE models)
model_vars <- fit$metadata()$stan_variables
if ("v_cmf_out" %in% model_vars) {
  cat("Generating v_bt per-block AR(1) trajectory plot...\n")
  save_v_bt_plot(fit, df, stan_data, plots_output_dir, model_spec)
}
if ("u_block_out" %in% model_vars) {
  cat("Generating u_block random effects plot...\n")
  save_u_block_plot(fit, plots_output_dir, model_spec)
}

if (isTRUE(cfg$use_dlnm)) {
  cat("Generating DLNM exposure-response and lag-response plots...\n")
  save_dlnm_response_plots(fit, prep, plots_output_dir, model_spec)
  cat("Generating DLNM lag-response plots at fixed exposure percentiles...\n")
  save_dlnm_lagresponse_plots(fit, prep, plots_output_dir, model_spec)
  if (length(prep$dlnm_ix_vars) > 0) {
    cat("Generating DLNM interaction surface plots...\n")
    save_dlnm_interaction_response_plots(fit, prep, plots_output_dir, model_spec)
  }
}

if (cfg$plot_ppc) {
  cat("Generating posterior predictive check plot...\n")
  save_ppc(df, fit, plots_output_dir, model_spec)
}



if (cfg$plot_timeseries) {
  cat("Generating time series plots...\n")
  save_timeseries_plots(df, plots_output_dir, model_spec, cfg$n_blocks_facet)
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
# Set cfg$plot_morans_I = FALSE to skip (loads full y_pred matrix).
if (isTRUE(cfg$plot_morans_I) && requireNamespace("spdep", quietly = TRUE)) {

  post_y_pred  <- fit$draws("y_pred", format = "matrix")
  mean_y_pred  <- colMeans(post_y_pred)
  df$n_bt      <- stan_data$n_bt

  df$pearson_resid_stan <- (df$y_bt - mean_y_pred) /
    sqrt(pmax(mean_y_pred * (1 - mean_y_pred / df$n_bt), 1e-6))
  df$year <- lubridate::year(df$year_month_date)

  df_spatial <- df %>%
    mutate(block_chr = as.character(.data[[cfg$block_col]])) %>%
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

  moran_stan_df <- do.call(rbind, moran_stan_list)
  if (is.null(moran_stan_df) || nrow(moran_stan_df) == 0) {
    cat("Skipping Stan Moran's I plot: not enough blocks per year (need >= 10).\n")
  } else {

    moran_stan_df <- moran_stan_df %>%
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
                paste0("moransI_stan_by_year_", model_spec, ".png")),
      p_moran_stan, width = 11, height = 5, dpi = 150
    )

    write.csv(
      moran_stan_df,
      file.path(plots_output_dir,
                paste0("moransI_stan_by_year_", model_spec, ".csv")),
      row.names = FALSE
    )
    cat("Stan Moran's I correlogram saved to:", plots_output_dir, "\n")
  }

} else if (!isTRUE(cfg$plot_morans_I)) {
  cat("Skipping Stan Moran's I (plot_morans_I = FALSE).\n")
} else {
  cat("Skipping Stan Moran's I: package 'spdep' not installed.\n")
}

# =========================
# 5) SPATIAL RE vs. AR TERM: DOES THE AR STATE ABSORB THE SPATIAL SIGNAL?
# =========================
if ("u_block_out" %in% model_vars) {
  cat("Generating spatial RE vs. AR term correlation checks...\n")
  save_spatial_re_ar_correlation_checks(fit, coords_sf, stan_data, plots_output_dir, model_spec)
}

# =========================
# TRACEPLOTS - placed at the end (often cause for crash)
# =========================
# Robust traceplot generation: only plot parameters that exist in the fit object
if (cfg$plot_traceplots) {
  cat("Generating trace plots...\n")
  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    cat("bayesplot package not installed; skipping trace plots.\n")
  } else {
    library(bayesplot)

    # Use metadata (cheap) instead of fit$summary() (expensive — summarises all
    # generated quantities and can hang/crash on large models)
    model_vars <- fit$metadata()$stan_variables

    # Whitelist: scalar/vector model parameters worth tracing
    # sigma_w is a vector[K] — drawn by root name, elements appear in the plot
    scalar_include <- c("alpha", "sigma_gp", "rho_gp", "sigma_icar",
                        "sigma_spatial", "phi_mix",
                        "delta1",
                        "tau", "sigma_v", "rho", "sigma_block_dev",
                        "sigma_time", "sigma_block",
                        "phi")
    scalar_vars <- intersect(scalar_include, model_vars)

    trace_dir <- file.path(plots_output_dir, "traceplots")
    dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

    # Helper: save chunked traceplots (avoids huge single ggplot)
    save_trace_chunks <- function(vars, draws_arr, file_prefix, chunk_size = 12, w, h) {
      chunks <- split(vars, ceiling(seq_along(vars) / chunk_size))
      for (i in seq_along(chunks)) {
        ggsave(
          file.path(trace_dir, paste0(file_prefix, "_part", i, "_", model_spec, ".png")),
          mcmc_trace(draws_arr, pars = chunks[[i]]), width = w, height = h
        )
      }
    }

    # Scalar/hyperparameter params
    if (length(scalar_vars) > 0) {
      draws_scalar <- fit$draws(variables = scalar_vars, format = "array")
      scalar_params <- dimnames(draws_scalar)[[3]]
      save_trace_chunks(scalar_params, draws_scalar, "traceplot_params", chunk_size = 12, w = 10, h = 8)
    } else {
      cat("No scalar parameters found for traceplot.\n")
    }

    # Lagged/DLNM weights
    if ("w_cb" %in% model_vars) {
      draws_w  <- fit$draws(variables = "w_cb", format = "array")
      w_params <- dimnames(draws_w)[[3]]
      cat("Plotting DLNM weight traceplots:", paste(head(w_params, 6), collapse = ", "), "...\n")
      save_trace_chunks(w_params, draws_w, "traceplot_weights_wcb", chunk_size = 12, w = 12, h = 10)
    } else if ("w" %in% model_vars) {
      draws_w  <- fit$draws(variables = "w", format = "array")
      w_params <- dimnames(draws_w)[[3]]
      cat("Plotting lag weight traceplots:", paste(w_params, collapse = ", "), "\n")
      save_trace_chunks(w_params, draws_w, "traceplot_weights_w", chunk_size = 12, w = 12, h = 10)
    }

    # Unlagged weights
    if ("w_unlagged" %in% model_vars) {
      draws_wu  <- fit$draws(variables = "w_unlagged", format = "array")
      wu_params <- dimnames(draws_wu)[[3]]
      save_trace_chunks(wu_params, draws_wu, "traceplot_weights_unlagged", chunk_size = 12, w = 12, h = 8)
    }

    # Interaction cross-basis weights
    if ("w_ix" %in% model_vars) {
      draws_wix  <- fit$draws(variables = "w_ix", format = "array")
      wix_params <- dimnames(draws_wix)[[3]]
      cat("Plotting interaction weight traceplots:", paste(head(wix_params, 6), collapse = ", "), "...\n")
      save_trace_chunks(wix_params, draws_wix, "traceplot_weights_ix", chunk_size = 12, w = 12, h = 8)
    }
  }
}