# =============================================================
# Variable sweep: run ICAR model with incrementally added predictors
# =============================================================
# Edit the `combinations` list below to define which variable sets to fit.
# All other settings are shared across runs (same model structure, same MCMC).
# cfg$use_dlnm (below) controls the whole sweep: when TRUE, every combo's lag
# vars go through the DLNM spline cross-basis (build_dlnm_stan_data() +
# hierarchical_state_space_AR_perCMF_blockRE_DLNM_ix.stan), not just the
# Group 4 combos that specify an `ix` interaction. That single compiled model
# handles both cases: P_ix/X_ix/w_ix are all data/parameters of size 0 when a
# combo has no `ix` (see the Stan file's own data-block comments), so no
# separate model or init function is needed for "DLNM without interaction".
# `has_ix` (set per-combo inside the loop) is the *only* thing that still
# gates interaction-specific output (w_ix summaries/traceplots, the
# save_dlnm_interaction_response_plots() call, the "_ix-" spec suffix).
# Set cfg$use_dlnm <- FALSE to fall back to the flat/binned lag design
# (build_stan_data() + the base blockRE model) for the whole sweep instead.
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
# numeric_vars = lag_vars minus any ending in _cat, plus continuous unlagged
# variables that should be standardized.
forced_unlagged_vars <- list(
  numeric = c("mean_ndvi", "water_containers"),
  binary  = c("is_urban", "is_WUI", "is_WI", "has_aljibes", "water_shortage")
)
get_combo_vars <- function(lag_vars, unlag_vars) {
  all_forced <- c(forced_unlagged_vars$numeric, forced_unlagged_vars$binary)
  list(
    lag   = setdiff(lag_vars, all_forced),
    unlag = unique(c(unlag_vars, intersect(lag_vars, all_forced)))
  )
}
get_numeric_vars <- function(lag_vars, unlag_vars) {
  unique(c(
    lag_vars[!grepl("_cat$", lag_vars)],
    intersect(unlag_vars, forced_unlagged_vars$numeric)
  ))
}
# ── Variable sweep combinations (RF-informed, 2026-07) ───────────────────────
# Precipitation variable: total_precip bundle (bake-off winner over
#   total_rainy_days; precip_max_day_resid_on_tp is the orthogonalised extreme-
#   event residual, only valid paired with total_precip).
# Core static predictors: is_urban (interaction modifier), is_WUI (RF #2),
#   water_containers (RF #5, strongest H-stat with precip).
# Optional static: water_shortage (RF above floor, mild H-stat with precip),
#   is_WI (near noise floor — borderline).
# Dropped: has_aljibes, nr_aljibes (both below RF noise floor).
# Climate variants: avg_temp (RF dominant but correlated with precip, r=0.65),
#   WS2M (collinear with temp r=-0.61 and precip r=-0.48 — test without temp).
# mean_ndvi in unlag — no lagged effect expected; treated as static vegetation index.
#
# Run the sweep twice (toggle use_icar below) to compare blockRE vs ICAR;
# LOO-IC then selects the better spatial structure per variable configuration.
# ─────────────────────────────────────────────────────────────────────────────
combinations <- list(

  # ── Group 1: Precipitation core — no temperature or wind ──────────────────
  # Establishes the baseline signal from total_precip + VPD alone.
  # Incrementally adds optional static predictors to test their contribution.

  # minimal model 
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization")),

  # minimal model + one extra spatial predictor 
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "mean_ndvi")),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","water_containers")),
  
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","is_WUI")),

  # minimal model + two extra spatial predictors  
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","mean_ndvi", "water_containers")),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","mean_ndvi", "water_shortage")),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","is_WUI", "water_containers")),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","is_WUI", "is_WI")),
  
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","is_WUI", "water_shortage")),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization","water_containers", "water_shortage")),

  # minimal model + three extra spatial predictors   
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "water_containers", "mean_ndvi")),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "water_shortage", "mean_ndvi")),
  
  # minimal model + four extra spatial predictors  
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "water_containers", "water_shortage", "mean_ndvi")),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "is_WI", "water_containers", "mean_ndvi")),

# full model - 1 predictor 
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "is_WI", "water_containers", "water_shortage", "mean_ndvi")),

# full model 
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "is_WI", "water_containers", "water_shortage", "mean_ndvi", "WS2M")),

  # # ── Group 2: Add avg_temp ─────────────────────────────────────────────────
  # # avg_temp is the dominant RF predictor (importance 0.307) but correlated
  # # with total_precip (Spearman r = 0.65) and total_rainy_days (r = 0.70)
  # # because the warm and wet seasons coincide in Cienfuegos. LOO-IC will
  # # reveal whether temp adds independent predictive value on top of precip.
  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization")),
  
  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization", "mean_ndvi")),

  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization", "water_containers")),

  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization", "is_WUI", "water_containers", "mean_ndvi")),
  
  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization", "is_WUI", "water_containers", "mean_ndvi", "WS2M")),

  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization", "is_WUI", "water_containers", "water_shortage", "mean_ndvi")),

  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization", "is_WUI", "water_containers", "is_WI", "mean_ndvi")),

  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "avg_temp"),
  #      unlag = c("HFP_urbanization", "is_WUI", "water_containers", "water_shortage", "is_WI", "mean_ndvi")),

  # # ── Group 3: WS2M instead of avg_temp ────────────────────────────────────
  # # WS2M tracks the dry/cool season (r = -0.48 with total_precip, r = -0.61
  # # with avg_temp). Tests whether wind carries residual signal that precip and
  # # VPD don't capture, without the temp-precip collinearity. Only the core and
  # # core+water_shortage variants are included here; if WS2M adds nothing in
  # # these, it will not help in the larger models either.
  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "WS2M"),
  #      unlag = c("is_urban", "is_WUI", "water_containers", "mean_ndvi")),

  # list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD", "WS2M"),
  #      unlag = c("is_urban", "is_WUI", "water_containers", "water_shortage", "mean_ndvi")),

  # Group 4: adds the is_urban × total_precip interaction on top of the same
  # DLNM cross-basis every other combo already uses (cfg$use_dlnm = TRUE
  # above). Exposure and lag bases: ns(df=3) for all variables.
  # Interaction chosen from RF H-statistics:
  #   is_urban × total_precip  (binary, active_level=0, H ~ 0.17–0.27 at lags 0–2)
  # Continuous modifiers (e.g. water_containers × total_precip, RF H > 0.45 at
  # lags 0,1,5) are not modeled as DLNM interactions here: a linear-in-modifier
  # tilt of the cross-basis, evaluated only at the modifier's mean and +1 SD,
  # is hard to interpret as anything other than an arbitrary two-point probe
  # of what is actually a continuous effect-modification surface. See
  # build_dlnm_stan_data() in helper_functions.r.
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "water_containers", "mean_ndvi"),
       ix    = list(
         list(binary_var = "is_urban",        active_level = 0,
              dlnm_var   = "total_precip",     label = "nonurban_x_tp")
       )),
  
  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "water_containers", "mean_ndvi"),
       ix    = list(
         list(binary_var = "is_urban",        active_level = 0,
              dlnm_var   = "total_precip",     label = "nonurban_x_tp")
       )),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "water_containers", "mean_ndvi", "WS2M"),
       ix    = list(
         list(binary_var = "is_urban",        active_level = 0,
              dlnm_var   = "total_precip",     label = "nonurban_x_tp")
       )),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI", "water_containers", "water_shortage", "mean_ndvi", "WS2M"),
       ix    = list(
         list(binary_var = "is_urban",        active_level = 0,
              dlnm_var   = "total_precip",     label = "nonurban_x_tp")
       )),

  list(lag   = c("total_precip", "precip_max_day_resid_on_tp", "avg_VPD"),
       unlag = c("HFP_urbanization", "is_WUI","is_WI", "water_containers", "water_shortage", "mean_ndvi", "WS2M"),
       ix    = list(
         list(binary_var = "is_urban",        active_level = 0,
              dlnm_var   = "total_precip",     label = "nonurban_x_tp")
       ))
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
    "env_epi_entomo_data_per_CMF_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv"
  else
    "env_epi_entomo_data_per_manzana_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv",
  response_start = "2016_01",   # 2015 rows used as lag lead-in only, not passed to Stan
  output_dir = if (hostname == "frietjes")
    "/home/rita/data/Entomo/fitting/stan"
  else
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan",

  use_time_RE     = FALSE,
  use_temporal_AR = TRUE,
  use_temporal_AR_perCMF = TRUE,
  use_spatial_AC  = FALSE,
  use_hsgp        = FALSE,
  use_icar        = FALSE,
  use_bym2        = FALSE,
  hsgp_m          = 20,
  hsgp_c          = 1.5,
  use_block_dev   = TRUE,
  use_dlnm        = TRUE,  # TRUE: every combo's lag_vars go through the DLNM
                            # spline cross-basis, whether or not it has `ix`
                            # (P_ix/X_ix/w_ix are size 0 when it doesn't).
                            # FALSE: whole sweep falls back to the flat/binned
                            # lag design instead.

  shapefile_path = if (hostname == "frietjes")
    "/home/rita/data/Entomo"
  else
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders",
  sf_block_col = if (spatial_level == "CMF") "Area_CMF" else "CODIGO_",
  block_col    = if (spatial_level == "CMF") "cmf"      else "manzana",

  n_blocks  = NULL,
  max_lag   = 6,
  kappa     = 4,

  chains          = 4,
  iter_warmup     = 1000,
  iter_sampling   = 1000,
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
  if (isTRUE(cfg$use_block_dev))
    file.path(stan_dir, "hierarchical_state_space_blockRE.stan")
  else
    file.path(stan_dir, "hierarchical_state_space.stan")
} else if (!isTRUE(cfg$use_spatial_AC)) {
  if (isTRUE(cfg$use_temporal_AR_perCMF)) {
    if (isTRUE(cfg$use_block_dev))
      file.path(stan_dir, "hierarchical_state_space_AR_perCMF_blockRE.stan")
    else
      file.path(stan_dir, "hierarchical_state_space_AR_perCMF.stan")
  } else {
    if (isTRUE(cfg$use_block_dev))
      file.path(stan_dir, "hierarchical_state_space_AR_blockRE.stan")
    else
      file.path(stan_dir, "hierarchical_state_space_AR.stan")
  }
} else if (isTRUE(cfg$use_bym2)) {
  file.path(stan_dir, "hierarchical_state_space_AR_BYM2.stan")
} else if (isTRUE(cfg$use_icar)) {
  if (isTRUE(cfg$use_temporal_AR_perCMF))
    file.path(stan_dir, "hierarchical_state_space_AR_perCMF_ICAR.stan")
  else if (isTRUE(cfg$use_block_dev))
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
  ar1_suffix     <- if (!isTRUE(cfg$use_temporal_AR))       "noAR1"
                   else if (isTRUE(cfg$use_temporal_AR_perCMF)) "AR1perCMF"
                   else                                         "AR1"
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

# ── Core budget: combos run SEQUENTIALLY, one at a time. Each combo's own
# MCMC chains still run in parallel per cfg$parallel_chains (chain-level
# parallelism within a single fit is cheap: those chains share one copy of
# the compiled model/data). Running multiple combos concurrently on top of
# that was reverted -- on a shared, memory-constrained box, stacking several
# full Stan fits at once (each with its own copy of the model, data, and
# log_lik draws in memory) is a much more likely crash cause than CPU
# contention, especially when swap is already in heavy use from other users'
# jobs. loo::loo()'s own internal parallelism is disabled (mc.cores = 1) to
# avoid adding to that footprint.
options(mc.cores = 1)
cat(sprintf(
  "Sequential sweep: 1 combo at a time x %d chain(s)/fit.\n",
  cfg$parallel_chains
))

dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)
sweep_dir <- file.path(cfg$output_dir, paste0("variable_sweep_", model_spec, "_", date_suffix))
dir.create(sweep_dir, recursive = TRUE, showWarnings = FALSE)

# ── One-time spatial setup (outside the loop) ─────────────────────────────────
# Use the last (fullest) combination to get block_ids — they don't depend on
# which variables are included, only on the data file and spatial resolution.
cfg$lag_vars      <- combinations[[length(combinations)]]$lag
cfg$unlagged_vars <- combinations[[length(combinations)]]$unlag
combo_vars <- get_combo_vars(cfg$lag_vars, cfg$unlagged_vars)
cfg$lag_vars      <- combo_vars$lag
cfg$unlagged_vars <- combo_vars$unlag
cfg$numeric_vars  <- get_numeric_vars(cfg$lag_vars, cfg$unlagged_vars)
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

# ── Compile Stan models ───────────────────────────────────────────────────────
# K and Ku are passed as data, not hard-coded in the Stan file, so the same
# compiled binary is valid for all variable combinations. Only compile the
# model(s) this sweep will actually use.
if (isTRUE(cfg$use_dlnm)) {
  # DLNM-ix model: P_cb/P_ix are data, so this one binary handles every combo,
  # whether or not it specifies `ix` (P_ix/X_ix/w_ix = 0 when it doesn't).
  dlnm_ix_stan_file <- file.path(stan_dir,
    "hierarchical_state_space_AR_perCMF_blockRE_DLNM_ix.stan")
  mod_dlnm_ix <- cmdstan_model(dlnm_ix_stan_file,
                                force_recompile = hostname == "frietjes")
  cat("DLNM-ix Stan model compiled.\n")
} else {
  mod <- cmdstan_model(cfg$stan_file, force_recompile = hostname == "frietjes")
  cat("Stan model compiled.\n")
}

# ── Per-combo worker ───────────────────────────────────────────────────────────
# Runs one combo end-to-end (design matrices -> MCMC -> plots -> LOO) and
# returns its result (rather than mutating shared state) so results can be
# collected uniformly after the sweep loop below. Console output goes straight
# to stdout as usual (combos run sequentially, so there's nothing to interleave).
run_one_combo <- function(combo_i) {
  cfg_i <- cfg   # per-worker copy; never mutate the shared `cfg`
  combo <- combinations[[combo_i]]
  has_ix <- !is.null(combo$ix)
  ix_labels  <- if (has_ix)
    paste(sapply(combo$ix, `[[`, "label"), collapse = "+")
  else ""

  cfg_i$lag_vars      <- combo$lag
  cfg_i$unlagged_vars <- combo$unlag
  combo_vars <- get_combo_vars(cfg_i$lag_vars, cfg_i$unlagged_vars)
  cfg_i$lag_vars      <- combo_vars$lag
  cfg_i$unlagged_vars <- combo_vars$unlag
  cfg_i$numeric_vars  <- get_numeric_vars(cfg_i$lag_vars, cfg_i$unlagged_vars)

  # predictor_spec encodes the variable set → unique output dir per combo
  predictor_spec <- paste0(
    "lag-",    paste(cfg_i$lag_vars,      collapse = "-"),
    "_unlag-", paste(cfg_i$unlagged_vars, collapse = "-"),
    if (has_ix) paste0("_ix-", gsub("\\+", "-", ix_labels)) else ""
  )
  run_output_dir   <- file.path(sweep_dir, predictor_spec)
  plots_output_dir <- file.path(run_output_dir, "plots")
  dir.create(run_output_dir,   recursive = TRUE, showWarnings = FALSE)
  dir.create(plots_output_dir, recursive = TRUE, showWarnings = FALSE)

  cat(sprintf(
    "\n========== Run %d / %d ==========\n  lag:   %s\n  unlag: %s%s\n",
    combo_i, length(combinations),
    paste(combo$lag, collapse = ", "),
    if (length(combo$unlag) == 0) "(none)" else paste(combo$unlag, collapse = ", "),
    if (has_ix) paste0("\n  ix:    ", ix_labels) else ""
  ))

  t_start <- proc.time()["elapsed"]
  loo_result <- NULL

  status <- tryCatch({

    # Rebuild design matrices for this combo (fast; spatial setup already done)
    if (isTRUE(cfg_i$use_dlnm)) {
      cfg_i$dlnm_vars    <- combo_vars$lag
      cfg_i$dlnm_argvar  <- setNames(
        lapply(combo_vars$lag, function(v) list(fun = "ns", df = 3)),
        combo_vars$lag
      )
      cfg_i$dlnm_arglag  <- list(fun = "ns", df = 3)
      cfg_i$dlnm_ix_vars <- combo$ix   # NULL when !has_ix -> P_ix = 0, handled by the Stan model
      prep <- build_dlnm_stan_data(cfg_i)
    } else {
      prep <- build_stan_data(cfg_i)
    }
    stan_data <- prep$stan_data
    df        <- prep$df

    # Inject pre-computed ICAR edges (only for spatial AC models)
    if (isTRUE(cfg_i$use_spatial_AC)) {
      stan_data$N_edges <- icar_edges$N_edges
      stan_data$node1   <- icar_edges$node1
      stan_data$node2   <- icar_edges$node2
    }

    # phi setup
    stan_data$fix_phi <- as.integer(isTRUE(cfg_i$fix_phi))
    if (isTRUE(cfg_i$fix_phi)) {
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
        cfg_i$phi_fixed
      }
      stan_data$phi_data <- phi_use
      cat(sprintf("phi fixed at %.1f\n", phi_use))
    } else {
      stan_data$phi_data <- 1.0
      cat("phi will be estimated\n")
    }

    # MCMC
    mod_use <- if (isTRUE(cfg_i$use_dlnm)) mod_dlnm_ix else mod
    init_fn <- if (isTRUE(cfg_i$use_dlnm)) {
      local({
        sd <- stan_data
        fix_phi_flag <- isTRUE(cfg_i$fix_phi)
        function() list(
          alpha       = rnorm(1, -7, 0.5),
          w_cb        = rnorm(sd$P_cb, 0, 0.05),
          w_ix        = rnorm(sd$P_ix, 0, 0.03),
          w_unlagged  = rnorm(sd$Ku,   0, 0.05),
          v_raw       = matrix(0, sd$B, sd$T),
          tau         = abs(rnorm(1, 0.4, 0.1)),
          rho         = rnorm(1, 0.4, 0.05),
          u_block_raw = rnorm(sd$B, 0, 0.1),
          sigma_block = abs(rnorm(1, 0.3, 0.05)),
          delta1      = abs(rnorm(1, 0.1, 0.05)),
          phi_raw     = abs(rnorm(1, 100, 20))
        )
      })
    } else {
      make_init_fun(
        stan_data, cfg_i$use_temporal_AR,
        use_hsgp               = isTRUE(cfg_i$use_hsgp) && !isTRUE(cfg_i$use_icar) && !isTRUE(cfg_i$use_bym2),
        use_icar               = isTRUE(cfg_i$use_icar) && !isTRUE(cfg_i$use_bym2),
        use_bym2               = isTRUE(cfg_i$use_bym2),
        use_time_RE            = isTRUE(cfg_i$use_time_RE),
        use_spatial_AC         = isTRUE(cfg_i$use_spatial_AC),
        use_block_dev          = isTRUE(cfg_i$use_block_dev),
        use_temporal_AR_perCMF = isTRUE(cfg_i$use_temporal_AR_perCMF)
      )
    }
    fit <- mod_use$sample(
      data            = stan_data,
      chains          = cfg_i$chains,
      iter_warmup     = cfg_i$iter_warmup,
      iter_sampling   = cfg_i$iter_sampling,
      thin            = if (!is.null(cfg_i$thin)) cfg_i$thin else 1,
      init            = init_fn,
      adapt_delta     = cfg_i$adapt_delta,
      max_treedepth   = cfg_i$max_treedepth,
      parallel_chains = cfg_i$parallel_chains,
      output_dir      = run_output_dir
    )
    invisible(file.remove(list.files(run_output_dir,
                                     pattern = "_(config|metric)\\.json$",
                                     full.names = TRUE)))

    # Model summary
    w_vars <- if (isTRUE(cfg_i$use_dlnm)) {
      c("w_cb", "w_unlagged", if (has_ix) "w_ix")
    } else {
      c("w", "w_unlagged")
    }
    summary_vars <- c("alpha", "delta1", w_vars)
    if (isTRUE(cfg_i$use_time_RE)) {
      summary_vars <- c(summary_vars, "sigma_time", "sigma_block")
    } else {
      if (isTRUE(cfg_i$use_spatial_AC)) {
        if (isTRUE(cfg_i$use_bym2))      summary_vars <- c(summary_vars, "sigma_spatial", "phi_mix")
        else if (isTRUE(cfg_i$use_icar)) summary_vars <- c(summary_vars, "sigma_icar")
        else                            summary_vars <- c(summary_vars, "sigma_gp", "rho_gp")
      }
      if (isTRUE(cfg_i$use_temporal_AR)) summary_vars <- c(summary_vars, "tau", "sigma_v", "rho")
      if (!isTRUE(cfg_i$use_bym2) && isTRUE(cfg_i$use_block_dev)) {
        if (isTRUE(cfg_i$use_temporal_AR_perCMF))
          summary_vars <- c(summary_vars, "sigma_block")
        else
          summary_vars <- c(summary_vars, "sigma_block_dev")
      }
    }
    if (!isTRUE(cfg_i$fix_phi)) summary_vars <- c(summary_vars, "phi")
    model_sum <- if (isTRUE(cfg_i$use_dlnm)) {
      fit$summary(variables = summary_vars)
    } else {
      rename_w_in_summary(fit$summary(variables = summary_vars), prep$lag_vars_expanded, prep$unlagged_vars)
    }
    summary_output <- capture.output({
      old_width <- options(width = 10000)
      print(as.data.frame(model_sum), digits = 3, row.names = FALSE)
      options(old_width)
    })
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
    if (cfg_i$plot_random_effects) {
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
    if (cfg_i$plot_ppc)        save_ppc(df, fit, plots_output_dir, model_spec)
    if (cfg_i$plot_timeseries) save_timeseries_plots(df, plots_output_dir, model_spec,
                                                      cfg_i$n_blocks_facet)

    # DLNM response plots: 2D/3D exposure-response and lag-response surfaces,
    # for every combo now that cfg_i$use_dlnm applies sweep-wide. The
    # interaction-comparison plots stay gated on has_ix (nothing to compare
    # against a reference group when P_ix = 0).
    if (isTRUE(cfg_i$use_dlnm)) {
      save_dlnm_response_plots(fit, prep, plots_output_dir, model_spec)
      save_dlnm_lagresponse_plots(fit, prep, plots_output_dir, model_spec)
      if (has_ix) save_dlnm_interaction_response_plots(fit, prep, plots_output_dir, model_spec)
    }

    # Per-CMF AR(1) trajectory plot (all runs)
    save_v_bt_plot(fit, df, stan_data, plots_output_dir, model_spec)

    # Traceplots
    if (cfg_i$plot_traceplots && requireNamespace("bayesplot", quietly = TRUE)) {
      library(bayesplot)
      model_vars     <- fit$metadata()$stan_variables
      scalar_include <- c("alpha", "sigma_gp", "rho_gp", "sigma_icar",
                          "sigma_spatial", "phi_mix", "delta1",
                          "tau", "sigma_v", "rho", "sigma_block_dev",
                          "sigma_time", "sigma_block", "phi")
      scalar_vars    <- intersect(scalar_include, model_vars)
      trace_dir <- file.path(plots_output_dir, "traceplots")
      dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)
      save_trace_chunks <- function(vars, draws_arr, file_prefix, chunk_size = 12, w, h) {
        chunks <- split(vars, ceiling(seq_along(vars) / chunk_size))
        for (i in seq_along(chunks))
          ggsave(
            file.path(trace_dir,
                      paste0(file_prefix, "_part", i, "_", model_spec, ".png")),
            mcmc_trace(draws_arr, pars = chunks[[i]]), width = w, height = h
          )
      }
      if (length(scalar_vars) > 0) {
        draws_scalar  <- fit$draws(variables = scalar_vars, format = "array")
        scalar_params <- dimnames(draws_scalar)[[3]]
        save_trace_chunks(scalar_params, draws_scalar, "traceplot_params", w = 10, h = 8)
      }
      if (isTRUE(cfg_i$use_dlnm)) {
        if ("w_cb" %in% model_vars) {
          draws_wcb <- fit$draws(variables = "w_cb", format = "array")
          save_trace_chunks(dimnames(draws_wcb)[[3]], draws_wcb,
                            "traceplot_weights_wcb", w = 12, h = 10)
        }
        if (has_ix && "w_ix" %in% model_vars) {
          draws_wix <- fit$draws(variables = "w_ix", format = "array")
          save_trace_chunks(dimnames(draws_wix)[[3]], draws_wix,
                            "traceplot_weights_wix", w = 12, h = 8)
        }
      } else {
        if ("w" %in% model_vars) {
          draws_w <- fit$draws(variables = "w", format = "array")
          save_trace_chunks(dimnames(draws_w)[[3]], draws_w,
                            "traceplot_weights_w", w = 12, h = 10)
        }
      }
      if ("w_unlagged" %in% model_vars) {
        draws_wu <- fit$draws(variables = "w_unlagged", format = "array")
        save_trace_chunks(dimnames(draws_wu)[[3]], draws_wu,
                          "traceplot_weights_unlagged", w = 12, h = 8)
      }
    }

    # LOO-CV — must run before CSV deletion (fit reads from CSV files)
    if (requireNamespace("loo", quietly = TRUE)) {
      log_lik_draws <- fit$draws("log_lik", format = "array")
      r_eff         <- loo::relative_eff(exp(log_lik_draws))
      loo_result    <- loo::loo(log_lik_draws, r_eff = r_eff)
      loo_output    <- capture.output(print(loo_result))
      writeLines(loo_output, file.path(run_output_dir, paste0("loo_", predictor_spec, ".txt")))
      saveRDS(loo_result,    file.path(run_output_dir, paste0("loo_", predictor_spec, ".rds")))
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
  log_row <- data.frame(
    run         = combo_i,
    lag_vars    = paste(combo$lag,   collapse = "|"),
    unlag_vars  = paste(combo$unlag, collapse = "|"),
    ix_vars     = if (has_ix) ix_labels else "",
    n_lag       = length(combo$lag),
    n_unlag     = length(combo$unlag),
    status      = status,
    elapsed_min = elapsed_min
  )

  list(predictor_spec = predictor_spec, loo_result = loo_result, log_row = log_row)
}

# ── Main sweep: run all combos one at a time ──────────────────────────────────
cat(sprintf("\nRunning %d combo(s) sequentially...\n", length(combinations)))
combo_results <- lapply(seq_along(combinations), run_one_combo)

sweep_log <- lapply(combo_results, function(r) r$log_row)
loo_list  <- list()   # named by predictor_spec
for (r in combo_results) {
  if (!is.null(r$loo_result)) loo_list[[r$predictor_spec]] <- r$loo_result
}

# ── Write sweep log ───────────────────────────────────────────────────────────
sweep_log_df <- do.call(rbind, sweep_log)
log_path     <- file.path(sweep_dir, "variable_sweep_log.csv")
write.csv(sweep_log_df, log_path, row.names = FALSE)
cat("\nSweep complete. Log written to:", log_path, "\n")
print(sweep_log_df)

# ── LOO comparison across all variable combinations ───────────────────────────
if (requireNamespace("loo", quietly = TRUE) && length(loo_list) >= 2) {
  loo_cmp <- loo::loo_compare(loo_list)
  print(loo_cmp, digits = 2, simplify = FALSE)

  cmp_df <- as.data.frame(loo_cmp)
  cmp_df$z_score <- cmp_df$elpd_diff / cmp_df$se_diff
  cmp_df$z_score[cmp_df$elpd_diff == 0] <- 0
  cat("\nz-score (elpd_diff / se_diff):\n")
  print(cmp_df["z_score"], digits = 2)

  cmp_out <- cbind(model = rownames(cmp_df), cmp_df)
  rownames(cmp_out) <- NULL

  writeLines(capture.output({
    print(loo_cmp, digits = 2, simplify = FALSE)
    cat("\nz-score (elpd_diff / se_diff):\n")
    print(cmp_df["z_score"], digits = 2)
  }), file.path(sweep_dir, "loo_comparison.txt"))
  writexl::write_xlsx(
    cmp_out,
    file.path(sweep_dir, "loo_comparison.xlsx")
  )
  cat("LOO comparison saved to:", sweep_dir, "\n")
} else if (length(loo_list) < 2) {
  cat("Fewer than 2 successful LOO results — skipping comparison.\n")
}
