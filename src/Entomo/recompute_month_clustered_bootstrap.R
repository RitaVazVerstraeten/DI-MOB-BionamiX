# recompute_month_clustered_bootstrap.R
#
# The argvar sweep's saved bootstrap comparison clusters by CMF (block),
# inherited unchanged from run_season_interaction_sweep.R. That doesn't
# address the actual concern: total_precip/avg_VPD/precip_max_day_resid_on_tp
# are municipality-wide, so every CMF repeats the same value in a given
# month -- resampling *which CMFs* are in a bootstrap draw still leaves every
# draw with all ~48 months of duplicated climate data. The correct cluster
# unit for this concern is MONTH, not CMF.
#
# bootstrap_elpd_comparison() only needs the already-saved pointwise
# log-likelihoods (loo_list_<date>.rds / waic_list_<date>.rds) plus a
# cluster_ids vector aligned to those rows -- no refit required. This script
# rebuilds just the row-ordering (via build_dlnm_stan_data(), same as any
# sweep scenario -- dlnm_argvar/dlnm_ix_vars don't affect which rows end up
# in stan_data or their order, only lag_vars/max_lag/block_col/response_start/
# n_blocks do, and those are identical across every scenario in the sweep)
# to recover prep$df$year_month in the exact row order the pointwise elpd
# vectors are indexed by, then recomputes both bootstrap comparisons
# clustered by month instead of by block.
#
# Usage: edit sweep_date below if you re-run the sweep on a different day,
# then: Rscript recompute_month_clustered_bootstrap.R
# =============================================================================

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

source(file.path(script_dir, "helper_functions.r"))

hostname  <- Sys.info()["nodename"]
sweep_dir <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/argvar_sweep"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/argvar_sweep"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/argvar_sweep"
}
sweep_dir <- path.expand(sweep_dir)

sweep_date <- "20260826"   # edit if the sweep's saved *_list_<date>.rds is from a different day

# =============================================================================
# Rebuild JUST the row ordering -- dlnm_argvar values here are placeholders
# (don't affect which rows end up in stan_data or their order, only
# lag_vars/max_lag/block_col/response_start/n_blocks do, all matched to
# run_argvar_sweep.R exactly). This still needs dlnm installed (for
# crossbasis() to run), same as any real fit, but does no MCMC sampling.
# =============================================================================
cfg <- list(
  data_dir       = if (hostname == "frietjes") "~/data/Entomo" else if (hostname == "stoofvlees") "~/entomo_data" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data",
  data_file_name = "env_epi_entomo_data_per_CMF_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv",
  block_col      = "cmf",
  n_blocks       = NULL,
  response_start = "2016_01",
  lag_vars       = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp"),
  dlnm_vars      = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp"),
  max_lag        = 5,
  dlnm_argvar    = list(total_precip = list(fun = "ns", df = 3), avg_VPD = list(fun = "ns", df = 3), precip_max_day_resid_on_tp = list(fun = "ns", df = 3)),
  dlnm_arglag    = list(fun = "ns", df = 3),
  dlnm_ix_vars   = NULL,
  unlagged_vars  = c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_shortage", "water_containers"),
  numeric_vars   = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi"),
  kappa          = 4
)
cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)

prep <- build_dlnm_stan_data(cfg)
month_ids <- prep$df$year_month
cat("N rows in rebuilt df:", length(month_ids), " | N distinct months:", length(unique(month_ids)), "\n")

# =============================================================================
# Recompute both bootstrap comparisons, clustered by month instead of block
# =============================================================================
recompute <- function(file_stub, criterion_label) {
  rds_path <- file.path(sweep_dir, paste0(file_stub, "_comparison_", sweep_date), paste0(file_stub, "_list_", sweep_date, ".rds"))
  if (!file.exists(rds_path)) {
    cat("Not found, skipping:", rds_path, "\n")
    return(invisible(NULL))
  }
  result_list <- readRDS(rds_path)

  if (length(month_ids) != length(result_list[[1]]$pointwise[, 1])) {
    stop(sprintf("Row count mismatch for %s: rebuilt df has %d rows, saved %s has %d -- cfg above doesn't match the sweep's row-determining settings.",
                  file_stub, length(month_ids), file_stub, length(result_list[[1]]$pointwise[, 1])))
  }

  cat("\n", strrep("=", 70), "\n")
  cat(criterion_label, "BOOTSTRAP COMPARISON -- clustered by MONTH (not block)\n")
  cat(strrep("=", 70), "\n\n")

  boot_cmp <- bootstrap_elpd_comparison(result_list, cluster_ids = month_ids, n_boot = 4000)
  print(boot_cmp, digits = 3, row.names = FALSE)

  out_file <- file.path(sweep_dir, paste0(file_stub, "_comparison_", sweep_date),
                         paste0(file_stub, "_bootstrap_comparison_MONTHCLUSTERED_", sweep_date, ".csv"))
  write.csv(boot_cmp, out_file, row.names = FALSE)
  cat("\nSaved to:", out_file, "\n")
}

recompute("loo",  "LOO")
recompute("waic", "WAIC")
