# run_boundary_knots_test.R
#
# One-off test (not part of the main season_interaction_sweep grid): does
# constraining precip_max_day_resid_on_tp's argvar Boundary.knots to a
# well-supported range remove the lag-5 spike in the precip_resid_x_season
# DLNM surface, without materially changing model fit (LOO/WAIC)?
#
# Background: the sweep's "(ref)" surface for precip_resid_x_season showed a
# sharp spike at (residual ~35, lag 5) driven by essentially ONE month
# (2017_01) -- see the observation-density heatmaps in
# Descriptive_Statistics_Environmental_Variables.rmd. ns()'s cross-basis
# currently places its argvar boundary knots at the literal min/max of the
# data, so the full flexible spline shape gets fit all the way out to that
# single-month extreme. Pinning Boundary.knots inside that range instead
# forces linear (not free-curving) behaviour beyond it -- the extreme month
# still gets a predicted effect, it just can't independently bend the curve
# based on 1-3 months' worth of information.
#
# Boundary.knots values below (in STANDARDIZED/z-score units, since
# build_dlnm_stan_data() z-scores dlnm_vars before building the cross-basis)
# are the 10th/90th percentile of precip_max_day_resid_on_tp computed
# directly from the actual CMF-level model input file
# (env_epi_entomo_data_per_CMF_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv):
#   raw 10th/90th pctile: -12.83 / 23.42 mm
#   mean = 5.34e-16, sd = 14.71 (already ~centered in the source file)
#   z-score 10th/90th:    -0.872 / 1.592
# Recompute if the underlying data file changes.
#
# Runs ONE config twice (same arm/df/lag as the rest of this branch's work --
# arglag ns(df=4), max_lag=5, "resid" interaction arm only): once
# unconstrained (matches the main sweep exactly) and once with the boundary
# constraint, so the two DLNM interaction surface plots are directly
# comparable side by side. LOO/WAIC also saved for a numeric fit comparison,
# though the real check here is visual (does the spike go away).
#
# Usage: Rscript run_boundary_knots_test.R
# =============================================================================

library(loo)

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

date_suffix <- format(Sys.Date(), "%Y%m%d")
hostname    <- Sys.info()["nodename"]

test_output_dir <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/boundary_knots_test"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/boundary_knots_test"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/boundary_knots_test_resid_2df"
}
test_output_dir <- path.expand(test_output_dir)
dir.create(test_output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Fixed config -- identical to run_season_interaction_sweep.R's "resid" arm,
# arglag ns(df=4), max_lag=5 (matches this branch's other current work)
# =============================================================================
lag_vars_fixed     <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
dlnm_vars_fixed    <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
numeric_vars_fixed <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi")

dlnm_argvar_unconstrained <- list(
  total_precip                = list(fun = "ns", df = 3),
  avg_VPD                     = list(fun = "ns", df = 3),
  precip_max_day_resid_on_tp  = list(fun = "ns", df = 2)
)

resid_boundary_z <- c(-0.872, 1.592)   # see header comment for derivation
dlnm_argvar_boundary_knots <- modifyList(dlnm_argvar_unconstrained, list(
  precip_max_day_resid_on_tp = list(fun = "ns", df = 2, Boundary.knots = resid_boundary_z)
))

unlagged_with_season <- c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_shortage", "water_containers", "is_rainy_season")

ix_resid <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "precip_max_day_resid_on_tp", label = "precip_resid_x_season")

test_variants <- list(
  unconstrained   = list(label = "unconstrained",   dlnm_argvar = dlnm_argvar_unconstrained),
  boundary_knots  = list(label = "boundary_knots",  dlnm_argvar = dlnm_argvar_boundary_knots)
)

arglag_df <- 4
max_lag   <- 5

# =============================================================================
# Run both variants
# =============================================================================
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))

loo_list   <- list()
waic_list  <- list()

for (variant_name in names(test_variants)) {
  variant   <- test_variants[[variant_name]]
  run_label <- paste0(date_suffix, "_arglagns", arglag_df, "df_lag", max_lag, "_ixresid_", variant$label)

  cat("\n", strrep("=", 70), "\n")
  cat("VARIANT:", run_label, "\n")
  cat(strrep("=", 70), "\n\n")

  .hierarch_cfg_override <- list(
    lag_vars      = lag_vars_fixed,
    dlnm_vars     = dlnm_vars_fixed,
    numeric_vars  = numeric_vars_fixed,
    dlnm_argvar   = variant$dlnm_argvar,
    dlnm_arglag   = list(fun = "ns", df = arglag_df),
    max_lag       = max_lag,
    dlnm_ix_vars  = list(ix_resid),
    unlagged_vars = unlagged_with_season,
    output_dir    = test_output_dir
  )
  .hierarch_run_suffix <- run_label
  loo_result           <- NULL
  waic_result          <- NULL

  tryCatch(
    eval(model_exprs, envir = globalenv()),
    error = function(e) cat("ERROR in variant", run_label, ":", conditionMessage(e), "\n")
  )

  if (exists("loo_result") && !is.null(loo_result)) {
    loo_list[[run_label]] <- loo_result
    saveRDS(loo_list, file.path(test_output_dir, "loo_list_partial.rds"))
  }
  if (exists("waic_result") && !is.null(waic_result)) {
    waic_list[[run_label]] <- waic_result
    saveRDS(waic_list, file.path(test_output_dir, "waic_list_partial.rds"))
  }
}

cat("\nDone. Compare the dlnm_ix_3d_precip_resid_x_season_ref_logscale_*.png\n")
cat("plots between the two run_label subdirectories under:\n  ", test_output_dir, "\n")
if (length(loo_list) == length(test_variants)) {
  print(loo::loo_compare(loo_list))
}
