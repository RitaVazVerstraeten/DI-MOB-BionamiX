# run_season_interaction_sweep.R
#
# Model-selection sweep over:
#   - dlnm_arglag: ns(df=3) vs ns(df=4)
#   - max_lag: 5 vs 6
#   - is_rainy_season interaction: none / precip_max_day_resid_on_tp x season /
#     total_precip x season / both together
#     (is_rainy_season is added to unlagged_vars as a main effect whenever at
#     least one interaction is present; dropped entirely -- no main effect,
#     no interaction -- in the "none" arm)
# 2 x 2 x 4 = 16 configs total.
#
# Sources Hierarch_StateSpace_Entomo_model.r once per configuration, exactly
# like run_exposure_response_functions_sweep.R.
#
# Results land in:
#   <output root>/season_interaction_sweep/<predictor_spec>/<model_spec>/<run_suffix>/
#
# LOO/WAIC comparison + bootstrap comparison written to the same root at the end.
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

sweep_output_dir <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/season_interaction_sweep"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/season_interaction_sweep"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/season_interaction_sweep"
}
dir.create(sweep_output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Configuration grid
# =============================================================================
# Pinned explicitly (rather than left to Hierarch_StateSpace_Entomo_model.r's
# own cfg defaults) so this sweep's results stay reproducible even if the
# base file's defaults drift later -- same rationale as
# run_exposure_response_functions_sweep.R's override list.
lag_vars_fixed     <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
dlnm_vars_fixed    <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
numeric_vars_fixed <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi")
dlnm_argvar_fixed  <- list(
  total_precip                = list(fun = "ns", df = 3),
  avg_temp                     = list(fun = "ns", df = 3),
  precip_max_day_resid_on_tp  = list(fun = "ns", df = 3)
)

unlagged_no_season   <- c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_shortage", "water_containers")
unlagged_with_season <- c(unlagged_no_season, "is_rainy_season")

ix_resid <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "precip_max_day_resid_on_tp", label = "precip_resid_x_season")
ix_total <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "total_precip",               label = "tp_x_season")

interaction_arms <- list(
  none  = list(ix_name = "none",  dlnm_ix_vars = NULL, unlagged_vars = unlagged_no_season),
  resid = list(ix_name = "resid", dlnm_ix_vars = list(ix_resid), unlagged_vars = unlagged_with_season),
  total = list(ix_name = "total", dlnm_ix_vars = list(ix_total), unlagged_vars = unlagged_with_season),
  both  = list(ix_name = "both",  dlnm_ix_vars = list(ix_resid, ix_total), unlagged_vars = unlagged_with_season)
)

arglag_dfs <- c(3, 4)
max_lags   <- c(5, 6)

configs <- list()
for (arglag_df in arglag_dfs) {
  for (ml in max_lags) {
    for (arm in interaction_arms) {
      configs[[length(configs) + 1]] <- list(
        arglag_df     = arglag_df,
        max_lag       = ml,
        ix_name       = arm$ix_name,
        dlnm_ix_vars  = arm$dlnm_ix_vars,
        unlagged_vars = arm$unlagged_vars
      )
    }
  }
}

# =============================================================================
# Build run_suffix labels
# =============================================================================
make_run_suffix <- function(cfg_i, date_suffix) {
  paste0(
    date_suffix,
    "_arglagns", cfg_i$arglag_df, "df",
    "_lag", cfg_i$max_lag,
    "_ix", cfg_i$ix_name
  )
}

# =============================================================================
# Run all configurations
# =============================================================================
# Parsed once, up front -- see run_exposure_response_functions_sweep.R for why
# (avoids re-source()-ing mid-save during a long sweep).
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))

loo_list   <- list()
waic_list  <- list()
run_labels <- character(length(configs))

for (i in seq_along(configs)) {
  cfg_i      <- configs[[i]]
  run_label  <- make_run_suffix(cfg_i, date_suffix)
  run_labels[i] <- run_label

  cat("\n", strrep("=", 70), "\n")
  cat("CONFIG", i, "of", length(configs), ":", run_label, "\n")
  cat(strrep("=", 70), "\n\n")

  .hierarch_cfg_override <- list(
    lag_vars      = lag_vars_fixed,
    dlnm_vars     = dlnm_vars_fixed,
    numeric_vars  = numeric_vars_fixed,
    dlnm_argvar   = dlnm_argvar_fixed,
    dlnm_arglag   = list(fun = "ns", df = cfg_i$arglag_df),
    max_lag       = cfg_i$max_lag,
    dlnm_ix_vars  = cfg_i$dlnm_ix_vars,
    unlagged_vars = cfg_i$unlagged_vars,
    output_dir    = sweep_output_dir
  )
  .hierarch_run_suffix <- run_label
  loo_result           <- NULL   # clear stale value; Hierarch will overwrite if fit succeeds
  waic_result          <- NULL

  tryCatch(
    eval(model_exprs, envir = globalenv()),
    error = function(e) cat("ERROR in config", i, "post-processing:", conditionMessage(e), "\n(loo_result/waic_result collected before error if LOO/WAIC completed)\n")
  )

  if (exists("loo_result") && !is.null(loo_result)) {
    loo_list[[run_label]] <- loo_result
    cat("LOO stored for:", run_label, "\n")
    saveRDS(loo_list, file.path(sweep_output_dir, "loo_list_partial.rds"))
  } else {
    cat("WARNING: loo_result not found after config", i, "— skipping LOO for this run.\n")
  }
  if (exists("waic_result") && !is.null(waic_result)) {
    waic_list[[run_label]] <- waic_result
    cat("WAIC stored for:", run_label, "\n")
    saveRDS(waic_list, file.path(sweep_output_dir, "waic_list_partial.rds"))
  } else {
    cat("WARNING: waic_result not found after config", i, "— skipping WAIC for this run.\n")
  }

  # Capture block ids once, for the cluster bootstrap comparison at the end.
  # All 16 configs share the same response period/rows (max_lag only changes
  # how far the lag window reaches into the 2015 lead-in, not which
  # 2016-01-onward rows are observations) -- safe to capture from the first
  # config that succeeds rather than re-capturing every iteration.
  if (!exists("block_ids_for_bootstrap") && exists("stan_data") && !is.null(stan_data$block))
    block_ids_for_bootstrap <- stan_data$block

  # Clean up override variables
  rm(".hierarch_cfg_override", ".hierarch_run_suffix", envir = globalenv())
}

# =============================================================================
# Criterion comparison (LOO, then WAIC)
# =============================================================================
write_criterion_comparison <- function(result_list, criterion_label, file_stub) {
  if (length(result_list) < 2) {
    cat("Fewer than 2 successful", criterion_label, "results — skipping comparison.\n")
    return(invisible(NULL))
  }
  cat("\n", strrep("=", 70), "\n")
  cat(criterion_label, "COMPARISON\n")
  cat(strrep("=", 70), "\n\n")

  comp <- loo_compare(result_list)
  print(comp, simplify = FALSE, digits = 2)

  cmp_df <- as.data.frame(comp)
  cmp_df$z_score <- cmp_df$elpd_diff / cmp_df$se_diff
  cmp_df$z_score[cmp_df$elpd_diff == 0] <- 0
  cat("\nz-score (elpd_diff / se_diff):\n")
  print(cmp_df["z_score"], digits = 2)

  comp_dir <- file.path(sweep_output_dir, paste0(file_stub, "_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  comp_file <- file.path(comp_dir, paste0(file_stub, "_comparison_", date_suffix, ".txt"))
  comp_output <- capture.output({
    cat(criterion_label, "comparison —", date_suffix, "\n\n")
    cat("Models (in order):\n")
    for (i in seq_along(run_labels)) cat(sprintf("  %d. %s\n", i, run_labels[i]))
    cat("\n")
    print(comp, simplify = FALSE, digits = 2)
    cat("\nz-score (elpd_diff / se_diff):\n")
    print(cmp_df["z_score"], digits = 2)
  })
  writeLines(comp_output, comp_file)
  cat("\n", criterion_label, "comparison saved to:", comp_file, "\n")

  saveRDS(result_list, file.path(comp_dir, paste0(file_stub, "_list_", date_suffix, ".rds")))
  cat(criterion_label, "objects saved to:",
      file.path(comp_dir, paste0(file_stub, "_list_", date_suffix, ".rds")), "\n")
  invisible(file.remove(file.path(sweep_output_dir, paste0(file_stub, "_list_partial.rds"))))
}

write_criterion_comparison(loo_list,  "LOO",  "loo")
write_criterion_comparison(waic_list, "WAIC", "waic")

# =============================================================================
# Bootstrap comparison: shape-preserving CI + win probability
# =============================================================================
write_bootstrap_comparison <- function(result_list, criterion_label, file_stub) {
  if (length(result_list) < 2) {
    cat("Fewer than 2 successful", criterion_label, "results — skipping bootstrap comparison.\n")
    return(invisible(NULL))
  }
  cat("\n", strrep("=", 70), "\n")
  cat(criterion_label, "BOOTSTRAP COMPARISON (", if (exists("block_ids_for_bootstrap")) "block-clustered" else "per-observation, no block ids captured", ")\n")
  cat(strrep("=", 70), "\n\n")

  boot_cmp <- bootstrap_elpd_comparison(
    result_list,
    cluster_ids = if (exists("block_ids_for_bootstrap")) block_ids_for_bootstrap else NULL,
    n_boot = 4000
  )
  print(boot_cmp, digits = 3, row.names = FALSE)

  comp_dir <- file.path(sweep_output_dir, paste0(file_stub, "_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)
  boot_file <- file.path(comp_dir, paste0(file_stub, "_bootstrap_comparison_", date_suffix, ".csv"))
  write.csv(boot_cmp, boot_file, row.names = FALSE)
  cat("\n", criterion_label, "bootstrap comparison saved to:", boot_file, "\n")
}

write_bootstrap_comparison(loo_list,  "LOO",  "loo")
write_bootstrap_comparison(waic_list, "WAIC", "waic")