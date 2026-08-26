# run_argvar_sweep.R
#
# Follow-up to run_season_interaction_sweep.R and run_boundary_knots_test.R:
# instead of sweeping arglag_df/max_lag, this sweeps each lagged variable's
# own argvar_df (total_precip, avg_VPD, precip_max_day_resid_on_tp) and
# whether the precip_resid_x_season interaction is present, arglag_df fixed
# at 3 and max_lag fixed at 5 throughout. Motivated by the parameter-budget
# discussion: the DLNM cross-basis parameters for all three lagged variables
# (plus the interaction) draw on the SAME ~48-month shared pool of
# independent weather variation, not 48 months each -- see
# doc/DLNM_Residual_Data_Support.tex and Bayesian_AR1_ICAR_model_explained.tex
# Section 2.6 for the full derivation.
#
# 7 named scenarios (argvar df per variable -- arglag df -- interaction?):
#   a: tp=3, VPD=3, resid=3           | arglag=3 | interaction=yes  (current default)
#   b: tp=3, VPD=3, resid=2           | arglag=3 | interaction=yes
#   c: tp=3, VPD=2, resid=2           | arglag=3 | interaction=yes
#   d: tp=3, VPD=3, resid=3           | arglag=3 | interaction=no   (base DLNM, no season)
#   e: tp=3, VPD=2, resid=3           | arglag=3 | interaction=no
#   f: tp=2, VPD=2, resid=2           | arglag=3 | interaction=yes
#   g: tp=2, VPD=lin, resid=2         | arglag=3 | interaction=yes
#
# Interaction=no follows the same convention as run_season_interaction_sweep.R's
# "none" arm: is_rainy_season is dropped entirely (no main effect, no
# interaction), not just the DLNM interaction switched off.
#
# COST WARNING: 7 full model fits, ~1-1.5h each based on prior runs on this
# branch -- roughly 7-10.5h sequential. Results save incrementally after each
# scenario (loo_list_partial.rds / waic_list_partial.rds), so this can be
# safely interrupted and resumed the same way as the other sweep scripts
# (existing chain CSVs in a run's output_dir are reloaded rather than
# re-sampled -- see Hierarch_StateSpace_Entomo_model.r's existing_csv check).
#
# Results land in:
#   <output root>/argvar_sweep/<predictor_spec>/<model_spec>/<run_suffix>/
# LOO/WAIC comparison + bootstrap comparison written to the sweep root at the end.
#
# Usage: Rscript run_argvar_sweep.R
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
  "/home/rita/data/Entomo/fitting/stan/argvar_sweep"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/argvar_sweep"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/argvar_sweep"
}
sweep_output_dir <- path.expand(sweep_output_dir)
dir.create(sweep_output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Fixed config -- same lagged variables / unlagged sets as
# run_season_interaction_sweep.R; only argvar_df per variable and the
# presence of the interaction vary across scenarios.
# =============================================================================
lag_vars_fixed     <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
dlnm_vars_fixed    <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
numeric_vars_fixed <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi")

unlagged_no_season   <- c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_shortage", "water_containers")
unlagged_with_season <- c(unlagged_no_season, "is_rainy_season")

ix_resid <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "precip_max_day_resid_on_tp", label = "precip_resid_x_season")

arglag_df_fixed <- 3
max_lag_fixed   <- 5

# argvar_df per variable, per scenario -- "lin" instead of a number switches
# that variable to fun="lin" (no df). Order/letters match the table this
# script was requested from.
scenario_defs <- list(
  a = list(argvar = list(total_precip = 3, avg_VPD = 3,   precip_max_day_resid_on_tp = 3), interaction = TRUE),
  b = list(argvar = list(total_precip = 3, avg_VPD = 3,   precip_max_day_resid_on_tp = 2), interaction = TRUE),
  c = list(argvar = list(total_precip = 3, avg_VPD = 2,   precip_max_day_resid_on_tp = 2), interaction = TRUE),
  d = list(argvar = list(total_precip = 3, avg_VPD = 3,   precip_max_day_resid_on_tp = 3), interaction = FALSE),
  e = list(argvar = list(total_precip = 3, avg_VPD = 2,   precip_max_day_resid_on_tp = 3), interaction = FALSE),
  f = list(argvar = list(total_precip = 2, avg_VPD = 2,   precip_max_day_resid_on_tp = 2), interaction = TRUE),
  g = list(argvar = list(total_precip = 2, avg_VPD = "lin", precip_max_day_resid_on_tp = 2), interaction = TRUE)
)

build_dlnm_argvar <- function(spec) {
  lapply(spec, function(v) if (identical(v, "lin")) list(fun = "lin") else list(fun = "ns", df = v))
}

# Short per-variable df tag for the run_suffix, e.g. "tp3vpd3resid2" -- "lin"
# stays as the literal word rather than a number so it's unambiguous in the
# folder name.
argvar_tag <- function(spec) {
  paste0("tp", spec$total_precip, "vpd", spec$avg_VPD, "resid", spec$precip_max_day_resid_on_tp)
}

configs <- list()
for (nm in names(scenario_defs)) {
  sc <- scenario_defs[[nm]]
  configs[[nm]] <- list(
    label         = nm,
    dlnm_argvar   = build_dlnm_argvar(sc$argvar),
    dlnm_ix_vars  = if (sc$interaction) list(ix_resid) else NULL,
    unlagged_vars = if (sc$interaction) unlagged_with_season else unlagged_no_season,
    run_suffix    = paste0(date_suffix, "_scenario", nm, "_", argvar_tag(sc$argvar),
                            "_arglag", arglag_df_fixed, "_ix", if (sc$interaction) "resid" else "none")
  )
}
cat(sprintf("%d argvar-sweep scenarios to run.\n", length(configs)))

# =============================================================================
# Run all configurations
# =============================================================================
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))

loo_list   <- list()
waic_list  <- list()
run_labels <- character(length(configs))

for (i in seq_along(configs)) {
  cfg_i      <- configs[[i]]
  run_label  <- cfg_i$run_suffix
  run_labels[i] <- run_label

  cat("\n", strrep("=", 70), "\n")
  cat("SCENARIO", cfg_i$label, "(", i, "of", length(configs), "):", run_label, "\n")
  cat(strrep("=", 70), "\n\n")

  .hierarch_cfg_override <- list(
    lag_vars      = lag_vars_fixed,
    dlnm_vars     = dlnm_vars_fixed,
    numeric_vars  = numeric_vars_fixed,
    dlnm_argvar   = cfg_i$dlnm_argvar,
    dlnm_arglag   = list(fun = "ns", df = arglag_df_fixed),
    max_lag       = max_lag_fixed,
    dlnm_ix_vars  = cfg_i$dlnm_ix_vars,
    unlagged_vars = cfg_i$unlagged_vars,
    output_dir    = sweep_output_dir
  )
  .hierarch_run_suffix <- run_label
  loo_result           <- NULL   # clear stale value; Hierarch will overwrite if fit succeeds
  waic_result          <- NULL

  tryCatch(
    eval(model_exprs, envir = globalenv()),
    error = function(e) cat("ERROR in scenario", cfg_i$label, "post-processing:", conditionMessage(e), "\n(loo_result/waic_result collected before error if LOO/WAIC completed)\n")
  )

  if (exists("loo_result") && !is.null(loo_result)) {
    loo_list[[run_label]] <- loo_result
    cat("LOO stored for:", run_label, "\n")
    saveRDS(loo_list, file.path(sweep_output_dir, "loo_list_partial.rds"))
  } else {
    cat("WARNING: loo_result not found after scenario", cfg_i$label, "— skipping LOO for this run.\n")
  }
  if (exists("waic_result") && !is.null(waic_result)) {
    waic_list[[run_label]] <- waic_result
    cat("WAIC stored for:", run_label, "\n")
    saveRDS(waic_list, file.path(sweep_output_dir, "waic_list_partial.rds"))
  } else {
    cat("WARNING: waic_result not found after scenario", cfg_i$label, "— skipping WAIC for this run.\n")
  }

  # Capture block ids once, for the cluster bootstrap comparison at the end.
  if (!exists("block_ids_for_bootstrap") && exists("stan_data") && !is.null(stan_data$block))
    block_ids_for_bootstrap <- stan_data$block

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
    cat("Scenarios (in order):\n")
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
