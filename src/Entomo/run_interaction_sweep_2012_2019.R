# run_interaction_sweep_2012_2019.R
#
# Model-selection sweep over three candidate DLNM interactions on the
# 2012-2019 NDXIbackfilled dataset, each tested for BOTH total_precip and its
# distribution residual (precip_max_day_resid_on_tp), same 4-arm design
# (none / resid / total / both) run_season_interaction_sweep.R uses for
# season alone:
#
#   - water_containers (continuous)        : tp x wc / resid x wc
#   - is_rainy_season  (binary, active=1)  : tp x season / resid x season
#   - is_urban         (binary, active=1)  : tp x urban / resid x urban
#
# RF screen support (results/Entomo/fitting/Random_Forest/2012_2019_NDXIbackfilled/
# rf_h_statistic_interactions.csv, re-run 2026-09-19 with both total_precip AND
# precip_max_day_resid_on_tp in the same interaction-screening forest):
#   - precip_max_day_resid_on_tp_lag5 x water_containers: H = 0.38 -- the
#     SINGLE STRONGEST pair of everything tested, driven by the residual at
#     the longest lag, not total_precip.
#   - total_precip_lag1 x water_containers: H = 0.31 (strong, borderline).
#     Rest of the water_containers pairs: H = 0.06-0.21 (weak-to-moderate).
#   - is_urban: NOTHING crosses the 0.3 "strong" threshold in this full run --
#     max is tp_resid_lag3 x is_urban at H = 0.19 (mild-moderate). An earlier,
#     smaller interaction-screening forest (without the residual columns)
#     showed a spike of H = 0.34 for total_precip_lag4 x is_urban; that did
#     NOT reproduce here (now H = 0.12) -- treat that spike as noise from the
#     under-specified earlier forest, not a real signal.
#   - is_rainy_season interactions: still NOT screened by RF at all (season
#     was never an interaction_targets entry) -- these three arms remain
#     theory-driven only.
#   -> Net effect on priority: water_containers axis has gotten stronger
#      support (especially its resid arm); urban axis has gotten weaker.
#      Consider dropping the urban axis entirely if you need to cut this
#      sweep down further.
#
# Runtime warning: a SINGLE fit on this dataset (roughly 2x the response rows
# of the 2016-2019 fits) has been observed to take 8+ hours locally. This
# sweep runs 1 baseline + 3 axes x 2 non-baseline arms (resid, total -- never
# both together, see make_axis() below) = 7 fits sequentially -- budget for
# several days of wall-clock time, or prioritise by cutting axes/arms below
# (water_containers first, given the RF support above; urban and season
# last/optional given weak-to-nonexistent RF signal). Consider running only
# the water_containers axis (2 extra fits) first and deciding whether to
# continue.
#
# Sources Hierarch_StateSpace_Entomo_model.r once per configuration, same
# pattern as run_season_interaction_sweep.R / run_exposure_response_functions_sweep.R.
# keep_raw_draws = FALSE on every config (see run_2012_2019_NDXIbackfilled.R) --
# this dataset's checkpoint+raw CSVs are multi-GB per run; 7 of them would
# fill the disk.
#
# Results land in:
#   <output root>/interaction_sweep_2012_2019/<predictor_spec>/<model_spec>/<run_suffix>/
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
  "/home/rita/data/Entomo/fitting/stan/interaction_sweep_2012_2019"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/interaction_sweep_2012_2019"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/interaction_sweep_2012_2019"
}
dir.create(sweep_output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Fixed config -- matches run_2012_2019_NDXIbackfilled.R exactly (pinned
# explicitly, not left to Hierarch_StateSpace_Entomo_model.r's own defaults,
# same rationale as run_season_interaction_sweep.R).
# =============================================================================
data_file_name_fixed <- "env_epi_entomo_data_per_CMF_2012_01_to_2019_12_NDXIbackfilled_noColinnearity.csv"
response_start_fixed <- "2012_01"
max_lag_fixed         <- 5

lag_vars_fixed     <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
dlnm_vars_fixed    <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
numeric_vars_fixed <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp",
                         "water_containers", "HFP_urbanization", "mean_ndvi")
dlnm_argvar_fixed  <- list(
  total_precip               = list(fun = "ns", df = 3),
  avg_VPD                    = list(fun = "ns", df = 2),
  precip_max_day_resid_on_tp = list(fun = "ns", df = 3)
)
dlnm_arglag_fixed <- list(fun = "ns", df = 3)

# Base unlagged main effects -- matches run_2012_2019_NDXIbackfilled.R.
# water_containers is already here (continuous main effect always present,
# with or without its interaction); is_rainy_season/is_urban are NOT --
# each gets added as a main effect only in the arms that also test it as an
# interaction modifier, same convention as run_season_interaction_sweep.R.
unlagged_base <- c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_containers")

# =============================================================================
# Interaction axis definitions
# =============================================================================
# Each axis: a modifier variable, whether it's continuous or binary (changes
# the dlnm_ix_vars spec shape), and a 2-arm interaction set built from it --
# resid-only or total-only, never both together in the same fit (deliberately
# no "both" arm: tp and its residual share the same total_precip signal by
# construction, so testing them simultaneously muddies which one is actually
# doing the work; each is tested against the plain "none" baseline instead).
# continuous_df = 2 (the model's own default) lets the effect-modification
# curve bend rather than being forced linear -- see helper_functions.r's
# build_dlnm_stan_data() interaction spec docs.
make_axis <- function(axis_name, modifier_var, modifier_type = c("continuous", "binary"), active_level = 1) {
  modifier_type <- match.arg(modifier_type)
  ix_spec <- function(dlnm_var, label) {
    if (modifier_type == "continuous") {
      list(continuous_var = modifier_var, dlnm_var = dlnm_var, label = label, continuous_df = 2)
    } else {
      list(binary_var = modifier_var, active_level = active_level, dlnm_var = dlnm_var, label = label)
    }
  }
  ix_resid <- ix_spec("precip_max_day_resid_on_tp", paste0("precip_resid_x_", axis_name))
  ix_total <- ix_spec("total_precip",               paste0("tp_x_", axis_name))

  # water_containers is already a main effect in unlagged_base -- don't add
  # it again. is_rainy_season/is_urban are not, so add them only when at
  # least one interaction using them is present.
  needs_main_effect <- !(modifier_var %in% unlagged_base)
  unlagged_with_modifier <- if (needs_main_effect) c(unlagged_base, modifier_var) else unlagged_base

  list(
    axis_name = axis_name,
    arms = list(
      resid = list(ix_name = paste0(axis_name, "_resid"), dlnm_ix_vars = list(ix_resid), unlagged_vars = unlagged_with_modifier),
      total = list(ix_name = paste0(axis_name, "_total"), dlnm_ix_vars = list(ix_total), unlagged_vars = unlagged_with_modifier)
    )
  )
}

axes <- list(
  make_axis("wc",     "water_containers", "continuous"),
  make_axis("season", "is_rainy_season",  "binary", active_level = 1),
  make_axis("urban",  "is_urban",         "binary", active_level = 1)
)

# =============================================================================
# Build the config list: 1 shared baseline ("none", no interactions at all)
# + 2 non-baseline arms (resid, total -- never both together) per axis
# = 1 + 3*2 = 7 configs total.
# =============================================================================
configs <- list(
  none = list(ix_name = "none", dlnm_ix_vars = NULL, unlagged_vars = unlagged_base)
)
for (axis in axes) {
  for (arm in axis$arms) {
    configs[[arm$ix_name]] <- arm
  }
}

cat(sprintf("Interaction sweep: %d configs (1 baseline + %d axes x 2 arms)\n",
            length(configs), length(axes)))

# =============================================================================
# Run all configurations
# =============================================================================
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))

loo_list   <- list()
waic_list  <- list()
run_labels <- character(length(configs))

for (i in seq_along(configs)) {
  cfg_i     <- configs[[i]]
  run_label <- paste0(date_suffix, "_ix", cfg_i$ix_name)
  run_labels[i] <- run_label

  cat("\n", strrep("=", 70), "\n")
  cat("CONFIG", i, "of", length(configs), ":", run_label, "\n")
  cat(strrep("=", 70), "\n\n")

  .hierarch_cfg_override <- list(
    data_file_name = data_file_name_fixed,
    response_start = response_start_fixed,
    max_lag        = max_lag_fixed,
    lag_vars       = lag_vars_fixed,
    dlnm_vars      = dlnm_vars_fixed,
    numeric_vars   = numeric_vars_fixed,
    dlnm_argvar    = dlnm_argvar_fixed,
    dlnm_arglag    = dlnm_arglag_fixed,
    dlnm_ix_vars   = cfg_i$dlnm_ix_vars,
    unlagged_vars  = cfg_i$unlagged_vars,
    output_dir     = sweep_output_dir,
    # Base script default is parallel_chains = 1 on a non-compute-node host
    # (this machine isn't frietjes/stoofvlees) -- override to run all 4
    # chains concurrently instead of sequentially. This machine has 12 cores
    # (nproc), so 4 leaves 8 free for the OS/other work.
    parallel_chains = 4,
    # This dataset's checkpoint.rds + raw chain CSVs run several GB per fit
    # (see run_2012_2019_NDXIbackfilled.R) -- with 7 fits in this sweep,
    # keeping them all would fill the disk. keep_raw_draws = FALSE skips the
    # checkpoint save AND deletes the raw per-chain CSVs at the end of each
    # config (see Hierarch_StateSpace_Entomo_model.r), so no chain output is
    # left on disk once a config finishes.
    keep_raw_draws = FALSE
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
  # All 7 configs share the same response period/rows (only dlnm_ix_vars/
  # unlagged_vars change between them) -- safe to capture from the first
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
