# test_exposure_response_functions.R
#
# Compares DLNM argvar / arglag specifications by sourcing
# Hierarch_StateSpace_Entomo_model.r once per configuration.
#
# Results land in:
#   results/Entomo/fitting/stan/test_exposure_response_functions/<predictor_spec>/<model_spec>/<run_suffix>/
#
# LOO comparison is written to the same root at the end.
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
  "/home/rita/data/Entomo/fitting/stan/test_exposure_response_functions_noHurr"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/test_exposure_response_functions"
}
dir.create(test_output_dir, recursive = TRUE, showWarnings = FALSE)

# Helper to build an argspec label
argspec_label <- function(spec) {
  if (!is.list(spec) || is.null(spec$fun)) return("default")
  if (spec$fun == "lin")    return("lin")
  if (spec$fun == "strata") return("strata")
  # Explicit knots (e.g. dlnm::logknots()) rather than df-based equal spacing:
  # spec$df is NULL here, so label by knot count instead of "nsNAdf".
  if (!is.null(spec$knots)) return(paste0(spec$fun, "LogK", length(spec$knots)))
  paste0(spec$fun, spec$df, "df")
}

# Helper to build per-variable arglag label
pervar_arglag_label <- function(named_arglag) {
  abbrevs <- c(
    total_precip               = "tp",
    precip_max_day_resid_on_tp = "resid",
    avg_VPD                    = "vpd"
  )
  parts <- sapply(names(named_arglag), function(v) {
    short <- if (v %in% names(abbrevs)) abbrevs[[v]] else v
    paste0(short, "_", argspec_label(named_arglag[[v]]))
  })
  paste0("lag_", paste(parts, collapse = "_"))
}


# max_lag isn't overridden by this script's .hierarch_cfg_override, so it
# stays whatever Hierarch_StateSpace_Entomo_model.r's own cfg$max_lag is --
# keep this in sync with that value (currently 5) so the log-knot positions
# computed here match what actually gets fit.
max_lag <- 5

# Log-spaced lag knots (nk=1 -> 3-column ns() basis, same dimensionality as
# the df=3 equal-spaced configs below -- dlnm::crossbasis() builds the lag
# basis with intercept=TRUE internally, so ncol = nk + 2). Front-loads spline
# flexibility toward short lags instead of spreading it evenly; see configs
# 8-9 below for the direct log-knots vs. equal-spaced-knots comparison.
logknots_lag <- list(fun = "ns", knots = dlnm::logknots(max_lag, nk = 1))

# nk=2 -> 4-column ns() basis (same dimensionality as the df=4 equal-spaced
# arglag tested manually), two log-spaced interior knots instead of one --
# see configs 14-15 for the direct nk=1 vs. nk=2 log-knots comparison.
logknots_lag2 <- list(fun = "ns", knots = dlnm::logknots(max_lag, nk = 2))

# =============================================================================
# Configuration grid
# =============================================================================
configs <- list(

  # 1 — simplest: linear lag, linear VPD
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 2),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "lin")
    ),
    dlnm_arglag = list(fun = "lin")
  ),

  # 2 — ns3 TP + VPD, ns2 RESID, linear lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(fun = "lin")
  ),

  # 3 — ns3 all, linear lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(fun = "lin")
  ),

  # 4 — ns3 all, ns2 lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(fun = "ns", df = 2)
  ),

  # 5 — ns3 all, ns3 lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(fun = "ns", df = 3)
  ),

  # 6 — ns3 TP + VPD, ns2 RESID, ns3 lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(fun = "ns", df = 3)
  ),

  # 7 — per-variable arglags
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    )
  ),

  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 2)
    ),
    dlnm_arglag = list(fun = "ns", df = 3)

  ),

  # 8 — log-knots counterpart of config 5 (ns3 all argvar): same argvar spec,
  # log-spaced lag knots instead of equal-spaced df=3, same 3-column
  # dimensionality -- direct LOO comparison against config 5 isolates the
  # effect of knot placement alone.
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = logknots_lag
  ),

  # 9 — log-knots counterpart of config 6 (ns3 TP+VPD, ns2 RESID argvar):
  # same pairing logic as config 8, against config 6 instead of config 5.
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = logknots_lag
  ),

  # 10 — ns4 all argvar (more exposure-response flexibility than configs
  # 3/5/8's ns3), equal-spaced df=3 lag. Same lag basis as config 5, so this
  # isolates the effect of exposure-dimension df alone (3 vs. 4).
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 4),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 4),
      avg_VPD                    = list(fun = "ns", df = 4)
    ),
    dlnm_arglag = list(fun = "ns", df = 3)
  ),

  # 11 — ns4 all argvar with log-knots lag: completes the 2x2 grid
  # {argvar df 3, 4} x {equal-spaced, log-spaced lag knots} formed by
  # configs 5, 8, 10, 11.
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 4),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 4),
      avg_VPD                    = list(fun = "ns", df = 4)
    ),
    dlnm_arglag = logknots_lag
  ),

  # 12 — per-variable log-knots counterpart of config 7 (same argvar spec):
  # total_precip keeps the standard equal-spaced df=3 lag (its effect is
  # expected to build gradually as breeding sites develop over several
  # weeks, so it may need flexibility across the full lag window), while
  # avg_VPD and precip_max_day_resid_on_tp switch to log-knots (their
  # effects are expected to act more acutely -- desiccation stress /
  # extreme-rainfall response -- so long-lag flexibility is more likely
  # noise than signal for these two). Direct LOO comparison against config
  # 7 isolates the effect of asymmetric (per-variable) knot placement.
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = logknots_lag,
      avg_VPD                    = logknots_lag
    )
  ),

  # 13 — linear baseline for precip_max_day_resid_on_tp: total_precip and
  # avg_VPD stay at ns3 (same as configs 5/6), RESID drops from ns2/ns3 down
  # to a plain linear exposure-response. Completes the argvar-flexibility
  # ladder for RESID (lin < ns2 < ns3 < ns4, i.e. configs 13, 6, 5, 10) --
  # if LOO doesn't favour the spline over this, RESID's effect is basically
  # linear and the extra flexibility isn't earning its keep.
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "lin"),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(fun = "ns", df = 3)
  ),

  # 14 — nk=2 log-knots counterpart of config 8 (same argvar as config 5):
  # tests whether a second log-spaced interior knot helps over the single-knot
  # version, direct LOO comparison against config 8 isolates knot count alone.
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = logknots_lag2
  ),

  # 15 — nk=2 log-knots counterpart of config 9 (same argvar as config 6):
  # same nk=1 vs. nk=2 comparison as config 14, against config 6/9's argvar
  # spec instead of config 5/8's.
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = logknots_lag2
  ), 

  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(
      total_precip               = logknots_lag2,
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3)
    )
  ),
  
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3)
    ),
    dlnm_arglag = list(
      total_precip               = logknots_lag2,
      precip_max_day_resid_on_tp = logknots_lag2,
      avg_VPD                    = list(fun = "ns", df = 3)
    )
  )
)

# =============================================================================
# Build run_suffix labels
# =============================================================================
make_run_suffix <- function(cfg_i, date_suffix) {
  av  <- cfg_i$dlnm_argvar
  al  <- cfg_i$dlnm_arglag

  arglag_is_per_var <- !is.null(names(al)) && any(names(al) %in% names(av))

  lag_label <- if (arglag_is_per_var) pervar_arglag_label(al) else paste0("lag_", argspec_label(al))

  paste0(
    date_suffix,
    "_TP_",    argspec_label(av$total_precip),
    "_RESID_", argspec_label(av$precip_max_day_resid_on_tp),
    "_VPD_",   argspec_label(av$avg_VPD),
    "_",       lag_label,
    "_ix_nonurban_x_tp"
  )
}

# =============================================================================
# Run all configurations
# =============================================================================
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
    lag_vars     = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp"),
    dlnm_vars    = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp"),
    numeric_vars = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers"),
    dlnm_ix_vars = list(
      list(binary_var = "is_urban", active_level = 0,
           dlnm_var   = "total_precip", label = "nonurban_x_tp")
    ),
    dlnm_argvar  = cfg_i$dlnm_argvar,
    dlnm_arglag  = cfg_i$dlnm_arglag,
    output_dir   = test_output_dir
  )
  .hierarch_run_suffix <- run_label
  loo_result           <- NULL   # clear stale value; Hierarch will overwrite if fit succeeds
  waic_result          <- NULL

  tryCatch(
    source(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"), local = FALSE),
    error = function(e) cat("ERROR in config", i, "post-processing:", conditionMessage(e), "\n(loo_result/waic_result collected before error if LOO/WAIC completed)\n")
  )

  if (exists("loo_result") && !is.null(loo_result)) {
    loo_list[[run_label]] <- loo_result
    cat("LOO stored for:", run_label, "\n")
    saveRDS(loo_list, file.path(test_output_dir, "loo_list_partial.rds"))
  } else {
    cat("WARNING: loo_result not found after config", i, "— skipping LOO for this run.\n")
  }
  if (exists("waic_result") && !is.null(waic_result)) {
    waic_list[[run_label]] <- waic_result
    cat("WAIC stored for:", run_label, "\n")
    saveRDS(waic_list, file.path(test_output_dir, "waic_list_partial.rds"))
  } else {
    cat("WARNING: waic_result not found after config", i, "— skipping WAIC for this run.\n")
  }

  # Capture block ids once, for the cluster bootstrap comparison at the end.
  # All 13 configs share the same lag_vars/dlnm_vars/response data (only the
  # argvar/arglag basis choice varies), so block assignment and row order
  # are identical across configs -- safe to capture from the first one that
  # succeeds rather than re-capturing (and overwriting) every iteration.
  if (!exists("block_ids_for_bootstrap") && exists("stan_data") && !is.null(stan_data$block))
    block_ids_for_bootstrap <- stan_data$block

  # Clean up override variables
  rm(".hierarch_cfg_override", ".hierarch_run_suffix", envir = globalenv())
}

# =============================================================================
# Criterion comparison (LOO, then WAIC)
# =============================================================================
# Shared by both: loo_compare() accepts a list of either loo() or waic()
# objects (same output structure, elpd_diff/se_diff columns named the same
# either way) -- called once per criterion below, never mixing the two lists
# in a single loo_compare() call. z_score makes the |z|>2 "meaningfully
# better" rule of thumb explicit instead of leaving elpd_diff/se_diff for the
# reader to divide by hand.
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

  comp_dir <- file.path(test_output_dir, paste0(file_stub, "_comparison_", date_suffix))
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
  invisible(file.remove(file.path(test_output_dir, paste0(file_stub, "_list_partial.rds"))))
}

write_criterion_comparison(loo_list,  "LOO",  "loo")
write_criterion_comparison(waic_list, "WAIC", "waic")

# =============================================================================
# Bootstrap comparison: shape-preserving CI + win probability
# =============================================================================
# Complements the se_diff-based comparison above with a resampling-based
# view that doesn't assume the total elpd difference is normally
# distributed -- see bootstrap_elpd_comparison() in helper_functions.r.
# Cluster (block) bootstrap if block ids were captured during the loop;
# falls back to per-observation resampling otherwise (understates
# uncertainty if the model has block-level structure, which this one does).
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

  comp_dir <- file.path(test_output_dir, paste0(file_stub, "_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)
  boot_file <- file.path(comp_dir, paste0(file_stub, "_bootstrap_comparison_", date_suffix, ".csv"))
  write.csv(boot_cmp, boot_file, row.names = FALSE)
  cat("\n", criterion_label, "bootstrap comparison saved to:", boot_file, "\n")
}

write_bootstrap_comparison(loo_list,  "LOO",  "loo")
write_bootstrap_comparison(waic_list, "WAIC", "waic")
