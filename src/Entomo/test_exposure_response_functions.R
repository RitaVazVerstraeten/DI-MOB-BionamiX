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
  "/home/rita/data/Entomo/fitting/stan/test_exposure_response_functions"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/test_exposure_response_functions"
}
dir.create(test_output_dir, recursive = TRUE, showWarnings = FALSE)

# Helper to build an argspec label
argspec_label <- function(spec) {
  if (is.null(spec)) return("default")
  if (spec$fun == "lin")    return("lin")
  if (spec$fun == "strata") return("strata")
  paste0(spec$fun, spec$df, "df")
}

# Helper to build per-variable arglag label
pervar_arglag_label <- function(named_arglag) {
  abbrevs <- c(
    total_precip               = "TP",
    precip_max_day_resid_on_tp = "RESID",
    avg_VPD                    = "VPD",
    hurricane_within_120km     = "HURR"
  )
  parts <- sapply(names(named_arglag), function(v) {
    short <- if (v %in% names(abbrevs)) abbrevs[[v]] else v
    paste0(short, "_", argspec_label(named_arglag[[v]]))
  })
  paste0("lag_", paste(parts, collapse = "_"))
}

hurr_argvar <- list(fun = "strata", breaks = 0.5)

# =============================================================================
# Configuration grid
# =============================================================================
configs <- list(

  # 1 — simplest: linear lag, linear VPD
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 2),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "lin"),
      hurricane_within_120km     = hurr_argvar
    ),
    dlnm_arglag = list(fun = "lin")
  ),

  # 2 — ns3 TP + VPD, ns2 RESID, linear lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3),
      hurricane_within_120km     = hurr_argvar
    ),
    dlnm_arglag = list(fun = "lin")
  ),

  # 3 — ns3 all, linear lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3),
      hurricane_within_120km     = hurr_argvar
    ),
    dlnm_arglag = list(fun = "lin")
  ),

  # 4 — ns3 all, ns2 lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3),
      hurricane_within_120km     = hurr_argvar
    ),
    dlnm_arglag = list(fun = "ns", df = 2)
  ),

  # 5 — ns3 all, ns3 lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 3),
      avg_VPD                    = list(fun = "ns", df = 3),
      hurricane_within_120km     = hurr_argvar
    ),
    dlnm_arglag = list(fun = "ns", df = 3)
  ),

  # 6 — ns3 TP + VPD, ns2 RESID, ns3 lag
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3),
      hurricane_within_120km     = hurr_argvar
    ),
    dlnm_arglag = list(fun = "ns", df = 3)
  ),

  # 7 — per-variable arglags
  list(
    dlnm_argvar = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3),
      hurricane_within_120km     = hurr_argvar
    ),
    dlnm_arglag = list(
      total_precip               = list(fun = "ns", df = 3),
      precip_max_day_resid_on_tp = list(fun = "ns", df = 2),
      avg_VPD                    = list(fun = "ns", df = 3),
      hurricane_within_120km     = list(fun = "ns", df = 2)
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
    "_HURR_strata_",
    lag_label
  )
}

# =============================================================================
# Run all configurations
# =============================================================================
loo_list   <- list()
run_labels <- character(length(configs))

for (i in seq_along(configs)) {
  cfg_i      <- configs[[i]]
  run_label  <- make_run_suffix(cfg_i, date_suffix)
  run_labels[i] <- run_label

  cat("\n", strrep("=", 70), "\n")
  cat("CONFIG", i, "of", length(configs), ":", run_label, "\n")
  cat(strrep("=", 70), "\n\n")

  .hierarch_cfg_override <- list(
    lag_vars     = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "hurricane_within_120km"),
    dlnm_vars    = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "hurricane_within_120km"),
    numeric_vars = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers"),
    dlnm_ix_vars = NULL,
    dlnm_argvar  = cfg_i$dlnm_argvar,
    dlnm_arglag  = cfg_i$dlnm_arglag,
    output_dir   = test_output_dir
  )
  .hierarch_run_suffix <- run_label
  loo_result           <- NULL   # clear stale value; Hierarch will overwrite if fit succeeds

  source(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"), local = FALSE)

  if (exists("loo_result") && !is.null(loo_result)) {
    loo_list[[run_label]] <- loo_result
    cat("LOO stored for:", run_label, "\n")
  } else {
    cat("WARNING: loo_result not found after config", i, "— skipping LOO for this run.\n")
  }

  # Clean up override variables
  rm(".hierarch_cfg_override", ".hierarch_run_suffix", envir = globalenv())
}

# =============================================================================
# LOO comparison
# =============================================================================
if (length(loo_list) >= 2) {
  cat("\n", strrep("=", 70), "\n")
  cat("LOO COMPARISON\n")
  cat(strrep("=", 70), "\n\n")

  loo_comp <- loo_compare(loo_list)
  print(loo_comp, simplify = FALSE, digits = 2)

  comp_dir <- file.path(test_output_dir, paste0("loo_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  comp_file <- file.path(comp_dir, paste0("loo_comparison_", date_suffix, ".txt"))
  comp_output <- capture.output({
    cat("LOO comparison —", date_suffix, "\n\n")
    cat("Models (in order):\n")
    for (i in seq_along(run_labels)) cat(sprintf("  %d. %s\n", i, run_labels[i]))
    cat("\n")
    print(loo_comp, simplify = FALSE, digits = 2)
  })
  writeLines(comp_output, comp_file)
  cat("\nLOO comparison saved to:", comp_file, "\n")

  saveRDS(loo_list, file.path(comp_dir, paste0("loo_list_", date_suffix, ".rds")))
  cat("LOO objects saved to:", file.path(comp_dir, paste0("loo_list_", date_suffix, ".rds")), "\n")
} else {
  cat("Fewer than 2 successful LOO results — skipping comparison.\n")
}
