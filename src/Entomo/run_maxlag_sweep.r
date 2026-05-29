# =============================================================================
# run_maxlag_sweep.r
#
# Runs Hierarch_StateSpace_Entomo_model.r repeatedly over a range of max_lag
# values. All other model settings (kappa, AR structure, predictors, etc.) are
# taken unchanged from the main script. For each max_lag, results are written to:
#   file.path(cfg$output_dir, lag_sweep_subdir)/
#     <predictor_spec>/..._lag<max_lag>_.../
#
# Usage:
#   Rscript run_maxlag_sweep.r
#   (or source("run_maxlag_sweep.r") from an R session in the same directory)
# =============================================================================

# ---------------------------------------------------------------------------
# SWEEP SETTINGS — adjust as needed
# ---------------------------------------------------------------------------
max_lag_values    <- c(2, 4, 5, 6, 12)       # lag values to test
lag_sweep_subdir  <- "lag_tests"         # subdirectory appended to cfg$output_dir

stan_output_dir <- if (Sys.info()["nodename"] == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan"
}
lag_tests_dir <- file.path(stan_output_dir, lag_sweep_subdir)

# Path to the Stan model script (assumed to be in the same directory as this script)
`%||%` <- function(a, b) if (!is.null(a)) a else b
stan_script <- file.path(dirname(sys.frame(1)$ofile %||% "."), "Hierarch_StateSpace_Entomo_model.r")
if (!file.exists(stan_script)) {
  stan_script <- "Hierarch_StateSpace_Entomo_model.r"
}
if (!file.exists(stan_script)) {
  stop("Hierarch_StateSpace_Entomo_model.r not found. Run this script from the same ",
       "directory, or set the stan_script path explicitly above.")
}

# ---------------------------------------------------------------------------
# Read the Stan model script once; patch per iteration
# ---------------------------------------------------------------------------
glmm_lines <- readLines(stan_script)

dir.create(lag_tests_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Sweep log
# ---------------------------------------------------------------------------
sweep_log <- list()
n_total   <- length(max_lag_values)

cat("=============================================================\n")
cat("  max_lag sweep: Hierarch_StateSpace_Entomo_model.r x", n_total, "runs\n")
cat("  max_lag values:", paste(max_lag_values, collapse = ", "), "\n")
cat("  Output base:  ", lag_tests_dir, "\n")
cat("=============================================================\n\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for (i in seq_along(max_lag_values)) {

  lag_val <- max_lag_values[i]
  cat("-------------------------------------------------------------\n")
  cat(sprintf("[%d/%d]  max_lag = %d\n", i, n_total, lag_val))
  cat("-------------------------------------------------------------\n")

  # -- Patch 0: fix script_dir and skip renv::restore --------------------------
  script_dir_val <- dirname(normalizePath(stan_script))
  patched <- gsub(
    pattern     = "renv::restore\\(project = script_dir, prompt = FALSE\\)",
    replacement = paste0('script_dir <- "', script_dir_val, '" # fixed by sweep'),
    x           = glmm_lines,
    perl        = TRUE
  )

  # -- Patch 1: replace max_lag in cfg -----------------------------------------
  patched <- gsub(
    pattern     = "(\\bmax_lag\\s*=\\s*)[0-9]+",
    replacement = paste0("\\1", lag_val),
    x           = patched,
    perl        = TRUE
  )

  # -- Patch 2: redirect output_dir into lag_sweep_subdir ----------------------
  patched <- gsub(
    pattern     = '("[^"]*fitting[/\\\\]stan[^"]*")',
    replacement = paste0('file.path(\\1, "', lag_sweep_subdir, '")'),
    x           = patched,
    perl        = TRUE
  )

  # -- Write patched script to a temp file --------------------------------------
  tmp_file <- tempfile(pattern = sprintf("stan_lag%d_", lag_val), fileext = ".r")
  writeLines(patched, tmp_file)
  on.exit(unlink(tmp_file), add = TRUE)

  # -- Run the patched script ---------------------------------------------------
  t_start <- proc.time()
  run_ok  <- tryCatch({
    source(tmp_file, local = FALSE)
    TRUE
  }, error = function(e) {
    cat(sprintf("\n[ERROR] max_lag = %d failed:\n  %s\n\n", lag_val, conditionMessage(e)))
    FALSE
  })
  elapsed <- (proc.time() - t_start)[["elapsed"]]

  unlink(tmp_file)
  on.exit(NULL)

  sweep_log[[i]] <- list(
    max_lag = lag_val,
    success = run_ok,
    elapsed = elapsed
  )

  cat(sprintf("\nmax_lag = %d  |  status: %s  |  time: %.1f s\n\n",
              lag_val,
              if (run_ok) "OK" else "FAILED",
              elapsed))
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
cat("=============================================================\n")
cat("  max_lag sweep complete\n")
cat("=============================================================\n")
log_df <- do.call(rbind, lapply(sweep_log, as.data.frame))
print(log_df)

log_file <- file.path(lag_tests_dir,
                      paste0("sweep_log_", format(Sys.Date(), "%Y%m%d"), ".csv"))
write.csv(log_df, log_file, row.names = FALSE)
cat("\nSweep log saved to:", log_file, "\n")
