# =============================================================================
# run_kappa_sweep.r
#
# Runs GLMM.r repeatedly over a range of kappa values (0 to 5).
# For each kappa, results are written to:
#   file.path(cfg$output_dir, kappa_sweep_subdir)/
#     <predictor_spec>/..._k<kappa>_.../
#
# Usage:
#   Rscript run_kappa_sweep.r
#   (or source("run_kappa_sweep.r") from an R session in the same directory)
# =============================================================================

# ---------------------------------------------------------------------------
# SWEEP SETTINGS — adjust these as needed
# ---------------------------------------------------------------------------
kappa_values      <- seq(0, 5, by = 1)   # e.g. 0, 1, 2, 3, 4, 5
                                          # Use seq(0, 5, by = 0.5) for finer steps
kappa_sweep_subdir <- "kappa_tests"       # subdirectory appended to cfg$output_dir

# Derive the sweep root the same way GLMM.r resolves cfg$output_dir, then
# append kappa_sweep_subdir — keeping kappa_tests *inside* cfg$output_dir.
glmm_output_dir <- if (Sys.info()["nodename"] == "frietjes") {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
}
kappa_tests_dir <- file.path(glmm_output_dir, kappa_sweep_subdir)

# Path to GLMM.r (assumed to be in the same directory as this script)
glmm_script <- file.path(dirname(sys.frame(1)$ofile %||% "."), "GLMM.r")
if (!file.exists(glmm_script)) {
  # Fallback: look relative to working directory
  glmm_script <- "GLMM.r"
}
if (!file.exists(glmm_script)) {
  stop("GLMM.r not found. Run this script from the same directory as GLMM.r, ",
       "or set the glmm_script path explicitly above.")
}

# ---------------------------------------------------------------------------
# Helper: resolve || for NULL (base R doesn't have %||%)
# ---------------------------------------------------------------------------
`%||%` <- function(a, b) if (!is.null(a)) a else b

# ---------------------------------------------------------------------------
# Read GLMM.r once; we'll patch it per iteration
# ---------------------------------------------------------------------------
glmm_lines <- readLines(glmm_script)

dir.create(kappa_tests_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Sweep log
# ---------------------------------------------------------------------------
sweep_log <- list()
n_total   <- length(kappa_values)

cat("=============================================================\n")
cat("  Kappa sweep: GLMM.r x", n_total, "runs\n")
cat("  Kappa values:", paste(kappa_values, collapse = ", "), "\n")
cat("  Output base: ", kappa_tests_dir, "\n")
cat("=============================================================\n\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for (i in seq_along(kappa_values)) {

  kappa_val <- kappa_values[i]
  cat("-------------------------------------------------------------\n")
  cat(sprintf("[%d/%d]  kappa = %g\n", i, n_total, kappa_val))
  cat("-------------------------------------------------------------\n")

  # -- Patch 1: replace the kappa setting in cfg --------------------------------
  # Targets the line:  kappa = <number>,  # multiplier ...
  patched <- gsub(
    pattern     = "(\\bkappa\\s*=\\s*)[0-9]+(?:\\.[0-9]+)?",
    replacement = paste0("\\1", kappa_val),
    x           = glmm_lines,
    perl        = TRUE
  )

  # -- Patch 2: wrap output_dir paths with file.path(..., kappa_sweep_subdir) --
  # Targets the quoted paths inside cfg$output_dir (both if/else branches).
  # Result in patched script: file.path("<original_path>", "kappa_tests")
  patched <- gsub(
    pattern     = '("[^"]*fitting[/\\\\]GLMM[^"]*")',
    replacement = paste0('file.path(\\1, "', kappa_sweep_subdir, '")'),
    x           = patched,
    perl        = TRUE
  )

  # -- Write patched script to a temp file --------------------------------------
  tmp_file <- tempfile(pattern = sprintf("GLMM_k%g_", kappa_val), fileext = ".r")
  writeLines(patched, tmp_file)
  on.exit(unlink(tmp_file), add = TRUE)

  # -- Run the patched script ---------------------------------------------------
  t_start <- proc.time()
  run_ok  <- tryCatch({
    source(tmp_file, local = FALSE)
    TRUE
  }, error = function(e) {
    cat(sprintf("\n[ERROR] kappa = %g failed:\n  %s\n\n", kappa_val, conditionMessage(e)))
    FALSE
  })
  elapsed <- (proc.time() - t_start)[["elapsed"]]

  unlink(tmp_file)
  on.exit(NULL)   # clear the on.exit registered above

  sweep_log[[i]] <- list(
    kappa   = kappa_val,
    success = run_ok,
    elapsed = elapsed
  )

  cat(sprintf("\nkappa = %g  |  status: %s  |  time: %.1f s\n\n",
              kappa_val,
              if (run_ok) "OK" else "FAILED",
              elapsed))
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
cat("=============================================================\n")
cat("  Kappa sweep complete\n")
cat("=============================================================\n")
log_df <- do.call(rbind, lapply(sweep_log, as.data.frame))
print(log_df)

log_file <- file.path(kappa_tests_dir, paste0("sweep_log_", format(Sys.Date(), "%Y%m%d"), ".csv"))
write.csv(log_df, log_file, row.names = FALSE)
cat("\nSweep log saved to:", log_file, "\n")
