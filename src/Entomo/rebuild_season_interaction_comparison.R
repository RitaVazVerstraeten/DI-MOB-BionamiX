# rebuild_season_interaction_comparison.R
#
# run_season_interaction_sweep.R's own end-of-loop LOO/WAIC/bootstrap
# comparison can't be trusted here: its pass-1 run mislabeled 8 of its 16
# configs (see cleanup_incorrect_season_interaction_runs.R for details), and
# the sweep has since been interrupted/resumed at least once (evidenced by
# run_suffix dates changing mid-run), so its in-memory loo_list/waic_list
# state is unverifiable from outside.
#
# This script sidesteps all of that: each config's Stan fit already saves its
# own loo_*.rds/waic_*.rds directly to its run_output_dir, independent of the
# sweep's final aggregation step. So this rebuilds the comparison straight
# from disk, resolving each of the 16 canonical (arglag_df, max_lag,
# interaction arm) configs to the correct run directory, and skipping any
# that haven't finished yet -- safe to run now for a partial comparison, or
# again later once the sweep completes for the full one. Doesn't touch the
# live sweep process at all.
#
# Bootstrap comparison is per-observation (iid), not block-clustered: the
# original sweep's block-clustered version requires stan_data$block captured
# live during a fit, which isn't available here without re-running data prep.
# If you want the clustered version, it needs stan_data$block reconstructed
# via build_dlnm_stan_data() on this sweep's fixed cfg -- ask and I'll add it.

suppressMessages(library(loo))

script_dir <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo"
source(file.path(script_dir, "helper_functions.r"))  # for bootstrap_elpd_comparison()

# Same hostname-based path resolution as run_season_interaction_sweep.R --
# this must match wherever that script's sweep_d actually pointed for the
# machine this is run on.
hostname   <- Sys.info()["nodename"]
sweep_root <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/season_interaction_sweep"
} else if (hostname == "stoofvlees") {
  path.expand("~/data/entomo/results/fitting/stan/season_interaction_sweep")
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/season_interaction_sweep"
}

pred_dir <- function(arm) file.path(sweep_root, switch(arm,
  none  = "dlnm-total_precip-avg_VPD-precip_max_day_resid_on_tp_unlag-HFP_urbanization-mean_ndvi-is_WUI-water_shortage-water_containers",
  resid = "dlnm-total_precip-avg_VPD-precip_max_day_resid_on_tp_unlag-HFP_urbanization-mean_ndvi-is_WUI-water_shortage-water_containers-is_rainy_season_ix-precip_resid_x_season",
  total = "dlnm-total_precip-avg_VPD-precip_max_day_resid_on_tp_unlag-HFP_urbanization-mean_ndvi-is_WUI-water_shortage-water_containers-is_rainy_season_ix-tp_x_season",
  both  = "dlnm-total_precip-avg_VPD-precip_max_day_resid_on_tp_unlag-HFP_urbanization-mean_ndvi-is_WUI-water_shortage-water_containers-is_rainy_season_ix-precip_resid_x_season-tp_x_season"
))

# Each arm's correct predictor_spec folder is disjoint from the mislabeled
# pass-1 runs (those all landed under the "resid" folder regardless of arm),
# so no date-based filtering is needed here -- just find any completed run
# matching this (df, lag, arm)'s run_suffix pattern in the right folder, and
# take the most recently completed one if more than one exists (e.g. a
# config that both pass 1 and pass 2 correctly fit -- they're the same model,
# either is fine).
resolve_run <- function(arglag_df, max_lag, arm) {
  model_dir <- file.path(pred_dir(arm),
                          sprintf("CMF_DLNM_AR1perCMF_noGP_blockRE_lag%d_k4_AllBlocks", max_lag))
  if (!dir.exists(model_dir)) return(NULL)

  pattern   <- sprintf("_arglagns%ddf_lag%d_ix%s$", arglag_df, max_lag, arm)
  run_dirs  <- list.dirs(model_dir, recursive = FALSE)
  run_dirs  <- run_dirs[grepl(pattern, basename(run_dirs))]
  if (length(run_dirs) == 0) return(NULL)

  loo_files <- file.path(run_dirs, sprintf("loo_CMF_DLNM_AR1perCMF_noGP_blockRE_lag%d_k4_AllBlocks.rds", max_lag))
  present   <- file.exists(loo_files)
  if (!any(present)) return(NULL)

  run_dirs <- run_dirs[present]
  loo_files <- loo_files[present]
  # Most recently completed, if duplicates exist.
  run_dirs[which.max(file.mtime(loo_files))]
}

arglag_dfs <- c(3, 4)
max_lags   <- c(5, 6)
arms       <- c("none", "resid", "total", "both")

configs <- expand.grid(arglag_df = arglag_dfs, max_lag = max_lags, arm = arms,
                        stringsAsFactors = FALSE)
configs$label <- sprintf("ns%ddf_lag%d_%s", configs$arglag_df, configs$max_lag, configs$arm)

cat(strrep("=", 70), "\n")
cat("Resolving 16 canonical configs against disk\n")
cat(strrep("=", 70), "\n\n")

loo_list  <- list()
waic_list <- list()

for (i in seq_len(nrow(configs))) {
  cfg_i    <- configs[i, ]
  run_dir  <- resolve_run(cfg_i$arglag_df, cfg_i$max_lag, cfg_i$arm)

  if (is.null(run_dir)) {
    cat(sprintf("  PENDING  %s -- not completed yet\n", cfg_i$label))
    next
  }

  loo_file  <- list.files(run_dir, pattern = "^loo_.*\\.rds$",  full.names = TRUE)[1]
  waic_file <- list.files(run_dir, pattern = "^waic_.*\\.rds$", full.names = TRUE)[1]

  if (!is.na(loo_file))  loo_list[[cfg_i$label]]  <- readRDS(loo_file)
  if (!is.na(waic_file)) waic_list[[cfg_i$label]] <- readRDS(waic_file)

  cat(sprintf("  OK       %s -- %s\n", cfg_i$label, basename(run_dir)))
}

cat(sprintf("\n%d of %d configs available (%d pending).\n",
            length(loo_list), nrow(configs), nrow(configs) - length(loo_list)))

# =============================================================================
# Output
# =============================================================================
out_dir <- file.path(sweep_root, paste0("comparison_corrected_", format(Sys.Date(), "%Y%m%d")))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_comparison <- function(result_list, criterion_label, file_stub) {
  if (length(result_list) < 2) {
    cat("Fewer than 2 available", criterion_label, "results -- skipping comparison.\n")
    return(invisible(NULL))
  }

  cat("\n", strrep("=", 70), "\n")
  cat(criterion_label, "COMPARISON (", length(result_list), "of 16 configs )\n")
  cat(strrep("=", 70), "\n\n")

  comp <- loo_compare(result_list)
  print(comp, simplify = FALSE, digits = 2)

  cmp_df <- as.data.frame(comp)
  cmp_df$z_score <- cmp_df$elpd_diff / cmp_df$se_diff
  cmp_df$z_score[cmp_df$elpd_diff == 0] <- 0
  cat("\nz-score (elpd_diff / se_diff):\n")
  print(cmp_df["z_score"], digits = 2)

  comp_file <- file.path(out_dir, paste0(file_stub, "_comparison.txt"))
  comp_output <- capture.output({
    cat(criterion_label, "comparison, rebuilt from disk --", format(Sys.Date()), "\n")
    cat("Configs included (", length(result_list), "of 16 ):\n")
    for (nm in names(result_list)) cat("  -", nm, "\n")
    cat("\n")
    print(comp, simplify = FALSE, digits = 2)
    cat("\nz-score (elpd_diff / se_diff):\n")
    print(cmp_df["z_score"], digits = 2)
  })
  writeLines(comp_output, comp_file)
  cat("\nSaved to:", comp_file, "\n")

  saveRDS(result_list, file.path(out_dir, paste0(file_stub, "_list.rds")))

  cat("\n", criterion_label, "BOOTSTRAP COMPARISON (per-observation, iid)\n")
  boot_cmp <- bootstrap_elpd_comparison(result_list, cluster_ids = NULL, n_boot = 4000)
  print(boot_cmp, digits = 3, row.names = FALSE)
  boot_file <- file.path(out_dir, paste0(file_stub, "_bootstrap_comparison.csv"))
  write.csv(boot_cmp, boot_file, row.names = FALSE)
  cat("Saved to:", boot_file, "\n")
}

write_comparison(loo_list,  "LOO",  "loo")
write_comparison(waic_list, "WAIC", "waic")
