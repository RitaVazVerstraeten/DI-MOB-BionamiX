# cleanup_incorrect_season_interaction_runs.R
#
# run_season_interaction_sweep.R's first (pass 1, Aug 11-12) pass through the
# 16-config grid had a bug: the "total" and "both" arms' dlnm_ix_vars never
# actually varied -- all three of resid/total/both silently fit the same
# single is_rainy_season x precip_max_day_resid_on_tp interaction, just under
# three different run_suffix labels. Verified directly against each run's
# saved model_summary_*.txt (w_ix parameter count: 9 for a single ns3df
# interaction, 12 for ns4df -- never 18, which a genuine "both" fit would
# have) and its interaction plot filenames (all say precip_resid_x_season,
# regardless of what the run_suffix/folder claims).
#
# This script MOVES (never deletes) those 8 confirmed-mislabeled run
# directories out of the way, into a dated quarantine folder alongside the
# sweep root, so the sweep's own output tree only contains genuinely correct
# runs. Nothing is destroyed -- if this list is ever wrong, everything is
# still sitting in the quarantine folder to move back.
#
# Defaults to a dry run (DRY_RUN <- TRUE below): prints exactly what it would
# move without touching anything. Flip to FALSE to actually perform the move.

DRY_RUN <- TRUE

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
resid_predictor_dir <- file.path(
  sweep_root,
  "dlnm-total_precip-avg_VPD-precip_max_day_resid_on_tp_unlag-HFP_urbanization-mean_ndvi-is_WUI-water_shortage-water_containers-is_rainy_season_ix-precip_resid_x_season"
)

# All 8 are pass-1 runs mis-filed under the precip_resid_x_season folder
# despite being tagged "total" or "both" in their run_suffix -- verified
# individually against saved model output (see header comment).
incorrect_runs <- c(
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag5_k4_AllBlocks", "20260811_arglagns3df_lag5_ixtotal"),
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag5_k4_AllBlocks", "20260812_arglagns3df_lag5_ixboth"),
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag6_k4_AllBlocks", "20260812_arglagns3df_lag6_ixtotal"),
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag6_k4_AllBlocks", "20260812_arglagns3df_lag6_ixboth"),
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag5_k4_AllBlocks", "20260812_arglagns4df_lag5_ixtotal"),
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag5_k4_AllBlocks", "20260812_arglagns4df_lag5_ixboth"),
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag6_k4_AllBlocks", "20260812_arglagns4df_lag6_ixtotal"),
  file.path(resid_predictor_dir, "CMF_DLNM_AR1perCMF_noGP_blockRE_lag6_k4_AllBlocks", "20260812_arglagns4df_lag6_ixboth")
)

quarantine_dir <- file.path(sweep_root, paste0("_incorrect_pass1_mislabeled_", format(Sys.Date(), "%Y%m%d")))

cat(strrep("=", 70), "\n")
cat(if (DRY_RUN) "DRY RUN -- nothing will be moved\n" else "LIVE RUN -- moving directories now\n")
cat(strrep("=", 70), "\n\n")

if (!DRY_RUN) dir.create(quarantine_dir, recursive = TRUE, showWarnings = FALSE)

for (run_dir in incorrect_runs) {
  if (!dir.exists(run_dir)) {
    cat("SKIP (already gone):", run_dir, "\n")
    next
  }
  # Preserve the lag5/lag6 split in the quarantine folder so moved runs
  # don't collide (both lags reuse the same run_suffix strings).
  lag_tag  <- basename(dirname(run_dir))
  dest_dir <- file.path(quarantine_dir, lag_tag)
  dest     <- file.path(dest_dir, basename(run_dir))

  cat(sprintf("MOVE:\n  from: %s\n  to:   %s\n", run_dir, dest))

  if (!DRY_RUN) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
    ok <- file.rename(run_dir, dest)
    if (!ok) cat("  WARNING: move failed (cross-device?) -- leaving in place.\n")
  }
}

cat("\n", strrep("=", 70), "\n")
if (DRY_RUN) {
  cat("Dry run complete. Set DRY_RUN <- FALSE at the top of this script and re-run to actually move these.\n")
} else {
  cat("Done. Quarantined runs are at:", quarantine_dir, "\n")
}
