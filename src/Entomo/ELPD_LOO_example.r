# model comparison through Leave-One-Out (LOO)

library(loo)

# ── List the specific LOO RDS files to compare ────────────────────────
results_root <- if (Sys.info()["nodename"] == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan"
}

pred <- "lag-total_rainy_days-avg_VPD-precip_max_day_unlag-is_urban-is_WUI"

# loo_files <- list(
#     # AR + Block RE
#    "AR1perCMF + blockRE"  = file.path(results_root, pred, "CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks",  "20260513_free_lag_str", "loo_CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks.rds"),
#     # AR no Block RE
#   "AR1perCMF, no blockRE" = file.path(results_root, pred, "CMF_AR1perCMF_noGP_noBlockRE_lag2_k2_AllBlocks", "20260513_free_lag_str", "loo_CMF_AR1perCMF_noGP_noBlockRE_lag2_k2_AllBlocks.rds"),
#     # only Block RE
#    "blockRE only"          = file.path(results_root, pred, "CMF_noAR1_noGP_blockRE_lag2_k2_AllBlocks",       "20260513_free_lag_str", "loo_CMF_noAR1_noGP_blockRE_lag2_k2_AllBlocks.rds")
#   # add more entries here as needed
# )


loo_files <- list(
    
   "free_lags"  = file.path(results_root, pred, "CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks",  "20260513_free_lag_str", "loo_CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks.rds"),

  "monotone_decay" = file.path(results_root, pred, "CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks", "20260513_monotone_decay", "loo_CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks.rds")
)

# ── Load and compare ───────────────────────────────────────────────────
loo_list <- lapply(loo_files, readRDS)

loo_cmp <- loo::loo_compare(loo_list)
print(loo_cmp, digits = 2, simplify = FALSE)

cmp_df <- as.data.frame(loo_cmp)
cmp_df$z_score <- cmp_df$elpd_diff / cmp_df$se_diff
cmp_df$z_score[cmp_df$elpd_diff == 0] <- 0
cat("\nz-score (elpd_diff / se_diff):\n")
print(cmp_df["z_score"], digits = 2)