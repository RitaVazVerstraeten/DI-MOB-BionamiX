# model comparison through Leave-One-Out (LOO)

library(dplyr)

# ── List the specific LOO files to compare ───────────────────────────
results_root <- if (Sys.info()["nodename"] == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan"
}

pred <- "lag-total_rainy_days-avg_VPD-precip_max_day_unlag-is_urban-is_WUI"

loo_files <- list(
    # AR + Block RE
   "AR1perCMF + blockRE"  = file.path(results_root, pred, "CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks",  "20260512_free_lag_str", "loo_CMF_AR1perCMF_noGP_blockRE_lag2_k2_AllBlocks.txt"),
    # AR no Block RE
  "AR1perCMF, no blockRE" = file.path(results_root, pred, "CMF_AR1perCMF_noGP_noBlockRE_lag2_k2_AllBlocks", "20260512_free_lag_str", "loo_CMF_AR1perCMF_noGP_noBlockRE_lag2_k2_AllBlocks.txt"), 
    # only Block RE
   "blockRE only"  = file.path(results_root, pred, "CMF_noAR1_noGP_blockRE_lag2_k2_AllBlocks", "20260513_free_lag_str", "loo_CMF_noAR1_noGP_blockRE_lag2_k2_AllBlocks.txt")
  # add more entries here as needed
)


# ── Parse and compare ──────────────────────────────────────────────────────────
parse_loo_txt <- function(path, label) {
  lines <- readLines(path)
  extract <- function(lab) {
    l <- grep(paste0("^", lab), lines, value = TRUE)
    as.numeric(strsplit(trimws(l), "\\s+")[[1]][2:3])
  }
  data.frame(
    model    = label,
    elpd_loo = extract("elpd_loo")[1],
    se_elpd  = extract("elpd_loo")[2],
    p_loo    = extract("p_loo")[1],
    looic    = extract("looic")[1],
    stringsAsFactors = FALSE
  )
}

loo_df <- do.call(rbind, mapply(parse_loo_txt, loo_files, names(loo_files), SIMPLIFY = FALSE))
loo_df <- loo_df[order(-loo_df$elpd_loo), ]
loo_df$elpd_diff      <- loo_df$elpd_loo - max(loo_df$elpd_loo)
loo_df$se_diff_approx <- sqrt(loo_df$se_elpd^2 + loo_df$se_elpd[loo_df$elpd_diff == 0]^2)
loo_df$z_score <- loo_df$elpd_diff / loo_df$se_diff_approx
loo_df$z_score[loo_df$elpd_diff == 0] <- 0

print(loo_df[, c("model", "elpd_loo", "se_elpd", "elpd_diff", "se_diff_approx", "z_score", "p_loo", "looic")],
      row.names = FALSE)