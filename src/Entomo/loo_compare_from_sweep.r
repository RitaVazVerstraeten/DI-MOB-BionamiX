library(loo)
library(writexl)

# ── Point this at a variable_sweep_* folder ───────────────────────────────────
sweep_dir <- if (Sys.info()["nodename"] == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/variable_sweep_CMF_AR1_noGP_blockRE_lag2_k2_AllBlocks_20260513"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/variable_sweep_CMF_AR1_noGP_blockRE_lag2_k2_AllBlocks_20260513"
}

# ── Load all LOO RDS files found anywhere under sweep_dir ─────────────────────
rds_files <- list.files(sweep_dir, pattern = "^loo_.*\\.rds$",
                        full.names = TRUE, recursive = TRUE)

if (length(rds_files) < 2) {
  stop("Fewer than 2 LOO RDS files found in: ", sweep_dir)
}

loo_list <- lapply(rds_files, readRDS)
names(loo_list) <- tools::file_path_sans_ext(basename(rds_files))
cat("Loaded", length(loo_list), "LOO results:\n")
cat(paste0("  ", names(loo_list), "\n"), sep = "")

# ── Compare ───────────────────────────────────────────────────────────────────
loo_cmp <- loo::loo_compare(loo_list)
print(loo_cmp, digits = 2, simplify = FALSE)

cmp_df <- as.data.frame(loo_cmp)
cmp_df$z_score <- cmp_df$elpd_diff / cmp_df$se_diff
cmp_df$z_score[cmp_df$elpd_diff == 0] <- 0
cat("\nz-score (elpd_diff / se_diff):\n")
print(cmp_df["z_score"], digits = 2)

# ── Save ──────────────────────────────────────────────────────────────────────
cmp_out <- cbind(model = rownames(cmp_df), cmp_df)
rownames(cmp_out) <- NULL

writeLines(capture.output(print(loo_cmp, digits = 2, simplify = FALSE)),
           file.path(sweep_dir, "loo_comparison.txt"))
writexl::write_xlsx(cmp_out, file.path(sweep_dir, "loo_comparison.xlsx"))
cat("Saved to:", sweep_dir, "\n")
