
source("plot_functions.r")

# Post-fit analysis for a glmmTMB beta-binomial GLMM.
# Sourced by GLMM.r and GLMM_stepwise_selection.R after model fitting.
#
# Required objects in the calling environment:
#   model          - fitted glmmTMB object
#   df_model       - complete-case data used to fit the model (with predictions attached after section 8)
#   df             - pre-model data (used to build df_observed)
#   cfg            - configuration list
#   formula_str    - character formula string
#   run_suffix     - date string used in filenames
#   run_output_dir - root output folder for this run
#   plots_output_dir  - subfolder for plots
#   resid_output_dir  - subfolder for residual diagnostics

# =============================================================================
# 7. INSPECT RESULTS
# =============================================================================
cat("Model Summary:\n")
print(summary(model))

coef_mat <- summary(model)$coefficients$cond
coef_table <- as_tibble(coef_mat, rownames = "term") %>%
  transmute(
    term,
    estimate        = Estimate,
    std_error       = `Std. Error`,
    z_value         = `z value`,
    p_value         = `Pr(>|z|)`,
    OR              = exp(estimate),
    pct_change_odds = (OR - 1) * 100
  ) %>%
  arrange(desc(abs(estimate)))

cat("\nFixed-effects interpretation (log-odds and odds ratios):\n")
print(coef_table, n = nrow(coef_table))

coef_table_file <- file.path(run_output_dir, paste0("glmm_fixed_effects_OR_", run_suffix, ".csv"))
write_csv(coef_table, coef_table_file)

save_glmm_coef_forest_plot(coef_table, cfg, plots_output_dir, run_suffix,
                           scale = c("logodds", "OR"))

summary_file <- file.path(run_output_dir, paste0("glmm_summary_", run_suffix, ".txt"))
sink(summary_file)
tryCatch({
  cat("Run suffix:", run_suffix, "\n")
  cat("Formula:", formula_str, "\n\n")
  op <- options(max.print = 999999)
  print(summary(model))
  options(op)
}, finally = sink())

# =============================================================================
# 7b. COLLINEARITY DIAGNOSTICS — VIF
# =============================================================================
# VIF = 1 / (1 - R²_j): how much the variance of coefficient j is inflated by
# correlation with the other predictors.
#   VIF < 5   : acceptable
#   VIF 5–10  : moderate concern, monitor
#   VIF > 10  : severe — consider dropping or combining lags

cat("\n=== COLLINEARITY DIAGNOSTICS (VIF) ===\n")

# --- Primary: performance::check_collinearity() supports glmmTMB natively ---
if (requireNamespace("performance", quietly = TRUE)) {
  vif_perf <- tryCatch(
    performance::check_collinearity(model, component = "conditional"),
    error = function(e) {
      cat("  performance::check_collinearity() failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (!is.null(vif_perf)) {
    cat("\nVIF from performance::check_collinearity():\n")
    print(vif_perf)

    vif_file <- file.path(run_output_dir, paste0("vif_collinearity_", run_suffix, ".csv"))
    write_csv(as.data.frame(vif_perf), vif_file)
    cat("  Saved to:", vif_file, "\n")
  }
} else {
  cat("  Install the 'performance' package for check_collinearity() support.\n")
  vif_perf <- NULL
}

# --- Fallback / complement: manual VIF from the fixed-effects model matrix ---
# Avoids any glmmTMB-specific issues and is fast via correlation-matrix inversion.
tryCatch({
  mm <- model.matrix(model)                          # fixed-effects design matrix
  mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
  mm_complete <- mm[complete.cases(mm), , drop = FALSE]

  if (ncol(mm_complete) > 1) {
    C    <- cor(mm_complete)
    vifs <- diag(solve(C))
    vif_df <- tibble(
      term      = names(vifs),
      VIF       = vifs,
      flag      = case_when(VIF > 10 ~ "HIGH (>10)",
                            VIF >  5 ~ "moderate (5-10)",
                            TRUE     ~ "ok")
    ) %>% arrange(desc(VIF))

    cat("\nManual VIF from model matrix (correlation-matrix inversion):\n")
    print(vif_df, n = nrow(vif_df))

    # --- Per-variable summary: max VIF across its lags ---
    lag_pattern <- paste(cfg$lag_vars, collapse = "|")
    vif_lag_summary <- vif_df %>%
      filter(grepl(lag_pattern, term)) %>%
      mutate(base_var = sub("_lag[0-9]+$", "", term)) %>%
      group_by(base_var) %>%
      summarise(max_VIF = max(VIF), mean_VIF = mean(VIF),
                worst_lag = term[which.max(VIF)], .groups = "drop") %>%
      arrange(desc(max_VIF))

    cat("\nMax VIF per lagged variable:\n")
    print(vif_lag_summary)

    vif_manual_file <- file.path(run_output_dir, paste0("vif_manual_", run_suffix, ".csv"))
    write_csv(vif_df, vif_manual_file)
    cat("  Saved to:", vif_manual_file, "\n")
  } else {
    cat("  Only one predictor — VIF not applicable.\n")
  }
}, error = function(e) {
  cat("  Manual VIF computation failed:", conditionMessage(e), "\n")
})

cat("=== End VIF diagnostics ===\n\n")

# =============================================================================
# 8. ADD PREDICTIONS (fitted p_bt)
# =============================================================================
preds_with_se <- predict(model, type = "response", se.fit = TRUE)
if (length(preds_with_se$fit) != nrow(df_model))
  stop("Mismatch: ", nrow(df_model), " model rows vs ", length(preds_with_se$fit), " predictions")

df_model <- df_model %>%
  mutate(
    p_bt_fitted       = preds_with_se$fit,
    p_bt_fitted_se    = preds_with_se$se.fit,
    p_bt_fitted_lower = p_bt_fitted - 1.96 * p_bt_fitted_se,
    p_bt_fitted_upper = p_bt_fitted + 1.96 * p_bt_fitted_se
  )

# =============================================================================
# 9. BUILD SUMMARY TABLE (one row per block × month)
# =============================================================================
# No baseline/reactive split: predictions are for the full n_bt directly.
# p_R_fitted = NA and omega = 0 are set as stubs so downstream plot functions
# that reference these columns continue to work without modification.
df_summary <- df_model %>%
  select(block, year_month_date,
         p_bt_fitted, p_bt_fitted_se, p_bt_fitted_lower, p_bt_fitted_upper) %>%
  mutate(
    p_R_fitted       = NA_real_,
    p_R_fitted_se    = NA_real_,
    p_R_fitted_lower = NA_real_,
    p_R_fitted_upper = NA_real_,
    omega            = 0
  )

model_pred_file   <- file.path(run_output_dir, paste0("glmm_model_predictions_",   run_suffix, ".csv"))
summary_pred_file <- file.path(run_output_dir, paste0("glmm_summary_predictions_",  run_suffix, ".csv"))
write_csv(df_model,   model_pred_file)
write_csv(df_summary, summary_pred_file)

cat("\nSaved outputs:\n")
cat("  Run folder:            ", run_output_dir,   "\n", sep = "")
cat("  Plots folder:          ", plots_output_dir, "\n", sep = "")
cat("  Residuals folder:      ", resid_output_dir, "\n", sep = "")
cat("  Summary TXT:           ", summary_file,     "\n", sep = "")
cat("  Fixed effects OR CSV:  ", coef_table_file,  "\n", sep = "")
cat("  Model predictions CSV: ", model_pred_file,  "\n", sep = "")
cat("  Summary predictions CSV:", summary_pred_file, "\n", sep = "")

# =============================================================================
# 10. PLOTS
# =============================================================================
df_observed <- df %>%
  transmute(block, year_month_date,
            p_observed = ifelse(n_bt > 0, y_bt / n_bt, NA_real_),
            cases)

df_summary_weighted <- df_summary %>%
  left_join(df_observed, by = c("block", "year_month_date")) %>%
  mutate(p_fitted_weighted = p_bt_fitted)

save_glmm_prob_timeseries_plot(df_summary  = df_summary,
                               df_observed = df_observed,
                               output_dir  = plots_output_dir,
                               run_suffix  = run_suffix,
                               cfg         = cfg)

save_glmm_calibplot_observed_vs_expected(df_summary  = df_summary,
                                         df_observed = df_observed,
                                         output_dir  = plots_output_dir,
                                         run_suffix  = run_suffix)

save_glmm_calibplot_weighted_avg(df         = df_summary_weighted,
                                 output_dir = plots_output_dir,
                                 run_suffix = run_suffix)

# =============================================================================
# 11. PLOT MODEL RESIDUALS
# =============================================================================
save_glmm_residuals_plot(model, resid_output_dir, run_suffix)

# =============================================================================
# 12. PLOT RANDOM EFFECTS
# =============================================================================
re <- ranef(model)$cond

u_post <- if (!is.null(re$block)) re$block[[1]] else rep(NA_real_, 1)
v_post <- if (!is.null(re$year_month)) {
  re$year_month[[1]]
} else if (!is.null(re$ar1_group)) {
  colMeans(as.matrix(re$ar1_group))
} else {
  rep(NA_real_, 1)
}

save_random_effects(u_post, v_post, plots_output_dir, run_suffix)

# =============================================================================
# 13. LARGE-RESIDUAL DIAGNOSTICS
# =============================================================================
pearson_resid <- residuals(model, type = "pearson")

df_resid_diag <- df_model %>%
  mutate(fitted_prob   = fitted(model),
         pearson_resid = pearson_resid,
         p_observed    = ifelse(n_trials > 0, y_bt / n_trials, NA_real_),
         abs_resid     = abs(pearson_resid)) %>%
  filter(abs_resid > 2) %>%
  select(block, year_month_date, y_bt, n_trials, n_bt,
         p_observed, fitted_prob, pearson_resid, abs_resid) %>%
  arrange(desc(abs_resid))

resid_diag_file <- file.path(resid_output_dir, paste0("glmm_large_residuals_", run_suffix, ".csv"))
write_csv(df_resid_diag, resid_diag_file)
cat("\nLarge-residual rows (|Pearson resid| > 2):", nrow(df_resid_diag),
    "out of", nrow(df_model), "model rows\n")

total_timepoints  <- df_model %>% count(block, name = "n_months")
block_resid_summary <- df_resid_diag %>%
  group_by(block) %>%
  summarise(n_large_resid     = n(),
            mean_pearson      = mean(pearson_resid),
            max_pearson       = max(abs_resid),
            mean_p_observed   = mean(p_observed, na.rm = TRUE),
            mean_fitted       = mean(fitted_prob, na.rm = TRUE),
            pct_positive_bias = mean(pearson_resid > 0)) %>%
  left_join(total_timepoints, by = "block") %>%
  mutate(pct_months_flagged = n_large_resid / n_months) %>%
  arrange(desc(pct_months_flagged))

cat("\nBlocks with large residuals in >30% of months:\n")
print(filter(block_resid_summary, pct_months_flagged > 0.30))

temporal_resid_summary <- df_resid_diag %>%
  group_by(year_month_date) %>%
  summarise(n_large_resid = n(), mean_pearson = mean(pearson_resid),
            pct_positive  = mean(pearson_resid > 0)) %>%
  arrange(desc(n_large_resid))
cat("\nMonths with most large residuals:\n")
print(head(temporal_resid_summary, 10))

block_resid_file <- file.path(resid_output_dir, paste0("glmm_block_resid_summary_", run_suffix, ".csv"))
write_csv(block_resid_summary, block_resid_file)
cat("Block residual summary saved to:", block_resid_file, "\n")

# =============================================================================
# 14. AR(1) TEMPORAL VARIANCE DIAGNOSTICS
# =============================================================================
df_re_plot <- df_model %>%
  mutate(fitted_prob   = fitted(model),
         pearson_resid = pearson_resid)

monthly_resid <- df_re_plot %>%
  group_by(year_month_date) %>%
  summarise(mean_resid    = mean(pearson_resid, na.rm = TRUE),
            median_resid  = median(pearson_resid, na.rm = TRUE),
            mean_fitted   = mean(fitted_prob, na.rm = TRUE),
            mean_observed = mean(ifelse(n_trials > 0, y_bt / n_trials, NA_real_), na.rm = TRUE),
            n = n(), .groups = "drop")

p_resid_time <- ggplot(monthly_resid, aes(x = year_month_date)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  geom_line(aes(y = mean_resid), colour = "steelblue") +
  geom_point(aes(y = mean_resid, size = n), colour = "steelblue", show.legend = FALSE) +
  labs(title    = "Mean Pearson residual over time (all rows)",
       subtitle = "Systematic pattern = unmeasured temporal signal absorbed by AR(1)",
       x = "Month", y = "Mean Pearson residual") +
  theme_minimal()
ggsave(file.path(resid_output_dir, paste0("ar1_resid_over_time_", run_suffix, ".png")),
       p_resid_time, width = 10, height = 5, dpi = 150)

block_monthly <- df_re_plot %>%
  group_by(block, year_month_date) %>%
  summarise(mean_fitted   = mean(fitted_prob, na.rm = TRUE),
            mean_observed = mean(ifelse(n_trials > 0, y_bt / n_trials, NA_real_), na.rm = TRUE),
            .groups = "drop")

set.seed(42)
sample_blocks <- sample(unique(block_monthly$block), min(50, n_distinct(block_monthly$block)))
p_block_traj <- ggplot(filter(block_monthly, block %in% sample_blocks),
                       aes(x = year_month_date, y = mean_fitted, group = block)) +
  geom_line(alpha = 0.2, colour = "steelblue") +
  geom_line(data = monthly_resid, aes(x = year_month_date, y = mean_fitted, group = NULL),
            colour = "red", linewidth = 1.2) +
  labs(title    = "Fitted trajectories for 50 random blocks",
       subtitle = "Red = city-wide mean fitted. Spread = block heterogeneity in AR(1)",
       x = "Month", y = "Fitted positivity rate") +
  theme_minimal()
ggsave(file.path(resid_output_dir, paste0("ar1_block_trajectories_", run_suffix, ".png")),
       p_block_traj, width = 10, height = 6, dpi = 150)

resid_ts <- monthly_resid %>% arrange(year_month_date) %>% pull(mean_resid)
acf_vals <- acf(resid_ts, lag.max = 24, plot = FALSE, na.action = na.pass)
acf_df   <- data.frame(lag = as.numeric(acf_vals$lag), acf = as.numeric(acf_vals$acf))
ci_bound <- qnorm(0.975) / sqrt(length(resid_ts))

p_acf <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_hline(yintercept = 0) +
  geom_hline(yintercept = c(-ci_bound, ci_bound), linetype = "dashed", colour = "blue") +
  geom_segment(aes(x = lag, xend = lag, y = 0, yend = acf)) +
  geom_point(size = 2) +
  labs(title    = "ACF of city-wide mean Pearson residuals",
       subtitle = "Significant spikes = AR(1) did not fully remove temporal autocorrelation",
       x = "Lag (months)", y = "Autocorrelation") +
  theme_minimal()
ggsave(file.path(resid_output_dir, paste0("ar1_acf_mean_resid_", run_suffix, ".png")),
       p_acf, width = 8, height = 5, dpi = 150)

cat("\nAR(1) temporal diagnostics saved to:", resid_output_dir, "\n")

# =============================================================================
# 15. SPATIAL AUTOCORRELATION OF RESIDUALS (MORAN'S I)
# =============================================================================
if (!requireNamespace("spdep", quietly = TRUE)) {
  cat("Skipping Moran's I: package 'spdep' not installed.\n")
} else {

  if (!exists("sf_blocks") || !cfg$sf_block_col %in% names(sf_blocks)) {
    sf_blocks <- sf::st_read(cfg$shapefile_file, quiet = TRUE)
    if (cfg$spatial_level == "CMF")
      sf_blocks <- sf_blocks %>% mutate(Area_CMF = paste(AS, CMF, sep = "_"))
  }

  block_mean_resid <- df_re_plot %>%
    group_by(block) %>%
    summarise(mean_resid = mean(pearson_resid, na.rm = TRUE), .groups = "drop") %>%
    mutate(block_chr = as.character(block))

  pts_resid <- suppressWarnings(sf::st_point_on_surface(sf_blocks))
  if (isTRUE(sf::st_is_longlat(pts_resid)))
    pts_resid <- sf::st_transform(pts_resid, 3857)
  else if (!is.na(cfg$spatial_crs))
    pts_resid <- sf::st_transform(pts_resid, cfg$spatial_crs)

  coords_resid <- sf_blocks %>%
    sf::st_drop_geometry() %>%
    mutate(block_chr = as.character(.data[[cfg$sf_block_col]]),
           x = sf::st_coordinates(pts_resid)[, 1],
           y = sf::st_coordinates(pts_resid)[, 2]) %>%
    select(block_chr, x, y) %>%
    distinct(block_chr, .keep_all = TRUE)

  resid_spatial <- block_mean_resid %>%
    left_join(coords_resid, by = "block_chr") %>%
    filter(!is.na(x), !is.na(y), !is.na(mean_resid))

  cat(sprintf("\nMoran's I: %d blocks with residuals and coordinates\n", nrow(resid_spatial)))

  coords_mat     <- as.matrix(resid_spatial[, c("x", "y")])
  dist_mat       <- as.matrix(dist(coords_mat))
  diag(dist_mat) <- NA_real_
  distance_breaks <- seq(0, 2000, by = 50)

  moran_results <- vector("list", length(distance_breaks) - 1)
  for (i in seq_len(length(distance_breaks) - 1)) {
    d_low  <- distance_breaks[i]; d_high <- distance_breaks[i + 1]
    w      <- matrix(0, nrow = nrow(dist_mat), ncol = ncol(dist_mat))
    w[!is.na(dist_mat) & dist_mat > d_low & dist_mat <= d_high] <- 1
    if (sum(w) == 0) {
      moran_results[[i]] <- data.frame(d_low = d_low, d_high = d_high,
        d_mid = (d_low + d_high) / 2, morans_I = NA_real_, p_value = NA_real_, significant = NA)
      next
    }
    lw <- spdep::mat2listw(w, style = "W", zero.policy = TRUE)
    mt <- spdep::moran.test(resid_spatial$mean_resid, lw, zero.policy = TRUE)
    moran_results[[i]] <- data.frame(d_low = d_low, d_high = d_high,
      d_mid = (d_low + d_high) / 2,
      morans_I    = unname(mt$estimate[["Moran I statistic"]]),
      p_value     = mt$p.value,
      significant = mt$p.value < 0.05)
  }

  moran_df <- do.call(rbind, moran_results) %>% filter(!is.na(morans_I))

  p_moran <- ggplot(moran_df, aes(x = d_mid, y = morans_I)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "gray50") +
    geom_line(colour = "steelblue", linewidth = 0.8) +
    geom_point(aes(colour = significant, shape = significant), size = 2.5) +
    scale_colour_manual(values = c("TRUE" = "red",  "FALSE" = "gray60"),
                        labels = c("TRUE" = "p < 0.05", "FALSE" = "p ≥ 0.05")) +
    scale_shape_manual(values  = c("TRUE" = 16, "FALSE" = 1),
                       labels  = c("TRUE" = "p < 0.05", "FALSE" = "p ≥ 0.05")) +
    scale_x_continuous(breaks = seq(0, 2000, by = 200)) +
    labs(title    = "Moran's I correlogram of block-level mean Pearson residuals",
         subtitle = "Significant values = residual spatial signal not captured by covariates",
         x = "Distance band midpoint (m)", y = "Moran's I", colour = NULL, shape = NULL) +
    theme_minimal()
  ggsave(file.path(resid_output_dir, paste0("moransI_correlogram_", run_suffix, ".png")),
         p_moran, width = 10, height = 5, dpi = 150)

  moran_file <- file.path(resid_output_dir, paste0("moransI_by_distance_", run_suffix, ".csv"))
  write_csv(moran_df, moran_file)

  n_sig     <- sum(moran_df$significant, na.rm = TRUE)
  first_sig <- moran_df %>% filter(significant) %>% slice(1)
  cat(sprintf("Distance bands tested: %d (50m annuli, 0-2000m)\n", nrow(moran_df)))
  cat(sprintf("Significant bands (p<0.05): %d\n", n_sig))
  if (nrow(first_sig) > 0)
    cat(sprintf("First significant band: %d-%dm (I=%.3f, p=%.4f)\n",
                first_sig$d_low, first_sig$d_high, first_sig$morans_I, first_sig$p_value))

  # --- By year ---
  years <- sort(unique(lubridate::year(df_re_plot$year_month_date)))
  moran_year_list <- vector("list", length(years))

  for (yr in years) {
    block_resid_yr <- df_re_plot %>%
      filter(lubridate::year(year_month_date) == yr) %>%
      group_by(block) %>%
      summarise(mean_resid = mean(pearson_resid, na.rm = TRUE), .groups = "drop") %>%
      mutate(block_chr = as.character(block)) %>%
      left_join(coords_resid, by = "block_chr") %>%
      filter(!is.na(x), !is.na(y), !is.na(mean_resid))

    if (nrow(block_resid_yr) < 10) next

    dist_yr <- as.matrix(dist(as.matrix(block_resid_yr[, c("x", "y")])))
    diag(dist_yr) <- NA_real_

    band_list <- vector("list", length(distance_breaks) - 1)
    for (i in seq_len(length(distance_breaks) - 1)) {
      d_low  <- distance_breaks[i]; d_high <- distance_breaks[i + 1]
      w      <- matrix(0, nrow = nrow(dist_yr), ncol = ncol(dist_yr))
      w[!is.na(dist_yr) & dist_yr > d_low & dist_yr <= d_high] <- 1
      if (sum(w) == 0) {
        band_list[[i]] <- data.frame(year = yr, d_low = d_low, d_high = d_high,
          d_mid = (d_low + d_high) / 2, morans_I = NA_real_, p_value = NA_real_, significant = NA)
        next
      }
      lw <- spdep::mat2listw(w, style = "W", zero.policy = TRUE)
      mt <- spdep::moran.test(block_resid_yr$mean_resid, lw, zero.policy = TRUE)
      band_list[[i]] <- data.frame(year = yr, d_low = d_low, d_high = d_high,
        d_mid = (d_low + d_high) / 2,
        morans_I    = unname(mt$estimate[["Moran I statistic"]]),
        p_value     = mt$p.value,
        significant = mt$p.value < 0.05)
    }
    moran_year_list[[which(years == yr)]] <- do.call(rbind, band_list)
  }

  moran_yr_df <- do.call(rbind, moran_year_list) %>%
    filter(!is.na(morans_I)) %>%
    mutate(year = factor(year))

  p_moran_yr <- ggplot(moran_yr_df, aes(x = d_mid, y = morans_I, colour = year, group = year)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "gray50") +
    geom_line(linewidth = 0.8) +
    geom_point(aes(shape = significant), size = 2) +
    scale_shape_manual(values  = c("TRUE" = 16, "FALSE" = 1),
                       labels  = c("TRUE" = "p < 0.05", "FALSE" = "p ≥ 0.05"),
                       na.value = 1) +
    scale_x_continuous(breaks = seq(0, 2000, by = 200)) +
    labs(title    = "Moran's I correlogram by year",
         subtitle = "Filled points = p < 0.05 · Each line = one calendar year",
         x = "Distance band midpoint (m)", y = "Moran's I", colour = "Year", shape = NULL) +
    theme_minimal()
  ggsave(file.path(resid_output_dir, paste0("moransI_correlogram_by_year_", run_suffix, ".png")),
         p_moran_yr, width = 11, height = 5, dpi = 150)

  moran_yr_file <- file.path(resid_output_dir, paste0("moransI_by_year_distance_", run_suffix, ".csv"))
  write_csv(moran_yr_df, moran_yr_file)
  cat("Year-stratified Moran's I correlogram saved to:", moran_yr_file, "\n")
}
