
# GLMM_quick.r
# Stripped version of GLMM.r that skips SE/CI calculation (the slowest step).
# All outputs go into a "quick/" subdirectory under the normal run folder.
#
# Removed vs GLMM.r:
#   - predict(..., se.fit = TRUE) → predict(...) only
#   - SE/CI columns dropped from df_expanded and df_summary
#   - Three timeseries plots that render CI bands (they need SE columns)
#
# Kept:
#   - Full data prep, model fit, coefficient table, model RDS
#   - Calibration plots (observed vs fitted, weighted)
#   - Residuals plot, random effects plot
#   - Sections 13 & 14 (large-residual diagnostics, AR(1) diagnostics)

library(tidyverse)
library(glmmTMB)
library(slider)
library(sf)

source("plot_functions.r")

if (!require("conflicted", quietly = TRUE)) {
  install.packages("conflicted")
  library(conflicted)
}
conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("lag", "dplyr")

# =========================
# SETTINGS  (keep in sync with GLMM.r)
# =========================
cfg <- list(
  include_block_re     = FALSE,
  include_time_re      = FALSE,
  include_ar1_temporal = TRUE,
  ar1_group            = "block",
  include_spatial_ar   = FALSE,

  link_function = "logit",

  lag_vars      = c("total_precip", "precip_max_day_resid", "avg_VPD", "mean_ndvi"),
  unlagged_vars = c("is_urban", "is_WUI"),
  numeric_vars  = c("total_precip", "precip_max_day_resid", "avg_VPD", "mean_ndvi"),

  interactions  = NULL,

  exclude_predictors = c("mean_ndvi_lag0", "mean_ndvi_lag1", "total_precip_lag1","total_precip_lag2", "avg_VPD_lag1", "avg_VPD_lag2"),
  # exclude_predictors = NULL,
  include_fourier = FALSE,

  shapefile_path = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
  } else {
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
  },
  sf_block_col = "CODIGO_",
  spatial_crs  = NA,

  data_file = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo/env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv"
  } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/data/env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv"
  },

  max_lag  = 2,
  kappa    = 1,

  output_dir = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
  } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
  },

  iter_max = 1e4,
  eval_max = 1e4
)

date_suffix <- format(Sys.Date(), "%Y%m%d")
if (!cfg$ar1_group %in% c("block", "global")) {
  stop("cfg$ar1_group must be either 'block' or 'global'")
}

run_suffix <- date_suffix

time_ar_spec  <- ifelse(cfg$include_ar1_temporal, paste0("AR1-", cfg$ar1_group), "noAR1")
space_ar_spec <- ifelse(cfg$include_spatial_ar, "AR-EXP", "noAR")

model_spec <- paste0(
  "space-", ifelse(cfg$include_block_re, "RE", "noRE"),
  "_time-", ifelse(cfg$include_time_re, "RE", "noRE"),
  "_time-", time_ar_spec,
  "_space-", space_ar_spec,
  "_lag", cfg$max_lag,
  "_k", cfg$kappa,
  "_link-", cfg$link_function
)

predictor_spec <- paste0(
  "lag-", paste(cfg$lag_vars, collapse = "-"),
  "_unlag-", paste(cfg$unlagged_vars, collapse = "-"),
  if (!is.null(cfg$interactions) && length(cfg$interactions) > 0)
    paste0("_ix-", paste(sapply(cfg$interactions, function(p) paste(p, collapse = "x")), collapse = "_"))
  else "",
  if (!is.null(cfg$exclude_predictors) && length(cfg$exclude_predictors) > 0)
    paste0("_excl-", paste(cfg$exclude_predictors, collapse = "-"))
  else "",
  if (isTRUE(cfg$include_fourier)) "_fourier" else ""
)

model_output_dir  <- file.path(cfg$output_dir, "quick", predictor_spec, model_spec)
run_output_dir    <- file.path(model_output_dir, run_suffix)
plots_output_dir  <- file.path(run_output_dir, "plots")
resid_output_dir  <- file.path(run_output_dir, "residuals_check")
dir.create(run_output_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(plots_output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(resid_output_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 1. LOAD DATA
# =========================
df <- read_csv(cfg$data_file)

df <- df %>%
  mutate(
    year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d"),
    year_month      = factor(year_month),
    block           = factor(manzana),
    n_bt            = Inspected_houses + cfg$kappa * cases,
    y_bt            = Houses_pos_IS,
    sin_annual      = sin(2 * pi * as.integer(format(year_month_date, "%m")) / 12),
    cos_annual      = cos(2 * pi * as.integer(format(year_month_date, "%m")) / 12)
  ) %>%
  select(-c(CMF, CP, AREA))

df <- df %>%
  mutate(landcover = factor(landcover), temp_cat = factor(temp_cat), precip_cat = factor(precip_cat))

sf_blocks <- st_read(cfg$shapefile_path, quiet = TRUE)

if (cfg$include_spatial_ar) {
  if (is.null(cfg$shapefile_path) || !file.exists(cfg$shapefile_path)) {
    stop("Spatial autocorrelation requested but shapefile not found: ", cfg$shapefile_path)
  }
  sf_blocks <- st_read(cfg$shapefile_path, quiet = TRUE)
  if (!cfg$sf_block_col %in% names(sf_blocks)) {
    stop("Spatial autocorrelation requested but sf block id column not found: ", cfg$sf_block_col)
  }
  pts <- suppressWarnings(st_point_on_surface(sf_blocks))
  if (isTRUE(sf::st_is_longlat(pts))) {
    pts <- st_transform(pts, 3857)
  } else if (!is.na(cfg$spatial_crs)) {
    pts <- st_transform(pts, cfg$spatial_crs)
  }
  xy <- st_coordinates(pts)
  coords_df <- sf_blocks %>%
    st_drop_geometry() %>%
    mutate(
      block = as.character(.data[[cfg$sf_block_col]]),
      x     = as.numeric(xy[, 1]),
      y     = as.numeric(xy[, 2])
    ) %>%
    select(block, x, y) %>%
    distinct(block, .keep_all = TRUE)
  df <- df %>%
    mutate(block_chr = as.character(block)) %>%
    left_join(coords_df, by = c("block_chr" = "block")) %>%
    select(-block_chr)
  df <- df %>%
    mutate(
      x_sc    = as.numeric(scale(x)),
      y_sc    = as.numeric(scale(y)),
      xy      = glmmTMB::numFactor(x_sc, y_sc),
      spatial = factor("all")
    )
  cat("Spatial coordinates added from shapefile:\n")
  cat("  ", cfg$shapefile_path, "\n", sep = "")
  cat("Rows with missing x/y after join:", sum(is.na(df$x) | is.na(df$y)), "\n\n")
}

time_levels <- df %>%
  distinct(year_month, year_month_date) %>%
  arrange(year_month_date) %>%
  pull(year_month) %>%
  unique()

df <- df %>%
  mutate(year_month_ar1 = factor(year_month, levels = time_levels, ordered = TRUE))

if (identical(cfg$ar1_group, "global")) {
  df <- df %>% mutate(ar1_group = factor("all"))
} else {
  df <- df %>% mutate(ar1_group = block)
}

# =========================
# 2. STANDARDIZE NUMERIC COVARIATES
# =========================
lag_vars      <- cfg$lag_vars
unlagged_vars <- cfg$unlagged_vars
numeric_vars  <- cfg$numeric_vars

for (var in numeric_vars) {
  if (var %in% names(df)) {
    df[[var]] <- (df[[var]] - mean(df[[var]], na.rm = TRUE)) / sd(df[[var]], na.rm = TRUE)
  }
}

# =========================
# 3. CREATE DISTRIBUTED LAGS
# =========================
L <- cfg$max_lag
for (var in lag_vars) {
  is_factor_var <- is.factor(df[[var]]) | is.character(df[[var]])
  for (l in 0:L) {
    lag_col <- paste0(var, "_lag", l)
    df <- df %>%
      group_by(block) %>%
      arrange(year_month_date, .by_group = TRUE) %>%
      mutate(!!lag_col := lag(
        .data[[var]],
        n       = l,
        default = if (is_factor_var) NA_character_ else NA_real_
      )) %>%
      ungroup()
  }
}
lagged_cols <- unlist(lapply(lag_vars, function(v) paste0(v, "_lag", 0:L)))

cat("\n=== CHECKING LAGGED COLUMNS FOR NAs ===\n")
cat("Total rows after lag creation:", nrow(df), "\n")
for (col in lagged_cols) {
  na_count <- sum(is.na(df[[col]]))
  na_pct   <- round(100 * na_count / nrow(df), 2)
  cat(sprintf("%s: %d NAs (%.2f%%)\n", col, na_count, na_pct))
}
cat("=== End lag NA check ===\n\n")

# =========================
# 4. EXPAND DATA FOR REACTIVE MIXTURE
# =========================
cat("\n=== DIAGNOSTIC: Data loss tracking ===\n")
cat("Before reactive mixture:", n_distinct(df$year_month_date), "months\n")

df_with_omega <- df %>%
  mutate(
    omega     = ifelse(cases > 0, pmin(cfg$kappa * cases / n_bt, 1), 0),
    has_cases = cases > 0
  )

df_baseline <- df_with_omega %>%
  mutate(
    n_trials       = floor((1 - omega) * n_bt),
    y_bt_adj       = floor(y_bt * (1 - omega)),
    reactive_shift = 0,
    type           = "baseline"
  ) %>%
  filter(n_trials > 0)

cat("After baseline filter(n_trials > 0):", n_distinct(df_baseline$year_month_date), "months,", nrow(df_baseline), "rows\n")

df_reactive <- df_with_omega %>%
  filter(has_cases) %>%
  mutate(
    n_trials       = floor(omega * n_bt),
    y_bt_adj       = floor(y_bt * omega),
    reactive_shift = log1p(cases),
    type           = "reactive"
  ) %>%
  filter(n_trials > 0)

cat("After reactive filter(n_trials > 0):", n_distinct(df_reactive$year_month_date), "months,", nrow(df_reactive), "rows\n")

df_expanded <- bind_rows(df_baseline, df_reactive) %>%
  select(-has_cases, -y_bt) %>%
  rename(y_bt = y_bt_adj) %>%
  arrange(block, year_month_date, type) %>%
  mutate(y_bt = pmin(y_bt, n_trials))

cat("After combining baseline + reactive:", n_distinct(df_expanded$year_month_date), "months,", nrow(df_expanded), "rows\n")
cat("=== End diagnostic ===\n\n")

# =========================
# 5. BUILD GLMM FORMULA
# =========================
interaction_terms <- c()
if (!is.null(cfg$interactions) && length(cfg$interactions) > 0) {
  for (pair in cfg$interactions) {
    if (length(pair) != 2) stop("Each interaction must be exactly 2 variable names, got: ", paste(pair, collapse = ", "))
    missing_vars <- setdiff(pair, names(df_expanded))
    if (length(missing_vars) > 0) stop("Interaction variable(s) not found in data: ", paste(missing_vars, collapse = ", "))
    interaction_terms <- c(interaction_terms, paste(pair[1], pair[2], sep = ":"))
  }
}

fixed_effects <- c(lagged_cols, unlagged_vars, "reactive_shift", interaction_terms)
if (isTRUE(cfg$include_fourier)) {
  fixed_effects <- c(fixed_effects, "sin_annual", "cos_annual")
  cat("Fourier terms added: sin_annual, cos_annual\n")
}
if (!is.null(cfg$exclude_predictors) && length(cfg$exclude_predictors) > 0) {
  unknown <- setdiff(cfg$exclude_predictors, fixed_effects)
  if (length(unknown) > 0)
    warning("exclude_predictors contains names not found in fixed_effects: ",
            paste(unknown, collapse = ", "))
  fixed_effects <- setdiff(fixed_effects, cfg$exclude_predictors)
  cat("Excluded predictors:", paste(cfg$exclude_predictors, collapse = ", "), "\n")
}

formula_str <- paste(
  "cbind(y_bt, n_trials - y_bt) ~",
  paste(fixed_effects, collapse = " + ")
)

random_effects <- c()
if (cfg$include_block_re)     random_effects <- c(random_effects, "(1 | block)")
if (cfg$include_time_re)      random_effects <- c(random_effects, "(1 | year_month)")
if (cfg$include_ar1_temporal) random_effects <- c(random_effects, "ar1(year_month_ar1 + 0 | ar1_group)")
if (cfg$include_spatial_ar)   random_effects <- c(random_effects, "exp(xy + 0 | spatial, range = 400)")

if (length(random_effects) > 0) {
  formula_str <- paste(formula_str, "+", paste(random_effects, collapse = " + "))
}

cat("GLMM Formula:\n")
cat(formula_str, "\n\n")

formula <- as.formula(formula_str)

# =========================
# 6. FIT GLMM
# =========================
required_cols <- unique(c(
  "y_bt", "n_trials",
  lagged_cols, unlagged_vars, "reactive_shift",
  unlist(cfg$interactions),
  if (cfg$include_block_re)     "block"                          else NULL,
  if (cfg$include_time_re)      "year_month"                     else NULL,
  if (cfg$include_ar1_temporal) c("year_month_ar1", "ar1_group") else NULL,
  if (cfg$include_spatial_ar)   c("x_sc", "y_sc", "xy", "spatial") else NULL
))

missing_required <- setdiff(required_cols, names(df_expanded))
if (length(missing_required) > 0) {
  stop("Missing required columns for model fit: ", paste(missing_required, collapse = ", "))
}

keep_rows <- complete.cases(df_expanded[, required_cols, drop = FALSE])

cat("Rows in df_expanded:", nrow(df_expanded), "\n")
cat("Rows used for model fit (complete cases):", sum(keep_rows), "\n")
cat("Rows excluded due to NA in model terms:", sum(!keep_rows), "\n")

if (sum(keep_rows) == 0) {
  stop("No complete rows available for model fitting after NA filtering.")
}

df_model <- df_expanded[keep_rows, , drop = FALSE]

if (exists(".glmm_data_prep_only") && isTRUE(.glmm_data_prep_only)) {
  cat("Data prep complete — stopping before model fit (.glmm_data_prep_only = TRUE)\n")
  stop(".glmm_data_prep_only")
}

cat("\nStarting model fit...\n")
cat("Formula: ", formula_str, "\n")
cat("Observations: ", nrow(df_model), "\n\n")

model <- glmmTMB(
  formula,
  family  = binomial(link = cfg$link_function),
  data    = df_model,
  control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 10))
)

cat("\nModel fit complete!\n\n")

model_file <- file.path(run_output_dir, paste0("glmm_model_", run_suffix, ".rds"))
saveRDS(model, model_file)

# =========================
# 7. INSPECT RESULTS
# =========================
cat("Model Summary:\n")
print(summary(model))

coef_mat   <- summary(model)$coefficients$cond
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

summary_file   <- file.path(run_output_dir, paste0("glmm_summary_", run_suffix, ".txt"))
summary_output <- capture.output(summary(model))
writeLines(c(
  paste0("Run suffix: ", run_suffix),
  paste0("Model spec folder: ", model_spec),
  paste0("Formula: ", formula_str),
  "",
  summary_output
), con = summary_file)

# =========================
# 8. ADD PREDICTIONS (fitted p_bt) — no SE
# =========================
cat("df_expanded rows:", nrow(df_expanded), "\n")

# predict() without se.fit — skips the slow delta-method SE calculation
fit <- predict(model, type = "response")
cat("Predictions length:", length(fit), "\n")

if (length(fit) == nrow(df_model)) {
  df_expanded <- df_expanded %>%
    mutate(fitted_prob = NA_real_)
  df_expanded$fitted_prob[keep_rows] <- fit
} else {
  stop("Mismatch: ", nrow(df_model), " model rows vs ", length(fit), " predictions")
}

# =========================
# 9. AGGREGATE BACK TO ORIGINAL OBSERVATION LEVEL
# =========================
df_summary <- df_expanded %>%
  select(block, year_month_date, type, omega, fitted_prob) %>%
  pivot_wider(
    names_from  = type,
    values_from = fitted_prob,
    values_fill = NA
  )

# When kappa = 0 there are no reactive rows — add as NA so rename always works
if (!"reactive" %in% names(df_summary)) df_summary$reactive <- NA_real_

df_summary <- df_summary %>%
  rename(
    p_bt_fitted = baseline,
    p_R_fitted  = reactive
  )

expanded_file     <- file.path(run_output_dir, paste0("glmm_expanded_predictions_", run_suffix, ".csv"))
summary_pred_file <- file.path(run_output_dir, paste0("glmm_summary_predictions_", run_suffix, ".csv"))
cfg_file          <- file.path(run_output_dir, paste0("glmm_config_", run_suffix, ".rds"))

write_csv(df_expanded, expanded_file)
write_csv(df_summary,  summary_pred_file)
saveRDS(cfg, cfg_file)

cat("\nSaved outputs:\n")
cat("  Run folder (quick):        ", run_output_dir, "\n", sep = "")
cat("  Model RDS:                 ", model_file, "\n", sep = "")
cat("  Summary TXT:               ", summary_file, "\n", sep = "")
cat("  Fixed effects OR CSV:      ", coef_table_file, "\n", sep = "")
cat("  Expanded predictions CSV:  ", expanded_file, "\n", sep = "")
cat("  Aggregated predictions CSV:", summary_pred_file, "\n", sep = "")
cat("  Config RDS:                ", cfg_file, "\n", sep = "")

# =========================
# 10. PLOTS (no CI bands — calibration plots only)
# =========================
# Timeseries plots with CI bands are skipped (they need SE columns).
# Run full GLMM.r to get those.

df_observed <- df %>%
  transmute(
    block,
    year_month_date,
    p_observed = ifelse(n_bt > 0, y_bt / n_bt, NA_real_),
    cases
  )

df_summary_weighted <- df_summary %>%
  left_join(df_observed, by = c("block", "year_month_date")) %>%
  mutate(
    p_fitted_weighted = ifelse(
      omega == 0 | is.na(p_R_fitted),
      p_bt_fitted,
      (1 - omega) * p_bt_fitted + omega * p_R_fitted
    )
  )

# Calibration plot: observed vs baseline fitted
save_glmm_calibplot_observed_vs_expected(
  df_summary  = df_summary,
  df_observed = df_observed,
  output_dir  = plots_output_dir,
  run_suffix  = run_suffix
)

# Calibration plot: observed vs weighted fitted
save_glmm_calibplot_weighted_avg(
  df         = df_summary_weighted,
  output_dir = plots_output_dir,
  run_suffix = run_suffix
)

# =========================
# 11. PLOT MODEL RESIDUALS
# =========================
save_glmm_residuals_plot(model, resid_output_dir, run_suffix)

# =========================
# 12. PLOT RANDOM EFFECTS
# =========================
save_glmm_random_effects_plot(model, plots_output_dir, run_suffix)

# =========================
# 13. LARGE-RESIDUAL DIAGNOSTICS
# =========================
pearson_resid <- residuals(model, type = "pearson")

df_resid_diag <- df_model %>%
  mutate(
    fitted_prob   = fitted(model),
    pearson_resid = pearson_resid,
    p_observed    = ifelse(n_trials > 0, y_bt / n_trials, NA_real_),
    abs_resid     = abs(pearson_resid)
  ) %>%
  filter(abs_resid > 2) %>%
  select(block, year_month_date, type, y_bt, n_trials, n_bt, omega,
         p_observed, fitted_prob, pearson_resid, abs_resid) %>%
  arrange(desc(abs_resid))

resid_diag_file <- file.path(resid_output_dir, paste0("glmm_large_residuals_", run_suffix, ".csv"))
write_csv(df_resid_diag, resid_diag_file)

cat("\nLarge-residual rows (|Pearson resid| > 2):", nrow(df_resid_diag),
    "out of", nrow(df_model), "model rows\n")
cat("  Saved to:", resid_diag_file, "\n")

block_resid_summary <- df_resid_diag %>%
  group_by(block) %>%
  summarise(
    n_large_resid     = n(),
    mean_pearson      = mean(pearson_resid),
    max_pearson       = max(abs_resid),
    mean_p_observed   = mean(p_observed, na.rm = TRUE),
    mean_fitted       = mean(fitted_prob, na.rm = TRUE),
    pct_positive_bias = mean(pearson_resid > 0)
  ) %>%
  arrange(desc(n_large_resid))

total_timepoints    <- df_model %>% count(block, name = "n_months")
block_resid_summary <- block_resid_summary %>%
  left_join(total_timepoints, by = "block") %>%
  mutate(pct_months_flagged = n_large_resid / n_months) %>%
  arrange(desc(pct_months_flagged))

cat("\nBlocks with large residuals in >30% of months:\n")
print(filter(block_resid_summary, pct_months_flagged > 0.30))

temporal_resid_summary <- df_resid_diag %>%
  group_by(year_month_date) %>%
  summarise(
    n_large_resid = n(),
    mean_pearson  = mean(pearson_resid),
    pct_positive  = mean(pearson_resid > 0)
  ) %>%
  arrange(desc(n_large_resid))

cat("\nMonths with most large residuals:\n")
print(head(temporal_resid_summary, 10))

df_model %>%
  mutate(pearson_resid = pearson_resid, large_resid = abs(pearson_resid) > 2) %>%
  group_by(large_resid) %>%
  summarise(
    mean_omega      = mean(omega, na.rm = TRUE),
    median_omega    = median(omega, na.rm = TRUE),
    pct_reactive    = mean(omega > 0, na.rm = TRUE),
    mean_n_trials   = mean(n_trials),
    mean_p_observed = mean(ifelse(n_trials > 0, y_bt / n_trials, NA), na.rm = TRUE)
  )

df_resid_diag %>%
  mutate(n_trials_cat = cut(n_trials,
                             breaks = c(0, 1, 2, 5, 10, 20, Inf),
                             labels = c("1", "2", "3-5", "6-10", "11-20", ">20"))) %>%
  count(n_trials_cat, name = "n_large_resid") %>%
  mutate(pct = round(100 * n_large_resid / sum(n_large_resid), 1))

block_resid_file <- file.path(resid_output_dir,
                               paste0("glmm_block_resid_summary_", run_suffix, ".csv"))
write_csv(block_resid_summary, block_resid_file)
cat("Block residual summary saved to:", block_resid_file, "\n")

# =========================
# 14. AR(1) TEMPORAL VARIANCE DIAGNOSTICS
# =========================
df_re_plot <- df_model %>%
  mutate(
    fitted_prob   = fitted(model),
    pearson_resid = pearson_resid
  )

monthly_resid <- df_re_plot %>%
  group_by(year_month_date) %>%
  summarise(
    mean_resid    = mean(pearson_resid, na.rm = TRUE),
    median_resid  = median(pearson_resid, na.rm = TRUE),
    mean_fitted   = mean(fitted_prob, na.rm = TRUE),
    mean_observed = mean(ifelse(n_trials > 0, y_bt / n_trials, NA_real_), na.rm = TRUE),
    n             = n(),
    .groups       = "drop"
  )

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

p_obs_vs_fit <- ggplot(monthly_resid, aes(x = year_month_date)) +
  geom_line(aes(y = mean_observed, colour = "Observed")) +
  geom_line(aes(y = mean_fitted,   colour = "Fitted")) +
  geom_point(aes(y = mean_observed, colour = "Observed"), size = 1.5) +
  geom_point(aes(y = mean_fitted,   colour = "Fitted"),   size = 1.5) +
  scale_colour_manual(values = c("Observed" = "black", "Fitted" = "steelblue")) +
  labs(title    = "Observed vs fitted positivity rate over time",
       subtitle = "Persistent gap = temporal variance not captured by the model",
       x = "Month", y = "Mean positivity rate", colour = NULL) +
  theme_minimal()

ggsave(file.path(resid_output_dir, paste0("ar1_obs_vs_fitted_time_", run_suffix, ".png")),
       p_obs_vs_fit, width = 10, height = 5, dpi = 150)

block_monthly <- df_re_plot %>%
  group_by(block, year_month_date) %>%
  summarise(
    mean_fitted   = mean(fitted_prob, na.rm = TRUE),
    mean_observed = mean(ifelse(n_trials > 0, y_bt / n_trials, NA_real_), na.rm = TRUE),
    .groups = "drop"
  )

set.seed(42)
sample_blocks <- sample(unique(block_monthly$block), min(50, n_distinct(block_monthly$block)))

p_block_traj <- ggplot(
  filter(block_monthly, block %in% sample_blocks),
  aes(x = year_month_date, y = mean_fitted, group = block)
) +
  geom_line(alpha = 0.2, colour = "steelblue") +
  geom_line(data = monthly_resid,
            aes(x = year_month_date, y = mean_fitted, group = NULL),
            colour = "red", linewidth = 1.2) +
  labs(title    = "Fitted trajectories for 50 random blocks",
       subtitle = "Red = city-wide mean fitted. Spread = block heterogeneity in AR(1)",
       x = "Month", y = "Fitted positivity rate") +
  theme_minimal()

ggsave(file.path(resid_output_dir, paste0("ar1_block_trajectories_", run_suffix, ".png")),
       p_block_traj, width = 10, height = 6, dpi = 150)

resid_ts <- monthly_resid %>% arrange(year_month_date) %>% pull(mean_resid)
acf_vals  <- acf(resid_ts, lag.max = 24, plot = FALSE, na.action = na.pass)
acf_df    <- data.frame(lag = as.numeric(acf_vals$lag), acf = as.numeric(acf_vals$acf))
ci_bound  <- qnorm(0.975) / sqrt(length(resid_ts))

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
