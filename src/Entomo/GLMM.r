
library(tidyverse)
library(glmmTMB)
library(slider)
library(sf)

# Source plot functions for GLMM diagnostics
source("plot_functions.r")

# Resolve namespace conflicts - prefer tidyverse/dplyr versions
if (!require("conflicted", quietly = TRUE)) {
  install.packages("conflicted")
  library(conflicted)
}
conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("lag", "dplyr")

# =========================
# SETTINGS
# =========================
cfg <- list(
  # Random effects to include
  include_block_re = FALSE,      # Random intercept for block (spatial)
  include_time_re = FALSE,      # Random intercept for time (temporal)
  include_ar1_temporal = TRUE, # AR(1) temporal autocorrelation (within group)
  ar1_group = "block",         # "block" (within-block AR1) or "global"
  include_spatial_ar = FALSE,  # Exponential spatial autocorrelation: exp(xy + 0 | spatial)
  # include_spatial_ar = TRUE,  # Matérn spatial autocorrelation: mat(xy + 0 | spatial)

  # Link function for binomial GLMM
  link_function = "logit",     # Options: "logit", "probit", "cloglog", "cauchit"
  # Predictors
  # lag_vars     : variables for which distributed lags are created
  # unlagged_vars: variables entered directly without lag
  # numeric_vars : continuous variables to z-score standardize
  #                (exclude factors, binary 0/1, and already-factored variables)
  lag_vars      = c("total_rainy_days", "avg_VPD", "precip_max_day"),
  unlagged_vars = c("is_urban", "water_containers"),
  numeric_vars  = c("precip_max_day", "avg_VPD", "water_containers"),

  # Interaction terms (NULL = none)
  # Each element is a character vector of exactly 2 variable names (column names
  # after lag expansion, e.g. "temp_cat_lag1", or unlagged names e.g. "is_urban")
  # Example: interactions = list(c("temp_cat_lag1", "avg_VPD_lag1"), c("is_urban", "water_containers"))
  interactions  = NULL,

  # Predictors to drop after lag expansion (NULL = keep all)
  # Use the fully expanded column name, e.g. "avg_VPD_lag1", "is_urban"
  exclude_predictors = c("avg_VPD_lag1"),
  # exclude_predictors = NULL,

  # Add sin/cos annual Fourier terms as fixed effects (2-parameter seasonal cycle).
  # Tests whether residual seasonality is independent of the climate covariates.
  include_fourier = FALSE,

  # Spatial coordinates from shapefile (used when include_spatial_ar = TRUE)
  shapefile_path = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
  } else {
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
    },
  sf_block_col = "CODIGO_",
  spatial_crs = NA,             # Optional projected CRS (e.g., 32719). NA = keep CRS unless lon/lat (then use EPSG:3857)

  # Data
  data_file = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo/env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv"
  } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/data/env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv"
  },

  # Lag settings
  max_lag = 2,
  kappa = 2,  # multiplier for cases in n_bt calculation

  # Output
  output_dir = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
    } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
    },
  # GLMM control
  iter_max = 1e4,
  eval_max = 1e4
)

date_suffix <- format(Sys.Date(), "%Y%m%d")
if (!cfg$ar1_group %in% c("block", "global")) {
  stop("cfg$ar1_group must be either 'block' or 'global'")
}

ar1_suffix <- ifelse(
  cfg$include_ar1_temporal,
  paste0("AR1-", cfg$ar1_group),
  "noAR1"
)

# Run suffix is the date
run_suffix <- date_suffix

# Output structure:
# <output_dir>/<predictor_spec>/<model_spec>/<run_suffix>/plots/
time_ar_spec <- ifelse(cfg$include_ar1_temporal, paste0("AR1-", cfg$ar1_group), "noAR1")
space_ar_spec <- ifelse(cfg$include_spatial_ar, "AR-EXP", "noAR")
# space_ar_spec <- ifelse(cfg$include_spatial_ar, "AR-Mat", "noAR")

model_spec <- paste0(
  "space-", ifelse(cfg$include_block_re, "RE", "noRE"),
  "_time-", ifelse(cfg$include_time_re, "RE", "noRE"),
  "_time-", time_ar_spec,
  "_space-", space_ar_spec,
  "_lag", cfg$max_lag,
  "_k", cfg$kappa,
  "_link-", cfg$link_function
)

# Encode predictor sets in folder name so each combination gets its own directory
predictor_spec <- paste0(
  "lag-", paste(cfg$lag_vars, collapse = "-"),
  "_unlag-", paste(cfg$unlagged_vars, collapse = "-"),
  if (!is.null(cfg$interactions) && length(cfg$interactions) > 0)
    paste0("_ix-", paste(sapply(cfg$interactions, function(p) paste(p, collapse = "x")), collapse = "_"))
  else "",
  if (isTRUE(cfg$include_fourier)) "_fourier" else ""
)

model_output_dir  <- file.path(cfg$output_dir, predictor_spec, model_spec)
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
    year_month = factor(year_month),  # For time RE
    block = factor(manzana), # for space RE
    n_bt = Inspected_houses + cfg$kappa * cases, # fixed number observations = universe + extra observations for cases
    y_bt = Houses_pos_IS,
    sin_annual = sin(2 * pi * as.integer(format(year_month_date, "%m")) / 12),
    cos_annual = cos(2 * pi * as.integer(format(year_month_date, "%m")) / 12)
  ) %>%
  select(-c(CMF, CP, AREA))

# create landcover as factor variable 
df <- df %>%
  mutate(landcover = factor(landcover), temp_cat = factor(temp_cat), precip_cat = factor(precip_cat))

# Optional: add spatial coordinates by matching block names to shapefile IDs

sf_blocks <- st_read(cfg$shapefile_path, quiet = TRUE)

if (cfg$include_spatial_ar) {
  if (is.null(cfg$shapefile_path) || !file.exists(cfg$shapefile_path)) {
    stop("Spatial autocorrelation requested but shapefile not found: ", cfg$shapefile_path)
  }

  sf_blocks <- st_read(cfg$shapefile_path, quiet = TRUE)
  if (!cfg$sf_block_col %in% names(sf_blocks)) {
    stop("Spatial autocorrelation requested but sf block id column not found: ", cfg$sf_block_col)
  }

  # Use representative point coordinates (stable for polygons)
  pts <- suppressWarnings(st_point_on_surface(sf_blocks))

  # If lon/lat, transform to projected CRS for distance-based covariance
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
      x = as.numeric(xy[, 1]),
      y = as.numeric(xy[, 2])
    ) %>%
    select(block, x, y) %>%
    distinct(block, .keep_all = TRUE)

  df <- df %>%
    mutate(block_chr = as.character(block)) %>%
    left_join(coords_df, by = c("block_chr" = "block")) %>%
    select(-block_chr)

  # Scale coordinates for better numerical stability, then encode for glmmTMB spatial covariance
  df <- df %>%
    mutate(
      x_sc = as.numeric(scale(x)),
      y_sc = as.numeric(scale(y)),
      xy = glmmTMB::numFactor(x_sc, y_sc)
    )

  # Single spatial field across all observations
  df <- df %>% mutate(spatial = factor("all"))

  cat("Spatial coordinates added from shapefile:\n")
  cat("  ", cfg$shapefile_path, "\n", sep = "")
  cat("Rows with missing x/y after join:", sum(is.na(df$x) | is.na(df$y)), "\n\n")
}

# Ordered monthly factor for AR(1) (must be in temporal order)
time_levels <- df %>%
  distinct(year_month, year_month_date) %>%
  arrange(year_month_date) %>%
  pull(year_month) %>%
  unique()

df <- df %>%
  mutate(year_month_ar1 = factor(year_month, levels = time_levels, ordered = TRUE))

# Grouping factor for AR(1): per-block or global
if (identical(cfg$ar1_group, "global")) {
  df <- df %>% mutate(ar1_group = factor("all"))
} else {
  df <- df %>% mutate(ar1_group = block)
}


# # =========================
# # OPTIONAL: SUBSET TO FIRST 100 BLOCKS (for testing)
# # =========================
# # Uncomment to enable faster testing
# set.seed(42)  # for reproducibility
# unique_blocks <- unique(df$block)
# blocks_to_keep <- sample(unique_blocks[1:min(100, length(unique_blocks))], size = min(100, length(unique_blocks)))
# df <- df %>% filter(block %in% blocks_to_keep)
# cat("Subset to", n_distinct(df$block), "blocks\n")


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
  is_factor_var <- is.factor(df[[var]]) | is.character(df[[var]])  # catch both
  
  for (l in 0:L) {
    lag_col <- paste0(var, "_lag", l)
    df <- df %>%
      group_by(block) %>%
      arrange(year_month_date, .by_group = TRUE) %>%
      mutate(!!lag_col := lag(
        .data[[var]], 
        n = l, 
        default = if (is_factor_var) NA_character_ else NA_real_  # explicit NA types
      )) %>%
      ungroup()
  }
}
lagged_cols <- unlist(lapply(lag_vars, function(v) paste0(v, "_lag", 0:L)))

# Check for NAs in lagged columns
cat("\n=== CHECKING LAGGED COLUMNS FOR NAs ===\n")
cat("Total rows after lag creation:", nrow(df), "\n")
for (col in lagged_cols) {
  na_count <- sum(is.na(df[[col]]))
  na_pct <- round(100 * na_count / nrow(df), 2)
  cat(sprintf("%s: %d NAs (%.2f%%)\n", col, na_count, na_pct))
}
cat("=== End lag NA check ===\n\n")

# =========================
# 4. EXPAND DATA FOR REACTIVE MIXTURE
# =========================
# For observations with cases > 0, split into baseline and reactive components
# Otherwise keep as single baseline observation

cat("\n=== DIAGNOSTIC: Data loss tracking ===\n")
cat("Before reactive mixture:", n_distinct(df$year_month_date), "months\n")

# omega represent the proportion of visits that was reactive (due to cases)
df_with_omega <- df %>%
  mutate(
    # omega represents the proportion of observation allocated to reactive surveillance
    omega = ifelse(cases > 0, pmin(cfg$kappa * cases / n_bt, 1), 0),  # ensure omega <= 1
    has_cases = cases > 0
  )

# Create baseline (systematic surveillance) rows
df_baseline <- df_with_omega %>%
  mutate( 
    # For baseline: use (1-omega) proportion of trials - this is the same as n_HH (universo)
    n_trials = floor((1 - omega) * n_bt), # systematic surveillance
    # Assume same proportion of positives
    y_bt_adj = floor(y_bt * (1 - omega)),
    reactive_shift = 0,
    type = "baseline"
  ) %>%
  filter(n_trials > 0)

cat("After baseline filter(n_trials > 0):", n_distinct(df_baseline$year_month_date), "months,", nrow(df_baseline), "rows\n")

# Create reactive rows (only for observations with cases > 0)
df_reactive <- df_with_omega %>%
  filter(has_cases) %>%
  mutate(
    # For reactive: use omega proportion of trials
    n_trials = floor(omega * n_bt),
    # Assume same proportion of positives (or higher due to reactive effect)
    y_bt_adj = floor(y_bt * omega),
    reactive_shift = log1p(cases),
    type = "reactive"
  ) %>%
  filter(n_trials > 0)

cat("After reactive filter(n_trials > 0):", n_distinct(df_reactive$year_month_date), "months,", nrow(df_reactive), "rows\n")

# Combine
df_expanded <- bind_rows(df_baseline, df_reactive) %>%
  select(-has_cases, -y_bt) %>%  # keep omega
  rename(y_bt = y_bt_adj) %>%
  arrange(block, year_month_date, type) %>%
  # Ensure y_bt doesn't exceed n_trials
  mutate(y_bt = pmin(y_bt, n_trials))

cat("After combining baseline + reactive:", n_distinct(df_expanded$year_month_date), "months,", nrow(df_expanded), "rows\n")

# Note: Lag columns already handled NAs via default=0 during creation, no further filtering needed
cat("=== End diagnostic ===\n\n")

# =========================
# 5. BUILD GLMM FORMULA
# =========================
# Interaction terms (added as var1:var2 — main effects already listed above)
interaction_terms <- c()
if (!is.null(cfg$interactions) && length(cfg$interactions) > 0) {
  for (pair in cfg$interactions) {
    if (length(pair) != 2) stop("Each interaction must be exactly 2 variable names, got: ", paste(pair, collapse = ", "))
    missing_vars <- setdiff(pair, names(df_expanded))
    if (length(missing_vars) > 0) stop("Interaction variable(s) not found in data: ", paste(missing_vars, collapse = ", "))
    interaction_terms <- c(interaction_terms, paste(pair[1], pair[2], sep = ":"))
  }
}

# Fixed effects: main effects + interactions, minus any explicitly excluded predictors
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
  "cbind(y_bt, n_trials - y_bt) ~", # y_bt successes (positives), n_trails - y_bt failures (zeros)
  paste(fixed_effects, collapse = " + ")
)

# Random effects
random_effects <- c()
if (cfg$include_block_re) {
  random_effects <- c(random_effects, "(1 | block)")
}
if (cfg$include_time_re) {
  random_effects <- c(random_effects, "(1 | year_month)")
}
if (cfg$include_ar1_temporal) {
  random_effects <- c(random_effects, "ar1(year_month_ar1 + 0 | ar1_group)")
}
if (cfg$include_spatial_ar) {
  random_effects <- c(random_effects, "exp(xy + 0 | spatial, range = 400)")
  # random_effects <- c(random_effects, "mat(xy + 0 | spatial, range = 400)")
}

if (length(random_effects) > 0) {
  formula_str <- paste(formula_str, "+", paste(random_effects, collapse = " + "))
}

cat("GLMM Formula:\n")
cat(formula_str, "\n\n")

formula <- as.formula(formula_str)

# =========================
# 6. FIT GLMM
# =========================

# glmmTMB drops rows with missing values in model terms.
# Build complete-case mask explicitly (without model.frame on random effects).
# Use individual column names (not formula terms like "var1:var2") for NA checking
required_cols <- unique(c(
  "y_bt", "n_trials",
  lagged_cols, unlagged_vars, "reactive_shift",
  unlist(cfg$interactions),  # individual vars from interaction pairs (already in lagged/unlagged but explicit)
  if (cfg$include_block_re) "block" else NULL,
  if (cfg$include_time_re) "year_month" else NULL,
  if (cfg$include_ar1_temporal) c("year_month_ar1", "ar1_group") else NULL,
  if (cfg$include_spatial_ar) c("x_sc", "y_sc", "xy", "spatial") else NULL
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
  stop("No complete rows available for model fitting after NA filtering. Check missingness in model terms.")
}

df_model <- df_expanded[keep_rows, , drop = FALSE]

# Early-exit hook: allows sourcing scripts to stop here after data prep
# without running the model fit. Set .glmm_data_prep_only <- TRUE before
# source("GLMM.r") to use this.
if (exists(".glmm_data_prep_only") && isTRUE(.glmm_data_prep_only)) {
  cat("Data prep complete — stopping before model fit (.glmm_data_prep_only = TRUE)\n")
  stop(".glmm_data_prep_only")
}

cat("\nStarting model fit...\n")
cat("Formula: ", formula_str, "\n")
cat("Observations: ", nrow(df_model), "\n\n")

model <- glmmTMB(
  formula,
  family = binomial(link = cfg$link_function),
  data = df_model,
  control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 10))
)

cat("\nModel fit complete!\n\n")

# Save fitted model object
model_file <- file.path(run_output_dir, paste0("glmm_model_", run_suffix, ".rds"))
saveRDS(model, model_file)


# =========================
# 7. INSPECT RESULTS
# =========================
cat("Model Summary:\n")
print(summary(model))

# Helper: fixed-effects interpretation table
coef_mat <- summary(model)$coefficients$cond
coef_table <- as_tibble(coef_mat, rownames = "term") %>%
  transmute(
    term,
    estimate = Estimate,
    std_error = `Std. Error`,
    z_value = `z value`,
    p_value = `Pr(>|z|)`,
    OR = exp(estimate),
    pct_change_odds = (OR - 1) * 100
  ) %>%
  arrange(desc(abs(estimate)))

cat("\nFixed-effects interpretation (log-odds and odds ratios):\n")
print(coef_table, n = nrow(coef_table))

# Save coefficient interpretation table
coef_table_file <- file.path(run_output_dir, paste0("glmm_fixed_effects_OR_", run_suffix, ".csv"))
write_csv(coef_table, coef_table_file)

# Save model summary and formula
summary_file <- file.path(run_output_dir, paste0("glmm_summary_", run_suffix, ".txt"))
summary_output <- capture.output(summary(model))
writeLines(c(
  paste0("Run suffix: ", run_suffix),
  paste0("Model spec folder: ", model_spec),
  paste0("Formula: ", formula_str),
  "",
  summary_output
), con = summary_file)

# =========================
# 8. ADD PREDICTIONS (fitted p_bt)
# =========================

cat("df_expanded rows:", nrow(df_expanded), "\n")
preds_with_se <- predict(model, type = "response", se.fit = TRUE)
cat("Predictions length:", length(preds_with_se$fit), "\n")

# Compute 95% confidence intervals
fit <- preds_with_se$fit
se <- preds_with_se$se.fit
lower <- fit - 1.96 * se
upper <- fit + 1.96 * se

# Add predictions and uncertainty back to full df_expanded safely
if (length(fit) == nrow(df_model)) {
  df_expanded <- df_expanded %>%
    mutate(
      fitted_prob = NA_real_,
      fitted_prob_se = NA_real_,
      fitted_prob_lower = NA_real_,
      fitted_prob_upper = NA_real_
    )
  df_expanded$fitted_prob[keep_rows] <- fit
  df_expanded$fitted_prob_se[keep_rows] <- se
  df_expanded$fitted_prob_lower[keep_rows] <- lower
  df_expanded$fitted_prob_upper[keep_rows] <- upper
} else {
  stop("Mismatch: ", nrow(df_model), " model rows vs ", length(fit), " predictions")
}

# =========================
# 9. AGGREGATE BACK TO ORIGINAL OBSERVATION LEVEL
# =========================

# For each original observation, get baseline p_bt and reactive p_R predictions and their uncertainty
df_summary <- df_expanded %>%
  select(block, year_month_date, type, omega, fitted_prob, fitted_prob_se, fitted_prob_lower, fitted_prob_upper) %>%
  pivot_wider(
    names_from = type, # splits fitted_prob into p_bt and p_R
    values_from = c(fitted_prob, fitted_prob_se, fitted_prob_lower, fitted_prob_upper),
    values_fill = NA
  )

# When kappa = 0 there are no reactive rows, so pivot_wider won't create
# the reactive columns — add them as NA so the rename below always works.
reactive_cols <- c("fitted_prob_reactive", "fitted_prob_se_reactive",
                   "fitted_prob_lower_reactive", "fitted_prob_upper_reactive")
for (col in reactive_cols) {
  if (!col %in% names(df_summary)) df_summary[[col]] <- NA_real_
}

df_summary <- df_summary %>%
  rename(
    p_bt_fitted = fitted_prob_baseline,
    p_R_fitted = fitted_prob_reactive,
    p_bt_fitted_se = fitted_prob_se_baseline,
    p_R_fitted_se = fitted_prob_se_reactive,
    p_bt_fitted_lower = fitted_prob_lower_baseline,
    p_R_fitted_lower = fitted_prob_lower_reactive,
    p_bt_fitted_upper = fitted_prob_upper_baseline,
    p_R_fitted_upper = fitted_prob_upper_reactive
  )

# Save data outputs for later checking/calling
expanded_file <- file.path(run_output_dir, paste0("glmm_expanded_predictions_", run_suffix, ".csv"))
summary_pred_file <- file.path(run_output_dir, paste0("glmm_summary_predictions_", run_suffix, ".csv"))
cfg_file <- file.path(run_output_dir, paste0("glmm_config_", run_suffix, ".rds"))

write_csv(df_expanded, expanded_file)
write_csv(df_summary, summary_pred_file)
saveRDS(cfg, cfg_file)

cat("\nSaved outputs:\n")
cat("  Predictor set:             ", predictor_spec, "\n", sep = "")
cat("  Model spec folder:         ", model_output_dir, "\n", sep = "")
cat("  Run folder:                ", run_output_dir, "\n", sep = "")
cat("  Plots folder:              ", plots_output_dir, "\n", sep = "")
cat("  Residuals folder:          ", resid_output_dir, "\n", sep = "")
cat("  Model RDS:                 ", model_file, "\n", sep = "")
cat("  Summary TXT:               ", summary_file, "\n", sep = "")
cat("  Fixed effects OR CSV:      ", coef_table_file, "\n", sep = "")
cat("  Expanded predictions CSV:  ", expanded_file, "\n", sep = "")
cat("  Aggregated predictions CSV:", summary_pred_file, "\n", sep = "")
cat("  Config RDS:                ", cfg_file, "\n", sep = "")

# =========================
# 10. PLOT p_bt_fitted, p_R_fitted, p_bt_observed
# =========================
df_observed <- df %>%
  transmute(
    block,
    year_month_date,
    p_observed = ifelse(n_bt > 0, y_bt / n_bt, NA_real_),
    cases
  )


# Use modular plot function instead
save_glmm_prob_timeseries_plot(
  df_summary = df_summary,
  df_observed = df_observed,
  output_dir = plots_output_dir,
  run_suffix = run_suffix,
  cfg = cfg
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

save_glmm_prob_timeseries_plot_random_blocks(
    df_summary = df_summary,
    df_observed = df_observed,
    output_dir = plots_output_dir,
    run_suffix = run_suffix,
    cfg = cfg,
    n_blocks = 10)

save_glmm_prob_timeseries_plot_weighted(
  df_summary = df_summary_weighted,
  output_dir = plots_output_dir,
  run_suffix = run_suffix,
  cfg = cfg
)

# QQ plot of observed vs expected (fitted) probabilities (aggregated over time)
save_glmm_qqplot_observed_vs_expected(
  df_summary = df_summary,
  df_observed = df_observed,
  output_dir = plots_output_dir,
  run_suffix = run_suffix
)

# QQ plot of observed vs weighted average fitted probability
save_glmm_qqplot_weighted_avg(
  df = df_summary_weighted,
  output_dir = plots_output_dir,
  run_suffix = run_suffix
)

# =========================
# 11. PLOT MODEL RESIDUALS (using plot_functions)
# =========================
save_glmm_residuals_plot(model, resid_output_dir, run_suffix)

# =========================
# 12. PLOT RANDOM EFFECTS (using plot_functions)
# =========================
save_glmm_random_effects_plot(model, plots_output_dir, run_suffix)

# =========================
# 13. LARGE-RESIDUAL DIAGNOSTICS
# =========================
# Pearson residuals are computed on df_model rows only.
# Threshold: |pearson_resid| > 2 (roughly 2 SD from expectation under the model).

pearson_resid <- residuals(model, type = "pearson")

df_resid_diag <- df_model %>%
  mutate(
    fitted_prob     = fitted(model),
    pearson_resid   = pearson_resid,
    p_observed      = ifelse(n_trials > 0, y_bt / n_trials, NA_real_),
    abs_resid       = abs(pearson_resid)
  ) %>%
  filter(abs_resid > 2) %>%
  select(
    block,
    year_month_date,
    type,
    y_bt,
    n_trials,
    n_bt,
    omega,
    p_observed,
    fitted_prob,
    pearson_resid,
    abs_resid
  ) %>%
  arrange(desc(abs_resid))

resid_diag_file <- file.path(resid_output_dir, paste0("glmm_large_residuals_", run_suffix, ".csv"))
write_csv(df_resid_diag, resid_diag_file)

cat("\nLarge-residual rows (|Pearson resid| > 2):", nrow(df_resid_diag),
    "out of", nrow(df_model), "model rows\n")
cat("  Saved to:", resid_diag_file, "\n")

# How many times does each block appear in large residuals?
block_resid_summary <- df_resid_diag %>%
  group_by(block) %>%
  summarise(
    n_large_resid     = n(),
    mean_pearson      = mean(pearson_resid),
    max_pearson       = max(abs_resid),
    mean_p_observed   = mean(p_observed, na.rm = TRUE),
    mean_fitted       = mean(fitted_prob, na.rm = TRUE),
    pct_positive_bias = mean(pearson_resid > 0)  # 1 = always underestimated
  ) %>%
  arrange(desc(n_large_resid))

# Flag "chronic" blocks — appear in large residuals in >30% of their time points
total_timepoints <- df_model %>% count(block, name = "n_months")

block_resid_summary <- block_resid_summary %>%
  left_join(total_timepoints, by = "block") %>%
  mutate(pct_months_flagged = n_large_resid / n_months) %>%
  arrange(desc(pct_months_flagged))

cat("\nBlocks with large residuals in >30% of months:\n")
print(filter(block_resid_summary, pct_months_flagged > 0.30))


# Are large residuals concentrated in specific time periods?
temporal_resid_summary <- df_resid_diag %>%
  group_by(year_month_date) %>%
  summarise(
    n_large_resid  = n(),
    mean_pearson   = mean(pearson_resid),
    pct_positive   = mean(pearson_resid > 0)
  ) %>%
  arrange(desc(n_large_resid))

cat("\nMonths with most large residuals:\n")
print(head(temporal_resid_summary, 10))

# Compare omega distribution in large vs normal residuals
df_model %>%
  mutate(
    pearson_resid = pearson_resid,
    large_resid   = abs(pearson_resid) > 2
  ) %>%
  group_by(large_resid) %>%
  summarise(
    mean_omega      = mean(omega, na.rm = TRUE),
    median_omega    = median(omega, na.rm = TRUE),
    pct_reactive    = mean(omega > 0, na.rm = TRUE),
    mean_n_trials   = mean(n_trials),
    mean_p_observed = mean(ifelse(n_trials > 0, y_bt/n_trials, NA), 
                          na.rm = TRUE)
  )



  # Distribution of n_trials among large residual rows
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
# These plots use ALL model rows (not just large residuals) to diagnose
# whether the AR(1) term is absorbing genuine temporal signal or
# inflating due to block heterogeneity / unmeasured temporal confounding.

df_re_plot <- df_model %>%
  mutate(
    fitted_prob   = fitted(model),
    pearson_resid = pearson_resid  # reuse vector computed in section 13
  )

monthly_resid <- df_re_plot %>%
  group_by(year_month_date) %>%
  summarise(
    mean_resid    = mean(pearson_resid, na.rm = TRUE),
    median_resid  = median(pearson_resid, na.rm = TRUE),
    mean_fitted   = mean(fitted_prob, na.rm = TRUE),
    mean_observed = mean(ifelse(n_trials > 0, y_bt / n_trials, NA_real_),
                         na.rm = TRUE),
    n             = n(),
    .groups       = "drop"
  )

# Plot 1: Mean Pearson residual over time (all rows, not just flagged)
# A systematic pattern here means an unmeasured temporal signal the AR(1)
# is absorbing — which inflates its variance estimate.
p_resid_time <- ggplot(monthly_resid, aes(x = year_month_date)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  geom_line(aes(y = mean_resid), colour = "steelblue") +
  geom_point(aes(y = mean_resid, size = n), colour = "steelblue", show.legend = FALSE) +
  labs(
    title    = "Mean Pearson residual over time (all rows)",
    subtitle = "Systematic pattern = unmeasured temporal signal absorbed by AR(1)",
    x = "Month", y = "Mean Pearson residual"
  ) +
  theme_minimal()

ggsave(
  file.path(resid_output_dir, paste0("ar1_resid_over_time_", run_suffix, ".png")),
  p_resid_time, width = 10, height = 5, dpi = 150
)

# Plot 2: Observed vs fitted positivity rate over time (city-wide mean)
# A persistent gap between lines is residual temporal variance the model
# failed to explain — even after the AR(1).
p_obs_vs_fit <- ggplot(monthly_resid, aes(x = year_month_date)) +
  geom_line(aes(y = mean_observed, colour = "Observed")) +
  geom_line(aes(y = mean_fitted,   colour = "Fitted")) +
  geom_point(aes(y = mean_observed, colour = "Observed"), size = 1.5) +
  geom_point(aes(y = mean_fitted,   colour = "Fitted"),   size = 1.5) +
  scale_colour_manual(values = c("Observed" = "black", "Fitted" = "steelblue")) +
  labs(
    title    = "Observed vs fitted positivity rate over time",
    subtitle = "Persistent gap = temporal variance not captured by the model",
    x = "Month", y = "Mean positivity rate", colour = NULL
  ) +
  theme_minimal()

ggsave(
  file.path(resid_output_dir, paste0("ar1_obs_vs_fitted_time_", run_suffix, ".png")),
  p_obs_vs_fit, width = 10, height = 5, dpi = 150
)

# Plot 3: Fitted trajectories for 50 random blocks vs city-wide mean
# If the spread of block lines is large relative to the city mean, the
# AR(1) variance is driven by block heterogeneity rather than a shared
# temporal trend — consider a block-level AR(1) or random intercept.
block_monthly <- df_re_plot %>%
  group_by(block, year_month_date) %>%
  summarise(
    mean_fitted   = mean(fitted_prob, na.rm = TRUE),
    mean_observed = mean(ifelse(n_trials > 0, y_bt / n_trials, NA_real_),
                         na.rm = TRUE),
    .groups = "drop"
  )

set.seed(42)
sample_blocks <- sample(unique(block_monthly$block), min(50, n_distinct(block_monthly$block)))

p_block_traj <- ggplot(
  filter(block_monthly, block %in% sample_blocks),
  aes(x = year_month_date, y = mean_fitted, group = block)
) +
  geom_line(alpha = 0.2, colour = "steelblue") +
  geom_line(
    data      = monthly_resid,
    aes(x = year_month_date, y = mean_fitted, group = NULL),
    colour    = "red", linewidth = 1.2
  ) +
  labs(
    title    = "Fitted trajectories for 50 random blocks",
    subtitle = "Red = city-wide mean fitted. Spread = block heterogeneity in AR(1)",
    x = "Month", y = "Fitted positivity rate"
  ) +
  theme_minimal()

ggsave(
  file.path(resid_output_dir, paste0("ar1_block_trajectories_", run_suffix, ".png")),
  p_block_traj, width = 10, height = 6, dpi = 150
)

# Plot 4: ACF of city-wide mean Pearson residuals over time
# Directly tests whether the AR(1) removed temporal autocorrelation.
# Significant spikes at lag > 0 mean the AR(1) did not fully account
# for the temporal structure — the true order may be higher, or a
# global time trend is needed.
resid_ts <- monthly_resid %>%
  arrange(year_month_date) %>%
  pull(mean_resid)

acf_vals <- acf(resid_ts, lag.max = 24, plot = FALSE, na.action = na.pass)
acf_df   <- data.frame(
  lag  = as.numeric(acf_vals$lag),
  acf  = as.numeric(acf_vals$acf)
)
ci_bound <- qnorm(0.975) / sqrt(length(resid_ts))

p_acf <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_hline(yintercept = 0) +
  geom_hline(yintercept = c(-ci_bound, ci_bound),
             linetype = "dashed", colour = "blue") +
  geom_segment(aes(x = lag, xend = lag, y = 0, yend = acf)) +
  geom_point(size = 2) +
  labs(
    title    = "ACF of city-wide mean Pearson residuals",
    subtitle = "Significant spikes = AR(1) did not fully remove temporal autocorrelation",
    x = "Lag (months)", y = "Autocorrelation"
  ) +
  theme_minimal()

ggsave(
  file.path(resid_output_dir, paste0("ar1_acf_mean_resid_", run_suffix, ".png")),
  p_acf, width = 8, height = 5, dpi = 150
)

cat("\nAR(1) temporal diagnostics saved to:", resid_output_dir, "\n")