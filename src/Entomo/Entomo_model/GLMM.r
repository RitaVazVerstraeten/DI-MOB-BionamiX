library(tidyverse)
library(glmmTMB)
library(slider)
library(sf)

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
  include_ar1_temporal = FALSE, # AR(1) temporal autocorrelation (within group)
  ar1_group = "block",         # "block" (within-block AR1) or "global"
  include_spatial_ar = TRUE,  # Exponential spatial autocorrelation: exp(xy + 0 | spatial)
  # include_spatial_ar = TRUE,  # Matérn spatial autocorrelation: mat(xy + 0 | spatial)

  # Spatial coordinates from shapefile (used when include_spatial_ar = TRUE)
  shapefile_path = "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp",
  sf_block_col = "CODIGO_",
  spatial_crs = NA,             # Optional projected CRS (e.g., 32719). NA = keep CRS unless lon/lat (then use EPSG:3857)
  
  # Data
  data_file = "/home/rita/PyProjects/DI-MOB-BionamiX/data/env_epi_entomo_data_per_manzana_2016_01_to_2019_12.csv",
  
  # Lag settings
  max_lag = 1,
  kappa = 2,  # multiplier for cases in n_bt calculation

  # Output
  output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM",
  
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

# Run suffix is now just the date (model spec is in parent folder)
run_suffix <- date_suffix

# Output structure:
# <output_dir>/<model_spec>/<run_suffix>/
time_ar_spec <- ifelse(cfg$include_ar1_temporal, paste0("AR1-", cfg$ar1_group), "noAR1")
space_ar_spec <- ifelse(cfg$include_spatial_ar, "AR-EXP", "noAR")
# space_ar_spec <- ifelse(cfg$include_spatial_ar, "AR-Mat", "noAR")

model_spec <- paste0(
  "space-", ifelse(cfg$include_block_re, "RE", "noRE"),
  "_time-", ifelse(cfg$include_time_re, "RE", "noRE"),
  "_time-", time_ar_spec,
  "_space-", space_ar_spec,
  "_lag", cfg$max_lag,
  "_k", cfg$kappa
)

model_output_dir <- file.path(cfg$output_dir, model_spec)
run_output_dir <- file.path(model_output_dir, run_suffix)
dir.create(run_output_dir, recursive = TRUE, showWarnings = FALSE)

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
    y_bt = Houses_pos_IS
  ) %>%
  select(-c(CMF, CP, AREA))

# Optional: add spatial coordinates by matching block names to shapefile IDs
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


# =========================
# OPTIONAL: SUBSET TO FIRST 100 BLOCKS (for testing)
# =========================
# Uncomment to enable faster testing
set.seed(42)  # for reproducibility
unique_blocks <- unique(df$block)
blocks_to_keep <- sample(unique_blocks[1:min(100, length(unique_blocks))], size = min(100, length(unique_blocks)))
df <- df %>% filter(block %in% blocks_to_keep)
cat("Subset to", n_distinct(df$block), "blocks\n")


# =========================
# 2. STANDARDIZE NUMERIC COVARIATES
# =========================
lag_vars <- c("avg_temp", "rel_hum", "total_precip", "mean_ndvi", "precip_max_day") # ndmi and ndwi removed due to collinearity
unlagged_vars <- c("WS2M", "is_urban", "has_aljibes", "nr_aljibes", "is_WI", "is_WUI", "water_shortage", "water_containers")

# Numeric variables to standardize (z-score)
numeric_vars <- c(lag_vars, "WS2M", "nr_aljibes", "water_containers")

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
  for (l in 0:L) {
    lag_col <- paste0(var, "_lag", l)
    df <- df %>%
      group_by(block) %>%
      arrange(year_month_date, .by_group = TRUE) %>%
      mutate(!!lag_col := lag(.data[[var]], n = l, default = 0)) %>%
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
  select(-omega, -has_cases, -y_bt) %>%
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
# Fixed effects
fixed_effects <- c(lagged_cols, unlagged_vars, "reactive_shift")
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
  random_effects <- c(random_effects, "exp(xy + 0 | spatial)")
  # random_effects <- c(random_effects, "mat(xy + 0 | spatial)")
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
required_cols <- unique(c(
  "y_bt", "n_trials", fixed_effects,
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

cat("\nStarting model fit...\n")
cat("Formula: ", formula_str, "\n")
cat("Observations: ", nrow(df_model), "\n\n")

model <- glmmTMB(
  formula,
  family = binomial(link = "logit"),
  data = df_model,
  control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 6))
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
preds <- predict(model, type = "response")
cat("Predictions length:", length(preds), "\n")

# Add predictions back to full df_expanded safely
if (length(preds) == nrow(df_model)) {
  df_expanded <- df_expanded %>%
    mutate(fitted_prob = NA_real_)
  df_expanded$fitted_prob[keep_rows] <- preds
} else {
  stop("Mismatch: ", nrow(df_model), " model rows vs ", length(preds), " predictions")
}

# =========================
# 9. AGGREGATE BACK TO ORIGINAL OBSERVATION LEVEL
# =========================
# For each original observation, get baseline p_bt and reactive p_R predictions 
df_summary <- df_expanded %>%
  select(block, year_month_date, type, fitted_prob) %>%
  pivot_wider(
    names_from = type,
    values_from = fitted_prob,
    values_fill = NA
  ) %>%
  rename(fitted_prob_baseline = baseline, fitted_prob_reactive = reactive)

# Save data outputs for later checking/calling
expanded_file <- file.path(run_output_dir, paste0("glmm_expanded_predictions_", run_suffix, ".csv"))
summary_pred_file <- file.path(run_output_dir, paste0("glmm_summary_predictions_", run_suffix, ".csv"))
cfg_file <- file.path(run_output_dir, paste0("glmm_config_", run_suffix, ".rds"))

write_csv(df_expanded, expanded_file)
write_csv(df_summary, summary_pred_file)
saveRDS(cfg, cfg_file)

cat("\nSaved outputs:\n")
cat("  Model spec folder: ", model_output_dir, "\n", sep = "")
cat("  Run folder: ", run_output_dir, "\n", sep = "")
cat("  Model RDS: ", model_file, "\n", sep = "")
cat("  Summary TXT: ", summary_file, "\n", sep = "")
cat("  Fixed effects OR CSV: ", coef_table_file, "\n", sep = "")
cat("  Expanded predictions CSV: ", expanded_file, "\n", sep = "")
cat("  Aggregated predictions CSV: ", summary_pred_file, "\n", sep = "")
cat("  Config RDS: ", cfg_file, "\n", sep = "")

# =========================
# 10. PLOT p_bt_fitted, p_R_fitted, p_bt_observed
# =========================
df_observed <- df %>%
  transmute(
    block,
    year_month_date,
    p_bt_observed = ifelse(n_bt > 0, y_bt / n_bt, NA_real_),
    cases
  )

df_plot <- df_summary %>%
  rename(
    p_bt_fitted = fitted_prob_baseline,
    p_R_fitted = fitted_prob_reactive
  ) %>%
  left_join(df_observed, by = c("block", "year_month_date"))

df_plot_ts <- df_plot %>%
  group_by(year_month_date) %>%
  summarise(
    p_bt_fitted = mean(p_bt_fitted, na.rm = TRUE),
    p_R_fitted = mean(p_R_fitted, na.rm = TRUE),
    p_bt_observed = mean(p_bt_observed, na.rm = TRUE),
    cases = sum(cases, na.rm = TRUE),
    .groups = "drop"
  )

df_plot_long <- df_plot_ts %>%
  pivot_longer(
    cols = c(p_bt_fitted, p_R_fitted, p_bt_observed),
    names_to = "series",
    values_to = "probability"
  )

max_prob <- max(df_plot_long$probability, na.rm = TRUE)
max_cases <- max(df_plot_ts$cases, na.rm = TRUE)
scale_factor <- ifelse(is.finite(max_cases) && max_cases > 0, max_prob / max_cases, 1)

p_probs <- ggplot(df_plot_long, aes(x = year_month_date, y = probability, color = series)) +
  geom_col(
    data = df_plot_ts,
    aes(x = year_month_date, y = cases * scale_factor),
    inherit.aes = FALSE,
    fill = "grey75",
    alpha = 0.5,
    width = 25
  ) +
  geom_line(linewidth = 1) +
  geom_point(size = 1.3) +
  scale_color_manual(
    values = c(
      p_bt_fitted = "#1f77b4",
      p_R_fitted = "#ff7f0e",
      p_bt_observed = "#d62728"
    )
  ) +
  scale_y_continuous(
    name = "Probability",
    sec.axis = sec_axis(~ . / scale_factor, name = "Cases")
  ) +
  labs(
    x = "Time",
    color = NULL,
    title = "Observed vs Fitted Probabilities with Cases",
    subtitle = "Lines: mean probabilities across blocks | Bars: total cases"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

plot_file <- file.path(run_output_dir, paste0("probabilities_timeseries_", run_suffix, ".png"))
ggsave(plot_file, p_probs, width = 12, height = 6, dpi = 150)
cat("  Probability plot PNG: ", plot_file, "\n", sep = "")