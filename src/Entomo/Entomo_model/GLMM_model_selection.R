library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(tibble)
library(stringr)
library(forcats)
library(ggplot2)
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
  shapefile_path = if (Sys.info()["nodename"] == "frietjes") {
    "data/Entomo/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
  } else {
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp"
    },
  
  sf_block_col = "CODIGO_",
  spatial_crs = NA,             # Optional projected CRS (e.g., 32719). NA = keep CRS unless lon/lat (then use EPSG:3857)
  
  # Data
  data_file = if (Sys.info()["nodename"] == "frietjes") {
    "home/rita/data/Entomo/env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv"
  } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/data/env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv"
  },
  
  # Lag settings
  max_lag = 2,
  kappa = 2,  # multiplier for cases in n_bt calculation

  # Output
  output_dir = if (Sys.info()["nodename"] == "frietjes") {
    "/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM_selection"
    } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM_selection"
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
dir.create(model_output_dir, recursive = TRUE, showWarnings = FALSE)

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
# lag_vars <- c("avg_temp",  "total_precip", "rel_hum", "mean_ndvi", "precip_max_day_resid") # ndmi and ndwi removed due to collinearity
# unlagged_vars <- c("is_urban", "has_aljibes", "nr_aljibes", "is_WI", "is_WUI", "water_shortage", "water_containers", "WS2M")

lag_vars <- c("avg_temp", "total_precip",  "mean_ndvi", "precip_max_day_resid") # RH, WS, ndmi and ndwi removed due to collinearity
unlagged_vars <- c("is_urban", "has_aljibes", "nr_aljibes", "is_WI", "is_WUI", "water_shortage", "water_containers")

# Numeric variables to standardize (z-score)
numeric_vars <- c(lag_vars, "nr_aljibes", "water_containers")

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
  random_effects <- c(random_effects, "exp(xy + 0 | spatial, range = 400)")
  # random_effects <- c(random_effects, "mat(xy + 0 | spatial, range = 400)")
}

if (length(random_effects) > 0) {
  formula_str <- paste(formula_str, "+", paste(random_effects, collapse = " + "))
}

cat("GLMM Formula:\n")
cat(formula_str, "\n\n")

formula <- as.formula(formula_str)

required_cols <- unique(c(
  "y_bt", "n_trials", fixed_effects,
  if (cfg$include_time_re) "year_month" else NULL,
  if (cfg$include_spatial_ar) c("x_sc", "y_sc", "xy", "spatial") else NULL
))
keep_rows <- complete.cases(df_expanded[, required_cols, drop = FALSE])
df_model <- df_expanded[keep_rows, , drop = FALSE]

cat("\nStarting model selection...\n")
cat("Initial formula: ", formula_str, "\n")

# Fit initial model
model_full <- glmmTMB(
  formula,
  family = binomial(link = "logit"),
  data = df_model,
  control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 10))
)
cat("Full model AIC:", AIC(model_full), "\n")

results <- tibble(predictor = character(), AIC = numeric())

for (pred in fixed_effects) {
  predictors_minus <- setdiff(fixed_effects, pred)
  formula_str_minus <- paste(
    "cbind(y_bt, n_trials - y_bt) ~",
    paste(predictors_minus, collapse = " + ")
  )
  if (cfg$include_time_re) {
    formula_str_minus <- paste(formula_str_minus, "+ (1 | year_month)")
  }
  if (cfg$include_spatial_ar) {
    formula_str_minus <- paste(formula_str_minus, "+ exp(xy + 0 | spatial, range = 400)")
  }
  formula_minus <- as.formula(formula_str_minus)
  model_minus <- glmmTMB(
    formula_minus,
    family = binomial(link = "logit"),
    data = df_model,
    control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 10))
  )
  results <- results %>% add_row(predictor = pred, AIC = AIC(model_minus))
  cat("Removed", pred, ": AIC =", AIC(model_minus), "\n")
  # Save results after each fit
  write_csv(results, file.path(model_output_dir, "model_selection_AIC.csv"))
  # Save model summary for each evaluated model
  summary_output <- capture.output(summary(model_minus))
  summary_file <- file.path(model_output_dir, paste0("model_summary_removed_", pred, ".txt"))
  writeLines(summary_output, summary_file)
}

# Save results
write_csv(results, file.path(model_output_dir, "model_selection_AIC.csv"))
cat("\nModel selection results saved to model_selection_AIC.csv\n")