# GLMM_model_selection.R
# Model selection for predictors in GLMM with spatial and temporal autocorrelation

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

# SETTINGS (same as GLMM.r)
cfg <- list(
  include_block_re = FALSE,
  include_time_re = TRUE,
  include_ar1_temporal = FALSE,
  ar1_group = "block",
  include_spatial_ar = TRUE,
  shapefile_path = "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp",
  sf_block_col = "CODIGO_",
  spatial_crs = NA,
  data_file = "/home/rita/PyProjects/DI-MOB-BionamiX/data/env_epi_entomo_data_per_manzana_2016_01_to_2019_12_noColinnearity.csv",
  max_lag = 2,
  kappa = 2,
  output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM_selection",
  iter_max = 1e4,
  eval_max = 1e4
)

# Load data
if (!dir.exists(cfg$output_dir)) dir.create(cfg$output_dir, recursive = TRUE)
df <- read_csv(cfg$data_file)
df <- df %>%
  mutate(
    year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d"),
    year_month = factor(year_month),
    block = factor(manzana),
    n_bt = Inspected_houses + cfg$kappa * cases,
    y_bt = Houses_pos_IS
  ) %>%
  select(-c(CMF, CP, AREA))

# -------------------------------------------------------------------
# Subset to first 100 blocks for test run
set.seed(42)  # for reproducibility
unique_blocks <- unique(df$block)
blocks_to_keep <- unique_blocks[1:min(100, length(unique_blocks))]
df <- df %>% filter(block %in% blocks_to_keep)
cat("Subset to", n_distinct(df$block), "blocks for test run\n")
# -------------------------------------------------------------------

# Add spatial coordinates
if (cfg$include_spatial_ar) {
  sf_blocks <- st_read(cfg$shapefile_path, quiet = TRUE)
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
      x = as.numeric(xy[, 1]),
      y = as.numeric(xy[, 2])
    ) %>%
    select(block, x, y) %>%
    distinct(block, .keep_all = TRUE)
  df <- df %>%
    mutate(block_chr = as.character(block)) %>%
    left_join(coords_df, by = c("block_chr" = "block")) %>%
    select(-block_chr)
  df <- df %>%
    mutate(
      x_sc = as.numeric(scale(x)),
      y_sc = as.numeric(scale(y)),
      xy = glmmTMB::numFactor(x_sc, y_sc)
    )
  df <- df %>% mutate(spatial = factor("all"))
}

# Standardize numeric covariates
lag_vars <- c("avg_temp",  "rel_hum", "total_precip", "mean_ndvi", "precip_max_day_resid")
unlagged_vars <- c("is_urban", "has_aljibes", "nr_aljibes", "is_WI", "is_WUI", "water_shortage", "water_containers")
numeric_vars <- c(lag_vars, "nr_aljibes", "water_containers", "WS2M")
for (var in numeric_vars) {
  if (var %in% names(df)) {
    df[[var]] <- (df[[var]] - mean(df[[var]], na.rm = TRUE)) / sd(df[[var]], na.rm = TRUE)
  }
}

# Create distributed lags
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

# Expand data for reactive mixture
# ...same as GLMM.r...
df_with_omega <- df %>%
  mutate(
    omega = ifelse(cases > 0, pmin(cfg$kappa * cases / n_bt, 1), 0),
    has_cases = cases > 0
  )
df_baseline <- df_with_omega %>%
  mutate(
    n_trials = floor((1 - omega) * n_bt),
    y_bt_adj = floor(y_bt * (1 - omega)),
    reactive_shift = 0,
    type = "baseline"
  ) %>%
  filter(n_trials > 0)
df_reactive <- df_with_omega %>%
  filter(has_cases) %>%
  mutate(
    n_trials = floor(omega * n_bt),
    y_bt_adj = floor(y_bt * omega),
    reactive_shift = log1p(cases),
    type = "reactive"
  ) %>%
  filter(n_trials > 0)
df_expanded <- bind_rows(df_baseline, df_reactive) %>%
  select(-omega, -has_cases, -y_bt) %>%
  rename(y_bt = y_bt_adj) %>%
  arrange(block, year_month_date, type) %>%
  mutate(y_bt = pmin(y_bt, n_trials))

# Model selection setup
all_predictors <- c(lagged_cols, unlagged_vars, "reactive_shift")

# Initial model with all predictors
formula_str <- paste(
  "cbind(y_bt, n_trials - y_bt) ~",
  paste(all_predictors, collapse = " + ")
)
if (cfg$include_time_re) {
  formula_str <- paste(formula_str, "+ (1 | year_month)")
}
if (cfg$include_spatial_ar) {
  formula_str <- paste(formula_str, "+ exp(xy + 0 | spatial)")
}
formula <- as.formula(formula_str)

required_cols <- unique(c(
  "y_bt", "n_trials", all_predictors,
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

# Stepwise backward selection
# Remove one predictor at a time, refit, and compare AIC
results <- tibble(predictor = character(), AIC = numeric())
for (pred in all_predictors) {
  predictors_minus <- setdiff(all_predictors, pred)
  formula_str_minus <- paste(
    "cbind(y_bt, n_trials - y_bt) ~",
    paste(predictors_minus, collapse = " + ")
  )
  if (cfg$include_time_re) {
    formula_str_minus <- paste(formula_str_minus, "+ (1 | year_month)")
  }
  if (cfg$include_spatial_ar) {
    formula_str_minus <- paste(formula_str_minus, "+ exp(xy + 0 | spatial)")
  }
  formula_minus <- as.formula(formula_str_minus)
  model_minus <- glmmTMB(
    formula_minus,
    family = binomial(link = "logit"),
    data = df_model,
    control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 6))
  )
  results <- results %>% add_row(predictor = pred, AIC = AIC(model_minus))
    cat("Removed", pred, ": AIC =", AIC(model_minus), "\n")
    # Save results after each fit
    write_csv(results, file.path(cfg$output_dir, "model_selection_AIC.csv"))
}

# Save results
write_csv(results, file.path(cfg$output_dir, "model_selection_AIC.csv"))
cat("\nModel selection results saved to model_selection_AIC.csv\n")

# Optionally, select predictors with lowest AIC and refit final model
best_predictors <- setdiff(all_predictors, results %>% filter(AIC == min(AIC)) %>% pull(predictor))
final_formula_str <- paste(
  "cbind(y_bt, n_trials - y_bt) ~",
  paste(best_predictors, collapse = " + ")
)
if (cfg$include_time_re) {
  final_formula_str <- paste(final_formula_str, "+ (1 | year_month)")
}
if (cfg$include_spatial_ar) {
  final_formula_str <- paste(final_formula_str, "+ exp(xy + 0 | spatial)")
}
final_formula <- as.formula(final_formula_str)
final_model <- glmmTMB(
  final_formula,
  family = binomial(link = "logit"),
  data = df_model,
  control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 6))
)
cat("\nFinal model formula:", final_formula_str, "\n")
cat("Final model AIC:", AIC(final_model), "\n")

saveRDS(final_model, file.path(cfg$output_dir, "glmm_final_model.rds"))
cat("Final model saved to glmm_final_model.rds\n")
