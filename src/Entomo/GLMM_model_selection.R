
library(glmmTMB)
library(tidyverse)
library(sf)

source("helper_functions.r")

if (!require("conflicted", quietly = TRUE)) {
  install.packages("conflicted"); library(conflicted)
}
conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("lag",    "dplyr")

# =============================================================================
# SETTINGS
# =============================================================================
cfg <- list(
  spatial_level = "CMF",

  # Random effects
  include_block_re     = TRUE,
  include_time_re      = FALSE,
  include_ar1_temporal = TRUE,
  ar1_group            = "block",
  include_spatial_ar   = FALSE,

  link_function = "logit",

  # Predictors
  lag_vars      = c("total_rainy_days", "avg_VPD", "precip_max_day_resid_on_trd", "hurricane_within_120km"),

  unlagged_vars = c("is_urban", "is_WUI", "is_WI","has_aljibes", "water_containers", "water_shortage", "pop_density", "landcover"),
  # unlagged_vars = c("is_urban", "is_WUI", "is_WI","has_aljibes", "water_containers"),
  numeric_vars  = c("total_rainy_days", "precip_max_day_resid_on_trd", "avg_VPD", "water_containers", "pop_density"),

  # Lag levels to drop after expansion (NULL = keep all)
  exclude_predictors = NULL,

  # Shapefile folder (used when include_spatial_ar = TRUE)
  shapefile_path = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo"
  } else {
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders"
  },
  spatial_crs = NA,

  data_file = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo/env_epi_entomo_data_per_CMF_2015_01_to_2019_12_noNDXI_noColinnearity.csv"
  } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/data/env_epi_entomo_data_per_CMF_2015_01_to_2019_12_noNDXI_noColinnearity.csv"
  },
  response_start = "2016_01",

  max_lag = 5,
  kappa   = 2,

  output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM/model_selection",

  iter_max = 1000,
  eval_max = 1500
)

# Derived spatial fields
cfg$block_col    <- if (cfg$spatial_level == "CMF") "cmf"      else "manzana"
cfg$sf_block_col <- if (cfg$spatial_level == "CMF") "Area_CMF" else "CODIGO_"
cfg$shapefile_file <- if (cfg$spatial_level == "CMF") {
  file.path(cfg$shapefile_path, "CMF", "Poligonos CMF Cienfuegos_28032025.shp")
} else {
  file.path(cfg$shapefile_path, "Manzanas_cleaned_05032026", "Mz_CMF_Correcto_2022026.shp")
}

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
date_suffix  <- format(Sys.Date(), "%Y%m%d")
run_suffix   <- date_suffix

time_ar_spec  <- ifelse(cfg$include_ar1_temporal, paste0("AR1-", cfg$ar1_group), "noAR1")
space_ar_spec <- ifelse(cfg$include_spatial_ar, "AR-EXP", "noAR")

model_spec <- paste0(
  "space-", ifelse(cfg$include_block_re, "RE", "noRE"),
  "_time-", ifelse(cfg$include_time_re,  "RE", "noRE"),
  "_time-", time_ar_spec,
  "_space-", space_ar_spec,
  "_lag", cfg$max_lag,
  "_k", cfg$kappa,
  "_betabinom-", cfg$link_function
)

predictor_spec <- paste0(
  "lag-",   paste(cfg$lag_vars,      collapse = "-"),
  "_unlag-", paste(cfg$unlagged_vars, collapse = "-"),
  if (!is.null(cfg$exclude_predictors) && length(cfg$exclude_predictors) > 0)
    paste0("_excl-", paste(cfg$exclude_predictors, collapse = "-"))
  else ""
)

sel_model_dir <- file.path(cfg$output_dir, predictor_spec, model_spec, run_suffix)
dir.create(sel_model_dir, recursive = TRUE, showWarnings = FALSE)
cat("Selection output:", sel_model_dir, "\n\n")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df <- load_base_data(cfg$data_file) %>%
  mutate(
    year_month = factor(year_month),
    block      = factor(.data[[cfg$block_col]]),
    n_bt       = Inspected_houses + cfg$kappa * cases,
    y_bt       = Houses_pos_IS,
    sin_annual = sin(2 * pi * as.integer(format(year_month_date, "%m")) / 12),
    cos_annual = cos(2 * pi * as.integer(format(year_month_date, "%m")) / 12),
    landcover  = factor(landcover),
    temp_cat   = factor(temp_cat),
    precip_cat = factor(precip_cat)
  ) %>%
  select(-any_of(cfg$block_col))

# Spatial coordinates (only needed when include_spatial_ar = TRUE)
if (cfg$include_spatial_ar) {
  if (!file.exists(cfg$shapefile_file))
    stop("Shapefile not found: ", cfg$shapefile_file)

  sf_blocks <- st_read(cfg$shapefile_file, quiet = TRUE)
  if (cfg$spatial_level == "CMF")
    sf_blocks <- sf_blocks %>% mutate(Area_CMF = paste(AS, CMF, sep = "_"))
  if (!cfg$sf_block_col %in% names(sf_blocks))
    stop("Block ID column not found: ", cfg$sf_block_col)

  pts <- suppressWarnings(st_point_on_surface(sf_blocks))
  if (isTRUE(sf::st_is_longlat(pts))) pts <- st_transform(pts, 3857)
  else if (!is.na(cfg$spatial_crs))   pts <- st_transform(pts, cfg$spatial_crs)

  xy <- st_coordinates(pts)
  coords_df <- sf_blocks %>%
    st_drop_geometry() %>%
    mutate(block = as.character(.data[[cfg$sf_block_col]]),
           x = as.numeric(xy[, 1]), y = as.numeric(xy[, 2])) %>%
    select(block, x, y) %>% distinct(block, .keep_all = TRUE)

  df <- df %>%
    mutate(block_chr = as.character(block)) %>%
    left_join(coords_df, by = c("block_chr" = "block")) %>%
    select(-block_chr) %>%
    mutate(x_sc = as.numeric(scale(x)), y_sc = as.numeric(scale(y)),
           xy = glmmTMB::numFactor(x_sc, y_sc), spatial = factor("all"))
}

# AR(1) ordered factor and grouping
time_levels <- df %>%
  distinct(year_month, year_month_date) %>%
  arrange(year_month_date) %>% pull(year_month) %>% unique()
df <- df %>%
  mutate(year_month_ar1 = factor(year_month, levels = time_levels, ordered = TRUE),
         ar1_group = if (cfg$ar1_group == "global") factor("all") else block)

# =============================================================================
# 2. STANDARDIZE
# =============================================================================
vars_to_std <- intersect(cfg$numeric_vars, names(df))
df[, vars_to_std] <- standardize_matrix(as.matrix(df[, vars_to_std]))

# =============================================================================
# 3. CREATE DISTRIBUTED LAGS
# =============================================================================
L <- cfg$max_lag
for (var in cfg$lag_vars) {
  is_factor_var <- is.factor(df[[var]]) | is.character(df[[var]])
  for (l in 0:L) {
    lag_col <- paste0(var, "_lag", l)
    df <- df %>%
      group_by(block) %>%
      arrange(year_month_date, .by_group = TRUE) %>%
      mutate(!!lag_col := lag(.data[[var]], n = l,
               default = if (is_factor_var) NA_character_ else NA_real_)) %>%
      ungroup()
  }
}
lagged_cols <- unlist(lapply(cfg$lag_vars, function(v) paste0(v, "_lag", 0:L)))

# Drop lead-in rows
response_date <- as.Date(paste0(cfg$response_start, "_01"), "%Y_%m_%d")
df <- df %>% filter(year_month_date >= response_date)

# =============================================================================
# 4. ADD REACTIVE SHIFT COVARIATE
# =============================================================================
df <- df %>%
  mutate(
    n_trials       = n_bt,
    reactive_shift = log1p(cases)
  )

# =============================================================================
# 5. BUILD FORMULA
# =============================================================================
fixed_effects <- c(lagged_cols, cfg$unlagged_vars, "reactive_shift")
if (!is.null(cfg$exclude_predictors))
  fixed_effects <- setdiff(fixed_effects, cfg$exclude_predictors)

random_effects <- c(
  if (cfg$include_block_re)     "(1 | block)",
  if (cfg$include_time_re)      "(1 | year_month)",
  if (cfg$include_ar1_temporal) "ar1(year_month_ar1 + 0 | ar1_group)",
  if (cfg$include_spatial_ar)   "exp(xy + 0 | spatial)"
)

formula_str <- paste(
  "cbind(y_bt, n_trials - y_bt) ~",
  paste(fixed_effects, collapse = " + "),
  if (length(random_effects) > 0) paste("+", paste(random_effects, collapse = " + ")) else ""
)
formula <- as.formula(formula_str)

required_cols <- unique(c(
  "y_bt", "n_trials", fixed_effects,
  if (cfg$include_block_re)     "block",
  if (cfg$include_time_re)      "year_month",
  if (cfg$include_ar1_temporal) c("year_month_ar1", "ar1_group"),
  if (cfg$include_spatial_ar)   c("x_sc", "y_sc", "xy", "spatial")
))
keep_rows <- complete.cases(df[, required_cols, drop = FALSE])
df_model  <- df[keep_rows, , drop = FALSE]

cat("Formula:", formula_str, "\n")
cat("Observations:", nrow(df_model), "\n\n")

# Hook: set .selection_prep_only <- TRUE before source() to stop here
if (exists(".selection_prep_only") && isTRUE(.selection_prep_only))
  stop(".selection_prep_only")

# =============================================================================
# 6. FULL MODEL
# =============================================================================
cat("Fitting full model...\n")
model_full <- glmmTMB(
  formula,
  family  = glmmTMB::betabinomial(link = cfg$link_function),
  data    = df_model,
  control = glmmTMBControl(optCtrl = list(
    iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 0))
)
full_aic <- AIC(model_full)
cat("Full model AIC:", round(full_aic, 2), "\n\n")

summary_output_full <- local({
  op <- options(max.print = 99999); on.exit(options(op))
  capture.output(summary(model_full))
})
writeLines(c(paste0("Formula: ", formula_str), "", summary_output_full),
           file.path(sel_model_dir, "model_summary_full.txt"))

# =============================================================================
# 7. LEAVE-ONE-OUT SELECTION LOOP
# =============================================================================
candidates <- setdiff(fixed_effects, "reactive_shift")
re_suffix  <- if (length(random_effects) > 0)
  paste("+", paste(random_effects, collapse = " + ")) else ""

results <- tibble(predictor = character(), AIC = numeric(),
                  delta_AIC = numeric(), converged = logical())

for (pred in candidates) {
  preds_minus       <- setdiff(fixed_effects, pred)
  formula_minus_str <- paste(
    "cbind(y_bt, n_trials - y_bt) ~",
    paste(preds_minus, collapse = " + "), re_suffix)

  cat(sprintf("Removing %-45s ... ", pred))

  model_minus <- tryCatch(
    glmmTMB(as.formula(formula_minus_str),
            family  = glmmTMB::betabinomial(link = cfg$link_function),
            data    = df_model,
            control = glmmTMBControl(optCtrl = list(
              iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 0))),
    error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL }
  )

  if (is.null(model_minus)) {
    results <- add_row(results, predictor = pred, AIC = NA_real_,
                       delta_AIC = NA_real_, converged = FALSE)
    next
  }

  aic_val   <- AIC(model_minus)
  delta     <- aic_val - full_aic
  converged <- model_minus$fit$convergence == 0
  cat(sprintf("AIC = %9.2f  ΔAIC = %+8.2f  %s\n", aic_val, delta,
              if (!converged) "(convergence warning)" else ""))

  results <- add_row(results, predictor = pred, AIC = aic_val,
                     delta_AIC = delta, converged = converged)

  write_csv(results %>% arrange(delta_AIC),
            file.path(sel_model_dir, "model_selection_AIC.csv"))

  summ_out <- local({
    op <- options(max.print = 99999); on.exit(options(op))
    capture.output(summary(model_minus))
  })
  writeLines(c(paste0("Formula: ", formula_minus_str), "", summ_out),
             file.path(sel_model_dir, paste0("summary_removed_", pred, ".txt")))
}

# =============================================================================
# 8. SUMMARY
# =============================================================================
results <- results %>% arrange(delta_AIC)
cat("\n=== Model selection complete ===\n")
cat(sprintf("Full model AIC: %.2f\n\n", full_aic))
print(results, n = nrow(results))
cat("\nΔAIC > 0: removing worsens fit (keep)  |  ΔAIC < 0: removing improves fit (drop)  |  |ΔAIC| < 2: negligible\n\n")

write_csv(results %>% arrange(delta_AIC),
          file.path(sel_model_dir, "model_selection_AIC.csv"))
cat("Results saved to:", file.path(sel_model_dir, "model_selection_AIC.csv"), "\n")
