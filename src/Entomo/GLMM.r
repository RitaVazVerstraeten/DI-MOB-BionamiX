
library(tidyverse)
library(glmmTMB)
library(slider)
library(sf)

# Source plot functions for GLMM diagnostics
source("plot_functions.r")
source("helper_functions.r")
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
  # Spatial resolution: "CMF" or "manzana"
  spatial_level = "CMF",

  # Random effects to include
  include_block_re     = TRUE,  # Random intercept for block (spatial)
  include_time_re      = FALSE,   # Random intercept for time (temporal)
  include_ar1_temporal = TRUE,  # AR(1) temporal autocorrelation (within group)
  ar1_group            = "block", # "block" (within-block AR1) or "global"
  include_spatial_ar   = FALSE,   # Exponential spatial autocorrelation: exp(xy + 0 | spatial)
  # include_spatial_ar = TRUE,   # Matérn spatial autocorrelation: mat(xy + 0 | spatial)

  # Link function for beta-binomial GLMM
  link_function = "logit",       # Options: "logit", "probit", "cloglog"

  # Predictors
  # lag_vars     : variables for which distributed lags are created
  # unlagged_vars: variables entered directly without lag
  # numeric_vars : continuous variables to z-score standardize
  #                (exclude factors, binary 0/1, and already-factored variables)
  lag_vars      = c("total_rainy_days", "avg_VPD", "precip_max_day_resid_on_trd", "hurricane_within_120km"),

  unlagged_vars = c("is_urban", "is_WUI", "is_WI","has_aljibes", "water_containers", "water_shortage", "pop_density", "landcover"),
  # unlagged_vars = c("is_urban", "is_WUI", "is_WI","has_aljibes", "water_containers"),
  numeric_vars  = c("total_rainy_days", "precip_max_day_resid_on_trd", "avg_VPD", "water_containers", "pop_density"),

  # Interaction terms (NULL = none)
  # Each element is a character vector of exactly 2 variable names (column names
  # after lag expansion, e.g. "temp_cat_lag1", or unlagged names e.g. "is_urban")
  # Example: interactions = list(c("temp_cat_lag1", "avg_VPD_lag1"), c("is_urban", "water_containers"))
  interactions  = NULL,

  # Predictors to drop after lag expansion (NULL = keep all)
  # Use the fully expanded column name, e.g. "avg_VPD_lag1", "is_urban"
  # exclude_predictors = c("total_rainy_days_lag1", "total_rainy_days_lag0"),
  # exclude_predictors = c("mean_ndvi_lag0", "mean_ndvi_lag1", "total_rainy_dayslag1", "total_rainy_days_lag0"),

  # Add sin/cos annual Fourier terms as fixed effects (2-parameter seasonal cycle).
  # Tests whether residual seasonality is independent of the climate covariates.
  include_fourier = FALSE,

  # Spatial coordinates from shapefile (used when include_spatial_ar = TRUE).
  # Point to the folder containing both the manzana and CMF shapefiles;
  # the correct file is selected automatically based on spatial_level.
  shapefile_path = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo"
  } else {
    "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders"
  },
  spatial_crs = NA,             # Optional projected CRS. NA = keep CRS unless lon/lat (then use EPSG:3857)

  # Data: extended-lag dataset (env 2015–2019, ento-epi 2016–2019).
  # response_start marks the start of the ento response period; 2015 rows
  # provide lag lead-in history and are dropped after distributed lags are built.
  data_file = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/data/Entomo/env_epi_entomo_data_per_CMF_2015_01_to_2019_12_noNDXI_noColinnearity.csv"
  } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/data/env_epi_entomo_data_per_CMF_2015_01_to_2019_12_noNDXI_noColinnearity.csv"
  },
  response_start = "2016_01",   # rows before this date are lag lead-in only

  # Lag settings
  max_lag = 5,
  kappa = 2,  # multiplier for cases in n_bt calculation

  # Output
  output_dir = if (Sys.info()["nodename"] == "frietjes") {
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
    } else {
    "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM"
    },
  # GLMM control
  iter_max = 1000,
  eval_max = 1500
)

# Derived from spatial_level — keep consistent with Hierarch_StateSpace model
cfg$block_col    <- if (cfg$spatial_level == "CMF") "cmf"      else "manzana"
cfg$sf_block_col <- if (cfg$spatial_level == "CMF") "Area_CMF" else "CODIGO_"
cfg$shapefile_file <- if (cfg$spatial_level == "CMF") {
  file.path(cfg$shapefile_path, "CMF", "Poligonos CMF Cienfuegos_28032025.shp")
} else {
  file.path(cfg$shapefile_path, "Manzanas_cleaned_05032026", "Mz_CMF_Correcto_2022026.shp")
}

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
  "_betabinom-", cfg$link_function
)

# Encode predictor sets in folder name so each combination gets its own directory
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
df <- load_base_data(cfg$data_file) %>%
  mutate(
    year_month    = factor(year_month),
    block         = factor(.data[[cfg$block_col]]),
    n_bt          = Inspected_houses + cfg$kappa * cases,
    y_bt          = Houses_pos_IS,
    sin_annual    = sin(2 * pi * as.integer(format(year_month_date, "%m")) / 12),
    cos_annual    = cos(2 * pi * as.integer(format(year_month_date, "%m")) / 12),
    landcover     = factor(landcover),
    temp_cat      = factor(temp_cat),
    precip_cat    = factor(precip_cat)
  ) %>%
  select(-any_of(cfg$block_col))

# Optional: add spatial coordinates by matching block names to shapefile IDs

if (cfg$include_spatial_ar) {
  if (!file.exists(cfg$shapefile_file)) {
    stop("Spatial autocorrelation requested but shapefile not found: ", cfg$shapefile_file)
  }

  sf_blocks <- st_read(cfg$shapefile_file, quiet = TRUE)
  if (cfg$spatial_level == "CMF") {
    sf_blocks <- sf_blocks %>% mutate(Area_CMF = paste(AS, CMF, sep = "_"))
  }
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
  cat("  ", cfg$shapefile_file, "\n", sep = "")
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

vars_to_std <- intersect(numeric_vars, names(df))
df[, vars_to_std] <- standardize_matrix(as.matrix(df[, vars_to_std]))

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

# Drop lead-in rows: 2015 rows existed only to provide lag history for early 2016 rows.
# response_start is the first month that should appear as a model observation.
response_date <- as.Date(paste0(cfg$response_start, "_01"), "%Y_%m_%d")
df <- df %>% filter(year_month_date >= response_date)

# Check for NAs in lagged columns — now on response-period rows only
cat("\n=== CHECKING LAGGED COLUMNS FOR NAs ===\n")
cat("Response rows (>= ", cfg$response_start, "):", nrow(df), "\n")
for (col in lagged_cols) {
  na_count <- sum(is.na(df[[col]]))
  na_pct <- round(100 * na_count / nrow(df), 2)
  cat(sprintf("%s: %d NAs (%.2f%%)\n", col, na_count, na_pct))
}
cat("=== End lag NA check ===\n\n")

# =========================
# 4. ADD REACTIVE SHIFT COVARIATE
# =========================
# Use the full n_bt and y_bt as-is; reactive_shift = log1p(cases) lets the model
# estimate the direction of surveillance bias directly from the data.
# Positive coefficient → targeting bias dominates (more positives found reactively).
# Negative coefficient → denominator inflation dominates (n_bt inflated, diluting p_bt).
df <- df %>%
  mutate(
    n_trials       = n_bt,
    reactive_shift = log1p(cases)
  )

cat("Observations after data prep:", nrow(df), "rows,",
    n_distinct(df$block), "blocks,",
    n_distinct(df$year_month_date), "months\n")

# =========================
# 5. BUILD GLMM FORMULA
# =========================
# Interaction terms (added as var1:var2 — main effects already listed above)
interaction_terms <- c()
if (!is.null(cfg$interactions) && length(cfg$interactions) > 0) {
  for (pair in cfg$interactions) {
    if (length(pair) != 2) stop("Each interaction must be exactly 2 variable names, got: ", paste(pair, collapse = ", "))
    missing_vars <- setdiff(pair, names(df))
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
#With ar1_group = "block", each block gets its own independent AR1 process. If set to "global", all observations share a single AR1 process across all blocks.
# the AR1 covariance function the corelation btw time points t and t+k is Corr(u_t, u_{t+k}) = rho^k, where rho is the autocorr coeff estimated from the data 

# For each block, a separate time series of latent values u_t is estimated — one per month. Consecutive months within a block are correlated by ρ, two months apart by ρ², etc. This captures the idea that mosquito occurrence in one month tends to persist into the next, independently within each block.

if (cfg$include_spatial_ar) {
  random_effects <- c(random_effects, "exp(xy + 0 | spatial)")
  # random_effects <- c(random_effects, "mat(xy + 0 | spatial, range = 400)")
}

# The right hand side of the bar splits the above specification independently among groups. Each group has its own separate u vector but shares the same parameters for the covariance structure.

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

na_counts <- sort(sapply(required_cols, function(col) sum(is.na(df_expanded[[col]]))), decreasing = TRUE)
na_counts <- na_counts[na_counts > 0]
if (length(na_counts) > 0) {
  cat("NA counts in required columns:\n")
  print(na_counts)
} else {
  cat("No NAs found in required columns.\n")
}

keep_rows <- complete.cases(df[, required_cols, drop = FALSE])

cat("Rows in df:", nrow(df), "\n")
cat("Rows used for model fit (complete cases):", sum(keep_rows), "\n")
cat("Rows excluded due to NA in model terms:", sum(!keep_rows), "\n")

if (sum(keep_rows) == 0) {
  stop("No complete rows available for model fitting after NA filtering. Check missingness in model terms.")
}

df_model <- df[keep_rows, , drop = FALSE]

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
  family = glmmTMB::betabinomial(link = cfg$link_function),
  data = df_model,
  control = glmmTMBControl(optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 10))
)

cat("\nModel fit complete!\n\n")

# Save fitted model object
model_file <- file.path(run_output_dir, paste0("glmm_model_", run_suffix, ".rds"))
# saveRDS(model, model_file)


source("GLMM_postfit.R")
