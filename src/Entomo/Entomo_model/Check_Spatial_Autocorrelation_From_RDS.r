library(tidyverse)
library(glmmTMB)
library(DHARMa)
library(spdep)
library(sf)

# =========================
# SETTINGS
# =========================
cfg <- list(
  # Folder where your fitted outputs were written
    # output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM",
  output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/GLMM/space-RE_time-noRE_ar-AR1-block_lag1_k2",

  # Optional explicit files (set to NULL to auto-pick latest)
  model_rds = NULL,
  expanded_predictions_csv = NULL,

  # Preferred: derive coordinates directly from block polygons in sf
  shapefile_path = "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Manzanas_cleaned_05032026/Mz_CMF_Correcto_2022026.shp",
  sf_block_col = "CODIGO_",

  # Spatial weights: inverse-distance weighting (IDW)
  idw_power = 1,
  idw_epsilon = 1e-6,

  # Monthly Moran's I (only if date column is available)
  min_blocks_per_month = 30,

  # Simulation diagnostics
  dharma_n_sim = 500
)

# =========================
# HELPERS
# =========================
latest_file <- function(dir_path, pattern) {
  files <- list.files(dir_path, pattern = pattern, full.names = TRUE, recursive = TRUE)
  if (length(files) == 0) return(NULL)
  info <- file.info(files)
  files[order(info$mtime, decreasing = TRUE)][1]
}

safe_moran <- function(df, value_col, x_col, y_col, idw_power = 1, idw_epsilon = 1e-6) {
  df <- df %>%
    filter(!is.na(.data[[value_col]]), !is.na(.data[[x_col]]), !is.na(.data[[y_col]]))

  if (nrow(df) < 3) {
    return(tibble(ok = FALSE, message = paste0("Not enough points (", nrow(df), ") for Moran's I")))
  }

  coords <- as.matrix(df[, c(x_col, y_col)])
  dist_mat <- as.matrix(dist(coords))
  diag(dist_mat) <- NA_real_

  # Handle duplicated coordinates robustly: zero distance (off-diagonal) gets zero weight.
  zero_offdiag <- !is.na(dist_mat) & dist_mat == 0
  if (any(zero_offdiag)) {
    dist_mat[zero_offdiag] <- NA_real_
  }

  w_mat <- 1 / (pmax(dist_mat, idw_epsilon)^idw_power)
  w_mat[is.na(w_mat)] <- 0
  diag(w_mat) <- 0

  if (all(rowSums(w_mat) == 0)) {
    return(tibble(ok = FALSE, message = "All IDW row sums are zero; check coordinates."))
  }

  lw <- mat2listw(w_mat, style = "W")

  mt <- moran.test(df[[value_col]], lw, zero.policy = TRUE)

  tibble(
    ok = TRUE,
    n = nrow(df),
    moran_I = unname(mt$estimate[["Moran I statistic"]]),
    expectation = unname(mt$estimate[["Expectation"]]),
    variance = unname(mt$estimate[["Variance"]]),
    p_value = mt$p.value,
    method = mt$method
  )
}

# =========================
# 1) FIND FILES
# =========================
model_file <- if (is.null(cfg$model_rds)) {
  latest_file(cfg$output_dir, "^glmm_model_.*\\.rds$")
} else cfg$model_rds

if (is.null(model_file) || !file.exists(model_file)) {
  stop("Could not find model .rds file. Set cfg$model_rds explicitly.")
}

expanded_file <- if (is.null(cfg$expanded_predictions_csv)) {
  run_id_tmp <- sub("^glmm_model_", "", basename(model_file))
  run_id_tmp <- sub("\\.rds$", "", run_id_tmp)
  candidate <- file.path(dirname(model_file), paste0("glmm_expanded_predictions_", run_id_tmp, ".csv"))
  if (file.exists(candidate)) candidate else latest_file(cfg$output_dir, "^glmm_expanded_predictions_.*\\.csv$")
} else cfg$expanded_predictions_csv

cat("Using model file:\n", model_file, "\n\n", sep = "")

if (!is.null(expanded_file)) {
  cat("Using expanded predictions file:\n", expanded_file, "\n\n", sep = "")
}

# output folder for diagnostics
run_id <- sub("^glmm_model_", "", basename(model_file))
run_id <- sub("\\.rds$", "", run_id)
diag_dir <- file.path(dirname(model_file), paste0("spatial_diagnostics_", run_id))
dir.create(diag_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 2) LOAD MODEL + BASIC DIAGNOSTICS
# =========================
model <- readRDS(model_file)
sm <- summary(model)

cat("AIC:", AIC(model), "\n")
cat("BIC:", BIC(model), "\n")
cat("nobs:", nobs(model), "\n\n")

capture.output(print(sm), file = file.path(diag_dir, "model_summary.txt"))

# =========================
# 3) NON-SPATIAL RESIDUAL DIAGNOSTICS (DHARMa)
# =========================
set.seed(123)
sim <- simulateResiduals(model, n = cfg$dharma_n_sim, plot = FALSE)

png(file.path(diag_dir, "dharma_diagnostics.png"), width = 1200, height = 900, res = 120)
plot(sim)
dev.off()

disp <- testDispersion(sim)
zi <- testZeroInflation(sim)
outl <- testOutliers(sim)

cat("DHARMa diagnostics:\n")
print(disp)
print(zi)
print(outl)

writeLines(c(
  paste("Dispersion p:", signif(disp$p.value, 4)),
  paste("Zero-inflation p:", signif(zi$p.value, 4)),
  paste("Outliers p:", signif(outl$p.value, 4))
), con = file.path(diag_dir, "dharma_tests.txt"))

# =========================
# 4) BUILD RESIDUAL TABLE WITH BLOCK INFO
# =========================
pearson_resid <- residuals(model, type = "pearson")
fitted_prob <- predict(model, type = "response")

model_df <- tryCatch(model.frame(model), error = function(e) NULL)

resid_df <- tibble(
  row_id = seq_along(pearson_resid),
  pearson_resid = as.numeric(pearson_resid),
  fitted_prob = as.numeric(fitted_prob)
)

if (!is.null(model_df)) {
  model_df <- as_tibble(model_df)
  if (nrow(model_df) == nrow(resid_df)) {
    for (nm in c("block", "year_month", "year_month_date", "type")) {
      if (nm %in% names(model_df)) resid_df[[nm]] <- model_df[[nm]]
    }
  }
}

# Fallback from expanded predictions file if needed
if (!"block" %in% names(resid_df) && !is.null(expanded_file) && file.exists(expanded_file)) {
  exp_df <- read_csv(expanded_file, show_col_types = FALSE)
  exp_df_fit <- exp_df %>% filter(!is.na(fitted_prob))

  if (nrow(exp_df_fit) == nrow(resid_df)) {
    for (nm in c("block", "year_month", "year_month_date", "type")) {
      if (nm %in% names(exp_df_fit)) resid_df[[nm]] <- exp_df_fit[[nm]]
    }
  }
}

write_csv(resid_df, file.path(diag_dir, "residuals_used_for_spatial_checks.csv"))

if (!"block" %in% names(resid_df)) {
  cat("No block column found in aligned model rows.\n")
  cat("Spatial autocorrelation checks skipped.\n")
  cat("You can still use DHARMa outputs in:", diag_dir, "\n")
  quit(save = "no")
}

# =========================
# 5) SPATIAL CHECKS (MORAN'S I)
# =========================
# Add spatial coordinates to residuals by matching resid_df$block to sf[[cfg$sf_block_col]]
if (is.null(cfg$shapefile_path) || !file.exists(cfg$shapefile_path)) {
  stop("Shapefile not found. Set cfg$shapefile_path to a valid .shp file.")
}

sf_blocks <- st_read(cfg$shapefile_path, quiet = TRUE)

if (!cfg$sf_block_col %in% names(sf_blocks)) {
  stop("sf block id column not found: ", cfg$sf_block_col)
}

cent <- suppressWarnings(st_point_on_surface(sf_blocks))
cent_xy <- st_coordinates(cent)

coords_df <- sf_blocks %>%
  st_drop_geometry() %>%
  mutate(
    block = as.character(.data[[cfg$sf_block_col]]),
    x = as.numeric(cent_xy[, 1]),
    y = as.numeric(cent_xy[, 2])
  ) %>%
  select(block, x, y) %>%
  distinct(block, .keep_all = TRUE)

cat("Using coordinates from shapefile centroids:\n", cfg$shapefile_path, "\n\n", sep = "")

resid_df <- resid_df %>% mutate(block = as.character(block))

# Join check: how many model blocks received coordinates
block_match <- resid_df %>%
  distinct(block) %>%
  left_join(coords_df %>% mutate(has_coord = TRUE), by = "block") %>%
  mutate(has_coord = ifelse(is.na(has_coord), FALSE, has_coord))

write_csv(block_match, file.path(diag_dir, "block_coordinate_match.csv"))

cat("Blocks in residual data:", nrow(block_match), "\n")
cat("Blocks matched to coordinates:", sum(block_match$has_coord), "\n")
cat("Blocks unmatched:", sum(!block_match$has_coord), "\n\n")

# 5a) Global (all months together), block-average Pearson residual
block_resid <- resid_df %>%
  group_by(block) %>%
  summarise(
    mean_pearson = mean(pearson_resid, na.rm = TRUE),
    n_obs = n(),
    .groups = "drop"
  ) %>%
  left_join(coords_df, by = "block")

global_moran <- safe_moran(
  df = block_resid,
  value_col = "mean_pearson",
  x_col = "x",
  y_col = "y",
  idw_power = cfg$idw_power,
  idw_epsilon = cfg$idw_epsilon
)

write_csv(global_moran, file.path(diag_dir, "moran_global_block_mean_residual.csv"))

# 5b) Monthly Moran's I, if date/time exists
monthly_moran <- tibble()
if ("year_month_date" %in% names(resid_df)) {
  tmp <- resid_df %>%
    filter(!is.na(year_month_date)) %>%
    group_by(year_month_date, block) %>%
    summarise(mean_pearson = mean(pearson_resid, na.rm = TRUE), .groups = "drop") %>%
    left_join(coords_df, by = "block")

  months <- sort(unique(tmp$year_month_date))

  monthly_list <- vector("list", length(months))
  for (i in seq_along(months)) {
    m <- months[i]
    dfm <- tmp %>% filter(year_month_date == m)

    if (nrow(dfm) < cfg$min_blocks_per_month) {
      monthly_list[[i]] <- tibble(
        year_month_date = m,
        ok = FALSE,
        n = nrow(dfm),
        moran_I = NA_real_,
        p_value = NA_real_,
        message = paste0("< min_blocks_per_month (", cfg$min_blocks_per_month, ")")
      )
      next
    }

    mt <- safe_moran(
      dfm,
      "mean_pearson",
      "x",
      "y",
      idw_power = cfg$idw_power,
      idw_epsilon = cfg$idw_epsilon
    )

    if (isTRUE(mt$ok[1])) {
      monthly_list[[i]] <- mt %>%
        transmute(
          year_month_date = m,
          ok,
          n,
          moran_I,
          p_value,
          message = NA_character_
        )
    } else {
      monthly_list[[i]] <- tibble(
        year_month_date = m,
        ok = FALSE,
        n = nrow(dfm),
        moran_I = NA_real_,
        p_value = NA_real_,
        message = mt$message[1]
      )
    }
  }

  monthly_moran <- bind_rows(monthly_list)
  write_csv(monthly_moran, file.path(diag_dir, "moran_monthly_residuals.csv"))

  if (nrow(monthly_moran) > 0) {
    p_month <- ggplot(monthly_moran, aes(x = year_month_date, y = moran_I)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
      geom_line(na.rm = TRUE) +
      geom_point(aes(color = p_value < 0.05), na.rm = TRUE) +
      scale_color_manual(values = c("TRUE" = "#d62728", "FALSE" = "#1f77b4"), na.translate = FALSE) +
      labs(
        x = "Month",
        y = "Moran's I (Pearson residuals)",
        color = "p < 0.05",
        title = "Monthly spatial autocorrelation in residuals"
      ) +
      theme_minimal()

    ggsave(file.path(diag_dir, "moran_monthly_timeseries.png"), p_month, width = 11, height = 5, dpi = 150)
  }
}

# =========================
# 6) SIMPLE DECISION SUPPORT
# =========================
recommendation <- c("Spatial autocorrelation recommendation:")

if (nrow(global_moran) == 1 && isTRUE(global_moran$ok[1])) {
  recommendation <- c(
    recommendation,
    paste0("- Global Moran's I = ", round(global_moran$moran_I[1], 4),
           ", p = ", signif(global_moran$p_value[1], 4), ".")
  )

  if (!is.na(global_moran$p_value[1]) && global_moran$p_value[1] < 0.05) {
    recommendation <- c(recommendation, "- Evidence of residual spatial autocorrelation: consider spatial structure.")
  } else {
    recommendation <- c(recommendation, "- No strong global residual spatial autocorrelation detected.")
  }
}

if (nrow(monthly_moran) > 0) {
  valid <- monthly_moran %>% filter(ok, !is.na(p_value))
  if (nrow(valid) > 0) {
    prop_sig <- mean(valid$p_value < 0.05)
    recommendation <- c(
      recommendation,
      paste0("- Proportion of months with significant Moran's I: ", round(100 * prop_sig, 1), "%.")
    )

    if (prop_sig >= 0.2) {
      recommendation <- c(recommendation, "- Persistent month-level spatial autocorrelation present; adding spatial terms is recommended.")
    } else {
      recommendation <- c(recommendation, "- Month-level spatial autocorrelation appears limited.")
    }
  }
}

writeLines(recommendation)

cat("\nDiagnostics saved in:\n", diag_dir, "\n", sep = "")
