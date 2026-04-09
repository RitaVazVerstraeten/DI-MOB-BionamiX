
# =====================================================
# Plot functions for GLMM entomological model
# =====================================================
library(patchwork)
#' Save GLMM Probability Time Series Plot (Mean Across Blocks)
#'
#' Plots mean observed and fitted probabilities (with 95% CI ribbon for p_bt_fitted) and total cases over time.
#' @param df_summary Data frame with summary predictions (wide format, includes uncertainty columns)
#' @param df_observed Data frame with observed values (block, year_month_date, p_observed, cases)
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @param cfg Model configuration list (for subtitle)
#' @return NULL (saves plot)
save_glmm_prob_timeseries_plot <- function(df_summary, df_observed, output_dir, run_suffix, cfg) {
  df_plot <- df_summary %>%
    dplyr::left_join(df_observed, by = c("block", "year_month_date"))

  df_plot_ts <- df_plot %>%
    dplyr::group_by(year_month_date) %>%
    dplyr::summarise(
      p_bt_fitted = mean(p_bt_fitted, na.rm = TRUE),
      p_bt_fitted_lower = mean(p_bt_fitted_lower, na.rm = TRUE),
      p_bt_fitted_upper = mean(p_bt_fitted_upper, na.rm = TRUE),
      p_R_fitted = mean(p_R_fitted, na.rm = TRUE),
      p_R_fitted_lower = mean(p_R_fitted_lower, na.rm = TRUE),
      p_R_fitted_upper = mean(p_R_fitted_upper, na.rm = TRUE),
      p_observed = mean(p_observed, na.rm = TRUE),
      cases = sum(cases, na.rm = TRUE),
      .groups = "drop"
    )

  df_plot_long <- df_plot_ts %>%
    tidyr::pivot_longer(
      cols = c(p_bt_fitted, p_R_fitted, p_observed),
      names_to = "series",
      values_to = "probability"
    ) %>%
    dplyr::mutate(
      lower = dplyr::case_when(
        series == "p_bt_fitted" ~ df_plot_ts$p_bt_fitted_lower[match(year_month_date, df_plot_ts$year_month_date)],
        series == "p_R_fitted" ~ df_plot_ts$p_R_fitted_lower[match(year_month_date, df_plot_ts$year_month_date)],
        TRUE ~ NA_real_
      ),
      upper = dplyr::case_when(
        series == "p_bt_fitted" ~ df_plot_ts$p_bt_fitted_upper[match(year_month_date, df_plot_ts$year_month_date)],
        series == "p_R_fitted" ~ df_plot_ts$p_R_fitted_upper[match(year_month_date, df_plot_ts$year_month_date)],
        TRUE ~ NA_real_
      )
    )

  subtitle_parts <- c(
    if (cfg$include_block_re) "Space RE: YES" else "Space RE: NO",
    if (cfg$include_time_re) "Time RE: YES" else "Time RE: NO",
    if (cfg$include_spatial_ar) "Space AR: YES" else "Space AR: NO",
    if (cfg$include_ar1_temporal) paste0("Time AR1: YES (", cfg$ar1_group, ")") else "Time AR1: NO",
    "Lines: mean probabilities across blocks | Bars: total cases"
  )
  plot_subtitle <- paste(subtitle_parts, collapse = " | ")

  # Add explanation for ribbon and NA values to the legend/caption
  plot_caption <- paste(
    "Shaded ribbon: 95% confidence interval for fitted probabilities (if available).",
    sep = "\n"
  )

  max_prob <- max(df_plot_long$probability, na.rm = TRUE)
  max_cases <- max(df_plot_ts$cases, na.rm = TRUE)
  scale_factor <- ifelse(is.finite(max_cases) && max_cases > 0, max_prob / max_cases, 1)

  # Only plot ribbon if aesthetics are present and not all NA
  # Only plot ribbon for p_bt_fitted if aesthetics are present and not all NA
  ribbon_data <- subset(df_plot_long, series == "p_bt_fitted")
  ribbon_ok <- nrow(ribbon_data) > 0 &&
    !all(is.na(ribbon_data$lower)) &&
    !all(is.na(ribbon_data$upper)) &&
    !all(is.na(ribbon_data$probability)) &&
    !all(is.na(ribbon_data$year_month_date))

  p_probs <- ggplot(df_plot_long, aes(x = year_month_date, y = probability, color = series, group = series)) +
    geom_col(
      data = df_plot_ts,
      aes(x = year_month_date, y = cases * scale_factor),
      inherit.aes = FALSE,
      fill = "grey75",
      alpha = 0.5,
      width = 25
    )

  if (ribbon_ok) {
    p_probs <- p_probs +
      geom_ribbon(
        data = ribbon_data,
        aes(x = year_month_date, ymin = lower, ymax = upper, fill = series),
        alpha = 0.2,
        color = NA,
        inherit.aes = FALSE
      )
  } else {
    warning("Skipping uncertainty ribbon: missing or invalid aesthetics.")
  }

  p_probs <- p_probs +
    geom_line(linewidth = 1) +
    geom_point(size = 1.3) +
    scale_color_manual(
      values = c(
        p_bt_fitted = "#1f77b4",
        p_R_fitted = "#ff7f0e",
        p_observed = "#d62728"
      )
    ) +
    scale_fill_manual(
      values = c(
        p_bt_fitted = "#1f77b4",
        p_R_fitted = "#ff7f0e"
      ),
      guide = "none"
    ) +
    scale_y_continuous(
      name = "Probability",
      sec.axis = sec_axis(~ . / scale_factor, name = "Cases")
    ) +
    labs(
      x = "Time",
      color = NULL,
      title = "Observed vs Fitted Probabilities with Cases",
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.caption = element_text(size = 10, hjust = 0)
    )

  print(p_probs)
  plot_file <- file.path(output_dir, paste0("probabilities_timeseries_", run_suffix, ".png"))
  ggsave(plot_file, p_probs, width = 12, height = 6, dpi = 150)
  cat("  Probability plot PNG: ", plot_file, "\n", sep = "")
}

#' Save Moran's I Plot for Spatial Residuals
#'
#' Plots Moran's I for spatial residuals (global and monthly), highlights significant autocorrelation in red.
#' @param monthly_moran Data frame with columns: year_month_date, moran_I, p_value
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @return NULL (saves plot)
save_glmm_moransI_plot <- function(monthly_moran, output_dir, run_suffix) {
  if (nrow(monthly_moran) == 0) return(invisible(NULL))
  p_month <- ggplot(monthly_moran, aes(x = year_month_date, y = moran_I)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    geom_line(na.rm = TRUE) +
    geom_point(aes(color = p_value < 0.05), size = 2, na.rm = TRUE) +
    scale_color_manual(values = c("TRUE" = "#d62728", "FALSE" = "#1f77b4"), na.translate = FALSE) +
    labs(
      x = "Month",
      y = "Moran's I (Pearson residuals)",
      color = "p < 0.05",
      title = "Monthly spatial autocorrelation in GLMM residuals",
      caption = "Red dots: significant spatial autocorrelation (p < 0.05).\nSpatial autocorrelation is bounded to 400m (only neighbors within 400m are considered)."
    ) +
    theme_minimal()
  ggsave(file.path(output_dir, paste0("glmm_moransI_monthly_timeseries_", run_suffix, ".png")), p_month, width = 11, height = 5, dpi = 150)
  cat("  Moran's I plot PNG: ", file.path(output_dir, paste0("glmm_moransI_monthly_timeseries_", run_suffix, ".png")), "\n", sep = "")
}


#' Save GLMM Probability Time Series Plot with Uncertainty
#'
#' Plots observed and fitted probabilities (with 95% CI ribbons) and cases over time.
#' @param df_summary Data frame with summary predictions (wide format, includes uncertainty columns)
#' @param df_observed Data frame with observed values (block, year_month_date, p_observed, cases)
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @param cfg Model configuration list (for subtitle)
#' @return NULL (saves plot)
save_glmm_prob_timeseries_plot_random_blocks <- function(
  df_summary,
  df_observed,
  output_dir,
  run_suffix,
  cfg,
  n_blocks = 10
) {

  set.seed(42)

  # Sample blocks
  blocks <- unique(df_summary$block)
  blocks_sample <- sample(blocks, size = min(n_blocks, length(blocks)))

  # Filter and join
  df_plot <- df_summary %>%
    dplyr::left_join(df_observed, by = c("block", "year_month_date")) %>%
    dplyr::filter(block %in% blocks_sample)

  # Subtitle
  subtitle_parts <- c(
    if (cfg$include_block_re) "Space RE: YES" else "Space RE: NO",
    if (cfg$include_time_re) "Time RE: YES" else "Time RE: NO",
    if (cfg$include_spatial_ar) "Space AR: YES" else "Space AR: NO",
    if (cfg$include_ar1_temporal) paste0("Time AR1: YES (", cfg$ar1_group, ")") else "Time AR1: NO"
  )
  plot_subtitle <- paste(subtitle_parts, collapse = " | ")

  plot_caption <- "Shaded ribbon: 95% confidence interval for fitted probabilities (if available)."

  # Pivot to long format for probabilities
  df_plot_long <- df_plot %>%
    tidyr::pivot_longer(
      cols = c(p_bt_fitted, p_R_fitted, p_observed),
      names_to = "series",
      values_to = "probability"
    ) %>%
    dplyr::mutate(
      lower = dplyr::case_when(
        series == "p_bt_fitted" ~ p_bt_fitted_lower,
        series == "p_R_fitted" ~ p_R_fitted_lower,
        TRUE ~ NA_real_
      ),
      upper = dplyr::case_when(
        series == "p_bt_fitted" ~ p_bt_fitted_upper,
        series == "p_R_fitted" ~ p_R_fitted_upper,
        TRUE ~ NA_real_
      )
    )

  ribbon_data <- df_plot_long %>%
    dplyr::filter(series %in% c("p_bt_fitted", "p_R_fitted"))

  # Plot
  p_probs <- ggplot(df_plot_long, aes(x = year_month_date, y = probability, color = series, group = interaction(series, block))) +
    geom_ribbon(
      data = ribbon_data,
      aes(ymin = lower, ymax = upper, fill = series, group = interaction(series, block)),
      inherit.aes = FALSE,
      alpha = 0.2,
      color = NA
    ) +
    geom_line(linewidth = 1) +
    geom_point(size = 1.3) +
    scale_color_manual(values = c(p_bt_fitted = "#1f77b4", p_R_fitted = "#ff7f0e", p_observed = "#d62728")) +
    scale_fill_manual(values = c(p_bt_fitted = "#1f77b4", p_R_fitted = "#ff7f0e"), guide = "none") +
    labs(
      x = "Time",
      y = "Probability",
      color = NULL,
      title = "Observed vs Fitted Probabilities for Random Blocks",
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.caption = element_text(size = 10, hjust = 0)
    ) +
    facet_wrap(~block, ncol = 2)

  print(p_probs)

  # Save plot
  plot_file <- file.path(output_dir, paste0("probabilities_timeseries_random_blocks_", run_suffix, ".png"))
  ggsave(plot_file, p_probs, width = 14, height = 10, dpi = 150)
  cat("  Probability plot (random blocks) PNG: ", plot_file, "\n", sep = "")
}

#' Save GLMM Probability Time Series Plot with Weighted Fitted Probability
#'
#' Plots observed and weighted fitted probabilities (with 95% CI ribbons) and cases over time.
#' @param df_summary Data frame with summary predictions (wide format, includes uncertainty columns)
#' @param df_observed Data frame with observed values (block, year_month_date, p_observed, cases)
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @param cfg Model configuration list (for subtitle)
#' @return NULL (saves plot)
save_glmm_prob_timeseries_plot_weighted <- function(df_summary_weighted, output_dir, run_suffix, cfg) {
  # df_summary_weighted must have: block, year_month_date, p_observed, omega, p_fitted_weighted, p_bt_fitted_lower, p_bt_fitted_upper, p_R_fitted_lower, p_R_fitted_upper
  df_plot <- df_summary_weighted %>%
    mutate(
      p_fitted_weighted_lower = ifelse(
        omega == 0 | is.na(p_R_fitted_lower),
        p_bt_fitted_lower,
        (1 - omega) * p_bt_fitted_lower + omega * p_R_fitted_lower
      ),
      p_fitted_weighted_upper = ifelse(
        omega == 0 | is.na(p_R_fitted_upper),
        p_bt_fitted_upper,
        (1 - omega) * p_bt_fitted_upper + omega * p_R_fitted_upper
      )
    )

  df_plot_ts <- df_plot %>%
    group_by(year_month_date) %>%
    summarise(
      p_fitted_weighted = mean(p_fitted_weighted, na.rm = TRUE),
      p_fitted_weighted_lower = mean(p_fitted_weighted_lower, na.rm = TRUE),
      p_fitted_weighted_upper = mean(p_fitted_weighted_upper, na.rm = TRUE),
      p_observed = mean(p_observed, na.rm = TRUE),
      .groups = "drop"
    )

  df_plot_long <- df_plot_ts %>%
    pivot_longer(
      cols = c(p_fitted_weighted, p_observed),
      names_to = "series",
      values_to = "probability"
    ) %>%
    mutate(
      lower = ifelse(series == "p_fitted_weighted", df_plot_ts$p_fitted_weighted_lower[match(year_month_date, df_plot_ts$year_month_date)], NA_real_),
      upper = ifelse(series == "p_fitted_weighted", df_plot_ts$p_fitted_weighted_upper[match(year_month_date, df_plot_ts$year_month_date)], NA_real_)
    )

  subtitle_parts <- c(
    if (cfg$include_block_re) "Space RE: YES" else "Space RE: NO",
    if (cfg$include_time_re) "Time RE: YES" else "Time RE: NO",
    if (cfg$include_spatial_ar) "Space AR: YES" else "Space AR: NO",
    if (cfg$include_ar1_temporal) paste0("Time AR1: YES (", cfg$ar1_group, ")") else "Time AR1: NO",
    paste0("Link: ", cfg$link_function),
    "Weighted fitted probability: (1-omega)*p_bt + omega*p_R"
  )
  plot_subtitle <- paste(subtitle_parts, collapse = " | ")
  plot_caption <- "Shaded ribbon: 95% confidence interval for weighted fitted probability."

  max_prob <- max(df_plot_long$probability, na.rm = TRUE)

  ribbon_data <- subset(df_plot_long, series == "p_fitted_weighted")
  ribbon_ok <- nrow(ribbon_data) > 0 &&
    !all(is.na(ribbon_data$lower)) &&
    !all(is.na(ribbon_data$upper)) &&
    !all(is.na(ribbon_data$probability)) &&
    !all(is.na(ribbon_data$year_month_date))

  p_probs <- ggplot(df_plot_long, aes(x = year_month_date, y = probability, color = series, group = series))

  if (ribbon_ok) {
    p_probs <- p_probs +
      geom_ribbon(
        data = ribbon_data,
        aes(x = year_month_date, ymin = lower, ymax = upper, fill = series),
        alpha = 0.2,
        color = NA,
        inherit.aes = FALSE
      )
  } else {
    warning("Skipping uncertainty ribbon: missing or invalid aesthetics.")
  }

  p_probs <- p_probs +
    geom_line(linewidth = 1) +
    geom_point(size = 1.3) +
    scale_color_manual(
      values = c(
        p_fitted_weighted = "#009E73",
        p_observed = "#d62728"
      ),
      labels = c(
        p_fitted_weighted = "Weighted Fitted",
        p_observed = "Observed"
      )
    ) +
    scale_fill_manual(
      values = c(
        p_fitted_weighted = "#009E73"
      ),
      guide = "none"
    ) +
    labs(
      x = "Time",
      color = NULL,
      title = "Observed vs Weighted Fitted Probabilities",
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.caption = element_text(size = 10, hjust = 0)
    )

  print(p_probs)
  plot_file <- file.path(output_dir, paste0("probabilities_timeseries_weighted_", run_suffix, ".png"))
  ggsave(plot_file, p_probs, width = 12, height = 6, dpi = 150)
  cat("  Probability plot (weighted) PNG: ", plot_file, "\n", sep = "")
}


#' Save GLMM Probability Time Series Plot for Random Manzanas
#'
#' Plots observed and fitted probabilities (with 95% CI ribbons) and cases over time for 10 random manzanas.
#' @param df_summary Data frame with summary predictions (wide format, includes uncertainty columns)
#' @param df_observed Data frame with observed values (block, year_month_date, p_observed, cases)
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @param cfg Model configuration list (for subtitle)
#' @param n_blocks Number of random manzanas to plot (default 10)
#' @return NULL (saves plot)
save_glmm_prob_timeseries_plot_random_blocks <- function(
  df_summary,
  df_observed,
  output_dir,
  run_suffix,
  cfg,
  n_blocks = 10
) {

  set.seed(123)

  # Sample blocks
  blocks <- unique(df_summary$block)
  blocks_sample <- sample(blocks, size = min(n_blocks, length(blocks)))

  # Filter and join
  df_plot <- df_summary %>%
    dplyr::left_join(df_observed, by = c("block", "year_month_date")) %>%
    dplyr::filter(block %in% blocks_sample)

  # Subtitle
  subtitle_parts <- c(
    if (cfg$include_block_re) "Space RE: YES" else "Space RE: NO",
    if (cfg$include_time_re) "Time RE: YES" else "Time RE: NO",
    if (cfg$include_spatial_ar) "Space AR: YES" else "Space AR: NO",
    if (cfg$include_ar1_temporal) paste0("Time AR1: YES (", cfg$ar1_group, ")") else "Time AR1: NO"
  )
  plot_subtitle <- paste(subtitle_parts, collapse = " | ")

  plot_caption <- "Shaded ribbon: 95% confidence interval for fitted probabilities (if available)."

  # Pivot long for probabilities
  df_plot_long <- df_plot %>%
    tidyr::pivot_longer(
      cols = c(p_bt_fitted, p_R_fitted, p_observed),
      names_to = "series",
      values_to = "probability"
    ) %>%
    dplyr::mutate(
      lower = dplyr::case_when(
        series == "p_bt_fitted" ~ p_bt_fitted_lower,
        series == "p_R_fitted" ~ p_R_fitted_lower,
        TRUE ~ NA_real_
      ),
      upper = dplyr::case_when(
        series == "p_bt_fitted" ~ p_bt_fitted_upper,
        series == "p_R_fitted" ~ p_R_fitted_upper,
        TRUE ~ NA_real_
      )
    )

  # Build ribbon data
  ribbon_data <- df_plot_long %>%
    dplyr::filter(
      (series == "p_bt_fitted") |
        (series == "p_R_fitted" & !is.na(lower) & !is.na(upper))
    )

  # Plot
  p_probs <- ggplot(df_plot_long, aes(x = year_month_date, y = probability, color = series, group = interaction(series, block))) +
    geom_ribbon(
      data = ribbon_data,
      aes(
        x = year_month_date,
        ymin = lower,
        ymax = upper,
        fill = series,
        group = interaction(series, block)
      ),
      inherit.aes = FALSE,
      alpha = 0.2,
      color = NA
    ) +
    geom_line(linewidth = 1) +
    geom_point(size = 1.3) +
    scale_color_manual(values = c(p_bt_fitted = "#1f77b4", p_R_fitted = "#ff7f0e", p_observed = "#d62728")) +
    scale_fill_manual(values = c(p_bt_fitted = "#1f77b4", p_R_fitted = "#ff7f0e"), guide = "none") +
    labs(
      x = "Time",
      y = "Probability",
      color = NULL,
      title = "Observed vs Fitted Probabilities for Random Blocks",
      subtitle = plot_subtitle,
      caption = plot_caption
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.caption = element_text(size = 10, hjust = 0)
    ) +
    facet_wrap(~block, ncol = 2, scales="free_y")

  print(p_probs)

  # Save plot
  plot_file <- file.path(output_dir, paste0("probabilities_timeseries_random_blocks_", run_suffix, ".png"))
  ggsave(plot_file, p_probs, width = 14, height = 10, dpi = 150)
  cat("  Probability plot (random blocks) PNG: ", plot_file, "\n", sep = "")
}

#' Save GLMM Observed vs Expected Calibration Plot
#'
#' Plots a calibration plot of observed vs expected (fitted) probabilities aggregated over time.
#' @param df_plot Data frame with columns: p_observed, p_bt_fitted (or similar)
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @return NULL (saves plot)
save_glmm_calibplot_observed_vs_expected <- function(df_summary, df_observed, output_dir, run_suffix) {
  # Merge summary and observed data
  df_plot <- df_summary %>%
    left_join(df_observed, by = c("block", "year_month_date"))
  # Remove NA values for fair comparison
  calib_df <- df_plot %>%
    filter(!is.na(p_observed) & !is.na(p_bt_fitted))
  calib_data <- data.frame(Observed = calib_df$p_observed, Expected = calib_df$p_bt_fitted)
  p_calib <- ggplot(calib_data, aes(x = Expected, y = Observed)) +
    geom_point(alpha = 0.7, color = "#0072B2") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
      x = "Expected (Fitted Probability)",
      y = "Observed Probability",
      title = "Calibration Plot: Observed vs Expected Probabilities",
      subtitle = "Paired by block/month"
    ) +
    theme_minimal()
  print(p_calib)
  calibplot_file <- file.path(output_dir, paste0("glmm_calibplot_observed_vs_expected_", run_suffix, ".png"))
  ggsave(calibplot_file, p_calib, width = 7, height = 7, dpi = 150)
  cat("  Calibration plot PNG: ", calibplot_file, "\n", sep = "")
}


#' Save Calibration Plot for Weighted Average Fitted Probability
#'
#' Plots a calibration plot of observed vs weighted average fitted probability: p_fit = (1-omega)*p_bt + omega*p_R
#' @param df Data frame with columns: p_observed, p_bt_fitted, p_R_fitted, omega
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @return NULL (saves plot)
save_glmm_calibplot_weighted_avg <- function(df, output_dir, run_suffix) {
  # Remove NA values for fair comparison
  calib_df <- df %>%
    filter(!is.na(p_observed) & !is.na(p_fitted_weighted))
  calib_data <- data.frame(Observed = df$p_observed, Weighted_Fitted = df$p_fitted_weighted)

  p_calib <- ggplot(calib_data, aes(x = Weighted_Fitted, y = Observed)) +
    geom_point(alpha = 0.7, color = "#009E73") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
      x = "Weighted Fitted Probability",
      y = "Observed Probability",
      title = "Calibration Plot: Observed vs Weighted Fitted Probability",
      subtitle = "Weighted average: (1-omega)*p_bt + omega*p_R"
    ) +
    theme_minimal()
  print(p_calib)
  calibplot_file <- file.path(output_dir, paste0("glmm_calibplot_weighted_avg_", run_suffix, ".png"))
  ggsave(calibplot_file, p_calib, width = 7, height = 7, dpi = 150)
  cat("  Calibration plot (weighted avg) PNG: ", calibplot_file, "\n", sep = "")
}


#' Save GLMM Residuals Plot
#'
#' Plots Pearson residuals vs fitted values for a glmmTMB model and saves as PNG.
#' @param model A fitted glmmTMB model
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @return NULL (saves plot)
save_glmm_residuals_plot <- function(model, output_dir, run_suffix) {
  resid_plot_file <- file.path(output_dir, paste0("glmm_residuals_plot_", run_suffix, ".png"))
  residuals_model <- residuals(model, type = "pearson")
  df_resid <- data.frame(
    fitted = fitted(model),
    residuals = residuals_model
  )
  p_resid <- ggplot(df_resid, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(
      x = "Fitted values",
      y = "Pearson residuals",
      title = "Residuals vs Fitted Values"
    ) +
    theme_minimal()
  ggsave(resid_plot_file, p_resid, width = 8, height = 6, dpi = 150)
  cat("  Residuals plot PNG: ", resid_plot_file, "\n", sep = "")
}

#' Save GLMM Random Effects Plot
#'
#' Plots histograms of random effects for each grouping factor in a glmmTMB model and saves as PNG.
#' @param model A fitted glmmTMB model
#' @param output_dir Output directory for plot
#' @param run_suffix Suffix for filename
#' @return NULL (saves plot)
save_glmm_random_effects_plot <- function(model, output_dir, run_suffix) {
  re_plot_file <- file.path(output_dir, paste0("glmm_random_effects_plot_", run_suffix, ".png"))
  re <- suppressWarnings(ranef(model)$cond)
  # ranef extracts the BLUPs (best linear unbiased prediction) from the model
  if (length(re) > 0) {
    re_df <- dplyr::bind_rows(lapply(names(re), function(grp) {
      data.frame(
        group = grp,
        level = rownames(re[[grp]]),
        effect = re[[grp]][, 1],
        stringsAsFactors = FALSE
      )
    }))
    p_re <- ggplot(re_df, aes(x = effect)) +
      geom_histogram(bins = 30, fill = "skyblue", color = "white") +
      facet_wrap(~ group, scales = "free_y") +
      labs(
        x = "Random effect value",
        y = "Count",
        title = "Distribution of Random Effects"
      ) +
      theme_minimal()
    ggsave(re_plot_file, p_re, width = 8, height = 6, dpi = 150)
    cat("  Random effects plot PNG: ", re_plot_file, "\n", sep = "")
  } else {
    cat("  No random effects to plot.\n")
  }
}



# Needed for CRAN checks and to avoid 'no visible global function definition' errors
#' @importFrom ggplot2 ggplot aes geom_point geom_hline labs theme_minimal ggsave geom_histogram facet_wrap
#' @importFrom glmmTMB ranef
NULL
# =====================================================
# Plot functions for Stan entomological model
# =====================================================

#' Save Random Effects Diagnostic Plot
#'
#' Creates a 2x2 grid plot showing spatial and temporal random effects diagnostics:
#' - Spatial: histogram and Q-Q plot
#' - Temporal: time series line plot and ACF (or placeholders if RE disabled)
#'
#' @param u_post Numeric vector of spatial random effects (u_block_out)
#' @param v_post Numeric vector of temporal random effects (v_time_out, or NA if disabled)
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filename
#' @return NULL (saves plot to PNG file)
save_random_effects <- function(u_post, v_post, output_dir, run_suffix) {
  png(file.path(output_dir, paste0("random_effects_", run_suffix, ".png")), width = 1000, height = 800)
  par(mfrow = c(2, 2))

  hist(u_post, breaks = 50, main = "Distribution of Spatial Random Effects (u_b)",
       xlab = "Effect", col = "lightblue", border = "white")
  abline(v = 0, lty = 2, col = "red", lwd = 2)

  qqnorm(u_post, main = "Q-Q Plot: Spatial Effects", pch = 19, cex = 0.5, col = "blue")
  qqline(u_post, col = "red", lwd = 2)

  if (!all(is.na(v_post))) {
    plot(v_post, type = "b", main = "Temporal Random Effects (v_t) with AR(1)",
         xlab = "Time", ylab = "Effect", col = "red", pch = 19)
    abline(h = 0, lty = 2, col = "gray")
    acf(v_post, main = "ACF of Temporal Effects", col = "darkred")
  } else {
    plot.new(); text(0.5, 0.5, "Temporal RE disabled\n(no v_time_out in model)")
    plot.new(); text(0.5, 0.5, "ACF unavailable\n(temporal RE disabled)")
  }

  par(mfrow = c(1, 1))
  dev.off()
}

#' Save Posterior Predictive Check Plot
#'
#' Posterior predictive check with three panels:
#'   1. Proportion of zeros across replicated datasets vs observed
#'   2. Distribution of non-zero counts: observed histogram overlaid with
#'      a sample of replicated datasets
#'   3. Fitted vs observed scatter using posterior mean (for bias diagnosis)
#'
#' @param df Data frame containing observed y_bt values
#' @param fit CmdStan fit object (used to extract y_pred draws)
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filename
#' @param n_draws_overlay Number of replicated datasets to overlay in panel 2
#' @return NULL (saves plot to PNG file)
save_ppc <- function(df, fit, output_dir, run_suffix, n_draws_overlay = 50) {
  y_pred_draws <- fit$draws("y_pred", format = "matrix")  # chains x iterations matrix
  y_obs        <- df$y_bt

  # --- Panel 1: proportion of y_bt == zeros ---
  prop_zero_rep <- rowMeans(y_pred_draws == 0)
  prop_zero_obs <- mean(y_obs == 0)

  p1 <- ggplot(data.frame(prop_zero = prop_zero_rep), aes(x = prop_zero)) +
    geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
    geom_vline(xintercept = prop_zero_obs, colour = "red", linewidth = 1) +
    annotate("text", x = prop_zero_obs, y = Inf,
             label = sprintf("observed\n%.2f", prop_zero_obs),
             colour = "red", hjust = -0.1, vjust = 1.5, size = 3) +
    labs(title = "Proportion of zeros",
         subtitle = "Histogram = replicated datasets; red = observed",
         x = "Proportion of zeros", y = "Count") +
    theme_minimal()

  # --- Panel 2: distribution of non-zero y_bt counts ---
  nonzero_obs <- y_obs[y_obs > 0]
  draw_idx    <- sample(nrow(y_pred_draws), min(n_draws_overlay, nrow(y_pred_draws)))

  rep_nonzero_df <- do.call(rbind, lapply(draw_idx, function(i) {
    vals <- y_pred_draws[i, ][y_pred_draws[i, ] > 0]
    if (length(vals) == 0) return(NULL)
    data.frame(count = vals, draw = i)
  }))

  obs_counts <- as.data.frame(table(count = nonzero_obs))
  obs_counts$count <- as.integer(as.character(obs_counts$count))

  p2 <- ggplot() +
    geom_histogram(
      data = rep_nonzero_df,
      aes(x = count, group = draw),
      binwidth = 1, center = 1, fill = "steelblue", alpha = 0.05, position = "identity"
    ) +
    geom_point(
      data = obs_counts,
      aes(x = count, y = Freq),
      colour = "red", size = 1.5
    ) +
    geom_line(
      data = obs_counts,
      aes(x = count, y = Freq),
      colour = "red", linewidth = 0.6
    ) +
    labs(title = "Distribution of non-zero counts",
         subtitle = sprintf("Blue = %d replicated datasets; red = observed", n_draws_overlay),
         x = "y_bt (non-zero only)", y = "Count") +
    scale_x_continuous(
      breaks = function(x) seq(ceiling(x[1]), floor(x[2]), by = 1),
      labels = function(x) ifelse(x %% 5 == 0, x, ""),
      minor_breaks = NULL
    ) +
    coord_cartesian(xlim = c(1, NA)) +
    theme_minimal() +
    theme(panel.grid.major.x = element_line(colour = "grey85", linewidth = 0.3))

  # --- Panel 3: fitted vs observed (posterior mean) ---
  post_mean <- colMeans(y_pred_draws)

  p3 <- ggplot(data.frame(observed = y_obs, predicted = post_mean),
               aes(observed, predicted)) +
    geom_point(alpha = 0.3, size = 0.8) +
    geom_abline(slope = 1, intercept = 0, colour = "red") +
    labs(title = "Fitted vs observed (posterior mean)",
         x = "Observed y_bt", y = "Posterior mean y_pred") +
    theme_minimal()

  p_combined <- p1 + p2 + p3 + patchwork::plot_layout(ncol = 3)

  ggsave(
    file.path(output_dir, paste0("posterior_predictive_check_", run_suffix, ".png")),
    p_combined, width = 15, height = 5, dpi = 150
  )
}

#' Save MCMC Trace Plots
#'
#' Creates trace plots for MCMC diagnostics using bayesplot package.
#' Generates three separate plots: main parameters, lagged weights (w), and
#' unlagged weights (w_unlagged). Conditionally includes temporal RE parameters
#' (sigma_v, rho) if enabled.
#'
#' @param fit Stan fit object (cmdstanr)
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filenames
#' @param use_temporal_re Logical flag indicating whether temporal RE is enabled
#' @return NULL (saves plots to PNG files or returns invisibly if bayesplot not installed)
save_trace_plots <- function(fit, output_dir, run_suffix, use_temporal_re) {
  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    cat("bayesplot package not installed; skipping trace plots.\n")
    return(invisible(NULL))
  }

  library(bayesplot)
  draws_array <- fit$draws(format = "array")

  params_main <- c("alpha", "sigma_u", "delta0", "delta1")
  if (use_temporal_re) params_main <- c(params_main, "sigma_v", "rho")

  ggsave(
    file.path(output_dir, paste0("traceplot_params_", run_suffix, ".png")),
    mcmc_trace(draws_array, pars = params_main), width = 10, height = 8
  )

  w_params <- grep("^w\\[", dimnames(draws_array)[[3]], value = TRUE)
  if (length(w_params) > 0) {
    ggsave(
      file.path(output_dir, paste0("traceplot_weights_w_", run_suffix, ".png")),
      mcmc_trace(draws_array, pars = w_params), width = 12, height = 10
    )
  }

  wu_params <- grep("^w_unlagged\\[", dimnames(draws_array)[[3]], value = TRUE)
  if (length(wu_params) > 0) {
    ggsave(
      file.path(output_dir, paste0("traceplot_weights_unlagged_", run_suffix, ".png")),
      mcmc_trace(draws_array, pars = wu_params), width = 12, height = 8
    )
  }
}

#' Save Time Series Diagnostic Plots
#'
#' Creates four time series plots: aggregate time series, block-specific time series,
#' residuals over time, and correlation distribution histogram.
#'
#' @param df Data frame with observed_p_bt, fitted_p_bt, year_month_date, block, N_HH, y_bt
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filenames
#' @param n_blocks_facet Integer number of blocks to show in faceted plot
#' @return NULL (saves four plots to PNG files)
save_timeseries_plots <- function(df, output_dir, run_suffix, n_blocks_facet = 9) {
  # Ensure tidyr is available
  if (!requireNamespace("tidyr", quietly = TRUE)) {
    cat("tidyr package not installed; skipping time series plots.\n")
    return(invisible(NULL))
  }
  
  library(tidyr)
  
  # Create output directory
  timeseries_dir <- output_dir  

  # --- Extract posterior draws for p_bt_out and y_pred if fit is available ---
  p1 <- NULL
  draws_ok <- FALSE
  if ("fit" %in% ls(envir = .GlobalEnv)) {
    fit_obj <- get("fit", envir = .GlobalEnv)
    if (inherits(fit_obj, "CmdStanMCMC")) {
      draws_p    <- tryCatch(fit_obj$draws("p_bt_out", format = "matrix"), error = function(e) NULL)
      draws_pred <- tryCatch(fit_obj$draws("y_pred",   format = "matrix"), error = function(e) NULL)
      if (!is.null(draws_p)) {
        draws_ok <- TRUE
        time_points <- sort(unique(df$year_month_date))
        n_draws     <- nrow(draws_p)

        # Helper: aggregate draws across blocks per time point
        agg_draws_by_time <- function(draws_mat, scale_by = NULL) {
          mat <- sapply(time_points, function(tp) {
            idx <- which(df$year_month_date == tp)
            if (length(idx) == 0) return(rep(NA, n_draws))
            vals <- draws_mat[, idx, drop = FALSE]
            if (!is.null(scale_by)) vals <- vals / matrix(scale_by[idx], nrow = n_draws, ncol = length(idx), byrow = TRUE)
            rowMeans(vals, na.rm = TRUE)
          })
          t(mat)  # n_time x n_draws
        }

        agg_p    <- agg_draws_by_time(draws_p)
        agg_pred <- if (!is.null(draws_pred)) agg_draws_by_time(draws_pred, scale_by = df$n_bt) else NULL

        summarise_draws <- function(mat) {
          data.frame(
            mean  = apply(mat, 1, mean,     na.rm = TRUE),
            lower = apply(mat, 1, quantile, probs = 0.025, na.rm = TRUE),
            upper = apply(mat, 1, quantile, probs = 0.975, na.rm = TRUE)
          )
        }

        p_summ    <- summarise_draws(agg_p)
        pred_summ <- if (!is.null(agg_pred)) summarise_draws(agg_pred) else NULL

        obs_summary <- df %>%
          group_by(year_month_date) %>%
          summarise(observed_mean = mean(observed_p_bt, na.rm = TRUE), .groups = "drop")

        plot_df <- data.frame(
          year_month_date = time_points,
          p_mean          = p_summ$mean,
          p_lower         = p_summ$lower,
          p_upper         = p_summ$upper,
          observed_mean   = obs_summary$observed_mean
        )
        if (!is.null(pred_summ)) {
          plot_df$pred_mean  <- pred_summ$mean
          plot_df$pred_lower <- pred_summ$lower
          plot_df$pred_upper <- pred_summ$upper
        }

        p1 <- ggplot(plot_df, aes(x = year_month_date)) +
          geom_ribbon(aes(ymin = p_lower, ymax = p_upper), fill = "blue", alpha = 0.18) +
          geom_line(aes(y = p_mean,        color = "Fitted p_bt"),   linewidth = 1) +
          geom_point(aes(y = p_mean,       color = "Fitted p_bt"),   size = 1.5) +
          geom_line(aes(y = observed_mean, color = "Observed y/n"),  linewidth = 1) +
          geom_point(aes(y = observed_mean, color = "Observed y/n"), size = 1.5)

        if (!is.null(pred_summ))
          p1 <- p1 +
            geom_ribbon(aes(ymin = pred_lower, ymax = pred_upper), fill = "#E69F00", alpha = 0.2) +
            geom_line(aes(y = pred_mean,  color = "Predicted y_pred/n"), linewidth = 1, linetype = "dashed") +
            geom_point(aes(y = pred_mean, color = "Predicted y_pred/n"), size = 1.5)

        p1 <- p1 +
          scale_color_manual(
            values = c("Fitted p_bt"       = "blue",
                       "Observed y/n"       = "red",
                       "Predicted y_pred/n" = "#E69F00"),
            breaks = c("Observed y/n", "Predicted y_pred/n", "Fitted p_bt")
          ) +
          labs(x = "Time", y = "Probability / Rate",
               title = "Time Series: observed rate, predicted rate, and fitted p_bt (mean across blocks)",
               color = NULL,
               caption = "Shaded ribbons: 95% CI for p_bt (blue) and y_pred/n_bt (orange)") +
          theme_minimal() +
          theme(legend.position = "bottom")
      }
    }
  }
  if (!draws_ok) {
    # Fallback: plot without uncertainty, no y_pred
    p1 <- df %>%
      group_by(year_month_date) %>%
      summarise(
        fitted_mean   = mean(fitted_p_bt,   na.rm = TRUE),
        observed_mean = mean(observed_p_bt, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      pivot_longer(cols = c(fitted_mean, observed_mean), names_to = "type", values_to = "probability") %>%
      ggplot(aes(x = year_month_date, y = probability, color = type)) +
      geom_line(linewidth = 1) +
      geom_point(size = 1.5) +
      scale_color_manual(values = c("fitted_mean" = "blue", "observed_mean" = "red"),
                         labels = c("Fitted p_bt", "Observed y/n")) +
      labs(x = "Time", y = "Probability",
           title = "Time Series: Observed vs Fitted Mosquito Probability (Mean Across Blocks)",
           color = NULL) +
      theme_minimal() +
      theme(legend.position = "bottom")
  }
  ggsave(file.path(timeseries_dir, paste0("timeseries_aggregate_", run_suffix, ".png")),
         p1, width = 12, height = 6, dpi = 150)
  
  # Plot 2: Block-specific time series (first n_blocks_facet blocks)
  block_ids <- sort(unique(df$block))[seq_len(min(n_blocks_facet, length(unique(df$block))))]
  p2_df <- df %>% filter(block %in% block_ids)
  p2 <- ggplot(p2_df, aes(x = year_month_date)) +
    geom_ribbon(aes(ymin = y_pred_rate_q05, ymax = y_pred_rate_q95), fill = "blue", alpha = 0.2) +
    geom_line(aes(y = y_pred_rate,  color = "Predicted rate (y_pred/n)"), alpha = 0.8, linewidth = 0.7) +
    geom_line(aes(y = observed_p_bt, color = "Observed rate (y/n)"),      alpha = 0.8, linewidth = 0.6) +
    geom_point(aes(y = observed_p_bt, color = "Observed rate (y/n)"),     size = 0.8, alpha = 0.7) +
    facet_wrap(~block, ncol = 3, scales = "free_y") +
    scale_color_manual(values = c(
      "Predicted rate (y_pred/n)" = "blue",
      "Observed rate (y/n)"       = "red"
    )) +
    labs(x = "Time", y = "Detection rate (positives / inspections)",
         title = "Time Series by Block: observed vs predicted detection rate",
         color = NULL) +
    theme_minimal() +
    theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 1, size = 7))
  
  ggsave(file.path(timeseries_dir, paste0("timeseries_by_block_", run_suffix, ".png")), 
         p2, width = 14, height = 10, dpi = 150)
  
  # Plot 3: Residuals over time
  p3 <- df %>%
    mutate(residual = observed_p_bt - fitted_p_bt) %>%
    group_by(year_month_date) %>%
    summarise(
      mean_residual = mean(residual, na.rm = TRUE),
      sd_residual = sd(residual, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    ggplot(aes(x = year_month_date, y = mean_residual)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_line(color = "darkred", linewidth = 1) +
    geom_point(color = "darkred", size = 2) +
    geom_ribbon(aes(ymin = mean_residual - sd_residual, ymax = mean_residual + sd_residual),
                alpha = 0.2, fill = "darkred") +
    labs(x = "Time", y = "Mean Residual (Observed - Fitted)",
         title = "Residuals Over Time (Mean ± SD Across Blocks)") +
    theme_minimal()
  
  ggsave(file.path(timeseries_dir, paste0("residuals_over_time_", run_suffix, ".png")), 
         p3, width = 12, height = 6, dpi = 150)
  
  # Plot 4: Correlation distribution
  df_corr <- df %>%
    group_by(block) %>%
    summarise(correlation = suppressWarnings(cor(observed_p_bt, fitted_p_bt, use = "complete.obs")), .groups = "drop") %>%
    filter(!is.na(correlation))
  
  p4 <- ggplot(df_corr, aes(x = correlation)) +
    geom_histogram(bins = 20, fill = "steelblue", color = "black", alpha = 0.7) +
    geom_vline(xintercept = median(df_corr$correlation), linetype = "dashed", color = "red", linewidth = 1) +
    labs(x = "Correlation (Observed vs Fitted)",
         y = "Number of Blocks",
         title = "Correlation p_observed vs p_bt_fitted over all timepoints",
         subtitle = paste0("Median correlation: ", round(median(df_corr$correlation), 3))) +
    theme_minimal()
  
  ggsave(file.path(timeseries_dir, paste0("correlation_distribution_", run_suffix, ".png")),
         p4, width = 10, height = 6, dpi = 150)
  
  # Print summary statistics
  cat("\nTime series plot summary statistics:\n")
  cat("  Overall correlation:", round(cor(df$observed_p_bt, df$fitted_p_bt), 3), "\n")
  cat("  RMSE:", round(sqrt(mean((df$observed_p_bt - df$fitted_p_bt)^2)), 4), "\n")
  cat("  MAE:", round(mean(abs(df$observed_p_bt - df$fitted_p_bt)), 4), "\n")
  cat("  Time series plots saved to:", timeseries_dir, "\n")
}

