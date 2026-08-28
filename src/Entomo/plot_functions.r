
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
      p_bt_fitted       = mean(p_bt_fitted,       na.rm = TRUE),
      p_bt_fitted_lower = mean(p_bt_fitted_lower, na.rm = TRUE),
      p_bt_fitted_upper = mean(p_bt_fitted_upper, na.rm = TRUE),
      p_observed        = mean(p_observed,         na.rm = TRUE),
      cases             = sum(cases,               na.rm = TRUE),
      .groups = "drop"
    )

  df_plot_long <- df_plot_ts %>%
    tidyr::pivot_longer(
      cols      = c(p_bt_fitted, p_observed),
      names_to  = "series",
      values_to = "probability"
    ) %>%
    dplyr::mutate(
      lower = dplyr::if_else(
        series == "p_bt_fitted",
        df_plot_ts$p_bt_fitted_lower[match(year_month_date, df_plot_ts$year_month_date)],
        NA_real_
      ),
      upper = dplyr::if_else(
        series == "p_bt_fitted",
        df_plot_ts$p_bt_fitted_upper[match(year_month_date, df_plot_ts$year_month_date)],
        NA_real_
      )
    )

  subtitle_parts <- c(
    if (cfg$include_block_re)     "Space RE: YES"  else "Space RE: NO",
    if (cfg$include_time_re)      "Time RE: YES"   else "Time RE: NO",
    if (cfg$include_spatial_ar)   "Space AR: YES"  else "Space AR: NO",
    if (cfg$include_ar1_temporal) paste0("Time AR1: YES (", cfg$ar1_group, ")") else "Time AR1: NO",
    "Lines: mean probabilities across blocks | Bars: total cases"
  )
  plot_subtitle <- paste(subtitle_parts, collapse = " | ")
  plot_caption  <- "Shaded ribbon: 95% CI for fitted p_bt."

  max_prob     <- max(df_plot_long$probability, na.rm = TRUE)
  max_cases    <- max(df_plot_ts$cases, na.rm = TRUE)
  scale_factor <- ifelse(is.finite(max_cases) && max_cases > 0, max_prob / max_cases, 1)

  ribbon_data <- subset(df_plot_long, series == "p_bt_fitted")
  ribbon_ok   <- nrow(ribbon_data) > 0 &&
    !all(is.na(ribbon_data$lower)) &&
    !all(is.na(ribbon_data$upper))

  p_probs <- ggplot(df_plot_long,
                    aes(x = year_month_date, y = probability, color = series, group = series)) +
    geom_col(data = df_plot_ts,
             aes(x = year_month_date, y = cases * scale_factor),
             inherit.aes = FALSE, fill = "grey75", alpha = 0.5, width = 25)

  if (ribbon_ok) {
    p_probs <- p_probs +
      geom_ribbon(data = ribbon_data,
                  aes(x = year_month_date, ymin = lower, ymax = upper),
                  fill = "#1f77b4", alpha = 0.2, color = NA, inherit.aes = FALSE)
  } else {
    warning("Skipping uncertainty ribbon: missing or invalid CI values.")
  }

  p_probs <- p_probs +
    geom_line(linewidth = 1) +
    geom_point(size = 1.3) +
    scale_color_manual(
      values = c(p_bt_fitted = "#1f77b4", p_observed = "#d62728"),
      labels = c(p_bt_fitted = "Fitted p_bt", p_observed = "Observed y_bt/n_bt")
    ) +
    scale_y_continuous(
      name     = "Probability",
      sec.axis = sec_axis(~ . / scale_factor, name = "Cases")
    ) +
    labs(x = "Time", color = NULL,
         title    = "Observed vs Fitted Detection Rate",
         subtitle = plot_subtitle,
         caption  = plot_caption) +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.caption = element_text(size = 10, hjust = 0))

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
  on.exit(dev.off(), add = TRUE)
  par(mfrow = c(2, 2))

  u_valid <- u_post[!is.na(u_post)]
  if (length(u_valid) > 1) {
    hist(u_valid, breaks = min(50, length(u_valid)), main = "Distribution of Spatial Random Effects (u_b)",
         xlab = "Effect", col = "lightblue", border = "white")
    abline(v = 0, lty = 2, col = "red", lwd = 2)
    qqnorm(u_valid, main = "Q-Q Plot: Spatial Effects", pch = 19, cex = 0.5, col = "blue")
    qqline(u_valid, col = "red", lwd = 2)
  } else {
    plot.new(); text(0.5, 0.5, "Spatial RE disabled\n(block RE not in model)", cex = 1.2)
    plot.new(); text(0.5, 0.5, "Spatial Q-Q unavailable", cex = 1.2)
  }

  if (!all(is.na(v_post)) && length(v_post) > 1) {
    plot(v_post, type = "b", main = "Temporal Random Effects (v_t) with AR(1)",
         xlab = "Time", ylab = "Effect", col = "red", pch = 19)
    abline(h = 0, lty = 2, col = "gray")
    acf(v_post, main = "ACF of Temporal Effects", col = "darkred")
  } else {
    plot.new(); text(0.5, 0.5, "Temporal RE disabled\n(no v_time_out in model)", cex = 1.2)
    plot.new(); text(0.5, 0.5, "ACF unavailable\n(temporal RE disabled)", cex = 1.2)
  }

  par(mfrow = c(1, 1))
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

  trace_dir <- file.path(output_dir, "traceplots")
  dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

  params_main <- c("alpha", "sigma_u", "delta0", "delta1")
  if (use_temporal_re) params_main <- c(params_main, "sigma_v", "rho")

  ggsave(
    file.path(trace_dir, paste0("traceplot_params_", run_suffix, ".png")),
    mcmc_trace(draws_array, pars = params_main), width = 10, height = 8
  )

  w_params <- grep("^w\\[", dimnames(draws_array)[[3]], value = TRUE)
  if (length(w_params) > 0) {
    ggsave(
      file.path(trace_dir, paste0("traceplot_weights_w_", run_suffix, ".png")),
      mcmc_trace(draws_array, pars = w_params), width = 12, height = 10
    )
  }

  wu_params <- grep("^w_unlagged\\[", dimnames(draws_array)[[3]], value = TRUE)
  if (length(wu_params) > 0) {
    ggsave(
      file.path(trace_dir, paste0("traceplot_weights_unlagged_", run_suffix, ".png")),
      mcmc_trace(draws_array, pars = wu_params), width = 12, height = 8
    )
  }
}

#' Save Per-CMF AR(1) State (v_bt) Spaghetti Plot
#'
#' Extracts posterior mean of v_cmf_out[b,t] and plots one line per block over time.
#'
#' @param fit CmdStan fit object
#' @param df Data frame with year_month_date column (used to map time indices to dates)
#' @param stan_data List with B (n blocks) and T (n time points)
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filename
#' @return NULL (saves plot to PNG file)
save_v_bt_plot <- function(fit, df, stan_data, output_dir, run_suffix) {
  draws_v <- tryCatch(fit$draws("v_cmf_out", format = "matrix"), error = function(e) NULL)
  if (is.null(draws_v)) {
    cat("v_cmf_out not found in fit; skipping v_bt plot.\n")
    return(invisible(NULL))
  }

  B <- stan_data$B
  T <- stan_data$T

  # df is lag-filtered: first max_lag time steps are absent, so unique dates < T.
  # Reconstruct the full T-length date vector by extrapolating backward from the
  # earliest surviving date using its time index.
  date_map   <- unique(df[, c("time", "year_month_date")])
  date_map   <- date_map[order(date_map$time), ]
  min_time   <- min(date_map$time)   # = max_lag + 1
  min_date   <- min(date_map$year_month_date)
  time_dates <- seq(
    from       = min_date - months(min_time - 1L),
    by         = "month",
    length.out = T
  )

  make_long <- function(draws_mat, value_name) {
    v_mean <- matrix(colMeans(draws_mat), nrow = B, ncol = T)
    do.call(rbind, lapply(seq_len(B), function(b) {
      data.frame(block = factor(b), year_month_date = time_dates, value = v_mean[b, ])
    }))
  }

  # Top panel: AR(1) states v_bt
  p_top <- ggplot(make_long(draws_v, "v_bt"),
                  aes(x = year_month_date, y = value, group = block, colour = block)) +
    geom_line(alpha = 0.4, linewidth = 0.35) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    labs(title = "Per-CMF AR(1) state v_bt (posterior mean)",
         subtitle = sprintf("B = %d blocks", B),
         x = NULL, y = "v_bt") +
    theme_minimal() +
    theme(legend.position = "none",
          axis.text.x = element_blank(), axis.ticks.x = element_blank())

  # Bottom panel: raw innovations v_raw (parameter, same [b,t] layout)
  draws_raw <- tryCatch(fit$draws("v_raw", format = "matrix"), error = function(e) NULL)

  if (!is.null(draws_raw)) {
    p_bot <- ggplot(make_long(draws_raw, "v_raw"),
                    aes(x = year_month_date, y = value, group = block, colour = block)) +
      geom_line(alpha = 0.4, linewidth = 0.35) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
      labs(title = "Raw innovations v_raw (posterior mean)",
           x = "Time", y = "v_raw") +
      theme_minimal() +
      theme(legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1))

    p_combined <- p_top / p_bot
    ggsave(
      file.path(output_dir, paste0("v_bt_per_block_", run_suffix, ".png")),
      p_combined, width = 12, height = 9, dpi = 150
    )
  } else {
    p_top <- p_top + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    ggsave(
      file.path(output_dir, paste0("v_bt_per_block_", run_suffix, ".png")),
      p_top, width = 12, height = 6, dpi = 150
    )
  }
  cat("v_bt per-block plot saved.\n")
}

#' Save Block Random Effects (u_block) Dot Plot
#'
#' Extracts posterior mean and 90% CI of u_block_out[b] and plots a lollipop chart
#' sorted by posterior mean.
#'
#' @param fit CmdStan fit object
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filename
#' @return NULL (saves plot to PNG file)
save_u_block_plot <- function(fit, output_dir, run_suffix) {
  draws_mat <- tryCatch(fit$draws("u_block_out", format = "matrix"), error = function(e) NULL)
  if (is.null(draws_mat)) {
    cat("u_block_out not found in fit; skipping u_block plot.\n")
    return(invisible(NULL))
  }

  u_df <- data.frame(
    block = seq_len(ncol(draws_mat)),
    u     = colMeans(draws_mat),
    q05   = apply(draws_mat, 2, quantile, 0.05),
    q95   = apply(draws_mat, 2, quantile, 0.95)
  )
  u_df <- u_df[order(u_df$u), ]
  u_df$rank <- seq_len(nrow(u_df))

  p_u <- ggplot(u_df, aes(x = rank, y = u)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    geom_linerange(aes(ymin = q05, ymax = q95), colour = "steelblue", alpha = 0.4, linewidth = 0.5) +
    geom_point(colour = "steelblue", size = 1.2) +
    labs(
      title    = "Block random effects u_block (posterior mean ± 90% CI)",
      subtitle = "Sorted by posterior mean",
      x = "Block (ranked)", y = "u_block"
    ) +
    theme_minimal()

  ggsave(
    file.path(output_dir, paste0("u_block_", run_suffix, ".png")),
    p_u, width = 10, height = 5, dpi = 150
  )
  cat("u_block plot saved.\n")
}

#' Save Spatial RE vs. AR Term Correlation Checks
#'
#' Two mechanistic checks on the "clean" model estimates themselves (not residuals):
#'   a) Moran's I on posterior-mean u_block_out directly -- does the block RE still
#'      have neighbor structure, i.e. is it behaving like a spatial effect? Reported
#'      two ways: (a1) a 50m-annuli correlogram (as for the residual checks, useful
#'      for seeing *at what distance* any structure appears) and (a2) a single pooled
#'      global test over all block pairs at once via k-NN (or inverse-distance)
#'      weights -- not binned, so it isn't diluted/underpowered the way each
#'      individual annulus in (a1) can be.
#'   b) Correlation of each block's time-averaged AR(1) state v_bar_b against its
#'      u_block -- a high correlation is direct evidence that v (AR) is carrying
#'      the same signal u_block would otherwise pick up spatially.
#' All plots are written to a "residual_spatial_correlation_checks" subfolder of
#' output_dir, alongside companion CSVs.
#'
#' @param fit CmdStan fit object
#' @param coords_sf Data frame with columns block_chr, x, y (ordered to match block index)
#' @param stan_data List with B (n blocks) and T (n time points)
#' @param output_dir Character string path to output directory (plots dir)
#' @param run_suffix Character string suffix for filenames
#' @param weight_type "knn" (default) for k-nearest-neighbor weights, or "idw" for
#'   inverse-squared-distance weights, used for the pooled global test in (a2)
#' @param k_neighbors Number of nearest neighbors to use when weight_type = "knn"
#' @return NULL (saves plots + CSVs to PNG/CSV files, or returns invisibly if u_block_out missing)
save_spatial_re_ar_correlation_checks <- function(fit, coords_sf, stan_data, output_dir, run_suffix,
                                                   weight_type = c("knn", "idw"), k_neighbors = 6) {
  weight_type <- match.arg(weight_type)

  draws_u <- tryCatch(fit$draws("u_block_out", format = "matrix"), error = function(e) NULL)
  if (is.null(draws_u)) {
    cat("u_block_out not found in fit; skipping spatial RE correlation checks.\n")
    return(invisible(NULL))
  }

  corr_dir <- file.path(output_dir, "residual_spatial_correlation_checks")
  dir.create(corr_dir, recursive = TRUE, showWarnings = FALSE)

  u_mean <- colMeans(draws_u)

  # --- a1) Moran's I on u_block_out itself: 50m-annuli correlogram -----------
  if (!requireNamespace("spdep", quietly = TRUE)) {
    cat("Skipping Moran's I on u_block: package 'spdep' not installed.\n")
  } else {
    u_df <- coords_sf %>%
      mutate(u_block = u_mean) %>%
      filter(!is.na(x), !is.na(y), !is.na(u_block), is.finite(u_block))

    if (nrow(u_df) < 10) {
      cat("Skipping Moran's I on u_block: fewer than 10 blocks with valid values.\n")
    } else {
      coords_u <- as.matrix(u_df[, c("x", "y")])
      dist_u   <- as.matrix(dist(coords_u))
      diag(dist_u) <- NA_real_

      distance_breaks_u <- seq(0, 2000, by = 50)
      band_list_u <- vector("list", length(distance_breaks_u) - 1)

      for (i in seq_len(length(distance_breaks_u) - 1)) {
        d_low  <- distance_breaks_u[i]
        d_high <- distance_breaks_u[i + 1]
        w <- matrix(0, nrow = nrow(dist_u), ncol = ncol(dist_u))
        w[!is.na(dist_u) & dist_u > d_low & dist_u <= d_high] <- 1
        if (sum(w) == 0) {
          band_list_u[[i]] <- data.frame(
            d_low = d_low, d_high = d_high, d_mid = (d_low + d_high) / 2,
            morans_I = NA_real_, p_value = NA_real_, significant = NA)
          next
        }
        lw <- spdep::mat2listw(w, style = "W", zero.policy = TRUE)
        mt <- tryCatch(
          spdep::moran.test(u_df$u_block, lw, zero.policy = TRUE),
          error   = function(e) NULL,
          warning = function(w) suppressWarnings(
            spdep::moran.test(u_df$u_block, lw, zero.policy = TRUE))
        )
        if (is.null(mt)) {
          band_list_u[[i]] <- data.frame(
            d_low = d_low, d_high = d_high, d_mid = (d_low + d_high) / 2,
            morans_I = NA_real_, p_value = NA_real_, significant = NA)
          next
        }
        band_list_u[[i]] <- data.frame(
          d_low       = d_low, d_high = d_high,
          d_mid       = (d_low + d_high) / 2,
          morans_I    = unname(mt$estimate[["Moran I statistic"]]),
          p_value     = mt$p.value,
          significant = mt$p.value < 0.05)
      }

      moran_u_df <- do.call(rbind, band_list_u) %>% filter(!is.na(morans_I))

      if (nrow(moran_u_df) == 0) {
        cat("Skipping u_block Moran's I correlogram plot: no valid distance bands.\n")
      } else {
        p_moran_u <- ggplot(moran_u_df, aes(x = d_mid, y = morans_I)) +
          geom_hline(yintercept = 0, linetype = "dashed", colour = "gray50") +
          geom_line(linewidth = 0.8, colour = "steelblue") +
          geom_point(aes(shape = significant), size = 2, colour = "steelblue") +
          scale_shape_manual(values  = c("TRUE" = 16, "FALSE" = 1),
                             labels  = c("TRUE" = "p < 0.05", "FALSE" = "p >= 0.05"),
                             na.value = 1) +
          scale_x_continuous(breaks = seq(0, 2000, by = 200)) +
          labs(
            title    = "Moran's I correlogram on posterior-mean u_block (spatial RE itself, not residuals)",
            subtitle = "Neighbor structure in the block random effect, by distance band -- does u_block behave spatially?",
            x = "Distance band midpoint (m)", y = "Moran's I",
            shape = NULL
          ) +
          theme_minimal()

        ggsave(
          file.path(corr_dir, paste0("moransI_u_block_correlogram_", run_suffix, ".png")),
          p_moran_u, width = 9, height = 5, dpi = 150
        )
        write.csv(
          moran_u_df,
          file.path(corr_dir, paste0("moransI_u_block_correlogram_", run_suffix, ".csv")),
          row.names = FALSE
        )
        cat("Moran's I correlogram on u_block saved to:", corr_dir, "\n")
      }

      # --- a2) Single pooled global Moran's I over ALL block pairs at once ----
      # (not binned -- every pair contributes to one properly-powered test)
      min_n <- if (weight_type == "knn") k_neighbors + 1 else 10
      if (nrow(u_df) < min_n) {
        cat(sprintf("Skipping global Moran's I on u_block: fewer than %d blocks with valid values.\n", min_n))
      } else {
        if (weight_type == "knn") {
          knn <- spdep::knearneigh(coords_u, k = k_neighbors)
          nb  <- spdep::knn2nb(knn)
          lw_global <- spdep::nb2listw(nb, style = "W", zero.policy = TRUE)
          weight_desc <- sprintf("k-NN weights (k = %d)", k_neighbors)
        } else {
          w_idw <- 1 / pmax(dist_u, 1e-6)^2
          w_idw[is.na(w_idw)] <- 0
          lw_global <- spdep::mat2listw(w_idw, style = "W", zero.policy = TRUE)
          weight_desc <- "inverse-squared-distance weights"
        }

        mt_global <- tryCatch(
          spdep::moran.test(u_df$u_block, lw_global, zero.policy = TRUE),
          error   = function(e) NULL,
          warning = function(w) suppressWarnings(
            spdep::moran.test(u_df$u_block, lw_global, zero.policy = TRUE))
        )

        if (is.null(mt_global)) {
          cat("Skipping global Moran's I on u_block: moran.test() failed.\n")
        } else {
          I_val <- unname(mt_global$estimate[["Moran I statistic"]])
          p_val <- mt_global$p.value

          u_lag <- spdep::lag.listw(lw_global, u_df$u_block, zero.policy = TRUE)
          moran_scatter_df <- data.frame(block_chr = u_df$block_chr, u_block = u_df$u_block, u_lag = u_lag)

          p_moran_global <- ggplot(moran_scatter_df, aes(x = u_block, y = u_lag)) +
            geom_hline(yintercept = mean(u_lag), linetype = "dotted", colour = "grey60") +
            geom_vline(xintercept = mean(u_df$u_block), linetype = "dotted", colour = "grey60") +
            geom_point(colour = "steelblue", alpha = 0.7) +
            geom_smooth(method = "lm", formula = y ~ x, colour = "darkred", se = TRUE) +
            labs(
              title    = "Global Moran's I on posterior-mean u_block (all pairs pooled, spatial RE itself)",
              subtitle = sprintf("I = %.3f, p = %.3g -- %s, n = %d blocks (single pooled test, not binned)",
                                  I_val, p_val, weight_desc, nrow(u_df)),
              caption  = "Shaded band: 95% CI of the linear fit",
              x = "u_block (posterior mean)",
              y = "Spatial lag of u_block (neighbor-weighted mean)"
            ) +
            theme_minimal()

          ggsave(
            file.path(corr_dir, paste0("moransI_u_block_global_", run_suffix, ".png")),
            p_moran_global, width = 7, height = 6, dpi = 150
          )

          moran_global_df <- data.frame(
            n = nrow(u_df), weight_type = weight_type,
            k_neighbors = if (weight_type == "knn") k_neighbors else NA_integer_,
            morans_I = I_val,
            expectation = unname(mt_global$estimate[["Expectation"]]),
            variance = unname(mt_global$estimate[["Variance"]]),
            p_value = p_val,
            significant = p_val < 0.05
          )
          write.csv(
            moran_global_df,
            file.path(corr_dir, paste0("moransI_u_block_global_", run_suffix, ".csv")),
            row.names = FALSE
          )
          cat(sprintf("Global Moran's I on u_block: I = %.3f, p = %.3g (%s, n = %d) -- saved to %s\n",
                      I_val, p_val, weight_desc, nrow(u_df), corr_dir))
        }
      }
    }
  }

  # --- b) Correlate time-averaged v_bar_b against u_block --------------------
  draws_v <- tryCatch(fit$draws("v_cmf_out", format = "matrix"), error = function(e) NULL)
  if (is.null(draws_v)) {
    cat("v_cmf_out not found in fit; skipping u_block vs v_bar correlation.\n")
    return(invisible(NULL))
  }

  B <- stan_data$B
  T <- stan_data$T
  v_mean_mat <- matrix(colMeans(draws_v), nrow = B, ncol = T)
  v_bar      <- rowMeans(v_mean_mat)

  uv_df <- data.frame(block = seq_len(B), u_block = u_mean, v_bar = v_bar) %>%
    filter(is.finite(u_block), is.finite(v_bar))

  if (nrow(uv_df) < 3) {
    cat("Skipping u_block vs v_bar correlation: fewer than 3 blocks with valid values.\n")
    return(invisible(NULL))
  }

  ct <- cor.test(uv_df$u_block, uv_df$v_bar)
  r  <- unname(ct$estimate)
  p  <- ct$p.value

  p_uv <- ggplot(uv_df, aes(x = u_block, y = v_bar)) +
    geom_point(colour = "steelblue", alpha = 0.7) +
    geom_smooth(method = "lm", formula = y ~ x, colour = "darkred", se = TRUE) +
    labs(
      title    = "u_block vs. time-averaged AR state: is the AR term absorbing the spatial signal?",
      subtitle = sprintf("Pearson r = %.3f, p = %.3g -- high |r| = AR(1) mechanistically carries the spatial signal",
                          r, p),
      caption  = "Shaded band: 95% CI of the linear fit",
      x = "u_block (posterior mean, spatial RE)",
      y = expression(bar(v)[b]~"(time-averaged AR(1) state)")
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 11))

  ggsave(
    file.path(corr_dir, paste0("u_block_vs_v_bar_correlation_", run_suffix, ".png")),
    p_uv, width = 8, height = 6, dpi = 150
  )
  write.csv(
    uv_df,
    file.path(corr_dir, paste0("u_block_vs_v_bar_correlation_", run_suffix, ".csv")),
    row.names = FALSE
  )
  cat(sprintf("u_block vs v_bar correlation saved to: %s (r = %.3f, p = %.3g)\n",
              corr_dir, r, p))
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
          summarise(
            observed_mean = mean(observed_p_bt, na.rm = TRUE),
            total_cases   = sum(C_bt,           na.rm = TRUE),
            .groups = "drop"
          )

        plot_df <- data.frame(
          year_month_date = time_points,
          p_mean          = p_summ$mean,
          p_lower         = p_summ$lower,
          p_upper         = p_summ$upper,
          observed_mean   = obs_summary$observed_mean,
          total_cases     = obs_summary$total_cases
        )
        if (!is.null(pred_summ)) {
          plot_df$pred_mean  <- pred_summ$mean
          plot_df$pred_lower <- pred_summ$lower
          plot_df$pred_upper <- pred_summ$upper
        }

        # Scale factor to map cases onto the left (probability) axis
        left_max  <- max(c(plot_df$p_upper, plot_df$pred_upper,
                           plot_df$observed_mean), na.rm = TRUE)
        cases_max <- max(plot_df$total_cases, na.rm = TRUE)
        c_scale   <- if (cases_max > 0) left_max / cases_max else 1

        p1 <- ggplot(plot_df, aes(x = year_month_date)) +
          geom_bar(aes(y = total_cases * c_scale), stat = "identity",
                   fill = "grey70", alpha = 0.5) +
          geom_ribbon(aes(ymin = p_lower, ymax = p_upper), fill = "blue", alpha = 0.18) +
          geom_line(aes(y = p_mean,        color = "Fitted p_bt"),   linewidth = 1) +
          geom_point(aes(y = p_mean,       color = "Fitted p_bt"),   size = 2) +
          geom_line(aes(y = observed_mean, color = "Observed y/n"),  linewidth = 1) +
          geom_point(aes(y = observed_mean, color = "Observed y/n"), size = 2)

        if (!is.null(pred_summ))
          p1 <- p1 +
            geom_ribbon(aes(ymin = pred_lower, ymax = pred_upper), fill = "#E69F00", alpha = 0.2) +
            geom_line(aes(y = pred_mean,  color = "Predicted y_pred/n"), linewidth = 1, linetype = "dashed") +
            geom_point(aes(y = pred_mean, color = "Predicted y_pred/n"), size = 2)

        p1 <- p1 +
          scale_y_continuous(
            name     = "Probability / Rate",
            sec.axis = sec_axis(~ . / c_scale, name = "Total dengue cases (municipality)")
          ) +
          scale_color_manual(
            values = c("Fitted p_bt"       = "blue",
                       "Observed y/n"       = "red",
                       "Predicted y_pred/n" = "#E69F00"),
            breaks = c("Observed y/n", "Predicted y_pred/n", "Fitted p_bt")
          ) +
          labs(x = "Time",
               title = "Time Series: observed rate, predicted rate, and fitted p_bt (mean across blocks)",
               color = NULL,
               caption = "Shaded ribbons: 95% CI for p_bt (blue) and y_pred/n_bt (orange). Grey bars: total dengue cases.") +
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

  # Scale factor for cases: map max cases to max predicted rate across selected blocks
  left_max_b  <- max(c(p2_df$y_pred_rate_q95, p2_df$observed_p_bt, p2_df$fitted_p_bt), na.rm = TRUE)
  cases_max_b <- max(p2_df$C_bt, na.rm = TRUE)
  c_scale_b   <- if (cases_max_b > 0) left_max_b / cases_max_b else 1

  p2 <- ggplot(p2_df, aes(x = year_month_date)) +
    geom_bar(aes(y = C_bt * c_scale_b), stat = "identity", fill = "grey70", alpha = 0.5) +
    geom_ribbon(aes(ymin = y_pred_rate_q05, ymax = y_pred_rate_q95), fill = "#E69F00", alpha = 0.2) +
    geom_line(aes(y = y_pred_rate,   color = "Predicted rate (y_pred/n)"), alpha = 0.8, linewidth = 0.7) +
    geom_line(aes(y = fitted_p_bt,   color = "Fitted p_bt"),               alpha = 0.8, linewidth = 0.7, linetype = "dashed") +
    geom_line(aes(y = observed_p_bt, color = "Observed rate (y/n)"),       alpha = 0.8, linewidth = 0.6) +
    geom_point(aes(y = observed_p_bt, color = "Observed rate (y/n)"),      size = 0.8, alpha = 0.7) +
    facet_wrap(~block, ncol = 3) +
    scale_y_continuous(
      name     = "Detection rate / Probability",
      sec.axis = sec_axis(~ . / c_scale_b, name = "Dengue cases (block)")
    ) +
    scale_color_manual(values = c(
      "Predicted rate (y_pred/n)" = "#E69F00",
      "Fitted p_bt"               = "blue",
      "Observed rate (y/n)"       = "red"
    )) +
    labs(x = "Time",
         title = "Time Series by Block: observed rate, predicted rate, and fitted p_bt",
         color = NULL,
         caption = "Orange ribbon: 90% CI of y_pred/n. Grey bars: dengue cases per block (right axis).") +
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

#' Exponentiate a dlnm crosspred/crossreduce object's effect estimates
#'
#' Converts a fitted crosspred()/crossreduce() object's log-odds-scale fit/CI
#' fields to the odds-ratio scale (exp()), leaving everything else (predvar,
#' lag structure, the model.link="identity" marker, etc.) untouched. Because
#' model.link stays "identity", the object can still be passed straight to
#' dlnm's own plot.crosspred()/plot.crossreduce() methods afterward without
#' those methods applying any further transformation of their own -- they'll
#' just plot whatever numbers are already sitting in the object.
#'
#' Given this project's genuinely rare outcome (pooled positivity ~0.4%), the
#' resulting odds ratio is also a reasonable approximation to a risk ratio --
#' the same rare-outcome justification already used for the attributable-
#' fraction functions (see compute_af_posterior()'s documentation).
#'
#' @param obj A crosspred or crossreduce object (from dlnm::crosspred()/crossreduce())
#' @return The same object, with all fit/low/high (and mat-prefixed) fields
#'   exponentiated
exp_crosspred <- function(obj) {
  for (fld in c("allfit", "alllow", "allhigh", "matfit", "matlow", "mathigh",
                "fit", "low", "high")) {
    if (!is.null(obj[[fld]])) obj[[fld]] <- exp(obj[[fld]])
  }
  obj
}

#' Save Unlagged-Variable Effects Forest Plot
#'
#' One point + whisker per unlagged predictor (w_unlagged): posterior mean on
#' the x-axis, variable name on the y-axis, whiskers = 95% credible interval
#' computed directly from the posterior draws (2.5%/97.5% quantiles) -- not
#' from a saved model_summary.txt, whose default quantile columns (q5/q95)
#' are a 90% interval. For plotting from an already-saved summary.txt instead
#' of a live fit object, see the standalone snippet in the project notes
#' rather than this function.
#'
#' scale = "logodds" (default, alongside "OR"; see below) plots w_unlagged
#' as-is -- this is the model's actual linear-predictor scale (see
#' hierarchical_state_space_*.stan: eta includes X_unlagged * w_unlagged,
#' then p_bt = inv_logit(eta)), NOT an odds ratio, unlike the DLNM lagged-
#' variable plots (save_dlnm_response_plots() etc.), which do exponentiate
#' via exp_crosspred(). scale = "OR" exponentiates for a like-for-like
#' comparison with those. Mirrors save_glmm_coef_forest_plot()'s
#' scale = c("logodds", "OR") pattern elsewhere in this file.
#'
#' Draws are transformed (exp()) BEFORE summarising when scale = "OR", so
#' mean/CI come directly from the OR-scale draws rather than from
#' exponentiating an already-computed log-odds mean -- exp() commutes
#' exactly with quantiles/medians but not with the mean (Jensen's
#' inequality), so this avoids a biased OR-scale point estimate.
#'
#' @param fit     CmdStanR fit object (must have a w_unlagged parameter)
#' @param prep    Return value of build_stan_data()/build_dlnm_stan_data()
#'   (uses prep$unlagged_vars for the w_unlagged[i] -> variable name mapping)
#' @param output_dir  Directory to write the PNG into
#' @param run_suffix  String appended to the filename
#' @param scale   "logodds", "OR", or both (default) -- one plot per value
save_unlagged_effects_plot <- function(fit, prep, output_dir, run_suffix,
                                       scale = c("logodds", "OR")) {
  scale <- match.arg(scale, several.ok = TRUE)
  if (length(scale) > 1) {
    for (s in scale) save_unlagged_effects_plot(fit, prep, output_dir, run_suffix, scale = s)
    return(invisible(NULL))
  }

  unlagged_vars <- prep$unlagged_vars
  if (is.null(unlagged_vars) || length(unlagged_vars) == 0) {
    cat("No unlagged variables in this model; skipping unlagged effects plot.\n")
    return(invisible(NULL))
  }

  draws <- fit$draws("w_unlagged", format = "matrix")
  idx   <- as.integer(sub("^w_unlagged\\[([0-9]+)\\]$", "\\1", colnames(draws)))
  colnames(draws) <- unlagged_vars[idx]

  if (scale == "OR") {
    draws    <- exp(draws)
    x_ref    <- 1
    x_label  <- "Odds ratio"
    subtitle <- "Posterior mean ± 95% credible interval (OR scale)"
    x_scale  <- ggplot2::scale_x_log10()
    file_tag <- "OR"
  } else {
    x_ref    <- 0
    x_label  <- "β coefficient (log-odds scale)"
    subtitle <- "Posterior mean ± 95% credible interval"
    x_scale  <- ggplot2::scale_x_continuous()
    file_tag <- "logodds"
  }

  df_plot <- data.frame(
    variable = colnames(draws),
    mean     = apply(draws, 2, mean),
    ci_low   = apply(draws, 2, quantile, probs = 0.025),
    ci_high  = apply(draws, 2, quantile, probs = 0.975)
  )
  # Order by effect size (largest positive at top) rather than alphabetically
  df_plot$variable <- factor(df_plot$variable,
                             levels = df_plot$variable[order(df_plot$mean)])

  p <- ggplot2::ggplot(df_plot, ggplot2::aes(x = mean, y = variable)) +
    ggplot2::geom_vline(xintercept = x_ref, linetype = "dashed",
                        colour = "gray40", linewidth = 0.5) +
    ggplot2::geom_errorbar(
      ggplot2::aes(xmin = ci_low, xmax = ci_high),
      width = 0.2, linewidth = 0.55, colour = "steelblue",
      orientation = "y") +
    ggplot2::geom_point(size = 2.5, colour = "steelblue4") +
    x_scale +
    ggplot2::labs(
      title    = "Unlagged variable effects",
      subtitle = subtitle,
      x        = x_label,
      y        = NULL
    ) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold", size = 12))

  out_file <- file.path(output_dir, paste0("unlagged_effects_", file_tag, "_", run_suffix, ".png"))
  ggplot2::ggsave(out_file, p, width = 8,
                  height = max(3, 0.4 * nrow(df_plot) + 1.5), dpi = 150)
  cat("Unlagged effects plot (", scale, ") saved to:", out_file, "\n")
  invisible(p)
}

#' Save DLNM Exposure-Response and Lag-Response Plots
#'
#' For each DLNM predictor, recovers the bivariate exposure-lag-response surface
#' from posterior draws of w_cb using dlnm::crosspred(), then saves:
#'   - overall cumulative effect (marginalised over all lags)
#'   - 3-D surface (effect by predictor value and lag)
#'
#' Predictor-to-column mapping is derived from the per-predictor crossbasis
#' column counts stored in prep$cb_mats, so it is robust to different df
#' settings across predictors.
#'
# Diverging colour scale shared by every DLNM OR plot below: "heatmap2" from
# the ltc-color-palettes explorer (https://loukesio.github.io/ltc-color-palettes/
# palette-explorer.html), a ColorBrewer RdBu-5 palette. Low->high = blue->white->red;
# individual call sites reverse it where their existing convention runs the other way.
dlnm_diverging_pal <- c("#2c7bb6", "#abd9e9", "#ffffff", "#fdae61", "#d7191c")
#' @param fit     CmdStanR fit object
#' @param prep    Return value of build_dlnm_stan_data() (contains cb_mats, dlnm_vars, df)
#' @param output_dir  Directory to write PNGs into
#' @param run_suffix  String appended to each filename
save_dlnm_response_plots <- function(fit, prep, output_dir, run_suffix) {
  if (!requireNamespace("dlnm", quietly = TRUE)) {
    cat("dlnm not installed; skipping DLNM response plots.\n")
    return(invisible(NULL))
  }

  # Output categories: 2D cumulative + 3D surface together ("overall"), vs.
  # the per-lag exposure-response slices in their own folder.
  dir_overall <- file.path(output_dir, "overall_exposure_response")
  dir_perlag  <- file.path(output_dir, "exposure_response_per_lag")
  dir.create(dir_overall, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir_perlag,  recursive = TRUE, showWarnings = FALSE)

  cb_mats        <- prep$cb_mats
  dlnm_vars      <- prep$dlnm_vars
  df             <- prep$df
  dlnm_var_stats <- prep$dlnm_var_stats   # list(var = list(mean, sd)), NULL if absent

  cb_ncols   <- sapply(dlnm_vars, function(v) ncol(cb_mats[[v]]))
  col_starts <- cumsum(c(1L, cb_ncols[-length(cb_ncols)]))

  w_cb_draws <- fit$draws("w_cb", format = "matrix")

  # ── Pass 1: compute all crosspred objects and grids ───────────────────────
  preds <- vector("list", length(dlnm_vars))
  names(preds) <- dlnm_vars

  for (i in seq_along(dlnm_vars)) {
    var  <- dlnm_vars[i]
    cols <- col_starts[i] + seq_len(cb_ncols[i]) - 1L

    if (!var %in% names(df)) {
      cat(sprintf("  Skipping %s: column not found in prep$df\n", var))
      next
    }

    stats_i <- if (!is.null(dlnm_var_stats) && var %in% names(dlnm_var_stats))
      dlnm_var_stats[[var]] else list(mean = 0, sd = 1)
    v_mean <- stats_i$mean
    v_sd   <- stats_i$sd

    x_orig_range <- range(df[[var]], na.rm = TRUE) * v_sd + v_mean
    cat(sprintf("  [%s] original range: [%.3f, %.3f]  cen=median=%.3f\n",
                var, x_orig_range[1], x_orig_range[2], v_mean))
    at_orig_nice <- pretty(x_orig_range, n = 40)
    at_std_nice  <- (at_orig_nice - v_mean) / v_sd

    obs_range <- range(df[[var]][is.finite(df[[var]])])
    keep      <- at_std_nice >= obs_range[1] & at_std_nice <= obs_range[2]
    at_std    <- at_std_nice[keep]
    at_orig   <- at_orig_nice[keep]

    draws_i     <- w_cb_draws[, cols, drop = FALSE]
    cb_colnames <- colnames(cb_mats[[var]])
    coef_i      <- setNames(colMeans(draws_i), cb_colnames)
    vcov_i      <- cov(draws_i)
    dimnames(vcov_i) <- list(cb_colnames, cb_colnames)

    pred_i <- tryCatch(
      exp_crosspred(dlnm::crosspred(cb_mats[[var]], coef = coef_i, vcov = vcov_i,
                      at = at_std, cen = 0, cumul = TRUE)),
      error = function(e) {
        cat(sprintf("  crosspred failed for %s: %s\n", var, conditionMessage(e)))
        NULL
      }
    )
    if (is.null(pred_i)) next

    # Second crosspred at a finer lag resolution, used only by the heatmap
    # below -- the ns() lag basis is continuous, so predicting at fractional
    # lags (rather than the 3-D surface's/slices' integer 0:L_val) gives the
    # smooth Lowe-et-al.-style transitions along the lag axis without
    # changing the resolution of any existing plot.
    L_val_i     <- as.integer(attr(cb_mats[[var]], "lag")[2])
    bylag_fine  <- max(L_val_i / 30, 0.05)
    pred_i_fine <- tryCatch(
      exp_crosspred(dlnm::crosspred(cb_mats[[var]], coef = coef_i, vcov = vcov_i,
                      at = at_std, cen = 0, cumul = TRUE, bylag = bylag_fine)),
      error = function(e) {
        cat(sprintf("  fine-lag crosspred failed for %s: %s\n", var, conditionMessage(e)))
        NULL
      }
    )

    preds[[var]] <- list(pred = pred_i, pred_fine = pred_i_fine,
                         at_std = at_std, at_orig = at_orig,
                         v_mean = v_mean, v_sd = v_sd,
                         L_val = L_val_i)
  }

  # ── Global z-range for comparable 3-D axes and colour scale ───────────────
  # One shared range across all predictors (so surfaces are visually
  # comparable), driven purely by the data -- NOT padded or widened to force
  # symmetry around 1. White is still anchored exactly at OR = 1 by splitting
  # the palette into two independent ramps that meet at 1 (dlnm_diverging_pal's
  # blue half below, red half above), each sized in proportion to how much of
  # the (possibly asymmetric) pooled range falls on that side -- rather than
  # by stretching the range itself to make both sides equal.
  pal_below <- dlnm_diverging_pal[1:3]  # blue -> light blue -> white
  pal_above <- dlnm_diverging_pal[3:5]  # white -> light red -> red

  all_z    <- unlist(lapply(preds, function(p) if (!is.null(p)) p$pred$matfit))
  z_global <- range(all_z, na.rm = TRUE)

  if (z_global[1] < 1 && z_global[2] > 1) {
    n_below <- max(1, round(50 * (1 - z_global[1]) / diff(z_global)))
    n_above <- max(1, 50 - n_below)
    pal <- c(colorRampPalette(pal_below)(n_below),
             colorRampPalette(pal_above)(n_above))
    z_breaks_global <- c(seq(z_global[1], 1, length.out = n_below + 1),
                          seq(1, z_global[2], length.out = n_above + 1)[-1])
  } else if (z_global[2] <= 1) {
    # Pooled range never reaches above 1: single blue ramp only.
    pal <- colorRampPalette(pal_below)(50)
    z_breaks_global <- seq(z_global[1], z_global[2], length.out = 51)
  } else {
    # Pooled range never dips below 1: single red ramp only.
    pal <- colorRampPalette(pal_above)(50)
    z_breaks_global <- seq(z_global[1], z_global[2], length.out = 51)
  }

  # Log-space companions of the above, for the log-scale 3-D/heatmap variants
  # below -- same proportional below/above split as the linear version, just
  # computed in log units so it meets exactly at log(OR) = 0 instead of OR = 1.
  log_all_z    <- log(all_z[all_z > 0])
  log_z_global <- range(log_all_z, na.rm = TRUE)

  if (log_z_global[1] < 0 && log_z_global[2] > 0) {
    n_below_log <- max(1, round(50 * (0 - log_z_global[1]) / diff(log_z_global)))
    n_above_log <- max(1, 50 - n_below_log)
    pal_log <- c(colorRampPalette(pal_below)(n_below_log),
                 colorRampPalette(pal_above)(n_above_log))
    log_z_breaks_global <- c(seq(log_z_global[1], 0, length.out = n_below_log + 1),
                              seq(0, log_z_global[2], length.out = n_above_log + 1)[-1])
  } else if (log_z_global[2] <= 0) {
    pal_log <- colorRampPalette(pal_below)(50)
    log_z_breaks_global <- seq(log_z_global[1], log_z_global[2], length.out = 51)
  } else {
    pal_log <- colorRampPalette(pal_above)(50)
    log_z_breaks_global <- seq(log_z_global[1], log_z_global[2], length.out = 51)
  }

  # ── Pass 2: plot ──────────────────────────────────────────────────────────
  for (i in seq_along(dlnm_vars)) {
    var <- dlnm_vars[i]
    if (is.null(preds[[var]])) next

    pred_i  <- preds[[var]]$pred
    at_std  <- preds[[var]]$at_std
    at_orig <- preds[[var]]$at_orig
    L_val   <- preds[[var]]$L_val
    lag_seq <- 0:L_val

    # ── Overall cumulative effect (original x-axis) ───────────────────────────
    png(file.path(dir_overall, paste0("dlnm_overall_", var, "_", run_suffix, ".png")),
        width = 800, height = 500)
    plot(pred_i, "overall",
         xaxt   = "n",
         main   = paste("Cumulative effect —", var),
         sub    = "Shaded band: 95% CI",
         xlab   = var,
         ylab   = "Odds ratio of p_bt",
         col    = "steelblue",
         ci.arg = list(col = adjustcolor("steelblue", 0.25), border = NA))
    axis(1, at = at_std, labels = round(at_orig, 2))
    abline(h = 1, lty = 2, col = "grey50")
    dev.off()

    # ── Overall cumulative effect, log-y-axis version ─────────────────────────
    # Equal-magnitude effects in opposite directions (e.g. OR=2 vs OR=0.5) are
    # equidistant from 1 on a log scale but wildly asymmetric on the linear OR
    # scale above -- this makes the two tails directly, visually comparable
    # instead of the high side dwarfing the low side just from the scale.
    png(file.path(dir_overall, paste0("dlnm_overall_", var, "_logscale_", run_suffix, ".png")),
        width = 800, height = 500)
    or_range <- range(c(pred_i$alllow, pred_i$allhigh), na.rm = TRUE)
    plot(at_std, pred_i$allfit, type = "n", log = "y",
         xaxt = "n", ylim = or_range,
         main = paste("Cumulative effect (log scale) —", var),
         sub  = "Shaded band: 95% CI",
         xlab = var, ylab = "Odds ratio of p_bt (log scale)")
    polygon(c(at_std, rev(at_std)), c(pred_i$alllow, rev(pred_i$allhigh)),
            col = adjustcolor("steelblue", 0.25), border = NA)
    lines(at_std, pred_i$allfit, col = "steelblue", lwd = 2)
    axis(1, at = at_std, labels = round(at_orig, 2))
    abline(h = 1, lty = 2, col = "grey50")
    dev.off()

    # ── 3-D surface (original x-axis, shared z-scale across predictors) ──────
    z_mat  <- pred_i$matfit
    z_mid  <- (z_mat[-1, -1] + z_mat[-1, -ncol(z_mat)] +
               z_mat[-nrow(z_mat), -1] + z_mat[-nrow(z_mat), -ncol(z_mat)]) / 4
    facet_col <- pal[cut(z_mid, breaks = z_breaks_global, include.lowest = TRUE)]

    png(file.path(dir_overall, paste0("dlnm_3d_", var, "_", run_suffix, ".png")),
        width = 800, height = 700)
    persp(x        = at_orig,
          y        = lag_seq,
          z        = z_mat,
          zlim     = z_global,
          xlab     = var,
          ylab     = "Lag (months)",
          zlab     = "Odds ratio of p_bt",
          main     = paste("DLNM surface —", var),
          theta    = 40, phi = 25, ltheta = 45,
          col      = facet_col,
          border   = NA,
          ticktype = "detailed",
          cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
    dev.off()

    # ── 2-D heatmap (Lowe et al. 2018-style smooth filled contour) ───────────
    # Same exposure x lag x OR surface as the 3-D plot above, same shared
    # colour scheme (pal/z_breaks_global, white anchored at OR = 1), but
    # rendered as a smooth filled contour (exposure on x, lag on y) rather
    # than a 3-D surface, using the finer-lag crosspred computed in Pass 1
    # so transitions along the lag axis aren't blocky at only L_val+1 points.
    pred_fine_i <- preds[[var]]$pred_fine
    if (!is.null(pred_fine_i)) {
      z_mat_fine    <- pred_fine_i$matfit
      fine_lag_seq  <- seq(pred_fine_i$lag[1], pred_fine_i$lag[2], by = pred_fine_i$bylag)
      # z_breaks_global's range comes from the coarse integer-lag grid
      # (pred_i$matfit); the finer interpolated grid plotted here can dip
      # past that range at a fractional lag the coarse grid never sampled,
      # leaving filled.contour() with no bin for that cell -- rendered as an
      # unpainted white gap. Clip into range before plotting: those cells are
      # already in the most extreme colour bin, so clipping is visually
      # indistinguishable from widening the scale, just without the gap.
      z_mat_fine_clip <- pmin(pmax(z_mat_fine, z_breaks_global[1]),
                               z_breaks_global[length(z_breaks_global)])

      png(file.path(dir_overall, paste0("dlnm_heatmap_", var, "_", run_suffix, ".png")),
          width = 800, height = 600)
      filled.contour(
        x      = at_orig,
        y      = fine_lag_seq,
        z      = z_mat_fine_clip,
        levels = z_breaks_global,
        col    = pal,
        xlab   = var,
        ylab   = "Lag (months)",
        main   = paste("DLNM heatmap —", var),
        cex.lab = 1.5, cex.main = 1.5,
        key.title = title(main = "OR", cex.main = 1.35),
        plot.axes = {
          axis(1, cex.axis = 1.5)
          axis(2, cex.axis = 1.5)
          contour(at_orig, fine_lag_seq, z_mat_fine, levels = 1,
                  add = TRUE, col = "black", lty = 2, lwd = 1, drawlabels = FALSE)
        }
      )
      dev.off()
    } else {
      cat(sprintf("  Skipping heatmap for %s: fine-lag crosspred unavailable\n", var))
    }

    # ── 3-D surface and heatmap, log-y-axis versions ──────────────────────────
    # log() taken per-cell before averaging corners for facet colour (not
    # log(mean of corners)) -- log() and averaging don't commute (Jensen's
    # inequality), so this avoids the same order-of-operations bias flagged
    # elsewhere in this codebase for back-transformations.
    z_mat_log <- log(z_mat)
    z_mid_log <- (z_mat_log[-1, -1] + z_mat_log[-1, -ncol(z_mat_log)] +
                  z_mat_log[-nrow(z_mat_log), -1] + z_mat_log[-nrow(z_mat_log), -ncol(z_mat_log)]) / 4
    facet_col_log <- pal_log[cut(z_mid_log, breaks = log_z_breaks_global, include.lowest = TRUE)]

    png(file.path(dir_overall, paste0("dlnm_3d_", var, "_logscale_", run_suffix, ".png")),
        width = 800, height = 700)
    persp(x        = at_orig,
          y        = lag_seq,
          z        = z_mat_log,
          zlim     = log_z_global,
          xlab     = var,
          ylab     = "Lag (months)",
          zlab     = "log(Odds ratio) of p_bt",
          main     = paste("DLNM surface (log scale) —", var),
          theta    = 40, phi = 25, ltheta = 45,
          col      = facet_col_log,
          border   = NA,
          ticktype = "detailed",
          cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
    dev.off()

    if (!is.null(pred_fine_i)) {
      # Same coarse-vs-fine range mismatch as the linear heatmap above, in
      # log space -- clip for the same reason.
      z_mat_fine_log <- log(z_mat_fine)
      z_mat_fine_log_clip <- pmin(pmax(z_mat_fine_log, log_z_breaks_global[1]),
                                   log_z_breaks_global[length(log_z_breaks_global)])

      png(file.path(dir_overall, paste0("dlnm_heatmap_", var, "_logscale_", run_suffix, ".png")),
          width = 800, height = 600)
      filled.contour(
        x      = at_orig,
        y      = fine_lag_seq,
        z      = z_mat_fine_log_clip,
        levels = log_z_breaks_global,
        col    = pal_log,
        xlab   = var,
        ylab   = "Lag (months)",
        main   = paste("DLNM heatmap (log scale) —", var),
        cex.lab = 1.5, cex.main = 1.5,
        key.title = title(main = "log(OR)", cex.main = 1.35),
        plot.axes = {
          axis(1, cex.axis = 1.5)
          axis(2, cex.axis = 1.5)
          contour(at_orig, fine_lag_seq, z_mat_fine_log, levels = 0,
                  add = TRUE, col = "black", lty = 2, lwd = 1, drawlabels = FALSE)
        }
      )
      dev.off()
    }

    # ── Per-lag slice plots (one per lag, same style as cumulative) ──────────
    for (l in lag_seq) {
      png(file.path(dir_perlag,
                    paste0("dlnm_lag", l, "_", var, "_", run_suffix, ".png")),
          width = 800, height = 500)
      plot(pred_i, "slices",
           lag    = l,
           xaxt   = "n",
           main   = paste0("Effect at lag ", l, " — ", var),
           sub    = "Shaded band: 95% CI",
           xlab   = var,
           ylab   = "Odds ratio of p_bt",
           col    = "steelblue",
           ci.arg = list(col = adjustcolor("steelblue", 0.25), border = NA))
      axis(1, at = at_std, labels = round(at_orig, 2))
      abline(h = 1, lty = 2, col = "grey50")
      dev.off()
    }

    cat(sprintf("  DLNM plots saved: %s\n", var))
  }

  # Returned so save_dlnm_interaction_response_plots() can fold this range
  # into its own, giving every 3-D surface in the model -- interaction or
  # not -- the same z-axis/colour scale.
  invisible(list(z_global = z_global, log_z_global = log_z_global))
}

#' Save DLNM Lag-Response Plots at Fixed Exposure Percentiles
#'
#' save_dlnm_response_plots() slices the DLNM surface by lag (exposure-response
#' curve at each fixed lag), via dlnm's plot(cp, "slices", lag = l). This is
#' the transpose view: lag-response curve (effect vs. lag) at each of a chosen
#' set of fixed exposure percentiles — what identifies *which lags* have a
#' credible interval excluding zero at a given exposure level, i.e. the
#' empirical basis for a "critical window" claim (e.g. "precip effect peaks
#' at lag 2, CI excludes 0 from lag 1-3").
#'
#' Uses dlnm::crossreduce(basis, type="var", value=X) rather than crosspred()
#' + plot(,"slices",var=X): crosspred's var= slicing requires an exact match
#' against its `at` prediction grid (no interpolation), so it would need the
#' target percentile folded into that grid first. crossreduce() computes the
#' reduced 1-D (lag-only) association directly at any value, with no grid
#' required — see the dlnm package's own "dlnmTS" vignette (Figure 5b) for
#' this exact use case (predictor-specific lag-response at a chosen value).
#'
#' @param fit     CmdStanR fit object (must have w_cb parameter)
#' @param prep    Return value of build_dlnm_stan_data()
#' @param output_dir   Directory to write PNGs/CSV into
#' @param run_suffix   String appended to each filename
#' @param percentiles  Numeric vector in (0,1): exposure percentiles to slice at
#' @return Invisibly, a data frame (variable, percentile, exposure_value, lag,
#'   estimate, ci_low, ci_high, significant) — estimate/ci_low/ci_high are on
#'   the odds-ratio scale (via exp_crosspred()); significant = CI excludes 1.
#'   Also written to <output_dir>/dlnm_lagresponse_critical_windows_<run_suffix>.csv
save_dlnm_lagresponse_plots <- function(fit, prep, output_dir, run_suffix,
                                         percentiles = c(0.10, 0.25, 0.50, 0.75, 0.90)) {
  if (!requireNamespace("dlnm", quietly = TRUE)) {
    cat("dlnm not installed; skipping DLNM lag-response plots.\n")
    return(invisible(NULL))
  }

  # Output category: lag-response curves at fixed exposure percentiles,
  # plus their numeric critical-window companion CSV.
  dir_lagresp <- file.path(output_dir, "lag_response_per_exposure")
  dir.create(dir_lagresp, recursive = TRUE, showWarnings = FALSE)

  cb_mats        <- prep$cb_mats
  dlnm_vars      <- prep$dlnm_vars
  df             <- prep$df
  dlnm_var_stats <- prep$dlnm_var_stats

  cb_ncols   <- sapply(dlnm_vars, function(v) ncol(cb_mats[[v]]))
  col_starts <- cumsum(c(1L, cb_ncols[-length(cb_ncols)]))

  w_cb_draws <- fit$draws("w_cb", format = "matrix")

  summary_rows <- list()

  for (i in seq_along(dlnm_vars)) {
    var  <- dlnm_vars[i]
    cols <- col_starts[i] + seq_len(cb_ncols[i]) - 1L

    if (!var %in% names(df)) {
      cat(sprintf("  Skipping %s: column not found in prep$df\n", var))
      next
    }

    stats_i <- if (!is.null(dlnm_var_stats) && var %in% names(dlnm_var_stats))
      dlnm_var_stats[[var]] else list(mean = 0, sd = 1)
    v_mean <- stats_i$mean
    v_sd   <- stats_i$sd

    # Target percentiles, computed on the original (back-transformed) scale,
    # then converted to the standardized scale the cross-basis was built on
    # (crossreduce's value= must be given in the basis's own coding).
    x_orig_obs <- df[[var]][is.finite(df[[var]])] * v_sd + v_mean
    perc_orig  <- as.numeric(quantile(x_orig_obs, probs = percentiles, na.rm = TRUE))
    perc_std   <- (perc_orig - v_mean) / v_sd

    draws_i     <- w_cb_draws[, cols, drop = FALSE]
    cb_colnames <- colnames(cb_mats[[var]])
    coef_i      <- setNames(colMeans(draws_i), cb_colnames)
    vcov_i      <- cov(draws_i)
    dimnames(vcov_i) <- list(cb_colnames, cb_colnames)

    L_val   <- as.integer(attr(cb_mats[[var]], "lag")[2])
    lag_seq <- 0:L_val

    for (p_idx in seq_along(percentiles)) {
      p_val   <- percentiles[p_idx]
      std_val <- perc_std[p_idx]
      orig_val <- perc_orig[p_idx]

      red_i <- tryCatch(
        # model.link must be an explicit non-NULL string here: crossreduce()'s
        # source does `if (model.link %in% c("log","logit"))` with no NULL
        # guard, and since we pass coef/vcov directly (no `model` object),
        # model.link stays NULL by default -> NULL %in% c(...) is logical(0)
        # -> "argument is of length zero". "identity" routes to the untransformed
        # fit/low/high branch, which is what we want (log-odds scale, no exp()).
        exp_crosspred(dlnm::crossreduce(cb_mats[[var]], coef = coef_i, vcov = vcov_i,
                          model.link = "identity",
                          type = "var", value = std_val,
                          lag = c(0, L_val), bylag = 1, cen = 0)),
        error = function(e) {
          cat(sprintf("  crossreduce failed for %s at p%d: %s\n",
                      var, round(p_val * 100), conditionMessage(e)))
          NULL
        }
      )
      if (is.null(red_i)) next

      est <- as.numeric(red_i$fit)
      lo  <- as.numeric(red_i$low)
      hi  <- as.numeric(red_i$high)
      sig <- (lo > 1 & hi > 1) | (lo < 1 & hi < 1)

      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        variable = var, percentile = p_val, exposure_value = orig_val,
        lag = lag_seq, estimate = est, ci_low = lo, ci_high = hi, significant = sig
      )

      png(file.path(dir_lagresp,
            sprintf("dlnm_lagresponse_p%02d_%s_%s.png", round(p_val * 100), var, run_suffix)),
          width = 800, height = 500)
      plot(red_i,
           main = sprintf("Lag-response - %s at %dth pct (%s = %.2f)",
                          var, round(p_val * 100), var, orig_val),
           sub  = "Shaded band: 95% CI",
           xlab = "Lag (months)",
           ylab = "Odds ratio of p_bt",
           col    = "steelblue",
           ci.arg = list(col = adjustcolor("steelblue", 0.25), border = NA))
      abline(h = 1, lty = 2, col = "grey50")
      dev.off()

      cat(sprintf("  [%s] p%d (%s=%.2f): significant lags = %s\n",
                  var, round(p_val * 100), var, orig_val,
                  if (any(sig)) paste(lag_seq[sig], collapse = ", ") else "none"))
    }

    cat(sprintf("  DLNM lag-response plots saved: %s\n", var))
  }

  result <- if (length(summary_rows) > 0) do.call(rbind, summary_rows) else
    data.frame(variable = character(), percentile = numeric(), exposure_value = numeric(),
               lag = integer(), estimate = numeric(), ci_low = numeric(), ci_high = numeric(),
               significant = logical())

  csv_path <- file.path(dir_lagresp, paste0("dlnm_lagresponse_critical_windows_", run_suffix, ".csv"))
  write_csv(result, csv_path)
  cat(sprintf("  Critical-window summary saved: %s\n", csv_path))

  invisible(result)
}

#' Save DLNM Interaction Response Plots
#'
#' For each interaction specified in prep$dlnm_ix_vars, plots:
#'   - Cumulative effect comparison: reference group (w_cb only) vs active group (w_cb + w_ix)
#'   - Per-lag slice comparison for each lag
#'   - 3-D surface for both groups side-by-side
#' Works for both interaction spec shapes from build_dlnm_stan_data():
#'   binary_var/active_level -> "reference" vs "active" group comparison.
#'   continuous_var (z-scored) -> "reference" is the effect at the modifier's
#'     mean and "active" is the effect at +1 SD (draws_base + w_ix is exactly
#'     the effect at modifier = +1 since the modifier is z-scored), i.e. the
#'     same draws_base/draws_active mechanics, just relabelled.
#'
#' @param fit     CmdStanR fit object (must have w_cb and w_ix parameters)
#' @param prep    Return value of build_dlnm_stan_data() with dlnm_ix_vars field populated
#' @param output_dir  Directory to write PNGs into
#' @param run_suffix  String appended to each filename
save_dlnm_interaction_response_plots <- function(fit, prep, output_dir, run_suffix,
                                                   external_z_range = NULL,
                                                   external_log_z_range = NULL) {
  if (!requireNamespace("dlnm", quietly = TRUE)) {
    cat("dlnm not installed; skipping DLNM interaction plots.\n")
    return(invisible(NULL))
  }
  if (is.null(prep$dlnm_ix_vars) || length(prep$dlnm_ix_vars) == 0) return(invisible(NULL))

  # Same subfolder convention as save_dlnm_response_plots(): overall/cumulative
  # + 3D surfaces together, per-lag slices in their own folder.
  dir_overall <- file.path(output_dir, "interaction_overall_exposure_response")
  dir_perlag  <- file.path(output_dir, "interaction_exposure_response_per_lag")
  dir.create(dir_overall, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir_perlag,  recursive = TRUE, showWarnings = FALSE)

  cb_mats      <- prep$cb_mats
  dlnm_vars    <- prep$dlnm_vars
  dlnm_ix_vars <- prep$dlnm_ix_vars
  df           <- prep$df
  dlnm_var_stats <- prep$dlnm_var_stats

  cb_ncols      <- sapply(dlnm_vars, function(v) ncol(cb_mats[[v]]))
  col_starts_cb <- cumsum(c(1L, cb_ncols[-length(cb_ncols)]))

  w_cb_draws <- fit$draws("w_cb", format = "matrix")
  w_ix_draws <- fit$draws("w_ix", format = "matrix")

  # Column offsets within w_ix: a binary spec occupies cb_ncols[dlnm_var]
  # columns; a continuous spec occupies cb_ncols[dlnm_var] * continuous_df
  # (see build_dlnm_stan_data()) -- must account for that here even though
  # this function only *plots* the binary ones, or offsets for every
  # interaction after a continuous one would be wrong.
  ix_ncols <- sapply(dlnm_ix_vars, function(ix) {
    base_n <- cb_ncols[which(dlnm_vars == ix$dlnm_var)]
    if (!is.null(ix$continuous_var)) base_n * ix$continuous_df else base_n
  })
  ix_col_starts <- cumsum(c(1L, ix_ncols[-length(ix_ncols)]))

  ref_col    <- "steelblue"
  active_col <- "firebrick"

  # ── Pass 1: compute pred_ref/pred_active for every binary interaction, and
  # plot the cumulative + per-lag slice comparisons (unaffected by the
  # z-range unification below) ───────────────────────────────────────────────
  ix_preds <- list()

  for (k in seq_along(dlnm_ix_vars)) {
    ix <- dlnm_ix_vars[[k]]
    if (!is.null(ix$continuous_var)) next  # continuous specs: see save_dlnm_continuous_interaction_plots()
    dlnm_var <- ix$dlnm_var
    label    <- ix$label

    var_idx  <- which(dlnm_vars == dlnm_var)
    n_cols   <- cb_ncols[var_idx]
    cb_cols  <- col_starts_cb[var_idx] + seq_len(n_cols) - 1L
    ix_cols  <- ix_col_starts[k]       + seq_len(n_cols) - 1L

    cb_names <- colnames(cb_mats[[dlnm_var]])

    # Reference group: baseline DLNM effect (w_cb for this variable)
    draws_base   <- w_cb_draws[, cb_cols, drop = FALSE]
    coef_ref     <- setNames(colMeans(draws_base), cb_names)
    vcov_ref     <- cov(draws_base)
    dimnames(vcov_ref) <- list(cb_names, cb_names)

    # Active group: baseline + interaction modifier (w_cb + w_ix), using joint draws
    draws_active  <- draws_base + w_ix_draws[, ix_cols, drop = FALSE]
    coef_active   <- setNames(colMeans(draws_active), cb_names)
    vcov_active   <- cov(draws_active)
    dimnames(vcov_active) <- list(cb_names, cb_names)

    # x-axis back-transformation
    stats_i  <- if (!is.null(dlnm_var_stats) && dlnm_var %in% names(dlnm_var_stats))
      dlnm_var_stats[[dlnm_var]] else list(mean = 0, sd = 1)
    v_mean   <- stats_i$mean
    v_sd     <- stats_i$sd

    x_orig_range <- range(df[[dlnm_var]], na.rm = TRUE) * v_sd + v_mean
    at_orig_nice <- pretty(x_orig_range, n = 40)
    at_std_nice  <- (at_orig_nice - v_mean) / v_sd
    obs_range    <- range(df[[dlnm_var]][is.finite(df[[dlnm_var]])], na.rm = TRUE)
    keep_pts     <- at_std_nice >= obs_range[1] & at_std_nice <= obs_range[2]
    at_std  <- at_std_nice[keep_pts]
    at_orig <- at_orig_nice[keep_pts]

    pred_ref <- tryCatch(
      exp_crosspred(dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_ref, vcov = vcov_ref,
                      at = at_std, cen = 0, cumul = TRUE)),
      error = function(e) { cat(sprintf("  crosspred (ref) failed for %s: %s\n", label, conditionMessage(e))); NULL }
    )
    pred_active <- tryCatch(
      exp_crosspred(dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_active, vcov = vcov_active,
                      at = at_std, cen = 0, cumul = TRUE)),
      error = function(e) { cat(sprintf("  crosspred (active) failed for %s: %s\n", label, conditionMessage(e))); NULL }
    )
    if (is.null(pred_ref) || is.null(pred_active)) next

    L_val   <- as.integer(attr(cb_mats[[dlnm_var]], "lag")[2])
    lag_seq <- 0:L_val

    # Second crosspred per group at a finer lag resolution, used only by the
    # heatmap in Pass 2 -- same technique as save_dlnm_response_plots()'s
    # main-effect heatmap (see that function for the full rationale): the
    # ns() lag basis is continuous, so predicting at fractional lags gives
    # smooth Lowe-et-al.-style transitions along the lag axis instead of the
    # blocky L_val+1-point resolution the 3-D surface/slices use.
    bylag_fine <- max(L_val / 30, 0.05)
    pred_ref_fine <- tryCatch(
      exp_crosspred(dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_ref, vcov = vcov_ref,
                      at = at_std, cen = 0, cumul = TRUE, bylag = bylag_fine)),
      error = function(e) { cat(sprintf("  fine-lag crosspred (ref) failed for %s: %s\n", label, conditionMessage(e))); NULL }
    )
    pred_active_fine <- tryCatch(
      exp_crosspred(dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_active, vcov = vcov_active,
                      at = at_std, cen = 0, cumul = TRUE, bylag = bylag_fine)),
      error = function(e) { cat(sprintf("  fine-lag crosspred (active) failed for %s: %s\n", label, conditionMessage(e))); NULL }
    )

    # ── Cumulative effect comparison ──────────────────────────────────────────
    y_lim <- range(pred_ref$alllow, pred_ref$allhigh,
                   pred_active$alllow, pred_active$allhigh, na.rm = TRUE)
    png(file.path(dir_overall, paste0("dlnm_ix_cumul_", label, "_", run_suffix, ".png")),
        width = 900, height = 500)
    plot(pred_ref, "overall", xaxt = "n", ylim = y_lim,
         main   = paste("Cumulative effect of", dlnm_var, "—", label),
         sub    = "Shaded band and dashed lines: 95% CI",
         xlab   = dlnm_var, ylab = "Odds ratio of p_bt",
         col    = ref_col,
         ci.arg = list(col = adjustcolor(ref_col, 0.20), border = NA))
    lines(at_std, pred_active$allfit, col = active_col, lwd = 2)
    lines(at_std, pred_active$alllow,  col = active_col, lwd = 1, lty = 2)
    lines(at_std, pred_active$allhigh, col = active_col, lwd = 1, lty = 2)
    axis(1, at = at_std, labels = round(at_orig, 2))
    abline(h = 1, lty = 2, col = "grey50")
    # Continuous modifier (z-scored in build_dlnm_stan_data()): draws_base is
    # the effect at the modifier's mean, draws_active = draws_base + w_ix is
    # the effect at +1 SD -- the same two-draws mechanics as the binary case,
    # just relabelled.
    legend_labels <- if (!is.null(ix$continuous_var)) {
      c(sprintf("Reference  (%s at mean)", ix$continuous_var),
        sprintf("+1 SD  (%s)", ix$continuous_var))
    } else {
      c(sprintf("Reference  (active_level ≠ %s)", ix$active_level),
        sprintf("Active group  (%s == %s)", ix$binary_var, ix$active_level))
    }
    legend("topright", legend = legend_labels,
           col = c(ref_col, active_col), lwd = 2, bty = "n")
    dev.off()

    # ── Per-lag slice comparison ──────────────────────────────────────────────
    for (l in lag_seq) {
      y_lim_lag <- range(pred_ref$matlow[, l + 1], pred_ref$mathigh[, l + 1],
                         pred_active$matlow[, l + 1], pred_active$mathigh[, l + 1], na.rm = TRUE)
      png(file.path(dir_perlag, paste0("dlnm_ix_lag", l, "_", label, "_", run_suffix, ".png")),
          width = 900, height = 500)
      plot(pred_ref, "slices", lag = l, xaxt = "n", ylim = y_lim_lag,
           main   = paste0("Effect at lag ", l, " — ", label),
           sub    = "Shaded band and dashed lines: 95% CI",
           xlab   = dlnm_var, ylab = "Odds ratio of p_bt",
           col    = ref_col,
           ci.arg = list(col = adjustcolor(ref_col, 0.20), border = NA))
      lines(at_std, pred_active$matfit[, l + 1], col = active_col, lwd = 2)
      lines(at_std, pred_active$matlow[,  l + 1], col = active_col, lwd = 1, lty = 2)
      lines(at_std, pred_active$mathigh[, l + 1], col = active_col, lwd = 1, lty = 2)
      axis(1, at = at_std, labels = round(at_orig, 2))
      abline(h = 1, lty = 2, col = "grey50")
      dev.off()
    }

    ix_preds[[label]] <- list(dlnm_var = dlnm_var, pred_ref = pred_ref, pred_active = pred_active,
                               pred_ref_fine = pred_ref_fine, pred_active_fine = pred_active_fine,
                               at_std = at_std, at_orig = at_orig, lag_seq = lag_seq)
  }

  if (length(ix_preds) == 0) return(invisible(NULL))

  # ── Global z-range across every interaction's ref+active surfaces, unioned
  # with the non-interaction surfaces' range if provided (external_z_range/
  # external_log_z_range, returned by save_dlnm_response_plots()) -- so every
  # 3-D surface in the model, interaction or not, shares one axis/colour
  # scale. Same blue-low/red-high convention, anchor-at-1 (linear) /
  # anchor-at-0 (log) splitting as save_dlnm_response_plots(). ─────────────
  pal_below <- dlnm_diverging_pal[1:3]  # blue -> light blue -> white
  pal_above <- dlnm_diverging_pal[3:5]  # white -> light red -> red

  all_z_ix <- unlist(lapply(ix_preds, function(p) c(p$pred_ref$matfit, p$pred_active$matfit)))
  z_global <- range(c(all_z_ix, external_z_range), na.rm = TRUE)

  if (z_global[1] < 1 && z_global[2] > 1) {
    n_below <- max(1, round(50 * (1 - z_global[1]) / diff(z_global)))
    n_above <- max(1, 50 - n_below)
    pal <- c(colorRampPalette(pal_below)(n_below),
             colorRampPalette(pal_above)(n_above))
    z_breaks_global <- c(seq(z_global[1], 1, length.out = n_below + 1),
                          seq(1, z_global[2], length.out = n_above + 1)[-1])
  } else if (z_global[2] <= 1) {
    pal <- colorRampPalette(pal_below)(50)
    z_breaks_global <- seq(z_global[1], z_global[2], length.out = 51)
  } else {
    pal <- colorRampPalette(pal_above)(50)
    z_breaks_global <- seq(z_global[1], z_global[2], length.out = 51)
  }

  log_all_z_ix <- log(all_z_ix[all_z_ix > 0])
  log_z_global <- range(c(log_all_z_ix, external_log_z_range), na.rm = TRUE)

  if (log_z_global[1] < 0 && log_z_global[2] > 0) {
    n_below_log <- max(1, round(50 * (0 - log_z_global[1]) / diff(log_z_global)))
    n_above_log <- max(1, 50 - n_below_log)
    pal_log <- c(colorRampPalette(pal_below)(n_below_log),
                 colorRampPalette(pal_above)(n_above_log))
    log_z_breaks_global <- c(seq(log_z_global[1], 0, length.out = n_below_log + 1),
                              seq(0, log_z_global[2], length.out = n_above_log + 1)[-1])
  } else if (log_z_global[2] <= 0) {
    pal_log <- colorRampPalette(pal_below)(50)
    log_z_breaks_global <- seq(log_z_global[1], log_z_global[2], length.out = 51)
  } else {
    pal_log <- colorRampPalette(pal_above)(50)
    log_z_breaks_global <- seq(log_z_global[1], log_z_global[2], length.out = 51)
  }

  # ── Pass 2: plot every interaction's ref/active 3-D surfaces + heatmaps,
  # linear + log ──────────────────────────────────────────────────────────
  for (label in names(ix_preds)) {
    p <- ix_preds[[label]]
    groups <- list(
      list(pred = p$pred_ref,    pred_fine = p$pred_ref_fine,    name = "ref"),
      list(pred = p$pred_active, pred_fine = p$pred_active_fine, name = "active")
    )
    for (grp in groups) {
      z_mat <- grp$pred$matfit
      z_mid <- (z_mat[-1, -1] + z_mat[-1, -ncol(z_mat)] +
                z_mat[-nrow(z_mat), -1] + z_mat[-nrow(z_mat), -ncol(z_mat)]) / 4
      fcol  <- pal[cut(z_mid, breaks = z_breaks_global, include.lowest = TRUE)]

      png(file.path(dir_overall, paste0("dlnm_ix_3d_", label, "_", grp$name, "_", run_suffix, ".png")),
          width = 800, height = 700)
      persp(x = p$at_orig, y = p$lag_seq, z = z_mat,
            zlim     = z_global,
            xlab     = p$dlnm_var, ylab = "Lag (months)", zlab = "Odds ratio of p_bt",
            main     = paste0("DLNM surface — ", label, " (", grp$name, ", pooled)"),
            theta    = 40, phi = 25, ltheta = 45,
            col      = fcol, border = NA, ticktype = "detailed",
            cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
      dev.off()

      # ── 2-D heatmap (Lowe et al. 2018-style smooth filled contour) ────────
      # Same technique as save_dlnm_response_plots()'s main-effect heatmap:
      # finer-lag crosspred for smooth transitions along the lag axis, same
      # shared colour scale as the 3-D surface above (so ref/active/every
      # other DLNM plot in the run stays visually comparable), clipped into
      # range so fractional lags the coarse grid never sampled don't leave
      # unpainted gaps in filled.contour().
      if (!is.null(grp$pred_fine)) {
        z_mat_fine   <- grp$pred_fine$matfit
        fine_lag_seq <- seq(grp$pred_fine$lag[1], grp$pred_fine$lag[2], by = grp$pred_fine$bylag)
        z_mat_fine_clip <- pmin(pmax(z_mat_fine, z_breaks_global[1]),
                                 z_breaks_global[length(z_breaks_global)])

        png(file.path(dir_overall, paste0("dlnm_ix_heatmap_", label, "_", grp$name, "_", run_suffix, ".png")),
            width = 800, height = 600)
        filled.contour(
          x      = p$at_orig,
          y      = fine_lag_seq,
          z      = z_mat_fine_clip,
          levels = z_breaks_global,
          col    = pal,
          xlab   = p$dlnm_var,
          ylab   = "Lag (months)",
          main   = paste0("DLNM heatmap — ", label, " (", grp$name, ", pooled)"),
          cex.lab = 1.5, cex.main = 1.5,
          key.title = title(main = "OR", cex.main = 1.35),
          plot.axes = {
            axis(1, cex.axis = 1.5)
            axis(2, cex.axis = 1.5)
            contour(p$at_orig, fine_lag_seq, z_mat_fine, levels = 1,
                    add = TRUE, col = "black", lty = 2, lwd = 1, drawlabels = FALSE)
          }
        )
        dev.off()
      } else {
        cat(sprintf("  Skipping heatmap for %s (%s): fine-lag crosspred unavailable\n", label, grp$name))
      }

      # Log-scale companion -- didn't exist before this unification; added so
      # the log-scale interaction surfaces can share log_z_global the same
      # way the non-interaction ones share it in save_dlnm_response_plots().
      z_mat_log <- log(z_mat)
      z_mid_log <- (z_mat_log[-1, -1] + z_mat_log[-1, -ncol(z_mat_log)] +
                    z_mat_log[-nrow(z_mat_log), -1] + z_mat_log[-nrow(z_mat_log), -ncol(z_mat_log)]) / 4
      fcol_log  <- pal_log[cut(z_mid_log, breaks = log_z_breaks_global, include.lowest = TRUE)]

      png(file.path(dir_overall, paste0("dlnm_ix_3d_", label, "_", grp$name, "_logscale_", run_suffix, ".png")),
          width = 800, height = 700)
      persp(x = p$at_orig, y = p$lag_seq, z = z_mat_log,
            zlim     = log_z_global,
            xlab     = p$dlnm_var, ylab = "Lag (months)", zlab = "log(Odds ratio) of p_bt",
            main     = paste0("DLNM surface (log scale) — ", label, " (", grp$name, ", pooled)"),
            theta    = 40, phi = 25, ltheta = 45,
            col      = fcol_log, border = NA, ticktype = "detailed",
            cex.axis = 1.5, cex.lab = 1.5, cex.main = 1.5)
      dev.off()

      if (!is.null(grp$pred_fine)) {
        z_mat_fine_log <- log(z_mat_fine)
        z_mat_fine_log_clip <- pmin(pmax(z_mat_fine_log, log_z_breaks_global[1]),
                                     log_z_breaks_global[length(log_z_breaks_global)])

        png(file.path(dir_overall, paste0("dlnm_ix_heatmap_", label, "_", grp$name, "_logscale_", run_suffix, ".png")),
            width = 800, height = 600)
        filled.contour(
          x      = p$at_orig,
          y      = fine_lag_seq,
          z      = z_mat_fine_log_clip,
          levels = log_z_breaks_global,
          col    = pal_log,
          xlab   = p$dlnm_var,
          ylab   = "Lag (months)",
          main   = paste0("DLNM heatmap (log scale) — ", label, " (", grp$name, ", pooled)"),
          cex.lab = 1.5, cex.main = 1.5,
          key.title = title(main = "log(OR)", cex.main = 1.35),
          plot.axes = {
            axis(1, cex.axis = 1.5)
            axis(2, cex.axis = 1.5)
            contour(p$at_orig, fine_lag_seq, z_mat_fine_log, levels = 0,
                    add = TRUE, col = "black", lty = 2, lwd = 1, drawlabels = FALSE)
          }
        )
        dev.off()
      }
    }

    cat(sprintf("  DLNM interaction plots saved: %s\n", label))
  }
}

#' Save DLNM Continuous-Modifier Interaction Plots
#'
#' For each continuous_var interaction in prep$dlnm_ix_vars, visualises the
#' full effect-modification surface rather than collapsing it to a single
#' reference/+1SD comparison:
#'   - Percentile-lines plot (cumulative): cumulative effect of the DLNM
#'     variable, one line per percentile of the observed modifier (default
#'     p10/25/50/75/90), coloured on a continuous scale by the modifier's
#'     value -- same idea as save_glmm_dlnm_plots()'s exposure-quantile
#'     lines, but here the lines vary over the *modifier*, not the exposure.
#'   - Per-lag percentile-lines plots: the same comparison but at each
#'     individual lag month instead of summed across all lags. A lag- or
#'     exposure-region-concentrated interaction (mixed-sign w_ix across the
#'     cross-basis) can cancel out in the cumulative summary above even
#'     though it's real and credible in the raw w_ix posterior -- check
#'     these slices (and the w_ix rows in the model summary directly)
#'     before concluding a flat cumulative plot means no interaction.
#'   - Heatmap: cumulative effect over the full (DLNM variable x modifier)
#'     grid, so the continuous surface is visible directly rather than
#'     inferred from a handful of lines.
#'
#' All three exploit that the interaction is a tensor product of the DLNM
#' sub-basis against ns(z, df = continuous_df) (z = z-scored modifier; see
#' build_dlnm_stan_data()): the coefficient draws at any modifier z-value are
#' draws_base + sum_j basis_row[j] * draws_ix_block[[j]], where basis_row is
#' the modifier's spline basis evaluated at that z (via predict() on a
#' reconstructed ns() object) -- no new model fit needed, just linear
#' recombination of the existing w_cb/w_ix posterior draws, generalizing the
#' single-column "draws_base + z * draws_ix" linear tilt to a curve that can
#' bend rather than being forced through a straight line.
#'
#' @param fit     CmdStanR fit object (must have w_cb and w_ix parameters)
#' @param prep    Return value of build_dlnm_stan_data() with dlnm_ix_vars field populated
#' @param output_dir  Directory to write PNGs/CSVs into
#' @param run_suffix  String appended to each filename
#' @param percentiles Numeric vector in (0,1): modifier percentiles for the line plot
#' @param grid_n  Number of modifier grid points for the heatmap (exposure grid
#'   reuses the same resolution as the rest of the DLNM plots)
save_dlnm_continuous_interaction_plots <- function(fit, prep, output_dir, run_suffix,
                                                    percentiles = c(0.10, 0.25, 0.50, 0.75, 0.90),
                                                    grid_n = 40) {
  if (!requireNamespace("dlnm", quietly = TRUE)) {
    cat("dlnm not installed; skipping DLNM continuous interaction plots.\n")
    return(invisible(NULL))
  }
  if (is.null(prep$dlnm_ix_vars) || length(prep$dlnm_ix_vars) == 0) return(invisible(NULL))

  # Same subfolder convention as save_dlnm_response_plots()/
  # save_dlnm_interaction_response_plots(): cumulative summary + heatmap
  # together, per-lag slices in their own folder.
  dir_overall <- file.path(output_dir, "interaction_overall_exposure_response")
  dir_perlag  <- file.path(output_dir, "interaction_exposure_response_per_lag")
  dir.create(dir_overall, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir_perlag,  recursive = TRUE, showWarnings = FALSE)

  cb_mats        <- prep$cb_mats
  dlnm_vars      <- prep$dlnm_vars
  dlnm_ix_vars   <- prep$dlnm_ix_vars
  df             <- prep$df
  dlnm_var_stats <- prep$dlnm_var_stats

  cb_ncols      <- sapply(dlnm_vars, function(v) ncol(cb_mats[[v]]))
  col_starts_cb <- cumsum(c(1L, cb_ncols[-length(cb_ncols)]))

  w_cb_draws <- fit$draws("w_cb", format = "matrix")
  w_ix_draws <- fit$draws("w_ix", format = "matrix")

  # Column offsets within w_ix: a continuous spec occupies cb_ncols[dlnm_var] *
  # continuous_df columns (continuous_df stacked cb_block-sized blocks); a
  # binary spec occupies just cb_ncols[dlnm_var] (see build_dlnm_stan_data()).
  ix_ncols <- sapply(dlnm_ix_vars, function(ix) {
    base_n <- cb_ncols[which(dlnm_vars == ix$dlnm_var)]
    if (!is.null(ix$continuous_var)) base_n * ix$continuous_df else base_n
  })
  ix_col_starts <- cumsum(c(1L, ix_ncols[-length(ix_ncols)]))

  all_diff_results <- list()  # accumulated across every continuous ix spec below

  for (k in seq_along(dlnm_ix_vars)) {
    ix <- dlnm_ix_vars[[k]]
    if (is.null(ix$continuous_var)) next  # binary specs: see save_dlnm_interaction_response_plots()

    dlnm_var       <- ix$dlnm_var
    label          <- ix$label
    continuous_var <- ix$continuous_var
    df_mod         <- ix$continuous_df

    var_idx  <- which(dlnm_vars == dlnm_var)
    n_cols   <- cb_ncols[var_idx]
    cb_cols  <- col_starts_cb[var_idx] + seq_len(n_cols) - 1L
    ix_cols  <- ix_col_starts[k]       + seq_len(n_cols * df_mod) - 1L
    cb_names <- colnames(cb_mats[[dlnm_var]])

    draws_base <- w_cb_draws[, cb_cols, drop = FALSE]
    draws_ix_full <- w_ix_draws[, ix_cols, drop = FALSE]
    # Split into df_mod cb_block-sized draws blocks, one per modifier spline
    # basis function (same stacking order used to build X_ix).
    draws_ix_blocks <- lapply(seq_len(df_mod), function(j) {
      draws_ix_full[, (j - 1) * n_cols + seq_len(n_cols), drop = FALSE]
    })

    # Reconstruct the exact z-scoring + spline basis build_dlnm_stan_data()
    # used for this modifier (same column, same rows, same df -> same
    # z-scoring and ns() knots).
    raw_mod  <- df[[continuous_var]]
    mod_mean <- mean(raw_mod, na.rm = TRUE)
    mod_sd   <- sd(raw_mod, na.rm = TRUE)
    if (!is.finite(mod_sd) || mod_sd == 0) {
      cat(sprintf("  Skipping continuous interaction plot for %s: zero/invalid SD.\n", label))
      next
    }
    mod_z      <- (raw_mod - mod_mean) / mod_sd
    mod_ns_basis <- splines::ns(mod_z, df = df_mod)
    # build_dlnm_stan_data() centers each spline basis column before it enters
    # X_ix (ns() doesn't guarantee mean-zero columns just because mod_z is
    # mean-zero); reuse that exact offset here so reconstructed predictions
    # match what the model was actually fit on.
    basis_center <- if (!is.null(ix$continuous_basis_center)) ix$continuous_basis_center else rep(0, df_mod)

    # DLNM variable's x-axis back-transform (same pattern as elsewhere)
    stats_i <- if (!is.null(dlnm_var_stats) && dlnm_var %in% names(dlnm_var_stats))
      dlnm_var_stats[[dlnm_var]] else list(mean = 0, sd = 1)
    v_mean <- stats_i$mean
    v_sd   <- stats_i$sd
    x_orig_range <- range(df[[dlnm_var]], na.rm = TRUE) * v_sd + v_mean
    at_orig_nice <- pretty(x_orig_range, n = grid_n)
    at_std_nice  <- (at_orig_nice - v_mean) / v_sd
    obs_range    <- range(df[[dlnm_var]][is.finite(df[[dlnm_var]])], na.rm = TRUE)
    keep_pts     <- at_std_nice >= obs_range[1] & at_std_nice <= obs_range[2]
    at_std  <- at_std_nice[keep_pts]
    at_orig <- at_orig_nice[keep_pts]

    # Coefficient draws at modifier z-value z_val: coef_base + the modifier's
    # spline basis evaluated at z_val, weighting each of the df_mod draws
    # blocks -- generalizes the old draws_base + z*draws_ix linear tilt to a
    # curve that can bend, since mod_basis_row is no longer just [z_val].
    crosspred_at_z <- function(z_val) {
      mod_basis_row <- as.numeric(predict(mod_ns_basis, newx = z_val)) - basis_center
      draws_z <- draws_base
      for (j in seq_len(df_mod)) draws_z <- draws_z + mod_basis_row[j] * draws_ix_blocks[[j]]
      coef_z  <- setNames(colMeans(draws_z), cb_names)
      vcov_z  <- cov(draws_z)
      dimnames(vcov_z) <- list(cb_names, cb_names)
      tryCatch(
        exp_crosspred(dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_z, vcov = vcov_z,
                        at = at_std, cen = 0, cumul = TRUE)),
        error = function(e) NULL
      )
    }

    # Precompute crosspred() once per percentile; reuse for both the cumulative
    # summary and the per-lag slices below (avoids redundant computation).
    mod_orig_at_p <- as.numeric(quantile(raw_mod, probs = percentiles, na.rm = TRUE))
    mod_z_at_p    <- (mod_orig_at_p - mod_mean) / mod_sd
    preds_by_p    <- lapply(mod_z_at_p, crosspred_at_z)

    # ── Percentile-lines plot (cumulative across all lags) ───────────────────
    # NB: a lag-concentrated interaction (e.g. RF found this pair's H-statistic
    # only at specific lags) can cancel out here if different lags/exposure
    # regions pull in opposite directions -- check the per-lag slices below
    # (and the raw w_ix posterior in the model summary) before concluding the
    # interaction isn't real just because this cumulative view looks flat.
    line_rows <- lapply(seq_along(percentiles), function(p_idx) {
      pred_z <- preds_by_p[[p_idx]]
      if (is.null(pred_z)) return(NULL)
      data.frame(
        exposure       = at_orig,
        fit            = pred_z$allfit,
        low            = pred_z$alllow,
        high           = pred_z$allhigh,
        modifier_value = mod_orig_at_p[p_idx],
        percentile     = percentiles[p_idx]
      )
    })
    curve_df <- do.call(rbind, line_rows)

    if (is.null(curve_df) || nrow(curve_df) == 0) {
      cat(sprintf("  Skipping continuous interaction line plot for %s: crosspred failed at all percentiles.\n", label))
    } else {
      curve_df$pct_label <- factor(
        sprintf("p%d (%s=%.2f)", round(curve_df$percentile * 100), continuous_var, curve_df$modifier_value),
        levels = sprintf("p%d (%s=%.2f)", round(percentiles * 100), continuous_var, mod_orig_at_p)
      )

      p_lines <- ggplot(curve_df, aes(x = exposure, y = fit, group = pct_label)) +
        geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50") +
        geom_ribbon(aes(ymin = low, ymax = high, fill = modifier_value), alpha = 0.10) +
        geom_line(aes(colour = modifier_value), linewidth = 1) +
        scale_colour_viridis_c(name = continuous_var, option = "plasma") +
        scale_fill_viridis_c(name = continuous_var, option = "plasma", guide = "none") +
        labs(
          title    = sprintf("Cumulative effect of %s across %s percentiles", dlnm_var, continuous_var),
          subtitle = sprintf("Lines at p%s of %s (not just mean/+1SD); shaded ribbon: 95%% CI",
                              paste(round(percentiles * 100), collapse = "/"), continuous_var),
          x = dlnm_var, y = "Cumulative odds ratio of p_bt"
        ) +
        theme_minimal()

      ggsave(file.path(dir_overall, paste0("dlnm_ix_continuous_lines_", label, "_", run_suffix, ".png")),
             p_lines, width = 9, height = 6, dpi = 150)
      write.csv(curve_df,
                file.path(dir_overall, paste0("dlnm_ix_continuous_lines_", label, "_", run_suffix, ".csv")),
                row.names = FALSE)
    }

    # ── Per-lag percentile-lines plots ────────────────────────────────────────
    # The cumulative plot above sums across all lags; if the true modification
    # is concentrated at a few lags (with others near zero or opposite-signed),
    # summing can cancel it out almost entirely. These slices show the effect
    # at each individual lag month instead, so a lag-concentrated interaction
    # is visible even when the cumulative summary isn't.
    L_val <- as.integer(attr(cb_mats[[dlnm_var]], "lag")[2])
    for (l in 0:L_val) {
      lag_rows <- lapply(seq_along(percentiles), function(p_idx) {
        pred_z <- preds_by_p[[p_idx]]
        if (is.null(pred_z)) return(NULL)
        data.frame(
          exposure       = at_orig,
          fit            = pred_z$matfit[, l + 1],
          low            = pred_z$matlow[, l + 1],
          high           = pred_z$mathigh[, l + 1],
          modifier_value = mod_orig_at_p[p_idx],
          percentile     = percentiles[p_idx]
        )
      })
      lag_curve_df <- do.call(rbind, lag_rows)
      if (is.null(lag_curve_df) || nrow(lag_curve_df) == 0) {
        cat(sprintf("  Skipping continuous interaction lag-%d plot for %s: crosspred failed at all percentiles.\n", l, label))
        next
      }

      lag_curve_df$pct_label <- factor(
        sprintf("p%d (%s=%.2f)", round(lag_curve_df$percentile * 100), continuous_var, lag_curve_df$modifier_value),
        levels = sprintf("p%d (%s=%.2f)", round(percentiles * 100), continuous_var, mod_orig_at_p)
      )

      p_lag <- ggplot(lag_curve_df, aes(x = exposure, y = fit, group = pct_label)) +
        geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50") +
        geom_ribbon(aes(ymin = low, ymax = high, fill = modifier_value), alpha = 0.10) +
        geom_line(aes(colour = modifier_value), linewidth = 1) +
        scale_colour_viridis_c(name = continuous_var, option = "plasma") +
        scale_fill_viridis_c(name = continuous_var, option = "plasma", guide = "none") +
        labs(
          title    = sprintf("Effect of %s at lag %d across %s percentiles", dlnm_var, l, continuous_var),
          subtitle = sprintf("Lines at p%s of %s -- not summed across lags; shaded ribbon: 95%% CI",
                              paste(round(percentiles * 100), collapse = "/"), continuous_var),
          x = dlnm_var, y = "Odds ratio of p_bt"
        ) +
        theme_minimal()

      ggsave(file.path(dir_perlag, paste0("dlnm_ix_continuous_lag", l, "_", label, "_", run_suffix, ".png")),
             p_lag, width = 9, height = 6, dpi = 150)
      write.csv(lag_curve_df,
                file.path(dir_perlag, paste0("dlnm_ix_continuous_lag", l, "_", label, "_", run_suffix, ".csv")),
                row.names = FALSE)
    }

    # ── Difference curve: is the modification actually credible? ────────────
    # The percentile-lines plots above show each percentile's *marginal* CI,
    # but those share draws_base (identical for every percentile) -- so
    # eyeballing whether two ribbons overlap is a poor, overly conservative
    # stand-in for testing whether the modifier changes the curve. The correct
    # test is the CI of the *difference* between two percentiles' coefficient
    # draws. Because draws_base is identical in both, it cancels out exactly:
    #   draws_z_hi - draws_z_lo = sum_j (basis_hi[j]-basis_lo[j]) * draws_ix_blocks[[j]]
    # -- depending only on the interaction draws, with none of the shared
    # main-effect noise that inflated the marginal ribbons. Exponentiated,
    # this is a ratio of odds ratios (ROR): how many times larger/smaller the
    # dlnm_var effect is at the high vs. low percentile of the modifier.
    # ROR = 1 (dashed line) is "no effect modification"; a 95% CI excluding 1
    # is the credible-interaction criterion -- this replaces "do the two
    # ribbons overlap" with the actual test.
    p_lo_idx <- 1
    p_hi_idx <- length(percentiles)
    basis_row_lo   <- as.numeric(predict(mod_ns_basis, newx = mod_z_at_p[p_lo_idx])) - basis_center
    basis_row_hi   <- as.numeric(predict(mod_ns_basis, newx = mod_z_at_p[p_hi_idx])) - basis_center
    basis_row_diff <- basis_row_hi - basis_row_lo

    draws_diff <- draws_ix_blocks[[1]] * 0
    for (j in seq_len(df_mod)) draws_diff <- draws_diff + basis_row_diff[j] * draws_ix_blocks[[j]]
    coef_diff <- setNames(colMeans(draws_diff), cb_names)
    vcov_diff <- cov(draws_diff)
    dimnames(vcov_diff) <- list(cb_names, cb_names)

    pred_diff <- tryCatch(
      exp_crosspred(dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_diff, vcov = vcov_diff,
                      at = at_std, cen = 0, cumul = TRUE)),
      error = function(e) NULL
    )

    if (is.null(pred_diff)) {
      cat(sprintf("  Skipping continuous interaction difference plot for %s: crosspred failed.\n", label))
    } else {
      diff_rows <- c(
        list(data.frame(lag = "cumulative", exposure = at_orig,
                         ror = pred_diff$allfit, ror_low = pred_diff$alllow, ror_high = pred_diff$allhigh)),
        lapply(0:L_val, function(l) data.frame(
          lag = as.character(l), exposure = at_orig,
          ror = pred_diff$matfit[, l + 1], ror_low = pred_diff$matlow[, l + 1], ror_high = pred_diff$mathigh[, l + 1]
        ))
      )
      diff_df <- do.call(rbind, diff_rows)
      diff_df$significant <- (diff_df$ror_low > 1 & diff_df$ror_high > 1) |
                              (diff_df$ror_low < 1 & diff_df$ror_high < 1)
      diff_df$label       <- label
      diff_df$pct_lo      <- round(percentiles[p_lo_idx] * 100)
      diff_df$pct_hi      <- round(percentiles[p_hi_idx] * 100)

      write.csv(diff_df,
                file.path(dir_overall, paste0("dlnm_ix_continuous_diff_", label, "_", run_suffix, ".csv")),
                row.names = FALSE)
      all_diff_results[[label]] <- diff_df

      cat(sprintf("  [%s] p%d vs p%d of %s: %d/%d (exposure x lag) points with CI excluding ROR=1 (cumulative %s)\n",
                  label, diff_df$pct_hi[1], diff_df$pct_lo[1], continuous_var,
                  sum(diff_df$significant), nrow(diff_df),
                  if (any(diff_df$significant[diff_df$lag == "cumulative"])) "CREDIBLE" else "not credible"))

      plot_diff_curve <- function(sub_df, title, file_path) {
        # geom_ribbon groups by the fill *level*, not by contiguous runs -- with
        # a non-contiguous TRUE region (e.g. significant at low exposure AND
        # again at high exposure, but not in between) it would otherwise stitch
        # all TRUE points into a single polygon that bridges straight across
        # the FALSE gap. run_id breaks the ribbon at every significant/not
        # transition so each disjoint stretch gets its own polygon.
        sub_df <- sub_df[order(sub_df$exposure), ]
        sub_df$run_id <- cumsum(c(TRUE, diff(sub_df$significant) != 0))
        p <- ggplot(sub_df, aes(x = exposure, y = ror)) +
          geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50") +
          geom_ribbon(aes(ymin = ror_low, ymax = ror_high, fill = significant, group = run_id), alpha = 0.25) +
          geom_line(colour = "black", linewidth = 1) +
          scale_fill_manual(values = c(`TRUE` = "firebrick", `FALSE` = "grey70"),
                             name = "CI excludes 1", guide = "none") +
          labs(
            title    = title,
            subtitle = sprintf("Ratio of odds ratios (p%d vs p%d of %s); red band = 95%% CI excludes ROR = 1",
                                diff_df$pct_hi[1], diff_df$pct_lo[1], continuous_var),
            x = dlnm_var, y = "Ratio of odds ratios (high / low modifier)"
          ) +
          theme_minimal()
        ggsave(file_path, p, width = 9, height = 6, dpi = 150)
      }

      plot_diff_curve(
        diff_df[diff_df$lag == "cumulative", ],
        sprintf("Effect modification of %s by %s: cumulative", dlnm_var, continuous_var),
        file.path(dir_overall, paste0("dlnm_ix_continuous_diff_cumulative_", label, "_", run_suffix, ".png"))
      )
      for (l in 0:L_val) {
        plot_diff_curve(
          diff_df[diff_df$lag == as.character(l), ],
          sprintf("Effect modification of %s by %s: lag %d", dlnm_var, continuous_var, l),
          file.path(dir_perlag, paste0("dlnm_ix_continuous_diff_lag", l, "_", label, "_", run_suffix, ".png"))
        )
      }
    }

    # ── Heatmap: full (exposure x modifier) surface ───────────────────────────
    mod_orig_grid <- seq(min(raw_mod, na.rm = TRUE), max(raw_mod, na.rm = TRUE), length.out = grid_n)
    mod_z_grid    <- (mod_orig_grid - mod_mean) / mod_sd

    heat_rows <- lapply(seq_along(mod_z_grid), function(i) {
      pred_z <- crosspred_at_z(mod_z_grid[i])
      if (is.null(pred_z)) return(NULL)
      data.frame(exposure = at_orig, modifier = mod_orig_grid[i], fit = pred_z$allfit)
    })
    heat_df <- do.call(rbind, heat_rows)

    if (is.null(heat_df) || nrow(heat_df) == 0) {
      cat(sprintf("  Skipping continuous interaction heatmap for %s: crosspred failed at all grid points.\n", label))
    } else {
      # Diverging scale centred on OR = 1 (no effect), not 0 -- limit is the
      # largest deviation from 1 in either direction, so the colour scale
      # stays symmetric around "no effect" on the ratio scale.
      limit <- max(abs(heat_df$fit - 1), na.rm = TRUE)
      p_heat <- ggplot(heat_df, aes(x = exposure, y = modifier, fill = fit)) +
        geom_tile() +
        # rev(): matches this plot's existing low=red/high=blue convention (see
        # the 3-D surface above); limits are symmetric around 1 by construction,
        # so gradientn's evenly-spaced default stops put white exactly at 1.
        scale_fill_gradientn(colours = rev(dlnm_diverging_pal),
                            limits = c(1 - limit, 1 + limit), name = "Odds ratio") +
        labs(
          title    = sprintf("Effect-modification surface: %s x %s", dlnm_var, continuous_var),
          subtitle = "Colour = cumulative odds ratio of p_bt",
          x = dlnm_var, y = continuous_var
        ) +
        theme_minimal() +
        theme(panel.grid = element_blank())

      ggsave(file.path(dir_overall, paste0("dlnm_ix_continuous_heatmap_", label, "_", run_suffix, ".png")),
             p_heat, width = 8, height = 6, dpi = 150)
      write.csv(heat_df,
                file.path(dir_overall, paste0("dlnm_ix_continuous_heatmap_", label, "_", run_suffix, ".csv")),
                row.names = FALSE)
    }

    cat(sprintf("  Continuous interaction plots saved: %s\n", label))
  }

  # Single combined summary across every continuous interaction spec in this
  # model -- the "at a glance" artifact for "is there a credible interaction,
  # and where": filter to significant == TRUE to see exactly which
  # (label, lag, exposure) combinations have a 95% CI excluding ROR = 1,
  # rather than having to inspect every per-label/per-lag plot individually.
  if (length(all_diff_results) > 0) {
    combined_diff_df <- do.call(rbind, all_diff_results)
    rownames(combined_diff_df) <- NULL
    write.csv(combined_diff_df,
              file.path(dir_overall, paste0("dlnm_ix_continuous_diff_summary_", run_suffix, ".csv")),
              row.names = FALSE)

    cumul_sig <- combined_diff_df[combined_diff_df$lag == "cumulative" & combined_diff_df$significant, ]
    any_sig   <- combined_diff_df[combined_diff_df$significant, ]
    cat(sprintf(
      "\nContinuous interaction summary: %d/%d labels credible cumulatively; %d/%d labels credible at >=1 lag.\n",
      length(unique(cumul_sig$label)), length(all_diff_results),
      length(unique(any_sig$label)), length(all_diff_results)
    ))
  }
  invisible(all_diff_results)
}

#' Save Backward Attributable-Number Time Series Plot
#'
#' Plots the monthly backward attributable number of positive houses (summed
#' across all CMFs, with 95% CI ribbon from compute_af_posterior_timeseries()),
#' alongside the mean observed exposure trend across CMFs for the same months --
#' analogous to Figure 3 (right panel) in Gasparrini & Leone (2014), adapted
#' from a single daily time series to a monthly panel pooled across CMFs.
#'
#' Note: b-AN can legitimately dip negative for some months (the
#' harvesting-paradox artifact of the backward counterfactual -- see
#' compute_af_posterior_timeseries()'s documentation) -- this is expected, not
#' a bug, and the zero line is drawn explicitly so it's easy to see when/where
#' it happens.
#'
#' @param af_ts Data frame returned by compute_af_posterior_timeseries()
#' @param df Data frame with year_month_date and the raw (original-scale)
#'   exposure column for `var`
#' @param var Character; name of the exposure variable (must be a column in df)
#' @param output_dir Directory to write the PNG into
#' @param run_suffix String appended to the filename
#' @return NULL (saves plot to PNG file)
save_af_timeseries_plot <- function(af_ts, df, var, output_dir, run_suffix) {
  exposure_trend <- df %>%
    group_by(year_month_date) %>%
    summarise(exposure_mean = mean(.data[[var]], na.rm = TRUE), .groups = "drop")

  plot_df <- af_ts %>% left_join(exposure_trend, by = "year_month_date")

  # Scale the exposure trend onto the AN axis for a dual-axis-style overlay
  an_range  <- range(c(plot_df$an_q025, plot_df$an_q975), na.rm = TRUE)
  exp_range <- range(plot_df$exposure_mean, na.rm = TRUE)
  scale_factor <- diff(an_range) / diff(exp_range)
  offset       <- an_range[1] - exp_range[1] * scale_factor

  p <- ggplot(plot_df, aes(x = year_month_date)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_ribbon(aes(ymin = an_q025, ymax = an_q975), fill = "steelblue", alpha = 0.2) +
    geom_line(aes(y = an_mean), colour = "steelblue", linewidth = 1) +
    geom_line(aes(y = exposure_mean * scale_factor + offset), colour = "firebrick",
              linewidth = 0.6, linetype = "dashed") +
    scale_y_continuous(
      name     = "Backward attributable number (positive houses), summed across CMFs",
      sec.axis = sec_axis(~ (. - offset) / scale_factor, name = paste0(var, " (mean across CMFs)"))
    ) +
    labs(
      title    = paste0("Backward attributable number over time -- ", var),
      subtitle = "Blue: monthly attributable positive-house count (95% CI). Red dashed: mean observed exposure.",
      caption  = "Dips below zero reflect the backward counterfactual's harvesting-paradox artifact, not true protection.",
      x = NULL
    ) +
    theme_minimal()

  ggsave(file.path(output_dir, paste0("af_timeseries_", var, "_", run_suffix, ".png")),
         p, width = 12, height = 6, dpi = 150)
  cat(sprintf("  AF time series plot saved: %s\n", var))
}

#' Save GLMM Coefficient Forest Plot
#'
#' Grouped forest plot of fixed-effect log-odds coefficients with 95% Wald CIs.
#' Terms are clustered into variable families (rainfall, VPD, land use, etc.)
#' shown as labelled facet strips. Significant terms (p < 0.05) are highlighted.
#'
#' @param coef_table  Tibble from GLMM_postfit with columns term, estimate, std_error, p_value
#' @param cfg         Model configuration list (used for lag_vars)
#' @param output_dir  Directory to write the PNG
#' @param run_suffix  String appended to the filename
#' @return Invisibly, the ggplot object
save_glmm_coef_forest_plot <- function(coef_table, cfg = NULL, output_dir, run_suffix,
                                       scale = c("logodds", "OR")) {
  scale <- match.arg(scale, several.ok = TRUE)
  if (length(scale) > 1) {
    for (s in scale) save_glmm_coef_forest_plot(coef_table, cfg, output_dir, run_suffix, scale = s)
    return(invisible(NULL))
  }

  # --- family definitions (first match wins) ---
  families <- list(
    list(pattern = "^total_rainy_days",
         group   = "Total rainy days",
         label   = function(t) paste("lag", sub(".*_lag", "", t))),
    list(pattern = "^avg_VPD",
         group   = "Vapour pressure deficit",
         label   = function(t) paste("lag", sub(".*_lag", "", t))),
    list(pattern = "^precip_max_day_resid_on_trd",
         group   = "Precipitation extremes",
         label   = function(t) paste("lag", sub(".*_lag", "", t))),
    list(pattern = "^hurricane_within_120km",
         group   = "Hurricane",
         label   = function(t) paste("lag", sub(".*_lag", "", t))),
    list(pattern = "^(is_urban|is_WUI|is_WI|landcover)",
         group   = "Land use",
         label   = function(t) dplyr::recode(t,
           is_urban  = "Urban",
           is_WUI    = "Wildland-urban interface",
           is_WI     = "Water interface",
           .default  = sub("^landcover", "Landcover: ", t))),
    list(pattern = "^(has_aljibes|water_containers|water_shortage)",
         group   = "Water access",
         label   = function(t) dplyr::recode(t,
           has_aljibes      = "Cisternae present",
           water_containers = "Water containers (per capita)",
           water_shortage   = "Water shortage zone")),
    list(pattern = "^pop_density",
         group   = "Demographics",
         label   = function(t) "Population density"),
    list(pattern = "^reactive_shift",
         group   = "Reactive surveillance",
         label   = function(t) "log(1 + dengue cases)")
  )

  group_order <- c("Total rainy days", "Vapour pressure deficit",
                   "Precipitation extremes", "Hurricane",
                   "Land use", "Water access", "Demographics",
                   "Reactive surveillance", "Other")

  # Clean display labels for unlagged vars (fallback for any unmatched names)
  unlagged_labels <- c(
    is_urban         = "Urban",
    is_WUI           = "Wildland-urban interface",
    is_WI            = "Water interface",
    has_aljibes      = "Cisternae present",
    water_containers = "Water containers (per capita)",
    water_shortage   = "Water shortage zone",
    pop_density      = "Population density",
    reactive_shift   = "log(1 + dengue cases)"
  )

  df_plot <- coef_table %>%
    dplyr::filter(term != "(Intercept)") %>%
    dplyr::mutate(
      ci_low  = estimate - 1.96 * std_error,
      ci_high = estimate + 1.96 * std_error,
      group   = NA_character_,
      label   = term
    )

  for (fam in families) {
    idx <- grepl(fam$pattern, df_plot$term) & is.na(df_plot$group)
    if (!any(idx)) next
    df_plot$group[idx] <- fam$group
    df_plot$label[idx] <- vapply(df_plot$term[idx], fam$label, character(1))
  }
  df_plot$group[is.na(df_plot$group)] <- "Other"
  df_plot$label[df_plot$label == df_plot$term] <-
    dplyr::recode(df_plot$label[df_plot$label == df_plot$term],
                  !!!as.list(unlagged_labels), .default = df_plot$label[df_plot$label == df_plot$term])

  # Within-group ordering: lags ascending (lag0 at top), unlagged alphabetical
  df_plot <- df_plot %>%
    dplyr::mutate(
      group    = factor(group, levels = group_order),
      lag_num  = suppressWarnings(as.integer(sub(".*_lag", "", term))),
      sort_key = ifelse(!is.na(lag_num), lag_num, 99L)
    ) %>%
    dplyr::arrange(group, sort_key, term) %>%
    dplyr::mutate(
      label = factor(label, levels = rev(unique(label))),  # rev → lag0 at top in ggplot
      significant = p_value < 0.05
    )

  if (scale == "OR") {
    df_plot <- df_plot %>%
      dplyr::mutate(x_val  = exp(estimate),
                    x_low  = exp(ci_low),
                    x_high = exp(ci_high))
    x_ref    <- 1
    x_label  <- "Odds Ratio"
    subtitle <- "OR scale · bars = 95% Wald CI · red = p < 0.05 · reference line = OR 1"
    x_scale  <- ggplot2::scale_x_log10(
      breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 10),
      labels = c("0.1", "0.25", "0.5", "1", "2", "4", "10"))
    file_tag <- "OR"
  } else {
    df_plot <- df_plot %>%
      dplyr::mutate(x_val  = estimate,
                    x_low  = ci_low,
                    x_high = ci_high)
    x_ref    <- 0
    x_label  <- "Log-odds coefficient"
    subtitle <- "Log-odds scale · bars = 95% Wald CI · red = p < 0.05"
    x_scale  <- ggplot2::scale_x_continuous()
    file_tag <- "logodds"
  }

  p <- ggplot2::ggplot(df_plot,
         ggplot2::aes(x = x_val, y = label, colour = significant)) +
    ggplot2::geom_vline(xintercept = x_ref, linetype = "dashed",
                        colour = "gray40", linewidth = 0.5) +
    ggplot2::geom_errorbar(
      ggplot2::aes(xmin = x_low, xmax = x_high),
      width = 0.35, linewidth = 0.55,
      orientation = "y") +
    ggplot2::geom_point(size = 2.2) +
    x_scale +
    ggplot2::scale_colour_manual(
      values = c("TRUE" = "#c0392b", "FALSE" = "gray55"),
      labels = c("TRUE" = "p < 0.05", "FALSE" = "p ≥ 0.05"),
      name   = NULL) +
    ggplot2::facet_grid(group ~ ., scales = "free_y", space = "free_y", switch = "y") +
    ggplot2::labs(
      title    = "Fixed-effect coefficients — beta-binomial GLMM",
      subtitle = subtitle,
      x        = x_label,
      y        = NULL
    ) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      strip.placement    = "outside",
      strip.text.y.left  = ggplot2::element_text(angle = 0, hjust = 1,
                                                  face = "bold", size = 9),
      strip.background   = ggplot2::element_rect(fill = "gray93", colour = NA),
      panel.spacing      = ggplot2::unit(0.5, "lines"),
      panel.grid.major.y = ggplot2::element_blank(),
      legend.position    = "bottom",
      axis.text.y        = ggplot2::element_text(size = 9),
      plot.title         = ggplot2::element_text(face = "bold", size = 12),
      plot.subtitle      = ggplot2::element_text(size = 10, colour = "gray40")
    )

  n_terms <- nrow(df_plot)
  fig_h   <- max(5, 0.3 * n_terms + 2.5)

  out_file <- file.path(output_dir,
                        paste0("glmm_coef_forest_", file_tag, "_", run_suffix, ".png"))
  ggplot2::ggsave(out_file, p, width = 9, height = fig_h, dpi = 150)
  cat("Forest plot saved to:", out_file, "\n")
  invisible(p)
}

#' DLNM exposure-lag-response plots for a single variable from a glmmTMB model.
#'
#' Produces three plots per variable:
#'   1. Cumulative effect curve (summed over all lags) with 95% CI
#'   2. Lag-response profile at the 75th-percentile exposure value
#'   3. Full exposure-lag-response heatmap
#'
#' @param var              Variable name (character), used in titles and filenames.
#' @param cb_obj           crossbasis object built for this variable.
#' @param cb_col_names_var Character vector of model coefficient names for this crossbasis.
#' @param model            Fitted glmmTMB object.
#' @param at               Numeric vector of exposure values to predict at.
#' @param cen              Centering value for the log-odds ratio (typically the median).
#' @param max_lag          Maximum lag used in the model.
#' @param output_dir       Directory to write PNGs.
#' @param run_suffix       String appended to filenames.
save_glmm_dlnm_plots <- function(var, cb_obj, cb_term_name, model,
                                 at, cen, max_lag, output_dir, run_suffix,
                                 scale_center = NULL, scale_sd = NULL) {
  if (!requireNamespace("dlnm", quietly = TRUE))
    stop("Package 'dlnm' required for DLNM plots")

  # Extract fixed-effect coefs and vcov from glmmTMB.
  # Matrix column cb_term_name produces coefficients named <cb_term_name><cb_col_name>
  # (e.g. cb_total_precipv1.l1). Grep by prefix to find them, then rename to match
  # the crossbasis object's own column names so crosspred() can work correctly.
  all_coef <- glmmTMB::fixef(model)$cond
  all_vcov <- as.matrix(vcov(model)$cond)

  cb_model_names <- grep(paste0("^", cb_term_name), names(all_coef), value = TRUE)
  if (length(cb_model_names) != ncol(cb_obj))
    stop(sprintf("Expected %d coefs for %s, found %d", ncol(cb_obj), cb_term_name, length(cb_model_names)))

  cb_coef <- all_coef[cb_model_names]
  cb_vcov <- all_vcov[cb_model_names, cb_model_names, drop = FALSE]

  names(cb_coef)                         <- colnames(cb_obj)
  rownames(cb_vcov) <- colnames(cb_vcov) <- colnames(cb_obj)

  pred <- tryCatch(
    suppressWarnings(
      dlnm::crosspred(cb_obj, coef = cb_coef, vcov = cb_vcov, at = at, cen = cen)
    ),
    error = function(e) {
      cat(sprintf("  crosspred() failed for %s: %s\n", var, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(pred)) return(invisible(NULL))

  # Resolve which matrix fields are populated.
  # Without model.link, dlnm stores log-scale effects in matfit.
  # With model.link it may route to matRRfit; fall back accordingly.
  mat_fit  <- if (!is.null(pred$matfit)  && length(pred$matfit)  > 0) pred$matfit  else log(pred$matRRfit)
  mat_low  <- if (!is.null(pred$matlow)  && length(pred$matlow)  > 0) pred$matlow  else log(pred$matRRlow)
  mat_high <- if (!is.null(pred$mathigh) && length(pred$mathigh) > 0) pred$mathigh else log(pred$matRRhigh)

  if (is.null(mat_fit) || length(mat_fit) == 0) {
    cat(sprintf("  No prediction matrix for %s (fields: %s)\n", var, paste(names(pred), collapse=", ")))
    return(invisible(NULL))
  }

  var_label <- gsub("_", " ", var)
  lags      <- seq(0, max_lag, length.out = ncol(mat_fit))

  # Back-transform exposure axis to original (unstandardized) scale where available
  if (!is.null(scale_center) && !is.null(scale_sd) &&
      var %in% names(scale_center) && var %in% names(scale_sd)) {
    x_orig <- pred$predvar * scale_sd[[var]] + scale_center[[var]]
    cen_orig <- cen * scale_sd[[var]] + scale_center[[var]]
  } else {
    x_orig   <- pred$predvar
    cen_orig <- cen
  }

  # --- 1. Cumulative effect curve ---
  # allfit (cumul=TRUE) doesn't always work with coef/vcov; sum over lags as fallback.
  if (length(pred$allfit) == length(pred$predvar)) {
    cum_fit  <- pred$allfit
    cum_low  <- pred$alllow
    cum_high <- pred$allhigh
  } else {
    cum_fit  <- rowSums(mat_fit)
    cum_low  <- rowSums(mat_low)
    cum_high <- rowSums(mat_high)
  }
  is_binary <- length(at) <= 2
  df_cumul  <- data.frame(
    exposure = x_orig,
    fit      = cum_fit,
    low      = cum_low,
    high     = cum_high
  )

  if (is_binary) {
    df_cumul$label <- ifelse(df_cumul$exposure == max(df_cumul$exposure), "Present", "Absent")
    p_cumul <- ggplot2::ggplot(df_cumul, ggplot2::aes(x = label, y = fit)) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
      ggplot2::geom_pointrange(ggplot2::aes(ymin = low, ymax = high),
                               colour = "steelblue", size = 0.8, linewidth = 1) +
      ggplot2::scale_x_discrete(limits = c("Absent", "Present")) +
      ggplot2::labs(
        title    = paste("Cumulative effect:", var_label),
        subtitle = paste0("Log-odds summed over lags 0–", max_lag, "; ref = absent"),
        x = var_label, y = "Cumulative log-odds ratio"
      ) +
      ggplot2::theme_minimal()
  } else {
    p_cumul <- ggplot2::ggplot(df_cumul, ggplot2::aes(x = exposure)) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = low, ymax = high),
                           fill = "steelblue", alpha = 0.25) +
      ggplot2::geom_line(ggplot2::aes(y = fit), colour = "steelblue", linewidth = 1) +
      ggplot2::labs(
        title    = paste("Cumulative effect:", var_label),
        subtitle = paste0("Log-odds summed over lags 0–", max_lag,
                          "; ref = ", round(cen_orig, 2)),
        x = var_label, y = "Cumulative log-odds ratio"
      ) +
      ggplot2::theme_minimal()
  }
  ggplot2::ggsave(file.path(output_dir, paste0("dlnm_cumul_", var, "_", run_suffix, ".png")),
                  p_cumul, width = 7, height = 5, dpi = 150)

  # --- 2. Lag-response profiles across exposure quantiles ---
  if (is_binary) {
    # Binary: single curve for "present" vs reference (absent)
    present_idx <- which.max(pred$predvar)
    df_lag <- data.frame(
      lag     = lags,
      fit     = mat_fit[present_idx, ],
      low     = mat_low[present_idx, ],
      high    = mat_high[present_idx, ],
      exp_val = x_orig[present_idx]
    )
    p_lag <- ggplot2::ggplot(df_lag, ggplot2::aes(x = lag)) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = low, ymax = high),
                           fill = "steelblue", alpha = 0.25) +
      ggplot2::geom_line(ggplot2::aes(y = fit), colour = "steelblue", linewidth = 1) +
      ggplot2::geom_point(ggplot2::aes(y = fit), colour = "steelblue", size = 2.5) +
      ggplot2::scale_x_continuous(breaks = 0:max_lag) +
      ggplot2::labs(
        title    = paste("Lag-response profile:", var_label),
        subtitle = paste0("Exposure = present (1); ref = absent"),
        x = "Lag (months)", y = "Log-odds ratio"
      ) +
      ggplot2::theme_minimal()
  } else {
    # Continuous: one curve per exposure quantile, coloured by exposure level
    quant_probs <- c(0.10, 0.25, 0.50, 0.75, 0.90)
    quant_idxs  <- sapply(quant_probs, function(q)
      which.min(abs(pred$predvar - quantile(pred$predvar, q))))

    df_lag_multi <- do.call(rbind, lapply(seq_along(quant_probs), function(i) {
      idx <- quant_idxs[i]
      data.frame(
        lag     = lags,
        fit     = mat_fit[idx, ],
        low     = mat_low[idx, ],
        high    = mat_high[idx, ],
        exp_val = x_orig[idx]
      )
    }))

    p_lag <- ggplot2::ggplot(df_lag_multi,
                             ggplot2::aes(x = lag, y = fit,
                                          colour = exp_val, group = factor(exp_val))) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = low, ymax = high, fill = exp_val),
                           alpha = 0.10, colour = NA) +
      ggplot2::geom_line(linewidth = 1) +
      ggplot2::geom_point(size = 2.5) +
      ggplot2::scale_colour_viridis_c(name = var_label, option = "plasma") +
      ggplot2::scale_fill_viridis_c(name = var_label, option = "plasma", guide = "none") +
      ggplot2::scale_x_continuous(breaks = 0:max_lag) +
      ggplot2::labs(
        title    = paste("Lag-response profiles:", var_label),
        subtitle = paste0("Lines at p10/p25/p50/p75/p90; ref = ", round(cen_orig, 2)),
        x = "Lag (months)", y = "Log-odds ratio"
      ) +
      ggplot2::theme_minimal()
  }
  ggplot2::ggsave(file.path(output_dir, paste0("dlnm_lagresponse_", var, "_", run_suffix, ".png")),
                  p_lag, width = 7, height = 5, dpi = 150)

  # --- 3. Exposure-lag-response heatmap ---
  df_heat <- expand.grid(exposure = x_orig, lag = lags)
  df_heat$fit <- as.vector(mat_fit)
  limit <- max(abs(df_heat$fit), na.rm = TRUE)

  p_heat <- ggplot2::ggplot(df_heat, ggplot2::aes(x = lag, y = exposure, fill = fit)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                                  midpoint = 0, limits = c(-limit, limit),
                                  name = "Log-OR") +
    ggplot2::scale_x_continuous(breaks = 0:max_lag) +
    ggplot2::labs(
      title    = paste("Exposure-lag-response surface:", var_label),
      subtitle = "Colour = log-odds ratio vs median exposure",
      x = "Lag (months)", y = var_label
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(panel.grid = ggplot2::element_blank())
  ggplot2::ggsave(file.path(output_dir, paste0("dlnm_heatmap_", var, "_", run_suffix, ".png")),
                  p_heat, width = 7, height = 5, dpi = 150)

  cat(sprintf("DLNM plots saved for %s\n", var))
  invisible(pred)
}

