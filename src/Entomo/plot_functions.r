
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
         caption = "Orange ribbon: 95% CI of y_pred/n. Grey bars: dengue cases per block (right axis).") +
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
#' @param fit     CmdStanR fit object
#' @param prep    Return value of build_dlnm_stan_data() (contains cb_mats, dlnm_vars, df)
#' @param output_dir  Directory to write PNGs into
#' @param run_suffix  String appended to each filename
save_dlnm_response_plots <- function(fit, prep, output_dir, run_suffix) {
  if (!requireNamespace("dlnm", quietly = TRUE)) {
    cat("dlnm not installed; skipping DLNM response plots.\n")
    return(invisible(NULL))
  }

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
    cat(sprintf("  [%s] original range: [%.3f, %.3f]  cen=mean=%.3f\n",
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
      dlnm::crosspred(cb_mats[[var]], coef = coef_i, vcov = vcov_i,
                      at = at_std, cen = 0, cumul = TRUE),
      error = function(e) {
        cat(sprintf("  crosspred failed for %s: %s\n", var, conditionMessage(e)))
        NULL
      }
    )
    if (is.null(pred_i)) next

    preds[[var]] <- list(pred = pred_i, at_std = at_std, at_orig = at_orig,
                         v_mean = v_mean, v_sd = v_sd,
                         L_val = as.integer(attr(cb_mats[[var]], "lag")[2]))
  }

  # ── Global z-range for comparable 3-D axes and colour scale ───────────────
  all_z <- unlist(lapply(preds, function(p) if (!is.null(p)) p$pred$matfit))
  z_global <- range(all_z, na.rm = TRUE)
  z_breaks_global <- seq(z_global[1], z_global[2], length.out = 51)
  pal <- colorRampPalette(c("firebrick", "white", "steelblue"))(50)

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
    png(file.path(output_dir, paste0("dlnm_overall_", var, "_", run_suffix, ".png")),
        width = 800, height = 500)
    plot(pred_i, "overall",
         xaxt   = "n",
         main   = paste("Cumulative effect —", var),
         xlab   = var,
         ylab   = "Effect on log-odds of p_bt",
         col    = "steelblue",
         ci.arg = list(col = adjustcolor("steelblue", 0.25), border = NA))
    axis(1, at = at_std, labels = round(at_orig, 2))
    abline(h = 0, lty = 2, col = "grey50")
    dev.off()

    # ── 3-D surface (original x-axis, shared z-scale across predictors) ──────
    z_mat  <- pred_i$matfit
    z_mid  <- (z_mat[-1, -1] + z_mat[-1, -ncol(z_mat)] +
               z_mat[-nrow(z_mat), -1] + z_mat[-nrow(z_mat), -ncol(z_mat)]) / 4
    facet_col <- pal[cut(z_mid, breaks = z_breaks_global, include.lowest = TRUE)]

    png(file.path(output_dir, paste0("dlnm_3d_", var, "_", run_suffix, ".png")),
        width = 800, height = 700)
    persp(x        = at_orig,
          y        = lag_seq,
          z        = z_mat,
          zlim     = z_global,
          xlab     = var,
          ylab     = "Lag (months)",
          zlab     = "Effect on log-odds of p_bt",
          main     = paste("DLNM surface —", var),
          theta    = 40, phi = 25, ltheta = 45,
          col      = facet_col,
          border   = NA,
          ticktype = "detailed")
    dev.off()

    # ── Per-lag slice plots (one per lag, same style as cumulative) ──────────
    for (l in lag_seq) {
      png(file.path(output_dir,
                    paste0("dlnm_lag", l, "_", var, "_", run_suffix, ".png")),
          width = 800, height = 500)
      plot(pred_i, "slices",
           lag    = l,
           xaxt   = "n",
           main   = paste0("Effect at lag ", l, " — ", var),
           xlab   = var,
           ylab   = "Effect on log-odds of p_bt",
           col    = "steelblue",
           ci.arg = list(col = adjustcolor("steelblue", 0.25), border = NA))
      axis(1, at = at_std, labels = round(at_orig, 2))
      abline(h = 0, lty = 2, col = "grey50")
      dev.off()
    }

    cat(sprintf("  DLNM plots saved: %s\n", var))
  }
}

#' Save DLNM Interaction Response Plots
#'
#' For each interaction specified in prep$dlnm_ix_vars, plots:
#'   - Cumulative effect comparison: reference group (w_cb only) vs active group (w_cb + w_ix)
#'   - Per-lag slice comparison for each lag
#'   - 3-D surface for both groups side-by-side
#'
#' @param fit     CmdStanR fit object (must have w_cb and w_ix parameters)
#' @param prep    Return value of build_dlnm_stan_data() with dlnm_ix_vars field populated
#' @param output_dir  Directory to write PNGs into
#' @param run_suffix  String appended to each filename
save_dlnm_interaction_response_plots <- function(fit, prep, output_dir, run_suffix) {
  if (!requireNamespace("dlnm", quietly = TRUE)) {
    cat("dlnm not installed; skipping DLNM interaction plots.\n")
    return(invisible(NULL))
  }
  if (is.null(prep$dlnm_ix_vars) || length(prep$dlnm_ix_vars) == 0) return(invisible(NULL))

  cb_mats      <- prep$cb_mats
  dlnm_vars    <- prep$dlnm_vars
  dlnm_ix_vars <- prep$dlnm_ix_vars
  df           <- prep$df
  dlnm_var_stats <- prep$dlnm_var_stats

  cb_ncols      <- sapply(dlnm_vars, function(v) ncol(cb_mats[[v]]))
  col_starts_cb <- cumsum(c(1L, cb_ncols[-length(cb_ncols)]))

  w_cb_draws <- fit$draws("w_cb", format = "matrix")
  w_ix_draws <- fit$draws("w_ix", format = "matrix")

  # Column offsets within w_ix: each interaction occupies cb_ncols[dlnm_var] columns
  ix_ncols      <- sapply(dlnm_ix_vars, function(ix) cb_ncols[which(dlnm_vars == ix$dlnm_var)])
  ix_col_starts <- cumsum(c(1L, ix_ncols[-length(ix_ncols)]))

  for (k in seq_along(dlnm_ix_vars)) {
    ix       <- dlnm_ix_vars[[k]]
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
      dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_ref, vcov = vcov_ref,
                      at = at_std, cen = 0, cumul = TRUE),
      error = function(e) { cat(sprintf("  crosspred (ref) failed for %s: %s\n", label, conditionMessage(e))); NULL }
    )
    pred_active <- tryCatch(
      dlnm::crosspred(cb_mats[[dlnm_var]], coef = coef_active, vcov = vcov_active,
                      at = at_std, cen = 0, cumul = TRUE),
      error = function(e) { cat(sprintf("  crosspred (active) failed for %s: %s\n", label, conditionMessage(e))); NULL }
    )
    if (is.null(pred_ref) || is.null(pred_active)) next

    L_val   <- as.integer(attr(cb_mats[[dlnm_var]], "lag")[2])
    lag_seq <- 0:L_val

    ref_col    <- "steelblue"
    active_col <- "firebrick"

    # ── Cumulative effect comparison ──────────────────────────────────────────
    y_lim <- range(pred_ref$alllow, pred_ref$allhigh,
                   pred_active$alllow, pred_active$allhigh, na.rm = TRUE)
    png(file.path(output_dir, paste0("dlnm_ix_cumul_", label, "_", run_suffix, ".png")),
        width = 900, height = 500)
    plot(pred_ref, "overall", xaxt = "n", ylim = y_lim,
         main   = paste("Cumulative effect of", dlnm_var, "—", label),
         xlab   = dlnm_var, ylab = "Effect on log-odds of p_bt",
         col    = ref_col,
         ci.arg = list(col = adjustcolor(ref_col, 0.20), border = NA))
    lines(at_std, pred_active$allfit, col = active_col, lwd = 2)
    lines(at_std, pred_active$alllow,  col = active_col, lwd = 1, lty = 2)
    lines(at_std, pred_active$allhigh, col = active_col, lwd = 1, lty = 2)
    axis(1, at = at_std, labels = round(at_orig, 2))
    abline(h = 0, lty = 2, col = "grey50")
    legend_labels <- if (!is.null(ix$modifier_var)) {
      c("Reference  (mean level)",
        sprintf("Active group  (%s: +1 SD)", ix$modifier_var))
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
      png(file.path(output_dir, paste0("dlnm_ix_lag", l, "_", label, "_", run_suffix, ".png")),
          width = 900, height = 500)
      plot(pred_ref, "slices", lag = l, xaxt = "n", ylim = y_lim_lag,
           main   = paste0("Effect at lag ", l, " — ", label),
           xlab   = dlnm_var, ylab = "Effect on log-odds of p_bt",
           col    = ref_col,
           ci.arg = list(col = adjustcolor(ref_col, 0.20), border = NA))
      lines(at_std, pred_active$matfit[, l + 1], col = active_col, lwd = 2)
      lines(at_std, pred_active$matlow[,  l + 1], col = active_col, lwd = 1, lty = 2)
      lines(at_std, pred_active$mathigh[, l + 1], col = active_col, lwd = 1, lty = 2)
      axis(1, at = at_std, labels = round(at_orig, 2))
      abline(h = 0, lty = 2, col = "grey50")
      dev.off()
    }

    # ── 3-D surfaces (reference and active, shared z-scale) ──────────────────
    z_global_ix <- range(pred_ref$matfit, pred_active$matfit, na.rm = TRUE)
    z_breaks_ix <- seq(z_global_ix[1], z_global_ix[2], length.out = 51)
    pal         <- colorRampPalette(c("firebrick", "white", "steelblue"))(50)

    for (grp in list(list(pred = pred_ref, name = "ref"), list(pred = pred_active, name = "active"))) {
      z_mat   <- grp$pred$matfit
      z_mid   <- (z_mat[-1, -1] + z_mat[-1, -ncol(z_mat)] +
                  z_mat[-nrow(z_mat), -1] + z_mat[-nrow(z_mat), -ncol(z_mat)]) / 4
      fcol    <- pal[cut(z_mid, breaks = z_breaks_ix, include.lowest = TRUE)]
      png(file.path(output_dir, paste0("dlnm_ix_3d_", label, "_", grp$name, "_", run_suffix, ".png")),
          width = 800, height = 700)
      persp(x = at_orig, y = lag_seq, z = z_mat,
            zlim     = z_global_ix,
            xlab     = dlnm_var, ylab = "Lag (months)", zlab = "Effect on log-odds of p_bt",
            main     = paste0("DLNM surface — ", label, " (", grp$name, ")"),
            theta    = 40, phi = 25, ltheta = 45,
            col      = fcol, border = NA, ticktype = "detailed")
      dev.off()
    }

    cat(sprintf("  DLNM interaction plots saved: %s\n", label))
  }
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

