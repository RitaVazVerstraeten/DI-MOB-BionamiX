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
#' Creates a scatter plot of observed vs predicted y_bt values with a 1:1 reference line.
#' Skips plotting if predictions are all NA.
#'
#' @param df Data frame containing observed y_bt values
#' @param y_pred Numeric vector of predicted y_bt values (posterior means)
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filename
#' @return NULL (saves plot to PNG file or returns invisibly if predictions are NA)
save_ppc <- function(df, y_pred, output_dir, run_suffix) {
  if (all(is.na(y_pred))) return(invisible(NULL))

  p <- ggplot(data.frame(observed = df$y_bt, predicted = y_pred), aes(observed, predicted)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(x = "Observed y_bt", y = "Predicted y_bt (posterior mean)", title = "Posterior Predictive Check") +
    theme_minimal()

  ggsave(file.path(output_dir, paste0("posterior_predictive_check_", run_suffix, ".png")), p, width = 8, height = 6)
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
  timeseries_dir <- file.path(output_dir, "timeseries_plots")
  dir.create(timeseries_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Plot 1: Aggregate time series (mean across blocks)
  p1 <- df %>%
    group_by(year_month_date) %>%
    summarise(
      fitted_mean = mean(fitted_p_bt, na.rm = TRUE),
      observed_mean = mean(observed_p_bt, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    pivot_longer(cols = c(fitted_mean, observed_mean), names_to = "type", values_to = "probability") %>%
    ggplot(aes(x = year_month_date, y = probability, color = type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 1.5) +
    scale_color_manual(values = c("fitted_mean" = "blue", "observed_mean" = "red"),
                       labels = c("Fitted p_bt", "Observed p_bt")) +
    labs(x = "Time", y = "Probability",
         title = "Time Series: Observed vs Fitted Mosquito Probability (Mean Across Blocks)",
         color = NULL) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  ggsave(file.path(timeseries_dir, paste0("timeseries_aggregate_", run_suffix, ".png")), 
         p1, width = 12, height = 6, dpi = 150)
  
  # Plot 2: Block-specific time series (first n_blocks_facet blocks)
  block_ids <- sort(unique(df$block))[seq_len(min(n_blocks_facet, length(unique(df$block))))]
  p2 <- df %>%
    filter(block %in% block_ids) %>%
    pivot_longer(cols = c(fitted_p_bt, observed_p_bt), names_to = "type", values_to = "probability") %>%
    ggplot(aes(x = year_month_date, y = probability, color = type)) +
    geom_line(alpha = 0.7) +
    geom_point(size = 0.8, alpha = 0.7) +
    facet_wrap(~block, ncol = 3, scales = "free_y") +
    scale_color_manual(values = c("fitted_p_bt" = "blue", "observed_p_bt" = "red"),
                       labels = c("Fitted", "Observed")) +
    labs(x = "Time", y = "Probability",
         title = "Time Series by Block: Observed vs Fitted Mosquito Probability",
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
    summarise(correlation = cor(observed_p_bt, fitted_p_bt, use = "complete.obs"), .groups = "drop") %>%
    filter(!is.na(correlation))
  
  p4 <- ggplot(df_corr, aes(x = correlation)) +
    geom_histogram(bins = 20, fill = "steelblue", color = "black", alpha = 0.7) +
    geom_vline(xintercept = median(df_corr$correlation), linetype = "dashed", color = "red", linewidth = 1) +
    labs(x = "Correlation (Observed vs Fitted)",
         y = "Number of Blocks",
         title = "Correlation p_bt_observed vs p_bt_fitted over all timepoints",
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
