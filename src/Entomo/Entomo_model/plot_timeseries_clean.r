# =====================================================
# Plot time series from fitted Stan model (clean version)
# =====================================================

library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)

# =========================
# 0) SETTINGS
# =========================
hostname <- Sys.info()["nodename"]

cfg <- list(
  data_dir = if (hostname == "frietjes") "~/data/Entomo" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/",
  data_file_name = "env_epi_entomo_data_per_manzana_2016_04_to_2019_12.csv",
  results_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting",

  # fit selection
  fit_file_name = "fit_20260305_stand_noTimeRE.rds",

  # data prep
  n_blocks = 100,
  lag_vars = c("avg_temp", "rel_hum", "total_precip", "mean_ndvi"),
  max_lag = 1,

  # plotting
  n_blocks_facet = 9,
  output_subdir = "timeseries_plots"
)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)
cfg$fit_file <- file.path(cfg$results_dir, cfg$fit_file_name)
date_suffix <- format(Sys.Date(), "%Y%m%d")
run_suffix <- paste0(date_suffix, "_stand")

# =========================
# 1) DATA PREPARATION
# =========================
load_and_prepare_df <- function(data_file, n_blocks, lag_vars, max_lag) {
  input_data <- read_csv(data_file, show_col_types = FALSE)

  input_data <- input_data %>%
    mutate(year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d")) %>%
    relocate(year_month_date, .after = year_month) %>%
    select(!c(CMF, CP, AREA))

  block_levels <- sort(unique(input_data$manzana))
  time_levels <- sort(unique(input_data$year_month_date))

  df <- input_data %>%
    mutate(
      block = match(manzana, block_levels),
      time = match(year_month_date, time_levels)
    ) %>%
    arrange(block, time) %>%
    rename(N_HH = Inspected_houses, C_bt = cases, y_bt = Houses_pos_IS) %>%
    mutate(N_HH = as.integer(N_HH), C_bt = as.integer(C_bt), y_bt = as.integer(y_bt))

  selected_blocks <- sort(unique(df$block))[seq_len(min(n_blocks, length(unique(df$block))))]
  df <- df %>% filter(block %in% selected_blocks)

  time_levels <- sort(unique(df$year_month_date))
  df <- df %>% mutate(time = match(year_month_date, time_levels))

  # keep lag handling consistent with calibration
  df <- df %>% mutate(across(all_of(lag_vars), ~coalesce(., 0)))
  rows_to_keep <- df$time > max_lag
  df <- df[rows_to_keep, ]

  list(df = df, time_levels = time_levels)
}

extract_fitted_p_bt <- function(fit, n_expected) {
  post_matrix <- fit$draws(format = "matrix")
  p_bt_cols <- grep("^p_bt_out\\[", colnames(post_matrix))
  if (length(p_bt_cols) == 0) stop("p_bt_out not found in posterior")

  fitted <- colMeans(post_matrix[, p_bt_cols])
  if (length(fitted) != n_expected) {
    stop("Length mismatch: fitted_p_bt=", length(fitted), " vs df rows=", n_expected)
  }
  fitted
}

# =========================
# 2) PLOT FUNCTIONS
# =========================
plot_aggregate_timeseries <- function(df) {
  df_agg <- df %>%
    group_by(year_month_date) %>%
    summarise(
      fitted_mean = mean(fitted_p_bt, na.rm = TRUE),
      observed_mean = mean(observed_p_bt, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    pivot_longer(cols = c(fitted_mean, observed_mean), names_to = "type", values_to = "probability")

  ggplot(df_agg, aes(x = year_month_date, y = probability, color = type)) +
    geom_line(linewidth = 1) +
    geom_point(size = 1.5) +
    scale_color_manual(values = c("fitted_mean" = "blue", "observed_mean" = "red"),
                       labels = c("Fitted p_bt", "Observed p_bt")) +
    labs(x = "Time", y = "Probability",
         title = "Time Series: Observed vs Fitted Mosquito Probability (Mean Across Blocks)",
         color = NULL) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

plot_block_timeseries <- function(df, n_blocks_facet = 9) {
  block_ids <- sort(unique(df$block))[seq_len(min(n_blocks_facet, length(unique(df$block))))]

  df_blocks <- df %>%
    filter(block %in% block_ids) %>%
    pivot_longer(cols = c(fitted_p_bt, observed_p_bt), names_to = "type", values_to = "probability")

  ggplot(df_blocks, aes(x = year_month_date, y = probability, color = type)) +
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
}

plot_residuals_over_time <- function(df) {
  df_resid <- df %>%
    mutate(residual = observed_p_bt - fitted_p_bt) %>%
    group_by(year_month_date) %>%
    summarise(
      mean_residual = mean(residual, na.rm = TRUE),
      sd_residual = sd(residual, na.rm = TRUE),
      .groups = "drop"
    )

  ggplot(df_resid, aes(x = year_month_date, y = mean_residual)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_line(color = "darkred", linewidth = 1) +
    geom_point(color = "darkred", size = 2) +
    geom_ribbon(aes(ymin = mean_residual - sd_residual, ymax = mean_residual + sd_residual),
                alpha = 0.2, fill = "darkred") +
    labs(x = "Time", y = "Mean Residual (Observed - Fitted)",
         title = "Residuals Over Time (Mean ± SD Across Blocks)") +
    theme_minimal()
}

plot_correlation_distribution <- function(df) {
  df_corr <- df %>%
    group_by(block) %>%
    summarise(correlation = cor(observed_p_bt, fitted_p_bt, use = "complete.obs"), .groups = "drop")
  df_corr_clean <- df_corr %>% filter(!is.na(correlation))

  p <- ggplot(df_corr_clean, aes(x = correlation)) +
    geom_histogram(bins = 20, fill = "steelblue", color = "black", alpha = 0.7) +
    geom_vline(xintercept = median(df_corr_clean$correlation), linetype = "dashed", color = "red", linewidth = 1) +
    labs(x = "Correlation (Observed vs Fitted)",
         y = "Number of Blocks",
         title = "Correlation p_bt_observed vs p_bt_fitted over all timepoints",
         subtitle = paste0("Median correlation: ", round(median(df_corr_clean$correlation), 3),
                           " (", nrow(df_corr_clean), "/", nrow(df_corr), " blocks with valid correlation)")) +
    theme_minimal()

  list(plot = p, df_corr = df_corr, df_corr_clean = df_corr_clean)
}

save_all_plots <- function(df, output_dir, run_suffix, n_blocks_facet) {
  p1 <- plot_aggregate_timeseries(df)
  ggsave(file.path(output_dir, paste0("timeseries_aggregate_", run_suffix, ".png")), p1, width = 12, height = 6, dpi = 150)

  p2 <- plot_block_timeseries(df, n_blocks_facet)
  ggsave(file.path(output_dir, paste0("timeseries_by_block_", run_suffix, ".png")), p2, width = 14, height = 10, dpi = 150)

  p3 <- plot_residuals_over_time(df)
  ggsave(file.path(output_dir, paste0("residuals_over_time_", run_suffix, ".png")), p3, width = 12, height = 6, dpi = 150)

  corr_obj <- plot_correlation_distribution(df)
  ggsave(file.path(output_dir, paste0("correlation_distribution_", run_suffix, ".png")),
         corr_obj$plot, width = 10, height = 6, dpi = 150)

  corr_obj
}

print_summary_stats <- function(df) {
  cat("\nSummary statistics:\n")
  cat("  Overall correlation:", round(cor(df$observed_p_bt, df$fitted_p_bt), 3), "\n")
  cat("  RMSE:", round(sqrt(mean((df$observed_p_bt - df$fitted_p_bt)^2)), 4), "\n")
  cat("  MAE:", round(mean(abs(df$observed_p_bt - df$fitted_p_bt)), 4), "\n")
}

# =========================
# 3) MAIN
# =========================
if (!file.exists(cfg$fit_file)) {
  stop("Fit file not found: ", cfg$fit_file)
}

cat("Loading fit object from:", cfg$fit_file, "\n")
fit <- readRDS(cfg$fit_file)

prep <- load_and_prepare_df(cfg$data_file, cfg$n_blocks, cfg$lag_vars, cfg$max_lag)
df <- prep$df
df$fitted_p_bt <- extract_fitted_p_bt(fit, nrow(df))
df$observed_p_bt <- df$y_bt / df$N_HH

output_dir <- file.path(cfg$results_dir, cfg$output_subdir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("\nGenerating time series plots...\n")
corr_obj <- save_all_plots(df, output_dir, run_suffix, cfg$n_blocks_facet)

cat("\nAll plots saved to:", output_dir, "\n")
print_summary_stats(df)
