# =====================================================
# Helper functions for Stan entomological model
# Data preparation, posterior extraction, and plotting
# =====================================================

# =========================
# DATA PREPARATION
# =========================

#' Load Base Entomological Data
#'
#' Reads CSV data file and creates a proper date column from year_month.
#' Removes unnecessary columns (CMF, CP, AREA) and reorders columns.
#'
#' @param data_file Character string path to the CSV data file
#' @return A data frame with cleaned entomological data including year_month_date
load_base_data <- function(data_file) {
  read_csv(data_file, show_col_types = FALSE) %>%
    mutate(year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d")) %>%
    relocate(year_month_date, .after = year_month) %>%
    select(!c(CMF, CP, AREA))
}

#' Index and Subset Data by Blocks
#'
#' Creates numeric block and time indices, renames key variables to Stan conventions,
#' and optionally subsets to a specified number of blocks. Ensures data is sorted
#' by block and time.
#'
#' @param input_data Data frame with manzana (block) and year_month_date columns
#' @param n_blocks Integer number of blocks to keep (NULL = keep all blocks)
#' @return List with elements: df (processed data frame), B (number of blocks), T (number of time points)
index_and_subset <- function(input_data, n_blocks) {
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

  all_blocks <- sort(unique(df$block))
  selected_blocks <- if (is.null(n_blocks)) {
    all_blocks
  } else {
    all_blocks[seq_len(min(n_blocks, length(all_blocks)))]
  }
  df <- df %>% filter(block %in% selected_blocks)

  time_levels <- sort(unique(df$year_month_date))
  df <- df %>% mutate(time = match(year_month_date, time_levels))

  list(df = df, B = length(selected_blocks), T = length(time_levels))
}

#' Build Distributed Lag Design Matrix
#'
#' Creates a 3D array of lagged variables (X_lag) for each observation, variable, and lag.
#' Flattens into a 2D matrix for Stan. Removes observations where lags cannot be computed
#' (first max_lag time points for each block). Handles block structure properly.
#'
#' @param df Data frame with block, time, and lagged variables
#' @param lag_vars Character vector of variable names to create lags for
#' @param B Integer number of blocks
#' @param max_lag Integer maximum lag order (L)
#' @return List with df (filtered data frame), X_lag_flat (N x K*(L+1) matrix), K (number of variables), Lp1 (L+1)
build_lag_design <- function(df, lag_vars, B, max_lag) {
  df <- df %>% mutate(across(all_of(lag_vars), ~coalesce(., 0)))
  K <- length(lag_vars)
  L <- max_lag
  Lp1 <- L + 1

  X_lag <- array(0, dim = c(nrow(df), K, Lp1))
  for (b in 1:B) {
    idx <- which(df$block == b)
    for (k in seq_len(K)) {
      x <- df[[lag_vars[k]]][idx]
      for (l in 0:L) {
        lagged <- if (l == 0) x else c(rep(NA_real_, l), x[1:(length(x) - l)])
        X_lag[idx, k, l + 1] <- lagged
      }
    }
  }

  rows_to_keep <- df$time > L
  df <- df[rows_to_keep, ]
  X_lag <- X_lag[rows_to_keep, , ]

  X_lag_flat <- matrix(0, nrow = nrow(df), ncol = K * Lp1)
  for (i in seq_len(nrow(df))) X_lag_flat[i, ] <- as.vector(X_lag[i, , ])

  list(df = df, X_lag_flat = X_lag_flat, K = K, Lp1 = Lp1)
}

#' Prepare Unlagged Covariates
#'
#' Extracts unlagged variables and standardizes continuous variables (z-score).
#' Binary variables are left unstandardized (0/1). Missing values are replaced with 0.
#'
#' @param df Data frame containing unlagged variables
#' @param unlagged_vars Character vector of all unlagged variable names
#' @param binary_unlagged_vars Character vector of binary variable names (subset of unlagged_vars)
#' @return List with df (data frame), X_unlagged_std (N x Ku standardized matrix), Ku (number of unlagged variables)
prepare_unlagged <- function(df, unlagged_vars, binary_unlagged_vars) {
  continuous_unlagged_vars <- setdiff(unlagged_vars, binary_unlagged_vars)
  df <- df %>% mutate(across(all_of(unlagged_vars), ~coalesce(., 0)))

  X_unlagged <- as.matrix(df[, unlagged_vars])
  X_unlagged_std <- X_unlagged

  if (length(continuous_unlagged_vars) > 0) {
    cont_idx <- match(continuous_unlagged_vars, colnames(X_unlagged))
    means <- colMeans(X_unlagged[, cont_idx, drop = FALSE])
    sds <- apply(X_unlagged[, cont_idx, drop = FALSE], 2, sd)
    sds[sds == 0 | is.na(sds)] <- 1
    X_unlagged_std[, cont_idx] <- scale(X_unlagged[, cont_idx, drop = FALSE], center = means, scale = sds)
  }

  list(df = df, X_unlagged_std = X_unlagged_std, Ku = ncol(X_unlagged))
}

#' Standardize Matrix (Z-score)
#'
#' Centers and scales each column to mean 0 and standard deviation 1.
#' Handles zero-variance columns by setting their scale to 1 (no scaling).
#'
#' @param x Numeric matrix to standardize
#' @return Standardized matrix with same dimensions as input
standardize_matrix <- function(x) {
  m <- colMeans(x)
  s <- apply(x, 2, sd)
  s[s == 0 | is.na(s)] <- 1
  scale(x, center = m, scale = s)
}

#' Build Complete Stan Data List
#'
#' Orchestrates the full data preparation pipeline: loads data, creates indices,
#' builds lag matrix, prepares unlagged covariates, and standardizes. Returns
#' both the Stan data list and the processed data frame.
#'
#' @param cfg List containing all configuration parameters (data_file, n_blocks, lag_vars,
#'            max_lag, unlagged_vars, binary_unlagged_vars, kappa)
#' @return List with stan_data (list ready for Stan) and df (processed data frame)
build_stan_data <- function(cfg) {
  input_data <- load_base_data(cfg$data_file)
  idx <- index_and_subset(input_data, cfg$n_blocks)

  # Standardize numeric vars on the raw time series before lagging,
  # so all lags of the same variable share the same mean/sd
  vars_to_std <- intersect(cfg$numeric_vars, names(idx$df))
  idx$df[, vars_to_std] <- standardize_matrix(as.matrix(idx$df[, vars_to_std]))

  lag <- build_lag_design(idx$df, cfg$lag_vars, idx$B, cfg$max_lag)
  binary_unlagged_vars <- setdiff(cfg$unlagged_vars, cfg$numeric_vars)
  unl <- prepare_unlagged(lag$df, cfg$unlagged_vars, binary_unlagged_vars)

  list(
    stan_data = list(
      N = nrow(unl$df),
      y = unl$df$y_bt,
      K = lag$K,
      Lp1 = lag$Lp1,
      X_lag_flat = lag$X_lag_flat,
      Ku = unl$Ku,
      X_unlagged = unl$X_unlagged_std,
      B = idx$B,
      T = idx$T,
      block = unl$df$block,
      time = unl$df$time,
      C_bt = unl$df$C_bt,
      n_bt = as.integer(unl$df$N_HH + cfg$kappa * unl$df$C_bt)
    ),
    df = unl$df
  )
}

# =========================
# MODEL FIT + EXTRACTION
# =========================

#' Create Initialization Function for Stan
#'
#' Returns a function that generates random initial values for MCMC chains.
#' Conditionally includes temporal random effect parameters (v_time_raw, sigma_v, rho)
#' only when temporal RE is enabled.
#'
#' @param stan_data List containing Stan data (used to determine dimensions)
#' @param use_temporal_re Logical flag indicating whether temporal RE is enabled in the model
#' @return Function that returns a list of initial values for one MCMC chain
make_init_fun <- function(stan_data, use_temporal_re, use_hsgp = FALSE,
                          use_time_RE = FALSE, use_spatial_AC = TRUE) {
  function() {
    init_vals <- list(
      alpha      = rnorm(1, -4.5, 0.4),
      w          = matrix(rnorm(stan_data$K * stan_data$Lp1, 0, 0.08), stan_data$K, stan_data$Lp1),
      sigma_w    = runif(stan_data$K, 0.1, 0.3),
      w_unlagged = rnorm(stan_data$Ku, 0, 0.1),
      delta0     = rnorm(1, 0.3, 0.2),
      delta1     = rnorm(1, 0, 0.1)
    )

    if (isTRUE(use_time_RE)) {
      # iid time RE + iid block RE model
      init_vals$v_time_raw  <- rnorm(stan_data$T, 0, 0.3)
      init_vals$sigma_time  <- runif(1, 0.05, 0.3)
      init_vals$u_block_raw <- rnorm(stan_data$B, 0, 0.3)
      init_vals$sigma_block <- runif(1, 0.05, 0.3)
    } else {
      # AR1 / GP model variants
      if (isTRUE(use_spatial_AC)) {
        init_vals$sigma_gp <- runif(1, 0.1, 0.5)
        init_vals$rho_gp   <- runif(1, 50, 200)
        if (isTRUE(use_hsgp)) {
          init_vals$beta_gp <- rnorm(stan_data$M^2, 0, 0.1)
        } else {
          init_vals$z_gp <- rnorm(stan_data$B, 0, 0.1)
        }
      }
      if (isTRUE(use_temporal_re)) {
        init_vals$v_global_raw    <- rnorm(stan_data$T, 0, 0.3)
        init_vals$v_block_dev_raw <- rnorm(stan_data$B, 0, 0.3)
        init_vals$sigma_v         <- runif(1, 0.1, 0.5)
        init_vals$sigma_block_dev <- runif(1, 0.05, 0.3)
        init_vals$rho             <- rnorm(1, 0.3, 0.15)
      }
    }

    init_vals
  }
}

#' Extract Posterior Means from Stan Fit
#'
#' Computes posterior means for key parameters and generated quantities using
#' flexible regex pattern matching. Returns NA vectors if a parameter is not found
#' (e.g., v_time_out when temporal RE is disabled).
#'
#' @param fit Stan fit object (cmdstanr)
#' @param n_rows Integer number of observations (for sizing output vectors)
#' @return List with elements: p_bt, p_R, u (spatial RE), v (temporal RE), y_pred
extract_means <- function(fit, n_rows) {
  post <- fit$draws(format = "matrix")
  vnames <- colnames(post)

  extract <- function(pattern, n_fallback = 1) {
    cols <- grep(pattern, vnames)
    if (length(cols) == 0) return(rep(NA_real_, n_fallback))
    colMeans(post[, cols])
  }

  list(
    p_bt = extract("^p_bt_out\\[", n_rows),
    p_R = extract("^p_R_out\\[", n_rows),
    u = extract("^u_block_out\\[", 1),
    v = extract("^v_time_out\\[", 1),
    y_pred = extract("^y_pred\\[", n_rows)
  )
}

# =========================
# PLOTTING
# =========================

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
    plot(v_post, type = "b", main = "Temporal Random Effects (v_t)",
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
