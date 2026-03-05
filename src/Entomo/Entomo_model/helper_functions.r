# =====================================================
# Helper functions for Stan entomological model
# Data preparation, posterior extraction, and plotting
# =====================================================

# =========================
# DATA PREPARATION
# =========================

load_base_data <- function(data_file) {
  read_csv(data_file, show_col_types = FALSE) %>%
    mutate(year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d")) %>%
    relocate(year_month_date, .after = year_month) %>%
    select(!c(CMF, CP, AREA))
}

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

standardize_matrix <- function(x) {
  m <- colMeans(x)
  s <- apply(x, 2, sd)
  s[s == 0 | is.na(s)] <- 1
  scale(x, center = m, scale = s)
}

build_stan_data <- function(cfg) {
  input_data <- load_base_data(cfg$data_file)
  idx <- index_and_subset(input_data, cfg$n_blocks)
  lag <- build_lag_design(idx$df, cfg$lag_vars, idx$B, cfg$max_lag)
  unl <- prepare_unlagged(lag$df, cfg$unlagged_vars, cfg$binary_unlagged_vars)

  list(
    stan_data = list(
      N = nrow(unl$df),
      y = unl$df$y_bt,
      N_HH = unl$df$N_HH,
      K = lag$K,
      Lp1 = lag$Lp1,
      X_lag_flat = standardize_matrix(lag$X_lag_flat),
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

make_init_fun <- function(stan_data, use_temporal_re) {
  function() {
    init_vals <- list(
      alpha = rnorm(1, -4.5, 0.4),
      w = matrix(rnorm(stan_data$K * stan_data$Lp1, 0, 0.08), stan_data$K, stan_data$Lp1),
      sigma_w = runif(stan_data$K, 0.1, 0.3),
      w_unlagged = rnorm(stan_data$Ku, 0, 0.1),
      u_block_raw = rnorm(stan_data$B, 0, 0.5),
      sigma_u = runif(1, 0.1, 0.5),
      delta0 = rnorm(1, 0.3, 0.2),
      delta1 = rnorm(1, 0, 0.1)
    )

    if (use_temporal_re) {
      init_vals$v_time_raw <- rnorm(stan_data$T, 0, 0.5)
      init_vals$sigma_v <- runif(1, 0.1, 0.5)
      init_vals$rho <- rnorm(1, 0, 0.25)
    }

    init_vals
  }
}

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

save_ppc <- function(df, y_pred, output_dir, run_suffix) {
  if (all(is.na(y_pred))) return(invisible(NULL))

  p <- ggplot(data.frame(observed = df$y_bt, predicted = y_pred), aes(observed, predicted)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(x = "Observed y_bt", y = "Predicted y_bt (posterior mean)", title = "Posterior Predictive Check") +
    theme_minimal()

  ggsave(file.path(output_dir, paste0("posterior_predictive_check_", run_suffix, ".png")), p, width = 8, height = 6)
}

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
