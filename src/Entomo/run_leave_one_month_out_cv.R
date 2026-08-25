# run_leave_one_month_out_cv.R
#
# Genuine leave-one-month-out (LOMO) cross-validation across a subset of the
# same grid as run_season_interaction_sweep.R (arglag ns(df=3/4) x
# max_lag=5 only [simplified from the sweep's 5/6 -- see max_lags below] x
# interaction arm none/resid/total/both -- 8 configs) -- a check on whether
# that sweep's PSIS-LOO comparison has been overstating how well the DLNM/interaction
# shape actually generalizes to unseen weather.
#
# Why LOO can't be trusted for that comparison: total_precip, avg_VPD, and
# precip_max_day_resid_on_tp are municipality-wide -- every CMF shares the
# exact same value in a given month. PSIS-LOO holds out one CMF-month row at
# a time, but ~148 other CMFs from that same month remain in the training
# set carrying nearly-identical predictor information. That's not a test of
# "does the fitted shape generalize to an unseen weather pattern" -- it's
# closer to "can we predict this one CMF's outcome having already seen 148
# others from the same month." Every LOO-based conclusion from the sweep
# (ns3df vs ns4df, none/resid/total/both) rests on that same optimistic
# assumption.
#
# What this does instead: for each (config, held-out month) pair, refit with
# that month's rows entirely absent from the likelihood (y, n_bt, X_cb, X_ix,
# X_unlagged, block, time, C_bt all subset to the training rows), then use
# the resulting posterior draws to compute the held-out predictive
# log-likelihood for that month's *actual* rows, evaluated completely
# outside that fold's own likelihood. See the single-config version of this
# script (git history) for the full derivation of why removing rows (rather
# than masking the likelihood inside Stan) is safe for this model's AR(1)
# structure -- summary: v[b, t] is built over a dense 1..T grid from a full
# B x T parameter matrix, independent of which rows have likelihood
# contributions, so a held-out month's v_level is estimated purely from the
# AR(1) prior interpolating between neighbouring in-sample months.
#
# COST WARNING -- read before running: this refits the full model once per
# (config, held-out month) pair. At ~1-1.5h per fit, even the reduced
# default of 8 held-out months x 8 configs (lag5 only) is ~60-95h (2.5-4
# days) sequential. held_out_months below defaults to a representative subset
# (every 6th response month, ~8 of 48) rather than the full 48 -- edit it to
# trade cost against precision. Results save incrementally after every
# (config, month) fold, so this can be safely interrupted and resumed
# (already-completed combinations are skipped on rerun).
#
# Usage: edit CONFIG / held_out_months below if needed, then:
#   Rscript run_leave_one_month_out_cv.R
# (from inside src/Entomo/, or with that as the working directory, so renv
# resolves cmdstanr from the project library.)

suppressMessages({
  library(cmdstanr)
  library(dplyr)
  library(readr)
})

script_dir <- tryCatch({
  p <- rstudioapi::getActiveDocumentContext()$path
  if (nzchar(p)) dirname(p) else stop("empty path")
}, error = function(e) tryCatch({
  frames <- sys.frames()
  for (f in rev(frames)) {
    if (!is.null(f$ofile) && nzchar(f$ofile))
      return(dirname(normalizePath(f$ofile, mustWork = FALSE)))
  }
  args <- commandArgs(trailingOnly = FALSE)
  fa   <- grep("--file=", args, value = TRUE)
  if (length(fa)) dirname(normalizePath(sub("--file=", "", fa[1]), mustWork = FALSE))
  else stop("no path")
}, error = function(e2) getwd()))

source(file.path(script_dir, "helper_functions.r"))

hostname <- Sys.info()["nodename"]
is_compute_node <- hostname %in% c("frietjes", "stoofvlees")

# =============================================================================
# Fixed config -- identical to run_season_interaction_sweep.R
# =============================================================================
data_dir  <- if (hostname == "frietjes") "~/data/Entomo" else if (hostname == "stoofvlees") "~/entomo_data" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data"
output_root <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/leave_one_month_out_cv"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/leave_one_month_out_cv"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/leave_one_month_out_cv"
}
output_root <- path.expand(output_root)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

# Final comparison is saved alongside this same sweep's existing LOO/WAIC
# comparisons (loo_comparison_<date>/, waic_comparison_<date>/) -- same root
# run_season_interaction_sweep.R itself writes to, same naming convention,
# so all three criteria for this 16-config grid sit side by side. Per-fold
# .rds progress files stay in output_root above (large/numerous, not really
# a "model selection criteria" artifact).
date_suffix <- format(Sys.Date(), "%Y%m%d")
sweep_output_dir <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/season_interaction_sweep"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/season_interaction_sweep"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/season_interaction_sweep"
}
sweep_output_dir <- path.expand(sweep_output_dir)

lag_vars_fixed     <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
dlnm_vars_fixed    <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
numeric_vars_fixed <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi")
dlnm_argvar_fixed  <- list(
  total_precip                = list(fun = "ns", df = 3),
  avg_temp                     = list(fun = "ns", df = 3),
  precip_max_day_resid_on_tp  = list(fun = "ns", df = 3)
)

unlagged_no_season   <- c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_shortage", "water_containers")
unlagged_with_season <- c(unlagged_no_season, "is_rainy_season")

ix_resid <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "precip_max_day_resid_on_tp", label = "precip_resid_x_season")
ix_total <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "total_precip",               label = "tp_x_season")

interaction_arms <- list(
  none  = list(ix_name = "none",  dlnm_ix_vars = NULL, unlagged_vars = unlagged_no_season),
  resid = list(ix_name = "resid", dlnm_ix_vars = list(ix_resid), unlagged_vars = unlagged_with_season),
  total = list(ix_name = "total", dlnm_ix_vars = list(ix_total), unlagged_vars = unlagged_with_season),
  both  = list(ix_name = "both",  dlnm_ix_vars = list(ix_resid, ix_total), unlagged_vars = unlagged_with_season)
)

arglag_dfs <- c(4)
max_lags   <- c(5)   # simplified from c(5, 6) -- lag6 dropped to cut the grid roughly in half

configs <- list()
for (arglag_df in arglag_dfs) {
  for (ml in max_lags) {
    for (arm in interaction_arms) {
      label <- sprintf("ns%ddf_lag%d_%s", arglag_df, ml, arm$ix_name)
      configs[[label]] <- list(
        label         = label,
        arglag_df     = arglag_df,
        max_lag       = ml,
        dlnm_ix_vars  = arm$dlnm_ix_vars,
        unlagged_vars = arm$unlagged_vars
      )
    }
  }
}
cat(sprintf("%d configs to run leave-one-month-out CV over.\n", length(configs)))

# =============================================================================
# Which months to hold out -- SAME subset applied to every config, so
# per-config results stay directly comparable. Defaults to a representative
# subset (see COST WARNING above); edit for full coverage or a cheaper trial.
# Resolved to actual month labels below, right after the first config's
# prep is built (need a real df$year_month to sample from).
# =============================================================================
n_held_out_months <- 1   # <- set to Inf (or a larger number) for full coverage; smaller for a quick trial

options(mc.cores = if (is_compute_node) 6 else 2)

dbetabinom_log <- function(y, n, a, b) {
  a <- pmax(a, 1e-6); b <- pmax(b, 1e-6)
  lchoose(n, y) + lbeta(y + a, n - y + b) - lbeta(a, b)
}
log_mean_exp_rows <- function(x) {
  m <- apply(x, 1, max)
  m + log(rowMeans(exp(x - m)))
}

# =============================================================================
# Per-config, per-month fold: refit with that month held out, score it.
# Returns NULL (and skips) if this fold's output already exists on disk.
# =============================================================================
run_one_fold <- function(cfg, prep, stan_data, df, mod, m, fold_file) {
  if (file.exists(fold_file)) {
    cat(sprintf("  [skip] %s already done\n", basename(fold_file)))
    return(invisible(NULL))
  }

  heldout_idx <- which(df$year_month == m)
  train_idx   <- setdiff(seq_len(stan_data$N), heldout_idx)

  sd_train <- stan_data
  sd_train$N          <- length(train_idx)
  sd_train$y          <- stan_data$y[train_idx]
  sd_train$n_bt       <- stan_data$n_bt[train_idx]
  sd_train$X_cb       <- stan_data$X_cb[train_idx, , drop = FALSE]
  sd_train$X_ix       <- if (stan_data$P_ix > 0) stan_data$X_ix[train_idx, , drop = FALSE] else stan_data$X_ix
  sd_train$X_unlagged <- stan_data$X_unlagged[train_idx, , drop = FALSE]
  sd_train$block      <- stan_data$block[train_idx]
  sd_train$time       <- stan_data$time[train_idx]
  sd_train$C_bt       <- stan_data$C_bt[train_idx]

  fit_m <- mod$sample(
    data            = sd_train,
    chains          = cfg$chains,
    iter_warmup     = cfg$iter_warmup,
    iter_sampling   = cfg$iter_sampling,
    init            = make_init_fun(
      sd_train, cfg$use_temporal_AR,
      use_hsgp               = isTRUE(cfg$use_hsgp) && !isTRUE(cfg$use_icar) && !isTRUE(cfg$use_bym2),
      use_icar               = isTRUE(cfg$use_icar) && !isTRUE(cfg$use_bym2),
      use_bym2               = isTRUE(cfg$use_bym2),
      use_time_RE            = isTRUE(cfg$use_time_RE),
      use_spatial_AC         = isTRUE(cfg$use_spatial_AC),
      use_block_dev          = isTRUE(cfg$use_block_dev),
      use_temporal_AR_perCMF = isTRUE(cfg$use_temporal_AR_perCMF),
      use_dlnm               = isTRUE(cfg$use_dlnm)
    ),
    adapt_delta     = cfg$adapt_delta,
    max_treedepth   = cfg$max_treedepth,
    parallel_chains = cfg$parallel_chains
  )

  # No output_dir was passed above, so CmdStan wrote this fold's raw chain
  # CSVs to R's default tempdir() -- which, across ~64 sequential folds in
  # one long-running session, would otherwise only get cleaned up when the
  # whole multi-day script exits. generated quantities includes two full
  # B x T matrices (v_cmf_out, v_level_out) per draw, so each fold's CSVs
  # run to the order of ~1-2GB; left uncleaned across every fold that adds
  # up to a lot of accumulated disk on a shared machine for no reason, since
  # only a handful of the generated quantities are actually needed below.
  # Delete them the moment the required draws are pulled out.
  csv_paths <- fit_m$output_files()

  max_rhat <- max(fit_m$summary(c("alpha", "w_cb", "w_unlagged", "tau", "rho"))$rhat, na.rm = TRUE)

  w_cb        <- fit_m$draws("w_cb", format = "matrix")
  w_unlagged  <- fit_m$draws("w_unlagged", format = "matrix")
  w_ix        <- if (stan_data$P_ix > 0) fit_m$draws("w_ix", format = "matrix") else matrix(0, nrow = nrow(w_cb), ncol = 0)
  alpha_draws <- as.vector(fit_m$draws("alpha", format = "matrix"))
  u_block     <- fit_m$draws("u_block_out", format = "matrix")
  v_level     <- fit_m$draws("v_level_out", format = "matrix")
  delta1_draws <- if (!isTRUE(cfg$fix_delta1)) as.vector(fit_m$draws("delta1", format = "matrix")) else rep(cfg$delta1_fixed, nrow(w_cb))
  phi_draws    <- if (!isTRUE(cfg$fix_phi)) as.vector(fit_m$draws("phi", format = "matrix")) else rep(sd_train$phi_data, nrow(w_cb))

  invisible(file.remove(csv_paths[file.exists(csv_paths)]))

  n_draws <- nrow(w_cb)

  ho_X_cb       <- stan_data$X_cb[heldout_idx, , drop = FALSE]
  ho_X_ix       <- if (stan_data$P_ix > 0) stan_data$X_ix[heldout_idx, , drop = FALSE] else matrix(0, nrow = length(heldout_idx), ncol = 0)
  ho_X_unlagged <- stan_data$X_unlagged[heldout_idx, , drop = FALSE]
  ho_block      <- stan_data$block[heldout_idx]
  ho_time       <- stan_data$time[heldout_idx]
  ho_C_bt       <- stan_data$C_bt[heldout_idx]
  ho_n_bt       <- stan_data$n_bt[heldout_idx]
  ho_y          <- stan_data$y[heldout_idx]
  n_ho          <- length(heldout_idx)

  x_effect <- ho_X_cb %*% t(w_cb) + ho_X_unlagged %*% t(w_unlagged) +
    (if (stan_data$P_ix > 0) ho_X_ix %*% t(w_ix) else matrix(0, n_ho, n_draws))

  v_level_cols <- paste0("v_level_out[", ho_block, ",", ho_time, "]")
  v_level_ho   <- t(v_level[, v_level_cols, drop = FALSE])
  u_block_cols <- paste0("u_block_out[", ho_block, "]")
  u_block_ho   <- t(u_block[, u_block_cols, drop = FALSE])

  eta  <- sweep(x_effect + v_level_ho + u_block_ho, 2, alpha_draws, `+`)
  p_bt <- plogis(eta)

  has_reactive <- ho_C_bt > 0
  p_R <- p_bt
  if (any(has_reactive)) {
    log_C_bt_rep <- matrix(log(ho_C_bt[has_reactive]), nrow = sum(has_reactive), ncol = n_draws)
    delta1_rep   <- matrix(delta1_draws, nrow = sum(has_reactive), ncol = n_draws, byrow = TRUE)
    p_R[has_reactive, ] <- plogis(eta[has_reactive, , drop = FALSE] + delta1_rep * log_C_bt_rep)
  }

  omega <- matrix(0, n_ho, n_draws)
  pi_mat <- p_bt
  reactive_and_observed <- has_reactive & ho_n_bt > 0
  if (any(reactive_and_observed)) {
    kappa_C_over_n <- pmin(1, cfg$kappa * ho_C_bt[reactive_and_observed] / ho_n_bt[reactive_and_observed])
    omega[reactive_and_observed, ] <- kappa_C_over_n
    om_rep <- matrix(kappa_C_over_n, nrow = sum(reactive_and_observed), ncol = n_draws)
    pi_mat[reactive_and_observed, ] <- (1 - om_rep) * p_bt[reactive_and_observed, , drop = FALSE] +
      om_rep * p_R[reactive_and_observed, , drop = FALSE]
  }
  zero_n <- ho_n_bt == 0
  if (any(zero_n)) pi_mat[zero_n, ] <- 0

  phi_rep <- matrix(phi_draws, nrow = n_ho, ncol = n_draws, byrow = TRUE)
  y_rep   <- matrix(ho_y,   nrow = n_ho, ncol = n_draws)
  n_rep   <- matrix(ho_n_bt, nrow = n_ho, ncol = n_draws)

  log_lik_ho <- dbetabinom_log(y_rep, n_rep, pi_mat * phi_rep, (1 - pi_mat) * phi_rep)
  elpd_i     <- log_mean_exp_rows(log_lik_ho) - log(n_draws)

  cat(sprintf("  month %s: held-out elpd = %.2f (%d rows, max Rhat = %.3f)\n",
              m, sum(elpd_i), n_ho, max_rhat))

  saveRDS(
    list(config = cfg$label, month = m, cmf = df$cmf[heldout_idx],
         elpd_i = elpd_i, max_rhat = max_rhat),
    fold_file
  )

  rm(fit_m, w_cb, w_unlagged, w_ix, v_level, u_block, log_lik_ho)
  gc()
  invisible(NULL)
}

# =============================================================================
# Main loop: outer over configs, inner over held-out months
# =============================================================================
held_out_months <- NULL   # resolved from the first config's actual months, below

for (i in seq_along(configs)) {
  cfg_i <- configs[[i]]
  cat("\n", strrep("=", 70), "\n")
  cat("CONFIG", i, "of", length(configs), ":", cfg_i$label, "\n")
  cat(strrep("=", 70), "\n\n")

  cfg <- list(
    data_dir = data_dir,
    data_file_name = "env_epi_entomo_data_per_CMF_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv",
    spatial_level = "CMF", block_col = "cmf",
    response_start = "2016_01", n_blocks = NULL,
    lag_vars = lag_vars_fixed, dlnm_vars = dlnm_vars_fixed, numeric_vars = numeric_vars_fixed,
    unlagged_vars = cfg_i$unlagged_vars,
    dlnm_argvar = dlnm_argvar_fixed,
    dlnm_arglag = list(fun = "ns", df = cfg_i$arglag_df),
    max_lag = cfg_i$max_lag, kappa = 4,
    dlnm_ix_vars = cfg_i$dlnm_ix_vars,
    use_time_RE = FALSE, use_temporal_AR = TRUE, use_temporal_AR_perCMF = TRUE,
    use_spatial_AC = FALSE, use_hsgp = FALSE, use_icar = FALSE, use_bym2 = FALSE,
    use_block_dev = TRUE, use_dlnm = TRUE,
    fix_delta1 = FALSE, delta1_fixed = 0,
    fix_phi = FALSE, phi_fixed = 25,
    chains = 4, iter_warmup = 1000, iter_sampling = 1000,
    adapt_delta = 0.95, max_treedepth = 12,
    parallel_chains = if (is_compute_node) 4 else 1,
    shrinkage_prior = "student_t",
    label = cfg_i$label
  )
  cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)
  cfg$stan_file <- file.path(script_dir, "hierarchical_state_space_AR_perCMF_blockRE_DLNM_ix.stan")

  out_dir <- file.path(output_root, cfg$label)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  prep      <- build_dlnm_stan_data(cfg)
  stan_data <- prep$stan_data
  df        <- prep$df
  stan_data$fix_phi  <- as.integer(isTRUE(cfg$fix_phi))
  stan_data$phi_data <- if (isTRUE(cfg$fix_phi)) cfg$phi_fixed else 1.0
  if (isTRUE(cfg$fix_delta1)) stan_data$delta1 <- cfg$delta1_fixed
  stopifnot(nrow(df) == stan_data$N)

  # Resolve the shared held-out month set once, from the first config --
  # every config has the same 48 response months, so this is config-invariant.
  if (is.null(held_out_months)) {
    all_months <- sort(unique(df$year_month))
    held_out_months <- if (is.infinite(n_held_out_months)) {
      all_months
    } else {
      all_months[round(seq(1, length(all_months), length.out = n_held_out_months))]
    }
    cat(sprintf("Held-out months (shared across all %d configs): %s\n",
                length(configs), paste(held_out_months, collapse = ", ")))
  }

  mod <- cmdstan_model(cfg$stan_file, force_recompile = FALSE)

  for (m in held_out_months) {
    fold_file <- file.path(out_dir, paste0("fold_", m, ".rds"))
    run_one_fold(cfg, prep, stan_data, df, mod, m, fold_file)
  }
}

# =============================================================================
# Assemble the overall comparison across all configs, using every fold found
# on disk (not just this run's -- picks up prior partial progress too)
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("ASSEMBLING LEAVE-ONE-MONTH-OUT COMPARISON ACROSS CONFIGS\n")
cat(strrep("=", 70), "\n\n")

config_labels <- names(configs)
per_config_folds <- lapply(config_labels, function(lbl) {
  out_dir <- file.path(output_root, lbl)
  ff <- file.path(out_dir, paste0("fold_", held_out_months, ".rds"))
  ff <- ff[file.exists(ff)]
  lapply(ff, readRDS)
})
names(per_config_folds) <- config_labels

n_done <- sapply(per_config_folds, length)
cat("Folds completed per config (of", length(held_out_months), "):\n")
print(n_done)

complete_configs <- config_labels[n_done == length(held_out_months)]
if (length(complete_configs) < 2) {
  cat("\nFewer than 2 configs have every held-out month done yet -- re-run this script to continue; comparison will assemble once at least 2 configs are complete.\n")
} else {
  # Build a common row order (month, cmf) shared across all complete configs,
  # from the first complete config's fold files, and align every other
  # config's elpd_i onto that same order before comparing.
  ref_folds <- per_config_folds[[complete_configs[1]]]
  row_key   <- unlist(lapply(ref_folds, function(f) paste(f$month, f$cmf, sep = "|")))

  elpd_mat <- sapply(complete_configs, function(lbl) {
    folds <- per_config_folds[[lbl]]
    keys  <- unlist(lapply(folds, function(f) paste(f$month, f$cmf, sep = "|")))
    vals  <- unlist(lapply(folds, `[[`, "elpd_i"))
    vals[match(row_key, keys)]
  })
  colnames(elpd_mat) <- complete_configs

  totals <- colSums(elpd_mat)
  cat("\nTotal leave-one-month-out elpd per config (", length(held_out_months), "held-out months ):\n")
  print(sort(totals, decreasing = TRUE))

  # Reuse bootstrap_elpd_comparison() (helper_functions.r) for a proper
  # elpd_diff / CI / prob_better comparison, same machinery already used for
  # this sweep's LOO comparison -- clustered by held-out month (not
  # per-observation), since rows sharing a held-out month share that fold's
  # posterior draws and aren't independent evidence for the comparison.
  pseudo_result_list <- lapply(complete_configs, function(lbl) {
    list(pointwise = matrix(elpd_mat[, lbl], ncol = 1, dimnames = list(NULL, "elpd_loo")))
  })
  names(pseudo_result_list) <- complete_configs
  month_cluster_ids <- sub("\\|.*$", "", row_key)

  boot_cmp <- bootstrap_elpd_comparison(pseudo_result_list, cluster_ids = month_cluster_ids, n_boot = 4000)
  cat("\nBootstrap comparison (clustered by held-out month):\n")
  print(boot_cmp, digits = 3, row.names = FALSE)

  # Saved alongside this sweep's existing loo_comparison_<date>/ and
  # waic_comparison_<date>/ folders -- same sweep_output_dir, same
  # "<criterion>_comparison_<date>" naming, so all three sit side by side.
  comp_dir <- file.path(sweep_output_dir, paste0("lomo_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  comp_file <- file.path(comp_dir, paste0("lomo_comparison_", date_suffix, ".txt"))
  comp_output <- capture.output({
    cat("Leave-one-month-out comparison —", date_suffix, "\n\n")
    cat(sprintf("%d held-out months (of 48 available): %s\n\n",
                length(held_out_months), paste(held_out_months, collapse = ", ")))
    cat("Configs (in order, all", length(configs), "from the sweep grid; ",
        length(complete_configs), "complete enough to compare so far):\n")
    for (i in seq_along(configs)) cat(sprintf("  %d. %s\n", i, config_labels[i]))
    cat("\nTotal leave-one-month-out elpd per config:\n")
    print(sort(totals, decreasing = TRUE))
    cat("\nBootstrap comparison (clustered by held-out month, 4000 resamples):\n")
    print(boot_cmp, digits = 3, row.names = FALSE)
  })
  writeLines(comp_output, comp_file)
  cat("\nSaved:", comp_file, "\n")

  boot_file <- file.path(comp_dir, paste0("lomo_bootstrap_comparison_", date_suffix, ".csv"))
  write.csv(boot_cmp, boot_file, row.names = FALSE)
  cat("Saved:", boot_file, "\n")

  saveRDS(pseudo_result_list, file.path(comp_dir, paste0("lomo_list_", date_suffix, ".rds")))
  saveRDS(elpd_mat, file.path(comp_dir, paste0("lomo_elpd_matrix_", date_suffix, ".rds")))
}
