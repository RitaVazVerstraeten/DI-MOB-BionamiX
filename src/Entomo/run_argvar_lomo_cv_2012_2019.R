# run_argvar_lomo_cv_2012_2019.R
#
# Genuine leave-one-month-out (LOMO) cross-validation for argvar/interaction
# model selection on the EXTENDED 2012-2019 dataset (run_interaction_sweep_
# 2012_2019.R's data), not the 2015-2019 one run_argvar_lomo_cv.R uses.
#
# 8 configs total:
#   - a_resid, b_resid, c_resid: three argvar_df choices (a/b/c, same
#     per-variable df combinations as run_argvar_lomo_cv.R's original grid),
#     each with ONLY precip_resid_x_season active.
#   - a_tp, b_tp, c_tp: the same three argvar_df choices, each with ONLY
#     tp_x_season active instead. resid and tp are never combined in the same
#     fit -- single-interaction tests only, 6 configs from 3 argvar choices.
#   - d: a's argvar_df (tp=3, vpd=3, resid=3) with NO interaction -- the
#     no-interaction counterpart to a_resid/a_tp, isolating whether the
#     season interaction itself helps for that specific argvar choice.
#   - e: a second, distinct no-interaction baseline, argvar_df unchanged from
#     run_argvar_lomo_cv.R's original grid.
# No boundary-knots doubling this time (unlike run_argvar_lomo_cv.R's 9x2=18)
# -- 8 configs, by design.
#
# Held-out months: 1/4 of the available response months (NOT full LOMO --
# the 2012-2019 response period is ~96 months vs the 2015-2019 grid's 48, and
# a single fit here is far more expensive -- see cost warning below), evenly
# spaced across the full period for year/season coverage, same deterministic
# no-RNG approach run_argvar_lomo_cv.R used for its original 12-month subset.
# ~96 months / 4 =~ 24 held-out months (resolved dynamically at runtime from
# the actual data, not hardcoded).
#
# COST WARNING -- read before running: a single fold of THIS script was
# measured at 8200s (~2.3h) per fit on a compute node at iter_warmup=1000,
# iter_sampling=1000, 4 chains -- this is a real, measured number, not a
# guess. iter_warmup has since been reduced to 500 below; at ~1500 total
# iterations instead of ~2000, a rough linear-scaling estimate is ~6150s
# (~1.7h)/fit, but warmup/sampling costs don't necessarily scale identically
# with iteration count -- time one fold at the new setting before trusting
# this. At 8 configs x ~24 held-out months = 192 folds, ~1.7h/fit works out
# to roughly 328h (~13.7 days) SEQUENTIAL. Results save incrementally after
# every (config, month) fold, so this is safe to interrupt and resume across
# multiple sessions -- already-completed combinations are skipped on rerun.
#
# See run_leave_one_month_out_cv.R for the full derivation of why
# leave-one-month-out (refitting with that month's rows entirely absent from
# the likelihood) is a safe, genuine test of this model's AR(1) structure --
# run_one_fold() below is copied verbatim from run_argvar_lomo_cv.R, it's
# fully config-agnostic. This script only differs in HOW the config grid is
# built (8 scenarios, not 18) and which dataset/held-out-month count are used
# (2012-2019 data, 1/4-month subsample, not 2015-2019's full-month LOMO).
#
# Usage: Rscript run_argvar_lomo_cv_2012_2019.R
# (from inside src/Entomo/, or with that as the working directory, so renv
# resolves cmdstanr from the project library.)
# =============================================================================

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
# Fixed config -- matches run_interaction_sweep_2012_2019.R's data setup
# exactly (same dataset, response_start, lag/arglag spec), NOT run_argvar_
# lomo_cv.R's 2015-2019 one.
# =============================================================================
data_dir <- if (hostname == "frietjes") "~/data/Entomo" else if (hostname == "stoofvlees") "~/entomo_data" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data"
output_root <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/argvar_lomo_cv_2012_2019"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/argvar_lomo_cv_2012_2019"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/argvar_lomo_cv_2012_2019"
}
output_root <- path.expand(output_root)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

date_suffix <- format(Sys.Date(), "%Y%m%d")

data_file_name_fixed <- "env_epi_entomo_data_per_CMF_2012_01_to_2019_12_NDXIbackfilled_noColinnearity.csv"
response_start_fixed  <- "2012_01"

lag_vars_fixed     <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
dlnm_vars_fixed    <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
numeric_vars_fixed <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi")

# No "water_shortage" here -- run_interaction_sweep_2012_2019.R's unlagged_base
# deliberately excludes it for this dataset (not run_argvar_lomo_cv.R's list,
# which includes it for the 2015-2019 data); follow that convention rather
# than the 2015-2019 one.
unlagged_no_season   <- c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_containers")
unlagged_with_season <- c(unlagged_no_season, "is_rainy_season")

ix_resid <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "precip_max_day_resid_on_tp", label = "precip_resid_x_season")
ix_total <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "total_precip",               label = "tp_x_season")

arglag_df_fixed <- 3
max_lag_fixed   <- 5

# 9 configs: a/b/c/d each get TWO single-interaction variants -- resid-only
# (precip_resid_x_season) and tp-only (tp_x_season) -- never combined
# together in the same fit (dropped the earlier "_both" idea entirely). e
# remains the sole no-interaction ("none") baseline, unchanged.
#
# d is deliberately a's argvar (tp=3, vpd=3, resid=3) with NO interaction --
# the no-interaction counterpart to a_resid/a_tp, same spline flexibility,
# isolating whether the season interaction itself helps for that specific
# argvar choice. Not a duplicate: a_resid/a_tp have dlnm_ix_vars set, d does
# not.
base_scenarios <- list(
  a = list(total_precip = 3, avg_VPD = 3, precip_max_day_resid_on_tp = 3),
  b = list(total_precip = 3, avg_VPD = 3, precip_max_day_resid_on_tp = 2),
  c = list(total_precip = 3, avg_VPD = 2, precip_max_day_resid_on_tp = 2)
)
d_argvar <- base_scenarios$a   # same argvar_df as a, used with no interaction
e_argvar <- list(total_precip = 3, avg_VPD = 2, precip_max_day_resid_on_tp = 3)

build_dlnm_argvar <- function(spec) {
  out <- lapply(names(spec), function(var_name) {
    v <- spec[[var_name]]
    if (identical(v, "lin")) return(list(fun = "lin"))
    list(fun = "ns", df = v)
  })
  setNames(out, names(spec))
}

all_configs <- list()
for (nm in names(base_scenarios)) {
  argvar <- base_scenarios[[nm]]
  all_configs[[paste0(nm, "_resid")]] <- list(
    label         = paste0(nm, "_resid"),
    dlnm_argvar   = build_dlnm_argvar(argvar),
    dlnm_ix_vars  = list(ix_resid),
    unlagged_vars = unlagged_with_season
  )
  all_configs[[paste0(nm, "_tp")]] <- list(
    label         = paste0(nm, "_tp"),
    dlnm_argvar   = build_dlnm_argvar(argvar),
    dlnm_ix_vars  = list(ix_total),
    unlagged_vars = unlagged_with_season
  )
}
all_configs[["d"]] <- list(
  label         = "d",
  dlnm_argvar   = build_dlnm_argvar(d_argvar),
  dlnm_ix_vars  = NULL,
  unlagged_vars = unlagged_no_season
)
all_configs[["e"]] <- list(
  label         = "e",
  dlnm_argvar   = build_dlnm_argvar(e_argvar),
  dlnm_ix_vars  = NULL,
  unlagged_vars = unlagged_no_season
)

# Optional single-config worker mode -- lets up to 8 instances of this exact
# script run concurrently (one config each, 4 cores/fit via parallel_chains
# below), instead of one process working through all 8 configs sequentially.
# Unset (default): fits all 8 configs in this one process, then runs the
# aggregation section at the end, same as before.
# Set to one of the 8 labels below: fits ONLY that config's held-out months,
# then EXITS without aggregating (every worker would otherwise race to
# write the same comparison files at slightly different times). Once all
# workers are done, run the script ONE more time with the env var unset --
# every fold file already exists, so the fitting loop skips instantly and it
# goes straight to assembling the final comparison from everything on disk.
#
# Example launch (8-way parallel on an idle 36-core host, 8x4=32/36 cores
# used, wall-clock roughly the time for ONE config's ~24 months instead of
# all 8's, i.e. ~13.7 sequential days collapses to ~1.7 days):
#   for cfg in a_resid a_tp b_resid b_tp c_resid c_tp d e; do
#     ARGVAR_LOMO_CONFIG=$cfg Rscript run_argvar_lomo_cv_2012_2019.R > log_$cfg.txt 2>&1 &
#   done
#   wait
#   Rscript run_argvar_lomo_cv_2012_2019.R   # env var unset -> aggregate everything
ARGVAR_LOMO_CONFIG <- Sys.getenv("ARGVAR_LOMO_CONFIG", unset = "")
configs <- all_configs
if (nzchar(ARGVAR_LOMO_CONFIG)) {
  if (!ARGVAR_LOMO_CONFIG %in% names(all_configs))
    stop(sprintf("ARGVAR_LOMO_CONFIG=%s is not one of the 8 configs (%s)",
                  ARGVAR_LOMO_CONFIG, paste(names(all_configs), collapse = ", ")))
  configs <- all_configs[ARGVAR_LOMO_CONFIG]
}
cat(sprintf("Fitting %d config%s (%s) on this host.\n",
            length(configs), if (length(configs) == 1) "" else "s",
            paste(names(configs), collapse = ", ")))

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
# Copied verbatim from run_argvar_lomo_cv.R / run_leave_one_month_out_cv.R --
# fully config-agnostic, only needs cfg/prep/stan_data/df/mod/m/fold_file.
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

  # No output_dir passed above, so CmdStan writes to tempdir(), uncleaned
  # otherwise until the whole multi-day/multi-week script exits.
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
    data_file_name = data_file_name_fixed,
    spatial_level = "CMF", block_col = "cmf",
    response_start = response_start_fixed, n_blocks = NULL,
    lag_vars = lag_vars_fixed, dlnm_vars = dlnm_vars_fixed, numeric_vars = numeric_vars_fixed,
    unlagged_vars = cfg_i$unlagged_vars,
    dlnm_argvar = cfg_i$dlnm_argvar,
    dlnm_arglag = list(fun = "ns", df = arglag_df_fixed),
    max_lag = max_lag_fixed, kappa = 4,
    dlnm_ix_vars = cfg_i$dlnm_ix_vars,
    use_time_RE = FALSE, use_temporal_AR = TRUE, use_temporal_AR_perCMF = TRUE,
    use_spatial_AC = FALSE, use_hsgp = FALSE, use_icar = FALSE, use_bym2 = FALSE,
    use_block_dev = TRUE, use_dlnm = TRUE,
    fix_delta1 = FALSE, delta1_fixed = 0,
    fix_phi = FALSE, phi_fixed = 25,
    chains = 4, iter_warmup = 500, iter_sampling = 1000,
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
  # every config has the same response months, so this is config-invariant.
  # 1/4 of available months, evenly spaced across the full period (NOT full
  # LOMO -- see cost warning in the header), same deterministic no-RNG
  # even-spacing approach run_argvar_lomo_cv.R originally used.
  if (is.null(held_out_months)) {
    all_months <- sort(unique(df$year_month))
    n_held_out_months <- round(length(all_months) / 4)
    held_out_months <- all_months[round(seq(1, length(all_months), length.out = n_held_out_months))]
    cat(sprintf("Held-out months (shared across all %d configs, %d of %d available -- 1/4 subsample): %s\n",
                length(configs), length(held_out_months), length(all_months), paste(held_out_months, collapse = ", ")))
  }

  mod <- cmdstan_model(cfg$stan_file, force_recompile = FALSE)

  for (m in held_out_months) {
    fold_file <- file.path(out_dir, paste0("fold_", m, ".rds"))
    run_one_fold(cfg, prep, stan_data, df, mod, m, fold_file)
  }
}

# =============================================================================
# Assemble the overall comparison across all configs, using every fold found
# on disk (not just this run's -- picks up prior partial progress too).
# Skipped entirely in single-config worker mode (ARGVAR_LOMO_CONFIG set) --
# see the usage note above: every worker would otherwise race to write the
# same comparison files. Run the script once more with the env var unset
# once all workers are done, to produce the real aggregated comparison.
# =============================================================================
if (!nzchar(ARGVAR_LOMO_CONFIG)) {
cat("\n", strrep("=", 70), "\n")
cat("ASSEMBLING LEAVE-ONE-MONTH-OUT COMPARISON ACROSS 2012-2019 ARGVAR CONFIGS\n")
cat(strrep("=", 70), "\n\n")

config_labels <- names(all_configs)
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

  pseudo_result_list <- lapply(complete_configs, function(lbl) {
    list(pointwise = matrix(elpd_mat[, lbl], ncol = 1, dimnames = list(NULL, "elpd_loo")))
  })
  names(pseudo_result_list) <- complete_configs
  month_cluster_ids <- sub("\\|.*$", "", row_key)

  boot_cmp <- bootstrap_elpd_comparison(pseudo_result_list, cluster_ids = month_cluster_ids, n_boot = 4000)
  cat("\nBootstrap comparison (clustered by held-out month):\n")
  print(boot_cmp, digits = 3, row.names = FALSE)

  comp_dir <- file.path(output_root, paste0("lomo_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  comp_file <- file.path(comp_dir, paste0("lomo_comparison_", date_suffix, ".txt"))
  comp_output <- capture.output({
    cat("Leave-one-month-out comparison (2012-2019 argvar/interaction grid) —", date_suffix, "\n\n")
    cat(sprintf("%d held-out months (of %d available): %s\n\n",
                length(held_out_months), length(all_months), paste(held_out_months, collapse = ", ")))
    cat("Configs (in order, all", length(all_configs), "from the grid; ",
        length(complete_configs), "complete enough to compare so far):\n")
    for (i in seq_along(all_configs)) cat(sprintf("  %d. %s\n", i, config_labels[i]))
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
} else {
  cat(sprintf("\nWorker mode (ARGVAR_LOMO_CONFIG=%s) done -- skipping aggregation. Re-run with the env var unset once all workers finish.\n", ARGVAR_LOMO_CONFIG))
}
