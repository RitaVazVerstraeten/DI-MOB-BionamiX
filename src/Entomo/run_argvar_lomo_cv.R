# run_argvar_lomo_cv.R
#
# Genuine leave-one-month-out (LOMO) cross-validation across the FULL
# run_argvar_sweep.R grid: 9 base scenarios (per-variable argvar_df +
# interaction none/resid/both) x {unconstrained, boundary-knots} = 18
# configs, 12 held-out months each -- 216 full refits total.
#
# Why all 18 (not a PSIS-LOO-prefiltered shortlist): if this is going into
# the paper's model-selection section, filtering candidates with PSIS-LOO
# first -- a criterion this project has already shown is unreliable here,
# since total_precip/avg_VPD/precip_max_day_resid_on_tp are municipality-wide
# and duplicated across ~149 CMFs per month -- before applying the rigorous
# method to what's left would be a real methodological weak point. LOMO
# across every candidate avoids that.
#
# COST WARNING -- read before running: 18 configs x 12 held-out months x
# ~1-1.5h/fit = ~216-324h (9-13.5 days) SEQUENTIAL. Deliberately not
# parallelized (stoofvlees is shared; running several fits at once would
# take most of its 48 cores). Results save incrementally after every
# (config, month) fold, so this is safe to interrupt and resume across
# multiple sessions -- already-completed combinations are skipped on rerun.
# Expect this to take well over a week of wall-clock time; there is no way
# to compress that without either parallel execution or fewer
# configs/months, both explicitly ruled out for this run.
#
# See run_leave_one_month_out_cv.R for the full derivation of why
# leave-one-month-out (refitting with that month's rows entirely absent from
# the likelihood) is a safe, genuine test of this model's AR(1) structure --
# run_one_fold() below is copied verbatim from there, it's fully
# config-agnostic. This script only differs in HOW the config grid and
# held-out-month count are built (from run_argvar_sweep.R's 18-scenario
# grid instead of the original 8-config ix-arm grid).
#
# Usage: Rscript run_argvar_lomo_cv.R
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
# Fixed config -- identical to run_argvar_sweep.R
# =============================================================================
data_dir <- if (hostname == "frietjes") "~/data/Entomo" else if (hostname == "stoofvlees") "~/entomo_data" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data"
output_root <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/argvar_lomo_cv"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/argvar_lomo_cv"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/argvar_lomo_cv"
}
output_root <- path.expand(output_root)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

# Final comparison saved alongside run_argvar_sweep.R's own LOO/WAIC
# comparisons (loo_comparison_<date>/, waic_comparison_<date>/) -- same
# sweep_output_dir, same naming convention, so all three criteria for this
# 18-config grid sit side by side (matches run_leave_one_month_out_cv.R's
# convention for the original sweep).
date_suffix <- format(Sys.Date(), "%Y%m%d")
sweep_output_dir <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/argvar_sweep"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/argvar_sweep"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/argvar_sweep"
}
sweep_output_dir <- path.expand(sweep_output_dir)

lag_vars_fixed     <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
dlnm_vars_fixed    <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp")
numeric_vars_fixed <- c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi")

unlagged_no_season   <- c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_shortage", "water_containers")
unlagged_with_season <- c(unlagged_no_season, "is_rainy_season")

ix_resid <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "precip_max_day_resid_on_tp", label = "precip_resid_x_season")
ix_total <- list(binary_var = "is_rainy_season", active_level = 1, dlnm_var = "total_precip",               label = "tp_x_season")

arglag_df_fixed <- 3
max_lag_fixed   <- 5

# 9 base scenarios -- identical to run_argvar_sweep.R. See that script's
# header for the reasoning behind each. interaction is "resid", "none", or
# "both" (precip_resid_x_season AND tp_x_season together, scenario i).
scenario_defs <- list(
  a = list(argvar = list(total_precip = 3, avg_VPD = 3,   precip_max_day_resid_on_tp = 3), interaction = "resid"),
  b = list(argvar = list(total_precip = 3, avg_VPD = 3,   precip_max_day_resid_on_tp = 2), interaction = "resid"),
  c = list(argvar = list(total_precip = 3, avg_VPD = 2,   precip_max_day_resid_on_tp = 2), interaction = "resid"),
  d = list(argvar = list(total_precip = 3, avg_VPD = 3,   precip_max_day_resid_on_tp = 3), interaction = "none"),
  e = list(argvar = list(total_precip = 3, avg_VPD = 2,   precip_max_day_resid_on_tp = 3), interaction = "none"),
  f = list(argvar = list(total_precip = 2, avg_VPD = 2,   precip_max_day_resid_on_tp = 2), interaction = "resid"),
  g = list(argvar = list(total_precip = 2, avg_VPD = "lin", precip_max_day_resid_on_tp = 2), interaction = "resid"),
  h = list(argvar = list(total_precip = 2, avg_VPD = "lin", precip_max_day_resid_on_tp = 2), interaction = "none"),
  i = list(argvar = list(total_precip = 3, avg_VPD = 3,   precip_max_day_resid_on_tp = 3), interaction = "both")
)

# Boundary.knots per variable, in standardized (z-score) units -- identical
# to run_argvar_sweep.R; see that script's header for derivation.
boundary_knots_by_var <- list(
  total_precip                = c(-1.1024, 0.9946),
  avg_VPD                     = c(-1.3004, 1.4406),
  precip_max_day_resid_on_tp  = c(-0.8720, 1.5924)
)

build_dlnm_argvar <- function(spec, use_boundary = FALSE) {
  out <- lapply(names(spec), function(var_name) {
    v <- spec[[var_name]]
    if (identical(v, "lin")) return(list(fun = "lin"))
    argspec <- list(fun = "ns", df = v)
    if (use_boundary) argspec$Boundary.knots <- boundary_knots_by_var[[var_name]]
    argspec
  })
  setNames(out, names(spec))
}

argvar_tag <- function(spec) {
  paste0("tp", spec$total_precip, "vpd", spec$avg_VPD, "resid", spec$precip_max_day_resid_on_tp)
}

# All 18 (9 base x {unconstrained, boundary-knots}). all_configs is the full
# set, used unconditionally by the final aggregation step so it always looks
# for every config regardless of which subset THIS host fitted -- see the
# two-machine split below. configs (possibly filtered) is what this host's
# fitting loop actually runs.
all_configs <- list()
for (nm in names(scenario_defs)) {
  sc <- scenario_defs[[nm]]
  for (use_boundary in c(FALSE, TRUE)) {
    cfg_label <- if (use_boundary) paste0(nm, "_boundary") else nm
    all_configs[[cfg_label]] <- list(
      label         = cfg_label,
      dlnm_argvar   = build_dlnm_argvar(sc$argvar, use_boundary = use_boundary),
      dlnm_ix_vars  = switch(sc$interaction,
                             resid = list(ix_resid),
                             both  = list(ix_resid, ix_total),
                             none  = NULL),
      unlagged_vars = if (sc$interaction == "none") unlagged_no_season else unlagged_with_season
    )
  }
}

# Split the 18 configs across two machines running independently (each its
# own plain sequential loop -- no concurrent fits on either host, so this
# doesn't touch the "not parallelizing on shared stoofvlees" decision).
# stoofvlees runs the 9 base (unconstrained) configs; frietjes -- currently
# idle -- runs the 9 "_boundary" configs. Roughly halves wall-clock (each
# fit only needs 4 cores, well within either host's idle capacity, so a
# single fit runs at about the same speed on either machine even though
# frietjes has fewer cores total -- the core count only matters for
# concurrent fits, which neither host is doing here).
#
# Controlled via the ARGVAR_LOMO_VARIANT env var so the exact same script
# file runs unmodified on both machines:
#   stoofvlees: ARGVAR_LOMO_VARIANT=base     Rscript run_argvar_lomo_cv.R
#   frietjes:   ARGVAR_LOMO_VARIANT=boundary Rscript run_argvar_lomo_cv.R
# Unset (or any other value) runs all 18 on one machine, e.g. for a single-
# host resume after merging both hosts' fold files together. NOTE: the final
# aggregation step below only produces a complete 18-config comparison once
# BOTH hosts' output_root folders (fold_*.rds under each config's
# subdirectory) have been merged onto one filesystem -- e.g.
# `rsync -av frietjes:~/data/entomo/results/fitting/stan/argvar_lomo_cv/ <this host's output_root>/`
# -- before re-running this script (or just its aggregation section).
run_variant <- Sys.getenv("ARGVAR_LOMO_VARIANT", unset = "all")
configs <- if (run_variant %in% c("base", "boundary")) {
  keep <- if (run_variant == "base") !grepl("_boundary$", names(all_configs)) else grepl("_boundary$", names(all_configs))
  all_configs[keep]
} else {
  all_configs
}
cat(sprintf("ARGVAR_LOMO_VARIANT=%s -- fitting %d of the 18 configs on this host (aggregation below always checks all 18).\n", run_variant, length(configs)))

# =============================================================================
# Which months to hold out -- SAME 12-month subset applied to every config,
# so per-config results stay directly comparable. Resolved to actual month
# labels below, right after the first config's prep is built.
# =============================================================================
n_held_out_months <- 12

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
# Copied verbatim from run_leave_one_month_out_cv.R -- fully config-agnostic,
# only needs cfg/prep/stan_data/df/mod/m/fold_file.
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

  # See run_leave_one_month_out_cv.R for why this cleanup matters: no
  # output_dir passed above, so CmdStan writes to tempdir(), uncleaned
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
    data_file_name = "env_epi_entomo_data_per_CMF_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv",
    spatial_level = "CMF", block_col = "cmf",
    response_start = "2016_01", n_blocks = NULL,
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
    held_out_months <- all_months[round(seq(1, length(all_months), length.out = n_held_out_months))]
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
cat("ASSEMBLING LEAVE-ONE-MONTH-OUT COMPARISON ACROSS ARGVAR-SWEEP CONFIGS\n")
cat(strrep("=", 70), "\n\n")

config_labels <- names(all_configs)   # always all 18, not just this host's subset -- see note above
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

  comp_dir <- file.path(sweep_output_dir, paste0("lomo_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  comp_file <- file.path(comp_dir, paste0("lomo_comparison_", date_suffix, ".txt"))
  comp_output <- capture.output({
    cat("Leave-one-month-out comparison (argvar sweep) —", date_suffix, "\n\n")
    cat(sprintf("%d held-out months (of 48 available): %s\n\n",
                length(held_out_months), paste(held_out_months, collapse = ", ")))
    cat("Configs (in order, all", length(all_configs), "from the argvar sweep grid; ",
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
