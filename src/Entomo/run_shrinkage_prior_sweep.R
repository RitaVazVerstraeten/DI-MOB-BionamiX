# run_shrinkage_prior_sweep.R
#
# Model-selection sweep over the shrinkage-prior family for w_cb/w_ix/w_unlagged:
#   student_t(3, 0, .)  -- heavy-tailed, the original default
#   normal(0, .)        -- light-tailed
#   laplace(0, .)        -- double_exponential; sharp peak, heavier tail than normal
# Same nominal scale per coefficient block (1.0 for w_cb, 0.5 for w_ix/w_unlagged)
# across all three, so the comparison isolates tail shape.
#
# Only shrinkage_prior + output_dir are overridden -- everything else (arglag,
# max_lag, season interaction, ...) stays at whatever
# Hierarch_StateSpace_Entomo_model.r's own cfg currently has. Deliberate choice
# to run on current defaults now rather than waiting for the (separate)
# season-interaction sweep to pick a winner -- unlike
# run_season_interaction_sweep.R / run_exposure_response_functions_sweep.R,
# this script does NOT pin lag_vars/dlnm_vars/etc, so re-running it later
# after the base file's defaults change will compare priors on whatever the
# new defaults are, not on today's.
#
# Validation follows a specific checklist; leave-one-block-out /
# leave-a-time-window-out CV is explicitly EXCLUDED here (compute cost, given
# this session's history of disk/RAM crashes on long runs) -- relying on
# Pareto-k instead:
#   1. Pareto-k diagnostics (write_pareto_k_summary) + elpd_diff vs se_diff
#      (write_criterion_comparison's z-score column) -- is the LOO/WAIC winner
#      actually trustworthy, or riding on a few bad-k points / noise?
#   3. Posterior predictive lag-response curve at the median exposure,
#      overlaid across all 3 priors -- does the prior choice change the
#      actual reported curve, or just the marginal likelihood?
#   4. Posterior vs. prior density for each w_cb[k], overlaid across all 3
#      priors -- is student_t's heavy tail actually being used, or does the
#      posterior sit well within where normal/laplace's tails already reach?
#
# Sources Hierarch_StateSpace_Entomo_model.r once per prior. Results land in:
#   <output root>/shrinkage_prior_sweep/<predictor_spec>/<model_spec>/<run_suffix>/
# =============================================================================

library(loo)

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
}, error = function(e2) {
  candidate <- file.path(getwd(), "src", "Entomo")
  if (file.exists(file.path(candidate, "helper_functions.r"))) candidate else getwd()
}))

date_suffix <- format(Sys.Date(), "%Y%m%d")
hostname    <- Sys.info()["nodename"]

sweep_output_dir <- if (hostname == "frietjes") {
  "/home/rita/data/Entomo/fitting/stan/shrinkage_prior_sweep"
} else if (hostname == "stoofvlees") {
  "~/data/entomo/results/fitting/stan/shrinkage_prior_sweep"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan/shrinkage_prior_sweep"
}
dir.create(sweep_output_dir, recursive = TRUE, showWarnings = FALSE)

priors <- c("student_t", "normal", "laplace")

# =============================================================================
# Run all three priors
# =============================================================================
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))

loo_list   <- list()
waic_list  <- list()
run_labels <- character(length(priors))
run_dirs   <- character(length(priors))  # each run's run_output_dir, for the post-loop draws re-reads below

for (i in seq_along(priors)) {
  prior_i   <- priors[i]
  run_label <- paste0(date_suffix, "_shrinkage_", prior_i)
  run_labels[i] <- run_label

  cat("\n", strrep("=", 70), "\n")
  cat("CONFIG", i, "of", length(priors), ":", run_label, "\n")
  cat(strrep("=", 70), "\n\n")

  .hierarch_cfg_override <- list(
    shrinkage_prior = prior_i,
    output_dir      = sweep_output_dir
  )
  .hierarch_run_suffix <- run_label
  loo_result           <- NULL   # clear stale value; Hierarch will overwrite if fit succeeds
  waic_result          <- NULL

  tryCatch(
    eval(model_exprs, envir = globalenv()),
    error = function(e) cat("ERROR in config", i, "post-processing:", conditionMessage(e), "\n(loo_result/waic_result collected before error if LOO/WAIC completed)\n")
  )

  if (exists("loo_result") && !is.null(loo_result)) {
    loo_list[[run_label]] <- loo_result
    cat("LOO stored for:", run_label, "\n")
    saveRDS(loo_list, file.path(sweep_output_dir, "loo_list_partial.rds"))
  } else {
    cat("WARNING: loo_result not found after config", i, "— skipping LOO for this run.\n")
  }
  if (exists("waic_result") && !is.null(waic_result)) {
    waic_list[[run_label]] <- waic_result
    cat("WAIC stored for:", run_label, "\n")
    saveRDS(waic_list, file.path(sweep_output_dir, "waic_list_partial.rds"))
  } else {
    cat("WARNING: waic_result not found after config", i, "— skipping WAIC for this run.\n")
  }

  if (exists("run_output_dir")) run_dirs[i] <- run_output_dir

  # Clean up override variables
  rm(".hierarch_cfg_override", ".hierarch_run_suffix", envir = globalenv())
}

# =============================================================================
# Criterion comparison (LOO, then WAIC) -- elpd_diff / se_diff / z-score
# =============================================================================
write_criterion_comparison <- function(result_list, criterion_label, file_stub) {
  if (length(result_list) < 2) {
    cat("Fewer than 2 successful", criterion_label, "results — skipping comparison.\n")
    return(invisible(NULL))
  }
  cat("\n", strrep("=", 70), "\n")
  cat(criterion_label, "COMPARISON\n")
  cat(strrep("=", 70), "\n\n")

  comp <- loo_compare(result_list)
  print(comp, simplify = FALSE, digits = 2)

  cmp_df <- as.data.frame(comp)
  cmp_df$z_score <- cmp_df$elpd_diff / cmp_df$se_diff
  cmp_df$z_score[cmp_df$elpd_diff == 0] <- 0
  cat("\nz-score (elpd_diff / se_diff) -- |z| < 2 means the difference is within noise:\n")
  print(cmp_df["z_score"], digits = 2)

  comp_dir <- file.path(sweep_output_dir, paste0(file_stub, "_comparison_", date_suffix))
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  comp_file <- file.path(comp_dir, paste0(file_stub, "_comparison_", date_suffix, ".txt"))
  comp_output <- capture.output({
    cat(criterion_label, "comparison —", date_suffix, "\n\n")
    cat("Models (in order):\n")
    for (i in seq_along(run_labels)) cat(sprintf("  %d. %s\n", i, run_labels[i]))
    cat("\n")
    print(comp, simplify = FALSE, digits = 2)
    cat("\nz-score (elpd_diff / se_diff):\n")
    print(cmp_df["z_score"], digits = 2)
  })
  writeLines(comp_output, comp_file)
  cat("\n", criterion_label, "comparison saved to:", comp_file, "\n")

  saveRDS(result_list, file.path(comp_dir, paste0(file_stub, "_list_", date_suffix, ".rds")))
  invisible(file.remove(file.path(sweep_output_dir, paste0(file_stub, "_list_partial.rds"))))
}

write_criterion_comparison(loo_list,  "LOO",  "loo")
write_criterion_comparison(waic_list, "WAIC", "waic")

# =============================================================================
# Item 1: Pareto-k trustworthiness comparison
# =============================================================================
# A prior "winning" on elpd is only as trustworthy as its Pareto-k diagnostics
# let it be -- heavy-tailed priors in particular can produce a few
# highly-influential observations where the PSIS-LOO approximation itself
# breaks down (high k). Compare bad/very-bad k counts across priors alongside
# the elpd_diff/z-score above, not instead of it.
write_pareto_k_summary <- function(loo_list) {
  if (length(loo_list) == 0) {
    cat("No LOO results collected — skipping Pareto-k comparison.\n")
    return(invisible(NULL))
  }
  cat("\n", strrep("=", 70), "\n")
  cat("PARETO-K TRUSTWORTHINESS COMPARISON (item 1)\n")
  cat(strrep("=", 70), "\n\n")

  summary_df <- do.call(rbind, lapply(names(loo_list), function(nm) {
    k <- loo_list[[nm]]$diagnostics$pareto_k
    data.frame(
      run            = nm,
      n_obs          = length(k),
      good_pct       = 100 * mean(k < 0.5),
      ok_pct         = 100 * mean(k >= 0.5 & k < 0.7),
      bad_pct        = 100 * mean(k >= 0.7 & k < 1.0),
      very_bad_pct   = 100 * mean(k >= 1.0),
      n_bad_or_worse = sum(k >= 0.7),
      max_k          = max(k)
    )
  }))
  print(summary_df, row.names = FALSE, digits = 3)

  out_file <- file.path(sweep_output_dir, paste0("pareto_k_summary_", date_suffix, ".csv"))
  write.csv(summary_df, out_file, row.names = FALSE)
  cat("\nPareto-k trustworthiness summary saved to:", out_file, "\n")
  cat("Interpretation: if the elpd-best prior also has a meaningfully higher\n",
      "n_bad_or_worse / max_k than the others, its LOO comparison is less\n",
      "trustworthy than the point estimate suggests -- weight the z-score and\n",
      "the item-3/item-4 plots below more heavily than the raw elpd ranking.\n")
  invisible(summary_df)
}

write_pareto_k_summary(loo_list)

# =============================================================================
# Items 3 & 4: re-read each run's w_cb draws (memory-safe targeted read, not
# a live fit$draws() call) and compare across priors.
#
# All 3 configs share identical dlnm_vars/dlnm_argvar/dlnm_arglag/max_lag (only
# shrinkage_prior differs), so the cross-basis objects built by the LAST
# config's eval() (still in globalenv as `prep`) are valid for all 3 runs --
# reused here rather than rebuilt.
# =============================================================================
if (!exists("prep") || is.null(prep$cb_mats)) {
  cat("No `prep` available after the sweep loop (all configs may have failed) — skipping items 3/4.\n")
} else {

  w_cb_draws_by_prior <- list()
  for (i in seq_along(priors)) {
    if (!nzchar(run_dirs[i])) next
    chain_csvs_i <- list.files(run_dirs[i], pattern = "^hierarchical_state_space.*\\.csv$", full.names = TRUE)
    if (length(chain_csvs_i) == 0) {
      cat("No chain CSVs found for", priors[i], "in", run_dirs[i], "— skipping its tail-check draws.\n")
      next
    }
    cc <- read_cmdstan_csv(chain_csvs_i, variables = "w_cb",
                            sampler_diagnostics = character(0), format = "draws_matrix")
    w_cb_draws_by_prior[[priors[i]]] <- cc$post_warmup_draws
    rm(cc); gc()
  }

  if (length(w_cb_draws_by_prior) == 0) {
    cat("No w_cb draws could be re-read for any prior — skipping items 3/4.\n")
  } else {

    cb_mats   <- prep$cb_mats
    dlnm_vars <- prep$dlnm_vars
    df_prep   <- prep$df
    dlnm_var_stats <- prep$dlnm_var_stats

    cb_ncols   <- sapply(dlnm_vars, function(v) ncol(cb_mats[[v]]))
    col_starts <- cumsum(c(1L, cb_ncols[-length(cb_ncols)]))
    coef_labels <- unlist(lapply(seq_along(dlnm_vars), function(i)
      paste0(dlnm_vars[i], "_", colnames(cb_mats[[dlnm_vars[i]]]))
    ))

    # ---- Item 4: posterior vs. prior density for each w_cb[k] -------------
    dens_prior <- function(x, prior_name, sigma = 1.0) {
      switch(prior_name,
        student_t = dt(x / sigma, df = 3) / sigma,
        normal    = dnorm(x, 0, sigma),
        laplace   = (1 / (2 * sigma)) * exp(-abs(x) / sigma)
      )
    }

    post_long <- do.call(rbind, lapply(names(w_cb_draws_by_prior), function(pr) {
      m <- w_cb_draws_by_prior[[pr]]
      do.call(rbind, lapply(seq_len(ncol(m)), function(k)
        data.frame(prior = pr, coef = coef_labels[k], value = m[, k])
      ))
    }))

    x_seq <- seq(min(post_long$value), max(post_long$value), length.out = 200)
    prior_curve_df <- do.call(rbind, lapply(names(w_cb_draws_by_prior), function(pr)
      data.frame(prior = pr, x = x_seq, density = dens_prior(x_seq, pr, sigma = 1.0))
    ))
    prior_curve_df_facets <- do.call(rbind, lapply(unique(post_long$coef), function(cf)
      cbind(prior_curve_df, coef = cf)
    ))

    p_tailcheck <- ggplot(post_long, aes(x = value, color = prior, fill = prior)) +
      geom_density(alpha = 0.15, linewidth = 0.7) +
      geom_line(data = prior_curve_df_facets, aes(x = x, y = density, color = prior),
                linetype = "dashed", linewidth = 0.5, inherit.aes = FALSE) +
      facet_wrap(~coef, scales = "free_y") +
      labs(title = "w_cb posterior vs. prior density, by shrinkage-prior family",
           subtitle = "Solid = posterior density; dashed = that family's prior density (same nominal scale, sigma=1.0)",
           x = "w_cb value", y = "Density", color = "Prior", fill = "Prior") +
      theme_minimal()

    tailcheck_file <- file.path(sweep_output_dir, paste0("shrinkage_prior_tailcheck_", date_suffix, ".png"))
    ggsave(tailcheck_file, p_tailcheck, width = 12, height = 8)
    cat("Item 4 (posterior-vs-prior tail check) saved to:", tailcheck_file, "\n")

    # ---- Item 3: posterior predictive lag-response curve at median exposure,
    #      overlaid across priors -------------------------------------------
    lagresp_rows <- list()
    for (i in seq_along(dlnm_vars)) {
      var  <- dlnm_vars[i]
      cols <- col_starts[i] + seq_len(cb_ncols[i]) - 1L
      if (!var %in% names(df_prep)) next

      stats_i <- if (!is.null(dlnm_var_stats) && var %in% names(dlnm_var_stats))
        dlnm_var_stats[[var]] else list(mean = 0, sd = 1)
      v_mean <- stats_i$mean
      v_sd   <- stats_i$sd
      x_orig_obs <- df_prep[[var]][is.finite(df_prep[[var]])] * v_sd + v_mean
      p50_orig <- as.numeric(quantile(x_orig_obs, probs = 0.5, na.rm = TRUE))
      p50_std  <- (p50_orig - v_mean) / v_sd

      L_val   <- as.integer(attr(cb_mats[[var]], "lag")[2])
      lag_seq <- 0:L_val
      cb_colnames <- colnames(cb_mats[[var]])

      for (pr in names(w_cb_draws_by_prior)) {
        draws_i <- w_cb_draws_by_prior[[pr]][, cols, drop = FALSE]
        coef_i  <- setNames(colMeans(draws_i), cb_colnames)
        vcov_i  <- cov(draws_i)
        dimnames(vcov_i) <- list(cb_colnames, cb_colnames)

        red_i <- tryCatch(
          exp_crosspred(dlnm::crossreduce(cb_mats[[var]], coef = coef_i, vcov = vcov_i,
                          model.link = "identity", type = "var", value = p50_std,
                          lag = c(0, L_val), bylag = 1, cen = 0)),
          error = function(e) {
            cat("  crossreduce failed for", var, "/", pr, ":", conditionMessage(e), "\n")
            NULL
          }
        )
        if (is.null(red_i)) next

        lagresp_rows[[length(lagresp_rows) + 1]] <- data.frame(
          variable = var, prior = pr, lag = lag_seq,
          estimate = as.numeric(red_i$fit), ci_low = as.numeric(red_i$low), ci_high = as.numeric(red_i$high)
        )
      }
    }

    if (length(lagresp_rows) > 0) {
      lagresp_df <- do.call(rbind, lagresp_rows)
      write_csv(lagresp_df, file.path(sweep_output_dir, paste0("lagresponse_p50_by_prior_", date_suffix, ".csv")))

      for (var in unique(lagresp_df$variable)) {
        d <- lagresp_df[lagresp_df$variable == var, ]
        p_overlay <- ggplot(d, aes(x = lag, y = estimate, color = prior, fill = prior)) +
          geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.15, color = NA) +
          geom_line(linewidth = 0.9) +
          geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
          labs(title = paste0("Lag-response at median ", var, ", by shrinkage-prior family"),
               subtitle = "Shaded band: 95% CI",
               x = "Lag (months)", y = "Odds ratio of p_bt", color = "Prior", fill = "Prior") +
          theme_minimal()
        ggsave(file.path(sweep_output_dir, sprintf("lagresponse_p50_%s_by_prior_%s.png", var, date_suffix)),
               p_overlay, width = 9, height = 6)
      }
      cat("Item 3 (posterior predictive lag-response overlay) plots saved to:", sweep_output_dir, "\n")
    } else {
      cat("No lag-response curves could be built — skipping item 3 plots.\n")
    }
  }
}
