
# Backward stepwise AIC selection.
# Reuses all data prep from GLMM_model_selection.R (sections 1–5), then
# runs a greedy stepwise loop: at each step, remove the predictor whose
# removal most improves AIC (ΔAIC most negative), stop when no removal
# gives ΔAIC < -2.

# =============================================================================
# 1. DATA PREP (sourced from GLMM_model_selection.R)
# =============================================================================
.selection_prep_only <- TRUE
tryCatch(
  source("GLMM_model_selection.R"),
  error = function(e) {
    if (!grepl("\\.selection_prep_only", conditionMessage(e))) stop(e)
  }
)
rm(.selection_prep_only)

# After sourcing: df_model, fixed_effects, random_effects, cfg, run_suffix,
# model_spec, predictor_spec are all available.

# =============================================================================
# 2. OUTPUT DIRECTORY
# =============================================================================
stepwise_dir <- file.path(
  cfg$output_dir, predictor_spec,
  paste0(model_spec, "_stepwise"), run_suffix
)
dir.create(stepwise_dir, recursive = TRUE, showWarnings = FALSE)
cat("Stepwise output:", stepwise_dir, "\n\n")

# =============================================================================
# 3. HELPER: fit one model, return NULL on error
# =============================================================================
fit_model <- function(formula_str) {
  tryCatch(
    glmmTMB(
      as.formula(formula_str),
      family  = glmmTMB::betabinomial(link = cfg$link_function),
      data    = df_model,
      control = glmmTMBControl(optCtrl = list(
        iter.max = cfg$iter_max, eval.max = cfg$eval_max, trace = 0))
    ),
    error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL }
  )
}

re_str <- if (length(random_effects) > 0)
  paste("+", paste(random_effects, collapse = " + ")) else ""

make_formula <- function(preds) paste(
  "cbind(y_bt, n_trials - y_bt) ~",
  paste(preds, collapse = " + "), re_str
)

# =============================================================================
# 4. STEPWISE LOOP
# =============================================================================

# When a main effect is dropped, any interaction term that contains it must
# also be dropped (a model with var1:var2 but not var1 is ill-specified).
# Dropping a standalone interaction term leaves main effects intact.
cascade_drop <- function(pred, fixed) {
  interactions_with_pred <- grep(
    paste0("(^|:)", gsub("([.|()\\^{}+$*?])", "\\\\\\1", pred), "($|:)"),
    fixed[grepl(":", fixed)], value = TRUE
  )
  unique(c(pred, interactions_with_pred))
}

current_fixed <- fixed_effects
step          <- 0L
step_log      <- list()

repeat {
  step <- step + 1L
  candidates <- setdiff(current_fixed, "reactive_shift")

  # --- Fit current model ---
  current_formula_str <- make_formula(current_fixed)
  cat(sprintf("\n=== Step %d | %d candidates ===\n",
              step, length(candidates)))
  cat("Fitting current model ... ")
  current_model <- fit_model(current_formula_str)
  if (is.null(current_model)) stop("Current model failed to fit at step ", step)
  current_aic <- AIC(current_model)
  cat(sprintf("AIC = %.2f\n", current_aic))

  # Save step summary
  summ_out <- local({
    op <- options(max.print = 99999, width = 200); on.exit(options(op))
    capture.output(summary(current_model))
  })
  writeLines(
    c(sprintf("Step: %d", step), paste0("Formula: ", current_formula_str), "", summ_out),
    file.path(stepwise_dir, sprintf("step%02d_model_summary.txt", step))
  )

  # --- Single-pass: try removing each candidate ---
  step_results <- tibble(predictor = character(), AIC = numeric(),
                         delta_AIC = numeric(), converged = logical())

  for (pred in candidates) {
    drop_set <- cascade_drop(pred, current_fixed)
    label <- if (length(drop_set) > 1)
      paste0(pred, " (+ ", length(drop_set) - 1, " interaction(s))")
    else pred
    cat(sprintf("  Removing %-55s ... ", label))
    m <- fit_model(make_formula(setdiff(current_fixed, drop_set)))

    if (is.null(m)) {
      step_results <- add_row(step_results, predictor = pred,
                              AIC = NA_real_, delta_AIC = NA_real_, converged = FALSE)
      next
    }

    aic_val   <- AIC(m)
    delta     <- aic_val - current_aic
    converged <- m$fit$convergence == 0
    cat(sprintf("AIC = %9.2f  ΔAIC = %+8.2f%s\n", aic_val, delta,
                if (!converged) "  (convergence warning)" else ""))

    step_results <- add_row(step_results, predictor = pred,
                            AIC = aic_val, delta_AIC = delta, converged = converged)
  }

  step_results <- step_results %>% arrange(delta_AIC)
  step_log[[step]] <- step_results %>% mutate(step = step, .before = 1)
  write_csv(step_results,
            file.path(stepwise_dir, sprintf("step%02d_aic_candidates.csv", step)))

  # --- Best candidate ---
  best <- step_results %>% filter(converged, !is.na(delta_AIC)) %>% slice(1)

  if (nrow(best) == 0 || best$delta_AIC >= 0) {
    cat(sprintf(
      "\nStopping: best removal ΔAIC = %s (threshold < 0)\n",
      if (nrow(best) == 0) "NA" else sprintf("%.2f", best$delta_AIC)
    ))
    break
  }

  drop_set <- cascade_drop(best$predictor, current_fixed)
  cat(sprintf("\nDropping '%s'  (ΔAIC = %.2f)\n", best$predictor, best$delta_AIC))
  if (length(drop_set) > 1)
    cat(sprintf("  Cascading drop of interactions: %s\n",
                paste(setdiff(drop_set, best$predictor), collapse = ", ")))
  current_fixed <- setdiff(current_fixed, drop_set)
}

# =============================================================================
# 5. FINAL SUMMARY + FULL POST-FIT OUTPUT
# =============================================================================
cat("\n=== Stepwise selection complete ===\n")
cat("Steps taken:", step, "\n")
cat("Final predictors:", paste(setdiff(current_fixed, "reactive_shift"), collapse = ", "), "\n")
cat(sprintf("Final AIC: %.2f\n\n", current_aic))

all_steps <- bind_rows(step_log)
write_csv(all_steps, file.path(stepwise_dir, "stepwise_full_log.csv"))

# Wire up objects and dirs expected by GLMM_postfit.R
model         <- current_model
formula_str   <- current_formula_str
run_output_dir   <- stepwise_dir
plots_output_dir <- file.path(stepwise_dir, "plots")
resid_output_dir <- file.path(stepwise_dir, "residuals_check")
dir.create(plots_output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(resid_output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# 6. STEPWISE SELECTION SUMMARY PLOT
# =============================================================================
# One row per step: the predictor that was actually dropped (min delta_AIC)
dropped_per_step <- all_steps %>%
  group_by(step) %>%
  filter(converged, !is.na(delta_AIC)) %>%
  slice_min(delta_AIC, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  filter(delta_AIC < 0) %>%
  mutate(
    aic_improvement = -delta_AIC,
    step_label      = paste0(step, " : ", predictor),
    var_type        = case_when(
      grepl(":", predictor)          ~ "interaction",
      grepl("_lag\\d+$", predictor)  ~ "meteorology",
      TRUE                           ~ "anthropogenic"
    )
  )

p_stepwise <- ggplot(
  dropped_per_step,
  aes(x = aic_improvement,
      y = reorder(step_label, -step),
      fill = var_type)
) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", aic_improvement)),
            hjust = -0.15, size = 3.5) +
  scale_fill_manual(
    values = c("meteorology" = "#2980B9", "anthropogenic" = "#C0392B", "interaction" = "#8E44AD"),
    labels = c("meteorology" = "Meteorological", "anthropogenic" = "Anthropogenic", "interaction" = "Interaction"),
    name   = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18)), limits = c(0, NA)) +
  labs(title = "Predictors dropped per step", x = "AIC improvement", y = "Step") +
  theme_minimal(base_size = 11) +
  theme(
    legend.position    = "bottom",
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank()
  )

print(p_stepwise)
ggsave(
  file.path(plots_output_dir, "stepwise_dropped_predictors.png"),
  p_stepwise,
  width  = 12,
  height = max(3, 0.55 * nrow(dropped_per_step) + 2),
  dpi    = 300,
  bg     = "white"
)
cat("Stepwise summary plot saved to:", plots_output_dir, "\n")

source("GLMM_postfit.R")

cat("All outputs saved to:", stepwise_dir, "\n")
