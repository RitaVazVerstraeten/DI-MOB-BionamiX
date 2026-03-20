
library(tidyverse)
library(glmmTMB)
library(purrr)

# =========================
# SETTINGS
# =========================
# Edit the kappa in GLMM.r (cfg$kappa) before running this sweep.
# The script sources GLMM.r for data prep only — no model is fitted there.

glmm_script   <- "GLMM.r"   # path relative to working directory

# SD values to test. The last entry should be the unconstrained estimate
# (read from a previous GLMM run, or update after a first pass).
sd_values <- c(0.5, 1.0, 1.5, 2.0, 3.0, 4.275)

# Output goes next to the GLMM run output, in its own subdirectory.
# Resolved after sourcing GLMM.r so it inherits the same run paths.

# =========================
# 1. DATA PREP (via GLMM.r)
# =========================
# Set the flag so GLMM.r stops after building df_model / formula.
# The stop() is caught below; df_model, formula, cfg etc. stay in scope.
.glmm_data_prep_only <- TRUE

tryCatch(
  source(glmm_script, local = FALSE),
  error = function(e) {
    if (!grepl(".glmm_data_prep_only", conditionMessage(e))) stop(e)
  }
)

rm(.glmm_data_prep_only)

cat("\ndf_model ready:", nrow(df_model), "rows\n")
cat("Formula:", formula_str, "\n\n")

# =========================
# 2. SWEEP OUTPUT DIRECTORY
# =========================
# Inherits run_output_dir from GLMM.r (same kappa / model spec)
sweep_dir <- file.path(run_output_dir, "ar1_variance_sweep")
dir.create(sweep_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 3. EXTRACT UNCONSTRAINED AR(1) PARAMETERS (REFERENCE)
# =========================
# Fit the unconstrained model first to get the baseline AIC and theta values.
cat("Fitting unconstrained reference model...\n")
model_ref <- glmmTMB(
  formula,
  family  = binomial(link = cfg$link_function),
  data    = df_model,
  control = glmmTMBControl(
    optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max)
  )
)

theta_ref  <- getME(model_ref, "theta")          # [log(sd), logit-rho]
sd_ref     <- exp(theta_ref[1])
rho_ref    <- 2 * plogis(theta_ref[2]) - 1       # back-transform logit to (-1, 1)
aic_ref    <- AIC(model_ref)

cat(sprintf("Unconstrained: SD = %.3f, rho = %.3f, AIC = %.1f\n\n",
            sd_ref, rho_ref, aic_ref))

# Replace the last sd_values entry with the true unconstrained estimate
sd_values[length(sd_values)] <- sd_ref

# =========================
# 4. CONSTRAINED SD SWEEP
# =========================
cat("Running AR(1) SD sweep...\n")

fit_constrained <- function(sd_val) {
  fit <- tryCatch(
    glmmTMB(
      formula,
      family  = binomial(link = cfg$link_function),
      data    = df_model,
      # Fix log(sd) = log(sd_val) via map; estimate rho freely (factor = 1)
      start   = list(theta = c(log(sd_val), theta_ref[2])),
      map     = list(theta = factor(c(NA, 1L))),
      control = glmmTMBControl(
        optCtrl = list(iter.max = cfg$iter_max, eval.max = cfg$eval_max)
      )
    ),
    error = function(e) {
      cat(sprintf("  SD = %.3f  FAILED: %s\n", sd_val, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)

  theta_fit <- getME(fit, "theta")
  rho_fit   <- 2 * plogis(theta_fit[2]) - 1

  tibble(
    sd_fixed     = sd_val,
    rho_est      = rho_fit,
    AIC          = AIC(fit),
    logLik       = as.numeric(logLik(fit)),
    delta_AIC    = AIC(fit) - aic_ref,
    converged    = fit$sdr$pdHess  # TRUE if Hessian is positive definite
  )
}

results <- map_dfr(sd_values, function(sv) {
  cat(sprintf("  SD = %.3f\n", sv))
  fit_constrained(sv)
})

# Mark the unconstrained row
results <- results %>%
  mutate(constrained = sd_fixed != sd_ref)

cat("\nResults:\n")
print(results, n = nrow(results))

results_file <- file.path(sweep_dir,
  paste0("ar1_sd_sweep_", run_suffix, ".csv"))
write_csv(results, results_file)
cat("\nResults saved to:", results_file, "\n")

# =========================
# 5. PLOTS
# =========================

# Plot 1: AIC vs fixed SD
# deltaAIC < 2 = no meaningful loss of fit → constraint is defensible
p_aic <- ggplot(results, aes(x = sd_fixed, y = AIC)) +
  geom_hline(yintercept = aic_ref, linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = aic_ref + 2, linetype = "dotted", colour = "orange") +
  geom_line(colour = "steelblue") +
  geom_point(aes(shape = constrained), colour = "steelblue", size = 3) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 17),
                     labels = c(`TRUE` = "Constrained", `FALSE` = "Unconstrained"),
                     name   = NULL) +
  annotate("text", x = sd_ref, y = aic_ref,
           label = sprintf("Unconstrained\nSD = %.2f", sd_ref),
           hjust = -0.1, vjust = 1, colour = "grey40", size = 3) +
  annotate("text", x = min(results$sd_fixed), y = aic_ref + 2,
           label = "\u0394AIC = 2", hjust = 0, vjust = -0.3,
           colour = "orange", size = 3) +
  labs(
    title    = "AIC vs constrained AR(1) SD",
    subtitle = "Dotted line = AIC + 2 (conventional fit-loss threshold)",
    x        = "Fixed AR(1) SD", y = "AIC"
  ) +
  theme_minimal()

ggsave(file.path(sweep_dir, paste0("ar1_aic_vs_sd_", run_suffix, ".png")),
       p_aic, width = 8, height = 5, dpi = 150)

# Plot 2: delta AIC vs fixed SD (easier to read the loss magnitude)
p_daic <- ggplot(results, aes(x = sd_fixed, y = delta_AIC)) +
  geom_hline(yintercept = 0,  linetype = "dashed",  colour = "grey50") +
  geom_hline(yintercept = 2,  linetype = "dotted",  colour = "orange") +
  geom_hline(yintercept = 10, linetype = "dotted",  colour = "red") +
  geom_line(colour = "steelblue") +
  geom_point(aes(shape = constrained), colour = "steelblue", size = 3) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 17), name = NULL,
                     labels = c(`TRUE` = "Constrained", `FALSE` = "Unconstrained")) +
  labs(
    title    = expression(Delta * "AIC vs constrained AR(1) SD"),
    subtitle = "Orange = 2 (weak evidence); Red = 10 (strong evidence against constraint)",
    x        = "Fixed AR(1) SD", y = expression(Delta * "AIC")
  ) +
  theme_minimal()

ggsave(file.path(sweep_dir, paste0("ar1_deltaAIC_vs_sd_", run_suffix, ".png")),
       p_daic, width = 8, height = 5, dpi = 150)

# Plot 3: Estimated rho vs fixed SD
# If rho changes a lot when SD is constrained, the two AR(1) parameters
# are compensating for each other — a sign of partial identifiability.
p_rho <- ggplot(results, aes(x = sd_fixed, y = rho_est)) +
  geom_hline(yintercept = rho_ref, linetype = "dashed", colour = "grey50") +
  geom_line(colour = "steelblue") +
  geom_point(aes(shape = constrained), colour = "steelblue", size = 3) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 17), name = NULL,
                     labels = c(`TRUE` = "Constrained", `FALSE` = "Unconstrained")) +
  annotate("text", x = sd_ref, y = rho_ref,
           label = sprintf("Unconstrained\nrho = %.2f", rho_ref),
           hjust = -0.1, colour = "grey40", size = 3) +
  labs(
    title    = "Estimated AR(1) rho vs fixed SD",
    subtitle = "Large shift in rho = SD and rho are compensating (partial identifiability)",
    x        = "Fixed AR(1) SD", y = "Estimated rho"
  ) +
  theme_minimal()

ggsave(file.path(sweep_dir, paste0("ar1_rho_vs_sd_", run_suffix, ".png")),
       p_rho, width = 8, height = 5, dpi = 150)

cat("\nAll sweep plots saved to:", sweep_dir, "\n")
