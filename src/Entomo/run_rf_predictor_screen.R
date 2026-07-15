# =============================================================
# Quick predictor screen via cluster-robust random forest
# =============================================================
# Screens candidate predictors (lagged + static) for predictive importance
# BEFORE committing them to an expensive Stan/DLNM sweep.
#
# Two things this does differently from a default randomForest()/ranger() call:
#
#   1. Lag structure: candidate lag variables are expanded into lag0..lagL
#      columns (same idea as the DLNM cross-basis), so a predictor whose effect
#      is concentrated at, say, lag 3-5 isn't missed by only looking at lag 0.
#
#   2. Cluster-robust importance: static/near-static CMF-level predictors
#      (urban_pct, is_WUI, pop_density, ...) repeat identically across ~48
#      months per CMF. A default row-level random forest can use these as a
#      "fingerprint" to recognise which CMF a row belongs to and just recall
#      that CMF's average outcome, rather than learning a generalisable
#      relationship -- inflating their importance. grf::regression_forest's
#      `clusters` argument does cluster-aware subsampling (whole CMFs in/out
#      of a tree together) so importance reflects generalisation to UNSEEN
#      CMFs, not row-level memorisation.
#
# This is a screen, not a substitute for the DLNM/Stan fit: it tells you which
# variables are worth spending Stan-fitting time on and roughly how much
# credit is shared between correlated predictors (e.g. avg_temp/total_precip),
# not their functional form, lag shape, or causal interpretation.

if (!requireNamespace("grf", quietly = TRUE)) install.packages("grf")
if (!requireNamespace("iml", quietly = TRUE)) install.packages("iml")
library(grf)
library(iml)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

# ── Script directory detection (same pattern as the other run_*.R scripts) ──
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
output_dir <- "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/Random_Forest"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
# renv::restore(project = script_dir, prompt = FALSE)  # uncomment if grf isn't in the lockfile yet
source(file.path(script_dir, "helper_functions.r"))

# =========================
# SETTINGS
# =========================
hostname   <- Sys.info()["nodename"]
block_col  <- "cmf"
max_lag    <- 5
response_start <- "2016_01"

data_dir  <- if (hostname == "frietjes") "~/data/Entomo" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data"
data_file <- file.path(data_dir, "env_epi_entomo_data_per_CMF_2015_01_to_2019_12_NDXIbackfilled_noColinnearity.csv")

# Cast a wide net here -- wider than the current DLNM cfg -- that's the point
# of a screen. Add/remove candidates freely.
lag_candidates <- c(
  "total_precip", "avg_temp", "avg_VPD", "WS2M",
  "total_rainy_days", "precip_max_day_resid_on_tp", "precip_max_day_resid_on_trd", "hurricane_within_120km"
)
unlag_candidates <- c(
  "urban_pct", "is_urban", "HFP_urbanization", "is_WUI", "is_WI", "pop_density", "mean_ndvi",
  "water_containers", "water_shortage", "has_aljibes", "nr_aljibes",
  "random_noise"   # pure noise, unrelated to the outcome -- an empirical "no
                    # signal" floor. Any real candidate scoring at or below
                    # this in the final importance table is indistinguishable
                    # from noise on this importance scale, regardless of its
                    # raw number (importance isn't calibrated like a p-value).
)

t_start <- Sys.time()

# =========================
# BUILD LAGGED + STATIC PREDICTORS
# =========================
df <- load_base_data(data_file)

block_levels <- sort(unique(df[[block_col]]))
time_levels  <- sort(unique(df$year_month_date))
set.seed(1)
df <- df %>%
  mutate(
    block = match(.data[[block_col]], block_levels),
    time  = match(year_month_date, time_levels),
    random_noise = rnorm(n())   # noise-floor reference column, see unlag_candidates above
  ) %>%
  arrange(block, time)

# Lag each candidate 0..max_lag within block (mirrors the "lagged correlation"
# pattern already used in Hierarch_StateSpace_Entomo_model.r / Descriptive_Stats).
lag_fns <- setNames(
  lapply(0:max_lag, function(l) function(x) dplyr::lag(x, l)),
  paste0("lag", 0:max_lag)
)
df_lagged <- df %>%
  group_by(block) %>%
  arrange(time, .by_group = TRUE) %>%
  mutate(across(all_of(lag_candidates), lag_fns, .names = "{.col}_{.fn}")) %>%
  ungroup()

lag_cols <- paste0(rep(lag_candidates, each = max_lag + 1), "_lag", 0:max_lag)

response_date <- as.Date(paste0(response_start, "_01"), "%Y_%m_%d")
df_model <- df_lagged %>%
  filter(year_month_date >= response_date) %>%
  mutate(
    positivity_rate = Houses_pos_IS / pmax(Inspected_houses, 1),
    across(where(is.logical), as.numeric)
  ) %>%
  select(block, all_of(lag_cols), all_of(unlag_candidates),
         Inspected_houses, positivity_rate) %>%
  drop_na()

cat(sprintf("Screening on %d rows, %d CMFs, %d candidate columns\n",
            nrow(df_model), n_distinct(df_model$block),
            length(lag_cols) + length(unlag_candidates)))
t_data_ready <- Sys.time()
cat(sprintf("Data prep took %.1f sec\n", as.numeric(difftime(t_data_ready, t_start, units = "secs"))))

lag_cols_for <- function(vars) paste0(rep(vars, each = max_lag + 1), "_lag", 0:max_lag)

Y        <- df_model$positivity_rate
clusters <- df_model$block                    # whole CMFs held in/out together
weights  <- df_model$Inspected_houses          # downweight low-N rows, same spirit as n_bt in the Stan model

# Fit a cluster-robust forest on a given (lag_vars, unlag_vars) choice.
# tune = FALSE uses grf's defaults -- deliberately used for bake-offs below so
# the comparison isolates the variable choice rather than each candidate
# getting separately auto-tuned hyperparameters. num_trees defaults to 2000
# for the main screen/bake-offs; the interaction-detection forests below pass
# a much smaller value (grf's per-tree bookkeeping -- sample/leaf indices,
# honesty splits -- scales with num.trees x N, not with feature count, so
# fewer columns alone doesn't shrink the object iml has to export).
fit_forest <- function(lag_vars_use, unlag_vars_use, tune = TRUE, num_trees = 2000) {
  cols <- c(lag_cols_for(lag_vars_use), unlag_vars_use)
  X <- df_model %>% select(all_of(cols)) %>% as.matrix()
  regression_forest(
    X, Y, clusters = clusters, sample.weights = weights,
    num.trees = num_trees, tune.parameters = if (tune) "all" else "none"
  )
}

# Out-of-bag R^2: predict(rf) with no newdata returns OOB predictions, so this
# is a fair (not-fit-on-itself) fit measure for comparing candidates.
oob_r2 <- function(rf) {
  pred <- predict(rf)$predictions
  1 - sum((Y - pred)^2) / sum((Y - mean(Y))^2)
}

# =========================
# MUTUALLY EXCLUSIVE PREDICTOR GROUPS (BUNDLES)
# =========================
# Each "choice" within a group can be a BUNDLE of >1 variable that must travel
# together -- e.g. precip_max_day_resid_on_tp is the residual of precip_max_day
# regressed on total_precip specifically, so it's only a valid orthogonalised
# variable when total_precip is also in the model. Pairing it with
# total_rainy_days instead would leave it correlated with total_rainy_days
# again, defeating the point of using a residual. total_rainy_days has its own
# matching residual, precip_max_day_resid_on_trd.
#
# Works on either axis (lag or unlag): for each group, every variable across
# every bundle is excluded from the "other" vars on that axis (so a bundle-
# specific residual never leaks into a run that's missing its paired base
# variable), then one forest per whole bundle is fit (everything else held
# fixed) and the winning bundle (by OOB R^2) is kept in its entirety.
#
# NOTE on urban_pct vs is_urban: these two aren't actually competing for the
# same role in the real Bayesian DLNM model -- urban_pct is meant as the
# continuous main effect, is_urban as the binary interaction modifier
# (nonurban_x_tp), so the model design doesn't need to pick one. This bake-off
# answers a narrower, still useful question: as a plain covariate, does the
# continuous or the thresholded version carry more raw predictive signal.
#
# NOTE on urban_pct/is_urban/pop_density and avg_temp/WS2M: unlike the precip
# amount-vs-frequency pair (a true redundant reparameterisation of the same
# measurement), these are correlated but conceptually distinct constructs
# (Spearman r = 0.61 for avg_temp/WS2M; urban_pct/is_urban/pop_density
# correlated per the earlier phi/point-biserial screen). A bake-off picks
# whichever ONE wins and discards the rest entirely -- treat a narrow margin
# in rf_bakeoff_comparison.png as "dominant proxy, not proof the others carry
# zero independent information", not a hard justification for dropping them
# from the eventual Bayesian model.
mutually_exclusive_lag_groups <- list(
  precip_amount_vs_frequency = list(
    total_precip     = c("total_precip",     "precip_max_day_resid_on_tp"),
    total_rainy_days = c("total_rainy_days", "precip_max_day_resid_on_trd")
  )#,
  # temp_vs_windspeed = list(
  #   avg_temp = c("avg_temp"),
  #   WS2M     = c("WS2M")
  # )
  # add more lag-axis groups here; each choice can be a single variable or a bundle
)
mutually_exclusive_unlag_groups <- list(
  urban_encoding = list(
    urban_pct   = c("urban_pct"),
    is_urban    = c("is_urban"),
    pop_density = c("pop_density"), 
    HFP_urbanization = c("HFP_urbanization")
  )
  # add more unlag-axis groups here (e.g. alternative landcover encodings)
  # groups aren't limited to 2 choices -- run_bake_off loops over names(bundles)
)

# Force specific bake-off winners instead of letting OOB R^2 decide. Useful
# when a downstream analysis needs a SPECIFIC variable regardless of which one
# wins on raw predictive power -- e.g. total_precip is used directly in the
# is_urban/water_shortage/avg_VPD interactions below, so if the bake-off
# picked total_rainy_days instead, the main importance table's "precip" line
# item would silently stop matching what the interaction sections are
# actually testing. Names must match a group name above; value must match one
# of that group's choice names. Leave empty (or omit a group) to let that
# group's bake-off run normally -- all candidates still get scored either way
# (see the "FORCED to" vs "keeping" log line), only the winner selection changes.
forced_bakeoff_winners <- list(
  precip_amount_vs_frequency = "total_precip"
  # e.g. urban_encoding = "urban_pct" to force that one too
)

# Fits one forest per bundle in `bundles`, holding `lag_context`/`unlag_context`
# fixed (minus the group's own variables on whichever axis is being resolved),
# prints each candidate's OOB R^2, and returns both the winning bundle and the
# full score table (the latter feeds the bake-off comparison plot below).
# forced_winner (optional): skip the R^2-based selection and use this choice
# name instead -- all candidates are still scored and logged for comparison.
run_bake_off <- function(grp_name, bundles, lag_context, unlag_context, axis = c("lag", "unlag"),
                          forced_winner = NULL) {
  axis <- match.arg(axis)
  all_group_vars <- unique(unlist(bundles))
  cat("\n--- Bake-off:", grp_name, "(", paste(names(bundles), collapse = " vs "), ") ---\n")

  scores <- sapply(names(bundles), function(choice) {
    bundle <- bundles[[choice]]
    if (axis == "lag") {
      lag_use   <- c(setdiff(lag_context, all_group_vars), bundle)
      unlag_use <- unlag_context
    } else {
      lag_use   <- lag_context
      unlag_use <- c(setdiff(unlag_context, all_group_vars), bundle)
    }
    rf_v <- fit_forest(lag_use, unlag_use, tune = FALSE)
    r2   <- oob_r2(rf_v)
    cat(sprintf("  %-30s %-55s OOB R^2 = %.4f\n", choice, paste(bundle, collapse = " + "), r2))
    r2
  })

  if (!is.null(forced_winner)) {
    winner <- forced_winner
    cat(sprintf("  -> FORCED to: %s  (bake-off would have picked: %s)\n",
                paste(bundles[[winner]], collapse = " + "), names(bundles)[which.max(scores)]))
  } else {
    winner <- names(bundles)[which.max(scores)]
    cat("  -> keeping:", paste(bundles[[winner]], collapse = " + "), "\n")
  }
  list(
    winner_bundle = bundles[[winner]],
    scores = tibble(group = grp_name, choice = names(scores), oob_r2 = as.numeric(scores),
                     is_winner = names(scores) == winner)
  )
}

# Resolve lag-axis groups first (using the raw unlag candidates as context),
# then unlag-axis groups using the now-resolved lag candidates as context --
# keeps each bake-off's "other predictors" as close as possible to what the
# final screen will actually use.
bakeoff_scores <- list()

resolved_lag_candidates <- lag_candidates
for (grp_name in names(mutually_exclusive_lag_groups)) {
  bundles <- mutually_exclusive_lag_groups[[grp_name]]
  result  <- run_bake_off(grp_name, bundles, resolved_lag_candidates, unlag_candidates, axis = "lag",
                           forced_winner = forced_bakeoff_winners[[grp_name]])
  bakeoff_scores[[grp_name]] <- result$scores
  resolved_lag_candidates <- c(setdiff(resolved_lag_candidates, unique(unlist(bundles))), result$winner_bundle)
}

resolved_unlag_candidates <- unlag_candidates
for (grp_name in names(mutually_exclusive_unlag_groups)) {
  bundles <- mutually_exclusive_unlag_groups[[grp_name]]
  result  <- run_bake_off(grp_name, bundles, resolved_lag_candidates, resolved_unlag_candidates, axis = "unlag",
                           forced_winner = forced_bakeoff_winners[[grp_name]])
  bakeoff_scores[[grp_name]] <- result$scores
  resolved_unlag_candidates <- c(setdiff(resolved_unlag_candidates, unique(unlist(bundles))), result$winner_bundle)
}
bakeoff_scores_df <- bind_rows(bakeoff_scores)
t_bakeoff_done <- Sys.time()
cat(sprintf("\nBake-offs (%d fits) took %.1f sec\n",
            nrow(bakeoff_scores_df), as.numeric(difftime(t_bakeoff_done, t_data_ready, units = "secs"))))

# =========================
# FINAL SCREEN, USING THE BAKE-OFF WINNERS
# =========================
rf  <- fit_forest(resolved_lag_candidates, resolved_unlag_candidates, tune = TRUE)
t_final_fit_done <- Sys.time()
cat(sprintf("Final tuned fit took %.1f sec\n",
            as.numeric(difftime(t_final_fit_done, t_bakeoff_done, units = "secs"))))
X_final <- df_model %>% select(all_of(lag_cols_for(resolved_lag_candidates)), all_of(resolved_unlag_candidates))

imp <- variable_importance(rf)
importance_df <- tibble(variable = colnames(X_final), importance = as.numeric(imp)) %>%
  arrange(desc(importance))

cat("\n=== Per-column importance (raw lag columns) ===\n")
print(importance_df, n = Inf)

# =========================
# AGGREGATE BY BASE VARIABLE (sums across lags -- the more useful summary)
# =========================
base_var <- sub("_lag[0-9]+$", "", importance_df$variable)
importance_by_var <- importance_df %>%
  mutate(base_var = base_var) %>%
  group_by(base_var) %>%
  summarise(importance = sum(importance), n_cols = n(), .groups = "drop") %>%
  arrange(desc(importance))

cat("\n=== Aggregated importance by variable (lags summed) ===\n")
print(importance_by_var, n = Inf)

# Save for reference
out_path <- file.path(output_dir, "rf_predictor_screen_importance.csv")
write_csv(importance_by_var, out_path)
cat("\nSaved:", out_path, "\n")

# =========================
# FIGURES
# =========================

# 1. Bake-off comparison: OOB R^2 per candidate, faceted by group. Shows not
#    just who won but by how much -- a near-tie is a weaker basis for the
#    "always use X" decision than a landslide.
if (nrow(bakeoff_scores_df) > 0) {
  p_bakeoff <- ggplot(bakeoff_scores_df, aes(x = choice, y = oob_r2, fill = is_winner)) +
    geom_col() +
    facet_wrap(~group, scales = "free_x") +
    scale_fill_manual(values = c(`TRUE` = "steelblue", `FALSE` = "grey70"), guide = "none") +
    labs(title = "Mutually-exclusive predictor bake-offs",
         subtitle = "Out-of-bag R^2 per candidate (blue = kept)",
         x = NULL, y = "OOB R²") +
    theme_minimal()
  ggsave(file.path(output_dir, "rf_bakeoff_comparison.png"), p_bakeoff, width = 9, height = 5, dpi = 300)
}

# 2. Aggregated importance bar chart -- the main "which variables matter" figure.
p_importance <- ggplot(importance_by_var, aes(x = reorder(base_var, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "RF variable importance (lags summed, cluster-robust)",
       subtitle = sprintf("regression_forest, clusters = %s, weighted by Inspected_houses", block_col),
       x = NULL, y = "Importance") +
  theme_minimal()
ggsave(file.path(output_dir, "rf_importance_by_variable.png"), p_importance, width = 8, height = 6, dpi = 300)

# 3. Variable x lag heatmap -- where in time each lagged predictor's signal
#    concentrates. Sanity-check this against the DLNM's own lag-response
#    curves once you fit it: a variable RF finds important mostly at lag 4-5
#    should show a comparable bump in the Stan model's per-lag slice plots.
#    Fill is on a log10 scale: importance is typically long-tailed (a handful
#    of big values, most cells small), so a linear scale washes out contrast
#    among the low-importance cells. Values are floored at a small epsilon so
#    exact zeros (never split on) don't become -Inf and drop out of the plot.
lag_importance_df <- importance_df %>%
  filter(variable %in% lag_cols_for(resolved_lag_candidates)) %>%
  mutate(
    lag = as.integer(sub(".*_lag", "", variable)),
    var = sub("_lag[0-9]+$", "", variable),
    importance_plot = pmax(importance, 1e-6)
  )
if (nrow(lag_importance_df) > 0) {
  p_heatmap <- ggplot(lag_importance_df, aes(x = lag, y = var, fill = importance_plot)) +
    geom_tile(color = "white") +
    scale_fill_viridis_c(trans = "log10", labels = scales::label_number(accuracy = 0.001)) +
    scale_x_continuous(breaks = 0:max_lag) +
    labs(title = "RF importance by variable and lag",
         subtitle = "Fill on log10 scale",
         x = "Lag (months)", y = NULL, fill = "Importance\n(log scale)") +
    theme_minimal()
  ggsave(file.path(output_dir, "rf_importance_by_variable_and_lag.png"), p_heatmap, width = 8, height = 6, dpi = 300)
}

# 4. Fit-quality check: OOB predicted vs. observed positivity rate for the
#    final tuned forest. Analogous to the PPC fitted-vs-observed panel used
#    for the Stan model -- a sense check, not a claim this RF is "the model".
final_pred <- predict(rf)$predictions
p_fit <- ggplot(tibble(observed = Y, predicted = as.numeric(final_pred)),
                 aes(x = observed, y = predicted)) +
  geom_point(alpha = 0.25, color = "darkred") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  labs(title = "Final forest: OOB predicted vs. observed positivity rate",
       x = "Observed", y = "OOB predicted") +
  theme_minimal()
ggsave(file.path(output_dir, "rf_final_fit_oob.png"), p_fit, width = 6, height = 6, dpi = 300)

cat("\nFigures saved to:", output_dir, "\n")

# =========================
# INTERACTION STRENGTH: FRIEDMAN'S H-STATISTIC
# =========================
# Model-agnostic interaction strength (Friedman & Popescu 2008). RF already
# implicitly captures any interaction anyway (a split on is_urban followed by
# a split on total_precip in a child node already models it) -- no engineered
# product columns needed. For a pair of features (j, k), H_jk measures how
# much of their JOINT partial-dependence effect is NOT explained by summing
# their INDIVIDUAL (marginal) partial-dependence effects -- i.e. how much the
# forest's prediction for j genuinely depends on the specific value of k,
# beyond what each contributes separately.
#
# Interpretation: H is roughly bounded in [0, 1] (can exceed it slightly due
# to estimation noise).
#   ~0        : effects are additive/separable -- no meaningful interaction
#   0.1 - 0.3 : mild-to-moderate interaction
#   > 0.3     : strong interaction -- the two features' effects are hard to
#               separate; a plain additive model would misrepresent this pair
# These are rule-of-thumb bands, not a significance test with a p-value --
# compare relative magnitudes across pairs/lags rather than reading a hard
# cutoff, and treat them as a screen for what's worth a closer look, not a
# final answer.
#
# Interaction$new(predictor, feature = X) computes X's H-statistic against
# EVERY other column present in `predictor`'s data, with no way to restrict
# that to a specific subset. Against the full ~50-column screening forest
# that's tens of thousands of grid-point x row evaluations per target, which
# is what OOM-killed the previous attempt. Fix: fit a small, dedicated forest
# using ONLY total_precip (lagged) + is_urban + water_shortage -- exactly the
# variables the two interactions of interest involve. This is a separate,
# simplified model purely for tractable interaction screening (it doesn't
# control for other confounders the way the full screen's `rf` does), not a
# replacement for the main importance results above.
# num.trees=200 (vs. 2000 for the main screen): grf's per-tree bookkeeping is
# what future has to export for iml, and that scales with num.trees x N, not
# feature count -- cutting columns from ~50 to a handful alone didn't shrink
# it (still ~2.2 GiB), which is what caused the OOM/size-cap errors on the
# last two attempts. 200 trees is plenty for a relative interaction-strength
# screen; it doesn't need the precision the tuned final importance model does.
ix_num_trees  <- 200
ix_lag_vars   <- c("total_precip")
ix_unlag_vars <- c("is_urban", "water_shortage", "water_containers")
rf_ix <- fit_forest(ix_lag_vars, ix_unlag_vars, tune = FALSE, num_trees = ix_num_trees)
X_ix  <- df_model %>% select(all_of(lag_cols_for(ix_lag_vars)), all_of(ix_unlag_vars))

# Extra safety margin even with the reduced feature set: iml resamples FROM
# the reference data at every grid point, so its size drives compute cost
# directly. A representative subsample is standard iml practice for exactly
# this reason, and grid.size below caps how many grid points get evaluated.
set.seed(1)
ix_sample_idx <- sample.int(nrow(X_ix), size = min(1500, nrow(X_ix)))
X_ix_sample   <- as.data.frame(X_ix[ix_sample_idx, ])
ix_grid_size  <- 10

# iml uses `future` for grid predictions. future's globals-size check runs
# during future *construction* (getGlobalsAndPackages()) regardless of which
# backend executes it -- plan("sequential") does NOT skip the check, even
# though sequential execution never actually ships anything to a separate
# process (so the cap this check enforces isn't protecting against a real
# risk here). num.trees=200 above should keep the exported size well under
# 500MB, but raise the cap a bit further as a safety margin anyway, and run
# sequentially since there's nothing to gain from R-level process parallelism
# for a forest this size.
old_plan     <- future::plan("sequential")
old_max_size <- getOption("future.globals.maxSize")
options(future.globals.maxSize = 1 * 1024^3)  # 1 GiB safety margin (num.trees=200 forest should be far smaller)

predictor <- Predictor$new(
  model = rf_ix,
  data  = X_ix_sample,
  predict.function = function(model, newdata) predict(model, as.matrix(newdata))$predictions
)

interaction_targets <- c("is_urban", "water_shortage", "water_containers")
h_stat_list <- list()
for (feat in interaction_targets) {
  t_h_start <- Sys.time()
  cat("\nComputing H-statistic for", feat, "vs. all other features...\n")
  interact <- Interaction$new(predictor, feature = feat, grid.size = ix_grid_size)
  # NOTE: iml's Interaction$results column names (.feature / .interaction) are
  # current as of iml 0.11.x -- if this errors, run str(interact$results) and
  # adjust the rename() below to match your installed version.
  # as_tibble() first: interact$results is a data.table, and dplyr verbs don't
  # strip that class on their own -- left as-is, print(n = Inf) later dispatches
  # to data.table's print method (no `n` argument -> ambiguous partial match
  # error) instead of tibble's.
  h_stat_list[[feat]] <- interact$results %>%
    as_tibble() %>%
    rename(other_feature = .feature, h_statistic = .interaction) %>%
    mutate(with_feature = feat)
  cat(sprintf("  took %.1f sec\n", as.numeric(difftime(Sys.time(), t_h_start, units = "secs"))))
}
h_stat_df <- bind_rows(h_stat_list) %>% arrange(desc(h_statistic))

cat("\n=== H-statistic: total_precip lags vs. is_urban / water_shortage ===\n")
h_stat_precip <- h_stat_df %>% filter(grepl("^total_precip_lag", other_feature))
print(h_stat_precip, n = Inf)

cat("\n=== Top 10 strongest interactions overall (either target, any feature) ===\n")
print(head(h_stat_df, 10))

write_csv(h_stat_df, file.path(output_dir, "rf_h_statistic_interactions.csv"))

p_hstat <- ggplot(h_stat_precip, aes(x = reorder(other_feature, h_statistic), y = h_statistic, fill = with_feature)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0.3, colour = "red", linetype = "dashed", linewidth = 0.8) +
  coord_flip() +
  labs(title = "Friedman's H-statistic: total_precip lags x is_urban / water_shortage",
       subtitle = "Higher = stronger interaction; ~0 means effects are additive\nRed dashed line: H = 0.3 (strong interaction threshold)",
       x = NULL, y = "H-statistic", fill = NULL) +
  theme_minimal()
ggsave(file.path(output_dir, "rf_h_statistic_interactions.png"), p_hstat, width = 8, height = 5, dpi = 300)

# =========================
# LAG-MATCHED CONTINUOUS x CONTINUOUS INTERACTION: avg_VPD x total_precip
# =========================
# avg_VPD and total_precip are both continuous AND both lagged -- "the same
# interaction at every lag" means testing avg_VPD_lagK x total_precip_lagK
# for each K in 0..max_lag, NOT the full cross-lag matrix (avg_VPD_lag3 x
# total_precip_lag0, _lag1, ...). Interaction$new(predictor, feature = X)
# always computes against EVERY other column present, so getting just the
# matched-lag pairs means fitting a tiny 2-column model per lag (rather than
# reusing a shared reduced forest and filtering out the cross-lag rows) --
# this also keeps each call fast, since there's only one "other column" to
# compute against.
#
# NOTE: these per-lag bivariate forests are even more stripped-down than the
# is_urban/water_shortage one above -- no other covariates at all, not even
# is_urban/water_shortage. This purely answers "does avg_VPD's effect on
# positivity rate depend on total_precip's value, at this specific lag,
# ignoring everything else." Treat as a first screen, not an adjusted estimate.
set.seed(1)
vpd_precip_h <- lapply(0:max_lag, function(k) {
  vpd_col    <- paste0("avg_VPD_lag", k)
  precip_col <- paste0("total_precip_lag", k)

  X_k  <- df_model %>% select(all_of(c(vpd_col, precip_col))) %>% as.matrix()
  rf_k <- regression_forest(X_k, Y, clusters = clusters, sample.weights = weights,
                             num.trees = ix_num_trees, tune.parameters = "none")

  sample_idx_k <- sample.int(nrow(X_k), size = min(1500, nrow(X_k)))
  predictor_k  <- Predictor$new(
    model = rf_k,
    data  = as.data.frame(X_k[sample_idx_k, ]),
    predict.function = function(model, newdata) predict(model, as.matrix(newdata))$predictions
  )
  interact_k <- Interaction$new(predictor_k, feature = precip_col, grid.size = ix_grid_size)
  results_k  <- as_tibble(interact_k$results)

  # With only 2 columns in this model, there's exactly one possible pairing --
  # take it positionally rather than matching .feature's exact string format
  # (uncertain across iml versions -- an exact-match filter here previously
  # matched 0 rows and silently produced an empty result: pull(.interaction)
  # on a 0-row filter returns a zero-length vector, and tibble()'s zero-length
  # recycling rule collapses the whole row to 0 rows with no error raised).
  if (nrow(results_k) != 1) {
    cat(sprintf("  NOTE: expected 1 row from Interaction$results at lag %d, got %d -- inspect manually:\n", k, nrow(results_k)))
    print(results_k)
  }
  h_val <- results_k$.interaction[1]

  tibble(lag = k, h_statistic = h_val)
}) %>% bind_rows()

cat("\n=== H-statistic: avg_VPD x total_precip, matched by lag ===\n")
print(vpd_precip_h, n = Inf)
write_csv(vpd_precip_h, file.path(output_dir, "rf_h_statistic_vpd_x_precip_by_lag.csv"))

p_vpd_precip <- ggplot(vpd_precip_h, aes(x = lag, y = h_statistic)) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = 0.3, colour = "red", linetype = "dashed", linewidth = 0.8) +
  scale_x_continuous(breaks = 0:max_lag) +
  labs(title = "avg_VPD x total_precip interaction strength by lag",
       subtitle = "Friedman's H-statistic, matched-lag pairs only -- higher = stronger interaction\nRed dashed line: H = 0.3 (strong interaction threshold)",
       x = "Lag (months)", y = "H-statistic") +
  theme_minimal()
ggsave(file.path(output_dir, "rf_h_statistic_vpd_x_precip_by_lag.png"), p_vpd_precip, width = 7, height = 5, dpi = 300)

future::plan(old_plan)                              # restore whatever plan was set before, in case this
options(future.globals.maxSize = old_max_size)      # script runs inside a larger session

t_hstat_done <- Sys.time()
cat(sprintf("\nH-statistic computation took %.1f sec\n",
            as.numeric(difftime(t_hstat_done, t_final_fit_done, units = "secs"))))

t_end <- Sys.time()
cat(sprintf("\nTotal runtime: %.1f sec (%.1f min)\n",
            as.numeric(difftime(t_end, t_start, units = "secs")),
            as.numeric(difftime(t_end, t_start, units = "mins"))))
