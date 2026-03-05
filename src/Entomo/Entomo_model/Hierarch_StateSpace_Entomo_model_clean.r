# =====================================================
# Mosquito hierarchical model with reactive surveillance
# Clean calibration script (function-based)
# =====================================================

if (!require("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
}
renv::restore()

library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)

# =========================
# 0) LOAD HELPER FUNCTIONS
# =========================
script_dir <- if (exists("rstudioapi") && requireNamespace("rstudioapi", quietly = TRUE) && 
                  !is.null(rstudioapi::getActiveDocumentContext()$path)) {
  dirname(rstudioapi::getActiveDocumentContext()$path)
} else {
  dirname(sys.frame(1)$ofile)
}
source(file.path(script_dir, "helper_functions.r"))

# =========================
# 1) SETTINGS
# =========================
hostname <- Sys.info()["nodename"]

cfg <- list(
  data_dir = if (hostname == "frietjes") "~/data/Entomo" else "/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/",
  data_file_name = "env_epi_entomo_data_per_manzana_2016_04_to_2019_12.csv",
  output_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting",

  # model variant
  use_temporal_re = FALSE,

  # data prep
  n_blocks = 100, # set NULL for all blocks
  lag_vars = c("avg_temp", "rel_hum", "total_precip", "mean_ndvi"),
  max_lag = 1,
  kappa = 2,
  unlagged_vars = c("is_urban", "has_aljibes", "is_WI", "is_WUI", "water_shortage", "WS2M"),
  binary_unlagged_vars = c("is_urban", "has_aljibes", "is_WI", "is_WUI", "water_shortage"),

  # MCMC
  chains = 2,
  iter_warmup = 100,
  iter_sampling = 100,
  thin = 2,
  adapt_delta = 0.95,
  max_treedepth = 12,
  parallel_chains = if (hostname == "frietjes") 2 else 1,

  # outputs
  save_random_effects_plot = TRUE,
  save_ppc_plot = TRUE,
  save_traceplots = TRUE
)

cfg$data_file <- file.path(cfg$data_dir, cfg$data_file_name)
cfg$stan_file <- if (cfg$use_temporal_re) {
  "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/Entomo_model/hierarchical_state_space.stan"
} else {
  "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/Entomo_model/hierarchical_state_space_no_time_re.stan"
}

date_suffix <- format(Sys.Date(), "%Y%m%d")
model_tag <- ifelse(cfg$use_temporal_re, "withTimeRE", "noTimeRE")
run_suffix <- paste0(date_suffix, "_stand_", model_tag)

options(mc.cores = if (hostname == "frietjes") 6 else 2)
dir.create(cfg$output_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 2) MAIN
# =========================
cat("Using hostname:", hostname, "\n")
cat("Data directory:", cfg$data_dir, "\n")
cat("Model variant:", ifelse(cfg$use_temporal_re, "with temporal RE", "without temporal RE"), "\n")

prep <- build_stan_data(cfg)
stan_data <- prep$stan_data
df <- prep$df

mod <- cmdstan_model(cfg$stan_file)
fit <- mod$sample(
  data = stan_data,
  chains = cfg$chains,
  iter_warmup = cfg$iter_warmup,
  iter_sampling = cfg$iter_sampling,
  thin = cfg$thin,
  init = make_init_fun(stan_data, cfg$use_temporal_re),
  adapt_delta = cfg$adapt_delta,
  max_treedepth = cfg$max_treedepth,
  parallel_chains = cfg$parallel_chains
)

fit$save_object(file.path(cfg$output_dir, paste0("fit_", run_suffix, ".rds")))

summary_vars <- c("alpha", "sigma_u", "delta0", "delta1", "w")
if (cfg$use_temporal_re) summary_vars <- c(summary_vars, "sigma_v", "rho")
summary_output <- capture.output(print(fit$summary(variables = summary_vars)))
cat(summary_output, sep = "\n")
writeLines(summary_output, file.path(cfg$output_dir, paste0("model_summary_", run_suffix, ".txt")))

post <- extract_means(fit, nrow(df))

if (cfg$save_random_effects_plot && !all(is.na(post$u))) {
  save_random_effects(post$u, post$v, cfg$output_dir, run_suffix)
}
if (cfg$save_ppc_plot) {
  save_ppc(df, post$y_pred, cfg$output_dir, run_suffix)
}
if (cfg$save_traceplots) {
  save_trace_plots(fit, cfg$output_dir, run_suffix, cfg$use_temporal_re)
}

cat("\nAll outputs saved to:", cfg$output_dir, "\n")
