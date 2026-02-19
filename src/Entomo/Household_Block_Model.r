# =====================================================
# Alternative household model at block level (Stan)
# Chapter 7: Entomo_model.pdf
# =====================================================

# --- 0. Load libraries ---
# library(rstan)  # Using cmdstanr instead in main model
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

# rstan_options(auto_write = TRUE)  # Not needed for cmdstanr
options(mc.cores = 1)

# --- 1. Paths and output ---
output_dir <- file.path("/home/rita/PyProjects/DI-MOB-BionamiX", "results", "Entomo", "household_block_model")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
date_suffix <- format(Sys.Date(), "%Y%m%d")

input_data <- read_csv("/media/rita/New Volume/Documenten/DI-MOB/Other Data/Env_data_cuba/data/env_epi_entomo_data_per_manzana_2016_04_to_2019_12.csv")

# --- 2. Prepare block-time dataset ---
env_vars <- c("avg_temp", "rel_hum", "total_precip", "WS2M", "mean_ndmi", "mean_ndwi", "mean_ndvi")

block_time <- input_data %>%
  mutate(year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d")) %>%
  group_by(manzana, year_month_date) %>%
  summarise(
    Y = sum(Houses_pos_IS, na.rm = TRUE),
    C = sum(cases, na.rm = TRUE),
    Inspected_houses = mean(Inspected_houses, na.rm = TRUE),
    across(all_of(env_vars), ~mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

block_levels <- sort(unique(block_time$manzana))
time_levels <- sort(unique(block_time$year_month_date))

# Household counts per block (Nb)
N_b_df <- input_data %>%
  group_by(manzana) %>%
  summarise(N_b = round(mean(Inspected_houses, na.rm = TRUE)), .groups = "drop")

block_time <- block_time %>%
  complete(manzana = block_levels, year_month_date = time_levels, fill = list(Y = 0, D = 0)) %>%
  left_join(N_b_df, by = "manzana") %>%
  mutate(
    N_b = as.integer(N_b),
    across(all_of(env_vars), ~coalesce(.x, 0))
  )

B <- length(block_levels)
T <- length(time_levels)
K <- length(env_vars)

Y_bt <- matrix(0L, nrow = B, ncol = T)
C_bt <- matrix(0L, nrow = B, ncol = T)
X_bt <- array(0, dim = c(B, T, K))

for (i in seq_len(nrow(block_time))) {
  b <- match(block_time$manzana[i], block_levels)
  t <- match(block_time$year_month_date[i], time_levels)
  Y_bt[b, t] <- as.integer(block_time$Y[i])
  C_bt[b, t] <- as.integer(block_time$C[i])
  X_bt[b, t, ] <- as.numeric(block_time[i, env_vars])
}

N_b <- N_b_df$N_b[match(block_levels, N_b_df$manzana)]

# --- 3. Prepare data for Stan ---
stan_data <- list(
  B = B,
  T = T,
  K = K,
  N_b = as.integer(N_b),
  Y = Y_bt,
  C = C_bt,
  X = X_bt
)

stan_file <- "/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo/household_block_model.stan"

# --- 4. Fit model ---
fit <- stan(
  file = stan_file,
  data = stan_data,
  chains = 2,
  iter = 500,
  warmup = 250,
  control = list(adapt_delta = 0.95, max_treedepth = 10)
)

# --- 5. Summarize results ---
summary_output <- capture.output(print(fit, pars = c("alpha0", "alpha1", "alpha2", "alpha_gamma", "alpha_phi", "theta", "sigma_u", "sigma_v", "rho")))
cat(summary_output, sep = "\n")
writeLines(summary_output, file.path(output_dir, paste0("household_block_model_summary_", date_suffix, ".txt")))

# --- 6. Posterior mean of r_bt (optional quick check) ---
post <- rstan::extract(fit)
r_bt_mean <- apply(post$r_bt, c(2, 3), mean)

avg_r_bt <- rowMeans(r_bt_mean)
df_r <- data.frame(block = block_levels, r_mean = avg_r_bt)

p <- ggplot(df_r, aes(x = r_mean)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Posterior Mean of r_bt (Block Averaged)", x = "Mean r_bt", y = "Blocks")

print(p)
ggsave(file.path(output_dir, paste0("alt_household_block_model_rbt_hist_", date_suffix, ".png")), p, width = 8, height = 6, dpi = 300)

cat("Outputs saved to:", output_dir, "\n")
