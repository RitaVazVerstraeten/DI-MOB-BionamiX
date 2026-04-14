# =====================================================
# Plot posterior lag weight profiles from model summary txt
# Reads w[k,l] estimates and plots shape per covariate
# =====================================================

library(dplyr)
library(ggplot2)
library(stringr)

# =========================
# SETTINGS
# =========================

# Path to the model summary txt file — adjust to your run
summary_txt <- file.choose()   # opens file picker; or set path directly:
# summary_txt <- "/home/rita/data/Entomo/fitting/stan/.../model_summary_....txt"

# Covariate names in the order they were passed to Stan (cfg$lag_vars)
lag_var_names <- c("total_rainy_days", "avg_VPD", "precip_max_day", "mean_ndvi")

# Maximum lag used in the model (cfg$max_lag)
max_lag <- 2

# Output directory for the plot (same folder as the txt by default)
output_dir <- dirname(summary_txt)

# =========================
# PARSE SUMMARY TXT
# =========================

lines <- readLines(summary_txt)

# Keep only lines that contain w[ (the lag weight parameters)
w_lines <- lines[grepl("^\\s*(\\d+\\s+)?w\\[", lines)]

# Parse into data frame: extract variable name and numeric columns
parsed <- lapply(w_lines, function(ln) {
  # Remove leading row index if present (printed tibble may include it)
  ln_clean <- sub("^\\s*\\d+\\s+", "", trimws(ln))
  # Split on whitespace
  parts <- str_split(ln_clean, "\\s+")[[1]]
  # parts[1] = variable name (e.g. "w[1,2]"), then mean, median, sd, mad, q5, q95, ...
  if (length(parts) < 7) return(NULL)
  data.frame(
    variable = parts[1],
    mean     = as.numeric(parts[2]),
    q5       = as.numeric(parts[6]),
    q95      = as.numeric(parts[7]),
    stringsAsFactors = FALSE
  )
})
w_df <- do.call(rbind, Filter(Negate(is.null), parsed))

# Extract k (covariate index) and l (lag index) from "w[k,l]"
w_df <- w_df %>%
  mutate(
    k   = as.integer(str_extract(variable, "(?<=\\[)\\d+")),
    l   = as.integer(str_extract(variable, "(?<=,)\\d+(?=\\])")),
    lag = l - 1L,   # Stan is 1-indexed: l=1 → lag 0, l=2 → lag 1, etc.
    covariate = lag_var_names[k]
  ) %>%
  filter(!is.na(k), !is.na(l), k <= length(lag_var_names), lag <= max_lag)

# =========================
# PLOT
# =========================

p <- ggplot(w_df, aes(x = lag, y = mean)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_ribbon(aes(ymin = q5, ymax = q95), fill = "steelblue", alpha = 0.25) +
  geom_line(colour = "steelblue", linewidth = 1) +
  geom_point(colour = "steelblue", size = 2.5) +
  facet_wrap(~ covariate, ncol = 2, scales = "free_y") +
  scale_x_continuous(breaks = 0:max_lag, labels = paste0("lag ", 0:max_lag)) +
  labs(
    x       = "Lag (months)",
    y       = "Weight (log-odds scale)",
    title   = "Posterior lag weight profiles",
    subtitle = "Mean ± 90% CI. Random walk prior — shape reveals functional form.",
    caption = "Ribbon: q5–q95 from model summary. Dashed line: zero."
  ) +
  theme_minimal() +
  theme(
    strip.text       = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

print(p)

out_file <- file.path(output_dir, "plots", "lag_weight_profiles.png")
ggsave(out_file, p, width = 10, height = 7, dpi = 150)
cat("Plot saved to:", out_file, "\n")
