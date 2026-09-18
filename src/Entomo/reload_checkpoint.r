# =====================================================
# Reload a model checkpoint without re-running Hierarch_StateSpace_Entomo_model.r
# =====================================================
# Use this after restarting R (e.g. to pick up a package that needed a fresh
# session) when you just want to call one of the save_*() plot functions by
# hand, without re-reading the data, rebuilding the DLNM cross-basis, or
# re-triggering every plot as a side effect of re-sourcing the full model
# script. Requires that Hierarch_StateSpace_Entomo_model.r has already run at
# least once for this model_spec (it writes the checkpoint automatically,
# right before the "GENERATE PLOTS" section).
#
# After running this script, fit/prep/df/stan_data/cfg/sf_blocks/municipality/
# block_ids/model_spec/run_output_dir/plots_output_dir/post are all restored
# in .GlobalEnv, and helper_functions.r/plot_functions.r are sourced, so e.g.
#   save_infestation_risk_gif(df, sf_blocks, cfg, plots_output_dir, model_spec,
#                              municipality = municipality)
# can be called directly.
#
# By default this finds the most recently modified checkpoint_*.rds anywhere
# under cfg$output_dir's base path (same hostname branches as the main
# script). To reload a specific, older run instead, set checkpoint_path
# explicitly before sourcing this file, e.g.:
#   checkpoint_path <- "/path/to/checkpoint_CMF_....rds"
#   source("reload_checkpoint.r")

setwd("/home/rita/PyProjects/DI-MOB-BionamiX/src/Entomo")
source("renv/activate.R")
suppressWarnings(suppressMessages({
  library(dplyr); library(ggplot2); library(sf); library(scales); library(patchwork)
}))
source("helper_functions.r")
source("plot_functions.r")

if (!exists("checkpoint_path")) {
  hostname <- Sys.info()["nodename"]
  base_dir <- if (hostname == "frietjes") "/home/rita/data/Entomo/fitting/stan"
              else if (hostname == "stoofvlees") "~/data/entomo/results/fitting/stan"
              else "/home/rita/PyProjects/DI-MOB-BionamiX/results/Entomo/fitting/stan"
  base_dir <- path.expand(base_dir)

  candidates <- list.files(base_dir, pattern = "^checkpoint_.*\\.rds$",
                            recursive = TRUE, full.names = TRUE)
  if (length(candidates) == 0) {
    stop("No checkpoint_*.rds found under ", base_dir,
         " -- run Hierarch_StateSpace_Entomo_model.r at least once first, ",
         "or set checkpoint_path explicitly before sourcing this file.")
  }
  checkpoint_path <- candidates[order(file.info(candidates)$mtime, decreasing = TRUE)][1]
}

cat("Loading checkpoint:", checkpoint_path, "\n")
checkpoint <- readRDS(checkpoint_path)
list2env(checkpoint, envir = .GlobalEnv)
cat("Restored:", paste(names(checkpoint), collapse = ", "), "\n")
cat("Ready -- call any save_*() plot function directly, e.g.:\n",
    "  save_infestation_risk_gif(df, sf_blocks, cfg, plots_output_dir, model_spec, municipality = municipality)\n",
    sep = "")
