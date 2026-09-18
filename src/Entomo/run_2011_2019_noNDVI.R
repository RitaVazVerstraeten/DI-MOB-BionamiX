# run_2011_2019_noNDVI.R
#
# Runs the standard CMF_DLNM_AR1perCMF_noGP_blockRE calibration
# (Hierarch_StateSpace_Entomo_model.r) on the full 2011-2019 dataset built in
# Create_entomo_epi_env_database.rmd ("Build 2011-2019 dataset (no
# NDVI/NDMI/NDWI at all)"): real entomological response for the ENTIRE
# 2011-2019 span (not just 2016-2019 with 2011-2015 as unused lag lead-in --
# the raw entomo file has genuine surveillance data back to 2011), with
# mean_ndvi dropped entirely (satellite NDVI/NDMI/NDWI only exist from 2015
# onward, so it can't be backfilled 5 years back the way the 2015-2019
# variant backfills 1 year).
#
# response_start is overridden to "2011_01" (the base script's default is
# "2016_01") so build_dlnm_stan_data() treats every 2011-2019 row with a
# real response as a Stan observation, instead of discarding 2011-2015 as
# pre-response lag history the way the 2015-2019 extended-lag variant does.
# Only the first max_lag months (2011_01-2011_05) are dropped, same as any
# DLNM fit -- they have no prior data to build a full lag window from.
#
# Before running on the SSH remote: copy
#   env_epi_entomo_data_per_CMF_2011_01_to_2019_12_noNDVI_noColinnearity.csv
# into that host's cfg$data_dir (~/data/Entomo on frietjes, ~/entomo_data on
# stoofvlees -- see Hierarch_StateSpace_Entomo_model.r's cfg$data_dir).
#
# Usage: Rscript run_2011_2019_noNDVI.R
# =============================================================================

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

.hierarch_cfg_override <- list(
  data_file_name = "env_epi_entomo_data_per_CMF_2011_01_to_2019_12_noNDVI_noColinnearity.csv",
  # Base script default is "2016_01" -- override so 2011-2015 rows (which now
  # carry real Houses_pos_IS/Inspected_houses, not NA) are treated as Stan
  # observations rather than discarded as pre-response lag history.
  response_start = "2012_01",
  # mean_ndvi dropped -- not available before 2015.
  unlagged_vars  = c("HFP_urbanization", "is_WUI", "water_containers"),
  numeric_vars   = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization")
)
.hierarch_run_suffix <- paste0(format(Sys.Date(), "%Y%m%d"), "_2012_2019_noNDVI")

# Same eval(parse(...), envir = globalenv()) pattern as the other run_*.R
# wrapper scripts (e.g. run_boundary_knots_test.R) rather than plain source(),
# so cfg override pickup and the existing_csv/checkpoint globals behave
# identically to those sweep scripts.
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))
eval(model_exprs, envir = globalenv())
