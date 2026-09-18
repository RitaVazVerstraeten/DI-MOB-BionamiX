# run_2011_2019_noNDVI.R
#
# Runs the standard CMF_DLNM_AR1perCMF_noGP_blockRE calibration
# (Hierarch_StateSpace_Entomo_model.r) on the extended 2011-2019 dataset
# built in Create_entomo_epi_env_database.rmd ("Build 2011-2019 dataset
# (no NDVI/NDMI/NDWI at all)"): meteo lag lead-in extended back to 2011
# using the full 2010-2024 station record, with mean_ndvi dropped entirely
# (satellite NDVI/NDMI/NDWI only exist from 2015 onward, so it can't be
# backfilled 5 years back the way the 2015-2019 variant backfills 1 year).
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
  # mean_ndvi dropped -- not available for the 2011-2015 lag lead-in period.
  unlagged_vars  = c("HFP_urbanization", "is_WUI", "water_containers"),
  numeric_vars   = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization")
)
.hierarch_run_suffix <- paste0(format(Sys.Date(), "%Y%m%d"), "_2011_2019_noNDVI")

# Same eval(parse(...), envir = globalenv()) pattern as the other run_*.R
# wrapper scripts (e.g. run_boundary_knots_test.R) rather than plain source(),
# so cfg override pickup and the existing_csv/checkpoint globals behave
# identically to those sweep scripts.
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))
eval(model_exprs, envir = globalenv())
