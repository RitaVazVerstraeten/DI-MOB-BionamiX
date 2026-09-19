# run_2012_2019_NDXIbackfilled.R
#
# Runs the standard CMF_DLNM_AR1perCMF_noGP_blockRE calibration
# (Hierarch_StateSpace_Entomo_model.r) on the full 2012-2019 dataset built in
# Create_entomo_epi_env_database.rmd ("Build 2012-2019 dataset (NDVI/NDMI/NDWI
# climatology-backfilled for 2011-2015)"): real entomological+epi response for
# the ENTIRE 2012-2019 span (not just 2016-2019 with 2012-2015 as unused lag
# lead-in -- the raw entomo file has genuine surveillance data back to 2011,
# but the DENV/case data only starts 2012, which is why the response period
# starts 2012 rather than 2011).
#
# NDVI/NDMI/NDWI: real satellite coverage only starts 2016, but this variant's
# response period starts 2012, so mean_ndvi is NOT backfilled the same way as
# the 2015-2019 variant (fill-the-gap-only). Instead every row's mean_ndvi is
# fixed to that CMF's own average for that calendar month, computed across
# whatever real years exist 2011-2019 (in practice 2016-2019) -- see the
# Create script section for the full rationale. This means mean_ndvi carries
# no real inter-annual variation in ANY year of this dataset, backfilled or
# observed, and 4 of the 8 response years (2012-2015) never had real NDVI at
# all to begin with.
#
# response_start is overridden to "2012_01" (the base script's default is
# "2016_01") so build_dlnm_stan_data() treats every 2012-2019 row with a
# real response as a Stan observation, instead of discarding 2012-2015 as
# pre-response lag history the way the 2015-2019 extended-lag variant does.
# Only the first max_lag months (2012_01-2012_05) are dropped, same as any
# DLNM fit -- they have no prior data to build a full lag window from (2011
# in this dataset exists purely to supply that lag history).
#
# Before running on the SSH remote: copy
#   env_epi_entomo_data_per_CMF_2012_01_to_2019_12_NDXIbackfilled_noColinnearity.csv
# into that host's cfg$data_dir (~/data/Entomo on frietjes, ~/entomo_data on
# stoofvlees -- see Hierarch_StateSpace_Entomo_model.r's cfg$data_dir).
#
# Usage: Rscript run_2012_2019_NDXIbackfilled.R
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
  data_file_name = "env_epi_entomo_data_per_CMF_2012_01_to_2019_12_NDXIbackfilled_noColinnearity.csv",
  # Base script default is "2016_01" -- override so 2012-2015 rows (which now
  # carry real Houses_pos_IS/Inspected_houses/cases, not NA) are treated as
  # Stan observations rather than discarded as pre-response lag history.
  response_start = "2012_01",
  # mean_ndvi restored -- this variant backfills it (CMF-month average),
  # unlike the earlier noNDVI variant.
  unlagged_vars  = c("HFP_urbanization", "mean_ndvi", "is_WUI", "water_containers"),
  numeric_vars   = c("total_precip", "avg_VPD", "precip_max_day_resid_on_tp", "water_containers", "HFP_urbanization", "mean_ndvi"),
  # This run has ~2x the response months of the 2016-2019 fits, so its
  # checkpoint.rds + raw chain CSVs would be several GB -- skip both to
  # avoid filling the disk during an unattended run. See the checkpoint
  # section and the end-of-script cleanup in Hierarch_StateSpace_Entomo_model.r.
  keep_raw_draws = FALSE
)
.hierarch_run_suffix <- paste0(format(Sys.Date(), "%Y%m%d"), "_2012_2019_NDXIbackfilled")

# Same eval(parse(...), envir = globalenv()) pattern as the other run_*.R
# wrapper scripts (e.g. run_boundary_knots_test.R) rather than plain source(),
# so cfg override pickup and the existing_csv/checkpoint globals behave
# identically to those sweep scripts.
model_exprs <- parse(file.path(script_dir, "Hierarch_StateSpace_Entomo_model.r"))
eval(model_exprs, envir = globalenv())
