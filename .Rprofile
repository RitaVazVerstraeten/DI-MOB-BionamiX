# Project-level .Rprofile for interactive convenience in VS Code
# Loaded for interactive sessions. Provides helpers to source files into
# the global environment and to assign values there.
if (interactive()) {
  # Source a file into the global environment (preserves usual source args)
  source_global <- function(file, ..., envir = .GlobalEnv) {
    base::source(file, local = envir, ...)
  }

  # Assign into global environment explicitly (useful from scripts)
  assign_global <- function(x, value) assign(x, value, envir = .GlobalEnv)

  # Ensure the helpers are available in the interactive global environment
  assign("source_global", source_global, envir = .GlobalEnv)
  assign("assign_global", assign_global, envir = .GlobalEnv)

  # Mark that the project profile loaded and print a short message so the
  # user can see it in the R console (helps troubleshooting when VS Code
  # doesn't automatically load project .Rprofile).
  options(DI_MOB_BionamiX.profile_loaded = TRUE)
  message("Loaded project .Rprofile: helpers available -> source_global(), assign_global()")

  # Optional: a convenience wrapper to source the currently open file in VS Code
  # if you use the built-in "Run Selected Text in Active Terminal" command
  # you can select the whole file and run it; alternatively call:
  # source_global("path/to/file.R")
}
