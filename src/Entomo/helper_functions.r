# =====================================================
# Helper functions for Stan entomological model
# Data preparation, posterior extraction, and plotting
# =====================================================

# =========================
# DATA PREPARATION
# =========================

#' Load Base Entomological Data
#'
#' Reads CSV data file and creates a proper date column from year_month.
#' Removes unnecessary columns (CMF, CP, AREA) and reorders columns.
#'
#' @param data_file Character string path to the CSV data file
#' @return A data frame with cleaned entomological data including year_month_date
load_base_data <- function(data_file) {
  read_csv(data_file, show_col_types = FALSE) %>%
    mutate(year_month_date = as.Date(paste0(year_month, "_01"), "%Y_%m_%d")) %>%
    relocate(year_month_date, .after = year_month) %>%
    select(!any_of(c("CMF", "CP", "AREA")))
}

#' Index and Subset Data by Blocks
#'
#' Creates numeric block and time indices, renames key variables to Stan conventions,
#' and optionally subsets to a specified number of blocks. Ensures data is sorted
#' by block and time.
#'
#' @param input_data Data frame with manzana (block) and year_month_date columns
#' @param n_blocks Integer number of blocks to keep (NULL = keep all blocks)
#' @return List with elements: df (processed data frame), B (number of blocks), T (number of time points)
index_and_subset <- function(input_data, n_blocks, block_col = "manzana") {
  block_levels <- sort(unique(input_data[[block_col]]))
  time_levels <- sort(unique(input_data$year_month_date))

  df <- input_data %>%
    mutate(
      block = match(.data[[block_col]], block_levels),
      time = match(year_month_date, time_levels)
    ) %>%
    arrange(block, time) %>%
    rename(N_HH = Inspected_houses, C_bt = cases, y_bt = Houses_pos_IS) %>%
    mutate(N_HH = as.integer(N_HH), C_bt = as.integer(C_bt), y_bt = as.integer(y_bt))

  all_blocks <- sort(unique(df$block))
  selected_blocks <- if (is.null(n_blocks)) {
    all_blocks
  } else {
    all_blocks[seq_len(min(n_blocks, length(all_blocks)))]
  }
  df <- df %>% filter(block %in% selected_blocks)

  time_levels <- sort(unique(df$year_month_date))
  df <- df %>% mutate(time = match(year_month_date, time_levels))

  list(df = df, B = length(selected_blocks), T = length(time_levels))
}

#' Build Distributed Lag Design Matrix
#'
#' Creates a 3D array of lagged variables (X_lag) for each observation, variable, and lag.
#' Flattens into a 2D matrix for Stan. Removes observations where lags cannot be computed
#' (first max_lag time points for each block). Handles block structure properly.
#'
#' @param df Data frame with block, time, and lagged variables
#' @param lag_vars Character vector of variable names to create lags for
#' @param B Integer number of blocks
#' @param max_lag Integer maximum lag order (L)
#' @return List with df (filtered data frame), X_lag_flat (N x K*(L+1) matrix), K (number of variables), Lp1 (L+1)
build_lag_design <- function(df, lag_vars, B, max_lag) {
  # Expand categorical lag variables into dummy indicator columns so that
  # categorical lags are represented as separate binary variables.
  lag_vars_expanded <- character()
  for (v in lag_vars) {
    if (!v %in% names(df)) stop(sprintf("Lag variable '%s' not found in data", v))
    if (is.numeric(df[[v]])) {
      # numeric: replace NAs with 0 and keep as-is
      df[[v]][is.na(df[[v]])] <- 0
      lag_vars_expanded <- c(lag_vars_expanded, v)
    } else {
      # categorical: convert to factor and create one-hot columns for each level
      fac <- as.factor(df[[v]])
      levels_fac <- levels(fac)
      # create safe level names
      safe_levels <- gsub("[^A-Za-z0-9]", "_", levels_fac)
      for (i in seq_along(levels_fac)) {
        colname <- paste0(v, "__", safe_levels[i])
        df[[colname]] <- as.integer(fac == levels_fac[i])
        df[[colname]][is.na(df[[colname]])] <- 0
        lag_vars_expanded <- c(lag_vars_expanded, colname)
      }
    }
  }
  K <- length(lag_vars_expanded)
  L <- max_lag
  Lp1 <- L + 1
  X_lag <- array(0, dim = c(nrow(df), K, Lp1))
  for (b in 1:B) {
    idx <- which(df$block == b)
    for (k in seq_len(K)) {
      x <- df[[lag_vars_expanded[k]]][idx]
      for (l in 0:L) {
        lagged <- if (l == 0) x else c(rep(NA_real_, l), x[1:(length(x) - l)])
        X_lag[idx, k, l + 1] <- lagged
      }
    }
  }

  rows_to_keep <- df$time > L
  df <- df[rows_to_keep, ]
  X_lag <- X_lag[rows_to_keep, , ]

  X_lag_flat <- matrix(0, nrow = nrow(df), ncol = K * Lp1)
  for (i in seq_len(nrow(df))) X_lag_flat[i, ] <- as.vector(X_lag[i, , ])

  # Return expanded variable names so downstream code can know which lag columns correspond
  list(df = df, X_lag_flat = X_lag_flat, K = K, Lp1 = Lp1, lag_vars_expanded = lag_vars_expanded)
}

#' Prepare Unlagged Covariates
#'
#' Extracts unlagged variables and standardizes continuous variables (z-score).
#' Binary variables are left unstandardized (0/1). Missing values are replaced with 0.
#'
#' @param df Data frame containing unlagged variables
#' @param unlagged_vars Character vector of all unlagged variable names
#' @param binary_unlagged_vars Character vector of binary variable names (subset of unlagged_vars)
#' @return List with df (data frame), X_unlagged_std (N x Ku standardized matrix), Ku (number of unlagged variables)
prepare_unlagged <- function(df, unlagged_vars, binary_unlagged_vars) {
  continuous_unlagged_vars <- setdiff(unlagged_vars, binary_unlagged_vars)
  df <- df %>% mutate(across(all_of(unlagged_vars), ~coalesce(., 0)))

  X_unlagged <- as.matrix(df[, unlagged_vars])
  X_unlagged_std <- X_unlagged

  if (length(continuous_unlagged_vars) > 0) {
    cont_idx <- match(continuous_unlagged_vars, colnames(X_unlagged))
    means <- colMeans(X_unlagged[, cont_idx, drop = FALSE])
    sds <- apply(X_unlagged[, cont_idx, drop = FALSE], 2, sd)
    sds[sds == 0 | is.na(sds)] <- 1
    X_unlagged_std[, cont_idx] <- scale(X_unlagged[, cont_idx, drop = FALSE], center = means, scale = sds)
  }

  list(df = df, X_unlagged_std = X_unlagged_std, Ku = ncol(X_unlagged))
}

#' Standardize Matrix (Z-score)
#'
#' Centers and scales each column to mean 0 and standard deviation 1.
#' Handles zero-variance columns by setting their scale to 1 (no scaling).
#'
#' @param x Numeric matrix to standardize
#' @return Standardized matrix with same dimensions as input
standardize_matrix <- function(x) {
  m <- colMeans(x, na.rm = TRUE)
  s <- apply(x, 2, sd, na.rm = TRUE)
  s[s == 0 | is.na(s)] <- 1
  scale(x, center = m, scale = s)
}

#' Build Complete Stan Data List
#'
#' Orchestrates the full data preparation pipeline: loads data, creates indices,
#' builds lag matrix, prepares unlagged covariates, and standardizes. Returns
#' both the Stan data list and the processed data frame.
#'
#' @param cfg List containing all configuration parameters (data_file, n_blocks, lag_vars,
#'            max_lag, unlagged_vars, binary_unlagged_vars, kappa)
#' @return List with stan_data (list ready for Stan) and df (processed data frame)
#' Build ICAR Edge List from Block Polygons
#'
#' Uses spdep::poly2nb() with a snap tolerance to define neighbours between
#' block polygons that may not share perfectly touching borders. Returns
#' deduplicated undirected edge pairs (node1 < node2) in the same ordering
#' as block_ids.
#'
#' @param sf_blocks sf object of block polygons (already loaded)
#' @param block_ids Character vector of block IDs in model order
#' @param sf_block_col Name of the block ID column in sf_blocks
#' @param snap_m Snap distance in metres (default 100)
#' @return List with N_edges, node1, node2
#' Compute BYM2 Scaling Factor
#'
#' Scales the ICAR component to unit marginal variance so that sigma_spatial
#' has a consistent interpretation regardless of graph structure. Uses the
#' geometric mean of the diagonal of the ICAR precision matrix pseudoinverse
#' (Riebler et al. 2016).
#'
#' @param node1 Integer vector of first nodes of each undirected edge
#' @param node2 Integer vector of second nodes of each undirected edge
#' @param B Integer number of blocks
#' @return Scalar scaling factor (pass as stan_data$scaling_factor)
compute_bym2_scaling <- function(node1, node2, B) {
  if (!requireNamespace("Matrix", quietly = TRUE))
    stop("Package 'Matrix' required for BYM2 scaling — install it with install.packages('Matrix')")
  # build adjacency matrix
  adj <- Matrix::sparseMatrix(
    i    = c(node1, node2),
    j    = c(node2, node1),
    x    = 1,
    dims = c(B, B)
  )
  # creates precision matrix of ICAR -> diagonal entry for block i is the number of neighbours for i, the off-diagonal entries are -1 for neighbours and 0 otherwise 
  Q     <- Matrix::Diagonal(x = Matrix::rowSums(adj)) - adj
  # regularize and invert
  Q_perturb <- Q + Matrix::Diagonal(B, 1e-6)
  Q_inv <- Matrix::solve(Q_perturb)

  # compute scaling factor (geometric mean of the marginal variance)
  sf    <- exp(mean(log(Matrix::diag(Q_inv))))
  cat(sprintf("BYM2 scaling factor: %.4f\n", sf))
  sf
}

build_icar_edges <- function(sf_blocks, block_ids, sf_block_col, snap_m = 100) {
  sf_model <- sf_blocks %>%
    mutate(block_chr = as.character(.data[[sf_block_col]])) %>%
    filter(block_chr %in% block_ids) %>%
    arrange(match(block_chr, block_ids))

  if (st_is_longlat(sf_model))
    sf_model <- st_transform(sf_model, 3857)

  # create the neighbours
  nb <- suppressWarnings(spdep::poly2nb(sf_model, snap = snap_m))
  # number of islands (no neighbours)
  n_islands <- sum(sapply(nb, function(x) length(x) == 1L && x == 0L))
  cat(sprintf("poly2nb (snap=%dm): %d blocks, %d islands\n",
              snap_m, length(nb), n_islands))

  edges <- do.call(rbind, lapply(seq_along(nb), function(i) {
    nbrs <- nb[[i]]
    if (length(nbrs) == 1L && nbrs == 0L) return(NULL)
    data.frame(node1 = i, node2 = nbrs)
  }))

  # create the ICAR nodes 
  node1_v      <- pmin(edges$node1, edges$node2)
  node2_v      <- pmax(edges$node1, edges$node2)
  unique_edges <- !duplicated(paste(node1_v, node2_v, sep = "_"))
  list(
    N_edges = sum(unique_edges),
    node1   = node1_v[unique_edges],
    node2   = node2_v[unique_edges]
  )
}

#' Build Stan Data with DLNM Cross-Basis Predictors
#'
#' Replaces the distributed-lag flat matrix (X_lag_flat / K / Lp1) with a
#' DLNM cross-basis matrix (X_cb / P_cb) built via dlnm::crossbasis().
#' Uses the full (unfiltered) data to construct lag matrices, then filters
#' rows where time <= max_lag exactly as build_lag_design does.
#'
#' Extended-lag mode (cfg$response_start set, e.g. "2015_01"):
#'   The dataset includes pre-response rows (e.g. 2015) that have NA for the
#'   entomological response variables. These rows are used when building the
#'   cross-basis lag matrix so that early response observations (e.g. Jan 2016)
#'   have a full lag history. Only rows where Inspected_houses is not NA (i.e.
#'   the response period) are passed to Stan as observations; the full time span
#'   (including 2015) is retained for the AR(1) state index T so the AR(1)
#'   warms up naturally through the pre-response period.
#'
#' @param cfg Config list. Must include dlnm_vars (character vector of predictor
#'   names), max_lag, and optionally dlnm_argvar (named list of per-variable
#'   argvar specs) and dlnm_arglag (single arglag spec shared across vars).
#'   Set cfg$response_start (e.g. "2016_01") to activate extended-lag mode.
#'   Defaults: argvar = list(fun="ns", df=3), arglag = list(fun="ns", df=3).
#' @return Same shape as build_stan_data(): list(stan_data, df, dlnm_vars, cb_mats, unlagged_vars)
build_dlnm_stan_data <- function(cfg) {
  if (!requireNamespace("dlnm", quietly = TRUE))
    stop("Package 'dlnm' required — install.packages('dlnm')")

  input_data <- load_base_data(cfg$data_file)
  block_col  <- if (!is.null(cfg$block_col)) cfg$block_col else "manzana"
  idx        <- index_and_subset(input_data, cfg$n_blocks, block_col = block_col)

  # Scoped to dlnm_vars only (not cfg$numeric_vars as a whole): dlnm_vars need
  # to be standardized here, on the full pre-filter population, because the
  # cross-basis is built from the full time series (lag windows reach back
  # into pre-response rows). Unlagged numeric vars (e.g. mean_ndvi) are
  # standardized once, later, by prepare_unlagged() on the response-period
  # population — standardizing them here too would silently double-apply:
  # prepare_unlagged() re-derives mean/sd from the already-transformed column,
  # which produces a *different* final value than either pass alone, and
  # diverges from what's saved in dlnm_var_stats/returned in `df`.
  vars_to_std <- intersect(cfg$dlnm_vars, names(idx$df))

  # Save mean/SD for DLNM vars BEFORE standardizing (needed for back-transformation in plots)
  dlnm_var_stats <- setNames(lapply(cfg$dlnm_vars, function(v) {
    if (v %in% vars_to_std) {
      x <- idx$df[[v]][is.finite(idx$df[[v]])]
      s <- sd(x)
      list(mean = mean(x), sd = if (s == 0 | is.na(s)) 1 else s)
    } else {
      list(mean = 0, sd = 1)   # not standardized; original = standardized
    }
  }), cfg$dlnm_vars)

  idx$df[, vars_to_std] <- standardize_matrix(as.matrix(idx$df[, vars_to_std]))

  B      <- idx$B
  L      <- cfg$max_lag
  df_all <- idx$df   # full data (all time points)

  # Default basis specs
  default_argvar <- list(fun = "ns", df = 3)
  default_arglag <- list(fun = "ns", df = 3)

  # Strip arguments that are incompatible with the basis function.
  # lin accepts no extra args; strata only accepts breaks.
  # This prevents leftover df/knots fields from causing errors.
  clean_basis_spec <- function(spec) {
    if (!is.list(spec) || is.null(spec$fun)) return(spec)
    if (spec$fun == "lin")    return(list(fun = "lin"))
    if (spec$fun == "strata") return(list(fun = "strata", breaks = spec$breaks))
    spec
  }

  # dlnm_arglag may be either a single global spec (unnamed list with fun/df keys)
  # or a named list keyed by dlnm_var name for per-variable lag bases.
  arglag_is_per_var <- !is.null(cfg$dlnm_arglag) &&
                       !is.null(names(cfg$dlnm_arglag)) &&
                       any(names(cfg$dlnm_arglag) %in% cfg$dlnm_vars)

  cat("Building DLNM cross-bases (max_lag =", L, "):\n")
  cb_mats <- list()

  for (var in cfg$dlnm_vars) {
    if (!var %in% names(df_all))
      stop(sprintf("DLNM variable '%s' not found in data", var))

    # Build Q[N_all, L+1]: lag matrix before row filtering
    # Row order matches df_all row order (block x time, sorted)
    Q <- matrix(NA_real_, nrow = nrow(df_all), ncol = L + 1)
    for (b in seq_len(B)) {
      rows <- which(df_all$block == b)
      x    <- df_all[[var]][rows]
      for (l in 0:L) {
        Q[rows, l + 1] <- if (l == 0) x else c(rep(NA_real_, l), x[seq_len(length(x) - l)])
      }
    }

    argvar <- clean_basis_spec(
      if (!is.null(cfg$dlnm_argvar) && var %in% names(cfg$dlnm_argvar))
        cfg$dlnm_argvar[[var]] else default_argvar
    )
    arglag <- clean_basis_spec(
      if (arglag_is_per_var && var %in% names(cfg$dlnm_arglag))
        cfg$dlnm_arglag[[var]]
      else if (arglag_is_per_var)
        default_arglag
      else if (!is.null(cfg$dlnm_arglag))
        cfg$dlnm_arglag
      else
        default_arglag
    )

    cb_mats[[var]] <- dlnm::crossbasis(Q, lag = c(0, L), argvar = argvar, arglag = arglag)
    cat(sprintf("  %-32s  %d columns\n", var, ncol(cb_mats[[var]])))
  }

  # Determine which rows are response observations.
  # Extended-lag mode: dataset contains pre-response rows (e.g. 2015) with NA
  # ento data; those rows provided lag history above but are not Stan observations.
  # Standard mode: drop the first max_lag rows per block (no full history yet).
  if (!is.null(cfg$response_start)) {
    response_date <- as.Date(paste0(cfg$response_start, "_01"), "%Y_%m_%d")
    # Also require time > L so rows without a full lag window (no pre-response
    # data in this block) are always excluded, even if response_start is set
    # for a dataset that contains no pre-response rows.
    keep <- !is.na(df_all$y_bt) & df_all$year_month_date >= response_date & df_all$time > L
    n_pre <- sum(df_all$year_month_date < response_date)
    cat(sprintf(
      "Extended-lag mode: response from %s — %d response rows, %d pre-response rows used for lag history\n",
      cfg$response_start, sum(keep), n_pre
    ))
    if (n_pre == 0)
      warning("response_start is set but dataset contains no pre-response rows. ",
              "Use the 2015-2019 extended-lag dataset or set response_start = NULL.")
  } else {
    keep <- df_all$time > L
  }
  df_filt <- df_all[keep, ]

  X_cb <- do.call(cbind, lapply(cb_mats, function(cb) {
    m <- cb[keep, , drop = FALSE]
    matrix(as.numeric(m), nrow = sum(keep), ncol = ncol(cb))
  }))
  P_cb <- ncol(X_cb)
  cat("DLNM total cross-basis columns P_cb =", P_cb, "\n")

  # Per-variable column counts in X_cb (needed for interaction construction)
  cb_ncols      <- sapply(cfg$dlnm_vars, function(v) ncol(cb_mats[[v]]))
  col_starts_cb <- cumsum(c(1L, cb_ncols[-length(cb_ncols)]))

  # Build interaction cross-basis X_ix: for each interaction spec, multiply a
  # per-row binary indicator into the corresponding DLNM sub-block of X_cb.
  #   binary_var + active_level: 0/1 indicator, 1 where binary_var equals
  #     active_level (e.g. 0 for non-urban when is_urban is coded 1 = urban;
  #     1 for water_shortage TRUE). w_ix is then "effect when the indicator
  #     switches on" relative to the w_cb baseline.
  # Continuous modifiers are not supported: a linear-in-modifier tilt of the
  # whole cross-basis, evaluated only at the modifier's mean and +1 SD, is
  # hard to interpret as anything other than an arbitrary two-point probe of
  # what is actually a continuous effect-modification surface. Use a binary
  # split (e.g. above/below median) if a continuous variable's effect
  # modification needs to be tested.
  if (!is.null(cfg$dlnm_ix_vars) && length(cfg$dlnm_ix_vars) > 0) {
    ix_mats <- lapply(cfg$dlnm_ix_vars, function(ix) {
      dlnm_var <- ix$dlnm_var
      if (!dlnm_var %in% cfg$dlnm_vars)
        stop(sprintf("dlnm_ix_vars: DLNM variable '%s' not in cfg$dlnm_vars", dlnm_var))

      binary_var   <- ix$binary_var
      active_level <- ix$active_level
      if (!binary_var %in% names(df_filt))
        stop(sprintf("dlnm_ix_vars: binary variable '%s' not found in data", binary_var))
      raw_num  <- suppressWarnings(as.numeric(df_filt[[binary_var]]))
      modifier <- as.numeric(raw_num == active_level)
      if (any(is.na(modifier)))
        stop(sprintf("dlnm_ix_vars: NA in indicator for '%s' at active_level = %s",
                     binary_var, active_level))

      var_idx   <- which(cfg$dlnm_vars == dlnm_var)
      col_start <- col_starts_cb[var_idx]
      col_end   <- col_start + cb_ncols[var_idx] - 1L
      X_cb[, col_start:col_end, drop = FALSE] * modifier
    })
    X_ix <- do.call(cbind, ix_mats)
    P_ix <- ncol(X_ix)
    cat(sprintf("Interaction cross-basis: %d pair(s), P_ix = %d columns\n",
                length(cfg$dlnm_ix_vars), P_ix))
    for (ix in cfg$dlnm_ix_vars) {
      cat(sprintf("  %s (level=%s) x %s  [%d cols]\n",
                  ix$binary_var, ix$active_level, ix$dlnm_var,
                  cb_ncols[which(cfg$dlnm_vars == ix$dlnm_var)]))
    }
  } else {
    X_ix <- matrix(0.0, nrow = nrow(X_cb), ncol = 0L)
    P_ix <- 0L
    cat("No DLNM interaction cross-basis (cfg$dlnm_ix_vars not set)\n")
  }

  binary_unlagged_vars <- setdiff(cfg$unlagged_vars, cfg$numeric_vars)
  unl <- prepare_unlagged(df_filt, cfg$unlagged_vars, binary_unlagged_vars)

  list(
    stan_data = list(
      N          = nrow(unl$df),
      y          = unl$df$y_bt,
      P_cb       = P_cb,
      X_cb       = X_cb,
      P_ix       = P_ix,
      X_ix       = X_ix,
      Ku         = unl$Ku,
      X_unlagged = unl$X_unlagged_std,
      B          = idx$B,
      T          = idx$T,
      block      = unl$df$block,
      time       = unl$df$time,
      C_bt       = unl$df$C_bt,
      n_bt       = as.integer(unl$df$N_HH + cfg$kappa * unl$df$C_bt),
      kappa      = cfg$kappa
    ),
    df             = unl$df,
    dlnm_vars      = cfg$dlnm_vars,
    cb_mats        = cb_mats,
    dlnm_var_stats = dlnm_var_stats,
    unlagged_vars  = cfg$unlagged_vars,
    dlnm_ix_vars   = if (!is.null(cfg$dlnm_ix_vars)) cfg$dlnm_ix_vars else list()
  )
}

build_stan_data <- function(cfg) {
  input_data <- load_base_data(cfg$data_file)
  block_col <- if (!is.null(cfg$block_col)) cfg$block_col else "manzana"
  idx <- index_and_subset(input_data, cfg$n_blocks, block_col = block_col)

  # Standardize lag vars on the raw time series before lagging, so all lags of
  # the same variable share the same mean/sd. Scoped to cfg$lag_vars only (not
  # cfg$numeric_vars as a whole) — unlagged numeric vars are standardized once,
  # later, by prepare_unlagged() on the response-period population; including
  # them here too would double-standardize them (see build_dlnm_stan_data()).
  vars_to_std <- intersect(cfg$lag_vars, names(idx$df))
  idx$df[, vars_to_std] <- standardize_matrix(as.matrix(idx$df[, vars_to_std]))

  lag <- build_lag_design(idx$df, cfg$lag_vars, idx$B, cfg$max_lag)
  # Inform user which lag variables were expanded/used
  if (!is.null(lag$lag_vars_expanded)) {
    cat("Lag variables expanded:", paste(lag$lag_vars_expanded, collapse = ", "), "\n")
  }

  # Extended-lag mode: drop pre-response rows (e.g. 2015) that provided lag
  # history but have no entomological observations (y_bt = NA).
  if (!is.null(cfg$response_start)) {
    response_date <- as.Date(paste0(cfg$response_start, "_01"), "%Y_%m_%d")
    keep_resp <- !is.na(lag$df$y_bt) & lag$df$year_month_date >= response_date
    n_pre <- sum(!keep_resp)
    lag$df        <- lag$df[keep_resp, ]
    lag$X_lag_flat <- lag$X_lag_flat[keep_resp, ]
    cat(sprintf(
      "Extended-lag mode: response from %s — %d response rows, %d pre-response rows used for lag history\n",
      cfg$response_start, nrow(lag$df), n_pre
    ))
  }

  binary_unlagged_vars <- setdiff(cfg$unlagged_vars, cfg$numeric_vars)
  unl <- prepare_unlagged(lag$df, cfg$unlagged_vars, binary_unlagged_vars)

  list(
    stan_data = list(
      N = nrow(unl$df),
      y = unl$df$y_bt,
      K = lag$K,
      Lp1 = lag$Lp1,
      X_lag_flat = lag$X_lag_flat,
      Ku = unl$Ku,
      X_unlagged = unl$X_unlagged_std,
      B = idx$B,
      T = idx$T,
      block = unl$df$block,
      time = unl$df$time,
      C_bt = unl$df$C_bt,
      n_bt = as.integer(unl$df$N_HH + cfg$kappa * unl$df$C_bt),
      kappa = cfg$kappa
    ),
    df = unl$df,
    lag_vars_expanded = if (!is.null(lag$lag_vars_expanded)) lag$lag_vars_expanded else cfg$lag_vars,
    unlagged_vars     = cfg$unlagged_vars
  )
}

# =========================
# MODEL FIT + EXTRACTION
# =========================

#' Create Initialization Function for Stan
#'
#' Returns a function that generates random initial values for MCMC chains.
#' Conditionally includes temporal random effect parameters (v_time_raw, sigma_v, rho)
#' only when temporal RE is enabled.
#'
#' @param stan_data List containing Stan data (used to determine dimensions)
#' @param use_temporal_re Logical flag indicating whether temporal RE is enabled in the model
#' @return Function that returns a list of initial values for one MCMC chain
make_init_fun <- function(stan_data, use_temporal_re, use_hsgp = FALSE,
                          use_icar = FALSE, use_bym2 = FALSE,
                          use_time_RE = FALSE, use_spatial_AC = TRUE,
                          use_block_dev = TRUE,
                          use_temporal_AR_perCMF = FALSE,
                          use_dlnm = FALSE) {
  function() {
    init_vals <- list(
      alpha      = rnorm(1, -4.5, 0.4),
      w_unlagged = rnorm(stan_data$Ku, 0, 0.1),
      delta1     = runif(1, 0.01, 0.05),
      phi_raw    = runif(1, 15, 30)
    )
    if (isTRUE(use_dlnm)) {
      init_vals$w_cb <- rnorm(stan_data$P_cb, 0, 0.08)
      if (!is.null(stan_data$P_ix) && stan_data$P_ix > 0)
        init_vals$w_ix <- rnorm(stan_data$P_ix, 0, 0.05)
    } else {
      init_vals$w <- matrix(rnorm(stan_data$K * stan_data$Lp1, 0, 0.08), stan_data$K, stan_data$Lp1)
    }

    if (isTRUE(use_time_RE)) {
      init_vals$v_time_raw  <- rnorm(stan_data$T, 0, 0.3)
      init_vals$sigma_time  <- runif(1, 0.05, 0.3)
      init_vals$u_block_raw <- rnorm(stan_data$B, 0, 0.3)
      init_vals$sigma_block <- runif(1, 0.05, 0.3)
    } else {
      if (isTRUE(use_spatial_AC)) {
        if (isTRUE(use_bym2)) {
          init_vals$u_icar_raw    <- rnorm(stan_data$B, 0, 0.1)
          init_vals$u_het_raw     <- rnorm(stan_data$B, 0, 0.1)
          init_vals$sigma_spatial <- runif(1, 0.1, 0.5)
          init_vals$phi_mix       <- runif(1, 0.3, 0.7)
        } else if (isTRUE(use_icar)) {
          init_vals$u_icar_raw <- rnorm(stan_data$B, 0, 0.1)
          init_vals$sigma_icar <- runif(1, 0.1, 0.4)
        } else {
          init_vals$sigma_gp <- runif(1, 0.1, 0.5)
          init_vals$rho_gp   <- runif(1, 50, 200)
          if (isTRUE(use_hsgp)) {
            init_vals$beta_gp <- rnorm(stan_data$M^2, 0, 0.1)
          } else {
            init_vals$z_gp <- rnorm(stan_data$B, 0, 0.1)
          }
        }
      }
      if (isTRUE(use_temporal_AR_perCMF)) {
        init_vals$v_raw <- matrix(rnorm(stan_data$B * stan_data$T, 0, 0.3), stan_data$B, stan_data$T)
        init_vals$tau   <- runif(1, 0.3, 0.8)
        init_vals$rho   <- rnorm(1, 0.3, 0.15)
        if (isTRUE(use_icar)) {
          # DLNM+ICAR or perCMF+ICAR: ICAR spatial field replaces block RE
          init_vals$u_icar_raw <- rnorm(stan_data$B, 0, 0.1)
          init_vals$sigma_icar <- runif(1, 0.1, 0.4)
        } else if (isTRUE(use_block_dev)) {
          init_vals$u_block_raw <- rnorm(stan_data$B, 0, 0.3)
          init_vals$sigma_block <- runif(1, 0.05, 0.3)
        }
      } else if (!isTRUE(use_temporal_re) && !isTRUE(use_spatial_AC) && isTRUE(use_block_dev)) {
        init_vals$u_block_raw <- rnorm(stan_data$B, 0, 0.3)
        init_vals$sigma_block <- runif(1, 0.05, 0.3)
      } else if (isTRUE(use_temporal_re)) {
        init_vals$v_global_raw <- rnorm(stan_data$T, 0, 0.3)
        init_vals$sigma_v      <- runif(1, 0.1, 0.5)
        init_vals$rho          <- rnorm(1, 0.3, 0.15)
        if (!isTRUE(use_bym2) && isTRUE(use_block_dev)) {
          init_vals$v_block_dev_raw <- rnorm(stan_data$B, 0, 0.3)
          init_vals$sigma_block_dev <- runif(1, 0.05, 0.3)
        }
      }
    }

    init_vals
  }
}

#' Rename w[k,l] and w_unlagged[i] rows in a cmdstanr summary tibble to actual variable names
rename_w_in_summary <- function(model_sum, lag_vars_expanded, unlagged_vars = NULL) {
  # w[k, l] → variable_lagN
  w_rows <- grepl("^w\\[", model_sum$variable)
  if (any(w_rows) && !is.null(lag_vars_expanded)) {
    parsed  <- regmatches(model_sum$variable[w_rows],
                          regexpr("[0-9]+,[0-9]+", model_sum$variable[w_rows]))
    k_idx   <- as.integer(sub(",.*", "", parsed))
    l_idx   <- as.integer(sub(".*,", "", parsed))
    model_sum$variable[w_rows] <- paste0(lag_vars_expanded[k_idx], "_lag", l_idx - 1)
  }
  # w_unlagged[i] → variable name
  wu_rows <- grepl("^w_unlagged\\[", model_sum$variable)
  if (any(wu_rows) && !is.null(unlagged_vars)) {
    i_idx   <- as.integer(regmatches(model_sum$variable[wu_rows],
                                     regexpr("[0-9]+", model_sum$variable[wu_rows])))
    model_sum$variable[wu_rows] <- unlagged_vars[i_idx]
  }
  model_sum
}

#' Extract Posterior Means from Stan Fit
#'
#' Computes posterior means for key parameters and generated quantities using
#' flexible regex pattern matching. Returns NA vectors if a parameter is not found
#' (e.g., v_time_out when temporal RE is disabled).
#'
#' @param fit Stan fit object (cmdstanr)
#' @param n_rows Integer number of observations (for sizing output vectors)
#' @return List with elements: p_bt, p_R, u (spatial RE), v (temporal RE), y_pred
extract_means <- function(fit, n_rows) {
  post <- fit$draws(format = "matrix")
  vnames <- colnames(post)

  extract <- function(pattern, n_fallback = 1) {
    cols <- grep(pattern, vnames)
    if (length(cols) == 0) return(rep(NA_real_, n_fallback))
    colMeans(post[, cols])
  }

  list(
    p_bt = extract("^p_bt_out\\[", n_rows),
    p_R = extract("^p_R_out\\[", n_rows),
    u = extract("^u_block_out\\[", 1),
    v = extract("^v_cmf_out\\[", 1),
    y_pred = extract("^y_pred\\[", n_rows)
  )
}

# =========================
# PLOTTING
# =========================

#' Save Random Effects Diagnostic Plot
#'
#' Creates a 2x2 grid plot showing spatial and temporal random effects diagnostics:
#' - Spatial: histogram and Q-Q plot
#' - Temporal: time series line plot and ACF (or placeholders if RE disabled)
#'
#' @param u_post Numeric vector of spatial random effects (u_block_out)
#' @param v_post Numeric vector of temporal random effects (v_time_out, or NA if disabled)
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filename
#' @return NULL (saves plot to PNG file)
save_random_effects <- function(u_post, v_post, output_dir, run_suffix) {
  png(file.path(output_dir, paste0("random_effects_", run_suffix, ".png")), width = 1000, height = 800)
  par(mfrow = c(2, 2))

  hist(u_post, breaks = 50, main = "Distribution of Spatial Random Effects (u_b)",
       xlab = "Effect", col = "lightblue", border = "white")
  abline(v = 0, lty = 2, col = "red", lwd = 2)

  qqnorm(u_post, main = "Q-Q Plot: Spatial Effects", pch = 19, cex = 0.5, col = "blue")
  qqline(u_post, col = "red", lwd = 2)

  if (!all(is.na(v_post))) {
    plot(v_post, type = "b", main = "Temporal Random Effects (v_t)",
         xlab = "Time", ylab = "Effect", col = "red", pch = 19)
    abline(h = 0, lty = 2, col = "gray")
    acf(v_post, main = "ACF of Temporal Effects", col = "darkred")
  } else {
    plot.new(); text(0.5, 0.5, "Temporal RE disabled\n(no v_time_out in model)")
    plot.new(); text(0.5, 0.5, "ACF unavailable\n(temporal RE disabled)")
  }

  par(mfrow = c(1, 1))
  dev.off()
}

#' Save MCMC Trace Plots
#'
#' Creates trace plots for MCMC diagnostics using bayesplot package.
#' Generates three separate plots: main parameters, lagged weights (w), and
#' unlagged weights (w_unlagged). Conditionally includes temporal RE parameters
#' (sigma_v, rho) if enabled.
#'
#' @param fit Stan fit object (cmdstanr)
#' @param output_dir Character string path to output directory
#' @param run_suffix Character string suffix for filenames
#' @param use_temporal_re Logical flag indicating whether temporal RE is enabled
#' @return NULL (saves plots to PNG files or returns invisibly if bayesplot not installed)
save_trace_plots <- function(fit, output_dir, run_suffix, use_temporal_re) {
  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    cat("bayesplot package not installed; skipping trace plots.\n")
    return(invisible(NULL))
  }

  library(bayesplot)
  draws_array <- fit$draws(format = "array")

  trace_dir <- file.path(output_dir, "traceplots")
  dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

  params_main <- c("alpha", "sigma_u", "delta0", "delta1")
  if (use_temporal_re) params_main <- c(params_main, "tau", "sigma_v", "rho")

  ggsave(
    file.path(trace_dir, paste0("traceplot_params_", run_suffix, ".png")),
    mcmc_trace(draws_array, pars = params_main), width = 10, height = 8
  )

  w_params <- grep("^w\\[", dimnames(draws_array)[[3]], value = TRUE)
  if (length(w_params) > 0) {
    ggsave(
      file.path(trace_dir, paste0("traceplot_weights_w_", run_suffix, ".png")),
      mcmc_trace(draws_array, pars = w_params), width = 12, height = 10
    )
  }

  wu_params <- grep("^w_unlagged\\[", dimnames(draws_array)[[3]], value = TRUE)
  if (length(wu_params) > 0) {
    ggsave(
      file.path(trace_dir, paste0("traceplot_weights_unlagged_", run_suffix, ".png")),
      mcmc_trace(draws_array, pars = wu_params), width = 12, height = 8
    )
  }
}
