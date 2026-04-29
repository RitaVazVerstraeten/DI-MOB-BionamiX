library(sf)
library(spdep)
library(dplyr)

shp_path  <- "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/CMF/Poligonos CMF Cienfuegos_28032025.shp"
out_dir   <- "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP1_Cartographic_data/Administrative borders/Neighbours_tests"
snap_vals <- c(30, 50, 100, 150, 200, 250)

sf_mz <- st_read(shp_path, quiet = TRUE) |> st_make_valid()

# Dissolve blocks into CMF polygons; AS_CMF is the unique identifier
sf_cmf <- sf_mz |>
  filter(!is.na(CMF), as.character(CMF) != "NA") |>
  mutate(AS_CMF = paste(AS, CMF, sep = "_")) |>
  group_by(AS_CMF, AS, CMF) |>
  summarise(geometry = st_union(geometry), .groups = "drop") |>
  st_make_valid() |>
  arrange(AS_CMF)

cat(sprintf("CMF polygons (all): %d\n", nrow(sf_cmf)))

# Restrict to CMFs that intersect the Cienfuegos municipality boundary
muni_path <- paste0(
  "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/",
  "WP1_Cartographic_data/Administrative borders/",
  "Municipality/MUNICIPIO CFGOS 3022025.shp"
)
sf_muni <- st_read(muni_path, quiet = TRUE) |> st_make_valid()
if (st_crs(sf_muni) != st_crs(sf_cmf))
  sf_muni <- st_transform(sf_muni, st_crs(sf_cmf))

sf_cmf <- sf_cmf |> st_filter(sf_muni, .predicate = st_intersects)
cat(sprintf("CMF polygons (within municipality): %d\n", nrow(sf_cmf)))

# poly2nb requires a projected CRS
sf_proj <- if (st_is_longlat(sf_cmf)) st_transform(sf_cmf, 3857) else sf_cmf
coords  <- st_coordinates(suppressWarnings(st_centroid(sf_proj)))

for (snap_m in snap_vals) {
  nb        <- suppressWarnings(spdep::poly2nb(sf_proj, snap = snap_m))
  n_islands <- sum(sapply(nb, function(x) length(x) == 1L && x == 0L))
  cat(sprintf("snap=%dm: %d CMFs, %d islands\n", snap_m, nrow(sf_cmf), n_islands))

  # CMF polygons with neighbour counts
  sf_out <- sf_cmf |>
    mutate(
      n_neighbours = sapply(nb, function(x) if (length(x) == 1L && x == 0L) 0L else length(x)),
      is_island    = as.integer(sapply(nb, function(x) length(x) == 1L && x == 0L))
    )
  st_write(sf_out,
           file.path(out_dir, sprintf("CMF_neighbours_snap%d.gpkg", snap_m)),
           delete_dsn = TRUE, quiet = TRUE)

  # Neighbour lines: shortest path between polygon borders (unique undirected edges)
  edges <- do.call(rbind, lapply(seq_along(nb), function(i) {
    nbrs <- nb[[i]]
    if (length(nbrs) == 1L && nbrs == 0L) return(NULL)
    data.frame(i = i, j = nbrs)
  }))
  edges <- edges[edges$i < edges$j, ]

  lines_list <- lapply(seq_len(nrow(edges)), function(k) {
    st_linestring(matrix(c(coords[edges$i[k], ],
                           coords[edges$j[k], ]), nrow = 2, byrow = TRUE))
  })

  lines_sf <- st_sf(
    i    = edges$i,
    j    = edges$j,
    i_ID = sf_cmf$AS_CMF[edges$i],
    j_ID = sf_cmf$AS_CMF[edges$j],
    wt   = 1L,
    geometry = st_sfc(lines_list, crs = st_crs(sf_proj))
  )
  if (st_is_longlat(sf_cmf)) lines_sf <- st_transform(lines_sf, st_crs(sf_cmf))

  st_write(lines_sf,
           file.path(out_dir, sprintf("CMF_neighbour_lines_snap%d.gpkg", snap_m)),
           delete_dsn = TRUE, quiet = TRUE)
}

cat("Done. Files written to:", out_dir, "\n")
