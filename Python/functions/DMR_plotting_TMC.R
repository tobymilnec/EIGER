library(grid)
library(gtable)


plot_region <- function(df, labels) {
  # -- 1) subset & covariance
  region <- as.matrix(df)[labels, , drop = FALSE]
  S      <- cov(t(region))
  n      <- nrow(S)
  
  # -- 2) mask *upper* triangle + diagonal (keep strict lower),
  #        then flip vertically just like np.flipud
  S_masked    <- S
  S_masked[upper.tri(S_masked, diag = TRUE)] <- NA
  S_flip      <- S_masked[n:1, , drop = FALSE]
  
  # -- 3) rebuild the Python mesh exactly:
  rows   <- n:0
  cols   <- 0:n
  # Python: for r in rows, for c in cols: coords.append((c, r))
  pts <- do.call(rbind, lapply(rows, function(r) {
    data.frame(col = cols, row = r)
  }))
  t_mat <- matrix(c(
    1,   0.5,   # [ [1,  0.5],
    -1,   0.5    #   [-1, 0.5] ]
  ), nrow = 2, byrow = TRUE)
  
  xy <- as.matrix(pts) %*% t_mat
  # According to Python pcolormesh:
  #   X_mesh = coordinate_matrix[:,1]  = xy[,2]
  #   Y_mesh = coordinate_matrix[:,0]  = xy[,1]
  X_raw <- xy[, 2]
  Y_raw <- xy[, 1]
  
  # reshape into (n+1)x(n+1) so [i, j] matches Python’s grid
  Xgrid <- matrix(X_raw, nrow = n+1, byrow = TRUE)
  Ygrid <- matrix(Y_raw, nrow = n+1, byrow = TRUE)
  
  # -- 4) rescale Xgrid from [0, n] → [1, n] so it lines up with CpG ticks
  Xgrid <- (Xgrid - min(Xgrid)) / (max(Xgrid) - min(Xgrid)) * (n - 1) + 1
  
  # -- 5) extract each valid cell from the LOWER‐triangle of S_flip
  #    (that’s what Python’s flip + mask shows) as a 4-corner diamond:
  poly_list <- vector("list", n * (n-1) / 2)
  k <- 1
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      val <- S_flip[i, j]
      if (!is.na(val)) {
        xs <- c(
          Xgrid[i,   j],
          Xgrid[i,   j+1],
          Xgrid[i+1, j+1],
          Xgrid[i+1, j]
        )
        ys <- c(
          Ygrid[i,   j],
          Ygrid[i,   j+1],
          Ygrid[i+1, j+1],
          Ygrid[i+1, j]
        )
        poly_list[[k]] <- data.frame(
          x    = xs,
          y    = ys,
          cov  = val,
          cell = k
        )
        k <- k + 1
      }
    }
  }
  heat_df <- bind_rows(poly_list)
  
  # -- 6) top panel: time-series of each CpG
  line_df <- as.data.frame(region) %>%
    mutate(pos = seq_along(icgc_region$start)) %>%
    pivot_longer(-pos, names_to = "sample", values_to = "value") %>% 
    mutate(group = sub(".*\\_", "", sample))
  
  p1 <- ggplot(line_df, aes(x = pos, y = value, group = sample)) +
    geom_rect(aes(xmin = 13, xmax = 17, ymin = -Inf, ymax = Inf), fill="lightblue", alpha = 0.1) +
    geom_line(alpha = 0.7) +
    scale_x_continuous(
      breaks = 1:n,
      labels = icgc_region$start,
      limits = c(1, n),
      expand = c(0, 0)
    ) +
    theme_minimal(base_size = 9) +
    theme(
      axis.title.x    = element_blank(),
      axis.text.x     = element_text(angle = 90, vjust = 0.5, size = 6),
      axis.ticks.x    = element_blank(),
      legend.position = "none"
    ) +
    geom_vline(xintercept = 13, linetype = "dashed", color = "red") +
    geom_vline(xintercept = 17, linetype = "dashed", color = "red")
  
  
  # -- 7) bottom panel: the rotated, tip-down triangle
  p2 <- ggplot(heat_df, aes(x = x, y = y, group = cell, fill = cov)) +
    geom_polygon(color = NA) +
    scale_fill_viridis_c(na.value = NA) +
    scale_x_continuous(
      breaks = 1:n,
      labels = icgc_region$start,
      limits = c(1, n),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = range(heat_df$y),
      expand = c(0, 0)
    ) +
    coord_fixed() +
    theme_void()
  
  # 2) Turn them into gtables:
  g1 <- ggplotGrob(p1)
  g2 <- ggplotGrob(p2)
  
  # 3) Locate the panel columns in each gtable:
  panel_col1 <- g1$layout[g1$layout$name == "panel", "l"]
  panel_col2 <- g2$layout[g2$layout$name == "panel", "l"]
  
  # 4) Pull out the width spec for p1’s panel, and assign it to p2:
  g2$widths[panel_col2] <- g1$widths[panel_col1]
  
  # 5) Now draw them, stacking on top of each other:
  grid.newpage()
  grid.draw(rbind(g1, g2, size = "first"))
}






