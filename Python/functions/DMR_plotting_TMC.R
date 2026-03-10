library(grid)
library(gtable)
library(dplyr)
library(tidyr)
library(ggplot2)

plot_region <- function(df, labels, cpg_coords,
                        highlight_start = NULL, highlight_end = NULL) {
  # cpg_coords: data.frame with at least a 'start' column, one row per CpG in df
  # highlight_start / highlight_end: optional positional indices (1-based) to
  # shade and mark with dashed lines in the top panel.

  # -- 1) subset & covariance
  region <- as.matrix(df)[labels, , drop = FALSE]
  S <- cov(t(region))
  n <- nrow(S)

  # -- 2) mask upper triangle + diagonal (keep strict lower),
  #        then flip vertically like np.flipud
  s_masked <- S
  s_masked[upper.tri(s_masked, diag = TRUE)] <- NA
  s_flip <- s_masked[n:1, , drop = FALSE]

  # -- 3) rebuild the Python pcolormesh coordinate grid
  rows <- n:0
  cols <- 0:n
  pts <- do.call(rbind, lapply(rows, function(r) {
    data.frame(col = cols, row = r)
  }))
  t_mat <- matrix(c(
    1,  0.5,
    -1, 0.5
  ), nrow = 2, byrow = TRUE)

  xy <- as.matrix(pts) %*% t_mat
  x_raw <- xy[, 2]
  y_raw <- xy[, 1]

  # reshape into (n+1) x (n+1) so [i, j] matches Python's grid
  x_grid <- matrix(x_raw, nrow = n + 1, byrow = TRUE)
  y_grid <- matrix(y_raw, nrow = n + 1, byrow = TRUE)

  # -- 4) rescale x_grid from [0, n] to [1, n] so it lines up with CpG ticks
  x_grid <- (x_grid - min(x_grid)) / (max(x_grid) - min(x_grid)) * (n - 1) + 1

  # -- 5) extract each valid cell from the lower triangle of s_flip as a
  #        4-corner diamond polygon
  poly_list <- vector("list", n * (n - 1) / 2)
  k <- 1
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      val <- s_flip[i, j]
      if (!is.na(val)) {
        xs <- c(
          x_grid[i,     j],
          x_grid[i,     j + 1],
          x_grid[i + 1, j + 1],
          x_grid[i + 1, j]
        )
        ys <- c(
          y_grid[i,     j],
          y_grid[i,     j + 1],
          y_grid[i + 1, j + 1],
          y_grid[i + 1, j]
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
  line_df <- as.data.frame(region) |>
    mutate(pos = seq_len(n)) |>
    pivot_longer(-pos, names_to = "sample", values_to = "value") |>
    mutate(group = sub(".*\\_", "", sample))

  p1 <- ggplot(line_df, aes(x = pos, y = value, group = sample)) +
    geom_line(alpha = 0.7) +
    scale_x_continuous(
      breaks = seq_len(n),
      labels = cpg_coords$start,
      limits = c(1, n),
      expand = c(0, 0)
    ) +
    theme_minimal(base_size = 9) +
    theme(
      axis.title.x    = element_blank(),
      axis.text.x     = element_text(angle = 90, vjust = 0.5, size = 6),
      axis.ticks.x    = element_blank(),
      legend.position = "none"
    )

  if (!is.null(highlight_start) && !is.null(highlight_end)) {
    p1 <- p1 +
      geom_rect(
        aes(
          xmin = highlight_start, xmax = highlight_end,
          ymin = -Inf, ymax = Inf
        ),
        fill = "lightblue", alpha = 0.1
      ) +
      geom_vline(xintercept = highlight_start, linetype = "dashed", color = "red") +
      geom_vline(xintercept = highlight_end, linetype = "dashed", color = "red")
  }

  # -- 7) bottom panel: the rotated, tip-down covariance triangle
  p2 <- ggplot(heat_df, aes(x = x, y = y, group = cell, fill = cov)) +
    geom_polygon(color = NA) +
    scale_fill_viridis_c(na.value = NA) +
    scale_x_continuous(
      breaks = seq_len(n),
      labels = cpg_coords$start,
      limits = c(1, n),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = range(heat_df$y),
      expand = c(0, 0)
    ) +
    coord_fixed() +
    theme_void()

  # Align panel widths between the two plots and draw stacked
  g1 <- ggplotGrob(p1)
  g2 <- ggplotGrob(p2)

  panel_col1 <- g1$layout[g1$layout$name == "panel", "l"]
  panel_col2 <- g2$layout[g2$layout$name == "panel", "l"]
  g2$widths[panel_col2] <- g1$widths[panel_col1]

  grid.newpage()
  grid.draw(rbind(g1, g2, size = "first"))
}
