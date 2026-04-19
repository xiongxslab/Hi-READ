#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(stringr)
  library(grid)
})

# Rebuild bubble plots for the 4 selected example diseases:
# - TF-set stacked panels (2x1 and 3x1), each with:
#   (a) all significant terms
#   (b) shared terms + top-N specific terms per (TF, disease)
# - Disease-specific bubble plots (one disease per figure)
# - Per-TF bubble plots (one TF per figure, example diseases only)

parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (startsWith(a, "--")) {
      key <- substring(a, 3)
      if (i == length(args) || startsWith(args[[i + 1]], "--")) {
        out[[key]] <- TRUE
        i <- i + 1
      } else {
        out[[key]] <- args[[i + 1]]
        i <- i + 2
      }
    } else {
      i <- i + 1
    }
  }
  out
}

safe_slug <- function(x) {
  y <- gsub("[^A-Za-z0-9_]+", "_", x)
  y <- gsub("_+", "_", y)
  y <- gsub("^_+|_+$", "", y)
  ifelse(nchar(y) == 0, "NA", y)
}

tf_levels_for_set <- function(tf_set) {
  if (identical(tf_set, "CTCF_Rad21")) return(c("CTCF", "Rad21"))
  if (identical(tf_set, "KLF4_OCT4_NANOG")) return(c("KLF4", "OCT4", "NANOG"))
  stop(paste("Unsupported tf_set:", tf_set))
}

all_sig_csv_for_set <- function(shared_dir, tf_set) {
  if (identical(tf_set, "CTCF_Rad21")) {
    return(file.path(shared_dir, "direct_brain_shared_sig_CTCF_Rad21_all_sig_terms_data.csv"))
  }
  if (identical(tf_set, "KLF4_OCT4_NANOG")) {
    return(file.path(shared_dir, "direct_brain_shared_sig_KLF4_OCT4_NANOG_all_sig_terms_data.csv"))
  }
  stop(paste("Unsupported tf_set:", tf_set))
}

read_all_sig <- function(shared_dir, tf_set) {
  p <- all_sig_csv_for_set(shared_dir, tf_set)
  if (!file.exists(p)) stop(paste("Missing input:", p))
  d <- fread(p, data.table = FALSE)
  d$p.adjust <- as.numeric(d$p.adjust)
  d$Count <- as.numeric(d$Count)
  if (!("is_shared" %in% colnames(d))) d$is_shared <- FALSE
  d$is_shared <- as.logical(d$is_shared)
  d$is_shared[is.na(d$is_shared)] <- FALSE
  d
}

build_variant <- function(d, mode = c("all_sig", "shared_plus_topn"), topn_specific = 3L) {
  mode <- match.arg(mode)
  if (mode == "all_sig") {
    out <- d
  } else {
    out <- d %>%
      group_by(.data$group_label, .data$tf) %>%
      group_modify(~{
        g <- .x %>% arrange(.data$p.adjust, desc(.data$Count), .data$ID)
        g_shared <- g %>% filter(.data$is_shared)
        g_spec <- g %>% filter(!.data$is_shared) %>% slice_head(n = as.integer(topn_specific))
        bind_rows(g_shared, g_spec)
      }) %>%
      ungroup()
  }

  out %>%
    mutate(
      term_id = as.character(.data$ID),
      term_desc = as.character(.data$Description),
      term_label = ifelse(
        .data$is_shared,
        paste0("* ", str_trunc(.data$term_desc, width = 96, side = "right")),
        str_trunc(.data$term_desc, width = 98, side = "right")
      ),
      log_p = -log10(.data$p.adjust)
    )
}

term_order_by_disease <- function(d, disease_levels) {
  if (nrow(d) == 0) return(character(0))
  order_tbl <- d %>%
    group_by(.data$group_label, .data$term_id) %>%
    summarise(
      best_padj = min(.data$p.adjust, na.rm = TRUE),
      best_count = max(.data$Count, na.rm = TRUE),
      is_shared_any = any(.data$is_shared),
      .groups = "drop"
    )

  row_levels <- character(0)
  seen <- new.env(parent = emptyenv())
  for (dis in disease_levels) {
    sub <- order_tbl %>%
      filter(.data$group_label == .env$dis) %>%
      arrange(desc(.data$is_shared_any), .data$best_padj, desc(.data$best_count), .data$term_id)
    ids <- unique(as.character(sub$term_id))
    for (tid in ids) {
      if (!exists(tid, envir = seen, inherits = FALSE)) {
        row_levels <- c(row_levels, tid)
        assign(tid, TRUE, envir = seen)
      }
    }
  }

  remain <- setdiff(unique(as.character(order_tbl$term_id)), row_levels)
  if (length(remain) > 0) {
    tail_ids <- order_tbl %>%
      filter(.data$term_id %in% .env$remain) %>%
      group_by(.data$term_id) %>%
      summarise(best_padj = min(.data$best_padj, na.rm = TRUE), .groups = "drop") %>%
      arrange(.data$best_padj, .data$term_id) %>%
      pull(.data$term_id)
    row_levels <- c(row_levels, as.character(tail_ids))
  }
  row_levels
}

extract_legend_grob <- function(p) {
  g <- ggplotGrob(p)
  idx <- which(sapply(g$grobs, function(x) x$name) == "guide-box")
  if (length(idx) == 0) return(NULL)
  g$grobs[[idx[1]]]
}

plot_tf_bubble <- function(d_tf,
                           tf_name,
                           disease_levels,
                           row_levels,
                           show_x_text = TRUE,
                           show_legend = TRUE,
                           base_size = 6.4,
                           min_logp,
                           max_logp,
                           max_count,
                           size_breaks) {
  if (nrow(d_tf) == 0 || length(row_levels) == 0) {
    return(ggplot() + theme_void() + ggtitle(tf_name))
  }

  label_map <- d_tf %>%
    select(.data$term_id, .data$term_label) %>%
    distinct() %>%
    { setNames(.$term_label, .$term_id) }

  d_plot <- d_tf %>%
    mutate(
      disease = factor(.data$group_label, levels = disease_levels),
      term = factor(.data$term_id, levels = row_levels)
    )

  p <- ggplot(d_plot, aes(x = .data$disease, y = .data$term)) +
    geom_point(aes(size = .data$Count, color = .data$log_p), alpha = 0.86) +
    scale_color_viridis_c(option = "plasma", limits = c(min_logp, max_logp), oob = scales::squish, name = "-log10(p.adjust)") +
    scale_size_continuous(range = c(0.8, 4.0), limits = c(1, max_count), breaks = size_breaks, name = "Count") +
    scale_x_discrete(drop = FALSE) +
    scale_y_discrete(labels = unname(label_map[row_levels]), drop = FALSE) +
    labs(title = tf_name, x = NULL, y = NULL) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(size = max(6.6, base_size + 0.6), face = "bold", hjust = 0.5),
      axis.text.x = if (show_x_text) element_text(angle = 55, hjust = 1, vjust = 1, size = 6.0, color = "black") else element_blank(),
      axis.ticks.x = if (show_x_text) element_line(color = "black", linewidth = 0.25) else element_blank(),
      axis.text.y = element_text(size = 6.0, color = "black"),
      panel.grid.major.x = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      legend.position = if (show_legend) "right" else "none",
      legend.title = element_text(size = 6.4),
      legend.text = element_text(size = 6.0),
      plot.margin = margin(2, 6, ifelse(show_x_text, 18, 2), 4)
    )
  p
}

plot_disease_bubble <- function(d_dis,
                                disease_name,
                                tf_levels,
                                base_size = 6.4,
                                min_logp,
                                max_logp,
                                max_count,
                                size_breaks) {
  if (nrow(d_dis) == 0) {
    return(ggplot() + theme_void() + ggtitle(disease_name))
  }

  ord <- d_dis %>%
    group_by(.data$term_id) %>%
    summarise(
      best_padj = min(.data$p.adjust, na.rm = TRUE),
      best_count = max(.data$Count, na.rm = TRUE),
      is_shared_any = any(.data$is_shared),
      term_label = dplyr::first(.data$term_label),
      .groups = "drop"
    ) %>%
    arrange(desc(.data$is_shared_any), .data$best_padj, desc(.data$best_count), .data$term_id)

  row_levels <- as.character(ord$term_id)
  label_map <- setNames(as.character(ord$term_label), as.character(ord$term_id))

  d_plot <- d_dis %>%
    mutate(
      tf = factor(.data$tf, levels = tf_levels),
      term = factor(.data$term_id, levels = row_levels)
    )

  ggplot(d_plot, aes(x = .data$tf, y = .data$term)) +
    geom_point(aes(size = .data$Count, color = .data$log_p), alpha = 0.86) +
    scale_color_viridis_c(option = "plasma", limits = c(min_logp, max_logp), oob = scales::squish, name = "-log10(p.adjust)") +
    scale_size_continuous(range = c(0.8, 4.0), limits = c(1, max_count), breaks = size_breaks, name = "Count") +
    scale_x_discrete(drop = FALSE) +
    scale_y_discrete(labels = unname(label_map[row_levels]), drop = FALSE) +
    labs(title = disease_name, x = NULL, y = NULL) +
    theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(size = max(6.6, base_size + 0.6), face = "bold", hjust = 0.5),
      axis.text.x = element_text(size = 6.2, color = "black"),
      axis.text.y = element_text(size = 6.0, color = "black"),
      panel.grid.major.x = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = 6.4),
      legend.text = element_text(size = 6.0),
      plot.margin = margin(2, 6, 4, 4)
    )
}

save_plot_pdf_png <- function(p, out_pdf, width, height) {
  pdf(out_pdf, width = width, height = height, useDingbats = FALSE)
  print(p)
  dev.off()
  out_png <- sub("\\.pdf$", "_01.png", out_pdf)
  ggsave(out_png, p, width = width, height = height, dpi = 320, bg = "white")
}

save_vertical_panel <- function(plots, legend_grob, out_pdf, width, height) {
  grobs <- lapply(plots, ggplotGrob)
  if (length(grobs) > 1) {
    w_shared <- Reduce(unit.pmax, lapply(grobs, function(g) g$widths))
    grobs <- lapply(grobs, function(g) { g$widths <- w_shared; g })
  }

  n <- length(grobs)
  pdf(out_pdf, width = width, height = height, useDingbats = FALSE)
  grid.newpage()
  if (is.null(legend_grob)) {
    pushViewport(viewport(layout = grid.layout(n, 1)))
    for (i in seq_len(n)) {
      pushViewport(viewport(layout.pos.row = i, layout.pos.col = 1))
      grid.draw(grobs[[i]])
      upViewport()
    }
    dev.off()
  } else {
    pushViewport(viewport(layout = grid.layout(n, 2, widths = unit.c(unit(1, "null"), unit(0.24, "null")))))
    for (i in seq_len(n)) {
      pushViewport(viewport(layout.pos.row = i, layout.pos.col = 1))
      grid.draw(grobs[[i]])
      upViewport()
    }
    pushViewport(viewport(layout.pos.row = 1:n, layout.pos.col = 2))
    grid.draw(legend_grob)
    upViewport()
    dev.off()
  }

}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (is.null(args$run_root)) {
    cat("Usage: Rscript make_examples_shared_go_bubbleplots.R --run_root <.../20260209_hbi_direct_gemini2_5> [--spec_csv <...>] [--out_dir <...>] [--topn_specific 3] [--min_padj 0.05]\n")
    quit(status = 1)
  }

  run_root <- args$run_root
  shared_dir <- file.path(run_root, "analysis_min1", "correspondence_high_conf_min1", "shared_sig_panels")
  spec_csv <- ifelse(is.null(args$spec_csv), file.path(shared_dir, "NATURE_sharedGOID_examples_panels_spec.csv"), args$spec_csv)
  out_dir <- ifelse(is.null(args$out_dir), file.path(shared_dir, "bubbleplots_examples_refresh"), args$out_dir)
  topn_specific <- ifelse(is.null(args$topn_specific), 3L, as.integer(args$topn_specific))
  min_padj <- ifelse(is.null(args$min_padj), 0.05, as.numeric(args$min_padj))
  panel_width <- ifelse(is.null(args$panel_width), 7.2, as.numeric(args$panel_width))
  single_width <- ifelse(is.null(args$single_width), 6.2, as.numeric(args$single_width))
  base_size <- ifelse(is.null(args$base_size), 6.4, as.numeric(args$base_size))
  panel_height_scale <- ifelse(is.null(args$panel_height_scale), 1.0, as.numeric(args$panel_height_scale))
  single_height_scale <- ifelse(is.null(args$single_height_scale), 1.0, as.numeric(args$single_height_scale))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  if (!file.exists(spec_csv)) stop(paste("Missing spec_csv:", spec_csv))
  spec <- fread(spec_csv, data.table = FALSE)
  if (!all(c("panel", "tf_set", "disease") %in% colnames(spec))) {
    stop("spec_csv must include panel, tf_set, disease columns")
  }
  spec <- spec %>% arrange(.data$panel)

  manifest <- list()
  tf_sets <- unique(as.character(spec$tf_set))

  for (tf_set in tf_sets) {
    tfs <- tf_levels_for_set(tf_set)
    dis_levels <- spec %>% filter(.data$tf_set == .env$tf_set) %>% arrange(.data$panel) %>% pull(.data$disease) %>% as.character()

    d_all <- read_all_sig(shared_dir, tf_set) %>%
      filter(.data$tf %in% tfs, .data$group_label %in% dis_levels, !is.na(.data$p.adjust), .data$p.adjust < min_padj)

    if (nrow(d_all) == 0) next

    variants <- list(
      list(mode = "all_sig", tag = "all_sig_terms"),
      list(mode = "shared_plus_topn", tag = paste0("shared_terms_top", topn_specific))
    )

    for (v in variants) {
      d_v <- build_variant(d_all, mode = v$mode, topn_specific = topn_specific)
      if (nrow(d_v) == 0) next

      min_logp <- -log10(min_padj)
      max_logp <- max(d_v$log_p, na.rm = TRUE)
      max_count <- max(d_v$Count, na.rm = TRUE)
      bb <- unique(pretty(c(1, max_count), n = 4))
      size_breaks <- bb[bb >= 1 & bb <= max_count]

      # 1) TF-set stacked bubble panel (2x1 / 3x1)
      tf_row_levels <- list()
      plots <- list()
      for (i in seq_along(tfs)) {
        tf <- tfs[[i]]
        d_tf <- d_v %>% filter(.data$tf == .env$tf)
        tf_row_levels[[tf]] <- term_order_by_disease(d_tf, dis_levels)
        plots[[i]] <- plot_tf_bubble(
          d_tf = d_tf,
          tf_name = tf,
          disease_levels = dis_levels,
          row_levels = tf_row_levels[[tf]],
          show_x_text = (i == length(tfs)),
          show_legend = FALSE,
          base_size = base_size,
          min_logp = min_logp,
          max_logp = max_logp,
          max_count = max_count,
          size_breaks = size_breaks
        )
      }

      p_legend <- plot_tf_bubble(
        d_tf = d_v %>% filter(.data$tf == tfs[[1]]),
        tf_name = tfs[[1]],
        disease_levels = dis_levels,
        row_levels = tf_row_levels[[tfs[[1]]]],
        show_x_text = FALSE,
        show_legend = TRUE,
        base_size = base_size,
        min_logp = min_logp,
        max_logp = max_logp,
        max_count = max_count,
        size_breaks = size_breaks
      )
      legend_grob <- extract_legend_grob(p_legend)

      max_rows <- max(unlist(lapply(tf_row_levels, length)))
      h_set <- max(4.2, min(13.0, 1.6 * length(tfs) + 0.075 * max_rows))
      h_set <- max(3.8, h_set * panel_height_scale)
      out_set <- file.path(out_dir, paste0("examples_", tf_set, "_bubble_", length(tfs), "x1_", v$tag, ".pdf"))
      save_vertical_panel(plots, legend_grob, out_set, width = panel_width, height = h_set)
      out_set_csv <- sub("\\.pdf$", "_data.csv", out_set)
      fwrite(d_v, out_set_csv)

      manifest[[length(manifest) + 1]] <- data.frame(
        type = "tf_set_panel",
        tf_set = tf_set,
        variant = v$tag,
        disease = NA_character_,
        tf = NA_character_,
        file = out_set,
        stringsAsFactors = FALSE
      )

      # 2) Disease-specific bubble plots
      for (dis in dis_levels) {
        d_dis <- d_v %>% filter(.data$group_label == .env$dis)
        p_dis <- plot_disease_bubble(
          d_dis = d_dis,
          disease_name = dis,
          tf_levels = tfs,
          base_size = base_size,
          min_logp = min_logp,
          max_logp = max_logp,
          max_count = max_count,
          size_breaks = size_breaks
        )
        out_dis <- file.path(out_dir, paste0("example_disease_", safe_slug(dis), "_", tf_set, "_", v$tag, "_bubble_1x1.pdf"))
        h_dis <- max(3.1, min(8.2, 1.9 + 0.085 * length(unique(d_dis$term_id))))
        h_dis <- max(2.8, h_dis * single_height_scale)
        save_plot_pdf_png(p_dis, out_dis, width = single_width, height = h_dis)

        manifest[[length(manifest) + 1]] <- data.frame(
          type = "disease_single",
          tf_set = tf_set,
          variant = v$tag,
          disease = dis,
          tf = NA_character_,
          file = out_dis,
          stringsAsFactors = FALSE
        )
      }
    }

    # 3) One bubble plot per TF (all significant terms)
    d_tf_all <- build_variant(d_all, mode = "all_sig", topn_specific = topn_specific)
    min_logp_tf <- -log10(min_padj)
    max_logp_tf <- max(d_tf_all$log_p, na.rm = TRUE)
    max_count_tf <- max(d_tf_all$Count, na.rm = TRUE)
    bb_tf <- unique(pretty(c(1, max_count_tf), n = 4))
    size_breaks_tf <- bb_tf[bb_tf >= 1 & bb_tf <= max_count_tf]

    for (tf in tfs) {
      sub_tf <- d_tf_all %>% filter(.data$tf == .env$tf)
      row_levels <- term_order_by_disease(sub_tf, dis_levels)
      p_tf <- plot_tf_bubble(
        d_tf = sub_tf,
        tf_name = tf,
        disease_levels = dis_levels,
        row_levels = row_levels,
        show_x_text = TRUE,
        show_legend = TRUE,
        base_size = base_size,
        min_logp = min_logp_tf,
        max_logp = max_logp_tf,
        max_count = max_count_tf,
        size_breaks = size_breaks_tf
      )
      out_tf <- file.path(out_dir, paste0("example_tf_", tf, "_", tf_set, "_all_sig_terms_bubble_1x1.pdf"))
      h_tf <- max(3.3, min(9.2, 2.2 + 0.085 * length(row_levels)))
      h_tf <- max(3.0, h_tf * single_height_scale)
      save_plot_pdf_png(p_tf, out_tf, width = single_width, height = h_tf)

      manifest[[length(manifest) + 1]] <- data.frame(
        type = "tf_single",
        tf_set = tf_set,
        variant = "all_sig_terms",
        disease = NA_character_,
        tf = tf,
        file = out_tf,
        stringsAsFactors = FALSE
      )
    }
  }

  m <- bind_rows(manifest)
  out_manifest <- file.path(out_dir, "examples_shared_go_bubbleplots_manifest.csv")
  fwrite(m, out_manifest)
  cat(out_manifest, "\n")
}

main()
