#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggrepel)
  library(grid)
  library(png)
})

BIN_SIZE <- 8192
NUM_BINS <- 256
DIAGONAL_EXCLUSION <- 2

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

split_genes <- function(x) {
  if (is.null(x) || is.na(x) || !nzchar(x)) return(character(0))
  vals <- unlist(strsplit(x, "/"))
  vals <- trimws(vals)
  unique(vals[nzchar(vals)])
}

split_tokens <- function(x) {
  if (is.null(x) || is.na(x) || !nzchar(x)) return(character(0))
  vals <- unlist(strsplit(x, "[,;]"))
  vals <- trimws(vals)
  unique(vals[nzchar(vals)])
}

parse_bool_field <- function(x) {
  vals <- tolower(trimws(as.character(x)))
  vals %in% c("true", "t", "1", "yes")
}

safe_name <- function(x) {
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("(^_+|_+$)", "", x)
  x
}

read_npy_matrix <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  magic <- readBin(con, "raw", n = 6)
  if (!identical(as.raw(c(0x93, charToRaw("NUMPY"))), magic)) {
    stop("Not a valid NPY file: ", path)
  }
  ver <- readBin(con, "integer", n = 2, size = 1, signed = FALSE)
  header_len <- if (ver[1] == 1L) {
    readBin(con, "integer", n = 1, size = 2, endian = "little", signed = FALSE)
  } else {
    readBin(con, "integer", n = 1, size = 4, endian = "little", signed = FALSE)
  }
  header <- rawToChar(readBin(con, "raw", n = header_len))
  descr <- sub(".*'descr': '([^']+)'.*", "\\1", header)
  fortran_order <- grepl("'fortran_order': True", header, fixed = TRUE)
  shape_txt <- sub(".*'shape': \\(([^\\)]*)\\).*", "\\1", header)
  dims <- as.integer(trimws(unlist(strsplit(gsub("L", "", shape_txt), ","))))
  dims <- dims[!is.na(dims)]
  if (length(dims) != 2) {
    stop("Only 2D matrices are supported: ", path)
  }
  size <- if (descr == "<f4") 4L else if (descr == "<f8") 8L else stop("Unsupported dtype: ", descr)
  vals <- readBin(con, "numeric", n = prod(dims), size = size, endian = "little")
  matrix(vals, nrow = dims[1], ncol = dims[2], byrow = !fortran_order)
}

compute_column_delta <- function(mat) {
  idx <- seq_len(nrow(mat))
  mask <- abs(outer(idx, idx, "-")) >= DIAGONAL_EXCLUSION
  colSums(abs(mat) * mask)
}

load_example_specs <- function(example_csv, gene_rank_csv) {
  ex <- fread(example_csv)
  gene_rank <- fread(gene_rank_csv)
  gene_rank[, in_shared_intersection_flag := parse_bool_field(in_shared_intersection)]
  specs <- list()
  for (panel_id in unique(ex$panel)) {
    sub <- ex[panel == panel_id]
    disease_name <- unique(sub$group_label)[1]
    bucket <- unique(sub$bucket)[1]
    tf_set <- unique(sub$tf_set)[1]
    tfs <- sort(unique(sub$tf))
    union_genes <- sort(unique(unlist(lapply(sub$geneID, split_genes))))
    shared_genes <- sort(unique(gene_rank[disease == disease_name & in_shared_intersection_flag == TRUE, gene]))
    specs[[length(specs) + 1]] <- list(
      panel = panel_id,
      disease = disease_name,
      bucket = bucket,
      tf_set = tf_set,
      tfs = tfs,
      union_genes = union_genes,
      shared_genes = shared_genes
    )
  }
  specs
}

load_tf_scores <- function(tf_name, sv_root) {
  path <- file.path(sv_root, paste0("results_", tf_name), "sv_gene_scores.csv")
  if (!file.exists(path)) stop("Missing score file: ", path)
  dt <- fread(path)
  dt[, tf := tf_name]
  dt
}

select_best_locus <- function(spec, sv_root) {
  rows_list <- lapply(spec$tfs, load_tf_scores, sv_root = sv_root)
  dt <- rbindlist(rows_list, use.names = TRUE, fill = TRUE)
  dt <- dt[gene_name %in% spec$union_genes]
  if (nrow(dt) == 0) {
    stop("No sv_gene_scores rows found for example disease: ", spec$disease)
  }
  dt[, abs_delta := abs(as.numeric(column_delta_score))]
  dt <- dt[mutation_type == "deletion"]
  dt[, in_shared := gene_name %in% spec$shared_genes]

  locus <- dt[, .(
    n_genes = uniqueN(gene_name),
    genes_in_locus = paste(sort(unique(gene_name)), collapse = ";"),
    n_shared = uniqueN(gene_name[in_shared == TRUE]),
    shared_genes_in_locus = paste(sort(unique(gene_name[in_shared == TRUE])), collapse = ";"),
    total_delta = sum(abs_delta),
    shared_total_delta = sum(abs_delta[in_shared == TRUE]),
    max_delta = max(abs_delta)
  ), by = .(tf, sv_chrom, sv_window_start, sv_window_end, sv_file, mutation_type)]

  locus_with_shared <- locus[n_shared > 0]
  if (nrow(locus_with_shared) > 0) {
    setorder(locus_with_shared, -n_shared, -shared_total_delta, -total_delta, -n_genes, -max_delta)
    locus_ranked <- locus_with_shared
  } else {
    setorder(locus, -total_delta, -n_genes, -max_delta)
    locus_ranked <- locus
  }

  chosen <- locus_ranked[1]
  chosen_rows <- dt[
    tf == chosen$tf &
      sv_file == chosen$sv_file &
      sv_chrom == chosen$sv_chrom &
      sv_window_start == chosen$sv_window_start &
      sv_window_end == chosen$sv_window_end
  ]
  chosen_rows <- chosen_rows[order(-abs_delta)]
  chosen_rows <- chosen_rows[!duplicated(gene_name)]

  list(rank_table = locus_ranked, chosen = chosen, chosen_rows = chosen_rows)
}

locate_npy <- function(npy_root, tf_name, mutation_type, sv_file) {
  subdir <- if (mutation_type == "deletion") "deletion_rrtdiffusion" else if (mutation_type %in% c("dup", "duplication")) "duplication_rrtdiffusion" else "translocation_rrtdiffusion"
  path <- file.path(npy_root, tf_name, "H9_hESCs", subdir, "npy", sv_file)
  if (!file.exists(path)) stop("Missing npy file: ", path)
  path
}

render_surface_png <- function(mat, output_png, title_text) {
  zlim <- max(abs(range(mat, finite = TRUE)))
  zlim <- c(-zlim, zlim)
  cols <- colorRampPalette(c("#2166ac", "#f7f7f7", "#b2182b"))(128)
  zfacet <- (mat[-1, -1] + mat[-1, -ncol(mat)] + mat[-nrow(mat), -1] + mat[-nrow(mat), -ncol(mat)]) / 4
  facet_idx <- pmax(1, pmin(length(cols), round((zfacet - zlim[1]) / diff(zlim) * (length(cols) - 1)) + 1))

  png(output_png, width = 10, height = 4.8, units = "in", res = 220, type = if (identical(Sys.info()[["sysname"]], "Darwin")) "quartz" else "cairo")
  par(mar = c(2.2, 2.8, 2.4, 0.8))
  persp(
    x = seq_len(nrow(mat)),
    y = seq_len(ncol(mat)),
    z = mat,
    theta = 38,
    phi = 28,
    expand = 0.55,
    col = cols[facet_idx],
    border = NA,
    shade = 0.15,
    ticktype = "detailed",
    xlab = "Bin",
    ylab = "Bin",
    zlab = "Difference",
    main = title_text
  )
  dev.off()
}

render_delta_png <- function(delta_values,
                             chosen_rows,
                             chosen,
                             output_png,
                             title_text) {
  delta_dt <- data.table(
    bin_idx = 0:(length(delta_values) - 1),
    genomic_pos = chosen$sv_window_start + ((0:(length(delta_values) - 1)) + 0.5) * BIN_SIZE,
    delta = as.numeric(delta_values)
  )

  label_rows <- copy(chosen_rows)
  label_rows[, genomic_pos := chosen$sv_window_start + (as.numeric(tss_bin) + 0.5) * BIN_SIZE]
  label_rows[, track_delta := delta_values[as.integer(tss_bin) + 1L]]
  label_rows[, label := paste0(gene_name, " (", sprintf("%.2f", abs_delta), ")")]
  label_rows <- label_rows[order(-abs_delta)]

  p <- ggplot(delta_dt, aes(x = genomic_pos, y = delta)) +
    annotate(
      "rect",
      xmin = unique(chosen_rows$sv_pos)[1],
      xmax = unique(chosen_rows$sv_end)[1],
      ymin = -Inf,
      ymax = Inf,
      alpha = 0.12,
      fill = "#f4a582"
    ) +
    geom_area(fill = "#92c5de", alpha = 0.45) +
    geom_line(color = "#2166ac", linewidth = 0.45) +
    geom_point(data = label_rows, aes(x = genomic_pos, y = track_delta), color = "#b2182b", size = 1.7) +
    ggrepel::geom_label_repel(
      data = label_rows,
      aes(x = genomic_pos, y = track_delta, label = label),
      size = 2.4,
      label.size = 0.15,
      fill = scales::alpha("white", 0.88),
      box.padding = 0.20,
      point.padding = 0.15,
      segment.color = "#666666",
      min.segment.length = 0
    ) +
    labs(title = title_text, x = NULL, y = "Column delta") +
    theme_bw(base_size = 9) +
    theme(
      plot.title = element_text(size = 9, face = "bold"),
      axis.text.x = element_text(size = 7),
      axis.text.y = element_text(size = 7),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.margin = margin(6, 6, 4, 10)
    )

  ggsave(output_png, p, width = 10, height = 2.2, dpi = 220, bg = "white")
}

combine_pngs <- function(surface_png, delta_png, gene_png, out_png, out_pdf) {
  img1 <- readPNG(surface_png)
  img2 <- readPNG(delta_png)
  img3 <- readPNG(gene_png)

  draw_stack <- function() {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(3, 1, heights = unit(c(0.56, 0.20, 0.24), "npc"))))
    print_img <- function(img, row_id) {
      pushViewport(viewport(layout.pos.row = row_id, layout.pos.col = 1))
      grid.raster(img, interpolate = TRUE)
      popViewport()
    }
    print_img(img1, 1)
    print_img(img2, 2)
    print_img(img3, 3)
    popViewport()
  }

  png(out_png, width = 10, height = 9, units = "in", res = 220, type = if (identical(Sys.info()[["sysname"]], "Darwin")) "quartz" else "cairo")
  draw_stack()
  dev.off()

  pdf(out_pdf, width = 10, height = 9, onefile = FALSE, useDingbats = FALSE)
  draw_stack()
  dev.off()
}

write_manifest_md <- function(manifest_dt, out_md) {
  con <- file(out_md, "w")
  on.exit(close(con), add = TRUE)
  writeLines("# Example 3D difference map + delta track + gene track manifest\n", con)
  writeLines("Selection rule: prioritize loci that contain shared-intersection genes; within that subset rank by `n_shared`, `shared_total_delta`, `total_delta`, and `n_genes`.\n", con)
  for (i in seq_len(nrow(manifest_dt))) {
    row <- manifest_dt[i]
    writeLines(paste0("## ", row$panel, " | ", row$disease, "\n"), con)
    writeLines(paste0("- TF set: `", row$tf_set, "` | selected TF: `", row$selected_tf, "`"), con)
    writeLines(paste0("- Locus: `", row$sv_chrom, ":", row$sv_window_start, "-", row$sv_window_end, "`"), con)
    writeLines(paste0("- SV file: `", row$sv_file, "`"), con)
    writeLines(paste0("- Locus genes: `", row$locus_genes, "`"), con)
    writeLines(paste0("- Shared genes in locus: `", row$shared_genes_in_locus, "`"), con)
    writeLines(paste0("- Omitted example genes: `", row$omitted_example_genes, "`"), con)
    writeLines(paste0("- Output PDF: `", row$output_pdf, "`"), con)
    writeLines("", con)
  }
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  run_root <- if (!is.null(args$`run-root`)) {
    args$`run-root`
  } else if (nzchar(Sys.getenv("HIREAD_SV_RUN_ROOT", ""))) {
    Sys.getenv("HIREAD_SV_RUN_ROOT")
  } else {
    "."
  }
  sv_root <- if (!is.null(args$`sv-root`)) {
    args$`sv-root`
  } else if (nzchar(Sys.getenv("HIREAD_SV_SCORE_ROOT", ""))) {
    Sys.getenv("HIREAD_SV_SCORE_ROOT")
  } else {
    file.path(run_root, "sv_perturbation_analysis")
  }
  npy_root <- if (!is.null(args$`npy-root`)) {
    args$`npy-root`
  } else if (nzchar(Sys.getenv("HIREAD_SV_NPY_ROOT", ""))) {
    Sys.getenv("HIREAD_SV_NPY_ROOT")
  } else {
    file.path(run_root, "sv_npy")
  }
  track_script <- if (!is.null(args$`track-script`)) {
    args$`track-script`
  } else if (nzchar(Sys.getenv("HIREAD_SV_TRACK_SCRIPT", ""))) {
    Sys.getenv("HIREAD_SV_TRACK_SCRIPT")
  } else {
    file.path(run_root, "plot_gviz_gene_tracks.R")
  }
  bed_path <- if (!is.null(args$`bed-path`)) {
    args$`bed-path`
  } else if (nzchar(Sys.getenv("HIREAD_SV_BED_PATH", ""))) {
    Sys.getenv("HIREAD_SV_BED_PATH")
  } else {
    file.path(run_root, "hg38.gencode.v48.all_gene.TSS1kb.bed")
  }
  coding_list_path <- if (!is.null(args$`coding-list-path`)) {
    args$`coding-list-path`
  } else if (nzchar(Sys.getenv("HIREAD_SV_CODING_LIST_PATH", ""))) {
    Sys.getenv("HIREAD_SV_CODING_LIST_PATH")
  } else {
    file.path(run_root, "coding_genes_biomart.txt")
  }

  shared_dir <- file.path(run_root, "analysis_min1", "correspondence_high_conf_min1", "shared_sig_panels")
  example_csv <- if (!is.null(args$`example-csv`)) args$`example-csv` else file.path(shared_dir, "NATURE_sharedGOID_examples_4panels_data.csv")
  gene_rank_csv <- if (!is.null(args$`gene-rank-csv`)) args$`gene-rank-csv` else file.path(shared_dir, "gene_lists_sharedGOID_examples", "sharedGOID_examples_gene_delta_ranked.csv")
  output_dir <- if (!is.null(args$`output-dir`)) args$`output-dir` else file.path(shared_dir, "example_3d_delta_gene_tracks")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  specs <- load_example_specs(example_csv, gene_rank_csv)
  manifest_rows <- list()

  for (spec in specs) {
    sel <- select_best_locus(spec, sv_root)
    chosen <- sel$chosen
    chosen_rows <- copy(sel$chosen_rows)
    chosen_rows[, abs_delta := abs(as.numeric(column_delta_score))]
    chosen_rows <- chosen_rows[order(-abs_delta)]

    npy_path <- locate_npy(npy_root, chosen$tf, chosen$mutation_type, chosen$sv_file)
    mat <- read_npy_matrix(npy_path)
    delta_values <- compute_column_delta(mat)

    stem <- paste0(spec$panel, "_", safe_name(spec$disease), "_", chosen$tf)
    surface_png <- file.path(output_dir, paste0(stem, "_surface.png"))
    delta_png <- file.path(output_dir, paste0(stem, "_delta.png"))
    gene_png <- file.path(output_dir, paste0(stem, "_gene_track.png"))
    final_png <- file.path(output_dir, paste0(stem, "_3d_delta_gene_track.png"))
    final_pdf <- file.path(output_dir, paste0(stem, "_3d_delta_gene_track.pdf"))
    candidates_csv <- file.path(output_dir, paste0(stem, "_locus_candidates.csv"))
    genes_csv <- file.path(output_dir, paste0(stem, "_selected_locus_gene_deltas.csv"))

    fwrite(sel$rank_table, candidates_csv)
    fwrite(chosen_rows, genes_csv)

    render_surface_png(
      mat = mat,
      output_png = surface_png,
      title_text = paste0(spec$disease, " | ", chosen$tf, " | 3D difference map")
    )

    render_delta_png(
      delta_values = delta_values,
      chosen_rows = chosen_rows,
      chosen = chosen,
      output_png = delta_png,
      title_text = paste0(spec$disease, " | ", chosen$tf, " | column delta track")
    )

    highlight_genes <- paste(chosen_rows$gene_name, collapse = ",")
    gene_title <- paste(spec$disease, chosen$tf, "gene track", sep = " - ")
    cmd <- c(
      track_script,
      "--region-chrom", as.character(chosen$sv_chrom),
      "--region-start", as.character(chosen$sv_window_start),
      "--region-end", as.character(chosen$sv_window_end),
      "--output-file", gene_png,
      "--bed-path", bed_path,
      "--coding-list-path", coding_list_path,
      "--highlight-genes", highlight_genes,
      "--plot-width-in", "10",
      "--plot-height-in", "2.6",
      "--title", gene_title,
      "--hide-base-labels"
    )
    status <- system2("Rscript", args = cmd)
    if (!identical(status, 0L)) {
      stop("plot_gviz_gene_tracks.R region mode failed for ", spec$disease)
    }

    combine_pngs(surface_png, delta_png, gene_png, final_png, final_pdf)

    locus_genes <- sort(unique(chosen_rows$gene_name))
    omitted <- setdiff(spec$union_genes, locus_genes)
    manifest_rows[[length(manifest_rows) + 1]] <- data.table(
      panel = spec$panel,
      disease = spec$disease,
      bucket = spec$bucket,
      tf_set = spec$tf_set,
      selected_tf = chosen$tf,
      sv_chrom = chosen$sv_chrom,
      sv_window_start = chosen$sv_window_start,
      sv_window_end = chosen$sv_window_end,
      sv_file = chosen$sv_file,
      n_genes_in_locus = chosen$n_genes,
      locus_genes = chosen$genes_in_locus,
      shared_genes_in_locus = chosen$shared_genes_in_locus,
      omitted_example_genes = paste(omitted, collapse = ";"),
      locus_candidate_csv = candidates_csv,
      locus_gene_csv = genes_csv,
      output_png = final_png,
      output_pdf = final_pdf
    )
  }

  manifest_dt <- rbindlist(manifest_rows, use.names = TRUE, fill = TRUE)
  manifest_csv <- file.path(output_dir, "example_3d_delta_gene_track_manifest.csv")
  manifest_md <- file.path(output_dir, "example_3d_delta_gene_track_manifest.md")
  fwrite(manifest_dt, manifest_csv)
  write_manifest_md(manifest_dt, manifest_md)

  cat(manifest_csv, "\n")
  cat(manifest_md, "\n")
}

main()
