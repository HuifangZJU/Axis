suppressPackageStartupMessages(library(Matrix))

for (i in list.files("multi_human_LUAD/counts", full.names = TRUE)) {
  counts <- read.csv(i, row.names = 1)
  counts$DAPI <- NULL
  counts <- as.matrix(counts)
  counts <- t(counts)
  counts <- counts[, colSums(counts) > 0]
  expr <- log1p(t(t(counts) / colSums(counts) * 1e4))
  saveRDS(expr, file = sub(".csv", ".rds", sub("counts", "expr", i)))
}
