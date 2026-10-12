suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(RColorBrewer))
SpatialColors <- colorRampPalette(colors = rev(x = brewer.pal(n = 11, name = "Spectral")))
source("/hpc/group/jilab/hz/sennet/colon/multi/code/function/spatialplot.R")

expr <- readRDS("multi_human_LUAD/expr/LUAD_2_A_protein_axis.rds")
coord <- read.csv(paste0("multi_human_LUAD/spatial_axis/LUAD_2_A_protein_axis.csv"), row.names = 1)
coord <- coord[colnames(expr), ]
coord <- as.matrix(coord)
colnames(coord) <- c("row", "col")

for (genename in rownames(expr)) {
  df <- as.data.frame(cbind(coord, expr = expr[genename, rownames(coord)]))
  
  spatialplot(df = df, genename = genename, quan = 0.005, yreverse = FALSE)
  ggsave(paste0("multi_human_LUAD/plot/codex_protein_original/", genename, ".png"), dpi = 300, width = 4, height = 3)
}
