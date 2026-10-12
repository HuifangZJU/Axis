suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(RColorBrewer))
SpatialColors <- colorRampPalette(colors = rev(x = brewer.pal(n = 11, name = "Spectral")))
source(file.path(Sys.getenv("AXIS_R_HELPERS", "helpers"), "spatialplot.R"))  # spatial plotting helper (not distributed)
source(file.path(Sys.getenv("PRETSA_DIR", "PreTSA/function"), "spatialPred.R"))  # PreTSA (ref. 24)

expr <- readRDS("multi_human_LUAD/expr/LUAD_2_A_protein_axis.rds")
coord <- read.csv(paste0("multi_human_LUAD/spatial_axis/LUAD_2_A_protein_axis.csv"), row.names = 1)
coord <- coord[colnames(expr), ]
coord <- as.matrix(coord)
colnames(coord) <- c("row", "col")
colnames(expr) <- rownames(coord) <- paste0("CODEX:", rownames(coord))

coord_new <- read.csv(paste0("multi_human_LUAD/spatial_axis/LUAD_2_A_xenium_axis.csv"), row.names = 1)
coord_new <- as.matrix(coord_new)
colnames(coord_new) <- c("row", "col")
rownames(coord_new) <- paste0("Xenium:", rownames(coord_new))

markers <- c("CD4", "CD8", "CD20", "CD68", "Ki67",
             "E.Cadherin", "Collagen.IV", "a.SMA", "IDO1")

numknot <- 50
res <- spatialPred(expr = expr, coord = coord, knot = numknot, coord_new = coord_new)
saveRDS(res, file = paste0("multi_human_LUAD/expr/xenium_protein_pretsa_", numknot, ".rds"))

for (genename in markers) {
  df <- as.data.frame(cbind(coord_new, expr = res[genename, rownames(coord_new)]))
  
  spatialplot(df = df, genename = genename, quan = 0.005, yreverse = FALSE)
  ggsave(paste0("multi_human_LUAD/plot/xenium_protein_pretsa_", numknot, "_q0.005/", genename, ".png"), dpi = 300, width = 4, height = 3)
}
