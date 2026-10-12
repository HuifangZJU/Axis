suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(RColorBrewer))

df <- readRDS("multi_human_LUAD/xenium_DE.rds")

markers <- c("CD4", "CD8", "CD20", "CD68", "Ki67",
             "E.Cadherin", "Collagen.IV", "a.SMA", "IDO1")
df <- df %>%
  filter(marker %in% markers)

df_top <- df %>%
  group_by(marker) %>%
  arrange(P.Value, desc(abs(logFC))) %>%
  slice_head(n = 20) %>%
  ungroup()

top_gene <- unique(df_top$gene)
plotdf <- df %>%
  filter(gene %in% top_gene)

mat <- plotdf %>%
  select(gene, marker, logFC) %>%
  pivot_wider(names_from = marker,
              values_from = logFC) %>%
  tibble::column_to_rownames("gene") %>%
  as.matrix()

hl_gene <- hclust(dist(mat))
gene_order <- hl_gene$labels[hl_gene$order]

hl_marker <- hclust(dist(t(mat)))
marker_order <- hl_marker$labels[hl_marker$order]

plotdf$gene <- factor(plotdf$gene, levels = gene_order)
plotdf$marker <- factor(plotdf$marker, levels = marker_order)

ggplot(data = plotdf, aes(x = marker, y = gene, color = logFC, size = AveExpr)) +
  geom_point() +
  theme_classic() +
  labs(title = NULL, x = NULL, y = NULL, color = "Slope", size = "Expression") +
  scale_color_gradientn(colors = colorRampPalette(rev(brewer.pal(11, "Spectral")))(100),
                        limits = quantile(plotdf$logFC, c(0.01, 0.99)),
                        values = scales::rescale(c(quantile(plotdf$logFC, 0.01), 0, quantile(plotdf$logFC, 0.99))),
                        oob = scales::squish) +
  scale_size_continuous(range = c(0.5, 5)) +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("multi_human_LUAD/plot/xenium_DE/dotplot.pdf", width = 4, height = 8)
