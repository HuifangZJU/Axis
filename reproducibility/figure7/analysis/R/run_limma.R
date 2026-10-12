suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(limma))

expr <- readRDS("multi_human_LUAD/expr/LUAD_2_A_xenium_axis.rds")
colnames(expr) <- paste0("Xenium:", colnames(expr))
pred <- readRDS("multi_human_LUAD/expr/xenium_protein_pretsa_50.rds")
res <- t(apply(pred, 1, function(i){
  q <- quantile(i, probs = c(0.005, 1 - 0.005), na.rm = TRUE)
  i[i < q[1]] <- q[1]
  i[i > q[2]] <- q[2]
  return(i)
}))
res <- res[, colnames(expr)]

df <- data.frame()
for (mk in rownames(res)) {
  cat("marker:", mk, "\n")
  
  df.tmp <- data.frame(row.names = colnames(res), protein = res[mk, ])
  design <- model.matrix(object = as.formula("~ protein"),
                         data = df.tmp)
  lm.res <- lmFit(object = expr, design = design)
  lm.res <- eBayes(lm.res)
  s <- topTable(lm.res, coef = "protein", number = Inf)
  s$gene <- rownames(s)
  rownames(s) <- NULL
  
  s$marker <- mk
  df <- rbind(df, s)
}
saveRDS(df, file = "multi_human_LUAD/xenium_DE.rds")
