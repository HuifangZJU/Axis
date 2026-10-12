suppressPackageStartupMessages(library(Matrix))

pred <- readRDS("multi_human_LUAD/expr/xenium_protein_pretsa_50.rds")
res <- t(apply(pred, 1, function(i){
  q <- quantile(i, probs = c(0.005, 1 - 0.005), na.rm = TRUE)
  i[i < q[1]] <- q[1]
  i[i > q[2]] <- q[2]
  return(i)
}))
colnames(res) <- sub("Xenium:", "", colnames(res))
saveRDS(res, file = "multi_human_LUAD/expr/xenium_protein_predict.rds")
