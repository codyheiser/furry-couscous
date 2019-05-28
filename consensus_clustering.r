library(Seurat)
library(tidyverse)

# continuous data - GEO accession # GSM2743164
colon <- t(read.csv('inputs/GSM2743164_rep1_colon_rnaseq.processed.tsv', sep='\t', header = T, row.names=1, check.names = F))
CreateSeuratObject(counts=colon) %>% FindVariableFeatures() -> colon_seurat

# process data and find clusters using graph-based method
colon_seurat %>%
  ScaleData() %>%
  RunPCA() %>%
  FindNeighbors() %>%
  FindClusters(resolution=0.25) %>%
  RunTSNE(dims = 1:10, seed.use = 18, perplexity=30) -> colon_seurat

# visualize clusters with t-SNE
DimPlot(colon_seurat, reduction = "tsne")

# create heatmap of upregulated cluster markers
colon.markers <- FindAllMarkers(colon_seurat, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
top10 <- colon.markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)
DoHeatmap(colon_seurat, features = top10$gene) + NoLegend()

# export clusters to file
write.table(colon_seurat@meta.data$seurat_clusters, file = 'inputs/colon_clu_seurat.csv', row.names = F, col.names = F)
write.table(colon_seurat@reductions$tsne@cell.embeddings, file = 'dev/Rmethods_out/colon_seurat_tSNE.csv', row.names = T, col.names = T, sep = ',')

###################################################################################################################################
# discrete data - GEO accession # GSM1626793
retina <- t(read.csv('inputs/GSM1626793_P14Retina_1.processed.tsv', sep='\t', header = T, row.names = 1, check.names = F))
CreateSeuratObject(counts = retina) %>% FindVariableFeatures() -> retina_seurat

# process data and find clusters using graph-based method
retina_seurat %>%
  ScaleData() %>%
  RunPCA() %>%
  FindNeighbors() %>%
  FindClusters(resolution=0.3) %>%
  RunTSNE(dims = 1:10, seed.use = 18, perplexity=30) -> retina_seurat

# visualize clusters with t-SNE
DimPlot(retina_seurat, reduction = "tsne")

# create heatmap of upregulated cluster markers
retina.markers <- FindAllMarkers(retina_seurat, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
top10 <- retina.markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_logFC)
DoHeatmap(retina_seurat, features = top10$gene) + NoLegend()

# export clusters and UMAP projection to file
write.table(retina_seurat@meta.data$seurat_clusters, file = 'inputs/retina_clu_seurat.csv', row.names = F, col.names = F)
write.table(retina_seurat@reductions$tsne@cell.embeddings, file = 'dev/Rmethods_out/retina_seurat_tSNE.csv', row.names = T, col.names = T, sep = ',')
