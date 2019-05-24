library(Seurat)
library(tidyverse)

# continuous data - GEO accession # GSM2743164
colon <- t(read.csv('inputs/GSM2743164_rep1_colon_rnaseq_filtered.tsv.gz', sep='\t', header = T))
colon.clu <- read.csv('inputs/colon_clu.csv', header = F)
# discrete data - GEO accession # GSM1626793
retina <- read.csv('inputs/GSM1626793_P14Retina_1.processed.tsv', sep='\t', header = T, row.names = 1)
retina.clu <- read.csv('inputs/retina_clu.csv', header = F)

paste0('cell',1:ncol(colon)) -> colnames(colon)
CreateSeuratObject(counts = colon) %>%
  FindVariableFeatures() -> colon_seurat

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(colon_seurat), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(colon_seurat)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
CombinePlots(plots = list(plot1, plot2))

colon_seurat %>%
  ScaleData() %>%
  RunPCA(features=VariableFeatures(colon_seurat)) %>%
  FindNeighbors() %>%
  FindClusters() -> colon_seurat

colon_seurat_clu <- colon_seurat@meta.data$seurat_clusters
write.table(colon_seurat_clu, file = 'inputs/colon_clu_seurat.csv', row.names = F, col.names = F)

BuildClusterTree(colon_seurat, graph = 'RNA_nn') %>% PlotClusterTree()
Seurat::VariableFeaturePlot(colon_seurat)


CreateSeuratObject(counts = t(retina)) %>%
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA() %>%
  FindNeighbors() %>%
  FindClusters() -> retina_seurat

retina_seurat_clu <- retina_seurat@meta.data$seurat_clusters
write.table(retina_seurat_clu, file = 'inputs/retina_clu_seurat.csv', row.names = F, col.names = F)
