# Louvain cluster expression analysis using Seurat

# @author: C Heiser
# May 2019

library(Seurat)
library(tidyverse)

# continuous data - GEO accession # GSM2743164
colon <- t(read.csv('../inputs/GSM2743164_rep1_colon_rnaseq_filtered_CH.tsv.gz', sep='\t', header = T, row.names = 1, check.names = F))
colnames(colon) <- 0:(ncol(colon)-1)
# read in phenograph clusters and get into meta.data format
phenograph.colon <- read.csv('outputs/colon_clu.csv', header = F)
rownames(phenograph.colon) <- 0:(nrow(phenograph.colon)-1) # adjust row names to python indexing
colnames(phenograph.colon) <- 'Phenograph.Clusters' # give it a descriptive header for Seurat
phenograph.colon$Phenograph.Clusters <- as.factor(phenograph.colon$Phenograph.Clusters) # make it a factor

CreateSeuratObject(counts=colon, meta.data = phenograph.colon) %>% # pass counts and cluster IDs as meta.data
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA() %>%
  SetIdent(value = "Phenograph.Clusters") %>% # set cell ID from meta.data column defined above
  RunTSNE(dims = 1:10, seed.use = 18, perplexity=30) -> colon.seurat

# visualize clusters with t-SNE
DimPlot(colon.seurat, reduction = "tsne")

# create heatmap of upregulated cluster markers
colon.markers <- FindAllMarkers(colon.seurat, min.pct = 0.25, logfc.threshold = 0.25)
top10 <- colon.markers %>% 
  group_by(cluster) %>% 
  top_n(n = 10, wt = avg_logFC)
# output cluster expression heatmap to image file for cell type identification
colon.heatmap <- DoHeatmap(colon.seurat, features = top10$gene) + NoLegend()
# save expression heatmap to file
png(filename='../images/colon_heatmap.png', width=600, height=600, res=100)
colon.heatmap
dev.off()

######################################################################################################################################################
# discrete data - GEO accession # GSM1626793
retina <- data.frame(t(read.csv('../inputs/GSM1626793_P14Retina_1.processed.norowlabels.tsv', sep='\t', header = T, check.names = F))) %>%
  rownames_to_column(var='gene') %>%
  separate(gene, into = c('chr','pos','gene'), sep = ':', remove=T)
rownames(retina) <- make.names(retina$gene, unique = T)
retina <- select(retina, -c(gene, chr, pos))
colnames(retina) <- 1:ncol(retina)
# read in phenograph clusters and get into meta.data format
phenograph.retina <- read.csv('outputs/retina_clu.csv', header = F)
colnames(phenograph.retina) <- 'Phenograph.Clusters' # give it a descriptive header for Seurat
phenograph.retina$Phenograph.Clusters <- as.factor(phenograph.retina$Phenograph.Clusters) # make it a factor

CreateSeuratObject(counts=retina, meta.data = phenograph.retina) %>% # pass counts and cluster IDs as meta.data
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA() %>%
  SetIdent(value = "Phenograph.Clusters") %>% # set cell ID from meta.data column defined above
  RunTSNE(dims = 1:10, seed.use = 18, perplexity=30) -> retina.seurat

# visualize clusters with t-SNE
DimPlot(retina.seurat, reduction = "tsne")

# create heatmap of upregulated cluster markers
retina.markers <- FindAllMarkers(retina.seurat, min.pct = 0.25, logfc.threshold = 0.25)
top6 <- retina.markers %>% 
  group_by(cluster) %>% 
  top_n(n = 6, wt = avg_logFC)
# output cluster expression heatmap to image file for cell type identification
retina.heatmap <- DoHeatmap(retina.seurat, features = top6$gene) + NoLegend()
# save expression heatmap to file
png(filename='../images/retina_heatmap.png', width=600, height=600, res=100)
retina.heatmap
dev.off()
