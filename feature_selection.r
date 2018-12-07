# Feature Selection 

# use feature selection tools to identify information-rich genes in scRNAseq data
# C Heiser, Fall 2018

library(tidyverse)
library(Seurat)
library(monocle)


# try Seurat's findvariablegenes function to select features in mouse retina dataset (Mocosko, et al)
P14retina7 <- read.table('inputs/GSM1626799_P14Retina_7.digital_expression.txt.gz', header = T, sep = '\t')
rownames(P14retina7) <- P14retina7$gene # move gene IDs into rownames

P14retina7 %>% 
  select(-gene) %>% # remove extraneous gene names column
  CreateSeuratObject() %>% # create Seurat object
  FindVariableGenes() -> P14retina7 # find variable genes and return to variable

write.csv(P14retina7@data[match(P14retina7@var.genes, rownames(P14retina7@data)),], 
          file = 'inputs/GSM1626799_P14Retina_7_SeuratSelect.csv', row.names = T, col.names = T)


# try Monocle's dpFeature function on the same dataset (Mocosko, et al)
featureData <- AnnotatedDataFrame(rownames(P14retina7@data))
colnames(featureData) <- 'gene_short_name'
test <- newCellDataSet(as.matrix(P14retina7@data), featureData = featureData)

#' Select features for constructing the developmental trajectory
#'
#' @param cds the CellDataSet upon which to perform this operation
#' @param num_cells_expressed A logic flag to determine whether or not you want to skip the calculation of rho / sigma
#' @param num_dim Number of clusters. The algorithm use 0.5 of the rho as the threshold of rho and the delta
#' corresponding to the number_clusters sample with the highest delta as the density peaks and for assigning clusters
#' @param rho_threshold The threshold of local density (rho) used to select the density peaks
#' @param delta_threshold The threshold of local distance (delta) used to select the density peaks
#' @param qval_threshold A logic flag passed to densityClust function in desnityClust package to determine whether or not Gaussian kernel will be used for calculating the local density
#' @param verbose Verbose parameter for DDRTree
#' @return an list contains a updated CellDataSet object after clustering and tree construction as well as a vector including the selected top significant differentially expressed genes across clusters of cells
#' @references Rodriguez, A., & Laio, A. (2014). Clustering by fast search and find of density peaks. Science, 344(6191), 1492-1496. doi:10.1126/science.1242072
#' @export
dpFeature <- function(cds, num_cells_expressed = 5, num_dim = 5, rho_threshold = NULL, delta_threshold = NULL, qval_threshold = 0.01, verbose = F){
  #1. determine how many pca dimension you want:
  cds <- detectGenes(cds)
  fData(cds)$use_for_ordering[fData(cds)$num_cells_expressed > num_cells_expressed] <- T

  if(is.null(num_dim)){
    lung_pc_variance <- plot_pc_variance_explained(cds, return_all = T)
    ratio_to_first_diff <- diff(lung_pc_variance$variance_explained) / diff(lung_pc_variance$variance_explained)[1]
    num_dim <- which(ratio_to_first_diff < 0.1) + 1
  }

  #2. run reduceDimension with tSNE as the reduction_method
  # absolute_cds <- setOrderingFilter(absolute_cds, quake_id)
  cds <- reduceDimension(cds, return_all = F, max_components=2, norm_method = 'log', reduction_method = 'tSNE', num_dim = num_dim,  verbose = verbose)

  #3. initial run of clusterCells_Density_Peak
  cds <- clusterCells_Density_Peak(cds, rho_threshold = rho_threshold, delta_threshold = delta_threshold, verbose = verbose)

  #perform DEG test across clusters:
  cds@expressionFamily <- negbinomial.size()
  pData(cds)$Cluster <- factor(pData(cds)$Cluster)
  clustering_DEG_genes <- differentialGeneTest(cds, fullModelFormulaStr = '~Cluster', cores = detectCores() - 2)
  clustering_DEG_genes_subset <- lung_clustering_DEG_genes[fData(cds)$num_cells_expressed > num_cells_expressed, ]

  #use all DEG gene from the clusters
  clustering_DEG_genes_subset <- clustering_DEG_genes_subset[order(clustering_DEG_genes_subset$qval), ]
  ordering_genes <- row.names(subset(clustering_DEG_genes, qval < qval_threshold))

  cds <- setOrderingFilter(cds, ordering_genes = lung_ordering_genes)
  cds <- reduceDimension(cds, norm_method = 'log', verbose = T)
  plot_cell_trajectory(cds, color_by = 'Time')

  return(list(new_cds = cds, ordering_genes = ordering_genes))
}
