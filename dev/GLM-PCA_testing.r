# testing GLM-PCA for euclidean distance comparison
library(tidyverse)
setwd('../scrna2019/')
source('algs/glmpca.R')
setwd('../furry-couscous/')
source('fcc_utils.r')

# read in feature-selected data
qi <- t(read.csv('inputs/qi_s1.500feature.genelabels.tsv', sep='\t')) # transpose matrix to get cells by genes format
qi_clu <- t(read.csv('inputs/qi_clusters.csv', header = F))

# perform GLM-PCA analysis
qi_glmpca <- glmpca(Y=qi, L=2, verbose=T)
# plot results
plot.DR(qi_glmpca$factors, colorby = qi_clu, name='GLM-PCA')

# output the reduced-dimension dataset to file
write.csv(qi_glmpca$factors, file = 'dev/GLM-PCA_out/qi_GLM-PCA.csv', row.names = F, col.names = T)

####################################################################################################
# Discrete Data

# read in feature-selected data
retina1 <- t(read.csv('inputs/GSM1626793_P14Retina_1.1kcells.tsv', sep='\t', header = T))
retina1_clu <- t(read.csv('inputs/P14Retina_clusters.csv', header = F))

# perform GLM-PCA analysis
retina1_glmpca <- glmpca(Y=retina1, L=2, verbose=T)
# plot results
plot.DR(retina1_glmpca$factors, name='GLM-PCA')

# output the reduced-dimension dataset to file
write.csv(retina1_glmpca$factors, file = 'dev/GLM-PCA_out/retina1_GLM-PCA.csv', row.names = F, col.names = T)
