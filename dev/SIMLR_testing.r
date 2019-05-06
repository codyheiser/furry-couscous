# testing SIMLR for euclidean distance comparison
library(tidyverse)
library(SIMLR)
library(igraph)
source('fcc_utils.r')

# read in feature-selected data
qi <- read.csv('inputs/qi_s1.500feature.genelabels.tsv', sep='\t')
qi_clu <- t(read.csv('inputs/qi_clusters.csv', header = F))
qi <- t(qi) # transpose matrix to get cells by genes format
qi_norm <- arcsinh.norm(qi, margin=2) # normalize by arcsinh-tranforming fractional counts per gene

# use SIMLR to estimate number of clusters in dataset for feeding into DR algorithm
# input is raw counts
qi_clu_est <- SIMLR_Estimate_Number_of_Clusters(qi, NUMC = 2:14)
# perform SIMLR analysis with the estimated number of clusters from above
qi_SIMLR <- SIMLR(qi_norm, c = which.min(qi_clu_est$K1))

plt <- ggplot(data = data.frame(qi_SIMLR$ydata), aes(x = X1, y = X2))+
  geom_point(size=2.5, alpha=0.7, aes(color=factor(qi_clu)))+
  labs(x = paste0('SIMLR',' 1'), y = paste0('SIMLR',' 2'))+
  plot.opts
plt

# we can compare the density peak clustering on the whole dataset t-SNE to the SIMLR clusters
nmi_1 = compare(qi_clu, qi_SIMLR$y$cluster, method="nmi")
nmi_1

# output the reduced-dimension dataset to file
write.csv(qi_SIMLR$F, file = 'dev/SIMLR_out/qi_SIMLR_F.csv', row.names = F, col.names = F)
write.csv(qi_SIMLR$ydata, file = 'dev/SIMLR_out/qi_SIMLR_ydata.csv', row.names = F, col.names = F)

####################################################################################################
# Discrete Data

# read in feature-selected data
retina1 <- read.csv('inputs/GSM1626793_P14Retina_1.1kcells.tsv', sep='\t')
retina1_clu <- t(read.csv('inputs/P14Retina_clusters.csv', header = F))
retina1 <- t(retina1)
retina1_norm <- arcsinh.norm(retina1, margin=2) # normalize by arcsinh-tranforming fractional counts per gene

# use SIMLR to estimate number of clusters in dataset for feeding into DR algorithm
# input is raw counts
retina1_clu_est <- SIMLR_Estimate_Number_of_Clusters(retina1, NUMC = 2:14)

retina1_SIMLR <- SIMLR(retina1_norm, c = which.min(retina1_clu_est$K1))

plt <- ggplot(data = data.frame(retina1_SIMLR$ydata), aes(x = X1, y = X2))+
  geom_point(size=2.5, alpha=0.7, aes(color=factor(retina1_SIMLR$y$cluster)))+
  labs(x = paste0('SIMLR',' 1'), y = paste0('SIMLR',' 2'))+
  plot.opts
plt

# output the reduced-dimension dataset to file
write.csv(retina1_SIMLR$F, file = 'dev/SIMLR_out/retina1_SIMLR_F.csv', row.names = F, col.names = F)
write.csv(retina1_SIMLR$ydata, file = 'dev/SIMLR_out/retina1_SIMLR_ydata.csv', row.names = F, col.names = F)
