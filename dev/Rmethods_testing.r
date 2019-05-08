# testing dimensionality reduction techniques implemented in R
# @author: C Heiser

setwd('../scrna2019/')
source('algs/glmpca.R')
setwd('../furry-couscous/')
source('fcc_utils.r')
library(zinbwave)
library(scRNAseq)
library(SIMLR)

# read in feature-selected datasets for testing; both are in cells x genes format with cell and gene labels
# continuous data - GEO accession # GSM2743164
colon <- read.csv('inputs/GSM2743164_rep1_colon_rnaseq.preprocessed.tsv', sep='\t', header = T, row.names = 1)
# discrete data - GEO accession # GSM1626793
retina <- read.csv('inputs/GSM1626793_P14Retina_1.processed.tsv', sep='\t', header = T, row.names = 1)

############################################################################################################################################
# ZINB-WAVE

# create SummarizedExperiment object from counts data
# transpose matrix to genes x cells format
colon_expt <- SummarizedExperiment(assays = list(counts=data.matrix(t(colon))))

# perform ZINB-WAVE analysis
colon_zinbwave <- zinbwave(colon_expt, K=2, epsilon=1000, verbose=T)
# plot results
plot.DR(data.frame(colon_zinbwave@reducedDims$zinbwave), name = 'ZINB-WAVE')

# output the reduced-dimension dataset to file
write.csv(data.frame(colon_zinbwave@reducedDims$zinbwave), file = 'dev/Rmethods_out/colon_ZINB-WAVE.csv', row.names = F)


# create SummarizedExperiment object from counts data
# transpose matrix to genes x cells format
retina_expt <- SummarizedExperiment(assays = list(counts=data.matrix(t(retina))))

# perform ZINB-WAVE analysis
retina_zinbwave <- zinbwave(retina_expt, K=2, epsilon=1000, verbose=T, maxiter.optimize=150)
# plot results
plot.DR(data.frame(retina_zinbwave@reducedDims$zinbwave), name = 'ZINB-WAVE')

# output the reduced-dimension dataset to file
write.csv(data.frame(retina_zinbwave@reducedDims$zinbwave), file = 'dev/Rmethods_out/retina_ZINB-WAVE.csv', row.names = F)

############################################################################################################################################
# GLM-PCA

# perform GLM-PCA analysis
colon_glmpca <- glmpca(Y=t(colon), L=2, verbose=T)
# plot results
plot.DR(colon_glmpca$factors, name='GLM-PCA')

# output the reduced-dimension dataset to file
write.csv(colon_glmpca$factors, file = 'dev/Rmethods_out/colon_GLM-PCA.csv', row.names = F)


# perform GLM-PCA analysis
retina_glmpca <- glmpca(Y=t(retina), L=2, verbose=T)
# plot results
plot.DR(retina_glmpca$factors, name='GLM-PCA')

# output the reduced-dimension dataset to file
write.csv(retina_glmpca$factors, file = 'dev/Rmethods_out/retina_GLM-PCA.csv', row.names = F)

############################################################################################################################################
# SIMLR

colon_norm <- arcsinh.norm(colon, margin=1) # normalize by arcsinh-tranforming fractional counts per gene

# use SIMLR to estimate number of clusters in dataset for feeding into DR algorithm
# input is raw counts
colon_clu_est <- SIMLR_Estimate_Number_of_Clusters(t(colon), NUMC = 2:14)
# perform SIMLR analysis with the estimated number of clusters from above
colon_SIMLR <- SIMLR(t(colon_norm), c = which.min(colon_clu_est$K1))
# plot results
plot.DR(colon_SIMLR$ydata, name='SIMLR')

# output the reduced-dimension dataset to files
write.csv(qi_SIMLR$F, file = 'dev/Rmethods_out/colon_SIMLR_F.csv', row.names = F)
write.csv(qi_SIMLR$ydata, file = 'dev/Rmethods_out/colon_SIMLR_ydata.csv', row.names = F)


retina_norm <- arcsinh.norm(retina, margin=1) # normalize by arcsinh-tranforming fractional counts per gene

# use SIMLR to estimate number of clusters in dataset for feeding into DR algorithm
# input is raw counts
retina_clu_est <- SIMLR_Estimate_Number_of_Clusters(t(retina), NUMC = 2:14)
# perform SIMLR analysis with the estimated number of clusters from above
retina_SIMLR <- SIMLR(t(retina_norm), c = which.min(retina_clu_est$K1))
# plot results
plot.DR(retina_SIMLR$ydata, name='SIMLR')

# output the reduced-dimension dataset to files
write.csv(retina_SIMLR$F, file = 'dev/Rmethods_out/retina_SIMLR_F.csv', row.names = F)
write.csv(retina_SIMLR$ydata, file = 'dev/Rmethods_out/retina_SIMLR_ydata.csv', row.names = F)
