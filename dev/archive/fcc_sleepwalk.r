# sleepwalk implementation

# @author: C Heiser
# June 2019

source('fcc_utils.r')
library(sleepwalk)

retina <- read.csv('inputs/GSM1626793_P14Retina_1.processed.tsv', sep = '\t', header = T, row.names = 1, check.names = F) %>% arcsinh.norm(margin = 1)
retina_tSNE <- read.csv('dev/pymethods_out/retina_tSNE.csv', header = F) 
retina_UMAP <- read.csv('dev/pymethods_out/retina_UMAP.csv', header = F) 

sleepwalk(embeddings = list(retina_tSNE, retina_UMAP), featureMatrices = retina, pointSize = 4)


colon <- read.csv('inputs/GSM2743164_rep1_colon_rnaseq.processed.tsv', sep = '\t', header = T, row.names = 1, check.names = F) 
colon.arcsinh <- colon %>% arcsinh.norm(margin = 1)
colon_tSNE <- read.csv('dev/pymethods_out/colon_tSNE.csv', header = F) 
colon_UMAP <- read.csv('dev/pymethods_out/colon_UMAP.csv', header = F)
colon_FItSNE <- read.csv('dev/pymethods_out/colon_FItSNE.csv', header = F)
colon_SIMLR <- read.csv('dev/Rmethods_out/colon_SIMLR_ydata.csv', header = T)
colon_ZINBWAVE <- read.csv('dev/Rmethods_out/colon_ZINB-WAVE.csv', header = T)
colon_scvis <- read.csv('dev/scvis_out/colon/perplexity_30_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.tsv', sep='\t', header = T, row.names = 1) 
colon_DCA <- read.csv('dev/pymethods_out/colon_DCA.csv', header = F)

sleepwalk(embeddings = list(colon_tSNE, colon_UMAP, colon_FItSNE, colon_SIMLR), featureMatrices = colon.arcsinh, pointSize = 4, maxdists = 75)

sleepwalk(embeddings = list(colon_DCA, colon_scvis), featureMatrices = colon, pointSize = 4)
