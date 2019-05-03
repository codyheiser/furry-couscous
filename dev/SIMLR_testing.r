# testing SIMLR for euclidean distance comparison
library(tidyverse)
library(SIMLR)
source('fcc_utils.r')

# read in feature-selected data
qi <- read.csv('inputs/qi_s1.500feature.genelabels.tsv', sep='\t')
qi <- t(qi) # transpose matrix to get cells by genes format
qi_norm <- arcsinh.norm(qi, margin=1) # normalize by arcsinh-tranforming fractional counts per gene

qi_SIMLR <- SIMLR(qi_norm, c = 14)
