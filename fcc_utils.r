# R utility functions

# @author: C Heiser
# May 2019

require(tidyverse)
source('ggplot_config.r')

remove.zeros <- function(counts, margin=1){
  # remove rows (margin=1) or columns (margin=2) from 'counts' dataframe that contain only zeros
  counts[apply(counts[,-1], margin, function(x) !all(x==0)),]
}

arcsinh.norm <- function(counts, margin=2, norm='l1', scale=1000){
  # function to normalize and transform RNA counts data
  #  counts = dataframe or matrix of RNA counts values
  #  margin = normalize by dividing each element by maximum of its row (1) or column (2)
  #  norm = if NULL, do not normalize before arcsinh tranforming. if "l1", normalize to maximum value along `margin`
  #  scale = value to scale counts or normalized counts to before arcsinh tranformation
  if(is.null(norm)){
    out <- apply(counts*scale, MARGIN = c(1,2), FUN = asinh)
  }else if(norm=='l1'){
    out <- counts/apply(counts, MARGIN = margin, max)
    out <- apply(out*scale, MARGIN = c(1,2), FUN = asinh)
  }
  return(out)
}

plot.DR <- function(results, colorby='c', name=''){
  # function to plot dimensionality reduction (DR) latent space
  #  results = dataframe or matrix of latent dimensions
  #  colorby = vector of values to color points by
  #  name = string containing name of DR technique for axis labels (i.e. "t-SNE", "UMAP", "SIMLR")
  results %>%
    mutate(plt.colors = colorby) %>%
    ggplot(aes(x = results[,1], y = results[,2]))+
    geom_point(size=2.5, alpha=0.7, aes(color=factor(plt.colors)), show.legend = F)+
    labs(x = paste0(name,' 1'), y = paste0(name,' 2'))+
    theme(legend.position = 'none')+
    plot.opts -> plt

  return(plt)
}
