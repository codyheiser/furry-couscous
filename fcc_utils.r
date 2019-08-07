# R utility functions

# @author: C Heiser
# August 2019

require(tidyverse)
require(Seurat)
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


seurat.pipe <- function(counts, n.pcs=100, k=30, tsne=T, umap=F, perplexity=30, seed=18){
  # normalize, feature select, scale, cluster, and reduce dimensionality via standard Seurat pipeline
  #   counts = matrix of counts in cells x genes format, with cell and gene labels as rownames and colnames, respectively
  #   n.pcs = number of PCs to calculate and use for downstream reductions
  #   k = nearest-neighbor value to use for Knn graph to seed Louvain clustering algorithm
  #   tsne = perform t-SNE on PCs?
  #   umap = perform UMAP on PCs?
  #   perplexity = parameter for t-SNE and UMAP reductions
  #   seed = seed for random layout of t-SNE and UMAP for reproducible results
  start.time <- Sys.time()
  
  obj <- CreateSeuratObject(counts=counts)                                                  # initialize Seurat object to get feature names for scaling
  obj <- obj %>%
    NormalizeData() %>%                                                                     # normalize counts within each cell and log1p-transform
    FindVariableFeatures() %>%                                                              # feature selection using UMI binning
    ScaleData(features = rownames(obj@assays$RNA@meta.features)) %>%                        # scale all features to prevent genes from taking over dataset
    RunPCA(npcs = n.pcs) %>%                                                                # run principal component analysis
    FindNeighbors(reduction = 'pca', dims = 1:n.pcs, k.param = k) %>%                       # build nn graph from PCs
    FindClusters(random.seed = seed)                                                        # perform Louvain clustering using nn graph
  
  if (tsne) {
    obj <- obj %>% 
      RunTSNE(dims = 1:n.pcs, reduction = 'pca', seed.use = seed, perplexity = perplexity)  # reduce dimensions, primed by PCA
  }
  if (umap) {
    obj <- obj %>% 
      RunUMAP(dims = 1:n.pcs, reduction = 'pca', seed.use = seed, n.neighbors = perplexity)  # reduce dimensions, primed by PCA
  }
  
  print(Sys.time() - start.time)
  return(obj)
}


fcc_predictcelltype<-function(qdata, ref.expr, anntext="Query", normalize=FALSE, corcutoff=0){
  # predict cell types based on expression correlation with reference database
  # adapted from scUnifrac (https://github.com/liuqivandy/scUnifrac)
  #   qdata = matrix of counts in cells x genes format, with cell and gene labels as rownames and colnames, respectively
  #   ref.expr = matrix of expression by cell type from reference database. 
  #              can be loaded using `load(system.file("extdata", "ref.expr.Rdata", package = "scUnifrac"))`
  #   anntext = string to annotate output heatmap with
  #   normalize = normalize expression counts prior to correlating with reference?
  #   corcutoff = minimum correlation value to require before returning cell type prediction
  if (normalize){
    sumqdata<-apply(qdata,2,sum)
    # normalize the data    
    normdata<-t(log2(t(qdata)/sumqdata*10000+1))     
  }
  
  commongene<-intersect(rownames(qdata),rownames(ref.expr))
  
  # require more than 300 genes in common to predict cell types
  if (length(commongene)>300){
    tst.match <- qdata[commongene,]
    ref.match<-ref.expr[commongene,]
    
    cors <- cor(ref.match,tst.match)
    
    cors_index <- unlist(apply(cors,2,function(x){cutoffind<-tail(sort(x),3)>corcutoff;return(order(x,decreasing=T)[1:3][cutoffind])}))
    
    if (length(cors_index)>1){
      
      cors_in = cors[cors_index,]
      
      colnames(cors_in)<-colnames(qdata)
      
      rhc<-hclust(dist(t(cors_in)),method="ward.D2")
      hc<-hclust(dist((cors_in)),method="ward.D2")
      rownum<-nrow(cors_in)
      colnum<-ncol(cors_in)
      layout(matrix(c(4,2,3,1), 2, 2, byrow = TRUE),width=c(1,6),heights=c(1,6))
      par(mar=c(3,1,1,10))
      par(xaxs="i")
      par(yaxs="i")
      image(1:colnum,1:rownum,t(cors_in[hc$order,rhc$order]),col=colorRampPalette(c("gray","white","red"))(100),axes=F) 
      rowcex<-min(0.6, 50 / rownum * 0.6)
      axis(4,1:rownum,labels=rownames(cors_in)[hc$order],las=2,tick=F,cex.axis=rowcex)
      mtext(anntext,side=1,line=1,at=colnum/2,cex=1.5)
      par(mar=c(0,1,1,10))
      plot(as.dendrogram(rhc),axes=F,leaflab="none")
      par(mar=c(3,1,1,0))
      plot(as.dendrogram(hc),horiz=T,axes=F,leaflab="none")
      par(mar=c(2,1,2,1))
      maxval<-round(max(cors_in),digits=1)
      zval<-seq(0,maxval,0.1)
      image(1:length(zval),1,matrix(zval,ncol=1),col= colorRampPalette(c("gray","white", "red"))(100),axes=F,xlab="",ylab="")
      mtext("0",side=1,line=0.5,at=1,cex=0.8)
      mtext(maxval,side=1,line=0.5,at=length(zval),cex=0.8)
      
      box(lty="solid",col="black")
      
      return(cors_in)
      
    } else if (length(cors_index)==1) {
      
      cors_in=matrix(cors[cors_index,],nrow=1)
      colnames(cors_in)<-colnames(qdata)	
      rownames(cors_in)<-rownames(cors)[cors_index]		
      rownum<-nrow(cors_in)
      colnum<-ncol(cors_in)
      rhc<-hclust(dist(t(cors_in)),method="ward.D2")
      layout(matrix(c(3,2,0,1), 2, 2, byrow = TRUE),width=c(1,6),heights=c(1,6))
      par(mar=c(3,1,1,10))
      par(xaxs="i")
      par(yaxs="i")
      image(1:colnum,1:rownum,t(t(cors_in[,rhc$order])),col=colorRampPalette(c("gray","white","red"))(100),axes=F,xlab="",ylab="") 
      rowcex<-min(0.6, 50 / rownum * 0.6)
      axis(4,1:rownum,labels=rownames(cors_in),las=2,tick=F,cex.axis=rowcex)
      mtext(anntext,side=1,line=1,at=colnum/2,cex=1.5)
      par(mar=c(0,1,1,10))
      plot(as.dendrogram(rhc),axes=F,leaflab="none")
      
      par(mar=c(2,1,2,1))
      maxval<-round(max(cors_in),digits=1)
      zval<-seq(0,maxval,0.1)
      image(1:length(zval),1,matrix(zval,ncol=1),col= colorRampPalette(c("gray","white", "red"))(100),axes=F,xlab="",ylab="")
      mtext("0",side=1,line=0.5,at=1,cex=0.8)
      mtext(maxval,side=1,line=0.5,at=length(zval),cex=0.8)
      
      box(lty="solid",col="black")
      
      return(cors_in)
      
    }
  }
  
  print(paste0('No cell type correlations detected above ', corcutoff))
  
}
