# R utility functions

# @author: C Heiser
# August 2019

require(tidyverse)
require(Seurat)
require(scatterpie)
require(ggpubr)
# plotting preferences
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


seurat.pipe <- function(counts, n.pcs=100, k=30, tsne=T, umap=F, perplexity=30, seed=18, ...){
  # normalize, feature select, scale, cluster, and reduce dimensionality via standard Seurat pipeline
  #   counts = Seurat object, or matrix of counts in cells x genes format with cell and gene labels as rownames and colnames, respectively
  #   n.pcs = number of PCs to calculate and use for downstream reductions
  #   k = nearest-neighbor value to use for Knn graph to seed Louvain clustering algorithm
  #   tsne = perform t-SNE on PCs?
  #   umap = perform UMAP on PCs?
  #   perplexity = parameter for t-SNE and UMAP reductions
  #   seed = seed for random layout of t-SNE and UMAP for reproducible results
  #   ... = options to pass to CreateSeuratObject()
  start.time <- Sys.time()
  
  if (class(counts)[1]!='Seurat'){
    counts <- CreateSeuratObject(counts=counts, ...)                                        # initialize Seurat object to get feature names for scaling
  }
  
  obj <- counts %>%
    NormalizeData() %>%                                                                     # normalize counts within each cell and log1p-transform
    FindVariableFeatures() %>%                                                              # feature selection using UMI binning
    ScaleData(features = rownames(counts@assays$RNA@meta.features)) %>%                     # scale all features to prevent genes from taking over dataset
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


set.cell.type <- function(obj, names){
  # create metadata field matching seurat_clusters to cell types
  #   obj = Seurat object with Idents() set as desired for mapping - usually 'seurat_clusters'
  #   names = list of strings containing cell type names for each ident.
  #           should be in order of idents. 
  #           i.e. if seurat_clusters = c(0,1,2), then names = c('stem','goblet','tuft') would mean {0:'stem',1:'goblet',2:'tuft'}
  types <- data.frame(seurat_clusters=levels(obj$seurat_clusters), cell.type=names)
  obj$cell.type <- types$cell.type[match(obj$seurat_clusters, types$seurat_clusters)]
  return(obj)
}


dimplot.pie <- function(obj, group.by, split.by, r=2, reduction='tsne', ...){
  # plot reduced dimensions with ident contributions from condition (grouping.var) shown as pie charts
  #   obj = Seurat object with Idents() set as desired - usually 'seurat_clusters'
  #   group.by = meta.data column header to plot cells by
  #   split.by = meta.data column header to split cells by for pie chart - usually 'orig.ident' for integrated samples
  #   r = radius of each pie chart on the plot
  #   reduction = type of dimension reduction from seurat object to use
  #   ... = options to pass to seurat's DimPlot() function
  start.time <- Sys.time()
  
  # get condition names from split.by argument
  conditions <- eval(parse(text = paste0('unique(obj$',split.by,')')))
  # get reduction key from seurat object
  red.key <- eval(parse(text = paste0('obj@reductions$',reduction,'@key')))
  
  # initial plot of reduced dimensions to get coords for scatterpie
  plt <- DimPlot(obj, reduction = reduction, group.by = group.by, label = T)
  
  # overlay pie charts on t-SNE that explain batch makeup of each cluster
  df <- eval(parse(text = paste0('obj@meta.data %>%
                             group_by(',group.by,') %>%
                             dplyr::count(',split.by,') %>%
                             spread(',split.by,', n) %>%
                             mutate(total=',paste(conditions,collapse='+'),')'
                             )))
  
  for(i in 1:length(conditions)){
  eval(parse(text = paste0('df <- df %>% 
                           mutate(',conditions[i],'.pct = ',conditions[i],'/total*100)')))
  }
  
  suppressWarnings(
    eval(parse(text = paste0('batch.data <- df %>% 
                             full_join(plt$layers[[2]]$data, by = group.by) %>% 
                             mutate(',red.key,'2 = ',red.key,'2 - (r+2))')))
    )
    
  print(Sys.time() - start.time)
  return(DimPlot(obj, reduction = reduction, group.by = group.by, label = T, ...)+DR.opts+NoLegend()+
           geom_scatterpie(data = batch.data, 
                           aes(x=eval(parse(text=paste0(red.key,'1'))), 
                               y=eval(parse(text=paste0(red.key,'2'))), 
                               r=r), 
                           cols=conditions))
}


fcc.corr.expr <- function(obj, ident, grouping.var, plot.out=T, n.genes.label=10){
  # correlate average expression within an ident across condition (grouping.var) and return df
  #   obj = Seurat object with Idents() set as desired - usually 'seurat_clusters'
  #   ident = name of ident to correlate expression within
  #   grouping.var = meta.data column header to group cells by - usually 'orig.ident' for integrated samples
  #   plot.out = scatterplot correlation of expression?
  #   n.genes.label = top n non-correlated genes to label on plot - default top 10
  start.time <- Sys.time()
  
  s <- subset(obj, idents=ident)
  Idents(s) <- grouping.var
  ident.names <- levels(Idents(s))
  avg.expr <- log1p(AverageExpression(s)$RNA) %>%
    rownames_to_column('gene') %>%
    mutate(diff=eval(parse(text=paste0('abs(',ident.names[1],'-',ident.names[2],')')))) %>%
    arrange(-diff)
  
  top.expr <- avg.expr %>%
    top_n(n.genes.label, wt=diff)
  
  if(nrow(top.expr)==0){
    print(Sys.time() - start.time)
    stop(paste0('Ident ',ident,' absent in both groups'))
  }
  
  if(plot.out){
    cond1.genes <- (top.expr %>% dplyr::filter(eval(parse(text=paste0(ident.names[1],'>',ident.names[2])))))$gene
    cond2.genes <- (top.expr %>% dplyr::filter(eval(parse(text=paste0(ident.names[1],'<',ident.names[2])))))$gene
    
    plt.df <- avg.expr %>%
      column_to_rownames('gene')
    p1 <- ggplot(plt.df, eval(parse(text = paste0('aes(',ident.names[1],',',ident.names[2],')')))) +
      geom_point() +
      ggtitle(ident) +
      labs(x=ident.names[1], y=ident.names[2]) +
      plot.opts
    if(length(cond1.genes)>0){
      p1 <- LabelPoints(plot=p1, points=cond1.genes, repel=T, xnudge = 0, ynudge = 0, color='blue')
    }
    if(length(cond2.genes)>0){
      p1 <- LabelPoints(plot=p1, points=cond2.genes, repel=T, xnudge = 0, ynudge = 0, color='red')
    }
    
    top.expr <- list(expr=top.expr, plt=p1)
  }
  
  print(Sys.time() - start.time)
  return(top.expr)
}


fcc.corr.expr.per.ident <- function(obj, grouping.var, n.genes.label=10){
  # correlate average expression within an ident across condition (grouping.var) and return df
  #   obj = Seurat object with Idents() set as desired - usually 'seurat_clusters'
  #   grouping.var = meta.data column header to group cells by - usually 'orig.ident' for integrated samples
  #   n.genes.label = top n non-correlated genes to label on each plot - default top 10
  start.time <- Sys.time()
  
  start.idents <- levels(Idents(obj))
  
  markers <- NULL
  plt.list <- list()
  for (i in 1:length(start.idents)) {
    message(paste0('Correlating expression for ident: ',start.idents[i]))
    if (is_null(markers)) {
      tryCatch(
        {
          response <- fcc.corr.expr(obj, ident=start.idents[i], grouping.var=grouping.var, plot.out=T, n.genes.label=n.genes.label)
          markers <- response$expr %>%
            mutate(cluster=paste(start.idents[i]))
          plt.list[[i]] <- response$plt
        }, error=function(cond) {
          message(paste0(cond,'\n'))
        }
      )
    }else{
      tryCatch(
        {
          response <- fcc.corr.expr(obj, ident=start.idents[i], grouping.var=grouping.var, plot.out=T, n.genes.label=n.genes.label)
          markers <- rbind(markers, 
                           response$expr %>%
                             mutate(cluster=paste(start.idents[i])))
          plt.list[[i]] <- response$plt
        }, error=function(cond) {
          message(paste0(cond,'\n'))
        }
      )
    }
  }
  # clean plots for arranging in figure
  message('Generating figure ... ')
  clean.plts <- lapply(Filter(Negate(is.null), plt.list), FUN = function(x){return(x+labs(x=NULL,y=NULL)+theme_pubr())})
  # arrange plots into figure with common labels and clean graphs
  fig <- ggarrange(plotlist = clean.plts, ncol = ceiling(length(clean.plts)/3), nrow = ceiling(length(clean.plts)/ceiling(length(clean.plts)/3))) %>%
    annotate_figure(left = text_grob(plt.list[[1]]$labels$y, rot = 90, size = 14), bottom = text_grob(plt.list[[1]]$labels$x, size = 14))
  
  print(Sys.time() - start.time)
  return(list(expr=markers, plt=fig, plt.list=plt.list))
}


fcc.find.conserved.markers <- function(obj, grouping.var, n.genes.per.ident=10, max.cells.per.ident=300){
  # ID conserved cell type markers in each cluster across condition (grouping.var) and return df
  #   obj = Seurat object with Idents() set as desired - usually 'seurat_clusters'
  #   grouping.var = meta.data column header to group cells by - usually 'orig.ident' for integrated samples
  #   n.genes.per.group = top n genes to return per ident - default 10
  #   max.cells.per.ident = downsample idents when testing to speed up processing time - set to Inf to deactivate downsampling
  start.time <- Sys.time()
  
  markers <- NULL
  for (id in levels(Idents(obj))) {
    if (is_null(markers)) {
      tryCatch(
        {
          markers <- FindConservedMarkers(obj, ident.1=id, grouping.var=grouping.var, max.cells.per.ident=max.cells.per.ident) %>% 
            rownames_to_column('gene') %>% 
            top_n(n=n.genes.per.ident, wt=minimump_p_val) %>% 
            mutate(cluster=paste(id))
        }, error=function(cond) {
          message(paste0(cond,'\n'))
        }
      )
    } else {
      tryCatch(
        {
          markers <- rbind(markers, FindConservedMarkers(obj, ident.1=id, grouping.var=grouping.var, max.cells.per.ident=max.cells.per.ident) %>% 
                             rownames_to_column('gene') %>% 
                             top_n(n.genes.per.ident, wt=minimump_p_val) %>% 
                             mutate(cluster=paste(id)))
        }, error=function(cond) {
          message(paste0(cond,'\n'))
        }
      )
    }
  }
  
  print(Sys.time() - start.time)
  return(markers)
}


fcc.find.DE.markers <- function(obj, grouping.var, n.genes.per.ident=10, max.cells.per.ident=300){
  # ID differentially-expressed cell type markers in each cluster across condition (grouping.var) and return df
  #   obj = Seurat object with Idents() set as desired - usually 'seurat_clusters'
  #   grouping.var = meta.data column header to group cells by - usually 'orig.ident' for integrated samples
  #   n.genes.per.group = top n genes to return per ident - default 10
  #   max.cells.per.ident = downsample idents when testing to speed up processing time - set to Inf to deactivate downsampling
  start.time <- Sys.time()
  
  start.idents <- levels(Idents(obj))
  conditions <- unique(eval(parse(text=paste0('obj$',grouping.var))))
  
  if (length(conditions)!=2){
    message('Only capable of DE analysis of binary conditions')
    return()
  }
  
  obj$ident_condition <- paste(Idents(obj), eval(parse(text=paste0('obj$',grouping.var))), sep='_')
  Idents(obj) <- 'ident_condition'
  
  markers <- NULL
  
  for (i in start.idents) {
    message(paste0('Performing differential expression analysis between ', conditions[1], ' and ', conditions[2], ' for ident: ', i))
    if (is_null(markers)) {
      tryCatch(
        {
          markers <- FindMarkers(obj, ident.1=paste(i, conditions[1], sep='_'), ident.2=paste(i, conditions[2], sep='_'), 
                                 max.cells.per.ident=max.cells.per.ident) %>% 
            rownames_to_column('gene') %>% 
            top_n(n=n.genes.per.ident, wt=p_val) %>% 
            mutate(cluster=i)
        }, error=function(cond) {
          message(paste0(cond,'\n'))
        }
      )
    } else {
      tryCatch(
        {
          markers <- rbind(markers, FindMarkers(obj, ident.1=paste(i, conditions[1], sep='_'), ident.2=paste(i, conditions[2], sep='_'), 
                                                max.cells.per.ident=max.cells.per.ident) %>% 
                             rownames_to_column('gene') %>% 
                             top_n(n=n.genes.per.ident, wt=p_val) %>% 
                             mutate(cluster=i))
        }, error=function(cond) {
          message(paste0(cond,'\n'))
        }
      )
    }
  }
  
  print(Sys.time() - start.time)
  return(markers)
}


fcc.predictcelltype <- function(qdata, ref.expr, anntext="Query", corcutoff=0){
  # predict cell types based on expression correlation with reference database
  # adapted from scUnifrac (https://github.com/liuqivandy/scUnifrac)
  #   qdata = matrix of counts in cells x genes format, with cell and gene labels as rownames and colnames, respectively
  #   ref.expr = matrix of expression by cell type from reference database. 
  #              can be loaded using `load(system.file("extdata", "ref.expr.Rdata", package = "scUnifrac"))`
  #   anntext = string to annotate output heatmap with
  #   corcutoff = minimum correlation value to require before returning cell type prediction
  
  commongene<-intersect(rownames(qdata),rownames(ref.expr))
  
  # require more than 300 genes in common to predict cell types
  if (length(commongene)>300){
    tst.match <- qdata[commongene,]
    ref.match<-ref.expr[commongene,]
    
    cors <- cor(ref.match,tst.match)
    
    cors_index <- unlist(apply(cors,2,function(x){cutoffind<-tail(sort(x),3)>corcutoff;return(order(x,decreasing=T)[1:3][cutoffind])}))
    
    tryCatch(
      {
        cors_in <- cors[cors_index,]
        cors_in <- cors_in[unique(rownames(cors_in)),]
        heatmap(cors_in, labCol=F, margins=c(0.5,10), cexRow=0.7, main=anntext)
        return(cors_in)
      }, error=function(cond) {
        message(paste0('No cell type correlations detected above ', corcutoff))
        message(paste0(cond,'\n'))
      }
    )
  }
}


fcc.predictcelltype.per.ident <- function(obj, ref.expr, corcutoff){
  # predict cell types for each ident in Seurat object based on expression correlation with reference database
  # adapted from scUnifrac (https://github.com/liuqivandy/scUnifrac)
  #   obj = Seurat object with Idents() set as desired - usually 'seurat_clusters'
  #   ref.expr = matrix of expression by cell type from reference database. 
  #              can be loaded using `load(system.file("extdata", "ref.expr.Rdata", package = "scUnifrac"))`
  #   corcutoff = minimum correlation value to require before returning cell type prediction
  start.time <- Sys.time()
  
  # use scUnifrac to predict cell type for each ident from expression
  for (id in levels(obj@active.ident)) {
    message(paste0('Predicting cell types from ident: ', id))
    cell.type.pred <- fcc.predictcelltype(as.matrix(obj@assays$RNA@counts[,WhichCells(obj, ident=id)]), 
                                          ref.expr = ref.expr, 
                                          corcutoff = corcutoff,
                                          anntext = id)
  }
  rm(cell.type.pred)
  print(Sys.time() - start.time)
}
