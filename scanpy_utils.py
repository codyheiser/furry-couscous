# -*- coding: utf-8 -*-
'''
scanpy utility functions

@author: C Heiser
October 2019
'''
# basics
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist            # unique pairwise and crosswise distances
from sklearn.neighbors import kneighbors_graph      # simple K-nearest neighbors graph
# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')


def reorder_adata(adata, descending = True):
    '''place cells in descending order of total counts'''
    if(descending):
        new_order = np.argsort(adata.X.sum(axis=1))[::-1]
    elif(not descending):
        new_order = np.argsort(adata.X.sum(axis=1))[:]
    adata.X = adata.X[new_order,:].copy()
    adata.obs = adata.obs.iloc[new_order].copy()


def arcsinh(adata, layer=None, scale=1000):
    '''
    return arcsinh-normalized values for each element in anndata counts matrix
    l1 normalization (sc.pp.normalize_total) should be performed before this transformation
        adata = AnnData object
        layer = name of lauer to perform arcsinh-normalization on. if None, use AnnData.X
        scale = factor to scale normalized counts to; default 1000
    '''
    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    adata.layers['arcsinh_norm'] = np.arcsinh(mat * scale)


def gf_icf(adata, layer=None):
    '''
    return GF-ICF scores for each element in anndata counts matrix
        adata = AnnData object
        layer = name of layer to perform GF-ICF normalization on. if None, use AnnData.X
    '''
    if layer is None:
        tf = adata.X.T / adata.X.sum(axis=1)
        tf = tf.T
        ni = adata.X.astype(bool).sum(axis=0)
        
        layer = 'X' # set input layer for naming output layer

    else:
        tf = adata.layers[layer].T / adata.layers[layer].sum(axis=1)
        tf = tf.T
        ni = adata.layers[layer].astype(bool).sum(axis=0)
        
    idf = np.log(adata.n_obs / (ni+1))
    
    adata.layers['{}_gf-icf'.format(layer)] = tf*idf


def knn_graph(dist_matrix, k, adata, save_rep='knn'):
    '''
    build simple binary k-nearest neighbor graph and add to anndata object
        dist_matrix = distance matrix to calculate knn graph for (i.e. pdist(adata.obsm['X_pca']))
        k = number of nearest neighbors to determine
        adata = AnnData object to add resulting graph to (in .uns slot)
        save_rep = name of .uns key to save knn graph to within adata (default adata.uns['knn'])
    '''
    adata.uns[save_rep] = kneighbors_graph(dist_matrix, k, mode='connectivity', include_self=False).toarray()


def subset_uns_by_ID(adata, uns_keys, obs_col, IDs):
    '''
    subset symmetrical distance matrices and knn graphs in adata.uns by one or more IDs defined in adata.obs
        adata = AnnData object 
        uns_keys = list of keys in adata.uns to subset. new adata.uns keys will be saved with ID appended to name (i.e. adata.uns['knn'] -> adata.uns['knn_ID1'])
        obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
        IDs = list of IDs to include in subset
    '''
    for key in uns_keys:
        tmp = adata.uns[key][adata.obs[obs_col].isin(IDs),:] # subset symmetrical uns matrix along axis 0
        tmp = tmp[:, adata.obs[obs_col].isin(IDs)] # subset symmetrical uns matrix along axis 1

        adata.uns['{}_{}'.format(key, '_'.join([str(x) for x in IDs]))] = tmp # save new .uns key by appending IDs to original key name


def recipe_fcc(adata, mito_names='MT-'):
    '''
    scanpy preprocessing recipe
        adata = AnnData object with raw counts data in .X 
        mito_names = substring encompassing mitochondrial gene names for calculation of mito expression

    -calculates useful .obs and .var columns ('total_counts', 'pct_counts_mito', 'n_genes_by_counts', etc.)
    -orders cells by total counts
    -stores raw counts (adata.raw.X)
    -provides GF-ICF normalization (adata.layers['X_gf-icf'])
    -normalization and log1p transformation of counts (adata.X)
    -identifies highly-variable genes using seurat method (adata.var['highly_variable'])
    '''
    reorder_adata(adata, descending=True) # reorder cells by total counts descending

    # raw
    adata.raw = adata # store raw counts before manipulation

    # obs/var
    adata.var['mito'] = adata.var_names.str.contains(mito_names) # identify mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mito'], inplace=True) # calculate standard qc .obs and .var
    adata.obs['ranked_total_counts'] = np.argsort(adata.obs['total_counts']) # rank cells by total counts

    # gf-icf
    gf_icf(adata, layer=None) # add gf-icf scores to adata.layers['gf-icf']

    # normalize/transform
    sc.pp.normalize_total(adata, target_sum=10000, layers=None, layer_norm=None, key_added='norm_factor')
    sc.pp.log1p(adata) # log1p transform counts

    # HVGs
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)


def gf_icf_markers(adata, n_genes=5, group_by='louvain'):
    '''
    return n_genes with top gf-icf scores for each group
        adata = AnnData object preprocessed using gf_icf() or recipe_fcc() function
        n_genes = number of top gf-icf scored genes to return per group
        group_by = how to group cells to ID marker genes
    '''
    markers = pd.DataFrame()
    for clu in adata.obs[group_by].unique():
        gf_icf_sum = adata.layers['gf-icf'][adata.obs[group_by]==str(clu)].sum(axis=0)
        gf_icf_mean = adata.layers['gf-icf'][adata.obs[group_by]==str(clu)].mean(axis=0)
        top = np.argpartition(gf_icf_sum, -n_genes)[-n_genes:]
        gene_IDs = adata.var.index[top]
        markers = markers.append(pd.DataFrame({group_by:np.repeat(clu,n_genes), 'gene':gene_IDs, 'gf-icf_sum':gf_icf_sum[top], 'gf-icf_mean':gf_icf_mean[top]}))

    return markers


def find_centroids(adata, use_rep, obs_col='louvain'):
    '''
    find cluster centroids
        adata = AnnData object
        use_rep = adata.obsm key containing space to calculate centroids in (i.e. 'X_pca')
        obs_col = adata.obs column name containing cluster IDs
        save_rep = adata.uns key
    '''
    # get unique cluster names
    names = sorted(adata.obs[obs_col].unique())
    # calculate centroids
    adata.uns['{}_centroids'.format(use_rep)] = np.array([np.mean(adata.obsm[use_rep][adata.obs[obs_col]==clu,:],axis=0) for clu in sorted(adata.obs[obs_col].unique())])
    # calculate distances between all centroids
    adata.uns['{}_centroid_distances'.format(use_rep)] = cdist(adata.uns['{}_centroids'.format(use_rep)], adata.uns['{}_centroids'.format(use_rep)])
    tmp = np.argsort(adata.uns['{}_centroid_distances'.format(use_rep)][0,])
    adata.uns['{}_centroid_ranks'.format(use_rep)] = pd.DataFrame({use_rep:tmp.argsort()}, index=names)



class DR_plot():
    '''
    class defining pretty plots of dimension-reduced embeddings such as PCA, t-SNE, and UMAP
        DR_plot().plot(): utility plotting function that can be passed any numpy array in the `data` parameter
                 .plot_IDs(): plot one or more cluster IDs on top of an .obsm from an `AnnData` object
                 .plot_centroids(): plot cluster centroids defined using find_centroids() function on `AnnData` object
    '''
    def __init__(self, dim_name='dim', figsize=(5,5)):
        '''
        dim_name = how to label axes ('dim 1' on x and 'dim 2' on y by default)
        figsize = size of resulting axes
        '''
        self.fig, self.ax = plt.subplots(1, figsize=figsize)

        plt.xlabel('{} 1'.format(dim_name), fontsize=14)
        self.ax.xaxis.set_label_coords(0.2, -0.025)
        plt.ylabel('{} 2'.format(dim_name), fontsize=14)
        self.ax.yaxis.set_label_coords(-0.025, 0.2)

        plt.annotate('', textcoords='axes fraction', xycoords='axes fraction', xy=(-0.006,0), xytext=(0.2,0), arrowprops=dict(arrowstyle= '<-', lw=2, color='black'))
        plt.annotate('', textcoords='axes fraction', xycoords='axes fraction', xy=(0,-0.006), xytext=(0,0.2), arrowprops=dict(arrowstyle= '<-', lw=2, color='black'))

        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
    

    def plot(self, data, color, pt_size=75, legend=None, save_to=None):
        '''
        general plotting function for dimensionality reduction outputs with cute arrows and labels
            data = np.array containing variables in columns and observations in rows
            color = list of length nrow(data) to determine how points should be colored
            pt_size = size of points in plot
            legend = None, 'full', or 'brief'
            save_to = path to .png file to save output, or None
        '''
        sns.scatterplot(data[:,0], data[:,1], s=pt_size, alpha=0.7, hue=color, legend=legend, edgecolor='none', ax=self.ax)

        if legend is not None:
            plt.legend(bbox_to_anchor=(1,1,0.2,0.2), loc='lower left', frameon=False, fontsize='small')

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight', dpi=1000)
    

    def plot_IDs(self, adata, use_rep, obs_col, IDs='all', pt_size=75, legend=None, save_to=None):
        '''
        general plotting function for dimensionality reduction outputs with cute arrows and labels
            adata = anndata object to pull dimensionality reduction from
            use_rep = adata.obsm key to plot from (i.e. 'X_pca')
            obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
            IDs = list of IDs to plot, graying out cells not assigned to those IDS (default 'all' IDs)
            pt_size = size of points in plot
            legend = None, 'full', or 'brief'
            save_to = path to .png file to save output, or None
        '''
        plotter = adata.obsm[use_rep]

        if IDs == 'all':
            sns.scatterplot(plotter[:,0], plotter[:,1], ax=self.ax, s=pt_size, alpha=0.7, hue=adata.obs[obs_col], legend=legend, edgecolor='none', palette='plasma')

        else:
            sns.scatterplot(plotter[-adata.obs[obs_col].isin(IDs), 0], plotter[-adata.obs[obs_col].isin(IDs), 1], ax=self.ax, s=pt_size, alpha=0.1, color='gray', legend=False, edgecolor='none')
            sns.scatterplot(plotter[adata.obs[obs_col].isin(IDs), 0], plotter[adata.obs[obs_col].isin(IDs), 1], ax=self.ax, s=pt_size, alpha=0.7, hue=adata.obs.loc[adata.obs[obs_col].isin(IDs), obs_col], legend=legend, edgecolor='none', palette='plasma')

        if legend is not None:
            plt.legend(bbox_to_anchor=(1,1,0.2,0.2), loc='lower left', frameon=False, fontsize='small')

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight', dpi=1000)
    

    def plot_centroids(self, adata, use_rep, obs_col, ctr_size=300, pt_size=75, legend=None, save_to=None):
        '''
        general plotting function for dimensionality reduction outputs with cute arrows and labels
            adata = anndata object to pull dimensionality reduction from
            use_rep = adata.obsm key to plot from (i.e. 'X_pca')
            obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
            pt_size = size of points in plot
            legend = None, 'full', or 'brief'
            save_to = path to .png file to save output, or None
        '''
        points = adata.obsm[use_rep]
        centroids = adata.uns['{}_centroids'.format(use_rep)]

        sns.scatterplot(points[:,0], points[:,1], ax=self.ax, s=pt_size, alpha=0.1, color='gray', legend=False, edgecolor='none')
        sns.scatterplot(centroids[:,0], centroids[:,1], ax=self.ax, s=ctr_size, alpha=0.7, hue=sorted(adata.obs[obs_col].unique()), legend=legend, edgecolor='none', palette='plasma')

        if legend is not None:
            plt.legend(bbox_to_anchor=(1,1,0.2,0.2), loc='lower left', frameon=False, fontsize='small')

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight', dpi=1000)
