# -*- coding: utf-8 -*-
'''
utility functions

@author: C Heiser
October 2019
'''
# basics
import numpy as np
import pandas as pd
import scanpy as sc
# scipy functions
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, cdist
# scikit packages
from sklearn.neighbors import kneighbors_graph          # simple K-nearest neighbors graph
from skbio.stats.distance import mantel					# Mantel test for correlation of symmetric distance matrices
# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')


# scanpy functions
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
    adata.uns['{}_centroid_distances'.format(use_rep)] = distance_matrix(adata.uns['{}_centroids'.format(use_rep)], adata.uns['{}_centroids'.format(use_rep)])
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



# fuzzy-lasagna functions
def threshold(mat, thresh=0.5, dir='above'):
    '''replace all values in a matrix above or below a given threshold with np.nan'''
    a = np.ma.array(mat, copy=True)
    mask=np.zeros(a.shape, dtype=bool)
    if dir=='above':
        mask |= (a > thresh).filled(False)

    elif dir=='below':
        mask |= (a < thresh).filled(False)

    else:
        raise ValueError("Choose 'above' or 'below' for threshold direction (dir)")

    a[~mask] = np.nan
    return a


def bin_threshold(mat, threshmin=None, threshmax=0.5):
    '''
    generate binary segmentation from probabilities
        thresmax = value on [0,1] to assign binary IDs from probabilities. values higher than threshmax -> 1.
            values lower than thresmax -> 0.
    '''
    a = np.ma.array(mat, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a < threshmin).filled(False)

    if threshmax is not None:
        mask |= (a > threshmax).filled(False)

    a[mask] = 1
    a[~mask] = 0
    return a


# furry-couscous manuscript functions
def distance_stats(pre, post):
    '''
    test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm.
    1) performs Pearson correlation of distance distributions
    2) normalizes unique distances using z-score for each dataset
    3) calculates Wasserstein or Earth-Mover's Distance for normalized distance distributions between datasets
        pre = vector of unique distances (pdist()) or distance matrix of shape (n_cells, m_cells) (cdist()) before transformation/projection
        post = vector of unique distances (pdist()) distance matrix of shape (n_cells, m_cells) (cdist()) after transformation/projection
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape == post.shape , 'Matrices contain different number of distances.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if pre.ndim==2:
        pre = pre.flatten()
        post = post.flatten()

    # calculate correlation coefficient using Pearson correlation
    corr_stats = pearsonr(x=pre, y=post)

    # normalize flattened distances by z-score within each set for fair comparison of probability distributions
    pre_norm = (pre-pre.min())/(pre.max()-pre.min())
    post_norm = (post-post.min())/(post.max()-post.min())

    # calculate EMD for the distance matrices
    EMD = wasserstein_distance(pre_norm, post_norm)

    return pre_norm, post_norm, corr_stats, EMD


def plot_cell_distances(pre_norm, post_norm, save_to=None):
    '''
    plot all unique cell-cell distances before and after some transformation. 
    Executes matplotlib.pyplot.plot(), does not initialize figure.
        pre_norm = flattened vector of normalized, unique cell-cell distances "pre-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm = flattened vector of normalized, unique cell-cell distances "post-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
    '''
    plt.plot(pre_norm, alpha=0.7, label='pre', color=sns.cubehelix_palette()[-1])
    plt.plot(post_norm, alpha=0.7, label='post', color=sns.cubehelix_palette()[2])
    plt.legend(loc='best',fontsize=14)
    plt.tick_params(labelleft=False, labelbottom=False)
    sns.despine()


def plot_distributions(pre_norm, post_norm):
    '''
    plot probability distributions for all unique cell-cell distances before and after some transformation. 
    Executes matplotlib.pyplot.plot(), does not initialize figure.
        pre_norm = flattened vector of normalized, unique cell-cell distances "pre-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm = flattened vector of normalized, unique cell-cell distances "post-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
    '''
    sns.distplot(pre_norm, hist=False, kde=True, label='pre', color=sns.cubehelix_palette()[-1])
    sns.distplot(post_norm, hist=False, kde=True, label='post', color=sns.cubehelix_palette()[2])
    plt.legend(loc='best',fontsize=14)
    plt.tick_params(labelleft=False, labelbottom=False)
    sns.despine()


def plot_cumulative_distributions(pre_norm, post_norm):
    '''
    plot cumulative probability distributions for all unique cell-cell distances before and after some transformation. 
    Executes matplotlib.pyplot.plot(), does not initialize figure.
        pre_norm = flattened vector of normalized, unique cell-cell distances "pre-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm = flattened vector of normalized, unique cell-cell distances "post-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
    '''
    num_bins = int(len(pre_norm)/100)
    pre_counts, pre_bin_edges = np.histogram (pre_norm, bins=num_bins)
    pre_cdf = np.cumsum (pre_counts)
    post_counts, post_bin_edges = np.histogram (post_norm, bins=num_bins)
    post_cdf = np.cumsum (post_counts)
    plt.plot(pre_bin_edges[1:], pre_cdf/pre_cdf[-1], label='pre', color=sns.cubehelix_palette()[-1])
    plt.plot(post_bin_edges[1:], post_cdf/post_cdf[-1], label='post', color=sns.cubehelix_palette()[2])
    plt.legend(loc='best',fontsize=14)
    plt.tick_params(labelleft=False, labelbottom=False)
    sns.despine()


def plot_distance_correlation(pre_norm, post_norm):
    '''
    plot correlation of all unique cell-cell distances before and after some transformation. 
    Executes matplotlib.pyplot.plot(), does not initialize figure.
        pre_norm: flattened vector of normalized, unique cell-cell distances "pre-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm: flattened vector of normalized, unique cell-cell distances "post-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
    '''
    plt.hist2d(x=pre_norm, y=post_norm, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.plot(np.linspace(max(min(pre_norm),min(post_norm)),1,100), np.linspace(max(min(pre_norm),min(post_norm)),1,100), linestyle='dashed', color=sns.cubehelix_palette()[-1]) # plot identity line as reference for regression
    plt.xlabel('Pre-Transformation', fontsize=14)
    plt.ylabel('Post-Transformation', fontsize=14)
    plt.tick_params(labelleft=False, labelbottom=False)
    sns.despine()


def joint_plot_distance_correlation(pre_norm, post_norm, figsize=(4,4)):
    '''
    plot correlation of all unique cell-cell distances before and after some transformation. Includes marginal plots of each distribution.
    Executes matplotlib.pyplot.plot(), does not initialize figure.
        pre_norm = flattened vector of normalized, unique cell-cell distances "pre-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm = flattened vector of normalized, unique cell-cell distances "post-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
    '''
    g = sns.JointGrid(x=pre_norm, y=post_norm, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(pre_norm, color=sns.cubehelix_palette()[-1], shade=False, bw=0.01, ax=g.ax_marg_x)
    sns.kdeplot(post_norm, color=sns.cubehelix_palette()[-1], shade=False, bw=0.01, vertical=True, ax=g.ax_marg_y)
    g.ax_joint.plot(np.linspace(max(min(pre_norm),min(post_norm)),1,100), np.linspace(max(min(pre_norm),min(post_norm)),1,100), linestyle='dashed', color=sns.cubehelix_palette()[-1]) # plot identity line as reference for regression
    plt.xlabel('Pre-Transformation', fontsize=14)
    plt.ylabel('Post-Transformation', fontsize=14)
    plt.tick_params(labelleft=False, labelbottom=False)


def compare_euclid(pre, post, plot_out=True):
    '''
    wrapper function for performing Mantel/Pearson correlation, EMD, and plotting outputs
        pre = distance matrix of shape (n_cells, n_cells) before transformation/projection
        post = distance matrix of shape (n_cells, n_cells) after transformation/projection
        plot_out = print plots as well as return stats?
    '''
    pre_flat_norm, post_flat_norm, corr_stats, EMD = distance_stats(pre, post)

    if plot_out:
        plt.figure(figsize=(15,5))

        plt.subplot(131)
        plot_distributions(pre_flat_norm, post_flat_norm)
        plt.title('Distance Distribution', fontsize=16)

        plt.subplot(132)
        plot_cumulative_distributions(pre_flat_norm, post_flat_norm)
        plt.title('Cumulative Distance Distribution', fontsize=16)

        plt.subplot(133)
        # plot correlation of distances
        plot_distance_correlation(pre_flat_norm, post_flat_norm)
        plt.title('Normalized Distance Correlation', fontsize=16)

        # add statistics as plot annotations
        plt.figtext(0.99, 0.15, 'R: {}\nn: {}'.format(round(corr_stats[0],4), corr_stats[2]), fontsize=14)
        plt.figtext(0.60, 0.15, 'EMD: {}'.format(round(EMD,4)), fontsize=14)

        plt.tight_layout()
        plt.show()

    return corr_stats, EMD


def knn_preservation(pre, post):
    '''
    test for k-nearest neighbor preservation (%) before and after transformation by a function or DR algorithm.
        pre = Knn graph of shape (n_cells, n_cells) before transformation/projection
        post = Knn graph of shape (n_cells, n_cells) after transformation/projection
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape == post.shape , 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])
    return np.round(((pre == post).sum()/pre.shape[0]**2)*100, 4)


def cluster_arrangement_sc(adata, pre, post, obs_col, IDs, ID_names, figsize=(6,6), legend=True):
    '''
    determine pairwise distance preservation between 3 IDs from adata.obs[obs_col]
        adata = anndata object to pull dimensionality reduction from
        pre = matrix to subset as pre-transformation (i.e. adata.X)
        post = matrix to subset as pre-transformation (i.e. adata.obsm['X_pca'])
        obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
        IDs = list of THREE IDs to compare (i.e. [0,1,2])
        figsize = size of resulting axes
        legend = None, 'full', or 'brief'
        save_to = path to .png file to save output, or None
    '''
    # distance calculations for pre_obj
    dist_0_1 = cdist(pre[adata.obs[obs_col]==IDs[0]], pre[adata.obs[obs_col]==IDs[1]]).flatten()
    dist_0_2 = cdist(pre[adata.obs[obs_col]==IDs[0]], pre[adata.obs[obs_col]==IDs[2]]).flatten()
    dist_1_2 = cdist(pre[adata.obs[obs_col]==IDs[1]], pre[adata.obs[obs_col]==IDs[2]]).flatten()
    dist = np.append(np.append(dist_0_1,dist_0_2), dist_1_2)
    dist_norm = (dist-dist.min())/(dist.max()-dist.min())
    dist_norm_0_1 = dist_norm[:dist_0_1.shape[0]]
    dist_norm_0_2 = dist_norm[dist_0_1.shape[0]:dist_0_1.shape[0]+dist_0_2.shape[0]]
    dist_norm_1_2 = dist_norm[dist_0_1.shape[0]+dist_0_2.shape[0]:]

    # distance calculations for post_obj
    post_0_1 = cdist(post[adata.obs[obs_col]==IDs[0]], post[adata.obs[obs_col]==IDs[1]]).flatten()
    post_0_2 = cdist(post[adata.obs[obs_col]==IDs[0]], post[adata.obs[obs_col]==IDs[2]]).flatten()
    post_1_2 = cdist(post[adata.obs[obs_col]==IDs[1]], post[adata.obs[obs_col]==IDs[2]]).flatten()
    post = np.append(np.append(post_0_1,post_0_2), post_1_2)
    post_norm = (post-post.min())/(post.max()-post.min())
    post_norm_0_1 = post_norm[:post_0_1.shape[0]]
    post_norm_0_2 = post_norm[post_0_1.shape[0]:post_0_1.shape[0]+post_0_2.shape[0]]
    post_norm_1_2 = post_norm[post_0_1.shape[0]+post_0_2.shape[0]:]

    # calculate EMD and Pearson correlation stats
    EMD = [wasserstein_distance(dist_norm_0_1, post_norm_0_1), wasserstein_distance(dist_norm_0_2, post_norm_0_2), wasserstein_distance(dist_norm_1_2, post_norm_1_2)]
    corr_stats = [pearsonr(x=dist_0_1, y=post_0_1)[0], pearsonr(x=dist_0_2, y=post_0_2)[0], pearsonr(x=dist_1_2, y=post_1_2)[0]]

    # generate jointplot
    g = sns.JointGrid(x=dist_norm, y=post_norm, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(dist_norm_0_1, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkorange', label=ID_names[0]+' - '+ID_names[1], legend=legend)
    sns.kdeplot(dist_norm_0_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkgreen', label=ID_names[0]+' - '+ID_names[2], legend=legend)
    sns.kdeplot(dist_norm_1_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkred', label=ID_names[1]+' - '+ID_names[2], legend=legend)
    if legend:
        g.ax_marg_x.legend(loc=(1.01,0.1))
    sns.kdeplot(post_norm_0_1, shade=False, bw=0.01, vertical=True,  color='darkorange', ax=g.ax_marg_y)
    sns.kdeplot(post_norm_0_2, shade=False, bw=0.01, vertical=True,  color='darkgreen', ax=g.ax_marg_y)
    sns.kdeplot(post_norm_1_2, shade=False, bw=0.01, vertical=True,  color='darkred', ax=g.ax_marg_y)
    g.ax_joint.plot(np.linspace(max(min(dist_norm),min(post_norm)),1,100), np.linspace(max(min(dist_norm),min(post_norm)),1,100), linestyle='dashed', color=sns.cubehelix_palette()[-1]) # plot identity line as reference for regression
    plt.xlabel('Pre-Transformation', fontsize=14)
    plt.ylabel('Post-Transformation', fontsize=14)
    plt.tick_params(labelleft=False, labelbottom=False)

    return EMD, corr_stats
