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
# distance metric functions
from scipy.stats import pearsonr                    # correlation coefficient
from scipy.spatial.distance import pdist            # unique pairwise distances
from ot import wasserstein_1d                       # POT implementation of Wasserstein distance between 1D arrays
# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')


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
def distance_stats(pre, post, downsample=False, verbose=True):
    '''
    test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm.
    1) performs Pearson correlation of distance distributions
    2) normalizes unique distances using min-max standardization for each dataset
    3) calculates Wasserstein or Earth-Mover's Distance for normalized distance distributions between datasets
        pre = vector of unique distances (pdist()) or distance matrix of shape (n_cells, m_cells) (cdist()) before transformation/projection
        post = vector of unique distances (pdist()) distance matrix of shape (n_cells, m_cells) (cdist()) after transformation/projection
        downsample = number of distances to downsample to (maximum of 50M [~10k cells, if symmetrical] is recommended for performance)
        verbose = print progress statements
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape == post.shape , 'Matrices contain different number of distances.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if pre.ndim==2:
        if verbose:
            print('Flattening distance matrices into 1D array of unique cell-cell distances...')
        pre = pre.flatten()
        post = post.flatten()

    # if dataset is large, randomly downsample to reasonable number of cells for calculation
    if downsample:
        assert downsample < len(pre), 'Must provide downsample value smaller than total number of cell-cell distances provided in pre and post'
        if verbose:
            print('Downsampling to {} total cell-cell distances...'.format(downsample))
        idx = np.random.choice(np.arange(len(pre)), downsample, replace=False)
        pre = pre[idx]
        post = post[idx]

    # calculate correlation coefficient using Pearson correlation
    if verbose:
        print('Correlating distances')
    corr_stats = pearsonr(x=pre, y=post)

    # min-max normalization for fair comparison of probability distributions
    if verbose:
        print('Normalizing unique distances')
    pre -= pre.min()
    pre /= pre.ptp()

    post -= post.min()
    post /= post.ptp()

    # calculate EMD for the distance matrices
    # by default, downsample to 50M distances to speed processing time, since this function often breaks with larger distributions
    if verbose:
        print("Calculating Earth-Mover's Distance between distributions")
    if len(pre) > 50000000:
        idx = np.random.choice(np.arange(len(pre)), 50000000, replace=False)
        pre_EMD = pre[idx]
        post_EMD = post[idx]
        EMD = wasserstein_1d(pre_EMD, post_EMD)
    else:
        EMD = wasserstein_1d(pre, post)

    return pre, post, corr_stats, EMD


def knn_preservation(pre, post):
    '''
    test for k-nearest neighbor preservation (%) before and after transformation by a function or DR algorithm.
        pre = Knn graph of shape (n_cells, n_cells) before transformation/projection
        post = Knn graph of shape (n_cells, n_cells) after transformation/projection
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape == post.shape , 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])
    return np.round(100 - ((pre != post).sum()/(pre.shape[0]**2))*100, 4)


def structure_preservation_sc(adata, latent, native='X', k=30, downsample=False, verbose=True):
    '''
    wrapper function for full structural preservation workflow applied to scanpy AnnData object
        adata = AnnData object with latent space to test in .obsm slot, and native (reference) space in .X or .obsm
        latent = adata.obsm key that contains low-dimensional latent space for testing
        native = adata.obsm key or .X containing high-dimensional native space, which should be direct input to dimension reduction
                 that generated latent .obsm for fair comparison. Default 'X', which uses adata.X.
        k = number of nearest neighbors to test preservation
        downsample = number of distances to downsample to (maximum of 50M [~10k cells, if symmetrical] is recommended for performance)
        verbose = print progress statements
    '''
    # 0) determine native space according to argument
    if native == 'X':
        native_space = adata.X.copy()
    else:
        native_space = adata.obsm[native].copy()
    
    # 1) calculate unique cell-cell distances
    if '{}_distances'.format(native) not in adata.uns.keys(): # check for existence in AnnData to prevent re-work
        if verbose:
            print('Calculating unique distances for native space, {}'.format(native))
        adata.uns['{}_distances'.format(native)] = pdist(native_space)
    
    if '{}_distances'.format(latent) not in adata.uns.keys(): # check for existence in AnnData to prevent re-work
        if verbose:
            print('Calculating unique distances for latent space, {}'.format(latent))
        adata.uns['{}_distances'.format(latent)] = pdist(adata.obsm[latent])
    
    # 2) get correlation and EMD values, and return normalized distance vectors for plotting distributions
    adata.uns['{}_norm_distances'.format(native)], adata.uns['{}_norm_distances'.format(latent)], corr_stats, EMD = distance_stats(pre=adata.uns['{}_distances'.format(native)], post=adata.uns['{}_distances'.format(latent)], verbose=verbose, downsample=downsample)

    # 3) determine neighbors
    if '{}_neighbors'.format(native) not in adata.uns.keys(): # check for existence in AnnData to prevent re-work
        if verbose:
            print('k-nearest neighbor calculation for native space, {}'.format(native))
        adata.uns['{}_neighbors'.format(native)] = sc.pp.neighbors(adata, n_neighbors=k, use_rep=native, knn=True, metric='euclidean', copy=True).uns['neighbors']
    
    if '{}_neighbors'.format(latent) not in adata.uns.keys(): # check for existence in AnnData to prevent re-work
        if verbose:
            print('k-nearest neighbor calculation for latent space, {}'.format(latent))
        adata.uns['{}_neighbors'.format(latent)] = sc.pp.neighbors(adata, n_neighbors=k, use_rep=latent, knn=True, metric='euclidean', copy=True).uns['neighbors']

    # 4) calculate neighbor preservation
    if verbose:
        print('Determining nearest neighbor preservation')
    knn_pres = knn_preservation(pre=adata.uns['{}_neighbors'.format(native)]['distances'], post=adata.uns['{}_neighbors'.format(latent)]['distances'])

    if verbose:
        print('Done!')
    return corr_stats, EMD, knn_pres


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


def cluster_arrangement_sc(adata, pre, post, obs_col, IDs, ID_names=None, figsize=(6,6), legend=True):
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
    dist_0_1 = pdist(pre[adata.obs[obs_col]==IDs[0]])
    dist_0_2 = pdist(pre[adata.obs[obs_col]==IDs[0]])
    dist_1_2 = pdist(pre[adata.obs[obs_col]==IDs[1]])
    # combine and min-max normalize
    dist = np.append(np.append(dist_0_1,dist_0_2), dist_1_2)
    dist -= dist.min()
    dist /= dist.ptp()
    # split normalized distances by cluster pair
    dist_norm_0_1 = dist[:dist_0_1.shape[0]]
    dist_norm_0_2 = dist[dist_0_1.shape[0]:dist_0_1.shape[0]+dist_0_2.shape[0]]
    dist_norm_1_2 = dist[dist_0_1.shape[0]+dist_0_2.shape[0]:]

    # distance calculations for post_obj
    post_0_1 = pdist(post[adata.obs[obs_col]==IDs[0]])
    post_0_2 = pdist(post[adata.obs[obs_col]==IDs[0]])
    post_1_2 = pdist(post[adata.obs[obs_col]==IDs[1]])
    # combine and min-max normalize
    post = np.append(np.append(post_0_1,post_0_2), post_1_2)
    post -= post.min()
    post /= post.ptp()
    # split normalized distances by cluster pair
    post_norm_0_1 = post[:post_0_1.shape[0]]
    post_norm_0_2 = post[post_0_1.shape[0]:post_0_1.shape[0]+post_0_2.shape[0]]
    post_norm_1_2 = post[post_0_1.shape[0]+post_0_2.shape[0]:]

    # calculate EMD and Pearson correlation stats
    EMD = [wasserstein_1d(dist_norm_0_1, post_norm_0_1), wasserstein_1d(dist_norm_0_2, post_norm_0_2), wasserstein_1d(dist_norm_1_2, post_norm_1_2)]
    corr_stats = [pearsonr(x=dist_0_1, y=post_0_1)[0], pearsonr(x=dist_0_2, y=post_0_2)[0], pearsonr(x=dist_1_2, y=post_1_2)[0]]

    if ID_names is None:
        ID_names = IDs.copy()

    # generate jointplot
    g = sns.JointGrid(x=dist, y=post, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(dist_norm_0_1, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkorange', label=ID_names[0]+' - '+ID_names[1], legend=legend)
    sns.kdeplot(dist_norm_0_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkgreen', label=ID_names[0]+' - '+ID_names[2], legend=legend)
    sns.kdeplot(dist_norm_1_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkred', label=ID_names[1]+' - '+ID_names[2], legend=legend)
    if legend:
        g.ax_marg_x.legend(loc=(1.01,0.1))
    sns.kdeplot(post_norm_0_1, shade=False, bw=0.01, vertical=True,  color='darkorange', ax=g.ax_marg_y)
    sns.kdeplot(post_norm_0_2, shade=False, bw=0.01, vertical=True,  color='darkgreen', ax=g.ax_marg_y)
    sns.kdeplot(post_norm_1_2, shade=False, bw=0.01, vertical=True,  color='darkred', ax=g.ax_marg_y)
    g.ax_joint.plot(np.linspace(max(dist.min(),post.min()),1,100), np.linspace(max(dist.min(),post.min()),1,100), linestyle='dashed', color=sns.cubehelix_palette()[-1]) # plot identity line as reference for regression
    plt.xlabel('Pre-Transformation', fontsize=14)
    plt.ylabel('Post-Transformation', fontsize=14)
    plt.tick_params(labelleft=False, labelbottom=False)

    return corr_stats, EMD
