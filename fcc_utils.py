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
def distance_stats(pre, post, downsample=False):
    '''
    test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm.
    1) performs Pearson correlation of distance distributions
    2) normalizes unique distances using min-max standardization for each dataset
    3) calculates Wasserstein or Earth-Mover's Distance for normalized distance distributions between datasets
        pre = vector of unique distances (pdist()) or distance matrix of shape (n_cells, m_cells) (cdist()) before transformation/projection
        post = vector of unique distances (pdist()) distance matrix of shape (n_cells, m_cells) (cdist()) after transformation/projection
        downsample = number of distances to downsample to (maximum of 50M [~10k cells, if symmetrical] is recommended for performance)
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape == post.shape , 'Matrices contain different number of distances.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if pre.ndim==2:
        pre = pre.flatten()
        post = post.flatten()

    # if dataset is large, randomly downsample to reasonable number of cells for calculation
    if downsample:
        assert downsample < len(pre), 'Must provide downsample value smaller than total number of cell-cell distances provided in pre and post'
        idx = np.random.choice(np.arange(len(pre)), downsample, replace=False)
        pre = pre[idx]
        post = post[idx]

    # calculate correlation coefficient using Pearson correlation
    corr_stats = pearsonr(x=pre, y=post)

    # min-max normalization for fair comparison of probability distributions
    pre -= pre.min()
    pre /= pre.ptp()

    post -= post.min()
    post /= post.ptp()

    # calculate EMD for the distance matrices
    EMD = wasserstein_1d(pre, post)

    return pre, post, corr_stats, EMD


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
    dist_0_1 = pdist(pre[adata.obs[obs_col]==IDs[0]], pre[adata.obs[obs_col]==IDs[1]])
    dist_0_2 = pdist(pre[adata.obs[obs_col]==IDs[0]], pre[adata.obs[obs_col]==IDs[2]])
    dist_1_2 = pdist(pre[adata.obs[obs_col]==IDs[1]], pre[adata.obs[obs_col]==IDs[2]])
    # combine and min-max normalize
    dist = np.append(np.append(dist_0_1,dist_0_2), dist_1_2)
    dist -= dist.min()
    dist /= dist.ptp()
    # split normalized distances by cluster pair
    dist_norm_0_1 = dist[:dist_0_1.shape[0]]
    dist_norm_0_2 = dist[dist_0_1.shape[0]:dist_0_1.shape[0]+dist_0_2.shape[0]]
    dist_norm_1_2 = dist[dist_0_1.shape[0]+dist_0_2.shape[0]:]

    # distance calculations for post_obj
    post_0_1 = pdist(post[adata.obs[obs_col]==IDs[0]], post[adata.obs[obs_col]==IDs[1]])
    post_0_2 = pdist(post[adata.obs[obs_col]==IDs[0]], post[adata.obs[obs_col]==IDs[2]])
    post_1_2 = pdist(post[adata.obs[obs_col]==IDs[1]], post[adata.obs[obs_col]==IDs[2]])
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

    return EMD, corr_stats
