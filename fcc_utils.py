# utility functions

# @author: C Heiser
# September 2019

# basics
import numpy as np
import pandas as pd
import scipy
import scanpy as sc
# scikit packages
from skbio.stats.distance import mantel					# Mantel test for correlation of symmetric distance matrices
# plotting packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')



# scanpy functions
def reorder_adata(adata, descending = True):
    '''place cells in descending order of total counts'''
    if(descending==True):
        new_order = np.argsort(adata.X.sum(axis=1))[::-1]
    elif(descending==False):
        new_order = np.argsort(adata.X.sum(axis=1))[:]
    adata.X = adata.X[new_order,:].copy()
    adata.obs = adata.obs.iloc[new_order].copy()


def gf_icf(adata):
    '''return GF-ICF scores for each element in anndata counts matrix'''
    tf = adata.X.T / adata.X.sum(axis=1)
    tf = tf.T
    
    ni = adata.X.astype(bool).sum(axis=0)
    idf = np.log(adata.n_obs / (ni+1))
    
    adata.layers['gf-icf'] = tf*idf


def recipe_fcc(adata, mito_names='MT-'):
    '''
    scanpy preprocessing recipe
        adata = AnnData object with raw counts data in .X 
        mito_names = substring encompassing mitochondrial gene names for calculation of mito expression

    -calculates useful .obs and .var columns ('total_counts', 'pct_counts_mito', 'n_genes_by_counts', etc.)
    -orders cells by total counts
    -stores raw counts (adata.raw.X)
    -provides GF-ICF normalization (adata.layers['gf-icf'])
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
    gf_icf(adata) # add gf-icf scores to adata.layers['gf-icf']

    # normalize/transform
    sc.pp.normalize_total(adata, target_sum=10000, layers='all', layer_norm='after', key_added='norm_factor')
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


def plot_DR(data, color, pt_size=75, dim_name='dim', figsize=(5,5), legend=None, save_to=None):
    '''general plotting function for dimensionality reduction outputs with cute arrows and labels'''
    _, ax = plt.subplots(1, figsize=figsize)
    sns.scatterplot(data[:,0], data[:,1], s=pt_size, alpha=0.7, hue=color, legend=legend, edgecolor='none')

    plt.xlabel('{} 1'.format(dim_name), fontsize=14)
    ax.xaxis.set_label_coords(0.2, -0.025)
    plt.ylabel('{} 2'.format(dim_name), fontsize=14)
    ax.yaxis.set_label_coords(-0.025, 0.2)

    plt.annotate('', textcoords='axes fraction', xycoords='axes fraction', xy=(-0.006,0), xytext=(0.2,0), arrowprops=dict(arrowstyle= '<-', lw=2, color='black'))
    plt.annotate('', textcoords='axes fraction', xycoords='axes fraction', xy=(0,-0.006), xytext=(0,0.2), arrowprops=dict(arrowstyle= '<-', lw=2, color='black'))

    plt.tick_params(labelbottom=False, labelleft=False)
    sns.despine(left=True, bottom=True)
    if legend is not None:
        plt.legend(bbox_to_anchor=(1,1,0.2,0.2), loc='lower left', frameon=False, fontsize='small')
    plt.tight_layout()

    if save_to is None:
        plt.show()
    else:
        plt.savefig(fname=save_to, transparent=True, bbox_inches='tight', dpi=1000)
        
    plt.close()


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
    Test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm.
    1) performs Mantel (for symmetrical matrices) or Pearson test for correlation of distance matrices
    2) normalizes unique distances (upper triangle of distance matrix) using z-score for each dataset
    3) calculates Wasserstein or Earth-Mover's Distance for normalized euclidean distance distributions between datasets
        pre = distance matrix of shape (n_cells, n_cells) before transformation/projection
        post = distance matrix of shape (n_cells, n_cells) after transformation/projection
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape == post.shape , 'Matrices contain different number of distances.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])

    if pre.shape[0]==pre.shape[1] and np.allclose(pre,pre.T): # test for matrix symmetry
        # calculate correlation coefficient and p-value for distance matrices using Mantel test
        corr_stats = mantel(x=pre, y=post)

        # for each matrix, take the upper triangle (it's symmetrical) for calculating EMD and plotting distance differences
        pre_flat = pre[np.triu_indices(pre.shape[1],1)]
        post_flat = post[np.triu_indices(post.shape[1],1)]

    else:
        pre_flat = pre.flatten()
        post_flat = post.flatten()

        # calculate correlation coefficient using Pearson correlation
        corr_stats = scipy.stats.pearsonr(x=pre_flat, y=post_flat)

    # normalize flattened distances by z-score within each set for fair comparison of probability distributions
    pre_flat_norm = (pre_flat-pre_flat.min())/(pre_flat.max()-pre_flat.min())
    post_flat_norm = (post_flat-post_flat.min())/(post_flat.max()-post_flat.min())

    # calculate EMD for the distance matrices
    EMD = scipy.stats.wasserstein_distance(pre_flat_norm, post_flat_norm)

    return pre_flat_norm, post_flat_norm, corr_stats, EMD


def plot_cell_distances(pre_norm, post_norm, save_to=None):
    '''
    plot all unique cell-cell distances before and after some transformation. Executes matplotlib.pyplot.plot(), does not initialize figure.
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
    plot probability distributions for all unique cell-cell distances before and after some transformation. Executes matplotlib.pyplot.plot(), does not initialize figure.
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
    plot cumulative probability distributions for all unique cell-cell distances before and after some transformation. Executes matplotlib.pyplot.plot(), does not initialize figure.
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
    plot correlation of all unique cell-cell distances before and after some transformation. Executes matplotlib.pyplot.plot(), does not initialize figure.
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
    Test for K-nearest neighbor preservation (%) before and after transformation by a function or DR algorithm.
        pre = Knn graph of shape (n_cells, n_cells) before transformation/projection
        post = Knn graph of shape (n_cells, n_cells) after transformation/projection
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape == post.shape , 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])
    return np.round(((pre == post).sum()/pre.shape[0]**2)*100, 4)


def cluster_arrangement(pre_obj, post_obj, pre_type, post_type, clusters, cluster_names, figsize=(6,6), pre_transform='arcsinh', legend=True):
    '''
    pre_obj = RNA_counts object
    post_obj = DR object
    pre_type =
    post_type =
    clusters = list of barcode IDs i.e. ['0','1','2'] to calculate pairwise distances between clusters 0, 1 and 2
    cluster_names = list of cluster names for labeling i.e. ['Bipolar Cells','Rods','Amacrine Cells'] for clusters 0, 1 and 2, respectively
    figsize = size of output figure to plot
    pre_transform = apply transformation to pre_obj counts? (None, 'arcsinh', or 'log2')
    legend = show legend on plot
    '''
    # distance calculations for pre_obj
    dist_0_1 = pre_obj.barcode_distance_matrix(data_type=pre_type, ranks=[clusters[0],clusters[1]], transform=pre_transform).flatten()
    dist_0_2 = pre_obj.barcode_distance_matrix(data_type=pre_type, ranks=[clusters[0],clusters[2]], transform=pre_transform).flatten()
    dist_1_2 = pre_obj.barcode_distance_matrix(data_type=pre_type, ranks=[clusters[1],clusters[2]], transform=pre_transform).flatten()
    dist = np.append(np.append(dist_0_1,dist_0_2), dist_1_2)
    dist_norm = (dist-dist.min())/(dist.max()-dist.min())
    dist_norm_0_1 = dist_norm[:dist_0_1.shape[0]]
    dist_norm_0_2 = dist_norm[dist_0_1.shape[0]:dist_0_1.shape[0]+dist_0_2.shape[0]]
    dist_norm_1_2 = dist_norm[dist_0_1.shape[0]+dist_0_2.shape[0]:]

    # distance calculations for post_obj
    post_0_1 = post_obj.barcode_distance_matrix(data_type=post_type, ranks=[clusters[0],clusters[1]]).flatten()
    post_0_2 = post_obj.barcode_distance_matrix(data_type=post_type, ranks=[clusters[0],clusters[2]]).flatten()
    post_1_2 = post_obj.barcode_distance_matrix(data_type=post_type, ranks=[clusters[1],clusters[2]]).flatten()
    post = np.append(np.append(post_0_1,post_0_2), post_1_2)
    post_norm = (post-post.min())/(post.max()-post.min())
    post_norm_0_1 = post_norm[:post_0_1.shape[0]]
    post_norm_0_2 = post_norm[post_0_1.shape[0]:post_0_1.shape[0]+post_0_2.shape[0]]
    post_norm_1_2 = post_norm[post_0_1.shape[0]+post_0_2.shape[0]:]

    # calculate EMD and Pearson correlation stats
    EMD = [scipy.stats.wasserstein_distance(dist_norm_0_1, post_norm_0_1), scipy.stats.wasserstein_distance(dist_norm_0_2, post_norm_0_2), scipy.stats.wasserstein_distance(dist_norm_1_2, post_norm_1_2)]
    corr_stats = [scipy.stats.pearsonr(x=dist_0_1, y=post_0_1)[0], scipy.stats.pearsonr(x=dist_0_2, y=post_0_2)[0], scipy.stats.pearsonr(x=dist_1_2, y=post_1_2)[0]]

    # generate jointplot
    g = sns.JointGrid(x=dist_norm, y=post_norm, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(dist_norm_0_1, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkorange', label=cluster_names[0]+' - '+cluster_names[1], legend=legend)
    sns.kdeplot(dist_norm_0_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkgreen', label=cluster_names[0]+' - '+cluster_names[2], legend=legend)
    sns.kdeplot(dist_norm_1_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkred', label=cluster_names[1]+' - '+cluster_names[2], legend=legend)
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


def cluster_arrangement_general(pre, post, cluster_names, figsize=(6,6), legend=True):
    '''
    pre = list of three (3) matrices
    post = list of three (3) matrices
    cluster_names = list of cluster names for labeling i.e. ['Bipolar Cells','Rods','Amacrine Cells'] for clusters 0, 1 and 2, respectively
    figsize = size of output figure to plot
    pre_transform = apply transformation to pre_obj counts? (None, 'arcsinh', or 'log2')
    legend = show legend on plot
    '''
    # distance calculations for pre_obj
    dist_0_1 = scipy.spatial.distance_matrix(pre[0], pre[1]).flatten()
    dist_0_2 = scipy.spatial.distance_matrix(pre[0], pre[2]).flatten()
    dist_1_2 = scipy.spatial.distance_matrix(pre[1], pre[2]).flatten()
    dist = np.append(np.append(dist_0_1,dist_0_2), dist_1_2)
    dist_norm = (dist-dist.min())/(dist.max()-dist.min())
    dist_norm_0_1 = dist_norm[:dist_0_1.shape[0]]
    dist_norm_0_2 = dist_norm[dist_0_1.shape[0]:dist_0_1.shape[0]+dist_0_2.shape[0]]
    dist_norm_1_2 = dist_norm[dist_0_1.shape[0]+dist_0_2.shape[0]:]

    # distance calculations for post_obj
    post_0_1 = scipy.spatial.distance_matrix(pre[0], pre[1]).flatten()
    post_0_2 = scipy.spatial.distance_matrix(pre[0], pre[2]).flatten()
    post_1_2 = scipy.spatial.distance_matrix(pre[1], pre[2]).flatten()
    post = np.append(np.append(post_0_1,post_0_2), post_1_2)
    post_norm = (post-post.min())/(post.max()-post.min())
    post_norm_0_1 = post_norm[:post_0_1.shape[0]]
    post_norm_0_2 = post_norm[post_0_1.shape[0]:post_0_1.shape[0]+post_0_2.shape[0]]
    post_norm_1_2 = post_norm[post_0_1.shape[0]+post_0_2.shape[0]:]

    # calculate EMD and Pearson correlation stats
    EMD = [scipy.stats.wasserstein_distance(dist_norm_0_1, post_norm_0_1), scipy.stats.wasserstein_distance(dist_norm_0_2, post_norm_0_2), scipy.stats.wasserstein_distance(dist_norm_1_2, post_norm_1_2)]
    corr_stats = [scipy.stats.pearsonr(x=dist_0_1, y=post_0_1)[0], scipy.stats.pearsonr(x=dist_0_2, y=post_0_2)[0], scipy.stats.pearsonr(x=dist_1_2, y=post_1_2)[0]]

    # generate jointplot
    g = sns.JointGrid(x=dist_norm, y=post_norm, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(dist_norm_0_1, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkorange', label=cluster_names[0]+' - '+cluster_names[1], legend=legend)
    sns.kdeplot(dist_norm_0_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkgreen', label=cluster_names[0]+' - '+cluster_names[2], legend=legend)
    sns.kdeplot(dist_norm_1_2, shade=False, bw=0.01, ax=g.ax_marg_x,  color='darkred', label=cluster_names[1]+' - '+cluster_names[2], legend=legend)
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
