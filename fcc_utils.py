# furry couscous utility functions

# @author: C Heiser
# November 2018

import numpy as np
import scipy as sc
# scikit packages
from skbio.stats.distance import mantel      	# Mantel test for correlation of Euclidean distance matrices
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA        	# PCA
from sklearn.manifold import TSNE            	# t-SNE
from sklearn.metrics import silhouette_score 	# silhouette score for clustering
# density peak clustering
from pydpc import Cluster                    	# density-peak clustering
# DCA packages
import scanpy.api as scanpy
from dca.api import dca                      	# DCA
# UMAP
import umap                                  	# UMAP
# plotting packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')
# package for reading in data files
import h5py

def read_hdf5(filename):
    '''read in all replicates in an .hdf5 file'''
    hdf5in = h5py.File(filename, 'r')
    hdf5out = {} # initialize empty dictionary
    for key in list(hdf5in.keys()):
        hdf5out.update({key:hdf5in[key].value})
        
    hdf5in.close()
    return hdf5out

def arcsinh_norm(x, norm=True, scale=1000):
    '''
    Perform an arcsinh-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
    Useful for feeding into PCA or tSNE.
        norm = convert to fractional counts first? divide each count by sqrt of sum of squares of counts for cell.
        scale = factor to multiply values by before arcsinh-transform. scales values away from [0,1] in order to make arcsinh more effective.
    '''
    if not norm:
        return np.arcsinh(x * scale)
    
    else:
        return np.arcsinh(normalize(x, axis=0, norm='l2') * scale)
    
def log2_norm(x, norm = True):
    '''
    Perform a log2-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
    Useful for feeding into PCA or tSNE.
        norm = convert to fractional counts first? divide each count by sqrt of sum of squares of counts for cell.
    '''
    if not norm:
        return np.log2(x + 1)
    
    else:
        return np.log2(normalize(x, axis=0, norm='l2') + 1)
    
def compare_euclid(pre, post, plot_out=True):
    '''
    Test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm.
    1) calculates Euclidean distance matrix (n_cells, n_cells)
    2) performs Mantel test for correlation of distance matrices (skbio.stats.distance.mantel())
    3) normalizes unique distances (upper triangle of distance matrix) to maximum for each dataset, yielding fractions in range [0, 1]
    4) calculates Earth-Mover's Distance and Kullback-Leibler Divergence for euclidean distance fractions between datasets
    5) plots fractional Euclidean distances for both datasets, cumulative probability distribution for fractional distances, and correlation of distances in one figure
    
        pre = matrix of shape (n_cells, n_features) before transformation/projection
        post = matrix of shape (n_cells, n_features) after transformation/projection
        plot_out = print plots as well as return stats?
    '''
    # make sure the number of cells in each matrix is the same
    assert pre.shape[0] == post.shape[0] , 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])
    
    # generate distance matrices for pre- and post-transformation arrays
    dm_pre = sc.spatial.distance_matrix(pre,pre)
    dm_post = sc.spatial.distance_matrix(post,post)
    
    # calculate Spearman correlation coefficient and p-value for distance matrices using Mantel test
    mantel_stats = mantel(x=dm_pre, y=dm_post)
    
    # for each matrix, take the upper triangle (it's symmetrical) for calculating EMD and plotting distance differences
    pre_flat = dm_pre[np.triu_indices(dm_pre.shape[1],1)]
    post_flat = dm_post[np.triu_indices(dm_post.shape[1],1)]
    
    # normalize flattened distances within each set for fair comparison of probability distributions
    pre_flat_norm = (pre_flat/pre_flat.max())
    post_flat_norm = (post_flat/post_flat.max())
    
    # calculate EMD for the distance matrices
    EMD = sc.stats.wasserstein_distance(pre_flat_norm, post_flat_norm)
    
    # Kullback Leibler divergence
    # add very small number to avoid dividing by zero
    KLD = sc.stats.entropy(pre_flat_norm+0.00000001, post_flat_norm+0.00000001)
    
    if plot_out:
        plt.figure(figsize=(15,5))
        
        plt.subplot(131)
        # plot unique cell-cell distances
        plt.plot(pre_flat_norm, alpha=0.5, label='pre')
        plt.plot(post_flat_norm, alpha=0.5, label='post')
        plt.title('Normalized Unique Distances', fontsize=16)
        plt.legend(loc='best',fontsize=14)
        plt.tick_params(labelsize=12, labelbottom=False)
        
        plt.subplot(132)
        # calculate and plot the cumulative probability distributions for cell-cell distances in each dataset
        num_bins = int(len(pre_flat_norm)/100)
        pre_counts, pre_bin_edges = np.histogram (pre_flat_norm, bins=num_bins)
        pre_cdf = np.cumsum (pre_counts)
        post_counts, post_bin_edges = np.histogram (post_flat_norm, bins=num_bins)
        post_cdf = np.cumsum (post_counts)
        plt.plot(pre_bin_edges[1:], pre_cdf/pre_cdf[-1], label='pre')
        plt.plot(post_bin_edges[1:], post_cdf/post_cdf[-1], label='post')
        plt.title('Cumulative Probability of Normalized Distances', fontsize=16)
        plt.legend(loc='best',fontsize=14)
        plt.tick_params(labelsize=12)
        
        plt.subplot(133)
        # plot correlation of distances
        sns.scatterplot(pre_flat_norm, post_flat_norm, s=75, alpha=0.5)
        plt.figtext(0.95, 0.5, 'R: {}\np-val: {}\nn: {}\n\nEMD: {}\nKLD: {}'.format(round(mantel_stats[0],5),mantel_stats[1],mantel_stats[2],round(EMD,4),round(KLD,4)), fontsize=14)
        plt.title('Normalized Distance Correlation', fontsize=16)
        plt.xlabel('Pre-Transformation', fontsize=14)
        plt.ylabel('Post-Transformation', fontsize=14)
        plt.tick_params(labelleft=False, labelbottom=False)
        
        sns.despine()
        plt.tight_layout()
        plt.show()
        
    return mantel_stats, EMD, KLD
    
def fcc_PCA(x, n_comp, plot_out=True):
    '''perform PCA with n_components on matrix x of shape (n_cells, n_features)'''
    # fit PCA to data
    PCA_fit = PCA(n_components=n_comp).fit(x)
    # transform data to fit
    PCA_results = PCA_fit.transform(x)
    # plot PCA if desired
    if plot_out:
        plt.figure(figsize=(10,5))
        
        plt.subplot(121)
        sns.scatterplot(PCA_results[:,0], PCA_results[:,1], s=75, alpha=0.7)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.ylabel('PC2', fontsize=14)
        plt.xlabel('PC1', fontsize=14)
        plt.title('PCA', fontsize=16)
        
        plt.subplot(122)
        plt.plot(np.cumsum(np.round(PCA_fit.explained_variance_ratio_, decimals=3)*100))
        plt.tick_params(labelsize=12)
        plt.ylabel('% Variance Explained', fontsize=14)
        plt.xlabel('# of Features', fontsize=14)
        plt.title('PCA Analysis', fontsize=16)
        
        sns.despine()
        plt.tight_layout()
        plt.show()
        plt.close()
        
    return PCA_results

def fcc_DCA(x, n_threads=2, dca_norm=True):
    '''
    Use deep count autoencoder (https://github.com/theislab/dca) to denoise and preprocess matrix x of shape (n_cells, n_features).
    NOTE: DCA removes features with 0 counts for all cells prior to processing.
    '''
    adata = scanpy.AnnData(x) # generate AnnData object (https://github.com/theislab/scanpy) for passing to DCA
    scanpy.pp.filter_genes(adata, min_counts=1) # remove features with 0 counts for all cells
    dca(adata, threads=n_threads) # perform DCA analysis on AnnData object
    
    if dca_norm:
        scanpy.pp.normalize_per_cell(adata) # normalize features for each cell with scanpy's method
        scanpy.pp.log1p(adata) # log-transform data with scanpy's method
        
    return adata.X # return the denoised data as a np.ndarray

def fcc_tSNE(x, p = 30, plot_out=True):
    '''perform t-SNE with perplexity p on matrix x of shape (n_cells, n_features) to reduce to 2 features'''
    tSNE_results = TSNE(n_components=2, perplexity=p).fit_transform(x)
    # plot t-SNE if desired
    if plot_out:
        plt.figure(figsize=(5,5))
        sns.scatterplot(tSNE_results[:,0], tSNE_results[:,1], s=75, alpha=0.7)
        plt.xlabel('t-SNE 1', fontsize=14)
        plt.ylabel('t-SNE 2', fontsize=14)
        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    return tSNE_results

def fcc_UMAP(x, p = 30, min_dist=0.3, plot_out=True):
    '''
    perform UMAP with perplexity p and minimum distance min_dist on matrix x of shape (n_cells, n_features) 
    to reduce to 2 features
    '''
    UMAP_results = umap.UMAP(n_neighbors=p, min_dist=min_dist, metric='correlation').fit_transform(x)
    # plot t-SNE if desired
    if plot_out:
        plt.figure(figsize=(5,5))
        sns.scatterplot(UMAP_results[:,0], UMAP_results[:,1], s=75, alpha=0.7)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    return UMAP_results

# WORK IN PROGRESS
def gen_clusters(x, plot_out=True):
    '''
    Use pydpc to cluster matrix x of shape (n_cells, n_features). 
    You will be asked for density and delta cutoffs to determine cluster centers.
    Returns pydpc cluster object and silhouette score for clustering.
    '''
    matplotlib.use('agg')
    # output delta vs density plot of points
    clu = Cluster(x)
    clu.draw_decision_graph()
    
    # prompt user for cutoff values
    dens = float(input('Density Cutoff Value: '))
    delt = float(input('Delta Cutoff Value: '))
    
    # assign cluster centers and output plot
    clu.assign(dens, delt)
    
    if plot_out:
        # plot clusters with point densities
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].scatter(x[:, 0], x[:, 1], s=75, alpha=0.7)
        ax[0].scatter(x[clu.clusters, 0], x[clu.clusters, 1], s=90, c="red")
        ax[1].scatter(x[:, 0], x[:, 1], s=75, c=clu.density)
        ax[2].scatter(x[:, 0], x[:, 1], s=75, c=clu.membership, cmap=plt.cm.plasma, alpha=0.7)
        for _ax in ax:
            _ax.set_aspect('equal')
            _ax.tick_params(labelbottom=False, labelleft=False)
        
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
    
    # calculate silhouette score
    if clu.membership.max()==0:
        ss = 0
    
    else:
        ss = silhouette_score(x, clu.membership)
    
    return clu, ss
