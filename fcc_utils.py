# t-SNE generation utility functions

# @author: C Heiser
# November 2018

import numpy as np
import scipy as sc
# scikit packages
import skbio
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA        # PCA
from sklearn.manifold import TSNE            # t-SNE
# DCA packages
import scanpy.api as scanpy
from dca.api import dca                      # DCA
# plotting packages
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

def corr_distances(mat1, mat2, plot_out=True):
    '''
    Calculate correlation of two distance matrices using mantel test. 
    Returns stats and plot of distances. mat1 & mat2 are distance matrices.
    Output is list of R, p, and n values in order.
    '''
    # calculate Spearman correlation coefficient and p-value for distance matrices using Mantel test
    mantel_stats = skbio.stats.distance.mantel(x=mat1, y=mat2)
    
    if plot_out:
        # for each matrix, take the upper triangle (it's symmetrical), and remove all zeros for plotting distance differences
        mat1_flat = np.triu(mat1)[np.nonzero(np.triu(mat1))]
        mat2_flat = np.triu(mat2)[np.nonzero(np.triu(mat2))]
        
        plt.figure(figsize=(5,5))
        sns.scatterplot(mat1_flat, mat2_flat, s=75, alpha=0.6)
        plt.figtext(0.95, 0.5, 'R: {}\np-val: {}\nn: {}'.format(round(mantel_stats[0],5),mantel_stats[1],mantel_stats[2]), fontsize=12)
        plt.title('Distance Correlation')
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
    return mantel_stats

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
    '''test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm'''
    # generate distance matrices for pre- and post-transformation arrays
    dm_pre = sc.spatial.distance_matrix(pre,pre)
    dm_post = sc.spatial.distance_matrix(post,post)
    
    # calculate Spearman correlation coefficient and p-value for distance matrices using Mantel test
    mantel_stats = skbio.stats.distance.mantel(x=dm_pre, y=dm_post)
    
    # for each matrix, take the upper triangle (it's symmetrical), and remove all zeros for calculating EMD and plotting distance differences
    pre_flat = np.triu(dm_pre)[np.nonzero(np.triu(dm_pre))]
    post_flat = np.triu(dm_post)[np.nonzero(np.triu(dm_post))]
    
    # calculate EMD for the distance matrices
    EMD = sc.stats.wasserstein_distance(pre_flat.flatten(), post_flat.flatten())
    
    if plot_out:
        plt.figure(figsize=(5,5))
        sns.scatterplot(pre_flat, post_flat, s=75, alpha=0.6)
        plt.figtext(0.95, 0.5, 'R: {}\np-val: {}\nn: {}\nEMD: {}'.format(round(mantel_stats[0],5),mantel_stats[1],mantel_stats[2],round(EMD,5)), fontsize=12)
        plt.title('Distance Correlation')
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
    return mantel_stats, EMD
    
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
        sns.scatterplot(PCA_results[:,0], PCA_results[:,1], s=75)
        plt.ylabel('PC2')
        plt.xlabel('PC1')
        plt.title('PCA')
        
        plt.subplot(122)
        plt.plot(np.cumsum(np.round(PCA_fit.explained_variance_ratio_, decimals=3)*100))
        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    return PCA_results
    
def fcc_tSNE(x, plot_out=True):
    '''perform t-SNE on matrix x of shape (n_cells, n_features) to reduce to 2 features'''
    tSNE_results = TSNE(n_components=2).fit_transform(x)
    # plot t-SNE if desired
    if plot_out:
        plt.figure(figsize=(5,5))
        sns.scatterplot(tSNE_results[:,0], tSNE_results[:,1], s=75)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    return tSNE_results

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


    