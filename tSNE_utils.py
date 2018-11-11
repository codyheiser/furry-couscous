# t-SNE generation utility functions

# @author: C Heiser
# November 2018

import numpy as np
np.seterr(divide='ignore', invalid='ignore') # allow divide by zero for normalization
# scikit packages
import skbio
# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'whitegrid')
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

def corr_distances(mat1, mat2, plot_out = True):
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

def arcsinh_norm(x, frac = True, scale = 1000):
    '''
    Perform an arcsinh-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
    Useful for feeding into PCA or tSNE.
        frac = convert to fractional counts first? divide each count by sum of counts for cell.
        scale = factor to multiply values by before arcsinh-transform. scales values away from [0,1] in order to make arcsinh more effective.
    '''
    if not frac:
        return np.arcsinh(x * scale)
    
    else:
        return np.arcsinh(np.nan_to_num(x / np.sum(x, axis=0)) * scale)
    
def log2_norm(x, frac = True):
    '''
    Perform a log2-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
    Useful for feeding into PCA or tSNE.
        frac = convert to fractional counts first? divide each count by sum of counts for cell.
    '''
    if not frac:
        return np.log2(x + 1)
    
    else:
        return np.log2(np.nan_to_num(x / np.sum(x, axis=0)) + 1)
    
    