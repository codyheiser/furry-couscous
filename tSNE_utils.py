# t-SNE generation utility functions

# @author: C Heiser
# November 2018

import numpy as np
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

