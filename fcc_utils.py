# furry-couscous utility functions

# @author: C Heiser
# November 2018

# packages for reading in data files
import h5py
# basics
import numpy as np
import pandas as pd
import scipy as sc
# scikit packages
from skbio.stats.distance import mantel      	# Mantel test for correlation of Euclidean distance matrices
from sklearn.metrics import silhouette_score 	# silhouette score for clustering
# density peak clustering
from pydpc import Cluster                    	# density-peak clustering
# plotting packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')


def read_hdf5(filename):
		'''read in all replicates in an .hdf5 file'''
		hdf5in = h5py.File(filename, 'r')
		hdf5out = {} # initialize empty dictionary
		for key in list(hdf5in.keys()):
			hdf5out.update({key:hdf5in[key].value})
			
		hdf5in.close()
		return hdf5out


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
		plt.figtext(0.99, 0.18, 'R: {}\np-val: {}\nn: {}'.format(round(mantel_stats[0],4), mantel_stats[1], mantel_stats[2]), fontsize=14)
		plt.figtext(0.61, 0.18, 'EMD: {}\n\nKLD: {}'.format(round(EMD,4), round(KLD,4)), fontsize=14)
		plt.title('Normalized Distance Correlation', fontsize=16)
		plt.xlabel('Pre-Transformation', fontsize=14)
		plt.ylabel('Post-Transformation', fontsize=14)
		plt.tick_params(labelleft=False, labelbottom=False)
		
		sns.despine()
		plt.tight_layout()
		plt.show()

	return mantel_stats, EMD, KLD

