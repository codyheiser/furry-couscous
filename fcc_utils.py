# furry-couscous utility functions

# @author: C Heiser
# May 2019

# packages for reading in data files
import h5py
# basics
import numpy as np
import pandas as pd
import scipy as sc
# scikit packages
from sklearn.preprocessing import normalize
from skbio.stats.distance import mantel      			# Mantel test for correlation of Euclidean distance matrices
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


def distance_stats(pre, post):
	'''
	Test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm.
	1) performs Mantel test for correlation of distance matrices (skbio.stats.distance.mantel())
	2) normalizes unique distances (upper triangle of distance matrix) using z-score for each dataset
	3) calculates Earth-Mover's Distance and Kullback-Leibler Divergence for normalized euclidean distance distributions between datasets

		pre = distance matrix of shape (n_cells, n_cells) before transformation/projection
		post = distance matrix of shape (n_cells, n_cells) after transformation/projection
	'''
	# make sure the number of cells in each matrix is the same
	assert pre.shape == post.shape , 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])

	# calculate correlation coefficient and p-value for distance matrices using Mantel test
	mantel_stats = mantel(x=pre, y=post)

	# for each matrix, take the upper triangle (it's symmetrical) for calculating EMD and plotting distance differences
	pre_flat = pre[np.triu_indices(pre.shape[1],1)]
	post_flat = post[np.triu_indices(post.shape[1],1)]

	# normalize flattened distances by z-score within each set for fair comparison of probability distributions
	pre_flat_norm = (pre_flat-pre_flat.min())/(pre_flat.max()-pre_flat.min())
	post_flat_norm = (post_flat-post_flat.min())/(post_flat.max()-post_flat.min())

	# calculate EMD for the distance matrices
	EMD = sc.stats.wasserstein_distance(pre_flat_norm, post_flat_norm)

	# Kullback Leibler divergence
	# add very small number to avoid dividing by zero
	KLD = sc.stats.entropy(pre_flat_norm+0.00000001, post_flat_norm+0.00000001)

	return pre_flat_norm, post_flat_norm, mantel_stats, EMD, KLD


def plot_cell_distances(pre_norm, post_norm, save_to=None):
	'''
	plot all unique cell-cell distances before and after some transformation. Executes matplotlib.pyplot.plot(), does not initialize figure.
		pre_norm: flattened vector of normalized, unique cell-cell distances "pre-transformation".
			Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
		post_norm: flattened vector of normalized, unique cell-cell distances "post-transformation".
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
		pre_norm: flattened vector of normalized, unique cell-cell distances "pre-transformation".
			Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
		post_norm: flattened vector of normalized, unique cell-cell distances "post-transformation".
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
		pre_norm: flattened vector of normalized, unique cell-cell distances "pre-transformation".
			Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
		post_norm: flattened vector of normalized, unique cell-cell distances "post-transformation".
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
	nbins = 12
	n, _ = np.histogram(pre_norm, bins=nbins)
	sy, _ = np.histogram(pre_norm, bins=nbins, weights=post_norm)
	sy2, _ = np.histogram(pre_norm, bins=nbins, weights=post_norm*post_norm)
	mean = sy / n
	std = np.sqrt(sy2/n - mean*mean)
	plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, elinewidth=2, color=sns.cubehelix_palette()[-1], linestyle='none', marker='o') # plot SD errorbars
	plt.plot(np.linspace(max(min(pre_norm),min(post_norm)),1,100), np.linspace(max(min(pre_norm),min(post_norm)),1,100), linestyle='dashed', color=sns.cubehelix_palette()[-1]) # plot identity line as reference for regression
	plt.xlabel('Pre-Transformation', fontsize=14)
	plt.ylabel('Post-Transformation', fontsize=14)
	plt.tick_params(labelleft=False, labelbottom=False)
	sns.despine()


def joint_plot_distance_correlation(pre_norm, post_norm):
	'''
	plot correlation of all unique cell-cell distances before and after some transformation. Includes marginal plots of each distribution.
	Executes matplotlib.pyplot.plot(), does not initialize figure.
		pre_norm: flattened vector of normalized, unique cell-cell distances "pre-transformation".
			Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
		post_norm: flattened vector of normalized, unique cell-cell distances "post-transformation".
			Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
	'''
	g = sns.JointGrid(x=pre_norm, y=post_norm, space=0)
	g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
	sns.kdeplot(pre_norm, color=sns.cubehelix_palette()[-1], shade=False, bw=0.01, ax=g.ax_marg_x)
	sns.kdeplot(post_norm, color=sns.cubehelix_palette()[-1], shade=False, bw=0.01, vertical=True, ax=g.ax_marg_y)
	nbins = 12
	n, _ = np.histogram(pre_norm, bins=nbins)
	sy, _ = np.histogram(pre_norm, bins=nbins, weights=post_norm)
	sy2, _ = np.histogram(pre_norm, bins=nbins, weights=post_norm*post_norm)
	mean = sy / n
	std = np.sqrt(sy2/n - mean*mean)
	g.ax_joint.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, elinewidth=2, color=sns.cubehelix_palette()[-1], linestyle='none', marker='o') # plot SD errorbars
	g.ax_joint.plot(np.linspace(max(min(pre_norm),min(post_norm)),1,100), np.linspace(max(min(pre_norm),min(post_norm)),1,100), linestyle='dashed', color=sns.cubehelix_palette()[-1]) # plot identity line as reference for regression
	plt.xlabel('Pre-Transformation', fontsize=14)
	plt.ylabel('Post-Transformation', fontsize=14)
	plt.tick_params(labelleft=False, labelbottom=False)


def compare_euclid(pre, post, plot_out=True):
	'''
	wrapper function for performing Mantel test, EMD, KLD, and plotting outputs

		pre = distance matrix of shape (n_cells, n_cells) before transformation/projection
		post = distance matrix of shape (n_cells, n_cells) after transformation/projection
		plot_out = print plots as well as return stats?
	'''
	pre_flat_norm, post_flat_norm, mantel_stats, EMD, KLD = distance_stats(pre, post)

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
		plt.figtext(0.99, 0.15, 'R: {}\nn: {}'.format(round(mantel_stats[0],4), mantel_stats[2]), fontsize=14)
		plt.figtext(0.60, 0.15, 'EMD: {}\nKLD: {}'.format(round(EMD,4), round(KLD,4)), fontsize=14)

		plt.tight_layout()
		plt.show()

	return mantel_stats, EMD, KLD


def knn_preservation(pre, post):
	'''
	Test for K-nearest neighbor preservation (%) before and after transformation by a function or DR algorithm.

		pre = Knn graph of shape (n_cells, n_cells) before transformation/projection
		post = Knn graph of shape (n_cells, n_cells) after transformation/projection
	'''
	# make sure the number of cells in each matrix is the same
	assert pre.shape == post.shape , 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(pre.shape[0], post.shape[0])
	return np.round(((pre == post).sum()/pre.shape[0]**2)*100, 4)
