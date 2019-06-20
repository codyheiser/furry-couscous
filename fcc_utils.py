# utility functions

# @author: C Heiser
# June 2019

# basics
import numpy as np
import pandas as pd
import scipy as sc
# scikit packages
from sklearn.preprocessing import normalize
from skbio.stats.distance import mantel      			# Mantel test for correlation of symmetric distance matrices
# plotting packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')


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
		corr_stats = sc.stats.pearsonr(x=pre_flat, y=post_flat)

	# normalize flattened distances by z-score within each set for fair comparison of probability distributions
	pre_flat_norm = (pre_flat-pre_flat.min())/(pre_flat.max()-pre_flat.min())
	post_flat_norm = (post_flat-post_flat.min())/(post_flat.max()-post_flat.min())

	# calculate EMD for the distance matrices
	EMD = sc.stats.wasserstein_distance(pre_flat_norm, post_flat_norm)

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
