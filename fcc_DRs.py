# furry-couscous dimensionality reduction objects

# @author: C Heiser
# January 2019

# utility functions
from fcc_utils import *
# packages for reading in data files
import os
import zipfile
import gzip
# basics
import numpy as np
import pandas as pd
import scipy as sc
# scikit packages
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA        	# PCA
from sklearn.manifold import TSNE            	# t-SNE
from sklearn.model_selection import KFold		# K-fold cross-validation
from sklearn.neighbors import kneighbors_graph	# K-nearest neighbors graph
# density peak clustering
from pydpc import Cluster                    	# density-peak clustering
# DCA packages
import scanpy.api as scanpy
from dca.api import dca                      	# DCA
# UMAP
from umap import UMAP                           # UMAP
# FIt-SNE
import sys; sys.path.append('/Users/Cody/git/FIt-SNE')	# ensure path to FIt-SNE repo is correct!
from fast_tsne import fast_tsne					# FIt-SNE
# NVR
import nvr 										# NVR
# plotting packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')


class RNA_counts():
	'''
	Object containing scRNA-seq counts data
		data = pd.DataFrame containing counts data.
		labels = list of index_col and header values to pass to pd.read_csv(). None if no cell or gene IDs, respectively.
		cells_axis = 0 if cells as rows, 1 if cells as columns.
	'''
	def __init__(self, data, labels=[0,0], cells_axis=0, barcodes=None):
		'''initialize object from np.ndarray or pd.DataFrame (data)'''
		self.data = pd.DataFrame(data) # store pd.DataFrame as data attribute

		self.cell_labels = labels[0] # column containing cell IDs
		self.gene_labels = labels[1] # row containing gene IDs

		if cells_axis == 1: # put cells on 0 axis if not already there
			self.data = self.data.transpose()

		if self.cell_labels!=None: # if cell IDs present, save as metadata
			self.cell_IDs = self.data.index

		if self.gene_labels!=None: # if gene IDs present, save as metadata
			self.gene_IDs = self.data.columns

		self.counts = np.ascontiguousarray(self.data) # store counts matrix as counts attribute (no labels, np.array format)

		if barcodes is not None: # if barcodes df provided, merge with data
			data_coded = self.data.merge(pd.DataFrame(barcodes), left_index=True, right_on='Cell Barcode', how='left')
			data_coded = data_coded.set_index('Cell Barcode', drop=True)
			data_coded = data_coded.astype({'Barcode':'category'})
			self.data_coded = data_coded # create 'coded' attribute that has data and barcodes
			self.barcodes = data_coded['Barcode'] # make barcodes attribute pd.Series for passing to other classes

		else:
			self.barcodes = None


	def distance_matrix(self, transform=None, ranks='all', **kwargs):
		'''
		calculate Euclidean distances between cells in matrix of shape (n_cells, n_cells)
			norm = how to normalize data prior to calculating distances (None, "arcsinh", "log2")
			**kwargs = keyword arguments to pass to normalization functions
		'''
		# transform data first, if necessary
		if transform is None:
			transformed = self.counts

		if transform == 'arcsinh':
			transformed = self.arcsinh_norm(**kwargs)

		elif transform == 'log2':
			transformed = self.log2_norm(**kwargs)

		# then subset data by rank-ordered barcode appearance
		if ranks=='all':
			return sc.spatial.distance_matrix(transformed, transformed)

		elif not isinstance(ranks, (list,)): # make sure input is list-formatted
			ranks = [ranks]

		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		ints = [x for x in ranks if type(x)==int] # pull out rank values
		IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
		ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
		ranks_counts = transformed[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed counts array
		return sc.spatial.distance_matrix(ranks_counts, ranks_counts)


	def knn_graph(self, k, **kwargs):
		'''
		calculate k nearest neighbors for each cell in distance matrix of shape (n_cells, n_cells)
			k = number of nearest neighbors to test
			**kwargs = keyword arguments to pass to distance_matrix() function
		'''
		# TODO: determine if distance matrix should be calculated on whole or on subset (ranks of **kwargs)
		return kneighbors_graph(self.distance_matrix(**kwargs), k, mode='connectivity', include_self=False).toarray()


	def top_barcodes(self, ranks):
		'''return list of top-ranked barcodes by prevalence in dataset'''
		assert self.barcodes is not None, 'Barcodes not assigned.\n'

		if not isinstance(ranks, (list,)): # make sure input is list-formatted
			ranks = [ranks]

		ints = [x for x in ranks if type(x)==int] # pull out rank values
		IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
		return list(self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index) + IDs


	def barcode_counts(self, IDs='all'):
		'''given list of barcode IDs, return pd.Series of number of appearances in dataset'''
		assert self.barcodes is not None, 'Barcodes not assigned.\n'

		if IDs=='all':
			return self.barcodes.value_counts()

		if not isinstance(IDs, (list,)): # make sure input is list-formatted
			IDs = [IDs]

		return self.barcodes.value_counts()[self.barcodes.value_counts().index.isin(IDs)]


	def arcsinh_norm(self, norm='l1', scale=1000, ranks='all'):
		'''
		Perform an arcsinh-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
		Useful for feeding into PCA or tSNE.
			scale = factor to multiply values by before arcsinh-transform. scales values away from [0,1] in order to make arcsinh more effective.
			ranks = which barcodes to include as list of indices or strings with barcode IDs
			norm = normalization strategy prior to Log2 transorm.
				None: do not normalize data
				'l1': divide each count by sum of counts for each cell
				'l2': divide each count by sqrt of sum of squares of counts for cell.
		'''
		if not norm:
			out = np.arcsinh(self.counts * scale)

		else:
			out = np.arcsinh(normalize(self.counts, axis=1, norm=norm) * scale)

		if ranks=='all':
			return out

		elif not isinstance(ranks, (list,)): # make sure input is list-formatted
			ranks = [ranks]

		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		ints = [x for x in ranks if type(x)==int] # pull out rank values
		IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
		ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
		return out[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed counts array


	def log2_norm(self, norm='l1', ranks='all'):
		'''
		Perform a log2-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
		Useful for feeding into PCA or tSNE.
			norm = normalization strategy prior to Log2 transorm.
				None: do not normalize data
				'l1': divide each count by sum of counts for each cell
				'l2': divide each count by sqrt of sum of squares of counts for cell.
		'''
		if not norm:
			out = np.log2(self.counts + 1)

		else:
			out = np.log2(normalize(self.counts, axis=1, norm=norm) + 1)

		if ranks=='all':
			return out

		elif not isinstance(ranks, (list,)): # make sure input is list-formatted
			ranks = [ranks]

		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		ints = [x for x in ranks if type(x)==int] # pull out rank values
		IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
		ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
		return out[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed counts array


	@classmethod
	def from_file(cls, datafile, labels=[0,0], cells_axis=0, barcodefile=None):
		'''initialize object from outside file (datafile)'''
		filetype = os.path.splitext(datafile)[1] # extract file extension to save as metadata

		if filetype == '.zip': # if compressed, open the file and update filetype
			zf = zipfile.ZipFile(datafile)
			datafile = zf.open(os.path.splitext(datafile)[0]) # update datafile with zipfile object
			filetype = os.path.splitext(os.path.splitext(datafile)[0])[1] # update filetype


		if filetype == '.csv': # read comma-delimited tables
			data = pd.read_csv(datafile, header=labels[1], index_col=labels[0])

		elif filetype == '.txt': # read tab-delimited text files
				data = pd.read_table(datafile, header=labels[1], index_col=labels[0])


		if filetype == '.gz': # if file is g-zipped, read accordingly
			filetype = os.path.splitext(os.path.splitext(datafile)[0])[1] # update filetype

			if filetype == '.csv':
				data = pd.read_csv(gzip.open(datafile), header=labels[1], index_col=labels[0])

			elif filetype == '.txt':
				data = pd.read_table(gzip.open(datafile), header=labels[1], index_col=labels[0])


		if barcodefile: # if barcodes provided, read in file
			barcodes = pd.read_csv(barcodefile, index_col=0).T

		else:
			barcodes = None


		return cls(data, labels=labels, cells_axis=cells_axis, barcodes=barcodes)


	@classmethod
	def drop_set(cls, counts_obj, drop_index, axis, num=False):
		'''
		drop cells (axis 0) or genes (axis 1) with a pd.Index list. return RNA_counts object with reduced data.
			keep_index: list of indices to keep
			axis: 0 to subset cells, 1 to subset genes
			num: numerical index (iloc) or index by labels (loc)?
		'''
		if counts_obj.barcodes is not None:
			codes = pd.DataFrame(counts_obj.barcodes)
			codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

		else:
			codes=None

		if not num:
			return cls(counts_obj.data.drop(drop_index, axis=axis), labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)

		else:
			return cls(counts_obj.data.drop(counts_obj.data.columns[drop_index], axis=axis), labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)


	@classmethod
	def keep_set(cls, counts_obj, keep_index, axis, num=False):
		'''
		keep cells (axis 0) or genes (axis 1) with a pd.Index list. return RNA_counts object with reduced data.
			keep_index: list of indices to keep
			axis: 0 to subset cells, 1 to subset genes
			num: numerical index (iloc) or index by labels (loc)?
		'''
		if counts_obj.barcodes is not None:
			codes = pd.DataFrame(counts_obj.barcodes)
			codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

		else:
			codes=None

		if not num:
			if axis==0:
				return cls(counts_obj.data.loc[keep_index,:], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)

			elif axis==1:
				return cls(counts_obj.data.loc[:,keep_index], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)

		else:
			if axis==0:
				return cls(counts_obj.data.iloc[keep_index,:], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)

			elif axis==1:
				return cls(counts_obj.data.iloc[:,keep_index], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)


	@classmethod
	def downsample_rand(cls, counts_obj, n_cells, seed=None):
		'''randomly downsample a dataframe of shape (n_cells, n_features) to n_cells and generate new counts object'''
		np.random.seed(seed) # set seed for reproducible sampling if desired
		cells_out = np.random.choice(counts_obj.data.shape[0], n_cells, replace=False)

		if counts_obj.barcodes is not None:
			codes = pd.DataFrame(counts_obj.barcodes)
			codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

		else:
			codes=None

		return cls(counts_obj.data.iloc[cells_out], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)


	@classmethod
	def downsample_proportional(cls, counts_obj, clu_membership, n_cells, seed=None):
		'''
		downsample a dataframe of shape (n_cells, n_features) to total n_cells using cluster membership.
		finds proportion of cells in each cluster (DR.clu.membership attribute) and maintains each percentage.
			counts_obj = RNA_counts object with data to downsample
			clu_membership = DR.clu.membership np.array generated from assocated RNA_counts data object
			n_cells = total number of cells desired in downsampled RNA_counts object
		'''
		np.random.seed(seed) # set seed for reproducible sampling if desired
		IDs, clu_counts = np.unique(clu_membership, return_counts=True) # get cluster IDs and number of cells in each

		cells_out = np.array([]) # initialize array of output cell indices
		for ID, count in zip(IDs, clu_counts):
			clu_num = int(count/clu_counts.sum()*n_cells) + 1 # number of cells to sample for given cluster
			cells_out = np.append(cells_out, np.random.choice(np.where(clu_membership == ID)[0], clu_num, replace=False))

		if counts_obj.barcodes is not None:
			codes = pd.DataFrame(counts_obj.barcodes)
			codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

		else:
			codes=None

		return cls(counts_obj.data.iloc[cells_out], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)


	@classmethod
	def kfold_split(cls, counts_obj, n_splits, seed=None, shuffle=True):
		'''split cells using k-fold strategy to reduce data size and cross-validate'''
		kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed) # generate KFold object for splitting data
		splits = {'train':[], 'test':[]} # initiate empty dictionary to dump matrix subsets into

		if counts_obj.barcodes is not None:
			codes = pd.DataFrame(counts_obj.barcodes)
			codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

		else:
			codes=None

		for train_i, test_i in kf.split(counts_obj.data):
			splits['train'].append(cls(counts_obj.data.iloc[train_i], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes))
			splits['test'].append(cls(counts_obj.data.iloc[test_i], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes))

		return splits


	@classmethod
	def nvr_select(cls, counts_obj, scale=1000):
		if scale:
			hqGenes = nvr.parseNoise(counts_obj.counts) # identify non-noisy genes
			selected_genes = nvr.select_genes(counts_obj.arcsinh_norm(scale=scale)[:,hqGenes]) # select features from arsinh-transformed, non-noisy data

		else:
			selected_genes = nvr.select_genes(counts_obj.arcsinh_norm(scale=scale)) # select features from arsinh-transformed, non-noisy data

		if counts_obj.barcodes is not None:
			codes = pd.DataFrame(counts_obj.barcodes)
			codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

		else:
			codes=None

		print('\nSelected {} variable genes\n'.format(selected_genes.shape[0]))
		return cls(counts_obj.data.iloc[:,selected_genes], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)


	@classmethod
	def var_select(cls, counts_obj, n_features):
		'''select n_features (genes) with highest variance across all cells in dataset'''
		v = counts_obj.data.var(axis=0).nlargest(n_features).index # get top n variant gene IDs

		if counts_obj.barcodes is not None:
			codes = pd.DataFrame(counts_obj.barcodes)
			codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

		else:
			codes=None

		return cls(counts_obj.data[v], labels=[counts_obj.cell_labels, counts_obj.gene_labels], barcodes=codes)



class DR():
	'''Catch-all class for dimensionality reduction outputs for high-dimensional data of shape (n_cells, n_features)'''
	def __init__(self, matrix, barcodes=None):
		self.input = matrix # store input matrix as metadata

		if barcodes is not None:
			self.barcodes = barcodes # maintain given barcode information

		else:
			self.barcodes = None


	def distance_matrix(self, ranks='all'):
		'''
		calculate Euclidean distances between cells in matrix of shape (n_cells, n_cells)
			ranks: rank barcodes by occurrence in dataset and plot list of ranks (e.g. [1,2,3] or np.arange(1:4) for top 3 codes)
		'''
		if ranks == 'all':
			return sc.spatial.distance_matrix(self.results, self.results)

		else:
			assert self.barcodes is not None, 'Barcodes not assigned.\n'
			ints = [x for x in ranks if type(x)==int] # pull out rank values
			IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
			ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
			ranks_results = self.results[self.barcodes.isin(list(ranks_i) + IDs)] # subset results array
			return sc.spatial.distance_matrix(ranks_results, ranks_results)


	def knn_graph(self, k, ranks='all'):
		'''
		calculate k nearest neighbors for each cell in distance matrix of shape (n_cells, n_cells)
			ranks: rank barcodes by occurrence in dataset and plot list of ranks (e.g. [1,2,3] or np.arange(1:4) for top 3 codes)
		'''
		if ranks=='all':
			return kneighbors_graph(self.distance_matrix(), k, mode='connectivity', include_self=False).toarray()

		else:
			assert self.barcodes is not None, 'Barcodes not assigned.\n'
			return kneighbors_graph(self.distance_matrix(ranks=ranks), k, mode='connectivity', include_self=False).toarray()


	def top_barcodes(self, ranks):
		'''return list of top-ranked barcodes by prevalence in dataset'''
		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		ints = [x for x in ranks if type(x)==int] # pull out rank values
		IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
		return list(self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index) + IDs


	def silhouette_score(self):
		'''calculate silhouette score of clustered results'''
		assert hasattr(self.clu, 'membership'), 'Clustering not yet determined. Assign clusters with self.clu.assign().\n'
		return silhouette_score(self.results, self.clu.membership) # calculate silhouette score


	def cluster_counts(self, prettyprint=True):
		'''return number of cells in each cluster'''
		assert hasattr(self.clu, 'membership'), 'Clustering not yet determined. Assign clusters with self.clu.assign().\n'
		IDs, counts = np.unique(self.clu.membership, return_counts=True)
		for ID, count in zip(IDs, counts):
			print('{} cells in cluster {} ({} %)\n'.format(count, ID, np.round(count/counts.sum()*100,3)))


	def plot_clusters(self):
		'''Visualize density peak clustering of DR results and calculate silhouette score'''
		assert hasattr(self.clu, 'clusters'), 'Clustering not yet determined. Assign clusters with self.clu.assign().\n'
		fig, ax = plt.subplots(1, 3, figsize=(15, 5))
		ax[0].scatter(self.results[:, 0], self.results[:, 1], s=75, alpha=0.7)
		ax[0].scatter(self.results[self.clu.clusters, 0], self.results[self.clu.clusters, 1], s=90, c="red")
		ax[1].scatter(self.results[:, 0], self.results[:, 1], s=75, alpha=0.7, c=self.clu.density)
		ax[2].scatter(self.results[:, 0], self.results[:, 1], s=75, alpha=0.7, c=self.clu.membership, cmap=plt.cm.plasma)
		IDs, counts = np.unique(self.clu.membership, return_counts=True) # get cluster counts and IDs
		bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9) # set up annotate box
		# add percentages of each cluster to plot
		for ID, count, x, y in zip(IDs, counts, self.results[self.clu.clusters, 0], self.results[self.clu.clusters, 1]):
			ax[2].annotate('{} %'.format(np.round(count/counts.sum()*100,2)), xy=(x, y), ha="center", va="center", size=12, bbox=bbox_props)

		for _ax in ax:
			_ax.set_aspect('equal')
			_ax.tick_params(labelbottom=False, labelleft=False)

		sns.despine(left=True, bottom=True)
		fig.tight_layout()



class fcc_PCA(DR):
	'''
	Object containing Principal Component Analysis of high-dimensional dataset of shape (n_cells, n_features) to reduce to n_components
	'''
	def __init__(self, matrix, n_components, barcodes=None):
		DR.__init__(self, matrix, barcodes) # inherits from DR object
		self.components = n_components # store number of components as metadata
		self.fit = PCA(n_components=self.components).fit(self.input) # fit PCA to data
		self.results = self.fit.transform(self.input) # transform data to fit
		self.clu = Cluster(self.results, autoplot=False) # get density-peak cluster information for results to use for plotting


	def plot(self):
		plt.figure(figsize=(10,5))

		plt.subplot(121)
		sns.scatterplot(x=self.results[:,0], y=self.results[:,1], s=75, alpha=0.7, hue=self.clu.density, legend=None, edgecolor='none')
		plt.tick_params(labelbottom=False, labelleft=False)
		plt.ylabel('PC2', fontsize=14)
		plt.xlabel('PC1', fontsize=14)
		plt.title('PCA', fontsize=16)

		plt.subplot(122)
		plt.plot(np.cumsum(np.round(self.fit.explained_variance_ratio_, decimals=3)*100))
		plt.tick_params(labelsize=12)
		plt.ylabel('% Variance Explained', fontsize=14)
		plt.xlabel('# of Features', fontsize=14)
		plt.title('PCA Analysis', fontsize=16)

		sns.despine()
		plt.tight_layout()
		plt.show()
		plt.close()


	def plot_barcodes(self, ranks='all'):
		'''
		Plot projection colored by barcode
			ranks: Rank barcodes by occurrence in dataset and plot list of ranks (e.g. [1,2,3] or np.arange(1:4) for top 3 codes). Default all codes plotted.
		'''
		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		plt.figure(figsize=(5,5))

		if ranks == 'all':
			sns.scatterplot(x=self.results[:,0], y=self.results[:,1], s=75, alpha=0.7, hue=self.barcodes, legend=None, edgecolor='none')

		else:
			ints = [x for x in ranks if type(x)==int] # pull out rank values
			IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
			ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
			ranks_codes = self.barcodes[self.barcodes.isin(list(ranks_i) + IDs)] # subset barcodes series
			ranks_results = self.results[self.barcodes.isin(list(ranks_i) + IDs)] # subset results array
			sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.1, color='gray', legend=None, edgecolor='none')
			sns.scatterplot(ranks_results[:,0], ranks_results[:,1], s=75, alpha=0.7, legend=False, hue=ranks_codes, edgecolor='none')

		plt.tick_params(labelbottom=False, labelleft=False)
		plt.ylabel('PC2', fontsize=14)
		plt.xlabel('PC1', fontsize=14)
		plt.title('PCA', fontsize=16)

		sns.despine()
		plt.tight_layout()
		plt.show()
		plt.close()



class fcc_tSNE(DR):
	'''
	Object containing t-SNE of high-dimensional dataset of shape (n_cells, n_features) to reduce to n_components
	'''
	def __init__(self, matrix, perplexity, n_components=2, barcodes=None):
		DR.__init__(self, matrix, barcodes) # inherits from DR object
		self.components = n_components # store number of components as metadata
		self.perplexity = perplexity # store tSNE perplexity as metadata
		self.results = TSNE(n_components=self.components, perplexity=self.perplexity).fit_transform(self.input)
		self.clu = Cluster(self.results.astype('double'), autoplot=False) # get density-peak cluster information for results to use for plotting


	def plot(self):
		plt.figure(figsize=(5,5))
		sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.7, hue=self.clu.density, legend=None, edgecolor='none')
		plt.xlabel('t-SNE 1', fontsize=14)
		plt.ylabel('t-SNE 2', fontsize=14)
		plt.tick_params(labelbottom=False, labelleft=False)
		sns.despine(left=True, bottom=True)
		plt.tight_layout()
		plt.show()
		plt.close()


	def plot_barcodes(self, ranks='all'):
		'''
		Plot projection colored by barcode
			ranks: Rank barcodes by occurrence in dataset and plot list of ranks (e.g. [1,2,3] or np.arange(1:4) for top 3 codes). Default all codes plotted.
		'''
		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		plt.figure(figsize=(5,5))

		if ranks == 'all':
			sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.7, hue=self.barcodes, legend=None, edgecolor='none')

		else:
			ints = [x for x in ranks if type(x)==int] # pull out rank values
			IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
			ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
			ranks_codes = self.barcodes[self.barcodes.isin(list(ranks_i) + IDs)] # subset barcodes series
			ranks_results = self.results[self.barcodes.isin(list(ranks_i) + IDs)] # subset results array
			sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.1, color='gray', legend=None, edgecolor='none')
			sns.scatterplot(ranks_results[:,0], ranks_results[:,1], s=75, alpha=0.7, legend=False, hue=ranks_codes, edgecolor='none')

		plt.xlabel('t-SNE 1', fontsize=14)
		plt.ylabel('t-SNE 2', fontsize=14)
		plt.tick_params(labelbottom=False, labelleft=False)
		sns.despine(left=True, bottom=True)
		plt.tight_layout()
		plt.show()
		plt.close()



class fcc_FItSNE(DR):
	'''
	Object containing FIt-SNE (https://github.com/KlugerLab/FIt-SNE) of high-dimensional dataset of shape (n_cells, n_features) to reduce to n_components
	'''
	def __init__(self, matrix, perplexity, seed=-1, barcodes=None, clean_workspace=True):
		DR.__init__(self, matrix, barcodes) # inherits from DR object
		self.perplexity = perplexity # store tSNE perplexity as metadata
		self.results = fast_tsne(self.input, perplexity=self.perplexity, seed=seed)
		self.clu = Cluster(self.results.astype('double'), autoplot=False) # get density-peak cluster information for results to use for plotting
		if clean_workspace:
			# get rid of files used by C++ to run FFT t-SNE
			os.remove('data.dat')
			os.remove('result.dat')


	def plot(self):
		plt.figure(figsize=(5,5))
		sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.7, hue=self.clu.density, legend=None, edgecolor='none')
		plt.xlabel('FIt-SNE 1', fontsize=14)
		plt.ylabel('FIt-SNE 2', fontsize=14)
		plt.tick_params(labelbottom=False, labelleft=False)
		sns.despine(left=True, bottom=True)
		plt.tight_layout()
		plt.show()
		plt.close()


	def plot_barcodes(self, ranks='all'):
		'''
		Plot projection colored by barcode
			ranks: Rank barcodes by occurrence in dataset and plot list of ranks (e.g. [1,2,3] or np.arange(1:4) for top 3 codes). Default all codes plotted.
		'''
		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		plt.figure(figsize=(5,5))

		if ranks == 'all':
			sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.7, hue=self.barcodes, legend=None, edgecolor='none')

		else:
			ints = [x for x in ranks if type(x)==int] # pull out rank values
			IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
			ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
			ranks_codes = self.barcodes[self.barcodes.isin(list(ranks_i) + IDs)] # subset barcodes series
			ranks_results = self.results[self.barcodes.isin(list(ranks_i) + IDs)] # subset results array
			sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.1, color='gray', legend=None, edgecolor='none')
			sns.scatterplot(ranks_results[:,0], ranks_results[:,1], s=75, alpha=0.7, legend=False, hue=ranks_codes, edgecolor='none')

		plt.xlabel('FIt-SNE 1', fontsize=14)
		plt.ylabel('FIt-SNE 2', fontsize=14)
		plt.tick_params(labelbottom=False, labelleft=False)
		sns.despine(left=True, bottom=True)
		plt.tight_layout()
		plt.show()
		plt.close()



class fcc_UMAP(DR):
	'''
	Object containing UMAP of high-dimensional dataset of shape (n_cells, n_features) to reduce to 2 components
	'''
	def __init__(self, matrix, perplexity, min_dist=0.3, metric='correlation', barcodes=None):
		DR.__init__(self, matrix, barcodes) # inherits from DR object
		self.perplexity = perplexity
		self.min_dist = min_dist
		self.metric = metric
		self.results = UMAP(n_neighbors=self.perplexity, min_dist=self.min_dist, metric=self.metric).fit_transform(self.input)
		self.clu = Cluster(self.results.astype('double'), autoplot=False)


	def plot(self):
		plt.figure(figsize=(5,5))
		sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.7, hue=self.clu.density, legend=None, edgecolor='none')
		plt.xlabel('UMAP 1', fontsize=14)
		plt.ylabel('UMAP 2', fontsize=14)
		plt.tick_params(labelbottom=False, labelleft=False)
		sns.despine(left=True, bottom=True)
		plt.tight_layout()
		plt.show()
		plt.close()


	def plot_barcodes(self, ranks='all'):
		'''
		Plot projection colored by barcode
			ranks: Rank barcodes by occurrence in dataset and plot list of ranks (e.g. [1,2,3] or np.arange(1:4) for top 3 codes). Default all codes plotted.
		'''
		assert self.barcodes is not None, 'Barcodes not assigned.\n'
		plt.figure(figsize=(5,5))

		if ranks == 'all':
			sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.7, hue=self.barcodes, legend=None, edgecolor='none')

		else:
			ints = [x for x in ranks if type(x)==int] # pull out rank values
			IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
			ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
			ranks_codes = self.barcodes[self.barcodes.isin(list(ranks_i) + IDs)] # subset barcodes series
			ranks_results = self.results[self.barcodes.isin(list(ranks_i) + IDs)] # subset results array
			sns.scatterplot(self.results[:,0], self.results[:,1], s=75, alpha=0.1, color='gray', legend=None, edgecolor='none')
			sns.scatterplot(ranks_results[:,0], ranks_results[:,1], s=75, alpha=0.7, legend=False, hue=ranks_codes, edgecolor='none')

		plt.xlabel('UMAP 1', fontsize=14)
		plt.ylabel('UMAP 2', fontsize=14)
		plt.tick_params(labelbottom=False, labelleft=False)
		sns.despine(left=True, bottom=True)
		plt.tight_layout()
		plt.show()
		plt.close()



class fcc_DCA(DR):
	'''
	Object containing DCA of high-dimensional dataset of shape (n_cells, n_features) to reduce to 33 components
		NOTE: DCA removes features with 0 counts for all cells prior to processing.
	'''
	def __init__(self, matrix, n_threads=2, norm=True, barcodes=None):
		DR.__init__(self, matrix, barcodes) # inherits from DR object
		self.DCA_norm = norm # store normalization decision as metadata
		self.adata = scanpy.AnnData(self.input) # generate AnnData object (https://github.com/theislab/scanpy) for passing to DCA
		scanpy.pp.filter_genes(self.adata, min_counts=1) # remove features with 0 counts for all cells
		dca(self.adata, threads=n_threads) # perform DCA analysis on AnnData object

		if self.DCA_norm:
			scanpy.pp.normalize_per_cell(self.adata) # normalize features for each cell with scanpy's method
			scanpy.pp.log1p(self.adata) # log-transform data with scanpy's method

		self.results = self.adata.X # return the denoised data as a np.ndarray
		self.clu = Cluster(self.results.astype('double'), autoplot=False)
