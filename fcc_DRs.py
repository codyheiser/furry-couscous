# furry-couscous dimensionality reduction objects

# @author: C Heiser
# November 2018

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
# density peak clustering
from pydpc import Cluster                    	# density-peak clustering
# DCA packages
import scanpy.api as scanpy
from dca.api import dca                      	# DCA
# UMAP
from umap import UMAP                           # UMAP
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
	def __init__(self, data, labels=[0,0], cells_axis=0):
		'''initialize object from np.ndarray or pd.DataFrame (data)'''
		self.data = data # store pd.DataFrame as data attribute

		if cells_axis == 1: # put cells on 0 axis if not already there
			self.data = self.data.transpose() 

		if labels[0]!=None: # if cell IDs present, save as metadata
			self.cell_IDs = self.data.index

		if labels[1]!=None: # if gene IDs present, save as metadata
			self.gene_IDs = self.data.columns

		self.counts = np.ascontiguousarray(self.data) # store counts matrix as counts attribute (no labels, np.array format)


	def distance_matrix(self):
		'''calculate Euclidean distances between cells in matrix of shape (n_cells, n_cells)'''
		return sc.spatial.distance_matrix(self.counts, self.counts)


	def arcsinh_norm(self, norm=True, scale=1000):
		'''
		Perform an arcsinh-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
		Useful for feeding into PCA or tSNE.
			norm = convert to fractional counts first? divide each count by sqrt of sum of squares of counts for cell.
			scale = factor to multiply values by before arcsinh-transform. scales values away from [0,1] in order to make arcsinh more effective.
		'''
		if not norm:
			return np.arcsinh(self.counts * scale)
		
		else:
			return np.arcsinh(normalize(self.counts, axis=0, norm='l2') * scale)

    
	def log2_norm(self, norm = True):
		'''
		Perform a log2-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
		Useful for feeding into PCA or tSNE.
			norm = convert to fractional counts first? divide each count by sqrt of sum of squares of counts for cell.
		'''
		if not norm:
			return np.log2(self.counts + 1)
		
		else:
			return np.log2(normalize(self.counts, axis=0, norm='l2') + 1)


	@classmethod
	def from_file(cls, datafile, labels=[0,0], cells_axis=0):
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

		return cls(data, labels=labels, cells_axis=cells_axis)


	@classmethod
	def downsample(cls, data, n_cells):
		'''downsample a dataframe of shape (n_cells, n_features) to n_cells and generate new counts object'''
		return cls(data.loc[np.random.choice(data.shape[0], n_cells, replace=False)])


	@classmethod
	def kfold_split(cls, data, n_splits, seed=None, shuffle=True):
		'''split cells using k-fold strategy to reduce data size and cross-validate'''
		kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed) # generate KFold object for splitting data
		splits = {'train':[], 'test':[]} # initiate empty dictionary to dump matrix subsets into

		for train_i, test_i in kf.split(data):
			splits['train'].append(cls(data.loc[train_i]))
			splits['test'].append(cls(data.loc[test_i]))

		return splits


	@classmethod
	def nvr_select(cls, data, scale=1000):
		hqGenes = nvr.parseNoise(data)
		dHq = nvr.mkIndexedArr(data, hqGenes)
		dataHq = nvr.pwArcsinh(dHq, scale)
		selected_genes=nvr.select_genes(dataHq)
		print('\nSelected {} variable genes\n'.format(selected_genes.shape[0]))
		return cls(dataHq[:,selected_genes], labels=[None,None])



class DR():
	'''Catch-all class for dimensionality reduction outputs for high-dimensional data of shape (n_cells, n_features)'''
	def __init__(self, matrix):
		self.input = matrix # store input matrix as metadata


	def distance_matrix(self):
		'''calculate Euclidean distances between cells in matrix of shape (n_cells, n_cells)'''
		return sc.spatial.distance_matrix(self.results, self.results)


	def plot_clusters(self):
		'''Visualize density peak clustering of DR results and calculate silhouette score'''
		try:
			fig, ax = plt.subplots(1, 3, figsize=(15, 5))
			ax[0].scatter(self.results[:, 0], self.results[:, 1], s=75, alpha=0.7)
			ax[0].scatter(self.results[self.clu.clusters, 0], self.results[self.clu.clusters, 1], s=90, c="red")
			ax[1].scatter(self.results[:, 0], self.results[:, 1], s=75, alpha=0.7, c=self.clu.density)
			ax[2].scatter(self.results[:, 0], self.results[:, 1], s=75, alpha=0.7, c=self.clu.membership, cmap=plt.cm.plasma)
			for _ax in ax:
				_ax.set_aspect('equal')
				_ax.tick_params(labelbottom=False, labelleft=False)
			
			sns.despine(left=True, bottom=True)
			fig.tight_layout()

			self.silhouette_score = silhouette_score(self.results, self.clu.membership) # calculate silhouette score

		except AttributeError as err:
			print('Clustering not yet determined. Assign clusters with self.clu.assign().\n', err)



class fcc_PCA(DR):
	'''
	Object containing Principal Component Analysis of high-dimensional dataset of shape (n_cells, n_features) to reduce to n_components
	'''
	def __init__(self, matrix, n_components):
		DR.__init__(self, matrix) # inherits from DR object
		self.components = n_components # store number of components as metadata
		self.fit = PCA(n_components=self.components).fit(self.input) # fit PCA to data
		self.results = self.fit.transform(self.input) # transform data to fit
		self.clu = Cluster(self.results, autoplot=False) # get density-peak cluster information for results to use for plotting


	def plot(self):
		plt.figure(figsize=(10,5))
		
		plt.subplot(121)
		plt.scatter(self.results[:,0], self.results[:,1], s=75, alpha=0.7, c=self.clu.density)
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



class fcc_tSNE(DR):
	'''
	Object containing t-SNE of high-dimensional dataset of shape (n_cells, n_features) to reduce to n_components
	'''
	def __init__(self, matrix, perplexity, n_components=2):
		DR.__init__(self, matrix) # inherits from DR object
		self.components = n_components # store number of components as metadata
		self.perplexity = perplexity # store tSNE perplexity as metadata
		self.results = TSNE(n_components=self.components, perplexity=self.perplexity).fit_transform(self.input)
		self.clu = Cluster(self.results.astype('double'), autoplot=False) # get density-peak cluster information for results to use for plotting


	def plot(self):
		plt.figure(figsize=(5,5))
		plt.scatter(self.results[:,0], self.results[:,1], s=75, alpha=0.7, c=self.clu.density)
		plt.xlabel('t-SNE 1', fontsize=14)
		plt.ylabel('t-SNE 2', fontsize=14)
		plt.tick_params(labelbottom=False, labelleft=False)
		sns.despine(left=True, bottom=True)
		plt.tight_layout()
		plt.show()
		plt.close()



class fcc_UMAP(DR):
	'''
	Object containing UMAP of high-dimensional dataset of shape (n_cells, n_features) to reduce to 2 components
	'''
	def __init__(self, matrix, perplexity, min_dist=0.3, metric='correlation'):
		DR.__init__(self, matrix) # inherits from DR object
		self.perplexity = perplexity
		self.min_dist = min_dist
		self.metric = metric
		self.results = UMAP(n_neighbors=self.perplexity, min_dist=self.min_dist, metric=self.metric).fit_transform(self.input)
		self.clu = Cluster(self.results.astype('double'), autoplot=False)


	def plot(self):
		plt.figure(figsize=(5,5))
		plt.scatter(self.results[:,0], self.results[:,1], s=75, alpha=0.7, c=self.clu.density)
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
	def __init__(self, matrix, n_threads=2, norm=True):
		DR.__init__(self, matrix) # inherits from DR object
		self.DCA_norm = norm # store normalization decision as metadata
		self.adata = scanpy.AnnData(self.input) # generate AnnData object (https://github.com/theislab/scanpy) for passing to DCA
		scanpy.pp.filter_genes(self.adata, min_counts=1) # remove features with 0 counts for all cells
		dca(self.adata, threads=n_threads) # perform DCA analysis on AnnData object
		
		if self.DCA_norm:
			scanpy.pp.normalize_per_cell(self.adata) # normalize features for each cell with scanpy's method
			scanpy.pp.log1p(self.adata) # log-transform data with scanpy's method
			
		self.results = self.adata.X # return the denoised data as a np.ndarray
		self.clu = Cluster(self.results.astype('double'), autoplot=False)


