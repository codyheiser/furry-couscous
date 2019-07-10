# slide-seq fusion objects

# @author: C Heiser
# July 2019

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
from sklearn.decomposition import PCA			# PCA
from sklearn.manifold import TSNE				# t-SNE
from sklearn.neighbors import kneighbors_graph	# K-nearest neighbors graph
# density peak clustering
from pydpc import Cluster						# density-peak clustering
# other DR methods
from umap import UMAP						# UMAP
# plotting packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')



class couscous():
    '''
    Object containing scRNA-seq counts data
    '''
    def __init__(self, data, data_type='counts', labels=[0,0], cells_axis=0, barcodes=None):
        '''
        initialize object
            data = pd.DataFrame containing counts or DR results data
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing input data
            labels = list containing [col, row] indices of labels in data. None if no cell or feature IDs, respectively.
            cells_axis = cells x features (0), or features x cells (1)
            barcodes = pd.DataFrame containing cell barcodes. Header of cell barcode column should be named 'Barcode'.
        '''
        self.cell_labels = labels[0] # index of column containing cell IDs
        self.feature_labels = labels[1] # index of row containing gene IDs

        if cells_axis == 1: # put cells on 0 axis if not already there
            data = data.transpose()

        if self.cell_labels!=None: # if cell IDs present, save as metadata
            self.cell_IDs = data.index

        if self.feature_labels!=None: # if feature IDs present, save as metadata
            self.feature_IDs = data.columns

        self.data = {data_type:data} # store relevant matrices in data attribute

        self.clu = {} # initiate dictionary of clustering results
        if data_type is not 'counts':
            self.clu[data_type] = Cluster(self.data[data_type], autoplot=False) # get density-peak cluster information for results to use for plotting

        if barcodes is not None: # if barcodes df provided, merge with data
            data_coded = data.merge(barcodes, left_index=True, right_index=True, how='left')
            data_coded = data_coded.astype({'Barcode':'category'})
            self.data_coded = data_coded # create 'coded' attribute that has data and barcodes
            self.barcodes = data_coded['Barcode'] # make barcodes attribute pd.Series for passing to other classes

        else:
            self.barcodes = None


    def distance_matrix(self, data_type='counts', transform=None, ranks='all', **kwargs):
        '''
        calculate Euclidean distances between cells in matrix of shape (n_cells, n_cells)
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to calculate distances in
            transform = how to normalize and transform data prior to calculating distances (None, "arcsinh", or "log2")
            ranks = which barcodes to return distances for. Can be list of ranks of most abundant barcodes (integers, i.e. [1,2,3] for top 3 barcodes),
                or names of barcode IDs (strings, i.e. ['0','1','2'] for barcodes with numbered IDs)
            **kwargs = keyword arguments to pass to normalization functions
        '''
        # transform data first, if necessary
        if transform is None:
            transformed = self.data[data_type]

        if transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        # then subset data by rank-ordered barcode appearance
        if ranks=='all':
            return sc.spatial.distance_matrix(transformed, transformed)

        elif not isinstance(ranks, (list,)): # make sure input is list-formatted
            ranks = [ranks]

        assert self.barcodes is not None, 'Barcodes not assigned.\n'
        ints = [x for x in ranks if type(x)==int] # pull out rank values
        IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
        ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
        ranks_counts = transformed[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed data
        return sc.spatial.distance_matrix(ranks_counts, ranks_counts)


    def knn_graph(self, k, data_type='counts', **kwargs):
        '''
        calculate k nearest neighbors for each cell in distance matrix of shape (n_cells, n_cells)
            k = number of nearest neighbors to determine
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to calculate distances in
            **kwargs = keyword arguments to pass to distance_matrix() function
        '''
        return kneighbors_graph(self.distance_matrix(**kwargs), k, mode='connectivity', include_self=False).toarray()


    def barcode_counts(self, IDs='all'):
        '''
        given list of barcode IDs, return pd.Series of number of appearances in dataset
            IDs = which barcodes to return distances for. List of names of barcode IDs (strings, i.e. ['0','1','2'] for barcodes with numbered IDs)
        '''
        assert self.barcodes is not None, 'Barcodes not assigned.\n'

        if IDs=='all':
            return self.barcodes.value_counts()

        if not isinstance(IDs, (list,)): # make sure input is list-formatted
            IDs = [IDs]

        return self.barcodes.value_counts()[self.barcodes.value_counts().index.isin(IDs)]


    def arcsinh_norm(self, data_type='counts', norm='l1', scale=1000, ranks='all'):
        '''
        Perform an arcsinh-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
        Useful for feeding into PCA or tSNE.
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to normalize
            norm = normalization strategy prior to Log2 transform.
                None: do not normalize data
                'l1': divide each feature by sum of features for each cell
                'l2': divide each feature by sqrt of sum of squares of features for cell.
            scale = factor to multiply values by before arcsinh-transform. scales values away from [0,1] in order to make arcsinh more effective.
            ranks = which barcodes to keep after normalization. Can be list of ranks of most abundant barcodes (integers, i.e. [1,2,3] for top 3 barcodes),
                or names of barcode IDs (strings, i.e. ['0','1','2'] for barcodes with numbered IDs)
        '''
        if not norm:
            out = np.arcsinh(np.ascontiguousarray(self.data[data_type]) * scale)

        else:
            out = np.arcsinh(normalize(np.ascontiguousarray(self.data[data_type]), axis=1, norm=norm) * scale)

        if ranks=='all':
            return out

        elif not isinstance(ranks, (list,)): # make sure input is list-formatted
            ranks = [ranks]

        assert self.barcodes is not None, 'Barcodes not assigned.\n'
        ints = [x for x in ranks if type(x)==int] # pull out rank values
        IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
        ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
        return out[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed counts array


    def log2_norm(self, data_type='counts', norm='l1', ranks='all'):
        '''
        Perform a log2-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
        Useful for feeding into PCA or tSNE.
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to normalize
            norm = normalization strategy prior to Log2 transorm.
                None: do not normalize data
                'l1': divide each feature by sum of features for each cell
                'l2': divide each feature by sqrt of sum of squares of features for cell.
            ranks = which barcodes to keep after normalization. Can be list of ranks of most abundant barcodes (integers, i.e. [1,2,3] for top 3 barcodes),
                or names of barcode IDs (strings, i.e. ['0','1','2'] for barcodes with numbered IDs)
        '''
        if not norm:
            out = np.log2(np.ascontiguousarray(self.data[data_type]) + 1) # add pseudocount of 1 to avoid log(0)

        else:
            out = np.log2(normalize(np.ascontiguousarray(self.data[data_type]), axis=1, norm=norm) + 1) # add pseudocount of 1 to avoid log(0)

        if ranks=='all':
            return out

        elif not isinstance(ranks, (list,)): # make sure input is list-formatted
            ranks = [ranks]

        assert self.barcodes is not None, 'Barcodes not assigned.\n'
        ints = [x for x in ranks if type(x)==int] # pull out rank values
        IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
        ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
        return out[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed counts array


    def fcc_PCA(self, n_components, data_type='counts', transform=None, **kwargs):
        '''
        principal component analysis
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to perform PCA on
            transform = how to normalize and transform data prior to reducing dimension (None, "arcsinh", or "log2")
            **kwargs = keyword arguments to pass to normalization functions
        '''
        # transform data first, if necessary
        if transform is None:
            transformed = self.data[data_type]

        if transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        self.PCA_fit = PCA(n_components=n_components).fit(transformed) # fit PCA to data
        self.data['PCA'] = self.PCA_fit.transform(transformed) # transform data to PCA space and save in data attribute
        self.clu['PCA'] = Cluster(self.data['PCA'], autoplot=False) # perform DPC on PCA results


    def fcc_tSNE(self, perplexity, data_type='counts', transform=None, seed=None, **kwargs):
        '''
        t-distributed stochastic neighbor embedding
            perplexity = value of nearest neighbors to attract one another in optimization
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to perform t-SNE on
            transform = how to normalize and transform data prior to reducing dimension (None, "arcsinh", or "log2")
            seed = random state to initialize t-SNE for reproducibility
            **kwargs = keyword arguments to pass to normalization functions
        '''
        # transform data first, if necessary
        if transform is None:
            transformed = self.data[data_type]

        if transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        self.tSNE_fit = TSNE(perplexity=perplexity, random_state=seed).fit(transformed)
        self.data['t-SNE'] = self.tSNE_fit.fit_transform(transformed)
        self.clu['t-SNE'] = Cluster(self.data['t-SNE'].astype('double'), autoplot=False) # perform DPC on t-SNE results


    def fcc_UMAP(self, perplexity, data_type='counts', transform=None, seed=None, **kwargs):
        '''
        uniform manifold approximation and projection
            perplexity = value of nearest neighbors to attract one another in optimization
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to perform UMAP on
            transform = how to normalize and transform data prior to reducing dimension (None, "arcsinh", or "log2")
            seed = random state to initialize UMAP for reproducibility
            **kwargs = keyword arguments to pass to normalization functions
        '''
        # transform data first, if necessary
        if transform is None:
            transformed = self.data[data_type]

        if transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        self.UMAP_fit = UMAP(n_neighbors=perplexity, random_state=seed).fit(transformed)
        self.data['UMAP'] = self.UMAP_fit.transform(transformed)
        self.clu['UMAP'] = Cluster(self.data['UMAP'].astype('double'), autoplot=False) # perform DPC on UMAP results


    def silhouette_score(self, data_type):
        '''
        calculate silhouette score of clustered results
            data_type = one of ['PCA', 't-SNE', 'UMAP', 'slide-seq'] describing space to evaluate clustering of
        '''
        assert hasattr(self.clu[data_type], 'membership'), 'Clustering not yet determined. Assign clusters with self.clu.assign().\n'
        return silhouette_score(self.data[data_type], self.clu[data_type].membership) # calculate silhouette score


    def cluster_counts(self, data_type):
        '''
        print number of cells in each cluster
            data_type = one of ['PCA', 't-SNE', 'UMAP', 'slide-seq'] describing space to evaluate clustering of
        '''
        assert hasattr(self.clu[data_type], 'membership'), 'Clustering not yet determined. Assign clusters with self.clu.assign().\n'
        IDs, counts = np.unique(self.clu[data_type].membership, return_counts=True)
        for ID, count in zip(IDs, counts):
            print('{} cells in cluster {} ({} %)\n'.format(count, ID, np.round(count/counts.sum()*100,3)))


    def plot_clusters(self, data_type):
        '''
        visualize density peak clustering
            data_type = one of ['PCA', 't-SNE', 'UMAP', 'slide-seq'] describing space to plot
        '''
        assert hasattr(self.clu[data_type], 'clusters'), 'Clustering not yet determined. Assign clusters with self.clu.assign().\n'
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].scatter(self.data[data_type][:, 0], self.data[data_type][:, 1], s=75, alpha=0.7)
        ax[0].scatter(self.data[data_type][self.clu[data_type].clusters, 0], self.data[data_type][self.clu[data_type].clusters, 1], s=90, c="red")
        ax[1].scatter(self.data[data_type][:, 0], self.data[data_type][:, 1], s=75, alpha=0.7, c=self.clu[data_type].density)
        ax[2].scatter(self.data[data_type][:, 0], self.data[data_type][:, 1], s=75, alpha=0.7, c=self.clu[data_type].membership, cmap=plt.cm.plasma)
        IDs, counts = np.unique(self.clu[data_type].membership, return_counts=True) # get cluster counts and IDs
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9) # set up annotate box
        # add percentages of each cluster to plot
        for ID, count, x, y in zip(IDs, counts, self.data[data_type][self.clu[data_type].clusters, 0], self.data[data_type][self.clu[data_type].clusters, 1]):
            ax[2].annotate('{} %'.format(np.round(count/counts.sum()*100,2)), xy=(x, y), ha="center", va="center", size=12, bbox=bbox_props)

        for _ax in ax:
            _ax.set_aspect('equal')
            _ax.tick_params(labelbottom=False, labelleft=False)

        sns.despine(left=True, bottom=True)
        fig.tight_layout()


    def plot(self, data_type, color=None, save_to=None, figsize=(5,5)):
        '''
        standard plot of first 2 dimensions of latent space
            data_type = one of ['PCA', 't-SNE', 'UMAP'] describing space to plot
            color = vector of values to color points by. Default coloring is by point density.
            save_to = path to .png file to save plot to
            figsize = size in inches of output figure
        '''
        if color is None:
            color = self.clu[data_type].density

        plotter = np.ascontiguousarray(self.data[data_type]) # coerce data to np array for plotting

        fig, ax = plt.subplots(1, figsize=figsize)
        sns.scatterplot(plotter[:,0], plotter[:,1], s=75, alpha=0.7, hue=color, legend=None, edgecolor='none')

        if data_type == 'PCA':
            dim_name = 'PC'

        else:
            dim_name = data_type

        plt.xlabel('{} 1'.format(dim_name), fontsize=14)
        ax.xaxis.set_label_coords(0.2, -0.025)
        plt.ylabel('{} 2'.format(dim_name), fontsize=14)
        ax.yaxis.set_label_coords(-0.025, 0.2)

        plt.annotate('', textcoords='axes fraction', xycoords='axes fraction', xy=(-0.006,0), xytext=(0.2,0), arrowprops=dict(arrowstyle= '<-', lw=2, color='black'))
        plt.annotate('', textcoords='axes fraction', xycoords='axes fraction', xy=(0,-0.006), xytext=(0,0.2), arrowprops=dict(arrowstyle= '<-', lw=2, color='black'))

        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        if save_to is None:
            plt.show()
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight', dpi=1000)

        plt.close()


    @classmethod
    def from_file(cls, datafile, data_type='counts', labels=[0,0], cells_axis=0, barcodefile=None):
        '''
        initialize object from outside files (datafile & beadfile)
            datafile = tab- or comma-delimited (.tsv/.txt/.csv) file containing data. May be .zip or .gz compressed.
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data
            labels = list containing [col, row] indices of labels in DataFrame
            cells_axis = cells x features (0), or features x cells (1)
            barcodefile = comma-delimited (.csv) file containing vertical vector of cell barcode IDs
        '''
        if '.csv' in datafile:
            data = pd.read_csv(datafile, header=labels[1], index_col=labels[0])

        elif '.txt' or '.tsv' in datafile:
            data = pd.read_csv(datafile, header=labels[1], index_col=labels[0], sep='\t')

        if barcodefile: # if barcodes provided, read in file
            barcodes = pd.read_csv(barcodefile, index_col=None, header=None, names=['Barcode'])

        else:
            barcodes = None

        return cls(data=data, data_type=data_type, labels=labels, cells_axis=cells_axis, barcodes=barcodes)



class pita(couscous):
    '''
    Object containing slide-seq data
    '''
    def __init__(self, data, bead_locs, data_type='counts', labels=[0,0], cells_axis=0, barcodes=None):
        '''
        initialize object
            data = pd.DataFrame containing counts or DR results data
            bead_locs = pd.DataFrame containing bead coordinates
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing input data
            labels = list containing [col, row] indices of labels in data. None if no cell or feature IDs, respectively.
            cells_axis = cells x features (0), or features x cells (1)
            barcodes = pd.DataFrame containing cell barcodes. Header of cell barcode column should be named 'Barcode'.
        '''
        couscous.__init__(self, data=data, data_type=data_type, labels=labels, cells_axis=cells_axis, barcodes=barcodes) # inherits from DR object

        self.data['slide-seq'] = bead_locs # store bead locations as attribute
        self.data['slide-seq'].sort_values(axis=0, by=['xcoord','ycoord'], inplace=True) # sort beads by image coordinates
        self.clu['slide-seq'] = Cluster(np.ascontiguousarray(self.data['slide-seq'].astype('double')), autoplot=False) # perform DPC on slide-seq positions for visualization by plot()


    def filter_beads(self, data_type='counts'):
        '''
        remove beads that have no RNA reads from the dataset
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to compare bead composition to
        '''
        self.data['slide-seq'] = self.data['slide-seq'].drop(self.data['slide-seq'].index.difference(self.data[data_type].index))


    def snippet(self, xmin, xmax, ymin, ymax): # TODO: make this a classmethod or extensible to all DRs of pita
        '''
        take snippet of image and make new pita object
        '''
        self.data['slide-seq'] = self.data['slide-seq'].loc[(self.data['slide-seq'].xcoord > xmin) & (self.data['slide-seq'].xcoord < xmax) & (self.data['slide-seq'].ycoord > ymin) & (self.data['slide-seq'].ycoord < ymax)]

        for key in self.data.keys():
            self.data[key] = self.data[key].loc[self.data['slide-seq'].index]


    def map_pixels(self):
        '''
        map bead IDs from 'bead space' to 'pixel space' by assigning bead ID values to evenly spaced grid
        '''
        # determine pixel bounds from bead locations
        xmin = np.floor(self.data['slide-seq']['xcoord'].min())
        xmax = np.ceil(self.data['slide-seq']['xcoord'].max())
        ymin = np.floor(self.data['slide-seq']['ycoord'].min())
        ymax = np.ceil(self.data['slide-seq']['ycoord'].max())

        # define grid for pixel space
        grid_x, grid_y = np.mgrid[xmin:xmax, ymin:ymax]

        # map beads to pixel grid
        self.pixel_map = sc.interpolate.griddata(self.data['slide-seq'].values, self.data['slide-seq'].index, (grid_x, grid_y), method='nearest')


    def assemble_pita(self, data_type, feature, plot_out=True, **kwargs):
        '''
        cast feature into pixel space to construct gene expression image
            data_type = one of ['counts','PCA','t-SNE','UMAP'] describing data to pull features from
            feature = name or index of feature to cast onto bead image
            plot_out = show resulting image?
            **kwargs = arguments to pass to show_pita() function
        '''
        if data_type!='counts':
            mapper = pd.DataFrame(self.data[data_type], index=self.data['counts'].index) # coerce to pandas df for reindexing

        else:
            mapper = self.data[data_type]

        assembled = np.array([mapper[feature].reindex(index=self.pixel_map[x], copy=True) for x in range(len(self.pixel_map))])

        if plot_out:
            self.show_pita(assembled, **kwargs)

        return assembled


    def show_pita(self, assembled, figsize=(7,7), save_to=None, **kwargs):
        '''
        plot assembled pita using imshow
            assembled = np.array output from assemble_pita()
            figsize = size in inches of output figure
            **kwargs = arguments to pass to imshow() function
        '''
        plt.figure(figsize=figsize)
        plt.imshow(assembled.T, origin='lower', **kwargs) # transpose matrix and origin=='lower' to show image with positive axes starting from lower left
        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)

        if save_to is None:
            plt.show()
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight', dpi=1000)

        plt.close()


    @classmethod
    def from_file(cls, datafile, beadfile, data_type='counts', labels=[0,0], cells_axis=0, barcodefile=None):
        '''
        initialize object from outside files (datafile & beadfile)
            datafile = tab- or comma-delimited (.tsv/.txt/.csv) file containing data. May be .zip or .gz compressed.
            beadfile = tab- or comma-delimited (.tsv/.txt/.csv) file containing bead locations for pita. May be .zip or .gz compressed.
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data
            labels = list containing [col, row] indices of labels in DataFrame
            cells_axis = cells x features (0), or features x cells (1)
            barcodefile = comma-delimited (.csv) file containing vertical vector of cell barcode IDs
        '''
        if '.csv' in datafile:
            data = pd.read_csv(datafile, header=labels[1], index_col=labels[0])

        elif '.txt' or '.tsv' in datafile:
            data = pd.read_csv(datafile, header=labels[1], index_col=labels[0], sep='\t')

        if '.csv' in beadfile:
            beads = pd.read_csv(beadfile, header=labels[1], index_col=labels[0])

        elif '.txt' or '.tsv' in beadfile:
            beads = pd.read_csv(beadfile, header=labels[1], index_col=labels[0], sep='\t')

        if barcodefile: # if barcodes provided, read in file
            barcodes = pd.read_csv(barcodefile, index_col=None, header=None, names=['Barcode'])

        else:
            barcodes = None

        return cls(data=data, bead_locs=beads, data_type=data_type, labels=labels, cells_axis=cells_axis, barcodes=barcodes)
