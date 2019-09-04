# scRNA-seq analysis and slide-seq fusion objects

# @author: C Heiser
# August 2019


# basic utilities and plotting
from fcc_utils import *
# packages for reading in data files
import os
# scikit packages
from sklearn.preprocessing import normalize     # matrix normalization
from sklearn.decomposition import PCA           # PCA
from sklearn.manifold import TSNE               # t-SNE
from sklearn.neighbors import kneighbors_graph  # K-nearest neighbors graph
from sklearn.metrics import silhouette_score    # silhouette score
# density peak clustering
from pydpc import Cluster                       # density-peak clustering
# other DR methods
from umap import UMAP                           # UMAP



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
            self.clu[data_type] = Cluster(np.ascontiguousarray(self.data[data_type]), autoplot=False) # get density-peak cluster information for results to use for plotting

        if barcodes is not None: # if barcodes df provided, merge with data
            data_coded = data.merge(barcodes, left_index=True, right_index=True, how='left')
            data_coded = data_coded.astype({'Barcode':'category'})
            self.data_coded = data_coded # create 'coded' attribute that has data and barcodes
            self.barcodes = data_coded['Barcode'] # make barcodes attribute pd.Series for passing to other classes

        else:
            self.barcodes = None


    def arcsinh_norm(self, data_type='counts', norm='l1', scale=1000, ranks='all'):
        '''
        Perform an arcsinh-transformation on a np.ndarray containing raw data of shape=(n_cells,n_genes).
        Useful for feeding into PCA or tSNE.
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to normalize
            norm = normalization strategy prior to arcsinh transform.
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


    def gficf_norm(self, data_type='counts', norm='l1', ranks='all'):
        '''
        Normalize a np.ndarray containing raw data of shape=(n_cells,n_genes) using the gene frequency - inverse cell frequency (GF-ICF) method.
        Adapted from Gambardella & di Bernardo (2019).  Useful for feeding into PCA or tSNE.
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to normalize
            norm = normalization strategy following GF-ICF score calculation for each element.
                None: do not normalize data
                'l1': divide each feature by sum of features for each cell
                'l2': divide each feature by sqrt of sum of squares of features for cell.
            ranks = which barcodes to keep after normalization. Can be list of ranks of most abundant barcodes (integers, i.e. [1,2,3] for top 3 barcodes),
                or names of barcode IDs (strings, i.e. ['0','1','2'] for barcodes with numbered IDs)
        '''
        tf = self.data[data_type].T / self.data[data_type].sum(axis=1)
        tf = tf.T

        ni = self.data[data_type].astype(bool).sum(axis=0)
        idf = np.log(self.data[data_type].shape[0] / (ni+1))

        if not norm:
            out = tf*idf

        else:
            out = normalize(tf*idf, axis=1, norm=norm)

        if ranks=='all':
            return out

        elif not isinstance(ranks, (list,)): # make sure input is list-formatted
            ranks = [ranks]

        assert self.barcodes is not None, 'Barcodes not assigned.\n'
        ints = [x for x in ranks if type(x)==int] # pull out rank values
        IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
        ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
        return out[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed counts array


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

        elif transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        elif transform == 'gficf':
            transformed = self.gficf_norm(data_type=data_type, **kwargs)

        # then subset data by rank-ordered barcode appearance
        if ranks=='all':
            return scipy.spatial.distance_matrix(transformed, transformed)

        elif not isinstance(ranks, (list,)): # make sure input is list-formatted
            ranks = [ranks]

        assert self.barcodes is not None, 'Barcodes not assigned.\n'
        ints = [x for x in ranks if type(x)==int] # pull out rank values
        IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
        ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
        ranks_counts = transformed[np.array(self.barcodes.isin(list(ranks_i) + IDs))] # subset transformed data
        return scipy.spatial.distance_matrix(ranks_counts, ranks_counts)


    def barcode_distance_matrix(self, ranks, data_type='counts', transform=None, **kwargs):
        '''
        calculate Euclidean distances between cells in two barcode groups within a dataset
            ranks = which TWO barcodes to calculate distances between. List of ranks of most abundant barcodes (integers, i.e. [1,2] for top 2 barcodes),
                or names of barcode IDs (strings, i.e. ['0','2'] for barcodes with numbered IDs)
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to calculate distances in
            transform = how to normalize and transform data prior to calculating distances (None, "arcsinh", or "log2")
            **kwargs = keyword arguments to pass to normalization functions
        '''
        assert self.barcodes is not None, 'Barcodes not assigned.\n'

        # transform data first, if necessary
        if transform is None:
            transformed = np.ascontiguousarray(self.data[data_type])

        elif transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        elif transform == 'gficf':
            transformed = self.gficf_norm(data_type=data_type, **kwargs)

        ranks_0 = transformed[np.array(self.barcodes.isin(list(ranks[0])))] # subset transformed counts array to first barcode ID
        ranks_1 = transformed[np.array(self.barcodes.isin(list(ranks[1])))] # subset transformed counts array to second barcode ID
        return scipy.spatial.distance_matrix(ranks_0, ranks_1)


    def knn_graph(self, k, data_type='counts', **kwargs):
        '''
        calculate k nearest neighbors for each cell in distance matrix of shape (n_cells, n_cells)
            k = number of nearest neighbors to determine
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing space to calculate distances in
            **kwargs = keyword arguments to pass to distance_matrix() function
        '''
        return kneighbors_graph(self.distance_matrix(data_type=data_type, **kwargs), k, mode='connectivity', include_self=False).toarray()


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

        elif transform == 'gficf':
            transformed = self.gficf_norm(data_type=data_type, **kwargs)

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

        elif transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        elif transform == 'gficf':
            transformed = self.gficf_norm(data_type=data_type, **kwargs)

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

        elif transform == 'arcsinh':
            transformed = self.arcsinh_norm(data_type=data_type, **kwargs)

        elif transform == 'log2':
            transformed = self.log2_norm(data_type=data_type, **kwargs)

        elif transform == 'gficf':
            transformed = self.gficf_norm(data_type=data_type, **kwargs)

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
        for _, count, x, y in zip(IDs, counts, self.data[data_type][self.clu[data_type].clusters, 0], self.data[data_type][self.clu[data_type].clusters, 1]):
            ax[2].annotate('{} %'.format(np.round(count/counts.sum()*100,2)), xy=(x, y), ha="center", va="center", size=12, bbox=bbox_props)

        for _ax in ax:
            _ax.set_aspect('equal')
            _ax.tick_params(labelbottom=False, labelleft=False)

        sns.despine(left=True, bottom=True)
        fig.tight_layout()


    def plot(self, data_type, color=None, feature_type='counts', features='total', transform='arcsinh', legend=None, save_to=None, figsize=(5,5), **kwargs):
        '''
        standard plot of first 2 dimensions of latent space
            data_type = one of ['PCA', 't-SNE', 'UMAP'] describing space to plot
            color = vector of values to color points by. Default coloring is by point density.
            feature_type = one of ['PCA', 't-SNE', 'UMAP'] describing space to overlay feature intensities from. Default 'counts' to plot gene expression.
            features = list of names or indices of features to cast onto plot. Default 'total' to sum all features in feature_type space for each cell.
            transform = transform data before generating feature image. One of ['arcsinh','log2',None].
            legend = one of ['brief', 'full', None] describing what kind of legend to show
            save_to = path to .png file to save plot to
            figsize = size in inches of output figure
            **kwargs = additional arguments to pass to arcsinh_norm or log2_norm methods.
        '''
        # determine how to display results
        # if no color vector or list of features is given, use point density
        if color is None and features is None:
            color = self.clu[data_type].density
            
        elif color is None:
            # transform data first, if necessary
            if transform is None:
                mapper = pd.DataFrame(self.data[feature_type], index=self.data['counts'].index) # coerce to pd.DF with bead ID index

            elif transform == 'arcsinh':
                mapper = pd.DataFrame(self.arcsinh_norm(data_type=feature_type, **kwargs), index=self.data['counts'].index) # coerce to pd.DF with bead ID index

            elif transform == 'log2':
                mapper = pd.DataFrame(self.log2_norm(data_type=feature_type, **kwargs), index=self.data['counts'].index) # coerce to pd.DF with bead ID index

            # if user wants total of all features (UMI count), sum them up
            if features == 'total':
                color = mapper.sum(axis=1)

            # if 'features' is a string, treat it as regex value
            elif isinstance(features, str):
                color = mapper.loc[:,self.feature_IDs.str.contains(features)].sum(axis=1)

            # if 'features' is an integer, treat it as column index
            elif isinstance(features, int):
                color = mapper.iloc[:,features]

            # if 'features' is a list of strings, treat them as regex values and return sum of (normalized) expression
            elif all(isinstance(elem, str) for elem in features):
                color = mapper.loc[:,self.feature_IDs.str.contains('|'.join(features))].sum(axis=1)

            # if 'features' is a list of integers, treat them as column indices and return sum of (normalized) expression
            elif all(isinstance(elem, int) for elem in features):
                color = mapper.iloc[:,features].sum(axis=1)
            
        plotter = np.ascontiguousarray(self.data[data_type]) # coerce data to np array for plotting

        _, ax = plt.subplots(1, figsize=figsize)
        sns.scatterplot(plotter[:,0], plotter[:,1], s=75, alpha=0.7, hue=color, legend=legend, edgecolor='none')

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
        if legend is not None:
            plt.legend(bbox_to_anchor=(1,1,0.2,0.2), loc='lower left', frameon=False, fontsize='small')
        plt.tight_layout()

        if save_to is None:
            plt.show()
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight', dpi=1000)

        plt.close()


    def plot_barcodes(self, data_type, ranks='all', legend=None, save_to=None, figsize=(5,5)):
        '''
        standard plot of first 2 dimensions of latent space, colored by barcodes
            data_type = one of ['PCA', 't-SNE', 'UMAP'] describing space to plot
            ranks = which barcodes to include as list of indices or strings with barcode IDs
            legend = one of ['brief', 'full', None] describing what kind of legend to show
            save_to = path to .png file to save plot to
            figsize = size in inches of output figure
        '''
        assert self.barcodes is not None, 'Barcodes not assigned.\n'

        plotter = np.ascontiguousarray(self.data[data_type]) # coerce data to np array for plotting

        _, ax = plt.subplots(1, figsize=figsize)

        if ranks == 'all':
            sns.scatterplot(plotter[:,0], plotter[:,1], s=75, alpha=0.7, hue=self.barcodes, legend=legend, edgecolor='none', palette='plasma')

        else:
            ints = [x for x in ranks if type(x)==int] # pull out rank values
            IDs = [x for x in ranks if type(x)==str] # pull out any specific barcode IDs
            ranks_i = self.barcodes.value_counts()[self.barcodes.value_counts().rank(axis=0, method='min', ascending=False).isin(ints)].index
            ranks_codes = self.barcodes[self.barcodes.isin(list(ranks_i) + IDs)] # subset barcodes series
            ranks_results = plotter[self.barcodes.isin(list(ranks_i) + IDs)] # subset results array
            sns.scatterplot(plotter[:,0], plotter[:,1], s=75, alpha=0.1, color='gray', legend=False, edgecolor='none')
            sns.scatterplot(ranks_results[:,0], ranks_results[:,1], s=75, alpha=0.7, hue=ranks_codes, edgecolor='none', palette='plasma')

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
        if legend is not None:
            plt.legend(bbox_to_anchor=(1,1,0.2,0.2), loc='lower left', frameon=False, fontsize='small')
        plt.tight_layout()

        if save_to is None:
            plt.show()
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight')

        plt.close()


    def plot_PCA(self, color=None, save_to=None, figsize=(10,5)):
        '''PCA plot including variance contribution per component'''
        assert self.data['PCA'] is not None, 'PCA not performed.\n'

        if color is None:
            color = self.clu['PCA'].density
        plt.figure(figsize=figsize)

        plt.subplot(121)
        sns.scatterplot(x=self.data['PCA'][:,0], y=self.data['PCA'][:,1], s=75, alpha=0.7, hue=color, legend=None, edgecolor='none')
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.xlabel('PC 1', fontsize=14)
        plt.ylabel('PC 2', fontsize=14)

        plt.subplot(122)
        plt.plot(np.cumsum(np.round(self.PCA_fit.explained_variance_ratio_, decimals=3)*100))
        plt.tick_params(labelsize=12)
        plt.ylabel('% Variance Explained', fontsize=14)
        plt.xlabel('# of Features', fontsize=14)
        sns.despine()

        plt.tight_layout()
        if save_to is None:
            plt.show()
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches='tight')

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
            
        if cells_axis==1:
            data = data.T # transpose matrix if needed
            labels = labels[::-1] # return labels in reverse order to capture cell and feature IDs properly

        if barcodefile: # if barcodes provided, read in file
            barcodes = pd.read_csv(barcodefile, index_col=None, header=None, names=['Barcode'])

        else:
            barcodes = None

        return cls(data=data, data_type=data_type, labels=labels, barcodes=barcodes)


    @classmethod
    def drop_set(cls, counts_obj, drop_index, axis, data_type='counts', num=False):
        '''
        drop cells (axis 0) or genes (axis 1) with a pd.Index list. return RNA_counts object with reduced data.
            counts_obj = RNA_counts object to use as template for new, subsetted RNA_counts object.
            drop_index = list of indices to drop
            axis = 0 to subset cells, 1 to subset genes
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to subset
            num = numerical index (iloc) or index by labels (loc)?
        '''
        if counts_obj.barcodes is not None:
            codes = pd.DataFrame(counts_obj.barcodes)
            codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

        else:
            codes=None

        if not num:
            return cls(counts_obj.data[data_type].drop(drop_index, axis=axis), data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)

        elif axis==1:
            return cls(counts_obj.data[data_type].drop(counts_obj.data.columns[drop_index], axis=axis), data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)

        elif axis==0:
            return cls(counts_obj.data[data_type].drop(counts_obj.data.index[drop_index], axis=axis), data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)


    @classmethod
    def keep_set(cls, counts_obj, keep_index, axis, data_type='counts', num=False):
        '''
        keep cells (axis 0) or genes (axis 1) with a pd.Index list. return RNA_counts object with reduced data.
            counts_obj = RNA_counts object to use as template for new, subsetted RNA_counts object.
            keep_index = list of indices to keep
            axis = 0 to subset cells, 1 to subset genes
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to subset
            num = numerical index (iloc) or index by labels (loc)?
        '''
        if counts_obj.barcodes is not None:
            codes = pd.DataFrame(counts_obj.barcodes)
            codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

        else:
            codes=None

        if not num:
            if axis==0:
                return cls(counts_obj.data[data_type].loc[keep_index,:], data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)

            elif axis==1:
                return cls(counts_obj.data[data_type].loc[:,keep_index], data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)

        else:
            if axis==0:
                return cls(counts_obj.data[data_type].iloc[keep_index,:], data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)

            elif axis==1:
                return cls(counts_obj.data[data_type].iloc[:,keep_index], data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)


    @classmethod
    def nvr_select(cls, counts_obj, parse_noise=True, data_type='counts', **kwargs):
        '''
        use neighborhood variance ratio (NVR) to feature-select RNA_counts object
        return RNA_counts object with reduced data.
            counts_obj = RNA_counts object to use as template for new, feature-selected RNA_counts object
            parse_noise = use pyNVR to get rid of noisy genes first?
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to feature-select
            **kwargs = keyword arguments to pass to arcsinh_norm() function
        '''
        if parse_noise:
            hqGenes = nvr.parseNoise(np.ascontiguousarray(counts_obj.data[data_type])) # identify non-noisy genes
            selected_genes = nvr.select_genes(counts_obj.arcsinh_norm(data_type=data_type, **kwargs)[:,hqGenes]) # select features from arsinh-transformed, non-noisy data

        else:
            selected_genes = nvr.select_genes(counts_obj.arcsinh_norm(data_type=data_type, **kwargs)) # select features from arsinh-transformed, non-noisy data

        if counts_obj.barcodes is not None:
            codes = pd.DataFrame(counts_obj.barcodes)
            codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

        else:
            codes=None

        print('\nSelected {} variable genes\n'.format(selected_genes.shape[0]))
        return cls(counts_obj.data[data_type].iloc[:,selected_genes], data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)


    @classmethod
    def var_select(cls, counts_obj, n_features, data_type='counts'):
        '''
        select n_features (genes) with highest variance across all cells in dataset
        return RNA_counts object with reduced data.
            counts_obj = RNA_counts object to use as template for new, feature-selected RNA_counts object
            n_features = total number of features desired in resulting dataset
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to feature-select
        '''
        v = counts_obj.data[data_type].var(axis=0).nlargest(n_features).index # get top n variant gene IDs

        if counts_obj.barcodes is not None:
            codes = pd.DataFrame(counts_obj.barcodes)
            codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

        else:
            codes=None

        return cls(counts_obj.data[data_type][v], data_type=data_type, labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)


    @classmethod
    def downsample_rand(cls, counts_obj, n_cells, data_type='counts', seed=None):
        '''
        randomly downsample a dataframe of shape (n_cells, n_features) to n_cells and generate new counts object
        return RNA_counts object with reduced data.
            counts_obj = RNA_counts object to use as template for new, subsetted RNA_counts object
            n_cells = total number of cells desired in downsampled RNA_counts object
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to downsample
            seed = random number generator seed for reproducible results
        '''
        np.random.seed(seed) # set seed for reproducible sampling if desired
        cells_out = np.random.choice(counts_obj.data[data_type].shape[0], n_cells, replace=False)

        if counts_obj.barcodes is not None:
            codes = pd.DataFrame(counts_obj.barcodes)
            codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

        else:
            codes=None

        return cls(counts_obj.data[data_type].iloc[cells_out], labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)


    @classmethod
    def downsample_proportional(cls, counts_obj, n_cells, data_type='counts', seed=None):
        '''
        downsample a dataframe of shape (n_cells, n_features) to total n_cells using cluster membership.
        finds proportion of cells in each cluster (DR.clu.membership attribute) and maintains each percentage.
        return RNA_counts object with reduced data.
            counts_obj = RNA_counts object to use as template for new, subsetted RNA_counts object
            n_cells = total number of cells desired in downsampled RNA_counts object
            data_type = one of ['counts', 'PCA', 't-SNE', 'UMAP'] describing data to downsample
            seed = random number generator seed for reproducible results
        '''
        np.random.seed(seed) # set seed for reproducible sampling if desired
        IDs, clu_counts = np.unique(counts_obj.clu[data_type].membership, return_counts=True) # get cluster IDs and number of cells in each

        cells_out = np.array([]) # initialize array of output cell indices
        for ID, count in zip(IDs, clu_counts):
            clu_num = int(count/clu_counts.sum()*n_cells) + 1 # number of cells to sample for given cluster
            cells_out = np.append(cells_out, np.random.choice(np.where(counts_obj.clu[data_type].membership == ID)[0], clu_num, replace=False))

        if counts_obj.barcodes is not None:
            codes = pd.DataFrame(counts_obj.barcodes)
            codes['Cell Barcode'] = codes.index # make barcodes mergeable when calling cls()

        else:
            codes=None

        return cls(counts_obj.data[data_type].iloc[cells_out], labels=[counts_obj.cell_labels, counts_obj.feature_labels], barcodes=codes)



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
        self.grid_x, self.grid_y = np.mgrid[xmin:xmax, ymin:ymax]

        # map beads to pixel grid
        self.pixel_map = scipy.interpolate.griddata(self.data['slide-seq'].values, self.data['slide-seq'].index, (self.grid_x, self.grid_y), method='nearest')


    def trim_pixels(self, threshold=100):
        '''
        trim_pixels
        '''
        tree = scipy.spatial.cKDTree(self.data['slide-seq'].values)
        xi = scipy.interpolate.interpnd._ndim_coords_from_arrays((self.grid_x, self.grid_y), ndim=self.data['slide-seq'].shape[1])
        dists, _ = tree.query(xi)

        self.show_pita(assembled=dists, figsize=(4,4))
        #self.show_pita(assembled=threshold(dists, thresh=threshold, dir='above'), figsize=(4,4))
        self.show_pita(assembled=bin_threshold(dists, threshmax=threshold), figsize=(4,4))

        self.pixel_map_trim = np.copy(self.pixel_map)
        self.pixel_map_trim[dists > threshold] = np.nan


    def assemble_pita(self, feature_type, features, transform=None, trimmed=True, plot_out=True, **kwargs):
        '''
        cast feature into pixel space to construct gene expression image
            feature_type = one of ['counts','PCA','t-SNE','UMAP'] describing data to pull features from
            features = list of names or indices of feature to cast onto bead image
            transform = transform data before generating feature image. One of ['arcsinh','log2',None].
            trimmed = show pixel map output from trim_pixels(), or uncropped map?
            plot_out = show resulting image?
            **kwargs = arguments to pass to show_pita() function
        '''
        # transform data first, if necessary
        if transform is None:
            mapper = pd.DataFrame(self.data[feature_type], index=self.data['counts'].index) # coerce to pd.DF with bead ID index

        if transform == 'arcsinh':
            mapper = pd.DataFrame(self.arcsinh_norm(data_type=feature_type, **kwargs), index=self.data['counts'].index) # coerce to pd.DF with bead ID index

        elif transform == 'log2':
            mapper = pd.DataFrame(self.log2_norm(data_type=feature_type, **kwargs), index=self.data['counts'].index) # coerce to pd.DF with bead ID index

        # determine which reference pixel map to use
        if trimmed:
            assert self.pixel_map_trim is not None, 'Pixel map not trimmed. Run self.trim_pixels(), or set trimmed=False to generate image.\n'
            ref = self.pixel_map_trim

        else:
            ref = self.pixel_map

        # determine how to display results
        # if user wants total of all features, sum them up
        if features == 'total':
            map_features = mapper.sum(axis=1)

        # if 'features' is a string, treat it as regex value
        elif isinstance(features, str):
            map_features = mapper.loc[:,self.feature_IDs.str.contains(features)].sum(axis=1)

        # if 'features' is an integer, treat it as column index
        elif isinstance(features, int):
            map_features = mapper.iloc[:,features]

        # if 'features' is a list of strings, treat them as regex values and return sum of (normalized) expression
        elif all(isinstance(elem, str) for elem in features):
            map_features = mapper.loc[:,self.feature_IDs.str.contains('|'.join(features))].sum(axis=1)

        # if 'features' is a list of integers, treat them as column indices and return sum of (normalized) expression
        elif all(isinstance(elem, int) for elem in features):
            map_features = mapper.iloc[:,features].sum(axis=1)

        else:
            raise ValueError('Please provide features as a list of strings or integers. e.g. [0,3,4], ["Vim","Aldoc","Ttr"].')

        assembled = np.array([map_features.reindex(index=ref[x], copy=True) for x in range(len(ref))])

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
