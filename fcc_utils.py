# -*- coding: utf-8 -*-
"""
utility functions

@author: C Heiser
October 2019
"""
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from ot import wasserstein_1d
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph

import seaborn as sns

sns.set(style="white")


# scanpy utility functions #
def arcsinh(adata, layer=None, scale=1000):
    """
    return arcsinh-normalized values for each element in anndata counts matrix
    l1 normalization (sc.pp.normalize_total) should be performed before this transformation
        adata = AnnData object
        layer = name of lauer to perform arcsinh-normalization on. if None, use AnnData.X
        scale = factor to scale normalized counts to; default 1000
    """
    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    adata.layers["arcsinh_norm"] = np.arcsinh(mat * scale)


def knn_graph(dist_matrix, k, adata, save_rep="knn"):
    """
    build simple binary k-nearest neighbor graph and add to anndata object
        dist_matrix = distance matrix to calculate knn graph for (i.e. pdist(adata.obsm['X_pca']))
        k = number of nearest neighbors to determine
        adata = AnnData object to add resulting graph to (in .uns slot)
        save_rep = name of .uns key to save knn graph to within adata (default adata.uns['knn'])
    """
    adata.uns[save_rep] = kneighbors_graph(
        dist_matrix, k, mode="connectivity", include_self=False
    ).toarray()


def subset_uns_by_ID(adata, uns_keys, obs_col, IDs):
    """
    subset symmetrical distance matrices and knn graphs in adata.uns by one or more IDs defined in adata.obs
        adata = AnnData object 
        uns_keys = list of keys in adata.uns to subset. new adata.uns keys will be saved with ID appended to name
            (i.e. adata.uns['knn'] -> adata.uns['knn_ID1'])
        obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
        IDs = list of IDs to include in subset
    """
    for key in uns_keys:
        tmp = adata.uns[key][
            adata.obs[obs_col].isin(IDs), :
        ]  # subset symmetrical uns matrix along axis 0
        tmp = tmp[
            :, adata.obs[obs_col].isin(IDs)
        ]  # subset symmetrical uns matrix along axis 1

        adata.uns[
            "{}_{}".format(key, "_".join([str(x) for x in IDs]))
        ] = tmp  # save new .uns key by appending IDs to original key name


def find_centroids(adata, use_rep, obs_col="louvain"):
    """
    find cluster centroids
        adata = AnnData object
        use_rep = 'X' or adata.obsm key containing space to calculate centroids in (i.e. 'X_pca')
        obs_col = adata.obs column name containing cluster IDs
    """
    # calculate centroids
    clu_names = adata.obs[obs_col].unique().astype(str)
    if use_rep == "X":
        adata.uns["{}_centroids".format(use_rep)] = np.array(
            [
                np.mean(adata.X[adata.obs[obs_col] == clu, :], axis=0)
                for clu in clu_names
            ]
        )
    else:
        adata.uns["{}_centroids".format(use_rep)] = np.array(
            [
                np.mean(adata.obsm[use_rep][adata.obs[obs_col] == clu, :], axis=0)
                for clu in clu_names
            ]
        )
    # calculate distances between all centroids
    adata.uns["{}_centroid_distances".format(use_rep)] = cdist(
        adata.uns["{}_centroids".format(use_rep)],
        adata.uns["{}_centroids".format(use_rep)],
    )
    # build networkx minimum spanning tree between centroids
    G = nx.from_numpy_matrix(adata.uns["{}_centroid_distances".format(use_rep)])
    G = nx.relabel_nodes(G, mapping=dict(zip(list(G.nodes), clu_names)), copy=True)
    adata.uns["{}_centroid_MST".format(use_rep)] = nx.minimum_spanning_tree(G)


# dimensionality reduction plotting class #
class DR_plot:
    """
    class defining pretty plots of dimension-reduced embeddings such as PCA, t-SNE, and UMAP
        DR_plot().plot(): utility plotting function that can be passed any numpy array in the `data` parameter
                 .plot_IDs(): plot one or more cluster IDs on top of an .obsm from an `AnnData` object
                 .plot_centroids(): plot cluster centroids defined using find_centroids() function on `AnnData` object
    """

    def __init__(self, dim_name="dim", figsize=(5, 5), ax_labels=True):
        """
        dim_name = how to label axes ('dim 1' on x and 'dim 2' on y by default)
        figsize = size of resulting axes
        ax_labels = draw arrows and dimension names in lower left corner of plot
        """
        self.fig, self.ax = plt.subplots(1, figsize=figsize)
        self.cmap = plt.get_cmap("plasma")

        if ax_labels:
            plt.xlabel("{} 1".format(dim_name), fontsize=14)
            self.ax.xaxis.set_label_coords(0.2, -0.025)
            plt.ylabel("{} 2".format(dim_name), fontsize=14)
            self.ax.yaxis.set_label_coords(-0.025, 0.2)

            plt.annotate(
                "",
                textcoords="axes fraction",
                xycoords="axes fraction",
                xy=(-0.006, 0),
                xytext=(0.2, 0),
                arrowprops=dict(arrowstyle="<-", lw=2, color="black"),
            )
            plt.annotate(
                "",
                textcoords="axes fraction",
                xycoords="axes fraction",
                xy=(0, -0.006),
                xytext=(0, 0.2),
                arrowprops=dict(arrowstyle="<-", lw=2, color="black"),
            )

        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

    def plot(self, data, color, pt_size=75, legend=None, save_to=None):
        """
        general plotting function for dimensionality reduction outputs with cute arrows and labels
            data = np.array containing variables in columns and observations in rows
            color = list of length nrow(data) to determine how points should be colored
            pt_size = size of points in plot
            legend = None, 'full', or 'brief'
            save_to = path to .png file to save output, or None
        """
        sns.scatterplot(
            data[:, 0],
            data[:, 1],
            s=pt_size,
            alpha=0.7,
            hue=color,
            legend=legend,
            edgecolor="none",
            ax=self.ax,
        )

        if legend is not None:
            plt.legend(
                bbox_to_anchor=(1, 1, 0.2, 0.2),
                loc="lower left",
                frameon=False,
                fontsize="small",
            )

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_IDs(self, adata, use_rep, obs_col, IDs="all", pt_size=75, save_to=None):
        """
        general plotting function for dimensionality reduction outputs with cute arrows and labels
            adata = anndata object to pull dimensionality reduction from
            use_rep = adata.obsm key to plot from (i.e. 'X_pca')
            obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
            IDs = list of IDs to plot, graying out cells not assigned to those IDS (default 'all' IDs)
            pt_size = size of points in plot
            save_to = path to .png file to save output, or None
        """
        plotter = adata.obsm[use_rep]
        # get color mapping from obs_col
        clu_names = adata.obs[obs_col].unique().astype(str)
        colors = self.cmap(np.linspace(0, 1, len(clu_names)))
        cdict = dict(zip(clu_names, colors))

        if IDs == "all":
            self.ax.scatter(
                plotter[:, 0],
                plotter[:, 1],
                s=pt_size,
                alpha=0.7,
                c=[cdict[x] for x in adata.obs[obs_col]],
                edgecolor="none",
            )

        else:
            sns.scatterplot(
                plotter[-adata.obs[obs_col].isin(IDs), 0],
                plotter[-adata.obs[obs_col].isin(IDs), 1],
                ax=self.ax,
                s=pt_size,
                alpha=0.1,
                color="gray",
                legend=False,
                edgecolor="none",
            )
            plt.scatter(
                plotter[adata.obs[obs_col].isin(IDs), 0],
                plotter[adata.obs[obs_col].isin(IDs), 1],
                s=pt_size,
                alpha=0.7,
                c=[
                    cdict[x]
                    for x in adata.obs.loc[adata.obs[obs_col].isin(IDs), obs_col]
                ],
                edgecolor="none",
            )

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_centroids(
        self,
        adata,
        use_rep,
        obs_col,
        ctr_size=300,
        pt_size=75,
        draw_edges=True,
        highlight_edges=False,
        save_to=None,
    ):
        """
        general plotting function for dimensionality reduction outputs with cute arrows and labels
            adata = anndata object to pull dimensionality reduction from
            use_rep = adata.obsm key to plot from (i.e. 'X_pca')
            obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
            ctr_size = size of centroid points in plot
            pt_size = size of points in plot
            draw_edges = draw edges of minimum spanning tree between all centroids?
            highlight_edges = list of edge IDs as tuples to highlight in red on plot
                e.g. `set(adata.uns['X_tsne_centroid_MST'].edges).difference(set(adata.uns['X_umap_centroid_MST'].edges))`
                with output {(0,3), (0,7)} says that edges from centroid 0 to 3 and 0 to 7 are found in 'X_tsne_centroids'
                but not in 'X_umap_centroids'. highlight the edges to show this.
            save_to = path to .png file to save output, or None
        """
        # get color mapping from obs_col
        clu_names = adata.obs[obs_col].unique().astype(str)
        colors = self.cmap(np.linspace(0, 1, len(clu_names)))

        # draw points in embedding first
        sns.scatterplot(
            adata.obsm[use_rep][:, 0],
            adata.obsm[use_rep][:, 1],
            ax=self.ax,
            s=pt_size,
            alpha=0.1,
            color="gray",
            legend=False,
            edgecolor="none",
        )

        # draw MST edges if desired, otherwise just draw centroids
        if not draw_edges:
            self.ax.scatter(
                adata.uns["{}_centroids".format(use_rep)][:, 0],
                adata.uns["{}_centroids".format(use_rep)][:, 1],
                s=ctr_size,
                c=colors,
                edgecolor="none",
            )
        else:
            pos = dict(zip(clu_names, adata.uns["{}_centroids".format(use_rep)][:, :2]))
            nx.draw_networkx(
                adata.uns["{}_centroid_MST".format(use_rep)],
                pos=pos,
                ax=self.ax,
                with_labels=False,
                width=2,
                node_size=ctr_size,
                node_color=colors,
            )
            # highlight edges if desired
            if highlight_edges:
                nx.draw_networkx_edges(
                    adata.uns["{}_centroid_MST".format(use_rep)],
                    pos=pos,
                    ax=self.ax,
                    edgelist=highlight_edges,
                    width=5,
                    edge_color="red",
                )

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)


# structural preservation utility functions #
def distance_stats(pre, post, downsample=False, verbose=True):
    """
    test for correlation between Euclidean cell-cell distances before and after transformation by a function or DR algorithm.
    1) performs Pearson correlation of distance distributions
    2) normalizes unique distances using min-max standardization for each dataset
    3) calculates Wasserstein or Earth-Mover's Distance for normalized distance distributions between datasets
        pre = vector of unique distances (pdist()) or distance matrix of shape (n_cells, m_cells)
            (cdist()) before transformation/projection
        post = vector of unique distances (pdist()) distance matrix of shape (n_cells, m_cells)
            (cdist()) after transformation/projection
        downsample = number of distances to downsample to
            (maximum of 50M [~10k cells, if symmetrical] is recommended for performance)
        verbose = print progress statements
    """
    # make sure the number of cells in each matrix is the same
    assert (
        pre.shape == post.shape
    ), 'Matrices contain different number of distances.\n{} in "pre"\n{} in "post"\n'.format(
        pre.shape[0], post.shape[0]
    )

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if pre.ndim == 2:
        if verbose:
            print(
                "Flattening distance matrices into 1D array of unique cell-cell distances..."
            )
        pre = pre.flatten()
        post = post.flatten()

    # if dataset is large, randomly downsample to reasonable number of cells for calculation
    if downsample:
        assert downsample < len(
            pre
        ), "Must provide downsample value smaller than total number of cell-cell distances provided in pre and post"
        if verbose:
            print("Downsampling to {} total cell-cell distances...".format(downsample))
        idx = np.random.choice(np.arange(len(pre)), downsample, replace=False)
        pre = pre[idx]
        post = post[idx]

    # calculate correlation coefficient using Pearson correlation
    if verbose:
        print("Correlating distances")
    corr_stats = pearsonr(x=pre, y=post)

    # min-max normalization for fair comparison of probability distributions
    if verbose:
        print("Normalizing unique distances")
    pre -= pre.min()
    pre /= pre.ptp()

    post -= post.min()
    post /= post.ptp()

    # calculate EMD for the distance matrices
    # by default, downsample to 50M distances to speed processing time,
    # since this function often breaks with larger distributions
    if verbose:
        print("Calculating Earth-Mover's Distance between distributions")
    if len(pre) > 50000000:
        idx = np.random.choice(np.arange(len(pre)), 50000000, replace=False)
        pre_EMD = pre[idx]
        post_EMD = post[idx]
        EMD = wasserstein_1d(pre_EMD, post_EMD)
    else:
        EMD = wasserstein_1d(pre, post)

    return pre, post, corr_stats, EMD


def knn_preservation(pre, post):
    """
    test for k-nearest neighbor preservation (%) before and after transformation by a function or DR algorithm.
        pre = Knn graph of shape (n_cells, n_cells) before transformation/projection
        post = Knn graph of shape (n_cells, n_cells) after transformation/projection
    """
    # make sure the number of cells in each matrix is the same
    assert (
        pre.shape == post.shape
    ), 'Matrices contain different number of cells.\n{} in "pre"\n{} in "post"\n'.format(
        pre.shape[0], post.shape[0]
    )
    return np.round(100 - ((pre != post).sum() / (pre.shape[0] ** 2)) * 100, 4)


def structure_preservation_sc(
    adata,
    latent,
    native="X",
    metric="euclidean",
    k=30,
    downsample=False,
    verbose=True,
    force_recalc=False,
):
    """
    wrapper function for full structural preservation workflow applied to scanpy AnnData object
        adata = AnnData object with latent space to test in .obsm slot, and native (reference) space in .X or .obsm
        latent = adata.obsm key that contains low-dimensional latent space for testing
        native = adata.obsm key or .X containing high-dimensional native space, which should be direct input to dimension
            reduction that generated latent .obsm for fair comparison. Default 'X', which uses adata.X.
        metric = distance metric to use. one of ['chebyshev','cityblock','euclidean','minkowski','mahalanobis','seuclidean'].
            default 'euclidean'.
        k = number of nearest neighbors to test preservation
        downsample = number of distances to downsample to
            (maximum of 50M [~10k cells, if symmetrical] is recommended for performance)
        verbose = print progress statements
        force_recalc = if True, recalculate all distances and neighbor graphs, regardless of their presence in AnnData object
    """
    # 0) determine native space according to argument
    if native == "X":
        native_space = adata.X.copy()
    else:
        native_space = adata.obsm[native].copy()

    # 1) calculate unique cell-cell distances
    if (
        "{}_distances".format(native) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print("Calculating unique distances for native space, {}".format(native))
        adata.uns["{}_distances".format(native)] = pdist(native_space, metric=metric)

    if (
        "{}_distances".format(latent) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print("Calculating unique distances for latent space, {}".format(latent))
        adata.uns["{}_distances".format(latent)] = pdist(
            adata.obsm[latent], metric=metric
        )

    # 2) get correlation and EMD values, and return normalized distance vectors for plotting distributions
    (
        adata.uns["{}_norm_distances".format(native)],
        adata.uns["{}_norm_distances".format(latent)],
        corr_stats,
        EMD,
    ) = distance_stats(
        pre=adata.uns["{}_distances".format(native)].copy(),
        post=adata.uns["{}_distances".format(latent)].copy(),
        verbose=verbose,
        downsample=downsample,
    )

    # 3) determine neighbors
    if (
        "{}_neighbors".format(native) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print(
                "{}-nearest neighbor calculation for native space, {}".format(k, native)
            )
        adata.uns["{}_neighbors".format(native)] = sc.pp.neighbors(
            adata,
            n_neighbors=k,
            use_rep=native,
            knn=True,
            metric="euclidean",
            copy=True,
        ).uns["neighbors"]

    if (
        "{}_neighbors".format(latent) not in adata.uns.keys() or force_recalc
    ):  # check for existence in AnnData to prevent re-work
        if verbose:
            print(
                "{}-nearest neighbor calculation for latent space, {}".format(k, latent)
            )
        adata.uns["{}_neighbors".format(latent)] = sc.pp.neighbors(
            adata,
            n_neighbors=k,
            use_rep=latent,
            knn=True,
            metric="euclidean",
            copy=True,
        ).uns["neighbors"]

    # 4) calculate neighbor preservation
    if verbose:
        print("Determining nearest neighbor preservation")
    if (
        adata.uns["{}_neighbors".format(native)]["params"]["n_neighbors"]
        != adata.uns["{}_neighbors".format(latent)]["params"]["n_neighbors"]
    ):
        warnings.warn(
            'Warning: Nearest-neighbor graphs constructed with different k values. k={} in "{}_neighbors", while k={} in "{}_neighbors". Consider re-generating neighbors graphs by setting force_recalc=True.'.format(
                adata.uns["{}_neighbors".format(native)]["params"]["n_neighbors"],
                native,
                adata.uns["{}_neighbors".format(latent)]["params"]["n_neighbors"],
                latent,
            )
        )
    knn_pres = knn_preservation(
        pre=adata.uns["{}_neighbors".format(native)]["distances"],
        post=adata.uns["{}_neighbors".format(latent)]["distances"],
    )

    if verbose:
        print("Done!")
    return corr_stats, EMD, knn_pres


# structure preservation plotting class #
class SP_plot:
    """
    class defining pretty plots of structural evaluation of dimension-reduced embeddings such as PCA, t-SNE, and UMAP
    """

    def __init__(
        self, pre_norm, post_norm, figsize=(4, 4), labels=["Native", "Latent"]
    ):
        """
        pre_norm = flattened vector of normalized, unique cell-cell distances "pre-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        post_norm = flattened vector of normalized, unique cell-cell distances "post-transformation".
            Upper triangle of cell-cell distance matrix, flattened to vector of shape ((n_cells^2)/2)-n_cells.
        figsize = size of resulting axes
        labels = name of pre- and post-transformation spaces for legend (plot_cell_distances, plot_distributions,
            plot_cumulative_distributions) or axis labels (plot_distance_correlation, joint_plot_distance_correlation)
            as list of two strings. False to exclude labels.
        """
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(1, figsize=self.figsize)
        self.palette = sns.cubehelix_palette()
        self.cmap = sns.cubehelix_palette(as_cmap=True)
        self.pre = pre_norm
        self.post = post_norm
        self.labels = labels

        plt.tick_params(labelbottom=False, labelleft=False)
        sns.despine()
        plt.tight_layout()

    def plot_cell_distances(self, legend=True, save_to=None):
        """
        plot all unique cell-cell distances before and after some transformation.
            legend = display legend on plot
            save_to = path to .png file to save output, or None
        """
        plt.plot(self.pre, alpha=0.7, label=self.labels[0], color=self.palette[-1])
        plt.plot(self.post, alpha=0.7, label=self.labels[1], color=self.palette[2])
        if legend:
            plt.legend(loc="best", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_distributions(self, legend=True, save_to=None):
        """
        plot probability distributions for all unique cell-cell distances before and after some transformation.
            legend = display legend on plot
            save_to = path to .png file to save output, or None
        """
        sns.distplot(
            self.pre, hist=False, kde=True, label=self.labels[0], color=self.palette[-1]
        )
        sns.distplot(
            self.post, hist=False, kde=True, label=self.labels[1], color=self.palette[2]
        )
        if legend:
            plt.legend(loc="best", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_cumulative_distributions(self, legend=True, save_to=None):
        """
        plot cumulative probability distributions for all unique cell-cell distances before and after some transformation.
            legend = display legend on plot
            save_to = path to .png file to save output, or None
        """
        num_bins = int(len(self.pre) / 100)
        pre_counts, pre_bin_edges = np.histogram(self.pre, bins=num_bins)
        pre_cdf = np.cumsum(pre_counts)
        post_counts, post_bin_edges = np.histogram(self.post, bins=num_bins)
        post_cdf = np.cumsum(post_counts)
        plt.plot(
            pre_bin_edges[1:],
            pre_cdf / pre_cdf[-1],
            label=self.labels[0],
            color=self.palette[-1],
        )
        plt.plot(
            post_bin_edges[1:],
            post_cdf / post_cdf[-1],
            label=self.labels[1],
            color=self.palette[2],
        )
        if legend:
            plt.legend(loc="best", fontsize="xx-large")
        else:
            plt.legend()
            self.ax.legend().remove()

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def plot_distance_correlation(self, save_to=None):
        """
        plot correlation of all unique cell-cell distances before and after some transformation.
            save_to = path to .png file to save output, or None
        """
        plt.hist2d(x=self.pre, y=self.post, bins=50, cmap=self.cmap)
        plt.plot(
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            linestyle="dashed",
            color=self.palette[-1],
        )  # plot identity line as reference for regression
        if self.labels:
            plt.xlabel(self.labels[0], fontsize="xx-large", color=self.palette[-1])
            plt.ylabel(self.labels[1], fontsize="xx-large", color=self.palette[2])

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)

    def joint_plot_distance_correlation(self, save_to=None):
        """
        plot correlation of all unique cell-cell distances before and after some transformation. 
        includes marginal plots of each distribution.
            save_to = path to .png file to save output, or None
        """
        plt.close()  # close matplotlib figure from __init__() and start over with seaborn.JointGrid()
        self.fig = sns.JointGrid(
            x=self.pre, y=self.post, space=0, height=self.figsize[0]
        )
        self.fig.plot_joint(plt.hist2d, bins=50, cmap=self.cmap)
        sns.kdeplot(
            self.pre,
            color=self.palette[-1],
            shade=False,
            bw=0.01,
            ax=self.fig.ax_marg_x,
        )
        sns.kdeplot(
            self.post,
            color=self.palette[2],
            shade=False,
            bw=0.01,
            vertical=True,
            ax=self.fig.ax_marg_y,
        )
        self.fig.ax_joint.plot(
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            np.linspace(max(min(self.pre), min(self.post)), 1, 100),
            linestyle="dashed",
            color=self.palette[-1],
        )  # plot identity line as reference for regression
        if self.labels:
            plt.xlabel(self.labels[0], fontsize="xx-large", color=self.palette[-1])
            plt.ylabel(self.labels[1], fontsize="xx-large", color=self.palette[2])

        plt.tick_params(labelbottom=False, labelleft=False)

        if save_to is None:
            return
        else:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=1000)


def cluster_arrangement(
    pre_obj,
    post_obj,
    clusters,
    cluster_names=None,
    figsize=(6, 6),
    pre_transform="arcsinh",
    legend=True,
    ax_labels=["Native", "Latent"],
):
    """
    determine pairwise distance preservation between 3 clusters
        pre_obj = RNA_counts object
        post_obj = DR object
        clusters = list of barcode IDs i.e. ['0','1','2'] to calculate pairwise distances between clusters 0, 1 and 2
        cluster_names = list of cluster names for labeling i.e. ['Bipolar Cells','Rods','Amacrine Cells'] for clusters
            0, 1 and 2, respectively
        figsize = size of output figure to plot
        pre_transform = apply transformation to pre_obj counts? (None, 'arcsinh', or 'log2')
        legend = display legend on plot
        ax_labels = list of two strings for x and y axis labels, respectively. if False, exclude axis labels.
    """
    # distance calculations for pre_obj
    dist_0_1 = pre_obj.barcode_distance_matrix(
        ranks=[clusters[0], clusters[1]], transform=pre_transform
    ).flatten()
    dist_0_2 = pre_obj.barcode_distance_matrix(
        ranks=[clusters[0], clusters[2]], transform=pre_transform
    ).flatten()
    dist_1_2 = pre_obj.barcode_distance_matrix(
        ranks=[clusters[1], clusters[2]], transform=pre_transform
    ).flatten()
    # combine and min-max normalize
    dist = np.append(np.append(dist_0_1, dist_0_2), dist_1_2)
    dist -= dist.min()
    dist /= dist.ptp()
    # split normalized distances by cluster pair
    dist_norm_0_1 = dist[: dist_0_1.shape[0]]
    dist_norm_0_2 = dist[dist_0_1.shape[0] : dist_0_1.shape[0] + dist_0_2.shape[0]]
    dist_norm_1_2 = dist[dist_0_1.shape[0] + dist_0_2.shape[0] :]

    # distance calculations for post_obj
    post_0_1 = post_obj.barcode_distance_matrix(
        ranks=[clusters[0], clusters[1]]
    ).flatten()
    post_0_2 = post_obj.barcode_distance_matrix(
        ranks=[clusters[0], clusters[2]]
    ).flatten()
    post_1_2 = post_obj.barcode_distance_matrix(
        ranks=[clusters[1], clusters[2]]
    ).flatten()
    # combine and min-max normalize
    post = np.append(np.append(post_0_1, post_0_2), post_1_2)
    post -= post.min()
    post /= post.ptp()
    # split normalized distances by cluster pair
    post_norm_0_1 = post[: post_0_1.shape[0]]
    post_norm_0_2 = post[post_0_1.shape[0] : post_0_1.shape[0] + post_0_2.shape[0]]
    post_norm_1_2 = post[post_0_1.shape[0] + post_0_2.shape[0] :]

    # calculate EMD and Pearson correlation stats
    EMD = [
        wasserstein_1d(dist_norm_0_1, post_norm_0_1),
        wasserstein_1d(dist_norm_0_2, post_norm_0_2),
        wasserstein_1d(dist_norm_1_2, post_norm_1_2),
    ]
    corr_stats = [
        pearsonr(x=dist_0_1, y=post_0_1)[0],
        pearsonr(x=dist_0_2, y=post_0_2)[0],
        pearsonr(x=dist_1_2, y=post_1_2)[0],
    ]

    if cluster_names is None:
        cluster_names = clusters.copy()

    # generate jointplot
    g = sns.JointGrid(x=dist, y=post, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(
        dist_norm_0_1,
        shade=False,
        bw=0.01,
        ax=g.ax_marg_x,
        color="darkorange",
        label=cluster_names[0] + " - " + cluster_names[1],
        legend=legend,
    )
    sns.kdeplot(
        dist_norm_0_2,
        shade=False,
        bw=0.01,
        ax=g.ax_marg_x,
        color="darkgreen",
        label=cluster_names[0] + " - " + cluster_names[2],
        legend=legend,
    )
    sns.kdeplot(
        dist_norm_1_2,
        shade=False,
        bw=0.01,
        ax=g.ax_marg_x,
        color="darkred",
        label=cluster_names[1] + " - " + cluster_names[2],
        legend=legend,
    )
    if legend:
        g.ax_marg_x.legend(loc=(1.01, 0.1))

    sns.kdeplot(
        post_norm_0_1,
        shade=False,
        bw=0.01,
        vertical=True,
        color="darkorange",
        ax=g.ax_marg_y,
    )
    sns.kdeplot(
        post_norm_0_2,
        shade=False,
        bw=0.01,
        vertical=True,
        color="darkgreen",
        ax=g.ax_marg_y,
    )
    sns.kdeplot(
        post_norm_1_2,
        shade=False,
        bw=0.01,
        vertical=True,
        color="darkred",
        ax=g.ax_marg_y,
    )
    g.ax_joint.plot(
        np.linspace(max(min(dist), min(post)), 1, 100),
        np.linspace(max(min(dist), min(post)), 1, 100),
        linestyle="dashed",
        color=sns.cubehelix_palette()[-1],
    )  # plot identity line as reference for regression
    if ax_labels:
        plt.xlabel(ax_labels[0], fontsize="xx-large", color=sns.cubehelix_palette()[-1])
        plt.ylabel(ax_labels[1], fontsize="xx-large", color=sns.cubehelix_palette()[2])

    plt.tick_params(labelleft=False, labelbottom=False)

    return corr_stats, EMD


def cluster_arrangement_sc(
    adata,
    pre,
    post,
    obs_col,
    IDs,
    ID_names=None,
    figsize=(6, 6),
    legend=True,
    ax_labels=["Native", "Latent"],
):
    """
    determine pairwise distance preservation between 3 IDs from adata.obs[obs_col]
        adata = anndata object to pull dimensionality reduction from
        pre = matrix to subset as pre-transformation (i.e. adata.X)
        post = matrix to subset as pre-transformation (i.e. adata.obsm['X_pca'])
        obs_col = name of column in adata.obs to use as cell IDs (i.e. 'louvain')
        IDs = list of THREE IDs to compare (i.e. [0,1,2])
        figsize = size of resulting axes
        legend = display legend on plot
        ax_labels = list of two strings for x and y axis labels, respectively. if False, exclude axis labels.
    """
    # distance calculations for pre_obj
    dist_0_1 = cdist(
        pre[adata.obs[obs_col] == IDs[0]], pre[adata.obs[obs_col] == IDs[1]]
    ).flatten()
    dist_0_2 = cdist(
        pre[adata.obs[obs_col] == IDs[0]], pre[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    dist_1_2 = cdist(
        pre[adata.obs[obs_col] == IDs[1]], pre[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    # combine and min-max normalize
    dist = np.append(np.append(dist_0_1, dist_0_2), dist_1_2)
    dist -= dist.min()
    dist /= dist.ptp()
    # split normalized distances by cluster pair
    dist_norm_0_1 = dist[: dist_0_1.shape[0]]
    dist_norm_0_2 = dist[dist_0_1.shape[0] : dist_0_1.shape[0] + dist_0_2.shape[0]]
    dist_norm_1_2 = dist[dist_0_1.shape[0] + dist_0_2.shape[0] :]

    # distance calculations for post_obj
    post_0_1 = cdist(
        post[adata.obs[obs_col] == IDs[0]], post[adata.obs[obs_col] == IDs[1]]
    ).flatten()
    post_0_2 = cdist(
        post[adata.obs[obs_col] == IDs[0]], post[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    post_1_2 = cdist(
        post[adata.obs[obs_col] == IDs[1]], post[adata.obs[obs_col] == IDs[2]]
    ).flatten()
    # combine and min-max normalize
    post = np.append(np.append(post_0_1, post_0_2), post_1_2)
    post -= post.min()
    post /= post.ptp()
    # split normalized distances by cluster pair
    post_norm_0_1 = post[: post_0_1.shape[0]]
    post_norm_0_2 = post[post_0_1.shape[0] : post_0_1.shape[0] + post_0_2.shape[0]]
    post_norm_1_2 = post[post_0_1.shape[0] + post_0_2.shape[0] :]

    # calculate EMD and Pearson correlation stats
    EMD = [
        wasserstein_1d(dist_norm_0_1, post_norm_0_1),
        wasserstein_1d(dist_norm_0_2, post_norm_0_2),
        wasserstein_1d(dist_norm_1_2, post_norm_1_2),
    ]
    corr_stats = [
        pearsonr(x=dist_0_1, y=post_0_1)[0],
        pearsonr(x=dist_0_2, y=post_0_2)[0],
        pearsonr(x=dist_1_2, y=post_1_2)[0],
    ]

    if ID_names is None:
        ID_names = IDs.copy()

    # generate jointplot
    g = sns.JointGrid(x=dist, y=post, space=0, height=figsize[0])
    g.plot_joint(plt.hist2d, bins=50, cmap=sns.cubehelix_palette(as_cmap=True))
    sns.kdeplot(
        dist_norm_0_1,
        shade=False,
        bw=0.01,
        ax=g.ax_marg_x,
        color="darkorange",
        label=ID_names[0] + " - " + ID_names[1],
        legend=legend,
    )
    sns.kdeplot(
        dist_norm_0_2,
        shade=False,
        bw=0.01,
        ax=g.ax_marg_x,
        color="darkgreen",
        label=ID_names[0] + " - " + ID_names[2],
        legend=legend,
    )
    sns.kdeplot(
        dist_norm_1_2,
        shade=False,
        bw=0.01,
        ax=g.ax_marg_x,
        color="darkred",
        label=ID_names[1] + " - " + ID_names[2],
        legend=legend,
    )
    if legend:
        g.ax_marg_x.legend(loc=(1.01, 0.1))

    sns.kdeplot(
        post_norm_0_1,
        shade=False,
        bw=0.01,
        vertical=True,
        color="darkorange",
        ax=g.ax_marg_y,
    )
    sns.kdeplot(
        post_norm_0_2,
        shade=False,
        bw=0.01,
        vertical=True,
        color="darkgreen",
        ax=g.ax_marg_y,
    )
    sns.kdeplot(
        post_norm_1_2,
        shade=False,
        bw=0.01,
        vertical=True,
        color="darkred",
        ax=g.ax_marg_y,
    )
    g.ax_joint.plot(
        np.linspace(max(dist.min(), post.min()), 1, 100),
        np.linspace(max(dist.min(), post.min()), 1, 100),
        linestyle="dashed",
        color=sns.cubehelix_palette()[-1],
    )  # plot identity line as reference for regression
    if ax_labels:
        plt.xlabel(ax_labels[0], fontsize="xx-large", color=sns.cubehelix_palette()[-1])
        plt.ylabel(ax_labels[1], fontsize="xx-large", color=sns.cubehelix_palette()[2])

    plt.tick_params(labelleft=False, labelbottom=False)

    return corr_stats, EMD
