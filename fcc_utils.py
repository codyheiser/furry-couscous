# -*- coding: utf-8 -*-
"""
utility functions

@author: C Heiser
October 2019
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from ot import wasserstein_1d
from scipy.spatial.distance import pdist, cdist
from scipy.stats import pearsonr

import seaborn as sns

sns.set(style="white")


# fuzzy-lasagna functions
def threshold(mat, thresh=0.5, dir="above"):
    """replace all values in a matrix above or below a given threshold with np.nan"""
    a = np.ma.array(mat, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if dir == "above":
        mask |= (a > thresh).filled(False)

    elif dir == "below":
        mask |= (a < thresh).filled(False)

    else:
        raise ValueError("Choose 'above' or 'below' for threshold direction (dir)")

    a[~mask] = np.nan
    return a


def bin_threshold(mat, threshmin=None, threshmax=0.5):
    """
    generate binary segmentation from probabilities
        thresmax = value on [0,1] to assign binary IDs from probabilities. values higher than threshmax -> 1.
            values lower than thresmax -> 0.
    """
    a = np.ma.array(mat, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a < threshmin).filled(False)

    if threshmax is not None:
        mask |= (a > threshmax).filled(False)

    a[mask] = 1
    a[~mask] = 0
    return a


# furry-couscous manuscript functions
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
                "Flattening pre-transformation distance matrix into 1D array..."
            )
        # if symmetric, only keep unique values (above diagonal)
        if np.allclose(pre, pre.T, rtol=1e-05, atol=1e-08):
            pre = pre[np.triu_indices(n=pre.shape[0], k=1)]
        # otherwise, flatten all distances
        else:
            pre = pre.flatten()

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if post.ndim == 2:
        if verbose:
            print(
                "Flattening post-transformation distance matrix into 1D array..."
            )
        # if symmetric, only keep unique values (above diagonal)
        if np.allclose(post, post.T, rtol=1e-05, atol=1e-08):
            post = post[np.triu_indices(n=post.shape[0], k=1)]
        # otherwise, flatten all distances
        else:
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
        native = adata.obsm key or .X containing high-dimensional native space, which should be direct input to dimension reduction
                 that generated latent .obsm for fair comparison. Default 'X', which uses adata.X.
        metric = distance metric to use. one of ['chebyshev','cityblock','euclidean','minkowski','mahalanobis','seuclidean']. default 'euclidean'.
        k = number of nearest neighbors to test preservation
        downsample = number of distances to downsample to (maximum of 50M [~10k cells, if symmetrical] is recommended for performance)
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
        labels = name of pre- and post-transformation spaces for legend (plot_cell_distances, plot_distributions, plot_cumulative_distributions) 
            or axis labels (plot_distance_correlation, joint_plot_distance_correlation) as list of two strings. False to exclude labels.
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
            plt.legend(loc="lower right", fontsize="xx-large")
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
