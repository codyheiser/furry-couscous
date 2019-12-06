# -*- coding: utf-8 -*-
"""
scanpy utility functions

@author: C Heiser
2019
"""
# basics
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.spatial.distance import cdist  # unique pairwise and crosswise distances
from sklearn.neighbors import kneighbors_graph  # simple K-nearest neighbors graph
from sklearn.preprocessing import normalize

# plotting packages
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

sns.set(style="white")


def reorder_adata(adata, descending=True):
    """place cells in descending order of total counts"""
    if descending:
        new_order = np.argsort(adata.X.sum(axis=1))[::-1]
    elif not descending:
        new_order = np.argsort(adata.X.sum(axis=1))[:]
    adata.X = adata.X[new_order, :].copy()
    adata.obs = adata.obs.iloc[new_order].copy()


def arcsinh(adata, layer=None, norm="l1", scale=1000):
    """
    return arcsinh-normalized values for each element in anndata counts matrix
        adata = AnnData object
        layer = name of layer to perform arcsinh-normalization on. if None, use AnnData.X
        norm = normalization strategy prior to Log2 transform.
            None: do not normalize data
            'l1': divide each count by sum of counts for each cell
            'l2': divide each count by sqrt of sum of squares of counts for cell
        scale = factor to scale normalized counts to; default 1000
    """
    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    adata.layers["arcsinh_norm"] = np.arcsinh(normalize(mat, axis=1, norm=norm) * scale)


def gf_icf(adata, layer=None):
    """
    return GF-ICF scores for each element in anndata counts matrix
        adata = AnnData object
        layer = name of layer to perform GF-ICF normalization on. if None, use AnnData.X
    """
    if layer is None:
        tf = adata.X.T / adata.X.sum(axis=1)
        tf = tf.T
        ni = adata.X.astype(bool).sum(axis=0)

        layer = "X"  # set input layer for naming output layer

    else:
        tf = adata.layers[layer].T / adata.layers[layer].sum(axis=1)
        tf = tf.T
        ni = adata.layers[layer].astype(bool).sum(axis=0)

    idf = np.log(adata.n_obs / (ni + 1))

    adata.layers["gf-icf"] = tf * idf


def knn_graph(dist_matrix, k, adata, save_rep="knn"):
    """
    build simple binary k-nearest neighbor graph and add to anndata object
        dist_matrix = distance matrix to calculate knn graph for (i.e. pdist(adata.obsm['X_pca']))
        k = number of nearest neighbors to determine
        adata = AnnData object to add resulting graph to (in .uns slot)
        save_rep = name of .uns key to save knn graph to within adata (default adata.uns['knn'])
    """
    adata.uns[save_rep] = {
        "graph": kneighbors_graph(
            dist_matrix, k, mode="connectivity", include_self=False, n_jobs=-1
        ).toarray(),
        "k": k,
    }


def subset_uns_by_ID(adata, uns_keys, obs_col, IDs):
    """
    subset symmetrical distance matrices and knn graphs in adata.uns by one or more IDs defined in adata.obs
        adata = AnnData object 
        uns_keys = list of keys in adata.uns to subset. new adata.uns keys will be saved with ID appended to name (i.e. adata.uns['knn'] -> adata.uns['knn_ID1'])
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


def recipe_fcc(adata, mito_names="MT-"):
    """
    scanpy preprocessing recipe
        adata = AnnData object with raw counts data in .X 
        mito_names = substring encompassing mitochondrial gene names for calculation of mito expression

    - calculates useful .obs and .var columns ('total_counts', 'pct_counts_mito', 'n_genes_by_counts', etc.)
    - orders cells by total counts
    - store raw counts (adata.layers['raw_counts'])
    - GF-ICF normalization (adata.layers['X_gf-icf'])
    - normalization and arcsinh transformation of counts (adata.layers['arcsinh_norm'])
    - normalization and log1p transformation of counts (adata.X, adata.layers['log1p_norm'])
    - identify highly-variable genes using seurat method (adata.var['highly_variable'])
    """
    reorder_adata(adata, descending=True)  # reorder cells by total counts descending

    # raw
    adata.layers["raw_counts"] = adata.X.copy()  # store raw counts before manipulation

    # obs/var
    adata.var["mito"] = adata.var_names.str.contains(
        mito_names
    )  # identify mitochondrial genes
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito"], inplace=True
    )  # calculate standard qc .obs and .var
    adata.obs["ranked_total_counts"] = np.argsort(
        adata.obs["total_counts"]
    )  # rank cells by total counts

    # arcsinh transform (adata.layers["arcsinh_norm"])
    arcsinh(adata, norm="l1", scale=1000)

    # gf-icf transform (adata.layers["gf-icf"])
    gf_icf(adata, layer=None)

    # log1p transform (adata.layers["log1p_norm"])
    sc.pp.normalize_total(
        adata, target_sum=10000, layers=None, layer_norm=None, key_added="norm_factor"
    )
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()  # save to .layers

    # HVGs
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)


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
                np.mean(adata.X[adata.obs[obs_col].astype(str) == clu, :], axis=0)
                for clu in clu_names
            ]
        )
    else:
        adata.uns["{}_centroids".format(use_rep)] = np.array(
            [
                np.mean(
                    adata.obsm[use_rep][adata.obs[obs_col].astype(str) == clu, :],
                    axis=0,
                )
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


def gf_icf_markers(adata, n_genes=5, group_by="louvain"):
    """
    return n_genes with top gf-icf scores for each group
        adata = AnnData object preprocessed using gf_icf() or recipe_fcc() function
        n_genes = number of top gf-icf scored genes to return per group
        group_by = how to group cells to ID marker genes
    """
    markers = pd.DataFrame()
    for clu in adata.obs[group_by].unique():
        gf_icf_sum = adata.layers["gf-icf"][adata.obs[group_by] == str(clu)].sum(axis=0)
        gf_icf_mean = adata.layers["gf-icf"][adata.obs[group_by] == str(clu)].mean(
            axis=0
        )
        top = np.argpartition(gf_icf_sum, -n_genes)[-n_genes:]
        gene_IDs = adata.var.index[top]
        markers = markers.append(
            pd.DataFrame(
                {
                    group_by: np.repeat(clu, n_genes),
                    "gene": gene_IDs,
                    "gf-icf_sum": gf_icf_sum[top],
                    "gf-icf_mean": gf_icf_mean[top],
                }
            )
        )

    return markers


def cnmf_markers(adata, spectra_score_file, n_genes=30, key="cnmf"):
    """
    read in gene spectra score output from cNMF and save top gene loadings 
    for each usage as dataframe in adata.uns
        adata = AnnData object
        spectra_score_file = '<name>.gene_spectra_score.<k>.<dt>.txt' file from cNMF containing usage gene loadings
        n_genes = number of top genes to list for each usage (rows of df)
        key = prefix of adata.uns keys to save
    """
    # load Z-scored GEPs which reflect gene enrichment, save to adata.uns
    adata.uns["{}_spectra".format(key)] = pd.read_csv(
        spectra_score_file, sep="\t", index_col=0
    ).T
    # obtain top n_genes for each GEP in sorted order and combine them into df
    top_genes = []
    for gep in adata.uns["{}_spectra".format(key)].columns:
        top_genes.append(
            list(
                adata.uns["{}_spectra".format(key)]
                .sort_values(by=gep, ascending=False)
                .index[:n_genes]
            )
        )
    # save output to adata.uns
    adata.uns["{}_markers".format(key)] = pd.DataFrame(
        top_genes, index=adata.uns["{}_spectra".format(key)].columns
    ).T


def rank_genes(
    adata,
    attr="varm",
    keys="usages",
    indices=None,
    labels=None,
    color="black",
    n_points=20,
    log=False,
    show=None,
    figsize=(7, 7),
):
    """
    Plot rankings. [Adapted from scanpy.plotting._anndata.ranking]
    See, for example, how this is used in pl.pca_ranking.
    Parameters
    ----------
    adata : AnnData
        The data.
    attr : {'var', 'obs', 'uns', 'varm', 'obsm'}
        The attribute of AnnData that contains the score.
    keys : str or list of str
        The scores to look up an array from the attribute of adata.
    indices : list of int
        The column indices of keys for which to plot (e.g. [0,1,2] for first three keys)
    Returns
    -------
    Returns matplotlib gridspec with access to the axes.
    """
    # default to all usages
    if indices is None:
        indices = range(getattr(adata, attr)[keys].shape[1])
    # get scores for each usage
    if isinstance(keys, str) and indices is not None:
        scores = np.array(getattr(adata, attr)[keys])[:, indices]
        keys = ["{}{}".format(keys[:-1], i + 1) for i in indices]
    n_panels = len(keys) if isinstance(keys, list) else 1
    if n_panels == 1:
        scores, keys = scores[:, None], [keys]
    if log:
        scores = np.log(scores)
    if labels is None:
        labels = (
            adata.var_names
            if attr in {"var", "varm"}
            else np.arange(scores.shape[0]).astype(str)
        )
    if isinstance(labels, str):
        labels = [labels + str(i + 1) for i in range(scores.shape[0])]
    if n_panels <= 5:
        n_rows, n_cols = 1, n_panels
    else:
        n_rows, n_cols = 2, int(n_panels / 2 + 0.5)
    plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
    left, bottom = 0.2 / n_cols, 0.13 / n_rows
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=0.2,
        left=left,
        bottom=bottom,
        right=1 - (n_cols - 1) * left - 0.01 / n_cols,
        top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
    )
    for iscore, score in enumerate(scores.T):
        plt.subplot(gs[iscore])
        indices = np.argsort(score)[::-1][: n_points + 1]
        for ig, g in enumerate(indices):
            plt.text(
                ig,
                score[g],
                labels[g],
                color=color,
                rotation="vertical",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize="large",
            )
        plt.title(keys[iscore].replace("_", " "), fontsize="x-large")
        plt.xlim(-0.9, ig + 0.9)
        score_min, score_max = np.min(score[indices]), np.max(score[indices])
        plt.ylim(
            (0.95 if score_min > 0 else 1.05) * score_min,
            (1.05 if score_max > 0 else 0.95) * score_max,
        )
        plt.tick_params(labelsize="x-large")
    if show == False:
        return gs


def cnmf_load_results(adata, cnmf_dir, name, k, dt, key="cnmf", n_points=15, **kwargs):
    """
    given adata object and corresponding cNMF output (cnmf_dir, name, k, dt to identify),
    read in relevant results and save to adata object, and output plot of gene loadings
    for each GEP usage.
        adata = AnnData object
        cnmf_dir = relative path to directory containing cNMF outputs
        name = name of cNMF replicate
        k = value used for consensus factorization
        dt = distance threshold value used for consensus clustering
        key = prefix of adata.uns keys to save
        n_points = how many top genes to include in rank_genes() plot
        **kwargs = keyword args to pass to cnmf_markers()
    """
    # read in cell usages
    usage = pd.read_csv(
        "{}/{}/{}.usages.k_{}.dt_{}.consensus.txt".format(
            cnmf_dir, name, name, str(k), str(dt).replace(".", "_")
        ),
        sep="\t",
        index_col=0,
    )
    usage.columns = ["usage_" + str(col) for col in usage.columns]
    usage_norm = usage.div(usage.sum(axis=1), axis=0)
    usage_norm.index = usage_norm.index.astype(str)
    adata.obs = pd.merge(
        left=adata.obs, right=usage_norm, how="left", left_index=True, right_index=True
    )

    # read in overdispersed genes determined by cNMF and add as metadata to adata.var
    overdispersed = np.genfromtxt(
        "{}/{}/{}.overdispersed_genes.txt".format(cnmf_dir, name, name), dtype=str
    )
    adata.var["cnmf_overdispersed"] = 0
    adata.var.loc[
        [x for x in adata.var.index if x in overdispersed], "cnmf_overdispersed"
    ] = 1

    # read top gene loadings for each GEP usage and save to adata.uns['cnmf_markers']
    cnmf_markers(
        adata,
        "{}/{}/{}.gene_spectra_score.k_{}.dt_{}.txt".format(
            cnmf_dir, name, name, str(k), str(dt).replace(".", "_")
        ),
        key=key,
        **kwargs
    )

    # plot gene loadings from consensus file above
    rank_genes(
        adata,
        attr="uns",
        keys="{}_spectra".format(key),
        labels=adata.uns["{}_spectra".format(key)].index,
        n_points=n_points,
    )


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
                c=[cdict[x] for x in adata.obs[obs_col].astype(str)],
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
                    for x in adata.obs.loc[
                        adata.obs[obs_col].isin(IDs), obs_col
                    ].astype(str)
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
