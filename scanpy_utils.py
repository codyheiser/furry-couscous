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
    """
    place cells in descending order of total counts
    
    Parameters:
        adata (AnnData.AnnData): AnnData object
        descending (bool): highest counts first

    Returns:
        AnnData.AnnData: adata cells are reordered in place
    """
    if descending:
        new_order = np.argsort(adata.X.sum(axis=1))[::-1]
    elif not descending:
        new_order = np.argsort(adata.X.sum(axis=1))[:]
    adata.X = adata.X[new_order, :].copy()
    adata.obs = adata.obs.iloc[new_order].copy()


def arcsinh_norm(adata, layer=None, norm="l1", scale=1e6):
    """
    return arcsinh-normalized values for each element in anndata counts matrix

    Parameters:
        adata (AnnData.AnnData): AnnData object
        layer (str or None): name of layer to perform arcsinh-normalization on. if None, use AnnData.X
        norm (str or None): normalization strategy following GF-ICF transform.
            None: do not normalize counts
            "l1": divide each count by sum of counts for each cell (analogous to sc.pp.normalize_total)
            "l2": divide each count by sqrt of sum of squares of counts for each cell
        scale (int): factor to scale normalized counts to; default 1e6 for TPM

    Returns:
        AnnData.AnnData: adata is edited in place to add arcsinh normalization to .layers
    """
    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    if norm is None:
        adata.layers["arcsinh_norm"] = np.arcsinh(mat * scale)
    else:
        adata.layers["arcsinh_norm"] = np.arcsinh(
            normalize(mat, axis=1, norm=norm) * scale
        )


def gf_icf(adata, layer=None, transform="arcsinh", norm=None):
    """
    return GF-ICF scores for each element in anndata counts matrix

    Parameters:
        adata (AnnData.AnnData): AnnData object
        layer (str or None): name of layer to perform GF-ICF normalization on. if None, use AnnData.X
        transform (str): how to transform ICF weights. arcsinh is recommended to retain counts of genes
            expressed in all cells. log transform eliminates these genes from the dataset.
        norm (str or None): normalization strategy following GF-ICF transform.
            None: do not normalize GF-ICF scores
            "l1": divide each score by sum of scores for each cell (analogous to sc.pp.normalize_total)
            "l2": divide each score by sqrt of sum of squares of scores for each cell

    Returns:
        AnnData.AnnData: adata is edited in place to add GF-ICF normalization to .layers["gf_icf"]
    """
    if layer is None:
        m = adata.X
    else:
        m = adata.layers[layer]

    # number of cells containing each gene (sum nonzero along columns)
    nt = m.astype(bool).sum(axis=0)
    assert np.all(
        nt
    ), "Encountered {} genes with 0 cells by counts. Remove these before proceeding (i.e. sc.pp.filter_genes(adata,min_cells=1))".format(
        np.size(nt) - np.count_nonzero(nt)
    )
    # gene frequency in each cell (l1 norm along rows)
    tf = m / m.sum(axis=1)[:, None]

    # inverse cell frequency (total cells / number of cells containing each gene)
    if transform == "arcsinh":
        idf = np.arcsinh(adata.n_obs / nt)
    elif transform == "log":
        idf = np.log(adata.n_obs / nt)
    else:
        raise ValueError("Please provide a valid transform (log or arcsinh).")

    # save GF-ICF scores to .layers and total GF-ICF per cell in .obs
    tf_idf = tf * idf
    adata.obs["gf_icf_total"] = tf_idf.sum(axis=1)
    if norm is None:
        adata.layers["gf_icf"] = tf_idf
    else:
        adata.layers["gf_icf"] = normalize(tf_idf, norm=norm, axis=1)


def recipe_fcc(
    adata, X_final="raw_counts", mito_names="MT-", target_sum=1e6, n_hvgs=2000
):
    """
    scanpy preprocessing recipe

    Parameters:
        adata (AnnData.AnnData): object with raw counts data in .X
        X_final (str): which normalization should be left in .X slot?
            ("raw_counts","gf_icf","log1p_norm","arcsinh_norm")
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        target_sum (int): total sum of counts for each cell prior to arcsinh 
            and log1p transformations; default 1e6 for TPM
        n_hvgs (int): number of HVGs to calculate using Seurat method

    Returns:
        AnnData.AnnData: adata is edited in place to include:
        - useful .obs and .var columns
            ("total_counts", "pct_counts_mito", "n_genes_by_counts", etc.)
        - cells ordered by "total_counts"
        - raw counts (adata.layers["raw_counts"])
        - GF-ICF transformation of counts (adata.layers["gf_icf"])
        - arcsinh transformation of normalized counts
            (adata.layers["arcsinh_norm"])
        - log1p transformation of normalized counts
            (adata.X, adata.layers["log1p_norm"])
        - highly variable genes (adata.var["highly_variable"])
    """
    reorder_adata(adata, descending=True)  # reorder cells by total counts descending

    # store raw counts before manipulation
    adata.layers["raw_counts"] = adata.X.copy()

    # identify mitochondrial genes
    adata.var["mito"] = adata.var_names.str.contains(mito_names)
    # calculate standard qc .obs and .var
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], inplace=True)
    # rank cells by total counts
    adata.obs["ranked_total_counts"] = np.argsort(adata.obs["total_counts"])

    # arcsinh transform (adata.layers["arcsinh_norm"]) and add total for visualization
    arcsinh_norm(adata, norm="l1", scale=target_sum)
    adata.obs["arcsinh_total_counts"] = np.arcsinh(adata.obs["total_counts"])

    # GF-ICF transform (adata.layers["gf_icf"], adata.obs["gf_icf_total"])
    gf_icf(adata)

    # log1p transform (adata.layers["log1p_norm"])
    sc.pp.normalize_total(
        adata,
        target_sum=target_sum,
        layers=None,
        layer_norm=None,
        key_added="TPM_norm_factor",
    )
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()  # save to .layers

    # HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, n_bins=20, flavor="seurat")

    # set .X as desired for downstream processing; default raw_counts
    adata.X = adata.layers[X_final].copy()


def knn_graph(dist_matrix, k, adata, save_rep="knn"):
    """
    build simple binary k-nearest neighbor graph and add to anndata object

    Parameters:
        dist_matrix (np.array): distance matrix to calculate knn graph for (i.e. pdist(adata.obsm['X_pca']))
        k (int): number of nearest neighbors to determine
        adata (AnnData.AnnData): AnnData object to add resulting graph to (in .uns slot)
        save_rep (str): name of .uns key to save knn graph to within adata (default adata.uns['knn'])

    Returns:
        AnnData.AnnData: adata is edited in place to add knn graph to .uns
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

    Parameters:
        adata (AnnData.AnnData): AnnData object 
        uns_keys (list of str): list of keys in adata.uns to subset. new adata.uns keys will
            be saved with ID appended to name (i.e. adata.uns['knn'] -> adata.uns['knn_ID1'])
        obs_col (str): name of column in adata.obs to use as cell IDs (i.e. 'louvain')
        IDs (list): list of IDs to include in subset

    Returns:
        AnnData.AnnData: adata is edited in place to generate new .uns keys corresponding to
        IDs and chosen .obs column to split on
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

    Parameters:
        adata (AnnData.AnnData): AnnData object
        use_rep (str): 'X' or adata.obsm key containing space to calculate centroids in (i.e. 'X_pca')
        obs_col (str): adata.obs column name containing cluster IDs

    Returns:
        AnnData.AnnData: adata is edited in place to include cluster centroids
        (adata.uns["X_centroids"])
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


def gf_icf_markers(adata, n_genes=10, group_by="louvain"):
    """
    return n_genes with top gf-icf scores for each group

    Parameters:
        adata (AnnData.AnnData): AnnData object preprocessed using gf_icf() or recipe_fcc() function
        n_genes (int): number of top gf-icf scored genes to return per group
        group_by (str): how to group cells to ID marker genes

    Returns:
        pd.DataFrame with top n_genes by gf-icf for each group
    """
    markers = pd.DataFrame()
    for clu in adata.obs[group_by].unique():
        gf_icf_sum = adata.layers["gf_icf"][adata.obs[group_by] == str(clu)].sum(axis=0)
        gf_icf_mean = adata.layers["gf_icf"][adata.obs[group_by] == str(clu)].mean(
            axis=0
        )
        top = np.argpartition(gf_icf_sum, -n_genes)[-n_genes:]
        gene_IDs = adata.var.index[top]
        markers = markers.append(
            pd.DataFrame(
                {
                    group_by: np.repeat(clu, n_genes),
                    "gene": gene_IDs,
                    "gf_icf_sum": gf_icf_sum[top],
                    "gf_icf_mean": gf_icf_mean[top],
                }
            )
        )

    return markers


def cnmf_markers(adata, spectra_score_file, n_genes=30, key="cnmf"):
    """
    read in gene spectra score output from cNMF and save top gene loadings 
    for each usage as dataframe in adata.uns

    Parameters:
        adata (AnnData.AnnData): AnnData object
        spectra_score_file (str): '<name>.gene_spectra_score.<k>.<dt>.txt' file from cNMF containing gene loadings
        n_genes (int): number of top genes to list for each usage (rows of df)
        key (str): prefix of adata.uns keys to save

    Returns:
        AnnData.AnnData: adata is edited in place to include gene spectra scores
        (adata.uns["cnmf_spectra"]) and list of top genes by spectra score (adata.uns["cnmf_markers"])
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

    Parameters:
        adata : AnnData
            The data.
        attr : {'var', 'obs', 'uns', 'varm', 'obsm'}
            The attribute of AnnData that contains the score.
        keys : str or list of str
            The scores to look up an array from the attribute of adata.
        indices : list of int
            The column indices of keys for which to plot (e.g. [0,1,2] for first three keys)
    Returns:
        matplotlib gridspec with access to the axes.
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
    Load results of cNMF.
    Given adata object and corresponding cNMF output (cnmf_dir, name, k, dt to identify),
    read in relevant results and save to adata object inplace, and output plot of gene
    loadings for each GEP usage.

    Parameters:
        adata (AnnData.AnnData): AnnData object
        cnmf_dir (str): relative path to directory containing cNMF outputs
        name (str): name of cNMF replicate
        k (int): value used for consensus factorization
        dt (int): distance threshold value used for consensus clustering
        key (str): prefix of adata.uns keys to save
        n_points (int): how many top genes to include in rank_genes() plot
        **kwargs: keyword args to pass to cnmf_markers()

    Returns:
        AnnData.AnnData: adata is edited in place to include overdispersed genes
            (adata.var["cnmf_overdispersed"]), usages (adata.obs["usage_#"]), gene
            spectra scores (adata.uns["cnmf_spectra"]), and list of top genes by
            spectra score (adata.uns["cnmf_markers"]).
        gridspec.Gridspec: plot of n_points gene loadings for each cNMF usage
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
    class defining pretty plots of dimension-reduced embeddings such as PCA/t-SNE/UMAP.

    DR_plot().plot(): utility plotting function that can be passed any numpy array in the `data` parameter
             .plot_IDs(): plot one or more cluster IDs on top of an .obsm from an `AnnData` object
             .plot_centroids(): plot cluster centroids defined using find_centroids() function on `AnnData` object
    """

    def __init__(self, dim_name="dim", figsize=(5, 5), ax_labels=True):
        """
        constructor for DR_plot class.

        Parameters:
            dim_name (str): how to label axes ('dim 1' on x and 'dim 2' on y by default)
            figsize (tuple of int): size of resulting axes
            ax_labels (bool): draw arrows and dimension names in lower left corner of plot
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
        general plotting function for dimensionality reduction outputs.

        Parameters:
            data (np.array): variables in columns and observations in rows
            color (list-like): list of length nrow(data) to determine how points should be colored
            pt_size (int): size of points in plot
            legend (None or str): 'full', or 'brief' as in sns. default is None.
            save_to (str): path to .png file to save output, or None
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
        plotting function for adata objects with points colored by categorical IDs

        Parameters:
            adata (AnnData.AnnData): object to pull dimensionality reduction from
            use_rep (str): adata.obsm key to plot from (i.e. 'X_pca')
            obs_col (str): name of column in adata.obs to use as cell IDs (i.e. 'louvain')
            IDs (list of str): list of IDs to plot, graying out cells not assigned to those IDS (default 'all' IDs)
            pt_size (int): size of points in plot
            save_to (str): path to .png file to save output, or None
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
        plotting function for adata objects with cluster centroids from calculate_centroids()

        Parameters:
            adata (AnnData.AnnData): object to pull dimensionality reduction from
            use_rep (str): adata.obsm key to plot from (i.e. 'X_pca')
            obs_col (str): name of column in adata.obs to use as cell IDs (i.e. 'louvain')
            ctr_size (int): size of centroid points in plot
            pt_size (int): size of points in plot
            draw_edges (bool): draw edges of minimum spanning tree between all centroids?
            highlight_edges (list): list of edge IDs as tuples to highlight in red on plot
                e.g. `set(adata.uns['X_tsne_centroid_MST'].edges).difference(set(adata.uns['X_umap_centroid_MST'].edges))`
                with output {(0,3), (0,7)} says that edges from centroid 0 to 3 and 0 to 7 are found in 'X_tsne_centroids'
                but not in 'X_umap_centroids'. highlight the edges to show this.
            save_to (str): path to .png file to save output, or None
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
