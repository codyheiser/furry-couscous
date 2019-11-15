# -*- coding: utf-8 -*-
"""
scRNA-seq quality control and cell filtering

@author: B Chen
June 2019
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from pydpc import Cluster
import scipy.io
import scipy
from sklearn.preprocessing import normalize
import sys


class library_data(object):
    def __init__(
        self, lib_counts=None, lib_cellID=None, lib_geneID=None, sort=True, subset=[]
    ):
        self.lib_counts = scipy.sparse.csr_matrix(
            lib_counts
        )  # must be count matrix, integers in some array format
        self.lib_size = np.array(self.lib_counts.sum(axis=1)).flatten()
        self.lib_rank = self.lib_size.argsort(axis=0)[::-1]  # greatest to least
        self.lib_geneID = lib_geneID  # must be 1 dimensional numpy string array
        if (
            sort
        ):  # rearrange cellIDs and count data by sorted ranking, least to greatest
            self.lib_counts = self.lib_counts[self.lib_rank]  # sorted
            self.lib_size = self.lib_size[self.lib_rank]
            self.lib_cellID = lib_cellID[self.lib_rank]
        else:
            self.lib_cellID = lib_cellID
        self.cell_number = self.lib_counts.shape[0]
        self.gene_number = self.lib_counts.shape[1]

    def find_inflection(self, inflection_percentiles=[0, 15, 30, 100]):
        lib_cumsum = np.cumsum(self.lib_size)
        x_vals = np.arange(0, self.cell_number)
        secant_coef = lib_cumsum[self.cell_number - 1] / self.cell_number
        secant_line = secant_coef * x_vals
        secant_dist = abs(lib_cumsum - secant_line)
        inflection_percentiles_inds = np.percentile(
            x_vals, inflection_percentiles
        ).astype(int)
        inflection_points = secant_dist.argsort()[::-1]
        percentile_points = inflection_points[inflection_percentiles_inds]
        color = plt.cm.tab10(np.linspace(0, 1, self.cell_number))
        plt.figure(figsize=(20, 10))
        plt.plot(np.array(lib_cumsum), label="Cumulative Sum")
        # plt.plot(np.array(secant_line), label="Secant Line")
        plt.plot(np.array(secant_dist), label="Secant Distance")
        for percentile in percentile_points:
            plt.axvline(
                x=percentile,
                ymin=0,
                c=color[percentile],
                linestyle="--",
                linewidth=2,
                label="Inflection point {}".format(percentile),
            )
        plt.legend()
        print(
            "Inflection point at {} for {} percentiles of greatest secant distances".format(
                percentile_points, inflection_percentiles
            )
        )


class dimension_reduction(object):
    def __init__(self, lib_data_in, inflection=0, seed=42, subset=False):
        self.lib_values = lib_data_in.lib_counts[:inflection, :]
        self.lib_size = lib_data_in.lib_size[:inflection]
        self.lib_rank = self.lib_size.argsort(axis=0)
        self.lib_geneID = lib_data_in.lib_geneID
        self.lib_cellID = lib_data_in.lib_cellID
        self.cell_number = lib_data_in.cell_number
        self.seed = seed
        np.random.seed(self.seed)
        self.UMAP = 0
        self.TSNE = 0

    def lib_size_normalize(self):
        self.lib_values = normalize(self.lib_values, norm="l1", axis=1)

    def arcsinh_transform(self, cofactor=1000):
        self.lib_values = np.arcsinh(self.lib_values * cofactor)

    def log1p_transform(self):
        self.lib_values = np.log1p(self.lib_values)

    def runPCA(self, n_pcs=100):
        print("Running PCA for {} components".format(n_pcs))
        self.lib_values_array = self.lib_values.toarray()
        _pca = PCA(n_components=n_pcs)
        self.PCA = _pca.fit(self.lib_values_array).transform(self.lib_values_array)

    def runUMAP(self, neighborhood_percent=0.005, n_components=2):
        n_neighbors_ = int(self.cell_number * neighborhood_percent)
        print("Running UMAP with {} neighbors".format(n_neighbors_))
        self.UMAP = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors_,
            min_dist=0.1,
            metric="correlation",
        ).fit_transform(self.PCA)

    def runTSNE(self, neighborhood_percent=0.005, n_components=2):
        n_neighbors_ = int(self.cell_number * neighborhood_percent)
        print("Running tSNE with {} perplexity".format(n_neighbors_))
        self.TSNE = TSNE(
            n_components=n_components,
            perplexity=n_neighbors_,
            n_iter=5000,
            metric="euclidean",
            init="pca",
            random_state=self.seed,
        ).fit_transform(self.PCA)


class gate_visualize(object):
    def __init__(self, dr_in, subset=False):
        self.lib_values = dr_in.lib_values
        self.lib_size = dr_in.lib_size
        self.lib_rank = dr_in.lib_rank
        self.PCA = dr_in.PCA
        self.UMAP = dr_in.UMAP
        self.TSNE = dr_in.TSNE
        self.seed = dr_in.seed
        self.lib_geneID = dr_in.lib_geneID
        self.DPC = None

    def runDPC(self, dr, x_cutoff, y_cutoff, force_rerun=False):
        if (self.DPC == None) or (force_rerun == True):
            self.DPC = Cluster(dr.astype("float64"))
        self.DPC.assign(x_cutoff, y_cutoff)

    def plotDPC(self):
        fig = plt.figure(figsize=(30, 10))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.scatter(
            self.UMAP[:, 0],
            self.UMAP[:, 1],
            c=self.DPC.membership,
            cmap="gist_rainbow",
            s=10,
        )
        ax2.scatter(
            self.UMAP[:, 0], self.UMAP[:, 1], c=self.lib_size, cmap="seismic", s=10
        )
        ax3.scatter(
            self.UMAP[:, 0], self.UMAP[:, 1], c=self.lib_rank, cmap="seismic", s=10
        )
        # plt.colorbar(p,ax=ax3)
        color = plt.cm.gist_rainbow(np.linspace(0, 1, len(self.DPC.clusters)))

        for i in range(len(self.DPC.clusters)):
            x = self.UMAP[self.DPC.clusters[i], 0]
            y = self.UMAP[self.DPC.clusters[i], 1]
            text = ax1.text(
                x,
                y,
                i,
                fontsize=15,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
                weight="bold",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="Circle",
                    pad=0.1,
                    alpha=0.5,
                ),
            )

        ax1.set_title("Density Peak Clusters", size=15, weight="bold")
        ax2.set_title("Library Size (Blue = Low Quality)", size=15, weight="bold")
        ax3.set_title(
            "Ranked Library Sizes (Blue = Low Quality)", size=15, weight="bold"
        )

    def plotGenes(self, feature_list, embedding="UMAP"):
        self.lib_geneID = pd.Series(self.lib_geneID)
        feature_inds = []
        gene_overlays = []
        for features in feature_list:
            feature_inds.append(np.where(self.lib_geneID.str.contains(features))[0])
        for features in feature_inds:
            gene_overlays.append(
                np.array(np.sum(self.lib_values[:, features], axis=1)).flatten()
            )
        if embedding == "UMAP":
            fig = plt.figure(figsize=(30, 30))
            for i in range(len(gene_overlays)):
                plt.subplot(3, 3, i + 1)
                plt.scatter(
                    self.UMAP[:, 0],
                    self.UMAP[:, 1],
                    c=gene_overlays[i],
                    cmap="hot",
                    s=20,
                )
                plt.title(feature_list[i], size=15, weight="bold")
        if embedding == "TSNE":
            fig = plt.figure(figsize=(30, 30))
            for i in range(len(gene_overlays)):
                plt.subplot(3, 3, i + 1)
                plt.scatter(
                    self.TSNE[:, 0],
                    self.TSNE[:, 1],
                    c=gene_overlays[i],
                    cmap="hot",
                    s=20,
                )
                plt.title(feature_list[i], size=15, weight="bold")

    def manual_gating(self, gate_out):  # embedding = "UMAP"
        color = plt.cm.gist_rainbow(np.linspace(0, 1, len(self.DPC.clusters)))
        clust_inds = np.delete(
            np.arange(0, len(self.DPC.membership), 1), gate_out
        )  # clusters that represent cells to keep
        cluster_ids = np.delete(np.arange(0, len(self.DPC.clusters), 1), gate_out)
        clust_mask = np.isin(self.DPC.membership, clust_inds)
        gate_out_inds = np.where(clust_mask == False)
        gated_embedding = self.UMAP[clust_mask]

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111)
        ax1.scatter(
            gated_embedding[:, 0],
            gated_embedding[:, 1],
            alpha=1,
            s=20,
            c=self.DPC.membership[clust_mask],
            cmap="gist_rainbow",
        )
        ax1.scatter(
            self.UMAP[gate_out_inds, 0],
            self.UMAP[gate_out_inds, 1],
            alpha=0.5,
            s=20,
            c="gray",
        )

        for i in range(len(self.DPC.clusters[cluster_ids])):
            x = self.UMAP[self.DPC.clusters[cluster_ids][i], 0]
            y = self.UMAP[self.DPC.clusters[cluster_ids][i], 1]
            text = ax1.text(
                x,
                y,
                i,
                fontsize=15,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
                weight="bold",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="Circle",
                    pad=0.1,
                    alpha=0.5,
                ),
            )
        return np.where(clust_mask)[0]
