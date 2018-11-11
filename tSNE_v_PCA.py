# t-SNE vs PCA testing

# @author: C Heiser
# 05 November 2018

import numpy as np
import pandas as pd
import scipy as sc
# scikit packages
import skbio
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# import tsne package
import sys; sys.path.append('/Users/Cody/git/FIt-SNE')
from fast_tsne import fast_tsne
# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'whitegrid')
# import utility functions
import tSNE_utils


# load the hdf5 files into dictionary
reps = {
'r00':tSNE_utils.read_hdf5('inputs/GSE102698ClosenessRep_0.hdf5'),
'r01':tSNE_utils.read_hdf5('inputs/GSE102698ClosenessRep_1.hdf5'),
'r02':tSNE_utils.read_hdf5('inputs/GSE102698ClosenessRep_2.hdf5')
}

# initiate df for dumping correlation data into
corr_out = pd.DataFrame()

for rep in reps.keys():
    for key in reps[rep].keys():
        
        print('generating raw distance matrix for rep {} with closeness {}'.format(rep,key))
        dm = sc.spatial.distance_matrix(x=reps[rep][key],y=reps[rep][key]) # generate distance matrix for raw data
        
        print('performing PCA for rep {} with closeness {}'.format(rep,key))
        pca = PCA(n_components=50).fit_transform(tSNE_utils.arcsinh_norm(reps[rep][key])) # perform PCA with 50 components
        
        print('clustering PCA results using k-means')
        pca_clust = KMeans(n_clusters=2).fit_predict(pca)
        
        print('generating PCA distance matrix for rep {} with closeness {}'.format(rep,key))
        dm_pca = sc.spatial.distance_matrix(x=pca,y=pca) # generate distance matrix for PCA
        
        print('performing t-SNE for rep {} with closeness {}'.format(rep,key))
        tsne = fast_tsne(pca,seed=18,perplexity=30) # perform tSNE with perplexity 30 using PCA output
        
        print('clustering t-SNE results using k-means')
        tsne_clust = KMeans(n_clusters=2).fit_predict(tsne)
        
        print('generating t-SNE distance matrix for rep {} with closeness {}'.format(rep,key))
        dm_tsne = sc.spatial.distance_matrix(x=tsne,y=tsne) # generate distance matrix for tSNE
        
        print('generating distance correlations for rep {} with closeness {}'.format(rep,key))
        corr = skbio.stats.distance.pwmantel(dms=[dm, dm_pca, dm_tsne]) # perform Mantel test on all combinations of distance matrices
        corr['Replicate']=rep
        corr['Closeness']=key
        
        corr_out = corr_out.append(corr)
        
        print('plotting results')
        # plot results
        plt.figure(figsize=(15,5))
        
        # plot PCA
        plt.subplot(131)
        sns.scatterplot(pca[:,0], pca[:,1], s=75, hue=pca_clust)
        plt.title('PCA {} {}'.format(rep,key))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # plot tSNE with PCA clusters
        plt.subplot(132)
        sns.scatterplot(tsne[:,0], tsne[:,1], s=75, hue=pca_clust)
        plt.title('t-SNE with PCA clusters {} {}'.format(rep,key))
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # plot tSNE with tSNE clusters
        plt.subplot(133)
        sns.scatterplot(tsne[:,0], tsne[:,1], s=75, hue=tsne_clust)
        plt.title('t-SNE {} {}'.format(rep,key))
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.savefig('outputs/PCA_v_tSNE_{}_{}.pdf'.format(rep,key))
        plt.close()
        
corr_out.to_csv('outputs/tSNE_PCA_correlations.csv')
