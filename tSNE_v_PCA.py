# t-SNE vs PCA testing

# @author: C Heiser
# 05 November 2018

import numpy as np
import pandas as pd
import scipy as sc
# scikit packages
import skbio
from sklearn.decomposition import PCA
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
        pca = PCA(n_components=50).fit_transform(np.log2(reps[rep][key]+1)) # perform PCA with 50 components
        
        # plot PCA
        plt.figure(figsize=(5,5))
        sns.scatterplot(pca[:,0], pca[:,1], s=75)
        plt.title('PCA {} {}'.format(rep,key))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.savefig('outputs/PCA_{}_{}.pdf'.format(rep,key))
        
        print('generating PCA distance matrix for rep {} with closeness {}'.format(rep,key))
        dm_pca = sc.spatial.distance_matrix(x=pca,y=pca) # generate distance matrix for PCA
        
        print('performing t-SNE for rep {} with closeness {}'.format(rep,key))
        tsne = fast_tsne(pca,seed=18,perplexity=30) # perform tSNE with perplexity 30 using PCA output
        
        # plot tSNE
        plt.figure(figsize=(5,5))
        sns.scatterplot(tsne[:,0], tsne[:,1], s=75)
        plt.title('t-SNE {} {}'.format(rep,key))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.savefig('outputs/tSNE_{}_{}.pdf'.format(rep,key))
        
        print('generating t-SNE distance matrix for rep {} with closeness {}'.format(rep,key))
        dm_tsne = sc.spatial.distance_matrix(x=tsne,y=tsne) # generate distance matrix for tSNE
        
        print('generating distance correlations for rep {} with closeness {}'.format(rep,key))
        corr = skbio.stats.distance.pwmantel(dms=[dm, dm_pca, dm_tsne]) # perform Mantel test on all combinations of distance matrices
        corr['Replicate']=rep
        corr['Closeness']=key
        
        corr_out = corr_out.append(corr)
        
corr_out.to_csv('outputs/tSNE_PCA_correlations.csv')
