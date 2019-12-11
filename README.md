### furry-couscous
Tools for manipulation and analysis of single-cell and spatial transcriptomic data

---
#### Contents
##### `scanpy_utils.py`:
Contains functions for processing single-cell data in Python using [Scanpy](https://scanpy.readthedocs.io/en/stable/).  

##### `seurat_utils.r` & `ggplot_config.r`:
Contains functions for processing single-cell data in R using [Seurat](https://www.rdocumentation.org/packages/Seurat/versions/3.1.1).  

##### `fcc_utils.py`:
Contains utility functions for manipulating datasets and comparing feature-reduced latent spaces.  
Consult `fcc_tutorial.ipynb` for info on how to process datasets using the __`furry-couscous`__ framework. 

##### `fcc.py`:
Defines python classes for manipulation, processing, and visualization of scRNA-seq gene expression counts, dimensionality reduction objects, and spatial transcriptomics data.  

---
#### Required Python Dependencies
Several graph-based analysis packages require the [iGraph C library](https://igraph.org/). Ensure this is installed before proceeding.  

Install Python packages using pip:  
```
pip install -r requirements.txt
```
