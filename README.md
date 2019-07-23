### furry-couscous
Tools for manipulation and analysis of single-cell and spatial transcriptomic data

---
#### Required Python Dependencies
Install using pip:  
```
pip install -r requirements.txt
```

---
#### Contents
##### `fcc_utils.py` & `fcc_utils.r`:
Contain utility functions for manipulating datasets and comparing feature-reduced latent spaces.  

##### `fcc.py`:
Defines python classes for manipulation, processing, and visualization of scRNA-seq gene expression counts, dimensionality reduction objects, and spatial transcriptomics data.  

Consult `fcc_classes_tutorial.ipynb` for info on how to create and manipulate _couscous_ and _pita_ objects.  

Example datasets used can be found in `inputs/`.  

---
#### Optional Packages

For feature selection using neighborhood variance ratio, install [__NVR__ from GitHub](https://github.com/KenLauLab/NVR).  
