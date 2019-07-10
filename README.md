### furry-couscous
Tools for manipulation and analysis of single-cell and spatial transcriptomic data

---
#### Required Python Dependencies
Install using pip:  
```
pip install -r requirements.txt
```

---
#### Optional Packages
In order to use the [__"FIt-SNE"__ implementation](https://arxiv.org/abs/1712.09005) of t-SNE, you'll need to download [FFTW](http://www.fftw.org/) and compile the code from the [FIt-SNE repo](https://github.com/KlugerLab/FIt-SNE).  

For feature selection using neighborhood variance ratio, install [__NVR__ from GitHub](https://github.com/KenLauLab/NVR).  

Clone the [scvis](https://github.com/shahcompbio/scvis) and [ZIFA](https://github.com/epierson9/ZIFA) packages and install with `python setup.py install` from their home directories.  

---
#### Contents
##### `fcc_utils.py` & `fcc_utils.r`:
Contain utility functions for manipulating datasets and comparing feature-reduced latent spaces.  

##### `fcc_DRs.py`:
Defines python classes for manipulation, processing, and visualization of scRNA-seq counts data and dimensionality reduction objects.  

Consult `fcc_classes_tutorial.ipynb` for info on how to create and manipulate _RNA_counts_ and _DR_ objects.  

Example datasets used can be found in `inputs/`, notebooks chronicling work along the way are located in `dev/`.  
