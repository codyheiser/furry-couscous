### furry-couscous
Review of dimensionality reduction strategies  

---
```
pip install -r requirements.txt # make sure you have all necessary python packages
```
  
In order to use the [__"FIt-SNE"__ implementation](https://arxiv.org/abs/1712.09005) of t-SNE, you'll need to download [FFTW](http://www.fftw.org/) and compile the code from the [FIt-SNE repo](https://github.com/KlugerLab/FIt-SNE).  

For feature selection using neighborhood variance ratio, install [__NVR__ from GitHub](https://github.com/KenLauLab/NVR).  

---
#### `fcc_utils.py`
Contains utility functions for reading files and comparing feature-reduced data.  

---
#### `fcc_DRs.py`
Defines classes for manipulation, processing, and visualization of scRNA-seq counts data and dimensionality reduction objects.  
