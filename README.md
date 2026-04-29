# EigenFit
Tools for viewing and analysing Raman map data in Python using Jupyter notebooks. 
Interactive display for grid (square/ rectangular) and line-profile Raman maps, suitable for single maps and for time-series of maps acquired at the same location (e.g. during _operando_ characterisation). 
Rapid analysis of low signal-to-noise ratio (SNR) data, and data where multiple peaks are present in different combinations throughout the map, by identifying spectra containing significant contributions from peaks, and 'skipping' noisy spectra. 

The name 'EigenFit' refers to the use of the eigenvectors of the covariance matrix calculated in the PCA process to identify spectra with significant peaks. 

**Note**: interactive display elements are currently optimised for **Jupyter 6**. Work is in progress to adapt to Jupyter 7. 

## Installation
```bash
pip install EigenFit
```

Or

```bash
pip install "EigenFit @ git+https://github.com/LiamBird/EigenFit.git"
```

## Requirements
* Python 3.9+
* numpy
* matplotlib
* lmfit
* sklearn
* ipywidgets
* BaselineRemoval

Examples to follow
