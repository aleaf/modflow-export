
MFexport
-----------------------------------------------
Fast & easy summarizing of MODFLOW data and export to GIS file formats.

### Version 0.0.0
[![Build Status](https://travis-ci.com/aleaf/MFexport.svg?branch=master)](https://travis-ci.org/aleaf/MFexport.svg)
[![Coverage Status](https://codecov.io/github/aleaf/MFexport/coverage.svg?branch=master)](https://codecov.io/github/aleaf/MFexport/coverage.svg?branch=master)





Getting Started
-----------------------------------------------

[Example Notebook](Examples/example.ipynb)


### Bugs

If you think you have discovered a bug in MFexport in which you feel that the program does not work as intended, then we ask you to submit a [Github issue](https://github.com/aleaf/mfsetup/labels/bug).


Installation
-----------------------------------------------

**Python versions:**

MFsetup requires **Python** 3.6 (or higher)

**Dependencies:**  
pyaml  
numpy  
pandas  
fiona  
rasterio  
shapely  
pyproj  
flopy  

### Install to site_packages folder
```
python setup.py install
```
### Install in current location (to current python path)
(i.e., for development)  

```  
pip install -e .
```



MODFLOW Resources
-----------------------------------------------

+ [MODFLOW Online Guide](https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/index.html?nwt_newton_solver.htm)
+ [MODFLOW 6](https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model)



Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.

