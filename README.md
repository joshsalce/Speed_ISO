# A First Look: Speed Effects on ISO (Code and Visuals)

## Description
This project is an attempt to find the effect that a hitter's sprint speed has on his statistic [Isolated Power (ISO)](https://www.mlb.com/glossary/advanced-stats/isolated-power). This repository includes notebooks containing cleaning and exploratory analysis of imported Statcast from the pybasell package, application of unsupervised machine learning techniques to data, and the  use of several different supervised machine learning classification methods to perform multi-class classification on a highly imbalanced dataset of singles, doubles, and triples.
Some of these methods include probability calibrated Naive Bayes and Support Vector Classifiers, a probabilistic Tensorflow neural network, and Weighted or Balanced Random Forest Classifiers. All methods for cleaning data, plotting visualizations for model performance, or implementing model towards individual predictions are included in the 'Scripts' folder.
All exploratory analyses of data are located in the 'Analysis Notebooks' section, and the use of supervised machine learning classifiers to preprocess data and train/test models are in all folders containing models.

### Motivation
Much of this project's inspiration has been inspired by Corbin Carroll, outfielder for the Arizona Diamondbacks. His ability to stretch hits that for nearly everyone else are singles into doubles, and from doubles to triples, caused me to wonder how his sprint speed could affect his ISO based on how often he could take an extra base at times where other hitters may not be able to.

### Packages and Tech Used
- Python
- Python Packages: [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), [datetime](https://docs.python.org/3/library/datetime.html), [sklearn](https://scikit-learn.org/stable/index.html), [imblearn](https://scikit-learn.org/stable/index.html), [TensorFlow](https://tensorflow.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [pybaseball](https://pypi.org/project/pybaseball/) 

## Table of Contents

| Component | Description |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [Analysis Notebooks](https://github.com/joshsalce/Speed_ISO/tree/main/Analysis%20Notebooks)| Jupyter Notebooks containing exploratory and unsupervised ML analysis of cleaned data | 
| [First Pass Models](https://github.com/joshsalce/Speed_ISO/tree/main/First%20Pass%20Models) | Jupyter Notebooks containing first set of instantiated models using first set of features |
| [Scripts](https://github.com/joshsalce/Speed_ISO/tree/main/Scripts) | Python scripts containing functions and code, imported into and ran inside of Jupyter Notebooks to reduce clutter |
| [Second Pass Models](https://github.com/joshsalce/Speed_ISO/tree/main/Second%20Pass%20Model) | Jupyter Notebooks containing second set of instantiated models after unsupervised ML performed on cleaned data, first set of features |
| [Third Pass Model](https://github.com/joshsalce/Speed_ISO/tree/main/Third%20Pass%20Model) | Contains Jupter Notebooks of new and final set of models trained on preprocessed data, as well as CSV of Statcast sprint speeds from 2015-2023 |
| [Visualizations](https://github.com/joshsalce/Speed_ISO/tree/main/Visualizations) | Includes relevant visuals from background research on probability calibration, generated figures from EDA, unsupervised ML, and visuals for model performance and predictions |


## Directions

## Credits

