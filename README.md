# A First Look: Speed Effects on ISO (Code and Visuals)

## Description
This project is an attempt to find the effect that a hitter's sprint speed has on his statistic [Isolated Power (ISO)](https://www.mlb.com/glossary/advanced-stats/isolated-power). This repository includes notebooks containing cleaning and exploratory analysis of imported Statcast from the pybaseball package, application of unsupervised machine learning techniques, light feature engineering, and the use of several different supervised machine learning classification methods to perform multi-class classification on a highly imbalanced dataset of singles, doubles, and triples. Some of these methods include probability calibrated Naive Bayes and Support Vector Classifiers, a probabilistic Tensorflow neural network, and Weighted or Balanced Random Forest Classifiers. All models were evaluated on several classification metrics, including precision, recall, ROC-AUC, and the recall of each individual class. The methods for cleaning data, plotting visualizations for model performance, or implementing model towards individual predictions are included in the 'Scripts' folder.
All exploratory analyses of data are located in the 'Analysis Notebooks' section, and the use of supervised machine learning classifiers to preprocess data and train/test models are in all folders containing models. A greater in-depth explanation of the code can be found at my [Medium page](https://medium.com/@joshsalce).

### Motivation
Much of this project's inspiration has been inspired by Corbin Carroll, outfielder for the Arizona Diamondbacks. His ability to stretch hits that for nearly everyone else are singles into doubles, and from doubles to triples, caused me to wonder how his sprint speed could affect his ISO based on how often he could take an extra base at times where other hitters may not be able to.

### Packages and Tech Used
- Python
- Python Packages: [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), [datetime](https://docs.python.org/3/library/datetime.html), [sklearn](https://scikit-learn.org/stable/index.html), [imblearn](https://scikit-learn.org/stable/index.html), [TensorFlow](https://tensorflow.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [pybaseball](https://pypi.org/project/pybaseball/) 

## Table of Contents

| Component | Description |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [Analysis Notebooks](https://github.com/joshsalce/Speed_ISO/tree/main/Analysis%20Notebooks)| Jupyter Notebooks containing exploratory and unsupervised ML analysis of cleaned data. Unsup. ML methods included are Principal Component Analysis (PCA) and Gaussian Mixture Models (GMM). | 
| [First Pass Models](https://github.com/joshsalce/Speed_ISO/tree/main/First%20Pass%20Models) | Jupyter Notebooks containing first set of instantiated models using first set of features. Models Used: (Probability Calibrated) Naive Bayes, SVC, and AdaBoost, as well as a Keras Sequential Network and Bagged Decision Tress |
| [Scripts](https://github.com/joshsalce/Speed_ISO/tree/main/Scripts) | Python scripts containing functions and code, imported into and ran inside of Jupyter Notebooks to reduce clutter |
| [Second Pass Models](https://github.com/joshsalce/Speed_ISO/tree/main/Second%20Pass%20Model) | Jupyter Notebooks containing second set of instantiated models after unsupervised ML performed on cleaned data, first set of features. All models tested are the same, but all models are trained on data with PCA and undersampling of non-minority classes applied. |
| [Third Pass Model](https://github.com/joshsalce/Speed_ISO/tree/main/Third%20Pass%20Model) | Contains Jupter Notebooks of new and final set of models trained on preprocessed data plus predictions for a set of individual players, as well as CSV of Statcast sprint speeds from 2015-2023. Models Used: (Class) Balanced Random Forest, Weighted Random Forest (Class Weight = Balanced) |
| [Visualizations](https://github.com/joshsalce/Speed_ISO/tree/main/Visualizations) | Includes relevant visuals from background research on probability calibration, generated figures from EDA, unsupervised ML, and visuals for model performance and predictions |

## Visualizations- Descriptions
- First Pass Performance: Visualizations of First Set of Models
  - [conf_matx.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/First%20Pass%20Performance/conf_matx.png)- Confusion Matrices for each model
  - [metrics.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/First%20Pass%20Performance/metrics.png)- Metrics DataFrame
  - [proba_dist.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/First%20Pass%20Performance/proba_dist.png)- Probability distribution of all classes for each model
  - [roc_auc.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/First%20Pass%20Performance/roc_auc.png)- ROC-AUC and Precision-Recall curves for each model
 
- Second Pass Performance: Visualizations of Second Set of Models
  - [conf_matx.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Second%20Pass%20Performance/conf_matx.png)- Confusion Matrices for each model
  - [metrics.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Second%20Pass%20Performance/metrics.png)- Metrics DataFrame
  - [proba_dist.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Second%20Pass%20Performance/proba_dist.png)- Probability distribution of all classes for each model
  - [roc_auc.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Second%20Pass%20Performance/roc_auc.png)- ROC-AUC and Precision-Recall curves for each model

- Third Pass Performance: Visualizations of Third Set of Models
  - Final Model
    - [error.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/Final%20Model/error.png)- Scatterplot of 2022 players' True ISO vs. predicted ISO
    - [residuals.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/Final%20Model/residuals.png)- Residual Plot of Predicted ISO values vs. True ISO minus Predicted ISO
    - [residuals_qualified.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/Final%20Model/residuals_qualified.png)- For all qualified 2022 players, plot of predicted ISO with sprint speed +/ 1.5 ft/sec minus predicted ISO at true sprint speed (Maximum and Miniumums  at +0.020 and ~-0.035 points of predicted ISO)
  - [Individual Residuals 2022](https://github.com/joshsalce/Speed_ISO/tree/main/Visualizations/Third%20Pass%20Performance/Individual%202022%20Residuals): Residuals of predicted ISO with sprint speed +/ 1.5 ft/sec minus predicted ISO with no change to sprint speed
    - Players: Ronald Acuña Jr., Mookie Betts, Alex Bregman, Corbin Carroll, Starling Marte, Kyle Schwarber, Giancarlo Stanton, Bobby Witt Jr. 
  - [conf_matx.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/conf_matx.png)- Confusion Matrices for each model
  - [metrics.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/metrics.png)- Metrics DataFrame
  - [proba_dist.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/proba_dist.png)- Probability distribution of all classes for each model
  - [roc_auc.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/roc_auc.png)- ROC-AUC and Precision-Recall curves for each model
  - [speed_v_change.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Third%20Pass%20Performance/speed_v_change.png)- Scatterplot of 2022 qualified players, 2022 sprint speed vs. 2022 sprint speed minus maximum sprint speed of all individual sprint speeds 2015-2023 
  
- [Hit_Type_Bar.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/Hit_Type_Bar.png)- Bar plot of hit type distributions for singles, doubles, triples
- [PCA_Imbalanced.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/PCA_Imbalanced.png)- 2D and 3D plots of cleaned data with PCA applied (2 and 3 component)
- [PCA_Undersampled.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/PCA_Undersampled.png)- 2D and 3D plots of cleaned data with PCA and undersampling of non-minority clases applied (2 and 3 component)
- [calc_distance_error.png](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/calc_distance_error.png)- Scatterplot of Statcast hit_distance_sc variable vs. personally calculated distance metric (see Exploratory notebook in Analysis Notebooks)
- fig8_Niculescu-Mizil_Caruana.png: Figure 8 from [Niculescu-Mizil and Caruana](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- [reflection.gif](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/reflection.gif)- GIF, reflecting Corbin Carroll's 2022 hits from Statcast hit-coordinates right-side up
- [translation.gif](https://github.com/joshsalce/Speed_ISO/blob/main/Visualizations/translation.gif)- GIF, translating Corbin Carroll's 2022 hits from Statcast hit-coordinates (after reflection) to the origin 

