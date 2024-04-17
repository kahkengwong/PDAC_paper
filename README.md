# PDAC Paper Repository
Author: Kah Keng Wong  
Email: [kahkeng@usm.my](mailto:kahkeng@usm.my); [kahkeng3@gmail.com](mailto:kahkeng3@gmail.com)  
GitHub: [kahkengwong/PDAC_paper](https://github.com/kahkengwong/PDAC_paper)

## Dataset Folder
This folder contains:
1. **Expression Matrix**: PDAC and asymptomatic control cases expression matrix for DEGs analysis using edgeR scripts in the 'PDAC_DEGs_edgeR.r' file.
2. **ML Datasets**: The training, test, and calibrate sets for ML.

## ML Scripts
Scripts should be used in the following sequence, corresponding to the flow of the main project/manuscript:
1. `TPE_with_xxx.py`: Scripts to perform Tree-structured Parzen Estimator (TPE) with each of the 5 ML algorithms (Logistic Regression - LR, Support Vector Machine - SVM, Random Forest - RF, XGBoost - XGB, Light Gradient Boosting Machine - GBM) in different settings.
2. `Pre-ensemble_xxx.py`: Execute the best genes combination and the best hyperparameters for each ML algorithm, returning their probabilities within the Python environment.
3. `Ensemble_Combining_Any_Two_or_Three_Models.py`: Perform ensemble modeling combining any two or three individual models.
4. `Ensemble_CLR_of_SVM-RF-GBM_Model.py`: Perform Custom Logistic Regression (CLR) on the best ensemble model i.e., the SVM:RF:GBM model.

## Data Source
This project utilizes data from the study "Detection and localization of early- and late-stage cancers using platelet RNA" by S. In 't Veld et al., 2022, available under GEO accession number GSE183635. Users of this data should cite the original article (DOI: 10.1016/j.ccell.2022.08.006).

## Contact
For further information or questions, please email [kahkeng@usm.my](mailto:kahkeng@usm.my) or [kahkeng3@gmail.com](mailto:kahkeng3@gmail.com).
