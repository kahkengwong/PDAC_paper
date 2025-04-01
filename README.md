# PDAC ML Paper Repository
Ensemble machine learning with tree-structured Parzen estimator (TPE); 3D plots of hyperparameters optimization (HPO) and objective function (OF) tracking of custom scoring function (CSF) scores for interpretability.

![GitHub last commit](https://img.shields.io/github/last-commit/kahkengwong/PDAC_paper)
![GitHub repo size](https://img.shields.io/github/repo-size/kahkengwong/PDAC_paper)
![GitHub languages](https://img.shields.io/github/languages/count/kahkengwong/PDAC_paper)
![GitHub top language](https://img.shields.io/github/languages/top/kahkengwong/PDAC_paper)
![Contributors](https://img.shields.io/github/contributors/kahkengwong/PDAC_paper)

Author: Kah Keng Wong  
Preprint of the study is available on SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4809282

Manuscript was accepted by _Biomedical Signal Processing and Control_ (Q1) on 16 March 2025 and paper is in production.

---

## Dataset Folder
This folder contains:
1. **Expression Matrix** - PDAC and asymptomatic control cases expression matrix for DEGs analysis using edgeR scripts in the `PDAC_DEGs_edgeR.r` file.
2. **ML Datasets** - The training, test, and calibration sets for ML.

---

## ML Scripts
Scripts should be used in the following sequence, corresponding to the flow of the main project/manuscript:
1. `TPE_with_xxx.py` - Scripts to perform TPE with each of the 5 ML algorithms i.e., LR, SVM, RF, XGB, and GBM in different settings.
2. `Pre-ensemble_xxx.py` - Execute the best genes combination and the best hyperparameters for each ML algorithm, returning their probabilities within the Python environment.
3. `Ensemble_Combining_Any_Two_or_Three_Models.py` - Perform ensemble modeling combining any two or three individual models.
4. `Ensemble_CLR_of_SVM-RF-GBM_Model.py` - Perform CLR on the best ensemble model i.e., the SVM:RF:GBM model.
5. `Ensemble_SVM-RF-GBM_PMs_and_Probabilities.joblib` - The final CLR-calibrated SVM:RF:GBM ensemble model serialized into a Joblib file containing the exact PM values and predicted probabilities for each case in the training and test sets. Access this file in any Python environment using the following codes:
    ```python
    import joblib
    model = joblib.load('Ensemble_SVM-RF-GBM_PMs_and_Probabilities.joblib')
    print(model)
    ```

---

## For command line Git users
1. **Clone the repository** - ```git clone https://github.com/kahkengwong/PDAC_paper```
2. **Navigate to the directory** - ```cd PDAC_paper```
3. **Accessing the serialized model** - Ensure you have Python and necessary libraries (e.g., `joblib`) installed, then run: ```python -c "import joblib; model = joblib.load('Ensemble_SVM-RF-GBM_PMs_and_Probabilities.joblib'); print(model)"```

---

## References
This project obtained the RNA-seq dataset from GSE183635 available on the Gene Expression Omnibus database (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635) and in the original article by S. In 't Veld et al., 2022 (DOI: 10.1016/j.ccell.2022.08.006).

---

## Contact
For further information or questions, please email [kahkeng@usm.my](mailto:kahkeng@usm.my)

---
