<div align="center">

# 🧬 Ensemble Machine Learning for Pancreatic Ductal Adenocarcinoma (PDAC) Classification
### ✨ Optimizing ML models with Tree-structured Parzen Estimator (TPE) for hyperparameter tuning and ensemble modeling, with 3D visualization of optimization processes and custom scoring for interpretability.

![Project Status](https://img.shields.io/badge/status-active-brightgreen?logo=check&logoColor=white)
[![Project Page](https://img.shields.io/badge/Code-GitHub-4E81BE?logo=github&logoColor=white)](https://github.com/kahkengwong/PDAC_paper)
![GitHub top language](https://img.shields.io/github/languages/top/kahkengwong/PDAC_paper?logo=python&logoColor=3572A5)
![GitHub languages](https://img.shields.io/github/languages/count/kahkengwong/PDAC_paper)
![GitHub repo size](https://img.shields.io/github/repo-size/kahkengwong/PDAC_paper)
![GitHub last commit](https://img.shields.io/github/last-commit/kahkengwong/PDAC_paper)
![Contributors](https://img.shields.io/github/contributors/kahkengwong/PDAC_paper)
[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-blue?logo=elsevier&logoColor=white)](https://www.sciencedirect.com/science/article/pii/S1746809425003787)
[![Preprint](https://img.shields.io/badge/Preprint-SSRN-red?logo=ssrn&logoColor=white)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4809282)
[![Dataset](https://img.shields.io/badge/Dataset-GEO-B08C00?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAABGUlEQVQ4jZ2QsUoDQRSGv7e2VrBKrEVsLOwEG1tb2NjY2NoqdgI7O7Gxs7GxsbEVrKKwsbGxsbEVrKIECxsb+z8M3+Ge8dELHly4sHPn3rx5MxgMHA4H2t3dDQaD4Xq9YrVaYRiG2+3WarX6/Pyc2Wzmi4uL5PN5+Xw+nU6nVqtVq9VqtVqWZZFOp1OpVKrVarVa/X5/sVi8vr7i4uJiOp3OZrO5vb29VCoVCoVCoVAoFApVKhUKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQKhUKhUCgUCoVCoVAoFAqFQqFQKBQKheL+/v7+/v6Ojo7+/v7+/v6Ojo5Go9FoNBr9P38B3zJ+5Zf9H5YAAAAASUVORK5CYII=)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635)

**Author:** Kah Keng Wong

</div>

---

## 📋 Overview
This repository contains the code and data for a machine learning (ML) study focused on classifying **Pancreatic Ductal Adenocarcinoma (PDAC)** using RNA-seq data. The project leverages **ensemble machine learning** with **Tree-structured Parzen Estimator (TPE)** for hyperparameter optimization (HPO) and tracks the **objective function (OF)** using a **custom scoring function (CSF)** for enhanced model interpretability. Key components include:

- **Differential Expression Analysis**: Identification of differentially expressed genes (DEGs) using edgeR for PDAC and asymptomatic control cases.
- **ML Pipeline**: Implementation of five ML algorithms (Logistic Regression, SVM, Random Forest, XGBoost, and Gradient Boosting Machine) with TPE-based HPO.
- **Ensemble Modeling**: Combining two or three models, with a focus on the best-performing **SVM:RF:GBM ensemble** calibrated using Calibrated Logistic Regression (CLR).
- **Visualization**: 3D plots of HPO and OF tracking to provide insights into model optimization and performance.

The dataset is sourced from the **GSE183635** RNA-seq dataset available on the [Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635), as published by S. In 't Veld et al., 2022 ([DOI: 10.1016/j.ccell.2022.08.006](https://doi.org/10.1016/j.ccell.2022.08.006)).

---

## 🎯 Aims of the Project
The objectives of this study are:
1. To optimize five ML algorithms (LR, SVM, RF, XGB, GBM) using TPE for hyperparameter tuning and evaluate their performance on PDAC classification.
2. To develop and evaluate ensemble models combining two or three algorithms, with a focus on the SVM:RF:GBM ensemble for superior predictive accuracy.
3. To enhance model interpretability using a custom scoring function (CSF) and 3D visualizations of HPO and OF tracking.
4. To provide a serialized, reproducible ensemble model (`Ensemble_SVM-RF-GBM_PMs_and_Probabilities.joblib`) for external validation and reuse.

---

## 🔀 Workflow of the Project
![Workflow](https://raw.githubusercontent.com/kahkengwong/PDAC_paper/main/Project_Workflow.jpg)

- **Data Preprocessing**: Differential expression analysis using edgeR (`PDAC_DEGs_edgeR.r`) to identify DEGs from the RNA-seq expression matrix.
- **ML Model Optimization**: TPE-based HPO for five ML algorithms, followed by selection of the best gene combinations and hyperparameters.
- **Ensemble Modeling**: Combining models (e.g., SVM:RF:GBM) and calibrating the best ensemble using CLR.
- **Model Serialization**: Saving the final SVM:RF:GBM ensemble model with predicted probabilities and performance metrics in a Joblib file.
- **Visualization**: Generating 3D plots to visualize HPO and OF tracking for interpretability.

---

## 📊 Project Key Findings
- TPE-based HPO effectively identified optimal hyperparameters, improving model performance across all algorithms.
- The **custom scoring function (CSF)** enhanced interpretability by quantifying model performance and feature importance.
- 3D visualizations of HPO and OF tracking provided clear insights into the optimization process and model behavior, and the example results:
![Results1](https://raw.githubusercontent.com/kahkengwong/PDAC_paper/main/Results1.jpg)

- The **SVM:RF:GBM ensemble** outperformed individual models, achieving high accuracy and robustness in PDAC classification: 
![Results2](https://raw.githubusercontent.com/kahkengwong/PDAC_paper/main/Results2.jpg)

---

## 📊 Dataset and ML Scripts
### Dataset Folder
- **Expression Matrix**: RNA-seq data for PDAC and asymptomatic control cases, used for DEG analysis with edgeR (`PDAC_DEGs_edgeR.r`).
- **ML Datasets**: Preprocessed training, test, and calibration sets for ML model development and evaluation.
- **Source**: Publicly available RNA-seq dataset from [GSE183635](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635).

### ML Scripts
The scripts should be executed in the following order to replicate the study:

| No | Script File | Description |
|----|-------------|-------------|
| 1  | `PDAC_DEGs_edgeR.r` | Performs differential expression analysis using edgeR to identify DEGs from the RNA-seq expression matrix. |
| 2  | `TPE_with_LR.py` | Optimizes Logistic Regression (LR) using TPE for HPO, evaluating performance on PDAC classification. |
| 3  | `TPE_with_SVM.py` | Optimizes Support Vector Machine (SVM) using TPE for HPO. |
| 4  | `TPE_with_RF.py` | Optimizes Random Forest (RF) using TPE for HPO. |
| 5  | `TPE_with_XGB.py` | Optimizes XGBoost (XGB) using TPE for HPO. |
| 6  | `TPE_with_GBM.py` | Optimizes Gradient Boosting Machine (GBM) using TPE for HPO. |
| 7  | `Pre-ensemble_LR.py` | Executes the best gene combination and hyperparameters for LR, returning probabilities. |
| 8  | `Pre-ensemble_SVM.py` | Executes the best gene combination and hyperparameters for SVM, returning probabilities. |
| 9  | `Pre-ensemble_RF.py` | Executes the best gene combination and hyperparameters for RF, returning probabilities. |
| 10 | `Pre-ensemble_XGB.py` | Executes the best gene combination and hyperparameters for XGB, returning probabilities. |
| 11 | `Pre-ensemble_GBM.py` | Executes the best gene combination and hyperparameters for GBM, returning probabilities. |
| 12 | `Ensemble_Combining_Any_Two_`<br>`or_Three_Models.py` | Combines two or three ML models to create ensemble models, evaluating their performance. |
| 13 | `Ensemble_CLR_of_SVM-RF-GBM_`<br>`Model.py` | Calibrates the best SVM:RF:GBM ensemble model using CLR, generating final predictions. |
| 14 | `Ensemble_SVM-RF-GBM_PMs_and_`<br>`Probabilities.joblib` | Serialized Joblib file containing the final CLR-calibrated SVM:RF:GBM ensemble model, including performance metrics (PMs) and predicted probabilities. |

### Accessing the Serialized Model
To load and inspect the final ensemble model in a Python environment:
    ```python
    import joblib
    model = joblib.load('Ensemble_SVM-RF-GBM_PMs_and_Probabilities.joblib')
    print(model)
    ```

### For command line Git users
1. **Clone the repository** - ```git clone https://github.com/kahkengwong/PDAC_paper```
2. **Navigate to the directory** - ```cd PDAC_paper```
3. **Accessing the serialized model** - Ensure you have Python and necessary libraries (e.g., `joblib`) installed, then run: ```python -c "import joblib; model = joblib.load('Ensemble_SVM-RF-GBM_PMs_and_Probabilities.joblib'); print(model)"```

---

## 🧾 License

This project is licensed under the [MIT License](https://github.com/kahkengwong/PDAC_paper/blob/main/LICENSE), encouraging collaboration and reuse with proper attribution. See the [LICENSE](https://github.com/kahkengwong/PDAC_paper/blob/main/LICENSE) file for details.

## 🤝🏻 Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub for suggestions or improvements.

## 📚 Citation

If you found this study useful, please cite:

```bibtex
@article{wong2025ensemble,
  title={Ensemble machine learning and tree-structured Parzen estimator to predict early-stage pancreatic cancer},
  author={Wong, Kah Keng},
  journal={Biomedical Signal Processing and Control},
  volume={108},
  pages={107867},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.bspc.2025.107867},
  url={https://doi.org/10.1016/j.bspc.2025.107867}
}
```

Wong KK (2025). Ensemble machine learning and tree-structured Parzen estimator to predict early-stage pancreatic cancer. *Biomedical Signal Processing and Control* 108: 107867. [DOI: 10.1016/j.bspc.2025.107867](https://doi.org/10.1016/j.bspc.2025.107867)

Please also cite the source dataset:

In 't Veld S et al. (2022). Integrated single-cell and bulk RNA sequencing in pancreatic cancer identifies monocyte signature associated with worse overall survival in patients. *Cancer Cell*. [DOI: 10.1016/j.ccell.2022.08.006](https://doi.org/10.1016/j.ccell.2022.08.006).

## 📩 Contact

For questions or further information, contact Kah Keng Wong: [kahkeng@usm.my](mailto:kahkeng@usm.my)
