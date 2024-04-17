import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import (accuracy_score, auc, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, log_loss, matthews_corrcoef, precision_recall_curve,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

def run_analysis():
    np.random.seed(1) 
    random.seed(1) 
    train_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-60-Training_n=82.xlsx")
    test_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-30-Test_n=41.xlsx")

    # Preprocess training data
    expression_data = train_data.drop(['ID', 'status'], axis=1)
    expression_data_plus1 = np.log2(expression_data + 1)
    expression_data_plus1.columns = expression_data_plus1.columns.astype(str)
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(expression_data_plus1), columns=expression_data_plus1.columns)
    normalized_data['ID'] = train_data['ID']
    normalized_data['status'] = train_data['status']

    # Preprocess test data
    test_expression_data = test_data.drop(['ID', 'status'], axis=1)
    test_expression_data_plus1 = np.log2(test_expression_data + 1)
    test_expression_data_plus1.columns = test_expression_data_plus1.columns.astype(str)
    normalized_test_data = pd.DataFrame(scaler.transform(test_expression_data_plus1), columns=test_expression_data_plus1.columns)
    normalized_test_data['ID'] = test_data['ID']
    normalized_test_data['status'] = test_data['status']
    
    # Best genes and hyperparameters for GBM
    GBM_best_gene_list = ["COX6B1", "TMEM258", "IFI27L2", "MYL6", "SRP14", "TNFAIP2", "GNG5", "DSTN", "FLOT1", "RNF181"]
    best_params = {'colsample_bytree': 0.9558355059476036, 'learning_rate': 0.0797347587963442, 'max_depth': 7, 'min_child_samples': 20, 'min_child_weight': 0.02683720841153188, 'min_split_gain': 0.009902230777907637, 'num_leaves': 27, 'reg_alpha': 0.030503513360885255, 'reg_lambda': 0.20651450414049877, 'subsample': 0.9254991481954664}
        
    # Train the best model
    GBM_best_model = LGBMClassifier(random_state=1, **best_params)
    GBM_best_model.fit(normalized_data[GBM_best_gene_list].values, normalized_data['status'])

    # Prediction and evaluation on the training set
    X_train = normalized_data[GBM_best_gene_list].values
    y_train = normalized_data['status']
    predicted_status_train = GBM_best_model.predict(X_train) 
    predicted_probs_train = GBM_best_model.predict_proba(X_train)[:, 1] 
    normalized_data['Probability'] = predicted_probs_train 

    # Prediction and evaluation on the test set
    X_test_train = normalized_test_data[GBM_best_gene_list].values
    y_test_train = normalized_test_data['status']
    test_set_predictions = GBM_best_model.predict(X_test_train)
    test_set_pred_probs = GBM_best_model.predict_proba(X_test_train)[:, 1]
    normalized_test_data['Probability'] = test_set_pred_probs 

    # Load and preprocess the calibration data
    calibration_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-10-Calib_n=13.xlsx")
    calibration_expression_data = calibration_data.drop(['ID', 'status'], axis=1)
    calibration_expression_data_plus1 = np.log2(calibration_expression_data + 1)
    calibration_expression_data_plus1.columns = calibration_expression_data_plus1.columns.astype(str)
    normalized_calibration_data = pd.DataFrame(scaler.transform(calibration_expression_data_plus1), columns=calibration_expression_data_plus1.columns)
    normalized_calibration_data['ID'] = calibration_data['ID']
    normalized_calibration_data['status'] = calibration_data['status']

    # Calibrate the model using the calibration set
    calibrated_model = CalibratedClassifierCV(GBM_best_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(normalized_calibration_data[GBM_best_gene_list].values, normalized_calibration_data['status'])

    # Get calibrated probabilities
    GBM_calibrated_probs_train = calibrated_model.predict_proba(X_train)[:, 1]
    GBM_calibrated_probs_test = calibrated_model.predict_proba(X_test_train)[:, 1]

    # Return the probabilities
    return GBM_best_model, GBM_best_gene_list, GBM_calibrated_probs_train, GBM_calibrated_probs_test

# Running the analysis and capturing returned probabilities
GBM_best_model, GBM_best_gene_list, GBM_calibrated_probs_train, GBM_calibrated_probs_test = run_analysis()