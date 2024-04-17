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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
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
    
    # Best genes and hyperparameters for XGB
    XGB_best_gene_list = ["COX6B1", "TMEM258", "IFI27L2", "MYL6", "SRP14", "TNFAIP2", "GNG5", "DSTN", "FLOT1", "RNF181", "COX7A2", "PLD4", "FTL", "CALM1", "NCF4", "POLR2G", "SMDT1"]
    best_params = {'colsample_bylevel': 0.5304520069830586, 'colsample_bynode': 0.7993819079563386, 'colsample_bytree': 0.7832483772513955, 'gamma': 0.9092109617676173, 'learning_rate': 0.17940576800599883, 'max_depth': 10, 'max_leaves': 70, 'min_child_weight': 2.2614637182063495, 'n_estimators': 550, 'reg_alpha': 0.04220766936857312, 'reg_lambda': 0.11215936867022136, 'scale_pos_weight': 1.062885864425209, 'subsample': 0.6699054339325979}

    # Train the best model using the best parameters from TPE
    XGB_best_model = XGBClassifier(random_state=1, **best_params)
    XGB_best_model.fit(normalized_data[XGB_best_gene_list].values, normalized_data['status'])
    
    # Prediction and evaluation on the training set
    X_train = normalized_data[XGB_best_gene_list].values
    y_train = normalized_data['status']
    predicted_status_train = XGB_best_model.predict(X_train)
    predicted_probs_train = XGB_best_model.predict_proba(X_train)[:, 1] 
    normalized_data['Probability'] = predicted_probs_train

    # Prediction and evaluation on the test set
    X_test_train = normalized_test_data[XGB_best_gene_list].values
    y_test_train = normalized_test_data['status']
    test_set_predictions = XGB_best_model.predict(X_test_train)
    test_set_pred_probs = XGB_best_model.predict_proba(X_test_train)[:, 1]
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
    calibrated_model = CalibratedClassifierCV(XGB_best_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(normalized_calibration_data[XGB_best_gene_list].values, normalized_calibration_data['status'])

    # Get calibrated probabilities
    XGB_calibrated_probs_train = calibrated_model.predict_proba(X_train)[:, 1]
    XGB_calibrated_probs_test = calibrated_model.predict_proba(X_test_train)[:, 1]

    # Return the probabilities
    return XGB_best_model, XGB_best_gene_list, XGB_calibrated_probs_train, XGB_calibrated_probs_test

# Running the analysis and capturing returned probabilities
XGB_best_model, XGB_best_gene_list, XGB_calibrated_probs_train, XGB_calibrated_probs_test = run_analysis()