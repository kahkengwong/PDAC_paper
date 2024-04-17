import os
import random
import numpy as np
import pandas as pd
import xlsxwriter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, auc, brier_score_loss, cohen_kappa_score, confusion_matrix,
                             log_loss, matthews_corrcoef, precision_recall_curve, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import json
from joblib import dump
    
# Load datasets
train_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-60-Training_n=82.xlsx")
test_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-30-Test_n=41.xlsx")

# Actual probabilities of train and test sets using ML modeling
SVM_calibrated_probs_train = SVM_calibrated_probs_train
SVM_calibrated_probs_test = SVM_calibrated_probs_test
RF_calibrated_probs_train = RF_calibrated_probs_train
RF_calibrated_probs_test = RF_calibrated_probs_test
GBM_calibrated_probs_train = GBM_calibrated_probs_train
GBM_calibrated_probs_test = GBM_calibrated_probs_test

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_proba):
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()    
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    f2_score = (5 * precision * sensitivity) / (4 * precision + sensitivity) if (precision + sensitivity) != 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    auprc = auc(recall_curve, precision_curve)
    brier = brier_score_loss(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    sharpness = np.std(y_proba)

    metrics = {'Confusion Matrix': conf_matrix, 'Sensitivity': sensitivity, 'Specificity': specificity, 'Accuracy': accuracy, 'Precision': precision, 'F1 Score': f1_score, 'F2 Score': f2_score, 'MCC': mcc, 'Kappa': kappa, 'ROC AUC': roc_auc, 'AUPRC': auprc, 'Brier score': brier, 'Logloss': logloss, 'Sharpness': sharpness, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP}

    return metrics

##########
# To perform CLR on the SVM:RF:GBM ensemble model
# Function to transform probabilities with custom logistic recalibration (CLR)
def transform_probabilities(probs, B0=0, B1=1):
    transformed_probs = 1 / (1 + np.exp(-(B0 + B1 * probs)))
    return transformed_probs

# Weights for the models
weights = [0.418, 0.448, 0.134]

# Function for ensemble predictions before CLR
def ensemble_predictions_pre_calibration(model_probs_train, model_probs_test, weights, y_train, y_test):
    ensemble_probs_train = np.sum([probs * weight for probs, weight in zip(model_probs_train, weights)], axis=0)
    ensemble_probs_test = np.sum([probs * weight for probs, weight in zip(model_probs_test, weights)], axis=0)

    train_metrics = calculate_metrics(y_train, (ensemble_probs_train > 0.5).astype(int), ensemble_probs_train)
    test_metrics = calculate_metrics(y_test, (ensemble_probs_test > 0.5).astype(int), ensemble_probs_test)

    return train_metrics, test_metrics, ensemble_probs_train, ensemble_probs_test

# Function for ensemble predictions after CLR
def ensemble_predictions_post_calibration(ensemble_probs_train, ensemble_probs_test, y_train, y_test, B0, B1):
    transformed_probs_train = transform_probabilities(ensemble_probs_train, B0, B1)
    transformed_probs_test = transform_probabilities(ensemble_probs_test, B0, B1)

    train_metrics = calculate_metrics(y_train, (transformed_probs_train > 0.5).astype(int), transformed_probs_train)
    test_metrics = calculate_metrics(y_test, (transformed_probs_test > 0.5).astype(int), transformed_probs_test)

    return train_metrics, test_metrics, transformed_probs_train, transformed_probs_test

# B0-B1 calibration ranges and combinations
B0_range = np.linspace(-15, 10, 100)
B1_range = np.linspace(0, 25, 100)

# Generate B0-B1 combinations
B0_B1_combinations = [(round(B0, 4), round(B1, 4)) for B0 in B0_range for B1 in B1_range]

# Limit to 50000 combinations
B0_B1_combinations = B0_B1_combinations[:50000]

# DataFrame to store all results
all_results_df = pd.DataFrame()

# Loop for B0-B1 Combinations
for B0, B1 in B0_B1_combinations:
    # Metrics before calibration
    train_metrics_pre, test_metrics_pre, ensemble_probs_train, ensemble_probs_test = ensemble_predictions_pre_calibration(
        [SVM_calibrated_probs_train, RF_calibrated_probs_train, GBM_calibrated_probs_train],
        [SVM_calibrated_probs_test, RF_calibrated_probs_test, GBM_calibrated_probs_test],
        weights, train_data['status'].values, test_data['status'].values
    )

    # Metrics after calibration
    train_metrics_post, test_metrics_post, transformed_probs_train, transformed_probs_test = ensemble_predictions_post_calibration(
        ensemble_probs_train, ensemble_probs_test, 
        train_data['status'].values, test_data['status'].values, 
        B0, B1
    )

    # Compile metrics
    combined_metrics = {
        **{f'Pre-Calibration Training {k}': v for k, v in train_metrics_pre.items()},
        **{f'Pre-Calibration Test {k}': v for k, v in test_metrics_pre.items()},
        **{f'Post-Calibration Training {k}': v for k, v in train_metrics_post.items()},
        **{f'Post-Calibration Test {k}': v for k, v in test_metrics_post.items()},
        'B0': B0, 'B1': B1
    }

    # Add to DataFrame
    temp_df = pd.DataFrame([combined_metrics])
    all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)

# Export the results
excel_writer = pd.ExcelWriter('CLR_calibration_of_SVM-RF-GBM-Ensemble-model.xlsx', engine='openpyxl')
all_results_df.to_excel(excel_writer, sheet_name='Metrics', index=False)
excel_writer.close()
##########

##########
# To extract the final model's PM and probabilities before and after CLR, and to serialize the final model to JSON and Joblib files
# Best B0 and B1 values
best_B0 = -8.9296
best_B1 = 18.197	

# Configuration settings
config = {
    "best_B0": -8.9296,
    "best_B1": 18.197,
    "weights_SVM_RF_GBM": [0.418, 0.448, 0.134] 
}

# Serialize to JSON
with open('CLR_SVM-RF-GBM_Ensemble_vFinal.json', 'w') as f:
    json.dump(config, f, indent=4)

# Metrics before calibration
train_metrics_pre, test_metrics_pre, ensemble_probs_train_pre, ensemble_probs_test_pre = ensemble_predictions_pre_calibration(
    [SVM_calibrated_probs_train, RF_calibrated_probs_train, GBM_calibrated_probs_train],
    [SVM_calibrated_probs_test, RF_calibrated_probs_test, GBM_calibrated_probs_test],
    weights, train_data['status'].values, test_data['status'].values
)

# Metrics after calibration
train_metrics_post, test_metrics_post, ensemble_probs_train_post, ensemble_probs_test_post = ensemble_predictions_post_calibration(
    ensemble_probs_train_pre, ensemble_probs_test_pre, 
    train_data['status'].values, test_data['status'].values, 
    best_B1, best_B0
)

# Model data
model_data = {
    "train_metrics_pre": train_metrics_pre,
    "test_metrics_pre": test_metrics_pre,
    "ensemble_probs_train_pre": ensemble_probs_train_pre,
    "ensemble_probs_test_pre": ensemble_probs_test_pre,
    "train_metrics_post": train_metrics_post,
    "test_metrics_post": test_metrics_post,
    "ensemble_probs_train_post": ensemble_probs_train_post,
    "ensemble_probs_test_post": ensemble_probs_test_post
}

# Serialize and save to Joblib file
dump(model_data, 'CLR_SVM-RF-GBM_Ensemble_vFinal.joblib')

# Compile all the results into a DataFrame
compiled_results = {
    **{f'Pre-Calibration Training {k}': [v] for k, v in train_metrics_pre.items()},
    **{f'Pre-Calibration Test {k}': [v] for k, v in test_metrics_pre.items()},
    **{f'Post-Calibration Training {k}': [v] for k, v in train_metrics_post.items()},
    **{f'Post-Calibration Test {k}': [v] for k, v in test_metrics_post.items()},
    'Best B1': [best_B1],
    'Best B0': [best_B0]
}

compiled_df = pd.DataFrame(compiled_results)

# Adding individual case probabilities to the DataFrame
train_probs_df_pre = pd.DataFrame({'ID': train_data['ID'], 'Status': train_data['status'], 'Train Probabilities Pre-Calibration': ensemble_probs_train_pre})
train_probs_df_post = pd.DataFrame({'Train Probabilities Post-Calibration': ensemble_probs_train_post})
test_probs_df_pre = pd.DataFrame({'ID': test_data['ID'], 'Status': test_data['status'], 'Test Probabilities Pre-Calibration': ensemble_probs_test_pre})
test_probs_df_post = pd.DataFrame({'Test Probabilities Post-Calibration': ensemble_probs_test_post})

# Combining everything into a single DataFrame
final_df = pd.concat([compiled_df, train_probs_df_pre, train_probs_df_post, test_probs_df_pre, test_probs_df_post], axis=1)

# Exporting the results
excel_writer = pd.ExcelWriter('PM_Probabilities_CLR_SVM-RF-GBM_Ensemble.xlsx', engine='openpyxl')
final_df.to_excel(excel_writer, sheet_name='Metrics and Probabilities', index=False)
excel_writer.close()
##########