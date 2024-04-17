import os
import random
import numpy as np
import pandas as pd
import xlsxwriter
from sklearn.metrics import (accuracy_score, auc, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, log_loss, matthews_corrcoef, precision_recall_curve,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

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

train_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-60-Training_n=82.xlsx")
test_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-30-Test_n=41.xlsx")

# Extract actual labels
y_train = train_data['status'].values
y_test = test_data['status'].values

# Actual probabilities of train and test sets using ML modeling
SVM_calibrated_probs_train = SVM_calibrated_probs_train
SVM_calibrated_probs_test = SVM_calibrated_probs_test
LR_calibrated_probs_train = LR_calibrated_probs_train
LR_calibrated_probs_test = LR_calibrated_probs_test
RF_calibrated_probs_train = RF_calibrated_probs_train
RF_calibrated_probs_test = RF_calibrated_probs_test
XGB_calibrated_probs_train = XGB_calibrated_probs_train
XGB_calibrated_probs_test = XGB_calibrated_probs_test
LGBM_calibrated_probs_train = LGBM_calibrated_probs_train
LGBM_calibrated_probs_test = LGBM_calibrated_probs_test

##########
# For any two individual models ensemble
def ensemble_predictions(model1_probs, model2_probs, y_true, weights):
    if round(sum(weights), 2) != 1.0:
        raise ValueError("The sum of weights must be 1.")

    weight_model1, weight_model2 = weights
    ensemble_probs = (model1_probs * weight_model1) + (model2_probs * weight_model2)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    metrics = calculate_metrics(y_true, ensemble_preds, ensemble_probs)
    return metrics

# Function to iterate over weight combinations for any two models
def iterate_weights(model1_probs_train, model1_probs_test, model2_probs_train, model2_probs_test, model_names):
    all_results_df = pd.DataFrame()
    step_size = 0.00019

    for weight_model1 in np.arange(0, 1 + step_size, step_size):
        weight_model2 = 1 - weight_model1

        train_metrics = ensemble_predictions(model1_probs_train, model2_probs_train, y_train, (weight_model1, weight_model2))
        test_metrics = ensemble_predictions(model1_probs_test, model2_probs_test, y_test, (weight_model1, weight_model2))

        # Flatten and combine training and test metrics
        combined_metrics = {f'Train {k}': v for k, v in train_metrics.items()}
        combined_metrics.update({f'Test {k}': v for k, v in test_metrics.items()})
        combined_metrics[f'Weight {model_names[0]}'] = weight_model1
        combined_metrics[f'Weight {model_names[1]}'] = weight_model2

        # Append to all_results_df using pd.concat
        temp_df = pd.DataFrame([combined_metrics])
        all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)

    return all_results_df

# Iterate over model pairs
model_pairs = [
    ('SVM', 'LR'), 
    ('SVM', 'RF'), 
    ('SVM', 'XGB'), 
    ('SVM', 'LGBM'), 
    ('LR', 'RF'), 
    ('LR', 'XGB'), 
    ('LR', 'LGBM'), 
    ('RF', 'XGB'), 
    ('RF', 'LGBM'), 
    ('XGB', 'LGBM')]

for model1, model2 in model_pairs:
    model1_train_probs = eval(f"{model1}_calibrated_probs_train")
    model1_test_probs = eval(f"{model1}_calibrated_probs_test")
    model2_train_probs = eval(f"{model2}_calibrated_probs_train")
    model2_test_probs = eval(f"{model2}_calibrated_probs_test")

    results_df = iterate_weights(model1_train_probs, model1_test_probs, model2_train_probs, model2_test_probs, (model1, model2))
    results_df.to_excel(f'Ensemble_2_{model1}-{model2}_5k-models.xlsx', index=False)
##########

##########
# For any three individual models ensemble
def ensemble_predictions(model1_probs, model2_probs, model3_probs, y_true, weights):
    if round(sum(weights), 2) != 1.0:
        raise ValueError("The sum of weights must be 1.")

    weight_model1, weight_model2, weight_model3 = weights
    ensemble_probs = (model1_probs * weight_model1) + (model2_probs * weight_model2) + (model3_probs * weight_model3)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    metrics = calculate_metrics(y_true, ensemble_preds, ensemble_probs)
    return metrics

# Function to iterate over weight combinations for any three models
def iterate_weights(model1_probs_train, model1_probs_test, model2_probs_train, model2_probs_test, model3_probs_train, model3_probs_test, model_names):
    all_results_df = pd.DataFrame()
    step_size = 0.01

    # Iterating through combinations of weights that sum up to 1
    for weight_model1 in np.arange(0, 1 + step_size, step_size):
        for weight_model2 in np.arange(0, 1 - weight_model1 + step_size, step_size):
            weight_model3 = 1 - weight_model1 - weight_model2
            if weight_model3 < 0 or weight_model3 > 1:
                continue

            train_metrics = ensemble_predictions(model1_probs_train, model2_probs_train, model3_probs_train, y_train, (weight_model1, weight_model2, weight_model3))
            test_metrics = ensemble_predictions(model1_probs_test, model2_probs_test, model3_probs_test, y_test, (weight_model1, weight_model2, weight_model3))

            # Flatten and combine training and test metrics
            combined_metrics = {f'Train {k}': v for k, v in train_metrics.items()}
            combined_metrics.update({f'Test {k}': v for k, v in test_metrics.items()})
            combined_metrics[f'Weight {model_names[0]}'] = weight_model1
            combined_metrics[f'Weight {model_names[1]}'] = weight_model2
            combined_metrics[f'Weight {model_names[2]}'] = weight_model3

            # Append to all_results_df using pd.concat
            temp_df = pd.DataFrame([combined_metrics])
            all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)

    return all_results_df

model_triplets = [
    ('LR', 'RF', 'SVM'),
    ('LR', 'RF', 'XGB'),
    ('LR', 'RF', 'LGBM'),
    ('LR', 'SVM', 'XGB'),
    ('LR', 'SVM', 'LGBM'),
    ('LR', 'XGB', 'LGBM'),
    ('RF', 'SVM', 'XGB'),
    ('RF', 'SVM', 'LGBM'),
    ('RF', 'XGB', 'LGBM'),
    ('SVM', 'XGB', 'LGBM'),
]

for model1, model2, model3 in model_triplets:
    model1_train_probs = eval(f"{model1}_calibrated_probs_train")
    model1_test_probs = eval(f"{model1}_calibrated_probs_test")
    model2_train_probs = eval(f"{model2}_calibrated_probs_train")
    model2_test_probs = eval(f"{model2}_calibrated_probs_test")
    model3_train_probs = eval(f"{model3}_calibrated_probs_train")
    model3_test_probs = eval(f"{model3}_calibrated_probs_test")

    results_df = iterate_weights(model1_train_probs, model1_test_probs, model2_train_probs, model2_test_probs, model3_train_probs, model3_test_probs, (model1, model2, model3))
    results_df.to_excel(f'Ensemble_3_{model1}-{model2}-{model3}_models.xlsx', index=False)
##########