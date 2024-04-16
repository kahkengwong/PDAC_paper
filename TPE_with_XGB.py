import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, log_loss, matthews_corrcoef, precision_recall_curve,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Define GL1
core_genes = ["COX6B1", "TMEM258", "IFI27L2", "MYL6", "SRP14", "TNFAIP2", "GNG5", "DSTN", "FLOT1", "RNF181"]
additional_genes = ["COX7A2", "PLD4", "FTL", "CALM1", "NCF4", "POLR2G", "SMDT1", "NDUFA3", "SELENOW", "COX17"]
gene_lists = [core_genes] + [core_genes + additional_genes[:i] for i in range(1, len(additional_genes) + 1)]

# Use the following 'core_genes' and 'additional_genes' for GL2
# core_genes = ["IFITM2", "MSRB2", "ELOB", "PFN1", "ATP5MG", "UBL5", "CMTM2", "DYNLRB1", "COX5B", "NOP10"]
# additional_genes = ["ZBTB8OS", "RPL41", "WASF2", "ATP5PF", "FIS1", "TNFRSF10C", "TSTA3", "NDUFB4", "DAD1", "COX7B"]

# Use the following 'core_genes' and 'additional_genes' for GL3
# core_genes = ["CARD16", "GCA", "CASP4", "FCER1G", "METTL22", "ATP5MPL", "CDC34", "NDUFA1", "TP53TG1", "PRSS50"]
# additional_genes = ["EIF1", "AC074327.1", "ATP5MD", "NDUFB10", "SCNM1", "ATP5F1E", "ANAPC11", "OST4", "METTL26", "VAMP8"]

# Use the following 'core_genes' and 'additional_genes' for GL4
# core_genes = ["GMFG", "CDA", "TXN", "GUK1", "SEC62", "MGST3", "NBDY", "CHMP2A", "MSRB1", "COX4I1"]
# additional_genes = ["NDUFB6", "QPCT", "NDUFS4", "UBE2C", "UBB", "NAA38", "PTPN1", "SFT2D1", "SSR4", "POLR2J"]

# Define CSFs
def CSF1(y_true, y_pred):
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    return sensitivity * 0.5 + specificity * 0.5

def CSF2(y_true, y_pred):
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    return sensitivity * 0.55 + specificity * 0.45

def CSF3(y_true, y_pred):
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    return sensitivity * 0.6 + specificity * 0.4

# Function to plot CSF score over TPE iterations
def plot_tpe_performance_seaborn(trials, num_genes, csf_name, max_evals):
    csf_scores = [-trial['result']['loss'] for trial in trials.trials]
    highest_score_trial = max(range(len(csf_scores)), key=csf_scores.__getitem__)
    highest_score = csf_scores[highest_score_trial]
    data = pd.DataFrame({'Trial': range(1, len(csf_scores) + 1), f'{csf_name} Score': csf_scores})
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Trial', y=f'{csf_name} Score', marker='o', markersize=5)
    plt.scatter(highest_score_trial + 1, highest_score, color='red', s=50, zorder=5)
    plt.title(f'TPE Performance over Iterations ({csf_name})')
    plt.xlabel('Iteration Number')
    plt.ylabel(f'{csf_name} Score')
    plt.grid(True)
    filename = f'GBM_{num_genes}genes_{csf_name}_{max_evals}_v2.svg'
    plt.savefig(filename, format='svg')
    plt.show()

# Perform the training with TPE and GBM, obtain the best model and its performance metrics in training and test sets before and after Platt calibration
def run_analysis(max_evals, n_startup_jobs, gene_list, csf, csf_name):
    np.random.seed(1) 
    random.seed(1) 
    full_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-60-Training_n=82.xlsx")
    test_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-30-Test_n=41.xlsx")

    # Preprocess training data
    expression_data = full_data.drop(['ID', 'status'], axis=1)
    expression_data_plus1 = np.log2(expression_data + 1)
    expression_data_plus1.columns = expression_data_plus1.columns.astype(str)
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(expression_data_plus1), columns=expression_data_plus1.columns)
    normalized_data['ID'] = full_data['ID']
    normalized_data['status'] = full_data['status']

    # Preprocess test data
    test_expression_data = test_data.drop(['ID', 'status'], axis=1)
    test_expression_data_plus1 = np.log2(test_expression_data + 1)
    test_expression_data_plus1.columns = test_expression_data_plus1.columns.astype(str)
    normalized_test_data = pd.DataFrame(scaler.transform(test_expression_data_plus1), columns=test_expression_data_plus1.columns)
    normalized_test_data['ID'] = test_data['ID']
    normalized_test_data['status'] = test_data['status']

    # Initialize 10-fold cross-validation and define hyperparameters space
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    all_metrics_df = pd.DataFrame()
    space = {
        'max_depth': hp.choice('max_depth', range(3, 11)),
        'max_leaves': hp.choice('max_leaves', range(1, 256)),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'n_estimators': hp.choice('n_estimators', range(100, 1000, 50)),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
        'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
        'gamma': hp.uniform('gamma', 0, 5),
        'reg_lambda': hp.uniform('reg_lambda', 0.1, 15),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 100)
    }

    iteration_results = pd.DataFrame(columns=['Iteration', 'Score', 'Hyperparameters'])
    
    # Define and perform TPE
    def objective(params):
        model = XGBClassifier(random_state=1, **params)
        score = cross_val_score(model, normalized_data[gene_list].values, normalized_data['status'], 
                            scoring=make_scorer(csf), cv=kfold, n_jobs=-1).mean()
        
        iteration_results.loc[len(iteration_results)] = {'Iteration': len(trials.trials) + 1, 'Score': score, 'Hyperparameters': params}

        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    gamma_val = 0.5
    best_params = fmin(
        fn=objective, 
        space=space, 
        algo=lambda *args, **kwargs: tpe.suggest(*args, **kwargs, n_startup_jobs=n_startup_jobs, gamma=gamma_val),
        max_evals=max_evals, 
        trials=trials
    )
    
    # Find the trial with the highest custom score
    best_trial_result = max(trials.results, key=lambda x: -x['loss'])
    best_trial_index = trials.results.index(best_trial_result)
    best_trial_score = -best_trial_result['loss'] 
    best_trial_iteration = best_trial_index + 1  
    print(f"Best Custom Score (CSF): {best_trial_score}, Iteration: {best_trial_iteration}")
    plot_tpe_performance_seaborn(trials, len(gene_list), csf_name, max_evals)
    best_trial_params_raw = trials.trials[best_trial_index]['misc']['vals']

    # Lists for hp.choice parameters
    sample_type_options = ['uniform', 'weighted']
    normalize_type_options = ['tree', 'forest']
    # Extracting parameters from the best trial and converting them
    best_trial_params_raw = trials.trials[best_trial_index]['misc']['vals']
    best_trial_params_converted = {}
    # Iterating through the raw parameters for conversion
    for key, val in best_trial_params_raw.items():
        if key == 'max_depth':
            best_trial_params_converted[key] = range(3, 11)[int(val[0])]
        elif key == 'max_leaves':
            best_trial_params_converted[key] = range(1, 256)[int(val[0])]
        elif key == 'n_estimators':
            best_trial_params_converted[key] = list(range(100, 1000, 50))[int(val[0])]
        else:
            best_trial_params_converted[key] = val[0] if val else None

    num_genes = len(gene_list)
    
    # Train the best model using the converted parameters
    best_model = XGBClassifier(random_state=1, **best_trial_params_converted)
    best_model.fit(normalized_data[gene_list].values, normalized_data['status'])

    # Prediction and evaluation on the training set
    X_full = normalized_data[gene_list].values
    y_full = normalized_data['status']
    predicted_status_full = best_model.predict(X_full) 
    predicted_probs_full = best_model.predict_proba(X_full)[:, 1]  
    normalized_data['Probability'] = predicted_probs_full 

    # Prediction and evaluation on the training set
    conf_matrix = confusion_matrix(y_full, predicted_status_full)
    TN, FP, FN, TP = conf_matrix.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    accuracy = (TP + TN) / (TP + FN + TN + FP) if (TP + FN + TN + FP) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    f2_score = (5 * precision * sensitivity) / (4 * precision + sensitivity) if (precision + sensitivity) != 0 else 0
    roc_auc = roc_auc_score(y_full, predicted_probs_full)
    precision_full, recall_full, _ = precision_recall_curve(y_full, predicted_probs_full)
    auprc_full = auc(recall_full, precision_full)
    kappa_train = cohen_kappa_score(y_full, predicted_status_full)
    mcc = matthews_corrcoef(y_full, predicted_status_full)
    brier_train = brier_score_loss(y_full, predicted_probs_full)
    logloss_train = log_loss(y_full, predicted_probs_full)
    sharpness_train = np.std(predicted_probs_full)

    # Prediction and evaluation on the test set
    X_test_full = normalized_test_data[gene_list].values
    y_test_full = normalized_test_data['status']
    test_set_predictions = best_model.predict(X_test_full) 
    test_set_pred_probs = best_model.predict_proba(X_test_full)[:, 1]
    normalized_test_data['Probability'] = test_set_pred_probs 

    # Generate confusion matrix and other metrics on the test set
    conf_matrix_test = confusion_matrix(y_test_full, test_set_predictions)
    TN_test, FP_test, FN_test, TP_test = conf_matrix_test.ravel()
    test_sensitivity = TP_test / (TP_test + FN_test) if (TP_test + FN_test) != 0 else 0
    test_specificity = TN_test / (TN_test + FP_test) if (TN_test + FP_test) != 0 else 0
    test_precision = TP_test / (TP_test + FP_test) if (TP_test + FP_test) != 0 else 0
    test_f1_score = 2 * (test_precision * test_sensitivity) / (test_precision + test_sensitivity) if (test_precision + test_sensitivity) != 0 else 0
    test_f2_score = (5 * test_precision * test_sensitivity) / (4 * test_precision + test_sensitivity) if (test_precision + test_sensitivity) != 0 else 0
    test_mcc = matthews_corrcoef(y_test_full, test_set_predictions)
    kappa_test = cohen_kappa_score(y_test_full, test_set_predictions)
    precision_test, recall_test, _ = precision_recall_curve(y_test_full, test_set_pred_probs)
    test_auprc = auc(recall_test, precision_test)
    test_set_accuracy = accuracy_score(y_test_full, test_set_predictions)
    test_set_auc = roc_auc_score(y_test_full, test_set_pred_probs)
    brier_test = brier_score_loss(y_test_full, test_set_pred_probs)
    logloss_test = log_loss(y_test_full, test_set_pred_probs)
    sharpness_test = np.std(test_set_pred_probs)

    # Append results for the combination of genes to the dataframe
    gene_metrics = pd.DataFrame({ 'Gene Combination': ' + '.join(gene_list), 'Best Hyperparameters': [best_trial_params_converted], 'Best Custom Score (CSF)': best_trial_score,'Iteration': best_trial_iteration, 'Training Sensitivity': sensitivity, 'Training Specificity': specificity, 'Training Accuracy': accuracy, 'Training Precision': precision, 'Training F1 Score': f1_score, 'Training F2 Score': f2_score, 'Training MCC': mcc, 'ROC AUC': roc_auc, 'Training AUPRC': auprc_full, 'Training Brier Score': brier_train, 'Training Log Loss': logloss_train, 'Training Sharpness': sharpness_train, 'Training Kappa': kappa_train, 'Training TN': TN,'Training FP': FP, 'Training FN': FN, 'Training TP': TP, 'Test Set Sensitivity': test_sensitivity, 'Test Set Specificity': test_specificity, 'Test Set Accuracy': test_set_accuracy, 'Test Set Precision': test_precision, 'Test Set F1 Score': test_f1_score, 'Test Set F2 Score': test_f2_score, 'Test Set MCC': test_mcc, 'Test Set ROC AUC': test_set_auc, 'Test Set AUPRC': test_auprc, 'Test Brier Score': brier_test, 'Test Log Loss': logloss_test, 'Test Sharpness': sharpness_test, 'Test Kappa': kappa_test, 'Test Set TN': TN_test, 'Test Set FP': FP_test, 'Test Set FN': FN_test, 'Test Set TP': TP_test}, index=[0])

    all_metrics_df = pd.concat([all_metrics_df, gene_metrics], ignore_index=True)

    # Load and preprocess the calibration data
    calibration_data = pd.read_excel("PDAC_Ctrl_All_Cases_Split-10-Calib_n=13.xlsx")
    calibration_expression_data = calibration_data.drop(['ID', 'status'], axis=1)
    calibration_expression_data_plus1 = np.log2(calibration_expression_data + 1)
    calibration_expression_data_plus1.columns = calibration_expression_data_plus1.columns.astype(str)
    normalized_calibration_data = pd.DataFrame(scaler.transform(calibration_expression_data_plus1), columns=calibration_expression_data_plus1.columns)
    normalized_calibration_data['ID'] = calibration_data['ID']
    normalized_calibration_data['status'] = calibration_data['status']

    # Calibrate the model using the calibration set
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(normalized_calibration_data[gene_list].values, normalized_calibration_data['status'])
    calibrated_probs_train = calibrated_model.predict_proba(X_full)[:, 1]
    calibrated_probs_test = calibrated_model.predict_proba(X_test_full)[:, 1]
    normalized_data['Calibrated Probability'] = calibrated_probs_train
    normalized_test_data['Calibrated Probability'] = calibrated_probs_test

    # Generate confusion matrix and other metrics on the training set with calibrated probabilities
    calib_predicted_status_full = (calibrated_probs_train >= 0.5).astype(int)
    conf_matrix_calib = confusion_matrix(y_full, calib_predicted_status_full)
    TN_calib, FP_calib, FN_calib, TP_calib = conf_matrix_calib.ravel()
    calib_sensitivity = TP_calib / (TP_calib + FN_calib) if (TP_calib + FN_calib) != 0 else 0
    calib_specificity = TN_calib / (TN_calib + FP_calib) if (TN_calib + FP_calib) != 0 else 0
    calib_accuracy = (TP_calib + TN_calib) / (TP_calib + FN_calib + TN_calib + FP_calib) if (TP_calib + FN_calib + TN_calib + FP_calib) != 0 else 0
    calib_precision = TP_calib / (TP_calib + FP_calib) if (TP_calib + FP_calib) != 0 else 0
    calib_f1_score = 2 * (calib_precision * calib_sensitivity) / (calib_precision + calib_sensitivity) if (calib_precision + calib_sensitivity) != 0 else 0
    calib_f2_score = (5 * calib_precision * calib_sensitivity) / (4 * calib_precision + calib_sensitivity) if (calib_precision + calib_sensitivity) != 0 else 0
    calib_mcc = matthews_corrcoef(y_full, calib_predicted_status_full)
    calib_precision_full, calib_recall_full, _ = precision_recall_curve(y_full, calibrated_probs_train)
    calib_auprc_train = auc(calib_recall_full, calib_precision_full)
    brier_calib_train = brier_score_loss(y_full, calibrated_probs_train)
    logloss_calib_train = log_loss(y_full, calibrated_probs_train)
    sharpness_calib_train = np.std(calibrated_probs_train)
    kappa_calib_train = cohen_kappa_score(y_full, calib_predicted_status_full)

    # Generate confusion matrix and other metrics on the test set with calibrated probabilities
    calib_test_set_predictions = (calibrated_probs_test >= 0.5).astype(int)
    conf_matrix_test_calib = confusion_matrix(y_test_full, calib_test_set_predictions)
    TN_test_calib, FP_test_calib, FN_test_calib, TP_test_calib = conf_matrix_test_calib.ravel()
    test_calib_sensitivity = TP_test_calib / (TP_test_calib + FN_test_calib) if (TP_test_calib + FN_test_calib) != 0 else 0
    test_calib_specificity = TN_test_calib / (TN_test_calib + FP_test_calib) if (TN_test_calib + FP_test_calib) != 0 else 0
    test_calib_precision = TP_test_calib / (TP_test_calib + FP_test_calib) if (TP_test_calib + FP_test_calib) != 0 else 0
    test_calib_f1_score = 2 * (test_calib_precision * test_calib_sensitivity) / (test_calib_precision + test_calib_sensitivity) if (test_calib_precision + test_calib_sensitivity) != 0 else 0
    test_calib_f2_score = (5 * test_calib_precision * test_calib_sensitivity) / (4 * test_calib_precision + test_calib_sensitivity) if (test_calib_precision + test_calib_sensitivity) != 0 else 0
    test_calib_mcc = matthews_corrcoef(y_test_full, calib_test_set_predictions)
    calib_precision_test, calib_recall_test, _ = precision_recall_curve(y_test_full, calibrated_probs_test)
    test_calib_auprc = auc(calib_recall_test, calib_precision_test)
    brier_calib_test = brier_score_loss(y_test_full, calibrated_probs_test)
    logloss_calib_test = log_loss(y_test_full, calibrated_probs_test)
    sharpness_calib_test = np.std(calibrated_probs_test)
    kappa_calib_test = cohen_kappa_score(y_test_full, calib_test_set_predictions)

    # Calibrated accuracy and ROC AUC for the training set
    calib_train_accuracy = accuracy_score(y_full, calib_predicted_status_full)
    calib_train_auc = roc_auc_score(y_full, calibrated_probs_train)

    # Calibrated accuracy and ROC AUC for the test set
    calib_test_accuracy = accuracy_score(y_test_full, calib_test_set_predictions)
    calib_test_auc = roc_auc_score(y_test_full, calibrated_probs_test)

    # Append recalculated performance metrics to gene_metrics
    calib_metrics = pd.DataFrame({'Calib Training Sensitivity': calib_sensitivity, 'Calib Training Specificity': calib_specificity, 'Calib Training Accuracy': calib_train_accuracy, 'Calib Training Precision': calib_precision, 'Calib Training F1 Score': calib_f1_score, 'Calib Training F2 Score': calib_f2_score, 'Calib Training MCC': calib_mcc, 'Calib Training ROC AUC': calib_train_auc, 'Calib Training AUPRC': calib_auprc_train, 'Calib Training Brier Score': brier_calib_train, 'Calib Training Log Loss': logloss_calib_train, 'Calib Training Sharpness': sharpness_calib_train, 'Calib Training Kappa': kappa_calib_train, 'Calib Training TN': TN_calib, 'Calib Training FP': FP_calib, 'Calib Training FN': FN_calib, 'Calib Training TP': TP_calib, 'Calib Test Set Sensitivity': test_calib_sensitivity, 'Calib Test Set Specificity': test_calib_specificity, 'Calib Test Set Accuracy': calib_test_accuracy, 'Calib Test Set Precision': test_calib_precision, 'Calib Test Set F1 Score': test_calib_f1_score, 'Calib Test Set F2 Score': test_calib_f2_score, 'Calib Test Set MCC': test_calib_mcc, 'Calib Test Set ROC AUC': calib_test_auc, 'Calib Test Set AUPRC': test_calib_auprc, 'Calib Test Set Brier Score': brier_calib_test, 'Calib Test Set Log Loss': logloss_calib_test, 'Calib Test Set Sharpness': sharpness_calib_test, 'Calib Test Set Kappa': kappa_calib_test, 'Calib Test Set TN': TN_test_calib, 'Calib Test Set FP': FP_test_calib, 'Calib Test Set FN': FN_test_calib, 'Calib Test Set TP': TP_test_calib}, index=[0])

    gene_metrics = pd.concat([gene_metrics, calib_metrics], axis=1)

    # Extract relevant columns (ID, status, probabilities) from the training and test datasets
    training_probabilities = normalized_data[['ID', 'status', 'Probability', 'Calibrated Probability']]
    test_probabilities = normalized_test_data[['ID', 'status', 'Probability', 'Calibrated Probability']]

    # Concatenate these with the gene_metrics dataframe
    training_data_with_metrics = pd.concat([training_probabilities, gene_metrics], axis=1)
    test_data_with_metrics = pd.concat([test_probabilities, gene_metrics], axis=1)

    # Save all metrics and probabilities
    filename = f"XGB_{num_genes}genes_{csf_name}_{max_evals}.xlsx"
    with pd.ExcelWriter(filename) as writer:
        training_data_with_metrics.to_excel(writer, sheet_name='Training Data', index=False)
        test_data_with_metrics.to_excel(writer, sheet_name='Test Data', index=False)
        iteration_results.to_excel(writer, sheet_name='Iteration Results', index=False)
    
    return trials

iteration_counts = [500, 1000]

csfs = [(CSF1, 'CSF1'), (CSF2, 'CSF2'), (CSF3, 'CSF3')]

for gene_list in gene_lists:
    for count in iteration_counts:
        n_startup_jobs = count // 2
        for csf, csf_name in csfs:
            print(f"Running analysis for {len(gene_list)} genes with {count} iterations and {csf_name}...")
            trials = run_analysis(max_evals=count, n_startup_jobs=n_startup_jobs, gene_list=gene_list, csf=csf, csf_name=csf_name)