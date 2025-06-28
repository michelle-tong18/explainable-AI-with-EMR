#!/usr/bin/env python3
"""
Classification Models Training Pipeline

This script trains various classification models for medical intervention prediction
using different time frames and data types.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Model storage
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier  # Uncomment if using LightGBM

# Model Training and Evaluation
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, balanced_accuracy_score

import matplotlib.pyplot as plt

# Set up the root directory for imports
import pyrootutils
root = pyrootutils.setup_root(
    search_from=os.path.abspath(''),
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

# Custom imports
from utils.file_management.config_loader import load_yaml, process_config_values
from utils.file_management.file_manager import FileManager
from utils.model_utils.data_utils import load_data, visualize_data, split_data, MultiClassToBinaryConverter
from utils.model_utils.eval_utils import eval_report, load_json
from utils.model_utils.shap_utils import shap_analysis_pipeline, aggregate_shap_feature_ranking

import pdb

def load_configuration():
    """Load and process configuration from YAML file."""
    config_path = str(root) + '/config/LBP_cohort.yaml'
    config = process_config_values(load_yaml(config_path))
    return FileManager(config.get('file_directory'))


def get_model_storage():
    """Define and return the model storage dictionary with hyperparameters."""
    return {
        'decision_tree_best': {
            'model': DecisionTreeClassifier(random_state=0),
            'hyperparams': {
                'model__criterion': ['gini', 'entropy'],
                'model__splitter': ['best', 'random'],
                'model__max_depth': [None, 2, 4, 6, 8],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__class_weight': ['balanced'],
            },
        },
        'random_forest_best': {
            'model': RandomForestClassifier(random_state=0),
            'hyperparams': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [10, 15, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__bootstrap': [True, False],
                'model__class_weight': ['balanced', 'balanced_subsample', None],
            },
        },
        'bagging_best': {
            'model': BaggingClassifier(estimator=RandomForestClassifier(random_state=0), random_state=0),
            'hyperparams': {
                'model__n_estimators': [10, 50, 100],
                'model__max_samples': [0.5, 0.7, 1.0],
                'model__max_features': [0.5, 0.7, 1.0],
                'model__bootstrap': [True, False],
                'model__bootstrap_features': [True, False],
            },
        },
        'adaboost_best': {
            'model': AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), random_state=0),
            'hyperparams': {
                'model__n_estimators': [50, 100, 180, 200],
                'model__learning_rate': [0.01, 0.1, 0.5, 1.0],
            }
        },
        'XGBoost_best': {
            'model': XGBClassifier(random_state=0),
            'hyperparams': {
                'model__objective': ['binary:logistic'],
                'model__max_depth': [3, 4, 5, 6],
                'model__learning_rate': [0.01, 0.1, 0.5, 1.0],
                'model__n_estimators': [50, 100, 150, 200],
                'model__lambda': [0, 10, 50, 100],
                'model__alpha': [0, 10, 50, 100],
                'model__scale_pos_weight': [1, 1.4, 1.7, 3.4, 4.8, 8.2],
            }
        },
    }

# --- TRAINING ---

def train_models(master_data_path, master_encoded_data_path, save_path_reference, 
                time_frame_list, data_type_list, y_col, labels_dict, paired_labels, 
                model_storage, n_splits=5, version_num = 1):
    """Train classification models for all combinations of time frames and data types."""
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score, adjusted=True)
    
    for time_frame in time_frame_list:  
        for data_type in data_type_list: 
            analysis_path = save_path_reference.replace('MODEL', 'classification_1class_meds').replace('INDEPENDENT_VAR', f'v{version_num}/{data_type}/{time_frame}')
            os.makedirs(analysis_path, exist_ok=True)
            # debug statement
            print('made directory ', analysis_path)
            # Load and visualize data
            df_categorical, df_data, continuous = load_data(master_data_path, master_encoded_data_path, analysis_path, data_type, time_frame)
            visualize_data(df_categorical, analysis_path)

            # Split data
            df_dev, df_test = split_data(df_data, y_col)
            categorical = [col for col in df_data.columns if col not in continuous]
            BinaryLabelLoader = MultiClassToBinaryConverter(y_col, categorical_cols=categorical)

            # Process each label pair
            for X_train, y_train, names in BinaryLabelLoader.process_data_stream(df_dev, paired_labels):
                X_train, y_train = np.array(X_train), np.array(y_train)
                label_0, label_1, input_variable = names.values()
                label_0_name = 'rest' if label_0 == 'rest' else labels_dict[label_0]
                label_1_name = labels_dict[label_1]

                results = []
                models = {}

                # Train each model
                for model_name, model_info in model_storage.items():
                    model_save_path = f'{analysis_path}/pred_{label_0_name}_vs_{label_1_name}/{model_name}'
                    save_path = model_save_path + f'/metrics.json'

                    # Skip if already trained
                    if os.path.exists(save_path):
                        metrics_dict = load_json(save_path)
                        results.append(metrics_dict)
                        continue

                    # Create pipeline
                    pipeline = ImbPipeline([
                        ('scaler', MinMaxScaler()),
                        ('oversample', SMOTE(random_state=42)),
                        ('model', model_info['model'])
                    ])

                    # Grid search
                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=model_info['hyperparams'],
                        cv=kf,
                        scoring=balanced_accuracy_scorer,
                        n_jobs=-1,
                        verbose=0
                    )

                    print(f"Grid Search for {data_type} {time_frame} {model_name} {label_0_name}_vs_{label_1_name}")
                    grid_search.fit(X_train, y_train)

                    best_pipeline = grid_search.best_estimator_
                    best_params = grid_search.best_params_

                    # Cross-validation predictions
                    y_pred = cross_val_predict(best_pipeline, X_train, y_train, cv=kf, method='predict')
                    y_pred_prob = cross_val_predict(best_pipeline, X_train, y_train, cv=kf, method='predict_proba')[:, 1]

                    run_info = {
                        'data_type': data_type,
                        'time_frame': time_frame,
                        'label_0': label_0_name,
                        'label_1': label_1_name,
                        'inputs': input_variable,
                        'model_name': model_name,
                        'best_params': best_params,
                        'set': 'validation',
                    }

                    metrics = eval_report(y_train, y_pred, y_pred_prob, model_save_path, run_info)
                    results.append(metrics)
                    models[model_name] = best_pipeline

                # Test best model
                test_save_path = f'{analysis_path}/pred_{label_0_name}_vs_{label_1_name}/best_model_test_set'
                if os.path.exists(test_save_path) or not models:
                    continue

                # Find best model
                results_df = pd.DataFrame(results)
                best_models_df = results_df.loc[results_df.groupby('labels')['balanced_accuracy_adjusted_chance'].idxmax()].reset_index()
                if len(best_models_df) > 1:
                    raise ValueError("Too many matches for best models")

                model_name = best_models_df['model'][0]
                best_pipeline = models.get(model_name, None)
                if best_pipeline is None:
                    continue

                # Evaluate on test set
                for X_test, y_test, names in BinaryLabelLoader.process_data_stream(df_test, [(label_0, label_1)]):
                    X_test, y_test = np.array(X_test), np.array(y_test)

                    y_pred = best_pipeline.predict(X_test)
                    y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

                    run_info = {
                        'data_type': data_type,
                        'time_frame': time_frame,
                        'label_0': label_0_name,
                        'label_1': label_1_name,
                        'inputs': input_variable,
                        'model_name': model_name,
                        'best_params': best_params,
                        'set': 'test',
                    }

                    eval_report(y_test, y_pred, y_pred_prob, test_save_path, run_info)

def generate_val_summary(master_data_path, master_encoded_data_path, save_path_reference,
                    time_frame_list, data_type_list, y_col, labels_dict, paired_labels,
                    model_storage, version_num):
    """Generate summary of all training results."""
    
    results = []

    for time_frame in time_frame_list:
        for data_type in data_type_list:
            analysis_path = save_path_reference.replace('MODEL', 'classification_1class_meds').replace('INDEPENDENT_VAR', f'v{version_num}/{data_type}/{time_frame}')
            # Load, select, and visualize data
            _, df_data, continuous = load_data(master_data_path, master_encoded_data_path, analysis_path, data_type, time_frame)
            # Prepare data for training
            df_dev, _ = split_data(df_data, y_col)
            categorical = [col for col in df_data.columns if col not in continuous]
            BinaryLabelLoader = MultiClassToBinaryConverter(y_col, categorical_cols=categorical)

            for _, _, names in BinaryLabelLoader.process_data_stream(df_dev, paired_labels):
                label1, label2, _ = names.values()
                label_name1 = 'rest' if label1 == 'rest' else labels_dict[label1]
                label_name2 = labels_dict[label2]

                for model_name in model_storage:
                    model_save_path = f'{analysis_path}/pred_{label_name1}_vs_{label_name2}/{model_name}'
                    metrics_path = os.path.join(model_save_path, 'metrics.json')
                    if os.path.exists(metrics_path):
                        metrics_dict = load_json(metrics_path)
                        results.append(metrics_dict)

    # Save summary
    results_df = pd.DataFrame(results)
    comparison_dir = save_path_reference.replace('MODEL', 'classification_1class_meds').replace('INDEPENDENT_VAR', f'v{version_num}')
    csv_path = f'{comparison_dir}/tree_models_validation_summary.csv'
    results_df.to_csv(csv_path, index=False)
    print(f'Saved validation summary: {csv_path}')

# --- TEST SET ---
def _map_label_name_to_num(labels_dict, label_name):
    """Map label name to corresponding number from labels dictionary."""
    return [num for num, name in labels_dict.items() if name == label_name][0]

def _select_best_models(df_dev, paired_labels, BinaryLabelLoader, labels_dict, 
                       model_storage, analysis_path):
    """
    Run hyperparameter tuning for all models and return best models summary.
    
    Args:
        df_dev: Development dataset
        paired_labels: List of label pairs for binary classification
        BinaryLabelLoader: Converter for multiclass to binary
        labels_dict: Dictionary mapping label numbers to names
        model_storage: Dictionary of models to test
        analysis_path: Path to save results
        
    Returns:
        DataFrame: Summary of best models for each label pair
    """
    all_results = []
     # Iterate over each label pair and predictor variable
    for X_train, y_train, names in BinaryLabelLoader.process_data_stream(df_dev, paired_labels):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        label_0, label_1, input_variable = names.values()
        label_0_name = 'rest' if label_0 == 'rest' else labels_dict[label_0]
        label_1_name = labels_dict[label_1]
        
        # Select the best model
        results = []
        for model_name, _ in model_storage.items():
            # Directory to save project analysis
            model_save_path = f'{analysis_path}/pred_{label_0_name}_vs_{label_1_name}/{model_name}'
            save_path = model_save_path + '/metrics.json'
            metrics_dict = load_json(save_path)
            results.append(metrics_dict)
        
        # Convert results to a DataFrame for better visualization
        print("Hyperparameter Tuning Results:")
        results_df = pd.DataFrame(results)
        #print(results_df)

        # Save tree results summary
        csv_path = f'{analysis_path}/pred_{label_0_name}_vs_{label_1_name}/tree_models_summary.csv'
        print('run summary: ', csv_path)
        results_df.to_csv(csv_path)

        # Select rows with the highest "metric" for each Output value
        best_models_df = results_df.loc[results_df.groupby('labels')['balanced_accuracy_adjusted_chance'].idxmax()].sort_values('labels').reset_index()
        
        all_results.append(best_models_df)
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def _evaluate_best_model_on_test_set(row, df_dev, df_test, BinaryLabelLoader, labels_dict,
                                   model_storage, analysis_path, balanced_accuracy_scorer,
                                   data_type, time_frame):
    """
    Train best model on dev set and evaluate on test set with SHAP analysis.
    
    Args:
        row: Row from best_models_df containing model info
        df_dev: Development dataset for training
        df_test: Test dataset for evaluation
        BinaryLabelLoader: Converter for multiclass to binary
        labels_dict: Dictionary mapping label numbers to names
        model_storage: Dictionary of available models
        analysis_path: Base path for saving results
        balanced_accuracy_scorer: Scoring function
        
    Returns:
        dict: Evaluation metrics
    """
    
    model_name = row['model']
    hyperparams = {key: [value] for key, value in row['best_parameters'].items()}
    label_0_name = row['label_0']
    label_1_name = row['label_1']
    model_info = model_storage[model_name]

    # Map label names back to numbers
    label_0 = 'rest' if label_0_name == 'rest' else _map_label_name_to_num(labels_dict, label_0_name)
    label_1 = _map_label_name_to_num(labels_dict, label_1_name)
    
    # Directory to save project analysis
    model_save_path = f'{analysis_path}/pred_{label_0_name}_vs_{label_1_name}/best_model_test_set'
    
    # # Skip if already processed
    # if os.path.exists(f'{model_save_path}/metrics.json'):
    #     metrics_dict = load_json(f'{model_save_path}/metrics.json')
    # If only want SHAP figures and not results
    #     if metrics_dict['model'] == model_name:
    #        return
    # if os.path.exists(f'{model_save_path}/SHAP_test/SHAP_decision_plot.png'):
    #     return
    
    # Train model on development set (only one iteration because we are only extracted one label pairing)
    for X_train, y_train, names in BinaryLabelLoader.process_data_stream(df_dev, [(label_0, label_1)]):
        # SHAP Variables
        X_SHAP_train = X_train.copy(deep=True)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Pipeline with feature scaling and model
        pipeline = ImbPipeline([
            ('scaler', MinMaxScaler()),  # Standardization
            ('oversample', SMOTE(random_state=42)),  # Oversample the minority class
            ('model', model_info['model']) # Model
        ])
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=hyperparams,
            scoring=balanced_accuracy_scorer,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"Starting Grid Search for {data_type} {time_frame} {model_name} {label_0_name}_vs_{label_1_name}...")
        grid_search.fit(X_train, y_train)
        
        # Best parameters and model
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters for {model_name} {label_0_name}_vs_{label_1_name}: {best_params}")

        # --- SHAP Explainer ---
        shap_analysis_pipeline(model = best_pipeline['model'], 
                                X_SHAP = X_SHAP_train, 
                                y_SHAP = np.array(y_train, dtype=bool), 
                                model_savepath = f'{model_save_path}/SHAP_train_val')
    
    # Collect test metrics
    results = []
        
    # Evaluate on test set (only one iteration because we are only extracted one label pairing)
    for X_test, y_test, names in BinaryLabelLoader.process_data_stream(df_test, [(label_0, label_1)]):
        # SHAP Variables
        X_SHAP_test = X_test.copy(deep=True)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # --- Performance Metrics ---
        # Use cross_val_predict for evaluation
        y_pred = best_pipeline.predict(X_test)
        y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Prepare run info
        _, _, input_variable = names.values()
        run_info = {
            'data_type': data_type,  # Will be filled by caller
            'time_frame': time_frame,  # Will be filled by caller
            'label_0': label_0_name,
            'label_1': label_1_name,
            'inputs': input_variable,
            'model_name': model_name,
            'best_params': best_params,
            'set': 'test',
        }
        
        # Evaluate and save metrics
        metrics = eval_report(y_test, y_pred, y_pred_prob, model_save_path, run_info)
        results.append(metrics)

        # SHAP analysis on test set
        shap_analysis_pipeline(
            model=best_pipeline['model'],
            X_SHAP=X_SHAP_test,
            y_SHAP=np.array(y_test, dtype=bool),
            model_savepath=f'{model_save_path}/SHAP_test'
        )
        
    return results

def generate_test_summary(master_data_path, master_encoded_data_path, save_path_reference,
                    time_frame_list, data_type_list, y_col, labels_dict, paired_labels,
                    model_storage, version_num):
    """
    Run the complete machine learning pipeline for all time frames and data types.
    
    Args:
        time_frame_list: List of time frames to process
        data_type_list: List of data types to process
        save_path_reference: Base save path template
        master_data_path: Path to master data
        master_encoded_data_path: Path to encoded data
        y_col: Target column name
        paired_labels: List of label pairs for binary classification
        labels_dict: Dictionary mapping label numbers to names
        model_storage: Dictionary of models to test
    """
    all_test_results = []
    
    for time_frame in time_frame_list:
        for data_type in data_type_list:
            print(f"\nProcessing {data_type} - {time_frame}")
            
            # Setup paths
            analysis_path = save_path_reference.replace('MODEL','classification_1class_meds').replace('INDEPENDENT_VAR',f'v{version_num}/{data_type}/{time_frame}') 
            os.makedirs(analysis_path, exist_ok=True)
            
            # Load, select, and visualize data
            _, df_data, continuous =  load_data(master_data_path, master_encoded_data_path, analysis_path, data_type, time_frame)

            # Prepare data for training
            df_dev, df_test = split_data(df_data, y_col)

            categorical = [col for col in df_data.columns if col not in continuous]
            BinaryLabelLoader = MultiClassToBinaryConverter(y_col, categorical_cols=categorical)
            
            os.makedirs(analysis_path, exist_ok=True)
            # Create a scorer object using balanced_accuracy_score
            balanced_accuracy_scorer = make_scorer(balanced_accuracy_score, adjusted=True)

            # Run model selection (loads existing results)
            print("Loading model selection results...")
            best_models_df = _select_best_models(
                df_dev, paired_labels, BinaryLabelLoader, labels_dict,
                model_storage, analysis_path
            )
            
            # Evaluate best models on test set
            print("Evaluating best models on test set...")
            for index, row in best_models_df.iterrows():
                results = _evaluate_best_model_on_test_set(
                    row, df_dev, df_test, BinaryLabelLoader, labels_dict,
                    model_storage, analysis_path, balanced_accuracy_scorer,
                    data_type, time_frame
                )
                
                all_test_results.append(results)
    
    return all_test_results

def aggregate_test_summary(master_data_path, master_encoded_data_path, save_path_reference,
                    time_frame_list, data_type_list, y_col, labels_dict, paired_labels,
                    version_num):
    """
    Run the complete machine learning pipeline for all time frames and data types.
    
    Args:
        time_frame_list: List of time frames to process
        data_type_list: List of data types to process
        save_path_reference: Base save path template
        master_data_path: Path to master data
        master_encoded_data_path: Path to encoded data
        y_col: Target column name
        paired_labels: List of label pairs for binary classification
        labels_dict: Dictionary mapping label numbers to names
    """
    # --- MODEL SUMMARY (TEST SET) ---
    all_test_results = []

    for time_frame in time_frame_list:
        for data_type in data_type_list:
            analysis_path = save_path_reference.replace('MODEL','classification_1class_meds').replace('INDEPENDENT_VAR',f'v{version_num}/{data_type}/{time_frame}') 

            # Load, select, and visualize data
            _, df_data, continuous =  load_data(master_data_path, master_encoded_data_path, analysis_path, data_type, time_frame)
            
            # Prepare data for training
            df_dev, _ = split_data(df_data, y_col)

            categorical = [col for col in df_data.columns if col not in continuous]
            BinaryLabelLoader = MultiClassToBinaryConverter(y_col, categorical_cols=categorical)
            
            # Iterate over each label pair and predictor variable
            for _, _, names in BinaryLabelLoader.process_data_stream(df_dev, paired_labels):
                label1, label2, _ = names.values()
                label_name1 = 'rest' if label1 == 'rest' else labels_dict[label1]
                label_name2 = labels_dict[label2] 

                # Directory to save project analysis
                model_save_path = f'{analysis_path}/pred_{label_name1}_vs_{label_name2}/best_model_test_set'
                #print(model_save_path)

                save_path = model_save_path + '/metrics.json'
                metrics_dict = load_json(save_path)
                all_test_results.append(metrics_dict)
        
    # Convert results to a DataFrame for better visualization
    print("Test Set Results:")
    test_results_df = pd.DataFrame(all_test_results)
    print(len(test_results_df))
    #print(results_df)

    # Save tree results summary
    comparison_dir = save_path_reference.replace('MODEL','classification_1class_meds').replace('INDEPENDENT_VAR',f'v{version_num}')
    csv_path = f'{comparison_dir}/tree_models_test_summary_v{version_num}.csv'
    print('run summary: ', csv_path)
    test_results_df.to_csv(csv_path)
    
    return test_results_df

# --- MAIN ---
def main():
    """Main function to orchestrate the classification training pipeline."""
    
    # Setup
    PlumsFiles = load_configuration()
    
    # Define paths
    master_data_path = PlumsFiles.get_datapath('model_output_dir').replace('MODEL', 'classification_1class_meds').replace('INDEPENDENT_VAR', 'master_data_for_analysis.csv') 
    master_encoded_data_path = PlumsFiles.get_datapath('model_output_dir').replace('MODEL', 'classification_1class_meds').replace('INDEPENDENT_VAR', 'master_numerical_data_for_analysis.csv') 
    save_path_reference = PlumsFiles.get_datapath('model_output_dir')
    
    # Define parameters
    # time_frame_list = ['2012_to_2024', '2012_to_2018', '2019_to_2024', '2012_to_2016', '2017_to_2019', '2020_to_2024']
    time_frame_list = ['2012_to_2024']
    data_type_list = ['tabular_text', 'tabular', 'text', 'diagnoses_tabular', 'diagnoses_text', 'psychosocial_tabular']
    version_num = 3

    y_col = 'interventiontype'
    labels_dict = {0: 'none', 1: 'nsaids', 2: 'opioids'}
    paired_labels = [(0, 1), (0, 2), (1, 2), ('rest', 0), ('rest', 1), ('rest', 2)]
    n_splits = 5
    
    # Get model configurations
    model_storage = get_model_storage()
    
    # Training Phase
    print("Starting model training phase...")
    train_models(
        master_data_path=master_data_path,
        master_encoded_data_path=master_encoded_data_path,
        save_path_reference=save_path_reference,
        time_frame_list=time_frame_list,
        data_type_list=data_type_list,
        y_col=y_col,
        labels_dict=labels_dict,
        paired_labels=paired_labels,
        model_storage=model_storage,
        n_splits=n_splits,
        version_num=version_num
    )
    
    # Validation Summary Phase
    print("Starting val summary generation phase...")
    generate_val_summary(
        master_data_path=master_data_path,
        master_encoded_data_path=master_encoded_data_path,
        save_path_reference=save_path_reference,
        time_frame_list=time_frame_list,
        data_type_list=data_type_list,
        y_col=y_col,
        labels_dict=labels_dict,
        paired_labels=paired_labels,
        model_storage=model_storage,
        version_num=version_num
    )
    
    # Test Set Summary Phase
    print("Starting test set summary generation phase...")
    test_results_df = generate_test_summary(
        master_data_path=master_data_path,
        master_encoded_data_path=master_encoded_data_path,
        save_path_reference=save_path_reference,
        time_frame_list=time_frame_list,
        data_type_list=data_type_list,
        y_col=y_col,
        labels_dict=labels_dict,
        paired_labels=paired_labels,
        model_storage=model_storage,
        version_num=version_num
    )
    test_results_df = aggregate_test_summary(
        master_data_path=master_data_path,
        master_encoded_data_path=master_encoded_data_path,
        save_path_reference=save_path_reference,
        time_frame_list=time_frame_list,
        data_type_list=data_type_list,
        y_col=y_col,
        labels_dict=labels_dict,
        paired_labels=paired_labels,
        version_num=version_num
    )
    
    # Feature Ranking Phase
    print("Starting SHAP feature ranking analysis...")
    aggregate_shap_feature_ranking(
        df=test_results_df,
        save_path_reference=save_path_reference,
        labels_dict=labels_dict,
        paired_labels=paired_labels,
        version_num=version_num
    )
    
    print("Classification training pipeline completed successfully!")


if __name__ == "__main__":
    main()