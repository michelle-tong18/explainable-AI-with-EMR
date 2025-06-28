# --- TRAINING UTILS ---
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score #accuracy_score, f1_score, recall_score, 
from sklearn.metrics import roc_curve, roc_auc_score, auc

import random 
import json

# --- Eval Helpers --- 
def plot_confusion_matrix(cm, labels, save_path=''):
    """Plot confusion matrix and normalized confusion matrix."""
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    
    cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
    cmp.plot(ax=ax[0])
    ax[0].set_title("Confusion Matrix")
    
    cm_norm = cm / np.sum(cm, axis=1, keepdims=True)
    cmp = ConfusionMatrixDisplay(cm_norm, display_labels=labels)
    cmp.plot(ax=ax[1])
    cmp.im_.set_clim(0, 1)
    ax[1].set_title("Normalized Confusion Matrix")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_roc_curve(fpr, tpr, label, title, save_path=''):
    """Plot ROC curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=label, color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def bootstrap_metrics(y_true, y_pred, y_pred_prob, n_bootstraps=1000, alpha=0.95, n_samples=100):
    """
    Calculates the confidence interval for AUC using bootstrap resampling.
    
    Parameters:
        y_true: array-like, shape (n_samples,)
            True binary labels.
        y_pred: array-like, shape (n_samples,)
            Predicted binary labels.
        y_pred_prob: array-like, shape (n_samples,)
            Target scores, typically probabilities or confidence values.
        n_bootstraps: int, default=1000
            Number of bootstrap samples.
        alpha: float, default=0.95
            Confidence level.
    
    Returns:
        mean_auc: float
            Mean AUC over all bootstrap samples.
        lower_ci: float
            Lower bound of the confidence interval.
        upper_ci: float
            Upper bound of the confidence interval.
    """
    def compute_confidence_interval(data, alpha):
        """Computes the mean and confidence interval bounds."""
        lower_percentile = ((1.0 - alpha) / 2.0) * 100
        upper_percentile = 100.0 - lower_percentile
        return {
            "mean": np.mean(data),
            "ci_lower": np.percentile(data, lower_percentile),
            "ci_upper": np.percentile(data, upper_percentile),
        }

    # Ensure a minimum number of samples
    if n_samples < 10:
        print('Warning - n_samples are less than 10, adjusting to len(y)')
        n_samples = len(y_true)

    # Initialize variables
    bootstrapped_list = {
        "precision": [], "recall": [], "specificity": [], "f1_score": [], "balanced_accuracy": [], "auc_": [], "tprs": []
    }

    mean_fpr = np.linspace(0, 1, n_samples)  # Fixed FPR points for interpolation
    random.seed(42)

    for _ in range(n_bootstraps):
        # Generate a bootstrap sample
        # Return a k sized list of elements chosen from the population with replacement. 
        indices = random.choices(range(n_samples), k=n_samples)
        # Skip bootstrap sample if it doesn't have at least one positive and one negative class
        if len(np.unique(y_true)) >= 2:
            while len(np.unique(y_true[indices])) < 2:
                indices = random.choices(range(n_samples), k=n_samples)
        
        #Updated Feb 2025
        metrics_report = classification_report(y_true[indices], y_pred[indices], output_dict=True)
        bootstrapped_list["precision"].append(metrics_report["1"]["precision"]) # how accurate are positive predictions
        bootstrapped_list["recall"].append(metrics_report["1"]["recall"]) # coverage of actual positive samples
        bootstrapped_list["specificity"].append(metrics_report["0"]["recall"]) # coverage of actual negative samples
        bootstrapped_list["f1_score"].append(metrics_report["1"]["f1-score"]) # metric of precision and recall for imbalanced classes
        bootstrapped_list["balanced_accuracy"].append(
            balanced_accuracy_score(y_true[indices], y_pred[indices])
        )

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true[indices], y_pred_prob[indices])
        bootstrapped_list["auc_"].append(auc(fpr, tpr))
        
        # Interpolate TPR at fixed FPR points
        bootstrapped_list["tprs"].append(np.interp(mean_fpr, fpr, tpr))
        bootstrapped_list["tprs"][-1][0] = 0.0  # Ensure TPR starts at 0

    # Compute confidence intervals
    bootstrap_report = {metric: compute_confidence_interval(values, alpha)
               for metric, values in bootstrapped_list.items() if metric != "tprs"}
    
    # Calculate tpr confidence interval
    bootstrap_report["tpr"] = {
        "mean": np.mean(bootstrapped_list["tprs"], axis=0),
        "ci_lower": np.percentile(bootstrapped_list["tprs"], (1 - alpha) / 2 * 100, axis=0),
        "ci_upper": np.percentile(bootstrapped_list["tprs"], (1 + alpha) / 2 * 100, axis=0),
    }
    bootstrap_report["tpr"]["mean"][-1] = 1.0  # Ensure TPR ends at 1
    bootstrap_report["fpr"] = {"mean": mean_fpr}

    return bootstrap_report

def extract_metrics(metrics_report, metric_keys):
    """ Extracts specified keys for both class labels and aggregate metrics. """
    metric_info = {}
    for metric_name in metric_keys:
        metric_info[f"{metric_name}_0"] = metrics_report["0"][metric_name]
        metric_info[f"{metric_name}_1"] = metrics_report["1"][metric_name]
        metric_info[f"{metric_name}_macro"] = metrics_report["macro avg"][metric_name] #macro avg of recall = balance acc
        metric_info[f"{metric_name}_weighted"] = metrics_report["weighted avg"][metric_name]
    return metric_info

def extract_bootstrap_metrics(bootstrap_report, metric_keys):
    """ Extracts mean and confidence intervals for the specified metrics. """
    metric_info = {}
    for metric_name in metric_keys:
        metric_info[f"{metric_name}_mean"] = bootstrap_report[metric_name]["mean"]
        metric_info[f"{metric_name}_ci_low"] = bootstrap_report[metric_name]["ci_lower"]
        metric_info[f"{metric_name}_ci_high"] = bootstrap_report[metric_name]["ci_upper"]
    return metric_info

def plot_feature_importance(model, X_labels, model_savepath=''):
    """Extract model feature importance, save as a df, and plot bar graph."""
    # TODO: Not implemented in Jan code iteration
    # Define the model
    #model = best_pipeline['model']
    # Create a pd.Series of features importances
    importances = pd.Series(data=model.feature_importances_, index=X_labels)

    # Sort importances
    importances_sorted = importances.sort_values(ascending=False)
    feature_df = pd.DataFrame(importances_sorted)
    if model_savepath:
        save_path = f'{model_savepath}/feature_importance.csv'
        feature_df.to_csv(save_path)

    # Draw a horizontal barplot of importances_sorted
    fig, ax = plt.subplots(figsize=(10, 5)) 
    importances_sorted.plot(kind='barh', color='lightgreen')
    plt.title('Feature Importance')
    if model_savepath:
        save_path = f'{model_savepath}/feature_importance.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    return feature_df

def save_json(data:dict, save_path:str):
    """
    Saves dictionary as a JSON file.
    """

    # Create a JSON Encoder class
    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    data_dict = data.copy()
    with open(save_path, 'w') as json_file:
        json.dump(data_dict, json_file, cls= json_serialize, indent=4)  # Use indent for readability
    return

def load_json(file_path:str) -> dict:
    """
    Loads a JSON file and returns its contents as a dictionary.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)  # Load JSON content
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
        return None


#--- Main Functions ---

def eval_report(y_true, y_pred, y_pred_prob, model_save_path, run_info):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    save_path = model_save_path + f'/confusion_matrix.png'
    plot_confusion_matrix(cm, labels=[run_info['label_0'], run_info['label_1']], save_path=save_path)
    
    # Performance Metrics: precision, recall (aka sensitivity), f1-score, support, accuracy
    metrics_report = classification_report(y_true, y_pred, output_dict=True)
    balanced_acc = balanced_accuracy_score(y_true, y_pred, adjusted=False)
    balanced_acc_chance = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    auc_val = roc_auc_score(y_true, y_pred_prob)
    #print(f"Performance metrics for {run_info['model_name']}:")
    #print(classification_report(y_true, y_pred))
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    save_path = model_save_path + '/roc_curve.png'
    plot_roc_curve(fpr, tpr, 
                label=f"{run_info['model_name']} (AUC = {auc_val:.2f})", 
                title=f"ROC Curve for {run_info['label_0']}_vs_{run_info['label_1']}", 
                save_path=save_path)
    
    # ROC curve with CIs
    bootstrap_report = bootstrap_metrics(y_true, y_pred, y_pred_prob, n_samples=int(len(y_true)))
    #mean_auc, lower_ci, upper_ci, mean_tpr, lower_tpr, upper_tpr, mean_fpr

    # Store results
    metrics_summary_dict = {
        # General Run Info
        'datatype': run_info['data_type'],
        'timeframe': run_info['time_frame'],
        'model': run_info['model_name'],
        'labels': f"{run_info['label_0']}_vs_{run_info['label_1']}",
        'label_0': run_info['label_0'],
        'label_1': run_info['label_1'],
        'input': run_info['inputs'],
        'best_parameters': run_info['best_params'],

        # Support Counts
        'support_0': metrics_report['0']['support'],
        'support_1': metrics_report['1']['support'],
        
        'confusion_matrix': cm,

        # Overall Metrics (label 0, 1, macro avg, weighted avg)
        **extract_metrics(metrics_report, ["precision", "recall", "f1-score"]),

        'accuracy': metrics_report['accuracy'],
        'balanced_accuracy': balanced_acc,
        'balanced_accuracy_adjusted_chance': balanced_acc_chance,
        
        'auc': auc_val,
        'fpr': fpr,
        'tpr': tpr,

        # Extracted Bootstrap Metrics
        **extract_bootstrap_metrics(bootstrap_report, metric_keys=["precision", "recall", "specificity", "f1_score", "balanced_accuracy"]),
        **extract_bootstrap_metrics(bootstrap_report, metric_keys=["auc_", "tpr"]),
        'fpr': bootstrap_report['fpr']
    }

    # Save Performance Metrics
    save_path = model_save_path + f'/metrics.json'
    save_json(metrics_summary_dict, save_path)

    return metrics_summary_dict

