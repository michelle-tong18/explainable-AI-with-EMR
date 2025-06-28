import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings

# --- FOR SHAP FEATURE IMPORTANCE EXTRACTION ---
def prepare_data(X_train, y_train, X_labels):
    """Prepare data for SHAP analysis."""
    X_SHAP = pd.DataFrame(X_train, columns=X_labels)
    y_SHAP = np.array(y_train, dtype=bool)
    return X_SHAP, y_SHAP

def create_explainer(model, X_SHAP):
    """Create a SHAP explainer."""
    # Note the input to the shap.force_plot changes depending on the model
    # random forests: explainer.expected_value[1], explanation[ii, :, 1], shap_values[1][ii, :]
    # XGBoost: explainer.expected_value, explanation[ii, :], shap_values[ii, :]
    
    explainer = shap.TreeExplainer(model) #model_output“raw”, “probability”, “log_loss”,
    explanation = explainer(X_SHAP)
    shap_values = explainer.shap_values(X_SHAP)
    
    if (len(np.shape(explanation))==3 and np.shape(explanation)[2] == 2):
        explanation = explanation[:,:,1]
    if (len(np.shape(shap_values))==3 and np.shape(shap_values)[2] == 2):
        shap_values = shap_values[:,:,1]
    
    return explainer, explanation, shap_values

def save_feature_importance(shap_values, X_SHAP, save_path=''):
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    # Create a DataFrame from the mean absolute SHAP values
    shap_df = pd.DataFrame({'Feature': X_SHAP.columns, 'Mean_SHAP': mean_abs_shap_values})
    # Sort the DataFrame by mean SHAP values in descending order
    shap_df = shap_df.sort_values(by='Mean_SHAP', ascending=False).reset_index(drop=True)
    # Save dataframe
    if save_path:
        shap_df.to_csv(save_path, index=True)
    return shap_df

class SHAP_visualizer:
    def __init__(self):
        pass

    def plot_force(explainer, shap_values, X_SHAP, save_path='', n=5):
        """Generate SHAP force plots."""
        if (np.shape(explainer.expected_value) == (2,)):
            expected_val = explainer.expected_value[1]
        else:
            expected_val = explainer.expected_value
        fig, axes = plt.subplots(n, 1, figsize=(30, 6)) # W x H
        for ii in range(n):
            plt.sca(axes[ii])
            shap.force_plot(
                expected_val,
                shap_values[ii, :],
                X_SHAP.iloc[ii, :],
                show=False,
                matplotlib=True
            )
            temp_fig = plt.gcf()
            axes[ii].imshow(temp_fig.canvas.buffer_rgba())
            axes[ii].axis('off')
            plt.close(temp_fig)
        fig.suptitle('SHAP Force Plot')

        fig.subplots_adjust(hspace=0.1)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        return

    def plot_waterfall(explanation, save_path='', n=4):
        """Generate SHAP waterfall plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for ii in range(n):
            plt.sca(axes[ii])
            shap.plots.waterfall(explanation[ii, :], show=False)
        fig.suptitle('SHAP Waterfall Plot')
        
        fig.set_size_inches(14,10)
        fig.subplots_adjust(wspace=1)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        return

    def plot_bar(shap_values, X_SHAP, explanation, save_path='', plot_type="bar"):
        """Generate SHAP summary plots."""
        # Set up the figure with 6 rows of subplots
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))  #12, 10 # 6 rows, 1 column
        # Flatten the 2D array of axes for easier indexing
        axes = axes.flatten()
        plt.sca(axes[0]) # Set the current subplot
        shap.summary_plot(shap_values[:, :], X_SHAP, plot_type=plot_type, show=False)
        plt.title('SHAP Feature Importance')
        plt.sca(axes[1]) # Set the current subplot
        shap.plots.bar(explanation[:, :], show=False)
        
        fig.set_size_inches(6,6)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        return

    def plot_violin(explanation, save_path='', plot_type=''):
        """Generate SHAP violin plots."""
        if plot_type in ['violin', 'layered_violin']:
            shap.plots.violin(explanation[:, :], plot_type=plot_type, show=False)
            plt.suptitle('SHAP Feature Importance')
        else:
            # Set up the figure with 6 rows of subplots
            fig, axes = plt.subplots(2, 1, figsize=(5, 8))  # (80,5) 6 rows, 1 column
            # Flatten the 2D array of axes for easier indexing
            axes = axes.flatten()
            plt.sca(axes[0]) # Set the current subplot
            shap.plots.violin(explanation[:, :], plot_type='violin', max_display=12, show=False)
            plt.sca(axes[1]) # Set the current subplot
            shap.plots.violin(explanation[:, :], plot_type='layered_violin', max_display=12, show=False)
            fig.suptitle('SHAP Feature Importance')
            fig.set_size_inches(5,8)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        return

    def plot_heatmap(explanation, order=None, save_path=''):
        """Generate SHAP heatmap plots."""
        # Set up the figure with 6 rows of subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 6 rows, 1 column
        # Flatten the 2D array of axes for easier indexing
        axes = axes.flatten()
        plt.sca(axes[0]) # Set the current subplot
        shap.plots.heatmap(explanation[:, :], show=False)
        plt.title('Heatmap')
        plt.sca(axes[1]) # Set the current subplot
        shap.plots.heatmap(explanation[:, :], instance_order=order, show=False)
        fig.set_size_inches(12,10)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        return

    def plot_dependence(shap_values, X_SHAP, save_path='', batch_size=12):
        """Generate SHAP dependence plots."""
        features = X_SHAP.columns
        for start_idx in range(0, len(features), batch_size):
            fig, axes = plt.subplots(3, 4, figsize=(16, 8))
            axes = axes.ravel()
            for ii, feature_idx in enumerate(range(start_idx, min(start_idx + batch_size, len(features)))):
                shap.dependence_plot(
                    features[feature_idx],
                    shap_values[:, :],
                    X_SHAP,
                    display_features=X_SHAP,
                    show=False,
                    ax=axes[ii]
                )
            for jj in range(ii + 1, len(axes)):
                axes[jj].axis('off')
            
            plt.suptitle(f'Dependence Plots {start_idx + 1}-{start_idx + batch_size}')
            plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
            
            if save_path:
                plt.savefig(f"{save_path}_batch_{start_idx}.png", bbox_inches='tight', dpi=300)
                plt.close()
            else:
                plt.show()

    def plot_decision(explainer, X_SHAP, save_path=''):
        """Generate SHAP decision plots."""
        shap_values = explainer.shap_values(X_SHAP)[1]
        shap_interaction_values = explainer.shap_interaction_values(X_SHAP)
        # Removed from nb to py
        # if isinstance(shap_interaction_values, list):
        #     shap_interaction_values = shap_interaction_values[1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # W X H

        axes = axes.ravel()

        # Set the current subplot
        plt.sca(axes[0])
        shap.decision_plot(explainer.expected_value, shap_values, X_SHAP, show=False, ignore_warnings=True)
        plt.title('Linear')
        plt.sca(axes[1])
        shap.decision_plot(explainer.expected_value, shap_values, X_SHAP, link="logit", show=False, ignore_warnings=True)
        plt.title('Logit')
        plt.sca(axes[2])
        shap.decision_plot(explainer.expected_value, shap_interaction_values, X_SHAP, show=False, ignore_warnings=True)
        plt.title('Interactions Linear')
        plt.sca(axes[3])
        shap.decision_plot(explainer.expected_value, shap_interaction_values, X_SHAP, link="logit", show=False, ignore_warnings=True)
        plt.title('Interactions Logit')
        plt.suptitle('Decision Plot')

        fig.set_size_inches(15,10)
        plt.tight_layout()

        #plt.subplots_adjust(hspace=0.7, wspace=1.1)  
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        return

# Main Execution
def shap_analysis_pipeline(model, X_SHAP, y_SHAP, model_savepath):
    """Complete pipeline for SHAP analysis."""
    os.makedirs(model_savepath, exist_ok=True)
    
    # Create an Expaliner Object (TreeExplainer or Explainer)
    explainer, explanation, shap_values = create_explainer(model, X_SHAP)
    
    # Subject feature importance
    SHAP_visualizer.plot_force(explainer, shap_values, X_SHAP, save_path=f"{model_savepath}/SHAP_force_plot_per_subject.png")
    SHAP_visualizer.plot_waterfall(explanation, save_path=f"{model_savepath}/SHAP_waterfall_plot_per_subject.png")

    # Mean feature importance
    save_feature_importance(shap_values, X_SHAP, save_path=f"{model_savepath}/SHAP_mean_importance.csv")
    SHAP_visualizer.plot_bar(shap_values, X_SHAP, explanation, save_path=f"{model_savepath}/SHAP_mean_importance_bar_chart_resized.png", plot_type="bar")
    SHAP_visualizer.plot_violin(explanation, save_path=f"{model_savepath}/SHAP_violin_plot_resized.png")
    # order by predictions
    SHAP_visualizer.plot_heatmap(explanation, order=np.argsort(y_SHAP), save_path=f"{model_savepath}/SHAP_heatmap.png")

    # SHAP_visualizer.plot_dependence(shap_values, X_SHAP, save_path=f"{model_savepath}/SHAP_dependence_plot")
    if (np.shape(explainer.expected_value) != (2,)):
        print(f"Explainer expected value: {explainer.expected_value:.2f}")
        SHAP_visualizer.plot_decision(explainer, X_SHAP.iloc[:], save_path=f"{model_savepath}/SHAP_decision_plot.png")

# --- FOR SHAP FEATURE RANKING ---
class SHAP_ranker:
    # --- SHAP FEATURE RANK AGGREGATION ---
    def __init__(self, save_path_reference, version_num):
        self.save_path_reference = save_path_reference
        self.version_num = version_num
        
    def rank_best_data_selection(self, df, metric='balanced_accuracy_mean'):

        # Find the indices of the rows with the highest balanced_accuracy for each label
        idx = df.groupby('labels')[metric].idxmax()
        # Use these indices to filter the DataFrame
        df = df.loc[idx]

        csv_list = []
        for index, row in df.iterrows():
            time_frame = row['timeframe']
            data_type = row['datatype']
            analysis_path = self.save_path_reference.replace('MODEL','classification_1class_meds').replace(
                    'INDEPENDENT_VAR',f'v{self.version_num}/{data_type}/{time_frame}') 
            
            label_name1 = row['label_0']
            label_name2 = row['label_1']
            # Directory to save project analysis
            model_save_path = f'{analysis_path}/pred_{label_name1}_vs_{label_name2}/best_model_test_set'

            csv_list.append(f"{model_save_path}/SHAP_test/SHAP_mean_importance.csv")
        return csv_list

    def rank_data_selection(self, time_frame_list, data_type_list, paired_labels, labels_dict):
        csv_list = []
        for time_frame in time_frame_list:
            for data_type in data_type_list:
                analysis_path = self.save_path_reference.replace('MODEL','classification_1class_meds').replace(
                    'INDEPENDENT_VAR',f'v{self.version_num}/{data_type}/{time_frame}') 

                # Iterate over each label pair and predictor variable
                for label1, label2 in paired_labels:
                    label_name1 = 'rest' if label1 == 'rest' else labels_dict[label1]
                    label_name2 = labels_dict[label2] 

                    # Directory to save project analysis
                    model_save_path = f'{analysis_path}/pred_{label_name1}_vs_{label_name2}/best_model_test_set'

                    csv_list.append(f"{model_save_path}/SHAP_test/SHAP_mean_importance.csv")
        return csv_list

    def _rank_prep(self, file_path, feature_col, metric_col):
        """Process individual CSV file and return normalized Borda scores"""
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Filter out rows with metric_col < 0.01
        df_filtered = df[df[metric_col] >= 0.01]
        
        if df_filtered.empty:
            return pd.Series(dtype=float)
        
        # Sort by metric_col descending
        df_sorted = df_filtered.sort_values(metric_col, ascending=False)
        
        # Assign Borda points (highest metric gets highest points)
        m = len(df_sorted)
        df_sorted['borda'] = range(m, 0, -1)
        
        # Normalize scores to sum to 1
        total_points = df_sorted['borda'].sum()
        df_sorted['normalized'] = df_sorted['borda'] / total_points
        
        return df_sorted.set_index(feature_col)['normalized']

    def rank_aggregation(self, mean_shap_path_list):
            # Process all files and collect scores
        all_scores = []
        all_features = set()

        for file in mean_shap_path_list:
            scores = self._rank_prep(file, feature_col='Feature', metric_col='Mean_SHAP')
            all_scores.append(scores)
            all_features.update(scores.index.tolist())

        # Create a DataFrame with all cities
        all_features = list(all_features)
        borda_df = pd.DataFrame(index=all_features)

        # Add each person's scores (0 for unranked cities)
        for i, scores in enumerate(all_scores):
            borda_df[f'model_{i+1}'] = scores.reindex(all_features).fillna(0)

        # Calculate total Borda scores
        borda_df['total_score'] = borda_df.sum(axis=1)
        borda_df = borda_df.sort_values(by='total_score' ,ascending=False)
        
        # Sumarize findings
        borda_df.index.name = 'feature'
        borda_df = borda_df.reset_index()
        borda_df['rank'] = np.arange(1,len(borda_df)+1)

        # # Get final ranking
        # final_ranking = borda_df['total_score'].sort_values(ascending=False)
        #print("Final Borda Count Ranking:")
        #print(final_ranking.to_string(float_format='%.3f'))
        
        return borda_df

# Main Execution
def aggregate_shap_feature_ranking(df, save_path_reference, labels_dict, paired_labels, version_num):

    myAggregator = SHAP_ranker(save_path_reference, version_num)
    # List of input CSV files (replace with your actual file paths)
    mean_shap_path_list = myAggregator.rank_best_data_selection(df=df[df['timeframe'] == "2012_to_2024"], metric='balanced_accuracy_mean')
    tab_text_ranked_df = myAggregator.rank_aggregation(mean_shap_path_list)
    print('best bala acc ranked feature importance')
    print(tab_text_ranked_df['feature'][0:10].to_string(float_format='%.3f'))

    for data_type in ['tabular_text', 'tabular', 'text']:
        mean_shap_path_list = myAggregator.rank_data_selection(time_frame_list=['2012_to_2024'], data_type_list=[data_type], paired_labels=paired_labels, labels_dict=labels_dict)
        data_ranked_df = myAggregator.rank_aggregation(mean_shap_path_list)
        print(f'{data_type} ranked feature importance')
        print(data_ranked_df['feature'][0:10].to_string(float_format='%.3f'))

    return
