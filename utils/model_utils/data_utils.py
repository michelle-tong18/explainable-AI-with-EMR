# --- DATA UTILS ---
import pandas as pd
import numpy as np

from tableone import TableOne

from sklearn.preprocessing import LabelEncoder
from itertools import combinations

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# --- Only for data preprocessing ---
def convert_categorical_to_binary(df, categorical_columns=None, keep_columns=None):
    """
    Converts specified categorical columns to a binary/one-hot encoded format.

    Params:
    - df (pd.DataFrame): The input dataframe.
    - categorical_columns (list[str], optional): List of columns to convert to binary. If None, all categorical columns are converted. Defaults to None.
    - keep_columns (str, optional): Column names to identify unique keys (e.g., patients) OR 'all'. Defaults to None.
    - keep_all_columns (bool): Whether to keep all original columns. Defaults to False.

    Returns:
    - df_binary (pd.DataFrame): Dataframe with binary features.
    - binary_columns (list[str]): List of binary feature column names.
    """
    if categorical_columns is None:
        # If no specific columns are provided, use all object-type columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Convert categorical columns to booleans and retain all columns
    if keep_columns == 'all':
        # Create a binary matrix of all specified features and drop duplicates
        df_binary = pd.get_dummies(df, columns=categorical_columns).drop_duplicates().reset_index(drop=True)
        binary_columns = [col for col in df_binary.columns if col not in df.columns]
    # Create a binary matrix for the specified categorical columns, keep specified columns, and drop duplicates
    elif keep_columns:
        df_binary = pd.concat([df[keep_columns], pd.get_dummies(df[categorical_columns])], axis=1).drop_duplicates().reset_index(drop=True)
        binary_columns = df_binary.columns.drop(keep_columns).tolist()
    # Create a binary matrix for the specified categorical columns and drop duplicates
    else:
        df_binary = pd.get_dummies(df[categorical_columns]).drop_duplicates().reset_index(drop=True)
        binary_columns = df_binary.columns.tolist()

    return df_binary, binary_columns

# --- Loading Helpers --- 
def get_categories(data_type):
    """
    Returns the category and continuous columns based on data type.
    """
    # Define categories
    # Define base categories
    chart_demographics = {'sex', 'obesity', 'ageatfirstimaging', 'yearatfirstimaging', 'interventiontype'}
    chart_psychosocial = {'preferredlanguage', 'raceethnicity', 'smokingstatus', 'socialsupport', 
                          'primaryinsurance', 'negativepsychstate'}
    chart_diagnoses = {'lbpduration', 'lbplaterality', 'sciatica', 'facetjointarthropathy', 'scoliosis',
                    'discpathology', 'spinalstenosis', 'sacroiliacjoint', 'radiculopathy', 'numbnesstingling', 
                    'osteoarthritisososteoarthritis', 'osteopeniaosteoporosis', 'fibromyalgiafibrosis'} # no bowelbladder, diabetes
    gpt_diagnoses = {'gpt_endplate', 'gpt_disc', 'gpt_scs', 'gpt_fj', 'gpt_lrs', 
                        'gpt_fs', 'gpt_sij', 'gpt_olisth', 'gpt_curv', 'gpt_frac'}

    # Define category mappings
    category_map = {
        'tabular': chart_demographics | chart_psychosocial | chart_diagnoses,
        'text': chart_demographics | chart_psychosocial | gpt_diagnoses,
        'tabular_text': chart_demographics | chart_psychosocial | chart_diagnoses | gpt_diagnoses,
        'diagnoses_tabular': chart_demographics | chart_demographics,
        'diagnoses_text': chart_demographics | gpt_diagnoses,
        'psychosocial_tabular': chart_demographics | chart_psychosocial  
    }
    categories = list(category_map.get(data_type, []))
    continuous_cols = ['ageatfirstimaging', 'yearatfirstimaging', 'gpt_disc', 'gpt_scs', 'gpt_fj', 'gpt_lrs', 'gpt_fs']
    continuous = [col for col in categories if col in continuous_cols]
    return categories, continuous



def get_timeframe(df_data, time_frame='all'):
    """
    Filters the dataframe based on the specified time frame.
    """
    if time_frame == '2012_to_2018':
        return df_data[df_data['yearatfirstimaging'] <= 2018]
    elif time_frame == '2019_to_2024':
        return df_data[df_data['yearatfirstimaging'] >= 2019]
    elif time_frame == '2012_to_2016':
        return df_data[df_data['yearatfirstimaging'] >= 2016]
    elif time_frame == '2017_to_2019':
        return df_data[(df_data['yearatfirstimaging'] >= 2017) & (df_data['yearatfirstimaging'] <= 2019)]
    elif time_frame == '2020_to_2024':
        return df_data[df_data['yearatfirstimaging'] >= 2020]
    return df_data

def patient_profile_loader(df_master, data_type, time_frame, label_col = 'interventiontype'):
    """
    Load patient profiles and generate summary statistics.
    """
    # Select a subset of data source(s) or content(s)
    categories, continuous = get_categories(data_type)
    df_data = df_master.set_index('patientdurablekey').dropna()
    df_data = df_data[categories]

    # Filter specific GPT-related rows if necessary
    if any([x.startswith('gpt_') for x in categories]):
        df_data = df_data[df_data['gpt_disc'] != -1] 
        # NOTE (TODO): Current implementation assumes all or no gpt columns are selected

    # Select a subset of data timeframes
    df_data = get_timeframe(df_data, time_frame)

    df_summary = TableOne(df_data, 
                          groupby=label_col, 
                          categorical=[col for col in categories if col not in continuous],
                          continuous=[col for col in categories if col in continuous],
                          pval=True)
    
    return df_data, df_summary, continuous

# --- Processor and Iterator Class --- 
class MultiClassToBinaryConverter:
    def __init__(self, y_col, categorical_cols=[], X_type='all'):
        """
        Initialize the dataset for one-vs-one or one-vs-rest classification.

        Parameters:
        - data: numpy array or list of input features
        - labels: numpy array or list of labels
        - task: str, either 'one_vs_one' or 'one_vs_rest'
        - class_pair: tuple, a pair of classes for one-vs-one (OvO) task.
        - X_type: either one or all input variables
        """
        self.categorical_cols = categorical_cols
        self.label_col = y_col
        self.X_type = X_type
        
        # Initialize label encoders for each categorical column
        self.label_encoders = {col: LabelEncoder() for col in (categorical_cols + [y_col])}
        
    def encode_categorical_cols(self, df_input, split_name='train'):
        """
        Encode categorical columns using LabelEncoder.
        """
        df = df_input.copy()

        for col in self.label_encoders:
            if col in df.columns:
                if split_name == 'train':
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                elif split_name == 'test':
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        else:
            assert('split not specified')
        return df
    
    def preprocess_data(self, df_data, label1=None, label2=None, split_name=None, to_encode:bool=True):
        """
        Preprocess the data by filtering for specified labels and encoding.
        """
        df_analysis = df_data.copy(deep=True)

        # Handle label filtering for OVO or OVR
        if (label1 is not None) and (label2 is not None):
            # one_vs_rest
            if label1 == 'rest':
                df_analysis[self.label_col] = np.where(df_analysis[self.label_col] == label2, 1, 0)
            # one_vs_one
            else:
                df_analysis = df_analysis[df_analysis[self.label_col].isin([label1, label2])]
                df_analysis[self.label_col] = np.where(df_analysis[self.label_col] == label2, 1, 0)
        else:
            print('categorical encoding only')

        # Convert categorical to numerical data
        if to_encode == True:
            df_analysis = self.encode_categorical_cols(df_analysis, split_name)

        return df_analysis
    
    def get_label_combinations(self, df):
        """
        Return all combinations of unique labels for the target column.
        """
        unique_labels = df[self.label_col].unique()
        return list(combinations(unique_labels, 2)) + [(label, 'rest') for label in unique_labels]
    
    def process_data_stream(self, df_data, paired_labels=None):
        """
        Generator yielding data for each pair of labels.
        """
        if paired_labels == None:
            paired_labels = self.get_label_combinations(df_data)

        for label1, label2 in paired_labels:
            # --- DATA CLEANING ---
            # Prepare the label column and covert categorical to numerical
            df_Xy = self.preprocess_data(df_data, label1=label1, label2=label2)
            X_cols = [col for col in df_data.columns if col not in self.label_col]
            
            # For OVO and OVR analysis with one input variable at a time
            if self.X_type == 'one':
                for input_var in X_cols:
                    yield df_Xy[input_var], df_Xy[self.label_col], {'label1': label1, 'label2': label2, 'input_var': input_var}
            # For OVO and OVR analysis with all inputs variables at the same time
            elif self.X_type == 'all':
                yield df_Xy[X_cols], df_Xy[self.label_col], {'label1': label1, 'label2': label2, 'input_var': X_cols}
    
    def __iter__(self):
        return iter(self.process_data_stream())

def split_data(df_data, y_col, test_size=0.15):
    X_cols = list(df_data.drop(y_col,axis=1).columns)
    # Split into development (train & val for crossval) and testing 
    X_dev, X_test, y_dev, y_test = train_test_split(np.array(df_data.drop(y_col,axis=1)), 
                                                    np.array(df_data[y_col]), 
                                                random_state=0,  
                                                test_size=test_size,  
                                                shuffle=True) 

    # Create a dataframe for each split
    df_dev = pd.DataFrame(data= np.concatenate((y_dev[:,np.newaxis],X_dev),axis=1),
                        columns= [y_col]+X_cols)
    df_test = pd.DataFrame(data= np.concatenate((y_test[:,np.newaxis],X_test),axis=1),
                        columns= [y_col]+X_cols)
    
    return df_dev, df_test

# --- Main Loading ---
def load_data(master_data_path, master_encoded_data_path, analysis_path, data_type, time_frame):
    """Load and visualize the dataset, generating histograms."""
    df_master = pd.read_csv(master_data_path)
    df_categorical, df_summary, continuous = patient_profile_loader(df_master, data_type, time_frame)
    df_master = pd.read_csv(master_encoded_data_path)
    df_data, df_summary2, _ = patient_profile_loader(df_master, data_type, time_frame)

    # --- DEBUG ---
    # print(df_summary)
    # print(df_summary2)
    # print(df_data.head())
    # print(df_data.info())
    # --- END ---
    
    save_path = f'{analysis_path}/tableone_stats.csv'
    df_summary.to_csv(save_path, index=True)
    save_path = f'{analysis_path}/tableone_stats_encoded.csv'
    df_summary2.to_csv(save_path, index=True)

    return df_categorical, df_data, continuous

def visualize_data(df_categorical, analysis_path=None, label_col='interventiontype'):
    """
    Plot histograms to display data distributions
    """
    df_categorical.hist(figsize=[23, 15])
    plt.tight_layout()
    plt.suptitle('data summary')
    if analysis_path: 
        save_path = f'{analysis_path}/distributions_for_data.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    for intervention in set(df_categorical[label_col]):
        df_categorical[df_categorical[label_col] == intervention].hist(figsize=[23, 15])
        plt.tight_layout()
        plt.suptitle(f'{intervention} summary')
        if analysis_path: 
            save_path = f'{analysis_path}/distributions_for_{intervention}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
    return
