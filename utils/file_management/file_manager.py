from datetime import datetime
import pandas as pd
import os

from datetime import datetime

class FileManager():
    def __init__(self, file_directory:dict):
        time_info = datetime.today().strftime('%Y%m%d')
        self.savepaths = file_directory
    
    def list_datapath_keys(self):
        return self.savepaths.keys()
    def get_datapath(self,key):
        return self.savepaths[key]
    
    def read_file(self, fpath):
        if fpath.endswith('.csv')==True:
            return pd.read_csv(fpath)
        elif fpath.endswith('.xlsx')==True:
            return pd.read_excel(fpath)
        #elif fpath.endswith('.parquet')==True:
            #Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
            #return pd.read_parquet(fpath)
        return 

    def save_df_to_csv(self,df_pd,fpath):
        assert fpath.endswith('.csv')==True, 'invalid filename extension, parameter must be your_text.csv'
        os.makedirs(os.path.dirname(fpath), exist_ok=True) 
        df_pd.to_csv(fpath, index=False)
        return True
    
    def save_df_to_parquet(self,df_pd,fpath):
        assert fpath.endswith('.parquet')==True, 'invalid filename extension, parameter must be your_text.parquet'
        os.makedirs(os.path.dirname(fpath), exist_ok=True) 
        df_pd.to_parquet(fpath, index=False)
        return True
    
#--- SAVING ---
def create_file(fpath):
    """Creates a new file or clears an existing file."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True) 
    with open(fpath, 'w') as file:
        file.write(f"File created {datetime.today().strftime('%Y-%m-%d')}.\n")

def add_line(fpath, line):
    """Adds a line to the file."""
    with open(fpath, 'a') as file:
        file.write(line + "\n")

# #AWS

#     def save_df_to_csv(self,df_pd,fpath):
#         assert fpath.endswith('.csv')==True, 'invalid filename extension, parameter must be your_text.csv'
#         wr.s3.to_csv(df_pd,fpath)
#         return

#     def save_df_to_parquet(self,df_pd,fpath):
#         assert fpath.endswith('.parquet')==True, 'invalid filename extension, parameter must be your_text.parquet'
#         wr.s3.to_parquet(df_pd,fpath)
#         return