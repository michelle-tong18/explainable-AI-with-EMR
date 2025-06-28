import duckdb
import numpy as np
import time

class Extractor(object):
    def __init__(self,num_results_flag:bool=True, display_results_flag:bool=True):
        #Set print counts flag
        self.num_results_flag = num_results_flag          #show the number of query results
        self.display_results_flag = display_results_flag  #display the query results
        
    def run_query(self,query_text,runtime_flag=True,df_type='pandas'):
        start_time = time.time()
        results_df = duckdb.sql(query_text).df()
        run_time = time.time() - start_time
        if runtime_flag == True:
            print(f"query runtime: {run_time//60:.0f}m {run_time%60:.2f}s")
        if self.num_results_flag==True:     
            print(f'total query results: ',len(results_df))
        return results_df

    def col_to_list(self, df, key_text:str, isdistinct:bool=True, n_show:str=5):
        key_list = list(df[key_text]) # alternatively can use .drop_duplicates()
        if isdistinct == True:
            key_list = list(set(key_list))
        if self.num_results_flag==True:     print(f'total unique {key_text}: ',len(set(key_list)))
        if self.display_results_flag==True: print(f'{key_text}: {key_list[0:n_show]}')
        return key_list

    def remove_invalid(self,col_list):
        new_list = [x for x in col_list if x is not None]
        new_list = [x for x in new_list if str(x) != 'nan']
        new_list = [x for x in new_list if x is not np.nan]
        return new_list
