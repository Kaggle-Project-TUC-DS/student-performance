import numpy as np
import pandas as pd
from typing import Tuple
#in this file we will collect the functions/utilities to load the preprocessed data for training purposes
#to do- merge the different preprocessing files and create 1 preprocessing function
#Stephan and Johannes

#load the preprocessed data
#work in progress

#define the dictionary to load in the data efficiently

import pandas as pd
import numpy as np

def load_data(file_path: str , dtypes: dict = None, n_rows: int = None):
    # If dtypes is not specified, set default data types for each column
    if dtypes is None:
        dtypes = {
            'level': np.uint8,  
            'session_id': np.int64,
            'level_group': 'category',
            'event_name': np.uint8,
            'name': np.uint8,
            'fqid': np.uint8,
            'room_fqid': np.uint8,
            'text_fqid': np.uint8,
            'fullscreen': np.uint8,
            'hq': np.uint8,
            'music': np.uint8,
            'hover_duration_mean': np.float32,
            'difference_clicks_mean': np.float32,
            "distance_clicks_mean": np.float32,
            "screen_distance_clicks_mean": np.float32,            
            'elapsed_time_std': np.float32,
            'page_std': np.float32,
            'room_coor_x_std': np.float32,
            'room_coor_y_std': np.float32,
            'screen_coor_x_std': np.float32,
            'screen_coor_y_std': np.float32,
            'hover_duration_std': np.float32,
            'difference_clicks_std': np.float32,
            "distance_clicks_std": np.float32,
            "screen_distance_clicks_std": np.float32,
            'index_sum_of_actions': np.int32,
            'difference_clicks_max': np.float32,
            'elapsed_time_max': np.float32,
            'clicks_per_second': np.float32,
            "sum_distance_clicks_max": np.float32,
        }
        
    # Read in the CSV file
    if n_rows is None:
        df = pd.read_csv(file_path, dtype=dtypes, index_col = 0)
    else:
        df = pd.read_csv(file_path, dtype=dtypes, nrows=n_rows, index_col= 0)
    
    # Set data types for columns with "_i" index in their name
    row, cols = df.shape
    if cols > 50:
        for column in df.columns:
            base_name = column.rsplit('_', 1)[0]  # get the base name by splitting on the last "_" character
            if base_name in dtypes:
                column_number = column.rsplit('_', 1)[1]  # get the number from the index by splitting on the last "_" character
                new_column_name = f"{base_name}_{column_number}"  # construct the new column name
                column_dtype = dtypes[base_name]
                df[new_column_name] = df[column].astype(column_dtype)  # set the same data type for all columns with the same base name

    return df


def load_labels(file_path: str = 'data/processed/labels.csv', dtypes: dict = None, n_rows: int = None) -> pd.DataFrame:
    if dtypes is None:
        dtypes= {
            'session': np.int64,
            'correct': np.uint8, 
            'q':np.uint8
            }
    # Read in the CSV file
    if n_rows is None:
        labels = pd.read_csv(file_path, dtype=dtypes, index_col = 0)
    else:
        labels = pd.read_csv(file_path, dtype=dtypes, nrows=n_rows, index_col= 0)

    return labels


def load_level_group_X_y(
        level_group: str,
        data_version: str = 'flattened', 
        dtypes_data: dict = None, 
        dtypes_labels: dict = None, 
        n_rows: int = None) -> Tuple[(pd.DataFrame, pd.DataFrame)]:
    
    if data_version == 'flattened':
        if level_group == '0_4':
            df = load_data(file_path='data/processed/df_0_4_flattened.csv', dtypes=dtypes_data, n_rows=n_rows)
            labels = load_labels(file_path='data/processed/labels_q1-3.csv', dtypes=dtypes_labels, n_rows=n_rows)
        elif level_group == '5_12':
            df = load_data(file_path='data/processed/df_5_12_flattened.csv', dtypes=dtypes_data, n_rows=n_rows)
            labels = load_labels(file_path='data/processed/labels_q4-13.csv', dtypes=dtypes_labels, n_rows=n_rows)
        elif level_group == '13_22':
            df = load_data(file_path='data/processed/df_13_22_flattened.csv', dtypes=dtypes_data, n_rows=n_rows)
            labels = load_labels(file_path='data/processed/labels_q14-18.csv', dtypes=dtypes_labels, n_rows=n_rows)

    return (df, labels)


def load_all_X_y(
        data: str = 'flattened', 
        file_path_labels: str = 'data/processed/labels.csv', 
        dtypes_data: dict = None, 
        dtypes_labels: dict = None, 
        n_rows: int = None) -> Tuple[(dict, pd.DataFrame)]:
    """Load all data and labels. Return a dictionary of dataframes and a dataframe of labels."""
    if data == 'flattened':
        df_0_4 = load_data(file_path='data/processed/df_0_4_flattened.csv', dtypes=dtypes_data, n_rows=n_rows)
        df_5_12 = load_data(file_path='data/processed/df_5_12_flattened.csv', dtypes=dtypes_data, n_rows=n_rows)
        df_13_22 = load_data(file_path='data/processed/df_13_22_flattened.csv', dtypes=dtypes_data, n_rows=n_rows)
        dict_dfs = dict({'0_4': df_0_4, '5_12': df_5_12, '13_22': df_13_22})

    labels = load_labels(file_path=file_path_labels, dtypes=dtypes_labels, n_rows=n_rows)
    return (dict_dfs, labels)


def load_all_as_dict(
        data: str = 'flattened', 
        file_path_labels: str = 'data/processed/labels.csv', 
        dtypes_data: dict = None, 
        dtypes_labels: dict = None, 
        n_rows: int = None) -> dict:
    """Load all data and labels. Return a dictionary of dataframes and a dataframe of labels."""
    dict_dfs, labels = load_all_X_y(
        data=data, 
        file_path_labels=file_path_labels, 
        dtypes_data=dtypes_data, 
        dtypes_labels=dtypes_labels, 
        n_rows=n_rows)
    dict_all = {}
    for q in labels['q'].unique():
        if q <= 3:
            dict_all[q] = {'X': dict_dfs['0_4'].iloc[:, 2:].values, 
                           'y': labels['correct'][labels['q'] == q].values}
        elif q <= 13:
            dict_all[q] = {'X': dict_dfs['5_12'].iloc[:, 2:].values, 
                           'y': labels['correct'][labels['q'] == q].values}
        elif q <= 18:
            dict_all[q] = {'X': dict_dfs['13_22'].iloc[:, 2:].values, 
                           'y': labels['correct'][labels['q'] == q].values}
    return dict_all
