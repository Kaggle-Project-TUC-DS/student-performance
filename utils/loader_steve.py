import numpy as np
import pandas as pd
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


def load_X_y(file_path_data: str, 
             file_path_labels: str = 'data/processed/labels.csv', 
             dtypes_data: dict = None, 
             dtypes_labels: dict = None, 
             n_rows: int = None):
    df = load_data(file_path=file_path_data, dtypes=dtypes_data, n_rows=n_rows)
    labels = load_labels(file_path=file_path_labels, dtypes=dtypes_labels, n_rows=n_rows)
    return (df, labels)

