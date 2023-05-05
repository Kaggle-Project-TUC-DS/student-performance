import numpy as np
import pandas as pd
#in this file we will collect the functions/utilities to load the preprocessed data for training purposes
#to do- merge the different preprocessing files and create 1 preprocessing function
#Stephan and Johannes

#load the preprocessed data
#work in progress

#define the dictionary to load in the data efficiently

def load_train_data(file_path: str = "data/processed/train.csv", dtypes: = dict = None, n_rows: int = None)
    if dtypes = None:    
        dtypes={
            'level': np.uint8,  
            'session_id':np.int64,
            'level_group':'category',
            'event_name':np.int64,
            'name':np.int64,
            'fqid':np.int64,
            'room_fqid':np.int64,
            'text_fqid':np.int64,
            'fullscreen':'category',
            'hq':'category',
            'music':'category',
            'hover_duration_mean':np.float32,
            'difference_clicks_mean':np.float64,
            'elapsed_time_std':np.float64,
            'page_std':np.float64,
            'room_coor_x_std':np.float64,
            'room_coor_y_std':np.float64,
            'screen_coor_x_std':np.float64,
            'screen_coor_y_std':np.float64,
            'hover_duration_std':np.float64,
            'difference_clicks_std':np.float64,
            'index_sum_of_actions':np.int32,
            'difference_clicks_max':np.float64,
            'elapsed_time_max':np.float64,
            'clicks_per_second':np.float64}
        
    if n_rows = None:
        df = pd.read_csv(file_path, dtype = dtypes)
    else:
        df = pd.read_csv(file_path, dtype = dtypes, nrows= n_rows)
        
    return df
