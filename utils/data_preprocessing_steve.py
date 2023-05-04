###
###Imports for the Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib as plt
import os
dtypes={
    'elapsed_time':np.int32,
    'event_name':'category',
    'name':'category',
    'level':np.uint8,
    'room_coor_x':np.float32,
    "index": np.int32,
    'room_coor_y':np.float32,
    'screen_coor_x':np.float32,
    'screen_coor_y':np.float32,
    'hover_duration':np.float32,
    'text':'category',
    'fqid':'category',
    'room_fqid':'category',
    'text_fqid':'category',
    'fullscreen':'category',
    'hq':'category',
    'music':'category',
    'level_group':'category'}

dataset_df = pd.read_csv('data/raw/train.csv', dtype=dtypes)

###Function to add variables to the whole dataset


def adding_new_variables(dataset_df):
    dataset_df = dataset_df.sort_values(['session_id','elapsed_time'])
    group = dataset_df.groupby(["session_id","level"])["elapsed_time"].diff()
    group = group.fillna(value= 0)
    dataset_df= dataset_df.assign(difference_clicks = group)

    return dataset_df

#Function to clean the sequential data for the training of the model

#For that Function to work we need to specify the variables in Categorical and Numerical & Counting

CATEGORICAL = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
NUMERICALmean = ['hover_duration',"difference_clicks"]
NUMERICALstd = ['elapsed_time','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration',"difference_clicks"]
COUNTING = ["index"]
MAXIMUM = ["difference_clicks", "elapsed_time"]

def feature_engineer_steve(dataset_df):
    dfs = []
    for c in CATEGORICAL:
        tmp = dataset_df.groupby(['session_id','level'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMERICALmean:
        tmp = dataset_df.groupby(['session_id','level'])[c].agg('mean')
        tmp.name = tmp.name + '_mean'
        dfs.append(tmp)
    for c in NUMERICALstd:
        tmp = dataset_df.groupby(['session_id','level'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)    
    for c in COUNTING:
        tmp = 1+ dataset_df.groupby(['session_id','level'])[c].agg('max')- dataset_df.groupby(['session_id','level'])[c].agg('min') 
        tmp.name = tmp.name + '_sum_of_actions'
        dfs.append(tmp)
    for c in MAXIMUM:
        tmp = dataset_df.groupby(['session_id','level'])[c].agg('max')- dataset_df.groupby(['session_id','level'])[c].agg('min') 
        tmp.name = tmp.name + '_max'
        dfs.append(tmp)
    
    dataset_df = pd.concat(dfs,axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id')
    
     #add mean-Clicks per second afterwards cause we need the time for each level
    dataset_df["mean_clicks_per_time"] = dataset_df["index_sum_of_actions"]/ dataset_df["elapsed_time_max"]
    return dataset_df

##test data preprocessing with Subset of 5 million rows
dataset_df = pd.read_csv("data/raw/train.csv", dtype = dtypes, nrows = 5000000)
os.chdir("N:\MASTER_DS\Code\Kaggle_competition\Kaggle-seminar\student-performance")

dataset_df_added = adding_new_variables(dataset_df)
dataset_df_level = feature_engineer_steve(dataset_df_added)
print(dataset_df_added)
print(dataset_df_level)