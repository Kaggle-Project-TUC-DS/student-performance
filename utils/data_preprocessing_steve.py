###Imports for the Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import gc
from typing import Tuple
from loader_steve import load_train_data

#Code
#############################################################################
#############################################################################

#Load in the Raw Dataset
dtypes_raw={
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

dataset_df = load_train_data(file_path= "data\raw\train.csv", dtypes= dtypes_raw)

from preprocessing_func import adding_new_variables_rescaling
dataset_df = adding_new_variables_rescaling(dataset_df)

#Define which variables get which treatement from the added dataset 
CATEGORICAL = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid', 'fullscreen', 'hq', 'music']
NUMERICALmean = ['hover_duration','difference_clicks']
NUMERICALstd = ['elapsed_time','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration', 'difference_clicks']
COUNTING = ['index']
MAXIMUM = ['difference_clicks', 'elapsed_time']

from preprocessing_func import feature_engineer_steve

dataset_df = feature_engineer_steve(dataset_df)

#load the data
df_0_4 = pd.read_csv("data\processed\df_0_4.csv", dtype=dtypes, index_col= 0)
df_0_4 = df_0_4.reset_index(drop=True)
df_5_12 = pd.read_csv("data\processed\df_5_12.csv", dtype=dtypes, index_col= 0)
df_5_12 = df_5_12.reset_index(drop=True)
df_13_22 = pd.read_csv("data\processed\df_13_22.csv", dtype=dtypes, index_col= 0)
df_13_22 = df_13_22.reset_index(drop=True)
#specify columns we want to exclude for the flattening
ex = ["level_group","music", "hq", "fullscreen"]
drop = ["level"]

#df_0_4_flattened, df_5_12_flattened, df_13_22_flattened = flatten_df(dataset_df, exclude= ex)
#make the dataframe, save it and delete it to save memory
#df_0_4 = flatten_df_one_at_a_time(df_0_4,exclude= ex)
df_0_4, df0_4_missing_sessions, df0_4_new_row = generate_rows(df_0_4,n_flatten= 5, level_g= "0-4")
df_0_4 = combine_rows(df_0_4,n_flatten= 5 ,drop= drop, only_one= ex)
df_0_4.to_csv('data/processed/df_0_4_flattened.csv')

#clear_memory(keep=["df_5_12","df_13_22"])


#df_5_12 = flatten_df_one_at_a_time(df_5_12, exclude= ex)
df_5_12, df5_12_missing_sessions, df5_12_new_rows = generate_rows(df_5_12,n_flatten= 8, level_g= "5-12")
df_5_12 = combine_rows(df_5_12,n_flatten= 8 ,drop= drop, only_one= ex)
df_5_12.to_csv('data/processed/df_5_12_flattened.csv')

#clear_memory(keep= ["df_13_22"])

#df_13_22 = flatten_df_one_at_a_time(df_13_22, exclude= ex)
df_13_22, df13_22_missing_sessions, df13_22_new_rows = generate_rows(df_13_22,n_flatten= 10, level_g= "13-22")
df_13_22 = combine_rows(df_13_22,n_flatten= 10 ,drop= drop, only_one= ex)
df_13_22.to_csv('data/processed/df_13_22_flattened.csv')
# Export results
#export the generated rows in a seperated df to controll later
df_generated_rows = pd.concat([df5_12_new_rows, df13_22_new_rows])
df_generated_rows.to_csv('data/processed/df_generated_rows.csv')

#'df_5_12_flattened.to_csv('data/processed/df_5_12_flattened.csv')
#'df_13_22_flattened.to_csv('data/processed/df_13_22_flattened.csv')



#Functions
#############################################################################
#############################################################################
#function to delete variables from memory


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



###Function to add variables to the whole dataset




#Function to clean the sequential data for the training of the model

#For that Function to work we need to specify the variables in Categorical and Numerical & Counting

CATEGORICAL = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid', 'fullscreen', 'hq', 'music']
NUMERICALmean = ['hover_duration','difference_clicks']
NUMERICALstd = ['elapsed_time','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration', 'difference_clicks']
COUNTING = ['index']
MAXIMUM = ['difference_clicks', 'elapsed_time']










#for be
#if we already have the dataset we can just load it in and not calculate it
#dataset_df = adding_new_variables_rescaling(dataset_df)
#dataset_df = feature_engineer_steve(dataset_df)
#new apporach: save these dataframes to avoid recalculkating them
#dataset_df.to_csv('data/processed/dataset_df_level.csv')

# load the dataframe
#split the dataframe into three different ones depending on the level group
dtypes = {
    'level': np.uint8,
    "level_group": "category",
    'event_name': np.uint8,
    'name': np.uint8,
    'fqid': np.uint8,
    'room_fqid': np.uint8,           
    "text_fqid": np.uint8,
    'fullscreen': np.uint8,
    'hq': np.uint8,
    'music': np.uint8,
    'hover_duration_mean': np.float32,
    'difference_clicks_mean': np.float32,
    'elapsed_time_std': np.float32,
    'page_std': np.float32,
    'room_coor_x_std': np.float32,
    'room_coor_y_std': np.float32,
    'screen_coor_x_std': np.float32,
    'screen_coor_y_std': np.float32,
    'hover_duration_std': np.float32,
    'difference_clicks_std': np.float32,
    'index_sum_of_actions': np.float32,
    'difference_clicks_max': np.float32,
    'elapsed_time_max': np.float32,
    'clicks_per_second': np.float32}

'''dataset_df = pd.read_csv("data/processed/dataset_df_level.csv", dtype=dtypes)
groups = dataset_df.groupby('level_group')

# Create a dictionary to store the resulting dataframes
result = {}

# Loop over each group

    # Add the group to the result dictionary
    result[name] = group

# Access the resulting dataframes using their keys
df_0_4 = result['0-4']
df_5_12 = result['5-12']
df_13_22 = result['13-22']
print(df_0_4.dtypes)

clear_memory(keep= ["df_0_4", "df_5_12","df_13_22"])'''
