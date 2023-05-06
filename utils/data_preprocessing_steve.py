###Imports for the Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import gc

#function to delete variables from memory
def clear_memory(keep=None):
    if keep is None:
        keep = []
    for name in list(globals().keys()):
        if not name.startswith('_') and name not in keep:
            value = globals()[name]
            if isinstance(value, pd.DataFrame):
                del globals()[name]
    gc.collect()

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


def adding_new_variables_rescaling(dataset_df):
    dataset_df = dataset_df.sort_values(['session_id','elapsed_time'])
    dataset_df['elapsed_time'] = dataset_df['elapsed_time']/1000
    group = dataset_df.groupby(['session_id','level'])['elapsed_time'].diff()
    group = group.fillna(value= 0)
    dataset_df= dataset_df.assign(difference_clicks = group)

    return dataset_df

#Function to clean the sequential data for the training of the model

#For that Function to work we need to specify the variables in Categorical and Numerical & Counting

CATEGORICAL = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid', 'fullscreen', 'hq', 'music']
NUMERICALmean = ['hover_duration','difference_clicks']
NUMERICALstd = ['elapsed_time','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration', 'difference_clicks']
COUNTING = ['index']
MAXIMUM = ['difference_clicks', 'elapsed_time']



def feature_engineer_steve(dataset_df):
    dfs = []
    tmp = dataset_df.groupby(['session_id','level'])["level_group"].first()
    tmp.name = tmp.name 
    dfs.append(tmp)
    for c in CATEGORICAL:
        if c not in ['fullscreen', 'hq', 'music']:
            tmp = dataset_df.groupby(['session_id','level'])[c].agg('nunique')
        else:
            tmp = dataset_df.groupby(['session_id','level'])[c].first().astype(int).fillna(0)
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
    
# add Clicks per second afterwards cause we need the time for each level first
    dataset_df['clicks_per_second'] = dataset_df['index_sum_of_actions']/ dataset_df['elapsed_time_max']
 
    return dataset_df

import pandas as pd
import numpy as np
#do not use this one it uses a lot of memory
def flatten_df(dataset_df_level, exclude=[]): 
    #split the dataframe into three different ones depending on the level group
    groups = dataset_df_level.groupby('level_group')

    # Create a dictionary to store the resulting dataframes
    result = {}

    # Loop over each group
    for name, group in groups:
        # Add the group to the result dictionary
        result[name] = group

    # Access the resulting dataframes using their keys
    df_0_4 = result['0-4']
    df_5_12 = result['5-12']
    df_13_22 = result['13-22']

    dfs = [df_0_4, df_5_12, df_13_22]
    flattened_dfs = []

    for df in dfs:
        # create a new index col
        df = df.reset_index()
        df['_index'] = df.index + 1
        df = df.set_index('_index')
        df = df.drop(["level_group"], axis= 1)

        # get the unique session_ids and columns
        session_ids = df['session_id'].unique()
        cols = ['session_id'] + [f'{col}_{i+1}' for i in range(len(session_ids)) for col in df.columns if col != 'session_id']

        # create a new dataframe to hold the flattened data
        new_df = pd.DataFrame(columns=cols)

        # define a function to apply to each group
        def flatten_group(group):
            # combine the columns into a single row
            row_data = [group[col].iloc[i] if pd.notnull(group[col].iloc[i]) else np.nan for i in range(len(group)) for col in group.columns if col != 'session_id'] 
            # add None values to the row if necessary to make it the same length as the columns
            if len(row_data) < len(cols) - 1:
                row_data += [np.nan] * (len(cols) - len(row_data) - 1)
            # add the session_id to the beginning of the row
            row_data = [group['session_id'].iloc[0]] + row_data
            return pd.Series(row_data, index=cols)

        # apply the function to each group and concatenate the results
        new_df = pd.concat([flatten_group(group) for _, group in df.groupby('session_id')], axis=1).T

        # fill NaN values with np.nan
        new_df = new_df.fillna(np.nan)

        # convert the columns back to their original datatypes
        for col in df.columns:
            if col != 'session_id':
                new_df[[f'{col}_{i+1}' for i in range(len(session_ids))]] = new_df[[f'{col}_{i+1}' for i in range(len(session_ids))]].astype(df[col].dtype, errors='ignore')

        # sort the columns
        new_df = new_df[cols]

        # remove columns that contain only NaN values
        new_df = new_df.dropna(axis=1, how='all')

        # remove specified columns from exclude list
        for col_name in exclude:
            new_df = new_df[[col for col in new_df.columns if not (col.startswith(col_name + '_') and col != col_name + '_1')]]

        flattened_dfs.append(new_df)

    return flattened_dfs[0], flattened_dfs[1], flattened_dfs[2]

def flatten_df_one_at_a_time(df, exclude=[]):
    # create a new index col
    df = df.reset_index()
    df['_index'] = df.index + 1
    df = df.set_index('_index')
    df = df.drop(["level_group"], axis=1)

    # get the unique session_ids and columns
    session_ids = df['session_id'].unique()
    cols = ['session_id'] + [f'{col}_{i+1}' for i in range(len(session_ids)) for col in df.columns if col != 'session_id']

    # create a new dataframe to hold the flattened data
    new_df = pd.DataFrame(columns=cols)

    # define a function to apply to each group
    def flatten_group(group):
        # combine the columns into a single row
        row_data = [group[col].iloc[i] if pd.notnull(group[col].iloc[i]) else np.nan for i in range(len(group)) for col in group.columns if col != 'session_id']
        # add None values to the row if necessary to make it the same length as the columns
        if len(row_data) < len(cols) - 1:
            row_data += [np.nan] * (len(cols) - len(row_data) - 1)
        # add the session_id to the beginning of the row
        row_data = [group['session_id'].iloc[0]] + row_data
        return pd.Series(row_data, index=cols)

    # apply the function to each group and concatenate the results
    new_df = pd.concat([flatten_group(group) for _, group in df.groupby('session_id')], axis=1).T

    # fill NaN values with np.nan
    new_df = new_df.fillna(np.nan)

    # convert the columns back to their original datatypes
    for col in df.columns:
        if col != 'session_id':
            new_df[[f'{col}_{i+1}' for i in range(len(session_ids))]] = new_df[[f'{col}_{i+1}' for i in range(len(session_ids))]].astype(df[col].dtype, errors='ignore')

    # sort the columns
    new_df = new_df[cols]

    # remove columns that contain only NaN values
    new_df = new_df.dropna(axis=1, how='all')

    # remove specified columns from exclude list
    for col_name in exclude:
        new_df = new_df[[col for col in new_df.columns if not (col.startswith(col_name + '_') and col != col_name + '_1')]]

    return new_df


#for be
dataset_df = adding_new_variables_rescaling(dataset_df)
dataset_df = feature_engineer_steve(dataset_df)

#split the dataframe into three different ones depending on the level group
dataset_df = dataset_df.groupby('level_group')

# Create a dictionary to store the resulting dataframes
result = {}
# Loop over each group
for name, group in dataset_df:
    # Add the group to the result dictionary
    result[name] = group

# Access the resulting dataframes using their keys
df_0_4 = result['0-4']
df_5_12 = result['5-12']
df_13_22 = result['13-22']

clear_memory(keep= ["df_0_4", "df_5_12","df_13_22"])

#specify columns we want to exclude for the flattening
ex = ["level", "music","hq","fullscreen"]

#df_0_4_flattened, df_5_12_flattened, df_13_22_flattened = flatten_df(dataset_df, exclude= ex)
#make the dataframe, save it and delete it to save memory
df_0_4 = flatten_df_one_at_a_time(df_0_4,exclude= ex)
df_0_4.to_csv('data/processed/df_0_4_flattened.csv')

clear_memory(keep=["df_5_12","df_13_22"])


df_5_12 = flatten_df_one_at_a_time(df_5_12, exclude= ex)
df_5_12.to_csv('data/processed/df_5_12_flattened.csv')

clear_memory(keep= ["df_13_22"])

df_13_22 = flatten_df_one_at_a_time(df_13_22, exclude= ex)
df_13_22.to_csv('data/processed/df_13_22_flattened.csv')
# Export results


#'df_5_12_flattened.to_csv('data/processed/df_5_12_flattened.csv')
#'df_13_22_flattened.to_csv('data/processed/df_13_22_flattened.csv')
