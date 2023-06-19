###Imports for the Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import gc
from typing import Tuple


def clear_memory(keep=None):
    if keep is None:
        keep = []
    for name in list(globals().keys()):
        if not name.startswith('_') and name not in keep:
            value = globals()[name]
            if isinstance(value, pd.DataFrame):
                del globals()[name]
    gc.collect()

def adding_new_variables_rescaling(dataset_df):
    
    dataset_df = dataset_df.sort_values(['session_id','elapsed_time'])
    dataset_df['elapsed_time'] = dataset_df['elapsed_time']/1000
    group = dataset_df.groupby(['session_id','level'])['elapsed_time'].diff()
    group = group.fillna(value= 0)
    dataset_df= dataset_df.assign(difference_clicks = group)

    return dataset_df

def feature_engineer_steve(dataset_df):
    CATEGORICAL = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid', 'fullscreen', 'hq', 'music']
    NUMERICALmean = ['hover_duration','difference_clicks', "distance_clicks", "screen_distance_clicks"]
    NUMERICALstd = ['elapsed_time','page', 'hover_duration', 'difference_clicks',"distance_clicks", "screen_distance_clicks"]
    COUNTING = ['index']
    MAXIMUM = ['difference_clicks', 'elapsed_time', "sum_distance_clicks"]
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
    
    # add martins sum of movement in level
    #happens before
    #dataset_df = adding_euclid_distance_sum_variable(dataset_df) #at this time is think it is just returning a single value and does not add anything to cumsum
    
    dataset_df = pd.concat(dfs,axis=1)
    dataset_df = dataset_df.fillna(-1)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.set_index('session_id') 
    
    # add Clicks per second afterwards cause we need the time for each level first
    dataset_df['clicks_per_second'] = dataset_df['index_sum_of_actions'] / dataset_df['elapsed_time_max']
    dataset_df["clicks_per_second"].replace([np.inf, -np.inf], 0, inplace=True)
    

    return dataset_df


def adding_euclid_distance_variable(dataset_df):
    # Sort the input DataFrame by the 'session_id' and 'elapsed_time' columns
    dataset_df = dataset_df.sort_values(['session_id','elapsed_time'])  
    # Interpolate missing values in the 'room_coor_x' and 'room_coor_y' columns
    coords = dataset_df.groupby(['session_id',"level"])[['room_coor_x', 'room_coor_y']].transform(lambda x: x.interpolate())   
    # Calculate the Euclidean distance between consecutive rows of the 'coords' DataFrame
    distance_clicks = np.linalg.norm(coords.diff(), axis=1).squeeze()  
    # Assign the calculated distances to a new column named 'distance_clicks'
    new_df = dataset_df.assign(distance_clicks=distance_clicks)    
    # Reset the index of the resulting DataFrame
    new_df.reset_index(inplace=True)    
    # Return the resulting DataFrame
    return new_df

def adding_screen_distance_clicks_variable(dataset_df):
    # Sort the input DataFrame by the 'session_id' and 'elapsed_time' columns
    dataset_df = dataset_df.sort_values(['session_id','elapsed_time'])    
    # Interpolate missing values in the 'screen_coor_x' and 'screen_coor_y' columns
    screen_coords = dataset_df.groupby(['session_id',"level"])[['screen_coor_x', 'screen_coor_y']].transform(lambda x: x.interpolate())
    # Calculate the Euclidean distance between consecutive rows of the 'screen_coords' DataFrame
    screen_distance_clicks = np.linalg.norm(screen_coords.diff(), axis=1).squeeze()    
    # Assign the calculated distances to a new column named 'screen_distance_clicks'
    new_df = dataset_df.assign(screen_distance_clicks=screen_distance_clicks)     
    # Return the resulting DataFrame
    return new_df

def adding_euclid_distance_cumsum_variable(dataset_df):
    # Replace NaN values in the 'distance_clicks' column with 0
    dataset_df['distance_clicks'] = dataset_df['distance_clicks'].fillna(0)
    # Compute the cumulative sum of the 'distance_clicks' column within each session
    sum_distance_clicks = dataset_df.groupby(['session_id'])['distance_clicks'].cumsum()
    # Assign the computed cumulative sum to a new column 'sum_distance_clicks' in the original dataframe
    new_df = dataset_df.assign(sum_distance_clicks=sum_distance_clicks) 
    return new_df

def adding_euclid_distance_sum_variable(dataset_df):
    # Replace NaN values in the 'distance_clicks' column with 0
    dataset_df['distance_clicks'] = dataset_df['distance_clicks'].fillna(0)
    # Compute the sum of the 'distance_clicks' column within each session and picks the max
    cumsum_distance_clicks_max = dataset_df.groupby(['session_id'])['distance_clicks'].sum()
    new_df = dataset_df.assign(cumsum_distance_clicks_max= cumsum_distance_clicks_max)
    return new_df

def get_adj_matrices(session_df,room_fqid_list):
    # looking at only one level in one session and generate a adjacence matrix
    adj_matrix_level = np.zeros((session_df['level'].nunique(),len(room_fqid_list),len(room_fqid_list)))
    adj_matrix_level_dist = np.zeros((session_df['level'].nunique(),len(room_fqid_list),len(room_fqid_list)))
    adj_matrix_level_time = np.zeros((session_df['level'].nunique(),len(room_fqid_list),len(room_fqid_list)))
    adj_matrix_level_velo = np.zeros((session_df['level'].nunique(),len(room_fqid_list),len(room_fqid_list)))
   
    # Iteriere über die eindeutigen Werte der Spalte 'level' im session_df_full DataFrame mit Hilfe des range-Befehls
    for level in range(session_df['level'].nunique()):
            
            # Filtere den session_df_full DataFrame nach dem aktuellen Level
            session_df_level = session_df[session_df['level'] == level]

            # Iteriere über die Indizes der room_fqid-Spalte im session_df_level DataFrame
            # Beginne mit dem ersten Index und ende einen Index vor dem letzten Index
            # Initialisieren des Raummerker 
            n = session_df_level['room_fqid'].index[0]

            for j in range(session_df_level['room_fqid'].index[0], session_df_level['room_fqid'].index[0] + len(session_df_level['room_fqid'])-1):                
                # Überprüfe, ob der Raum in der nächsten Zeile gleich dem Raum in der aktuellen Zeile ist
                h=1
                try:                    
                    if session_df_level['room_fqid'][j+1] == session_df_level['room_fqid'][j]:
                        pass
                    else:                
                        # Wenn die Räume nicht gleich sind, weise den aktuellen Raum und den vorherigen Raum zu Variablen zu
                        current_room = session_df_level['room_fqid'][j+1]
                        previous_room = session_df_level['room_fqid'][j]
                        
                        # Ermittle den Index des aktuellen und vorherigen Raums in der room_fqid_list
                        current_room_index = room_fqid_list.index(current_room)
                        previous_room_index = room_fqid_list.index(previous_room)
                        
                        # Erhöhe den Eintrag in der Adjazenzmatrix adj_matrix_level um 1
                        adj_matrix_level[level][previous_room_index][current_room_index] += 1                
                        # Erhöhe den Eintrag in der Adjazenzmatrix adj_matrix_level_dist um den im letzten Raum zurückgelegten Weg
                        adj_matrix_level_dist[level][previous_room_index][current_room_index] += session_df_level['sum_distance_clicks'][j]-session_df_level['sum_distance_clicks'][n]
                        # Erhöhe den Eintrag in der Adjazenzmatrix adj_matrix_level_time um die im letzten Raum vergange Zeit
                        adj_matrix_level_time[level][previous_room_index][current_room_index] += session_df_level['elapsed_time'][j]-session_df_level['elapsed_time'][n]
                        # Merker für Raumwechsel
                        n = j

                        if adj_matrix_level_time[level][previous_room_index][current_room_index] !=0:
                            adj_matrix_level_velo[level][previous_room_index][current_room_index] = int(adj_matrix_level_dist[level][previous_room_index][current_room_index]/adj_matrix_level_time[level][previous_room_index][current_room_index])                            
                except:
                    pass                    

    return  adj_matrix_level, adj_matrix_level_dist, adj_matrix_level_time, adj_matrix_level_velo

def calc_gravity(matrix):
    gesamtmasse = np.sum(matrix)
    zeilen, spalten = matrix.shape

    x_schwerpunkt = 0
    y_schwerpunkt = 0

    for i in range(zeilen):
        for j in range(spalten):
            masse = matrix[i, j]
            x_schwerpunkt += j * masse
            y_schwerpunkt += i * masse
    try:
        if gesamtmasse == 0 :
            pass
        else:
            x_schwerpunkt /= gesamtmasse
            y_schwerpunkt /= gesamtmasse
    except:
        pass

    return x_schwerpunkt, y_schwerpunkt

def adj_gravity(dataset_df):
    drop_df = dataset_df.drop_duplicates(subset=['session_id'])
    session_id_list = drop_df['session_id'].to_list()

    drop_df = dataset_df.drop_duplicates(subset=['room_fqid'])
    room_fqid_list = drop_df['room_fqid'].to_list() 

    level_count = dataset_df['level'].nunique()
    coordinats = 2

    adj_matrix_level_gravity = np.zeros((level_count, coordinats))
    adj_matrix_level_dist_gravity = np.zeros((level_count, coordinats))
    adj_matrix_level_time_gravity = np.zeros((level_count, coordinats))
    adj_matrix_level_velo_gravity = np.zeros((level_count, coordinats))

    # Erstellen des DataFrames
    gravity_df = pd.DataFrame({
                'level': 0,
                'session_id': 0,
                'gravity_x': adj_matrix_level_gravity[:, 0],
                'gravity_y': adj_matrix_level_gravity[:, 1],
                'dist_gravity_x': adj_matrix_level_dist_gravity[:, 0],
                'dist_gravity_y': adj_matrix_level_dist_gravity[:, 1],
                'time_gravity_x': adj_matrix_level_time_gravity[:, 0],
                'time_gravity_y': adj_matrix_level_time_gravity[:, 1],
                'velo_gravity_x': adj_matrix_level_velo_gravity[:, 0],
                'velo_gravity_y': adj_matrix_level_velo_gravity[:, 1]
                })

    drop_df = dataset_df.drop_duplicates(subset=['session_id'])  
    session_id_list = drop_df['session_id'].to_list()
    try:
        for session_id in range(len(session_id_list)):            
            session_df_full = dataset_df[dataset_df['session_id'] == session_id_list[session_id]]
            try:
                adj_matrix_level, adj_matrix_level_dist, adj_matrix_level_time, adj_matrix_level_velo = get_adj_matrices(session_df_full, room_fqid_list)            
            except:
                print('not used sessionindex and session',session_id,session_id_list[session_id])
            
            # Erstellen des DataFrames
            sub_df = pd.DataFrame({
                'level': 0,
                'session_id': 0,
                'gravity_x': adj_matrix_level_gravity[:, 0],
                'gravity_y': adj_matrix_level_gravity[:, 1],
                'dist_gravity_x': adj_matrix_level_dist_gravity[:, 0],
                'dist_gravity_y': adj_matrix_level_dist_gravity[:, 1],
                'time_gravity_x': adj_matrix_level_time_gravity[:, 0],
                'time_gravity_y': adj_matrix_level_time_gravity[:, 1],
                'velo_gravity_x': adj_matrix_level_velo_gravity[:, 0],
                'velo_gravity_y': adj_matrix_level_velo_gravity[:, 1]
                })
            
            for level in range(level_count):
                adj_matrix_level_gravity[level,:]  = calc_gravity(adj_matrix_level[level])
                adj_matrix_level_dist_gravity[level,:]  = calc_gravity(adj_matrix_level_dist[level])
                adj_matrix_level_time_gravity[level,:]  = calc_gravity(adj_matrix_level_time[level])
                adj_matrix_level_velo_gravity[level,:]  = calc_gravity(adj_matrix_level_velo[level])

                # Erstellen des DataFrames
                new_session_df  = pd.DataFrame({
                    'level': list(range(level_count)),
                    'session_id': session_id_list[session_id],
                    'gravity_x': adj_matrix_level_gravity[:, 0],
                    'gravity_y': adj_matrix_level_gravity[:, 1],
                    'dist_gravity_x': adj_matrix_level_dist_gravity[:, 0],
                    'dist_gravity_y': adj_matrix_level_dist_gravity[:, 1],
                    'time_gravity_x': adj_matrix_level_time_gravity[:, 0],
                    'time_gravity_y': adj_matrix_level_time_gravity[:, 1],
                    'velo_gravity_x': adj_matrix_level_velo_gravity[:, 0],
                    'velo_gravity_y': adj_matrix_level_velo_gravity[:, 1]
                })
                # Hinzufügen des neuen DataFrames zur vorhandenen DataFrame gravity_df
                sub_df.update(new_session_df)
            gravity_df = pd.concat([gravity_df, sub_df], ignore_index=True)          
    except:
        pass
    return gravity_df


def split_level_groups(df):
    # Split the dataframe into three different ones depending on the level group
    groups = df.groupby('level_group')

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

    # Sort each dataframe by "session_id" and "level"
    df_0_4 = df_0_4.sort_values(['session_id', 'level']).reset_index(drop=False)
    df_5_12 = df_5_12.sort_values(['session_id', 'level']).reset_index(drop=False)
    df_13_22 = df_13_22.sort_values(['session_id', 'level']).reset_index(drop=False)


    # Return the resulting dataframes
    return df_0_4, df_5_12, df_13_22

def combine_rows(df, n_flatten=5, only_one=None, drop=None):
    """
    Combines every n_flatten rows of a Pandas DataFrame into a new DataFrame, with each row containing the combined values from the n_flatten rows.

    Args:
        df (pandas.DataFrame): The DataFrame to combine.
        n_flatten (int): The number of rows to be combined into a single row.
        only_one (list): A list of column names to keep only the first occurrence of in the output DataFrame.
        drop (list): A list of column names to drop from the input DataFrame before performing the calculation.

    Returns:
        pandas.DataFrame: A new DataFrame containing one row for every n_flatten rows of the input DataFrame, with each row containing the combined values from the n_flatten rows.
    """
    # Create a copy of the input DataFrame to modify
    df = df.copy()

    # Drop specified columns from input DataFrame
    if drop:
        df = df.drop(columns=drop)

    # Check if all session_id's occur the same amount of times
    session_counts = df['session_id'].value_counts()
    if len(set(session_counts.values)) > 1:
        raise ValueError("Missing level: All session_id's should occur the same amount of times.")

    # Determine the number of rows and columns in the input DataFrame
    num_rows, num_cols = df.shape

    # Determine the number of new rows in the output DataFrame
    num_new_rows = num_rows // n_flatten

    # Reshape the flattened values into a new array with the desired shape
    values = df.values.flatten()
    new_values = values.reshape(num_new_rows, n_flatten*num_cols)

    # Create a new DataFrame from the reshaped values
    new_df = pd.DataFrame(new_values, columns=[f"{col}_{i}" for i in range(1, n_flatten+1) for col in df.columns])

    # Drop specified columns from output DataFrame
    if only_one:
        for col in only_one:
            keep_col = f"{col}_1"
            drop_cols = [f"{col}_{i}" for i in range(2, n_flatten+1)]
            new_df = new_df.drop(columns=drop_cols)

    return new_df
def generate_rows(df: pd.DataFrame, n_flatten: int, level_g: str):
    # Use value_counts() to get the count of each session_id
    counts = df['session_id'].value_counts()
    df['level'] = df['level'].astype(np.uint8)

    # Check if each group has the same number of rows
    if (counts % n_flatten).any():
        # Get the session_ids that need to be generated
        need_generating = counts[counts < n_flatten].index.tolist()
        num_generated_rows = 0
        generated_sessions = []
        generated_rows = []

        # Loop through the session_ids that need to be generated
        for session_id in need_generating:
            # Check if all levels are present in this session
            levels_present = set(df.loc[df['session_id'] == session_id, 'level'].unique())
            min_level = df['level'].min()
            max_level = df['level'].max()
            expected_levels = set(range(min_level, max_level + 1))
            if levels_present != expected_levels:
                # Generate new rows for missing levels
                missing_levels = expected_levels - levels_present
                new_rows = []
                for missing_level in missing_levels:
                    new_row = {"session_id": session_id, "level": missing_level}
                    for col in df.columns:
                        if col == "session_id" or col == "level":
                            continue
                        
                        else:
                            # Numeric column - set value to average of other values in column with the same level
                            other_values = df.loc[(df["session_id"] == session_id) & (df["level"] == missing_level), col]
                            if other_values.dtype.kind in 'biufc':
                                new_value = other_values.mean()
                                if np.isnan(new_value):
                                    new_value = df.loc[df["level"] == missing_level, col].mean()
                                new_row[col] = new_value
                    new_rows.append(new_row)

                # Add the new rows to the dataframe
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                num_generated_rows += len(new_rows)
                generated_sessions.append({"session_id": session_id, "num_rows": len(new_rows)})
                generated_rows.extend(new_rows)

        print(f"Generated {num_generated_rows} rows with indices: {list(range(len(df) - num_generated_rows, len(df)))}")

        # Create output dataframe 2
        df2 = pd.DataFrame(generated_sessions)
        df2 = df2.set_index("session_id")
        print("Generated rows per session id:")
        print(df2)

        # Create output dataframe 3
        df3 = pd.DataFrame(generated_rows)
        df3 = df3.reindex(df.columns, axis=1)
        print("Generated rows:")
        print(df3)
    else:
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
    df["level_group"] = level_g
    df = df.sort_values(by=["session_id", "level"])
    df = df.reset_index(drop=True)
    df3["level_group"] = level_g
    return df, df2, df3

#because i dont want to delete eversthing 

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
# do not use above 