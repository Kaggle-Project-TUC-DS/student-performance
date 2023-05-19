###Imports for the Data Preprocessing
import os

import numpy as np
import pandas as pd

from loader_steve import load_data  # instead of load_train_data?
# for Martins additional values
from preprocessing_func import adding_euclid_distance_variable, adding_screen_distance_clicks_variable, \
    adding_euclid_distance_cumsum_variable
from preprocessing_func import adding_new_variables_rescaling
from preprocessing_func import feature_engineer_steve
from preprocessing_func import generate_rows, combine_rows
from preprocessing_func import split_level_groups

# Load in the Raw Dataset
dtypes_raw = {
    'elapsed_time': np.int32,
    'event_name': 'category',
    'name': 'category',
    'level': np.uint8,
    'room_coor_x': np.float32,
    "index": np.int32,
    'room_coor_y': np.float32,
    'screen_coor_x': np.float32,
    'screen_coor_y': np.float32,
    'hover_duration': np.float32,
    'text': 'category',
    'fqid': 'category',
    'room_fqid': 'category',
    'text_fqid': 'category',
    'fullscreen': 'category',
    'hq': 'category',
    'music': 'category',
    'level_group': 'category'}


def pp_pipeline_noah(data=None, file_path=None, flatten=True, saveIntermediateFiles=True, dtypes=None, output=True):
    # set wd
    # get working directory and remove last folder
    # TODO: make this more robust
    wd = os.path.dirname(os.getcwd())
    os.chdir(wd)
    print('Working Directory: ', os.getcwd())

    if file_path and dtypes:
        data = load_data(file_path=file_path, dtypes=dtypes)
    elif data:
        print('Provide either data as a dataframe or a filepath. Neither of both was given.')

    dataset_df = adding_new_variables_rescaling(data)

    # Martins additional values
    dataset_df = adding_screen_distance_clicks_variable(dataset_df)
    dataset_df = adding_euclid_distance_variable(dataset_df)
    dataset_df = adding_euclid_distance_cumsum_variable(dataset_df)

    # save the raw data with added variables
    if saveIntermediateFiles:
        dataset_df.to_csv('data/processed/df_added_variables.csv')

    # Define which variables get which treatement from the added dataset
    CATEGORICAL = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid', 'fullscreen', 'hq', 'music']
    NUMERICALmean = ['hover_duration', 'difference_clicks', "distance_clicks", "screen_distance_clicks"]
    NUMERICALstd = ['elapsed_time', 'page', 'hover_duration', 'difference_clicks', "distance_clicks",
                    "screen_distance_clicks"]
    COUNTING = ['index']
    MAXIMUM = ['difference_clicks', 'elapsed_time', "sum_distance_clicks"]

    # copy them into the feature engineer function

    # Careful, werid fix of a problem: when changing the categories and variables. copy them into the right place in the deature engeneer function.
    # they cant be loaded across the files.

    dataset_df = feature_engineer_steve(dataset_df)

    if saveIntermediateFiles:
        # save the leveled data (aggregated)
        dataset_df.to_csv('data/processed/df_level.csv')

    # split the dataset into three parts based on level group
    df_0_4, df_5_12, df_13_22 = split_level_groups(dataset_df)

    # load the data
    # df_0_4 = pd.read_csv("data\processed\df_0_4.csv", dtype=dtypes, index_col= 0)
    # df_0_4 = df_0_4.reset_index(drop=True)
    # df_5_12 = pd.read_csv("data\processed\df_5_12.csv", dtype=dtypes, index_col= 0)
    # df_5_12 = df_5_12.reset_index(drop=True)
    # df_13_22 = pd.read_csv("data\processed\df_13_22.csv", dtype=dtypes, index_col= 0)
    # df_13_22 = df_13_22.reset_index(drop=True)
    # specify columns we want to exclude for the flattening ex: will only be present one time
    # #(we only need the music information one time and not for every level)
    # drop drop the coloumn completely. level is not required anymore
    ex = ["level_group", "music", "hq", "fullscreen", "session_id"]
    drop = ["level"]

    # df_0_4_flattened, df_5_12_flattened, df_13_22_flattened = flatten_df(dataset_df, exclude= ex)
    # make the dataframe, save it and delete it to save memory
    # df_0_4 = flatten_df_one_at_a_time(df_0_4,exclude= ex)

    df_0_4, df0_4_missing_sessions, df0_4_new_rows = generate_rows(df_0_4, n_flatten=5, level_g="0-4")
    df_0_4 = combine_rows(df_0_4, n_flatten=5, drop=drop, only_one=ex)

    if not output:
        df_0_4.to_csv('data/processed/df_0_4_flattened.csv')

    # clear_memory(keep=["df_5_12","df_13_22"])

    # df_5_12 = flatten_df_one_at_a_time(df_5_12, exclude= ex)
    df_5_12, df5_12_missing_sessions, df5_12_new_rows = generate_rows(df_5_12, n_flatten=8, level_g="5-12")
    df_5_12 = combine_rows(df_5_12, n_flatten=8, drop=drop, only_one=ex)

    if not output:
        df_5_12.to_csv('data/processed/df_5_12_flattened.csv')

    # clear_memory(keep= ["df_13_22"])

    # df_13_22 = flatten_df_one_at_a_time(df_13_22, exclude= ex)
    df_13_22, df13_22_missing_sessions, df13_22_new_rows = generate_rows(df_13_22, n_flatten=10, level_g="13-22")
    df_13_22 = combine_rows(df_13_22, n_flatten=10, drop=drop, only_one=ex)

    if not output:
        df_13_22.to_csv('data/processed/df_13_22_flattened.csv')

    # Export results
    # export the generated rows in a separated df to control later
    df_generated_rows = pd.concat([df0_4_new_rows, df5_12_new_rows, df13_22_new_rows])

    if saveIntermediateFiles:
        df_generated_rows.to_csv('data/processed/df_generated_rows.csv')

    if output:
        return df_0_4, df_5_12, df_13_22
    else:
        print("The output was saved to data/processed/")


if __name__ == "__main__":
    # With this specification the data preprocessing should do the same as the previous version of the script:
    pp_pipeline_noah(data=None, file_path="data/raw/train.csv", flatten=True, saveIntermediateFiles=True,
                     dtypes=dtypes_raw, output=False)