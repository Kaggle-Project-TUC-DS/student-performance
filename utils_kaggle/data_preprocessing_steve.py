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
    #wd = os.path.dirname(os.getcwd())
    #os.chdir(wd)
    #print('Working Directory: ', os.getcwd())

    if file_path and dtypes:
        data = load_data(file_path=file_path, dtypes=dtypes)
    elif data is None:
        print('Provide either data as a dataframe or a filepath. Neither of both was given.')
    else:
        data = data.astype(dtypes)

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

    grp_dict = split_level_groups(dataset_df)

    ex = ["level_group", "music", "hq", "fullscreen", "session_id"]
    drop = ["level"]

    df_generated_rows = pd.DataFrame()

    n_flatten = {'0-4': 5, '5-12': 8, '13-22': 10}

    for lvl_groups in grp_dict:
        grp_dict[lvl_groups], grps_missing_sessions, grps_new_rows = generate_rows(grp_dict[lvl_groups],
                                                                                   n_flatten=n_flatten[lvl_groups],
                                                                                   level_g=lvl_groups)
        grp_dict[lvl_groups] = combine_rows(grp_dict[lvl_groups], n_flatten=n_flatten[lvl_groups], drop=drop, only_one=ex)

        df_generated_rows = pd.concat([df_generated_rows, grps_new_rows])

        if not output:
            grp_dict[lvl_groups].to_csv('data/processed/df_' + str(lvl_groups) + '_flattened.csv')

    if saveIntermediateFiles:
        df_generated_rows.to_csv('data/processed/df_generated_rows.csv')

    if output:
        return grp_dict    # returns the whole dictionary
    else:
        print("The output was saved to data/processed/")  # Output = false ->


if __name__ == "__main__":
    # With this specification the data preprocessing should do the same as the previous version of the script:
    pp_pipeline_noah(data=None, file_path="data/raw/train.csv", flatten=True, saveIntermediateFiles=True,
                     dtypes=dtypes_raw, output=False)