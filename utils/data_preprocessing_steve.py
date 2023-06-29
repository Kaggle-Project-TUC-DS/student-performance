###Imports for the Data Preprocessing
import os
import numpy as np
import pandas as pd

# To make the imports compatible with Kaggle:
try:  # Kaggle variant:
    from loader_steve import load_data  # instead of load_train_data?
    # for Martins additional values
    from preprocessing_func import adding_euclid_distance_variable, adding_screen_distance_clicks_variable, \
        adding_euclid_distance_cumsum_variable
    from preprocessing_func import adding_new_variables_rescaling
    from preprocessing_func import feature_engineer_steve
    from preprocessing_func import generate_rows, combine_rows
    from preprocessing_func import split_level_groups
except ModuleNotFoundError:  # Local variant:
    from utils.loader_steve import load_data  # instead of load_train_data?
    # for Martins additional values
    from utils.preprocessing_func import adding_euclid_distance_variable, adding_screen_distance_clicks_variable, \
        adding_euclid_distance_cumsum_variable
    from utils.preprocessing_func import adding_new_variables_rescaling
    from utils.preprocessing_func import feature_engineer_steve
    from utils.preprocessing_func import generate_rows, combine_rows
    from utils.preprocessing_func import split_level_groups

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


def pp_pipeline_noah(data=None, file_path=None, group_by: str = 'level', flatten=True, saveIntermediateFiles=True, dtypes=None, output=True, df_mean_level=None):
    # Set wd: -> get working directory and remove last folder - currently compatible with folders: submission,
    # notebooks and content root TODO: Test this and make this more robust
    wd = os.getcwd()
    if wd[-10:] == 'submission':
        wd = wd[:-11]
        os.chdir(wd)
        print("Changed working directory: ", os.getcwd())
    elif wd[-9:] == 'notebooks':
        wd = wd[:-10]
        os.chdir(wd)
        print("Changed working directory: ", os.getcwd())
    elif wd[-5:] == 'utils':
        wd = wd[:-6]
        os.chdir(wd)
        print("Changed working directory: ", os.getcwd())
    # else:
        # print("No need to change working directory: ", wd)

    if file_path and dtypes:
        data = load_data(file_path=file_path, dtypes=dtypes)
    elif data is None:
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

    dataset_df = feature_engineer_steve(dataset_df, group_by=group_by)

    if saveIntermediateFiles:
        # save the leveled data (aggregated)
        dataset_df.to_csv('data/processed/df_level.csv')

    grp_dict = split_level_groups(dataset_df, group_by)

    if output and group_by == 'level_group':
        return grp_dict
    elif group_by == 'level_group':
        for lvl_groups in grp_dict:
            grp_dict[lvl_groups].to_csv('data/processed/df_' + str(lvl_groups) + '_group-by-level.csv')

    ex = ["level_group", "music", "hq", "fullscreen", "session_id"]
    drop = ["level"]

    df_generated_rows = pd.DataFrame()

    n_flatten = {'0-4': 5, '5-12': 8, '13-22': 10}

    for lvl_groups in grp_dict:

        grp_dict[lvl_groups], grps_missing_sessions, grps_new_rows = generate_rows(grp_dict[lvl_groups],
                                                                                   n_flatten=n_flatten[lvl_groups],
                                                                                   level_g=lvl_groups, mean_value_levels=df_mean_level)
        grp_dict[lvl_groups] = combine_rows(grp_dict[lvl_groups], n_flatten=n_flatten[lvl_groups], drop=drop,
                                            only_one=ex)

        df_generated_rows = pd.concat([df_generated_rows, grps_new_rows])

        if not output:
            grp_dict[lvl_groups].to_csv('data/processed/df_' + str(lvl_groups) + '_flattened.csv')

    if saveIntermediateFiles:
        df_generated_rows.to_csv('data/processed/df_generated_rows.csv')

    if output:
        return grp_dict  # returns the whole dictionary
    else:
        print("The output was saved to data/processed/")


if __name__ == "__main__":
    wd = os.getcwd()
    if wd[-5:] == 'utils':
        wd = wd[:-6]
        os.chdir(wd)
        print("Changed working directory: ", os.getcwd())

    train = load_data(file_path='data/raw/train.csv', n_rows=10000, dtypes=dtypes_raw)
    df_mean_level = pd.read_csv("data/processed/df_mean_level.csv")

    # With this specification the data preprocessing should do the same as the previous version of the script:
    pp_pipeline_noah(df_mean_level = df_mean_level ,data=train, file_path=None, group_by='level', flatten=True, saveIntermediateFiles=True,
                     dtypes=dtypes_raw, output=False)
