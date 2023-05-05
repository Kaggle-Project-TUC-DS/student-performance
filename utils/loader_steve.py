#in this file we will collect the functions/utilities to load the preprocessed data for training purposes
#to do- merge the different preprocessing files and create 1 preprocessing function
#Stephan and Johannes

#load the preprocessed data
#work in progress

#define the dictionary to load in the data efficiently
dtypes={
    "level_group": "category",
    'level': np.uint8,
    'event_name_nunique': np.uint8,
    'name_nunique':np.uint8,
    'fqid_nunique':np.uint8,
    'room_fqid_nunique':np.uint8,
    'text_fqid_nunique':np.uint8, 
    'hover_duration_mean': np.float32,
    'difference_clicks_mean':np.float32,
    'elapsed_time_std':np.float32,
    'page_std':np.float32,
    'room_coor_x_std':np.float32,
    'room_coor_y_std':np.float32,
    'screen_coor_x_std':np.float32,
    'screen_coor_y_std':np.float32,
    'hover_duration_std':np.float32,
    'difference_clicks_std':np.float32,
    'index_sum_of_actions':np.uint16,
    'difference_clicks_max':np.float32,
    'elapsed_time_max':np.float32,
    'clicks_per_second':np.float32
    }