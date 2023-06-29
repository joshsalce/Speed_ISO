import pybaseball
from pybaseball import statcast, statcast_batter, playerid_lookup, spraychart, statcast_sprint_speed, statcast_running_splits
from pybaseball.plotting import plot_bb_profile

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import homogeneity_score, silhouette_score
from sklearn.decomposition import PCA
from imblearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime

import ipympl
from mpl_toolkits.mplot3d import Axes3D


def join_speed_dataframes(dataframes):
    """
    Join a list of dataframes based on a common player_id , calculate average sprint speed to impute on

    Parameters:
    -----------
    dataframes: list
        A list of pandas DataFrames to be joined.
        
    Returns:
    --------
    merged_df : pandas.DataFrame
        The joined dataframe with the average value in the average value column.
    """
    dataframes = [i.drop(['team_id','team','position','age','competitive_runs','bolts','hp_to_1b'], axis = 1) for i in dataframes]

    # Merge the dataframes on the 'ID' column
    merged_df = pd.merge(dataframes[0], dataframes[1], on='player_id', how='outer')

    # Iterate over the remaining dataframes and merge with the merged_df
    for i in range(2, len(dataframes)):
        merged_df = pd.merge(merged_df, dataframes[i], on='player_id', how='outer')


    # Drop the '_x' and '_y' columns
    drop_columns = [col for col in merged_df.columns if col.endswith(('_x', '_y'))]


    # Calculate the average value for the repeated column
    value_cols = [col for col in merged_df.columns if col.startswith('sprint')]
    merged_df['sprint_speed'] = round(merged_df[value_cols].mean(axis=1, skipna=True),1)
    
    merged_df.drop(drop_columns, axis = 1, inplace=True)
    return merged_df.dropna()



def sample_left_skew(id, speed_df):
    """
    Create left-skewed distribution to impute sprint speed on (applies to triples only)

    Parameters:
    -----------
    id: numpy.float64
        Float containing an indvidual player's player_id
    speed_df : pandas.Dataframe
        Dataframe from outer join of multiple Statcast sprint speed dataframes
        
    Returns:
    --------
    sample : numpy.float64
        Randomly drawn sample from skewed distribution
    """
    # Grab mean and standard deviation of normal distribution
    mean = speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0]
    std = np.std(np.arange(speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0] - 1,
                    speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0] + 1, 0.05))

    samples = np.random.normal(mean, std, 10000)

    # Filter samples to keep only positive values
    positive_samples = samples[samples > mean]
    # Create new distribution only of positive samples, sample once
    positive_dist = np.random.normal(np.median(positive_samples), np.std(positive_samples) / 2, size = 1000)
    sample = np.random.choice(positive_dist)
    return sample


def sample_right_skew(id, speed_df):
    """
    Create right-skewed distribution to impute sprint speed on (applies to singles only)

    Parameters:
    -----------
    id: numpy.float64
        Float containing an indvidual player's player_id
    speed_df : pandas.Dataframe
        Dataframe from outer join of multiple Statcast sprint speed dataframes
        
    Returns:
    --------
    sample : numpy.float64
        Randomly drawn sample from skewed distribution
    """
    mean = speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0]
    std = np.std(np.arange(speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0] - 1,
                    speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0] + 1, 0.05))

    samples = np.random.normal(mean, std, 10000)

    # Filter samples to keep only negative values
    negative_samples = samples[samples < mean]

    # Create new distribution only of positive samples, sample once
    negative_dist = np.random.normal(np.median(negative_samples), np.std(negative_samples), size = 1000)
    sample = np.random.choice(negative_dist)
    return sample

  
def sample_normal(id, speed_df):
    """
    Create normal-skewed distribution to impute sprint speed on (applies to doubles only)

    Parameters:
    -----------
    id: numpy.float64
        Float containing an indvidual player's player_id
    speed_df : pandas.Dataframe
        Dataframe from outer join of multiple Statcast sprint speed dataframes
        
    Returns:
    --------
    sample : numpy.float64
        Randomly drawn sample from skewed distribution
    """
    # Grab mean and standard deviation of normal distribution
    mean = speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0]
    std = np.std(np.arange(speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0] - 1,
                    speed_df['sprint_speed'].loc[speed_df['player_id'] == id].values[0] + 1, 0.05))
    
    samples = np.random.normal(mean, std, 1000)

    # Sample once
    sample = np.random.choice(samples)
    return sample

def clean_data(data, speed_df):
    """
    Cleans Statcast requested public data, performs multiple operations:
        - variable filtering
        - filtering data to include non-homeruns hits ONLY
        - dropping of NA values
        - creating game_week column from game_date
        - calculates distance based on r-value multiplier multiplied to distance column values in polar coords
        - imputes sprint speed for individual players (different distributions for each hit type)
        - encodes y-values 

    Parameters:
    -----------
    id: numpy.float64
        Float containing an indvidual player's player_id
    speed_df : pandas.Dataframe
        Dataframe from outer join of multiple Statcast sprint speed dataframes
        
    Returns:
    --------
    X : numpy.array
        Array of x-values for training data
    y : numpy.array
        Array of y-values for training data
    label_encoder: sklearn LabelEncoder()
        Object that encodes y-values for training data
    """
    
    # Define variables of interest
    hit_vars = ['events','game_date','home_team','batter','bb_type', 'des', "launch_speed", "launch_angle",'hc_x', 'hc_y','hit_distance_sc']

    # Limit variables, and runners who do not have sprint speed data
    data = data[hit_vars]
    data = data[data['batter'].isin(speed_df['player_id'].unique())]
    data = data.dropna()

    # Handle the launch_speed, launch_angle, hc_x, hc_y, hit_distance_sc na values
    hit_types = ['triple','single','double','home_run']
    hits = data[data['events'].isin(hit_types)]

    # Create game_week column, where week of season is taken from game_date in Savant
    season_start = datetime.strptime('2021-04-01', '%Y-%m-%d').date()
    hits['game_date'] = pd.to_datetime(hits['game_date'])
    hits['game_date'] = hits.apply(lambda x: (x['game_date'].date() - season_start).days // 7, axis = 1)
    hits = hits.rename(columns = {'game_date':'game_week'})

    # Takes negative arctangent of translated x and y coordiantes (x/y), shifted by pi/2  to match appearance of spraychart
    hits['angle'] = ((-1*np.arctan((hits.hc_x - 130)/(210 - hits.hc_y))) + (np.pi / 2)) * (180/np.pi)

    # Given x and y, we calculate r value for polar coordiantes by simply taking hypotenuse of x and y
    hits['r'] = np.hypot((210 - hits.hc_y),(hits.hc_x - 130))

    # taking all home runs, average over the proportion of projected distance to r value
    r_value_multiplier = np.mean(hits[hits.events == 'home_run']['hit_distance_sc'] / hits[hits.events == 'home_run']['r'])

    # Scale up all values by r-value scalar to get better measure of total distance traveled
    hits['calc_distance'] = round(hits['r'] * r_value_multiplier,0)
    hits = hits.drop(['hc_x','hc_y'], axis = 1)

    # Eliminate home_runs, errors
    hits = hits[hits['events'] != 'home_run']
    hits = hits.rename(columns = {'batter':'player_id', 'hit_distance_sc': 'contact_distance'})
    hits = hits[~hits['des'].str.contains('fielding error')]

    # Impute sprint speed for each hit, create distributions for each hit type given the player's "average" sprint speed 
    hits['sprint_speed'] = hits.apply(lambda x: np.where(x['events'] == 'triple', sample_left_skew(x['player_id'],speed_df), np.where(x['events'] == 'single', sample_right_skew(x['player_id'],speed_df), sample_normal(x['player_id'],speed_df))), axis = 1)

    hits = hits.drop(['des','player_id'], axis = 1)
    hits = hits[['events','game_week','home_team','launch_speed','launch_angle','angle', 'calc_distance', 'contact_distance','sprint_speed']]

    # Take maximum of projected distance and r-value calculated distance
    hits['distance'] = hits[['calc_distance', 'contact_distance']].max(axis=1)
    hits = hits.drop(['calc_distance', 'contact_distance'], axis = 1)
    hits = hits[['events','game_week','home_team','launch_speed','launch_angle','angle','distance','sprint_speed']]

    X = hits.drop('events', axis=1)
    y = hits.events

    # Define the label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

# Grab training data
start_date = '2021-04-01'
end_date = '2021-10-03'
#end_date = '2022-05-03'

data = statcast(start_date, end_date, parallel = True)

'''
# Grab speed data for imputation
speed_data21 = statcast_sprint_speed(2021, 1)
speed_data22 = statcast_sprint_speed(2022, 1)

# Join speed data
speed_dfs = [speed_data21, speed_data22]
speed_data = join_speed_dataframes(speed_dfs)
'''
speed_data21 = statcast_sprint_speed(2021, 1)

# Clean training data, exported to Jupyter Notebook this script will be run in
X, y, label_encoder = clean_data(data, speed_data21)