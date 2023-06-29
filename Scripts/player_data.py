import pybaseball
from pybaseball import statcast, statcast_batter, playerid_lookup, statcast_sprint_speed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

def sample_left_skew_plyr(plyr_speed):
    """
    Create left-skewed distribution to impute sprint speed on (applies to triples only),
    given that the player's speed is within 1 ft/sec of the player's ACTUAL speed

    Note: Applied to each 0.1 ft/sec increment of +/- 1 of a player's speed

    Parameters:
    -----------
    plyr_speed: numpy.float64
        Float containing an indvidual player's sprint_speed
        
    Returns:
    --------
    sample : numpy.float64
        Randomly drawn sample from skewed distribution
    """
    # Grab mean and standard deviation of normal distribution
    mean = plyr_speed
    std = np.std(np.arange(plyr_speed - 1,plyr_speed + 1, 0.05))

    samples = np.random.normal(mean, std, 10000)

    # Filter samples to keep only positive values
    positive_samples = samples[samples > mean]
    # Create new distribution only of positive samples, sample once
    positive_dist = np.random.normal(np.median(positive_samples), np.std(positive_samples) / 2, size = 1000)
    sample = np.random.choice(positive_dist)
    return sample


def sample_right_skew_plyr(plyr_speed):
    """
    Create right-skewed distribution to impute sprint speed on (applies to singles only),
    given that the player's speed is within 1 ft/sec of the player's ACTUAL speed

    Note: Applied to each 0.1 ft/sec increment of +/- 1 of a player's speed

    Parameters:
    -----------
    plyr_speed: numpy.float64
        Float containing an indvidual player's sprint_speed
        
    Returns:
    --------
    sample : numpy.float64
        Randomly drawn sample from skewed distribution
    """
    mean = plyr_speed
    std = np.std(np.arange(plyr_speed - 1,plyr_speed + 1, 0.05))

    samples = np.random.normal(mean, std, 10000)

    # Filter samples to keep only negative values
    negative_samples = samples[samples < mean]

    # Create new distribution only of positive samples, sample once
    negative_dist = np.random.normal(np.median(negative_samples), np.std(negative_samples), size = 1000)
    sample = np.random.choice(negative_dist)
    return sample

  
def sample_normal_plyr(plyr_speed):
    """
    Create normal distribution to impute sprint speed on (applies to doubles only),
    given that the player's speed is within 1 ft/sec of the player's ACTUAL speed

    Note: Applied to each 0.1 ft/sec increment of +/- 1 of a player's speed

    Parameters:
    -----------
    plyr_speed: numpy.float64
        Float containing an indvidual player's sprint_speed
        
    Returns:
    --------
    sample : numpy.float64
        Randomly drawn sample from skewed distribution
    """
    # Grab mean and standard deviation of normal distribution
    mean = plyr_speed
    std = np.std(np.arange(plyr_speed - 1,plyr_speed + 1, 0.05))
    
    samples = np.random.normal(mean, std, 1000)

    # Sample once
    sample = np.random.choice(samples)
    return sample


def clean_plyr_data(data, sprint_speed):
    """
    Cleans Statcast requested public data, performs multiple operations:
        - variable filtering
        - filtering data to include non-homeruns hits ONLY
        - dropping of NA values
        - creating game_week column from game_date
        - calculates distance based on r-value multiplier multiplied to distance column values in polar coords
        - imputes sprint speed for individual players (different distributions for each hit type)

    Parameters:
    -----------
    data: pandas.DataFrame
        DataFrame containing Statcast requested player data, will be clearned
    sprint_speed : int
        Integer representing independent variable, used for imputation
        
    Returns:
    --------
    X : numpy.array
        Array of x-values for predicitions on trained and fitted model
    """
    
    # Define variables of interest
    hit_vars = ['events','game_date','home_team','batter','bb_type', 'des', "launch_speed", "launch_angle",'hc_x', 'hc_y','hit_distance_sc']

    # Limit variables, and runners who do not have sprint speed data
    data = data[hit_vars]
    data = data.dropna()

    # Handle the launch_speed, launch_angle, hc_x, hc_y, hit_distance_sc na values
    hit_types = ['triple','single','double','home_run']
    hits = data[data['events'].isin(hit_types)]

    # Create game_week column, where week of season is taken from game_date in Savant
    season_start = datetime.strptime('2022-04-01', '%Y-%m-%d').date()
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
    hits['sprint_speed'] = hits.apply(lambda x: np.where(x['events'] == 'triple', sample_left_skew_plyr(sprint_speed), np.where(x['events'] == 'single', sample_right_skew_plyr(sprint_speed), sample_normal_plyr(sprint_speed))), axis = 1)

    hits = hits.drop(['des','player_id'], axis = 1)
    hits = hits[['events','game_week','home_team','launch_speed','launch_angle','angle', 'calc_distance', 'contact_distance','sprint_speed']]

    # Take maximum of projected distance and r-value calculated distance
    hits['distance'] = hits[['calc_distance', 'contact_distance']].max(axis=1)
    hits = hits.drop(['calc_distance', 'contact_distance'], axis = 1)
    hits = hits[['events','game_week','home_team','launch_speed','launch_angle','angle','distance','sprint_speed']]

    X = hits.drop('events', axis=1)

    return X