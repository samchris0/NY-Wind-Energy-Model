import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import itertools
import os

from preprocessing.filterHistoricalForecast_v2 import produce_filtered_dataset

def retrieveData(dist):
    # returns data frame that includes all measurements with time as index and columns as wind speed at given locations
    
    file_path = f'data/filtered_historicalForecasts/{dist}km_historicalForecast2024.csv'
    print(f'Looking for file at {file_path}...', end=' ')
    if os.path.exists(file_path):
        print(f'FOUND!')
        df = pd.read_csv(file_path)
    else:
        print('File does not exist, creating filtered dataset')
        df = produce_filtered_dataset(
            unfiltered_data_path = 'data/unfiltered_historicalForecast2024.csv',
            turbine_data_path = 'data/uswtdb_v7_2_20241120.csv',
            coordinate_column_path = 'data/coordinate_columns.csv',
            output_folder = 'data/filtered_historicalForecasts',
            max_distance_km = dist
        )
        
    df['Date'] = df['Date'].apply(lambda x: x if " " in x else x + " 00:00:00")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df

def fill_missing_hours(df):
    """
    Identifies missing hourly timestamps in a DataFrame and adds rows with NaN values.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with datetime index or column
        datetime_col (str): Name of datetime column (if not using index)
        
    Returns:
        pd.DataFrame: DataFrame with complete hourly sequence
    """
    
    # Create complete hourly range for the year
    start_date = df.index.min().floor('D')  # Start at beginning of first day
    end_date = df.index.max().ceil('D')     # End at midnight of last day
    full_range = pd.date_range(start=start_date, end=end_date, freq='h', inclusive='left')
    
    # Reindex to add missing hours
    df_complete = df.reindex(full_range)
    
    return df_complete

def getMagDf(dist=10):
    """
    Create a dataframe with rows as time and columns as coordinates
    """

    data = retrieveData(dist)

    #create a set of all coordinates in column names
    data_coords = set()
    for col in data.columns:
        coord = col.split("_")[1]
        data_coords.add(coord)

    df_mag = pd.DataFrame(index=data.index)
    new_cols = {}

    data_coords = list(data_coords)
    for coord in data_coords:
        matching_cols = [col for col in data.columns if str(coord) in col]
        df_selected = data[matching_cols]
        new_cols[str(coord)] = np.sqrt((df_selected ** 2).sum(axis=1))

    new_df = pd.DataFrame(new_cols)
    df_mag = pd.concat([df_mag, new_df], axis=1)
    df_mag = fill_missing_hours(df_mag)
    
    return df_mag

def makeKernel(df):
    """
    Makes the kernel covariance matrix
    """
    vals = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vals)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_scaled)

    # Heuristic for distance
    dists = pdist(X_scaled.T)  # pairwise distances
    median_dist = np.median(dists)

    kernel = RBF(length_scale=median_dist/2)
    gp = GaussianProcessRegressor(kernel=kernel)
    K = gp.kernel(X_imputed, X_imputed)
    return K

def getCoordinateGrid(df):
    """
    Takes dataframe with coordinates as the columns and returns ordered lat and lon dicts with coordinates
    columns are given as (lat, lon) strings
    """

    coords = df.columns.to_list()
    coords = [coord.strip("()") for coord in coords]
    lat, lon = zip(*[(float(lat), float(lon)) for lat, lon in (coord.split(", ") for coord in coords)])
    lat = sorted(set(lat))
    lon = sorted(set(lon))
    lat_dict=dict(zip(range(len(lat)),lat))
    lon_dict=dict(zip(range(len(lon)),lon))
    return lat_dict, lon_dict, coords

def mutual_info_gain(K_sub, sigma2):
    n = K_sub.shape[0]
    return 0.5 * np.linalg.slogdet(np.eye(n) + (1 / sigma2) * K_sub)[1]

def greedy_select(lat_dict, lon_dict, coords, K_full, k=10, sigma2=10**(-5)):
    """
    Greedy Mutual Information Optimization to select points of maximum information

    Args:
        lat_dict: dictionary of integer range as keys and lat coordinates as values
        lon_dict: dictionary of integer range as keys and lon coordinates as values
        coords: list of coords within x km from turbine as strings 'xxx, yyy'
        K_full: Covariance Kernel Matrix
        k: number of sensors to select
        sigma2: variance of observed values, not relevant to building distribution so set to small
        value for stability

    Returns:
        selected_indices: indices of selected coordinates
        selected_coords: selected coordinates
    """
    
    selected_indices = []
    selected_coords = []

    remaining_indices = list(itertools.product(list(lat_dict.keys()), list(lon_dict.keys())))

    remaining_indices = [i for i in remaining_indices if f"{lat_dict[i[0]]}, {lon_dict[i[1]]}" in coords]
    
    for _ in range(k):
        print(f'finding {_} out of {k}')
        best_gain = -np.inf
        best_idx = None
        
        for i in remaining_indices:
            candidate = selected_indices + [i]

            lats, lons = zip(*candidate)

            K_sub = K_full[np.ix_(lats, lons)]
            gain = mutual_info_gain(K_sub, sigma2)
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        print(best_idx)
        selected_indices.append(best_idx)
        selected_coords.append((lat_dict[best_idx[0]], lon_dict[best_idx[1]]))
        remaining_indices.remove(best_idx)

    return selected_indices, selected_coords


"""x_km_from_turbines = 10
df_mag = getMagDf(x_km_from_turbines)

#square root to stabilize variance
df_mag_sqrt = np.sqrt(df_mag)
K = makeKernel(df_mag_sqrt)

lat_dict, lon_dict = getCoordinateGrid(df_mag_sqrt)

best_indices, best_coords = greedy_select(lat_dict, lon_dict, K)
print(best_coords)"""

