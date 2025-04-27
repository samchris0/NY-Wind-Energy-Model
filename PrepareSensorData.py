import numpy as np
import pandas as pd
import os

from preprocessing.filterHistoricalForecast_v2 import produce_filtered_dataset

def retrieve_data(dist):
    """
    Returns date indexed dataframe of historical forecast data filtered within dist km of turbines
    
    Parameters:
        dist (int): Distance in km from turbines to filter data
    
    Returns:
        df (pd.DataFrame): DataFrame with date as index and sensor data as columns
    """
    
    file_path = f'data/filtered_historicalForecasts/{dist}km_historicalForecast2024.csv'
    print(f'Looking for file at {file_path}...', end=' ')
    if os.path.exists(file_path):
        print(f'FOUND!\n')
        df = pd.read_csv(file_path)
    else:
        print('File does not exist, creating filtered dataset\n')
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


def get_mag_df(dist=10):
    """
    Returns a DataFrame with the magnitude of wind speed for each sensor (forward filled for missing hours)
    
    Parameters:
        dist (int): Distance in km from turbines to filter data
        
    Returns:
        df_mag (pd.DataFrame): DataFrame with date as index and sensor magnitude as columns
    """

    data = retrieve_data(dist)

    sensor_ids = sorted(set(col.split('_')[1] for col in data.columns if col.startswith('u80_')))

    mag_columns = {}

    for sensor_id in sensor_ids:
        u_col = f'u80_{sensor_id}'
        v_col = f'v80_{sensor_id}'
        if u_col in data.columns and v_col in data.columns:
            mag_columns[sensor_id] = np.sqrt(data[u_col]**2 + data[v_col]**2)

    # Concatenate all columns at once
    df_mag = pd.DataFrame(mag_columns, index=data.index)
    df_mag = df_mag[sorted(df_mag.columns)]

    df_mag = fill_missing_hours(df_mag).ffill() # forward fill missing values instead of leaving them as NaN
        
    return df_mag


def get_coordinate_dicts(df):
    """
    Returns ordered lat and lon dicts with coordinates
    
    Parameters:
        df (pd.DataFrame): DataFrame containing sensor data over time with 1 column per sensor 
        
    Returns:
        lat_dict (dict): Dictionary with sensor IDs as keys and latitude as values
        lon_dict (dict): Dictionary with sensor IDs as keys and longitude as values
    """
    
    sensor_ids = df.columns.astype(int)
            
    df_sensor_coords = pd.read_csv('data/coordinate_columns.csv')
    
    lat_dict = {sensor_id: df_sensor_coords[df_sensor_coords['sensor_id'] == sensor_id]['latitude'].iloc[0] for sensor_id in sensor_ids}
    lon_dict = {sensor_id: df_sensor_coords[df_sensor_coords['sensor_id'] == sensor_id]['longitude'].iloc[0] for sensor_id in sensor_ids}

    return lat_dict, lon_dict
