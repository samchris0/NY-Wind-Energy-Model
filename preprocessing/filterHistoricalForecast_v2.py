#This file goes through historical data and removes all data that is not within X km of a wind turbine

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import ast

def filter_nearby_sensors(reference_coords, sensor_dict, max_distance_km):
    """
    Returns sensor_ids whose coordinates are farther than max_distance_km from all reference points.
    
    Parameters:
        reference_coords (list of tuples): List of (lat, lon) reference coordinates.
        sensor_dict (dict): Dictionary {sensor_id: (lat, lon)} for target sensors.
        max_distance_km (float): Maximum allowed distance in kilometers.
    
    Returns:
        list of sensor_ids: IDs of sensors outside the distance threshold.
    """
    # Convert lat/lon to radians
    ref_radians = np.radians(reference_coords)
    sensor_ids = list(sensor_dict.keys())
    sensor_coords = list(sensor_dict.values())
    tgt_radians = np.radians(sensor_coords)

    # Build KDTree
    tree = cKDTree(ref_radians)
    EARTH_RADIUS_KM = 6371.0

    # Query distances
    distances, _ = tree.query(tgt_radians, distance_upper_bound=max_distance_km / EARTH_RADIUS_KM)

    # Return sensor IDs whose distance is inf (i.e., no close reference point)
    filtered_ids = [sensor_id for sensor_id, d in zip(sensor_ids, distances) if d == np.inf]

    return filtered_ids


def produce_filtered_dataset(unfiltered_data_path,turbine_data_path,coordinate_column_path,output_folder, max_distance_km):
    """
    Filters the dataset to remove data points that are within max_distance_km of any wind turbine.
    Saves the spatially filtered historical dataset to a new CSV file
    
    Parameters:
        unfiltered_data_path (str): Path to the unfiltered dataset CSV file.
        turbine_data_path (str): Path to the CSV file containing wind turbine data.
        coordinate_column_path (str): Path to the CSV file containing sensor coordinates.
        output_folder (str): Path to the folder where the filtered dataset will be saved.
        max_distance_km (float): Maximum allowed distance in kilometers.
        
    Returns:
        df of filtered data
    
    """
    df = pd.read_csv(turbine_data_path)
        
    df_ny = df[df["t_state"] == "NY"] # Filter for New York State (NY)
        
    # Select only coordinates
    ny_turbine_coords = list(zip(df_ny["ylat"], df_ny["xlong"]))
    
    df_sensor_coords = pd.read_csv(coordinate_column_path)
    df_sensor_coords = df_sensor_coords.drop_duplicates(subset='sensor_id')
 
    sensor_dict = dict(zip(df_sensor_coords['sensor_id'], zip(df_sensor_coords['latitude'], df_sensor_coords['longitude'])))    
        
    distant_sensor_ids = filter_nearby_sensors(ny_turbine_coords, sensor_dict, max_distance_km)
    
    columns_to_drop = [f"{prefix}_{id}" for id in distant_sensor_ids for prefix in ("u80", "v80")]
    
    print('Loading: ', unfiltered_data_path, end=' ')
    df_unfiltered = pd.read_csv(unfiltered_data_path)
    print('...Done')
    
    no_matches = [col for col in columns_to_drop if col not in df_unfiltered.columns]
    if no_matches:
        print('Error: Attempting to drop sensors that do not appear in the unfiltered dataframe')
    
    df_unfiltered = df_unfiltered.drop(columns=[col for col in columns_to_drop if col in df_unfiltered.columns])
    print(f'Successfully Dropped {len(columns_to_drop)} Columns ')
    
    output_path = output_folder + f'/{max_distance_km}km_historicalForecast2024.csv'
    
    df_unfiltered.to_csv(output_path, index=False)

    print(f"Saved filtered CSV to: {output_path}")
    
    return df_unfiltered

