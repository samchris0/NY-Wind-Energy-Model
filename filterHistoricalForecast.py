#This file goes through historical data and removes all data that is not within 6 km of a wind turbine
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import ast

def filter_nearby_coordinates(reference_coords, target_coords, max_distance_km):
    """
    Returns coordinates that exceed the maximum amount of distance from the reference coords
    
    Parameters:
        reference_coords (list of tuples): List of (lat, lon) reference coordinates.
        target_coords (list of tuples): List of (lat, lon) target coordinates.
        max_distance_km (float): Maximum allowed distance in kilometers.
    
    Returns:
        list of tuples: Filtered target coordinates not within max_distance_km.
    """
    # Convert lat/lon to radians for haversine distance calculations
    ref_radians = np.radians(reference_coords)
    tgt_radians = np.radians(target_coords)

    # Build a KDTree for fast spatial queries
    tree = cKDTree(ref_radians)

    # Earth’s radius in km
    EARTH_RADIUS_KM = 6371.0

    # Query the tree for distances to the nearest reference point
    distances, _ = tree.query(tgt_radians, distance_upper_bound=max_distance_km / EARTH_RADIUS_KM)

    # Filter points within the max distance
    filtered_coords = [str(coord) for coord, d in zip(target_coords, distances) if d == np.inf]

    return filtered_coords

# Select folder to save data to
folder_path = '/Users/schristianson/Desktop/NY Wind Energy Model/historicalForecast10km2024'

# Load the dataset
df = pd.read_csv("/Users/schristianson/Desktop/NY Wind Energy Model/uswtdb_v7_2_20241120.csv")

# Filter for New York State (NY)
df_ny = df[df["t_state"] == "NY"]

# Select only coordinates
ny_turbine_coords = list(zip(df_ny["ylat"], df["xlong"]))

# Get forecast  coordinates
df = pd.read_csv("/Users/schristianson/Desktop/NY Wind Energy Model/historicalForecasts2024/historicalForecast01.csv")
columns = df.columns[1:]

data_coords = [ast.literal_eval(col[4:]) for col in columns]

max_distance = 10  # Kilometers

distant_targets = filter_nearby_coordinates(ny_turbine_coords, data_coords, max_distance)

columns_to_drop = [col for col in df.columns if any(sub in col for sub in distant_targets)]
df.drop(columns=columns_to_drop, inplace=True)
df.to_csv(os.path.join(folder_path,f'historicalForecast{max_distance}km01.csv'), index=False)

for i in range(2,13):
    print(i)
    getPath = f'/Users/schristianson/Desktop/NY Wind Energy Model/historicalForecasts2024/historicalForecast{i:02}.csv'
    df = pd.read_csv(getPath)
    df.drop(columns=columns_to_drop, inplace=True)
    df.to_csv(os.path.join(folder_path,f'historicalForecast{max_distance}km{i:02}.csv'), index=False)