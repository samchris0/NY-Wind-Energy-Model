import random
import numpy as np
from geopy.distance import geodesic

def random_selection(sensor_ids,k):
    """
    Randomly select k sensor ids from the list of sensor ids.
    Parameters:
        sensor_ids (list): List of sensor ids to select from.
        k (int): Number of sensor ids to select.
    Returns:
        list: List of randomly selected sensor ids.
    """
    
    if k > len(sensor_ids):
        raise ValueError("k cannot be greater than the number of sensor ids")
    return random.sample(sensor_ids, k)


def geographic_distance(id1, id2, lat_dict, lon_dict):
    coord1 = (lat_dict[id1], lon_dict[id1])
    coord2 = (lat_dict[id2], lon_dict[id2])
    return geodesic(coord1, coord2).km  # use haversine-like distance in km

def greedy_geographic_selection(sensor_ids, lat_dict, lon_dict, k):
    """
    Greedily selects k sensor IDs that are geographically far from each other by maximizing pairwise distance in each iteration.

    Parameters:
        sensor_ids (list[int]): List of sensor IDs to choose from
        lat_dict (dict): sensor_id → latitude
        lon_dict (dict): sensor_id → longitude
        k (int): Number of sensors to select

    Returns:
        selected_ids (list[int]): Selected sensor IDs
    """
    if k > len(sensor_ids):
        raise ValueError("k cannot be greater than the number of available sensors")
    
    sensor_ids = list(sensor_ids)
    selected_ids = [sensor_ids[0]]  # Start with an arbitrary sensor

    while len(selected_ids) < k:
        best_candidate = None
        best_min_dist = -np.inf

        for candidate in sensor_ids:
            if candidate in selected_ids:
                continue
            
            # Calculate min distance from this candidate to any already selected sensor
            dists = [geographic_distance(candidate, sid, lat_dict, lon_dict) for sid in selected_ids]
            min_dist = min(dists)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate

        selected_ids.append(best_candidate)

    return selected_ids


def greedy_variance_selection(df_mag, k):
    """
    Selects the k sensors (columns) in df_mag with the highest variance over time.

    Parameters:
        df_mag (pd.DataFrame): DataFrame where columns are sensor IDs and rows are timestamps
        k (int): Number of sensors to select

    Returns:
        list: List of top-k sensor IDs (column names) with highest variance
    """
    if k > df_mag.shape[1]:
        raise ValueError("k cannot be greater than the number of sensors in df_mag")

    variances = df_mag.var(axis=0)  # Variance for each sensor (column)
    top_k = variances.nlargest(k).index.tolist()
    
    return top_k    