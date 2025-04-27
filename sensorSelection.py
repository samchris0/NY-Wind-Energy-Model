import os
import pandas as pd
import numpy as np
import json

from PrepareSensorData import get_mag_df, get_coordinate_dicts
from OptimizedSelection import get_optimized_sensors
from LazySelections import random_selection, greedy_geographic_selection, greedy_variance_selection

def get_sensor_selections(dist, k, types=['random', 'geographic', 'variance', 'optimized']):
    df_mag = get_mag_df(dist)  # Fetch or create the filtered data if it doesn't exist
    df_mag.columns = df_mag.columns.astype(int)
    lat_dict, lon_dict = get_coordinate_dicts(df_mag)
    
    # Sqrt of df_mag for variance selection
    df_mag_sqrt = np.sqrt(df_mag)
    
    sensor_selections = {}  # Dictionary to hold the final sensor selections
    
    # Attempt to load selection cache from CSV
    csv_path = 'data/selection_cache.csv'
    
    # If the file doesn't exist, create it
    if not os.path.exists(csv_path):
        df_empty = pd.DataFrame(columns=['selection_type', 'dist', 'k', 'selections'])
        df_empty.to_csv(csv_path, index=False)
    
    # Load the CSV if it exists
    if os.path.exists(csv_path):
        print(f"Selection cache file found at {csv_path}")
        df_selections = pd.read_csv(csv_path)
        filtered_df = df_selections[(df_selections['dist'] == dist) & (df_selections['k'] == k)]
        changes_made = False
        
        for type in types:
            if type in filtered_df['selection_type'].values:
                # Load the selection from the cache
                selection = filtered_df[filtered_df['selection_type'] == type]['selections'].values[0]
                sensor_selections[type] = json.loads(selection)
                print(f"Loaded {type} selection from cache.")
            else:
                changes_made = True
                if type == 'random':
                    lazy_random_selection = random_selection(list(df_mag_sqrt.columns), k=k)
                    new_row = pd.DataFrame({'selection_type': [type], 'dist': [dist], 'k': [k], 'selections': [json.dumps(lazy_random_selection)]})
                    df_selections = pd.concat([df_selections, new_row], ignore_index=True)
                    sensor_selections[type] = lazy_random_selection
                    print(f"Computed {type} selection.")
                elif type == 'geographic':
                    lazy_geographic_selection = greedy_geographic_selection(list(df_mag_sqrt.columns.astype(int)), lat_dict, lon_dict, k=k)
                    new_row = pd.DataFrame({'selection_type': [type], 'dist': [dist], 'k': [k], 'selections': [json.dumps(lazy_geographic_selection)]})
                    df_selections = pd.concat([df_selections, new_row], ignore_index=True)
                    sensor_selections[type] = lazy_geographic_selection
                    print(f"Computed {type} selection.")
                elif type == 'variance':
                    lazy_variance_selection = greedy_variance_selection(df_mag_sqrt, k=k)
                    new_row = pd.DataFrame({'selection_type': [type], 'dist': [dist], 'k': [k], 'selections': [json.dumps(lazy_variance_selection)]})
                    df_selections = pd.concat([df_selections, new_row], ignore_index=True)
                    sensor_selections[type] = lazy_variance_selection
                    print(f"Computed {type} selection.")
                elif type == 'optimized':
                    optimal_selection = get_optimized_sensors(df_mag_sqrt, k=k)
                    new_row = pd.DataFrame({'selection_type': [type], 'dist': [dist], 'k': [k], 'selections': [json.dumps(optimal_selection)]})
                    df_selections = pd.concat([df_selections, new_row], ignore_index=True)
                    sensor_selections[type] = optimal_selection
                    print(f"Computed {type} selection.")
        
        # Save the updated selections to the CSV file (if any changes were made)
        if changes_made:
            df_selections.to_csv(csv_path, index=False)
            print(f"Updated selection cache data to {csv_path}")
    
        return df_mag_sqrt, lat_dict, lon_dict, sensor_selections
    
    else:
        print('Error loading selection cache file.')
        return None, None, None, None