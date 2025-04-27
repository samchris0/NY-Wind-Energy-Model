from herbie import Herbie
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from makeNYGrid import makeNYGrid
import shutil
import os
import csv

#delete cached cfgrib to avoid download errors
year = '2024'
grib_folder = '/Users/schristianson/data/hrrr'
shutil.rmtree(grib_folder)  # Deletes everything in the folder, including subdirectories
os.makedirs(grib_folder)

forecast_folder_path = f"/Users/schristianson/Desktop/NY Wind Energy Model/historicalForecasts{year}"

date_ranges = []

if os.path.exists(forecast_folder_path) and os.path.isdir(forecast_folder_path):
    # List all files in the folder
    files = sorted(os.listdir(forecast_folder_path))
    last_file = files[-1]
    file_path = os.path.join(forecast_folder_path, last_file)
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        historical_data = [row for row in reader]  # Convert reader object to list of dicts
        start_date = historical_data[-1]['Date']
        start_month = pd.to_datetime(start_date).month
        #get starting month
else:
    os.mkdir(forecast_folder_path)
    start_month = 1

for i in range(start_month+1, 13):
    start_date = f"{year}-{i:02}-01 00:00"
    end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd()
    date_ranges.append(list(pd.date_range(start=start_date, end=end_date, freq='h')))

x,y,coords = makeNYGrid()
n_coords = len(coords)
coords_df = pd.DataFrame([[c[0], c[1]] for c in coords], columns=['longitude','latitude'])


# Convert coords_df into xarray-friendly format
lats = coords_df["latitude"].values
lons = coords_df["longitude"].values

for date_range in date_ranges:
    historical_data = []
    print(date_range[0])
    month = date_range[0].month
    for count, date in enumerate(date_range, start=1):
        H = Herbie(
            date,  
            model="hrrr",  
            fxx=1,  
            product="sfc",
        )
        
        grib_path = H.get_localFilePath()
        print(grib_path)

        ds = H.xarray(r":[U\|V]GRD:[1\|8]0 m")  # List of two xarrays

        # Convert lat/lon to xarray DataArray for batch processing
        #points = xr.Dataset({'latitude': (['points'], lats), 'longitude': (['points'], lons)})

        # Nearest neighbor lookup for all coordinates at once
        matched_80m = ds[0].herbie.pick_points(coords_df, method="nearest")
        matched_10m = ds[1].herbie.pick_points(coords_df, method="nearest")

        # Construct a row with all wind data
        temp_dict = {'Date': date}

        for i in range(len(coords_df)):
            temp_dict[f'u80_({lats[i]}, {lons[i]})'] = matched_80m.u[i].item()
            temp_dict[f'v80_({lats[i]}, {lons[i]})'] = matched_80m.v[i].item()
            temp_dict[f'u10_({lats[i]}, {lons[i]})'] = matched_10m.u10[i].item()
            temp_dict[f'v10_({lats[i]}, {lons[i]})'] = matched_10m.v10[i].item()

        historical_data.append(temp_dict)
        
    historicalForecast = pd.DataFrame(historical_data)
    historicalForecast.to_csv(os.path.join(forecast_folder_path,f'historicalForecast{month:02}.csv'), index=False)

