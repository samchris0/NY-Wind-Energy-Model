import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from coordinates import getCoordinates, clusterTurbines
from getHistoricalWeather import getHistoricalWeather

def trainModel(numClusters, getData=False):
    if getData == True:
        #Load in the coordinates for the points of interest
        turbines = getCoordinates()
        coords = clusterTurbines(turbines, numClusters)

        #get weather data
        weather_data = getHistoricalWeather(coords)
        weather_data.to_csv(f'historical_weather_{numClusters}_clusters.csv')
    else:
        weather_data = pd.read_csv(f'historical_weather_{numClusters}_clusters.csv')
    