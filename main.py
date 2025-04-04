import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def main():

    #Load in the coordinates for the points of interest
    #numClusters = 7
    #coords = getCoordinates()
    
    df = pd.read_csv('/Users/schristianson/Desktop/NY Wind Energy Model/historicalForecast10km2024/historicalForecast10km01.csv')
    print(df.shape)

    df = pd.read_csv('/Users/schristianson/Desktop/NY Wind Energy Model/historicalForecasts2024/historicalForecast01.csv')
    print(df.shape)
    #Load real time data and forecast for each location
    #Data of interest: Wind speed at turbine height, turbulence, wind variability, temperature
    
    

    #Forecast wind energy output with ML model
    

    #Get current energy demand, balance, and pricing, and energy demand forecast

    #Estimate wind utilization and penetration rate

    #Forecast wind prices

    

if __name__ == "__main__":
    main()

