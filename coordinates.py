#The purpose of this file is to query the US Wind Turbine Data Base to for relevant turbine information
import requests
import pandas as pd

####
#Note: Function not currently working. Data set is saved in CSV form and used in filterHistoricalForecast
#get function to start working to always have most up-to-date-data
#

def getCoordinates():
    
    
    selected_columns = ["t_hh", "t_cap", "p_year", "xlong", "ylat"]

    # Base URL for the USAWTDB API
    API_URL = "https://eersc.usgs.gov/api/uswtdb/v1/turbines"

    try:
        # Make the GET request
        response = requests.get(API_URL) #, params=params)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()

            turbines = [
                {col: turbine.get(col, None) for col in selected_columns} for turbine in data if turbine.get('t_state') == 'NY'
            ]

        else:
            print(f"Error: {response.status_code}, {response.text}")  

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    turbines = pd.DataFrame(turbines)
    return turbines

def clusterTurbines(turbines, numClusters):
    coordinates = list(turbines[["xlong", "ylat"]].itertuples(index=False, name=None))
    

    return coordinates

#df = getCoordinates()











"""
def getCoordinates():
    #return the list of coordinates of the points of interest. Lat (deg), Lon (deg), Height (m)
    coords = []
    #Noble Clinton Project, 80m
    coords.append((44.91969,-73.998, 80))
    #Maple Ridge Project, 80m
    coords.append((43.79069,-75.5941, 80))
    #Noble Wethersfield, 80m
    coords.append((42.64119,-78.2327, 80))
    #Dutch Hill/Cohorton, 80m
    coords.append((42.52609,-77.4563, 80))
    #Awkwright Summit, 95m
    coords.append((42.39377,-79.21597, 95))
    #Eight Point Wind, 117m
    coords.append((42.09926,-77.72263, 117))
    #Blue Stone, 120m
    coords.append((42.1039,-75.50745, 120))
    return coords"""