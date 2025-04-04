import requests
import pandas as pd
from coordinates import getCoordinates
from retry_requests import retry
from getHistoricalWeather import getHistoricalWeather


def getEnergyData():
    results = []
    base_url = 'https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key=3z4gd7XUypM2MGoNL6fv4IorLJ3gxI4zeArQJba0'
    
    offset = 0
    page_length = 5000
    
    params =    {
                "frequency": "local-hourly",
                'data[0]': "value",
                "facets[respondent][]": "NYIS",
                "facets[fueltype][]": "WND",           
                "offset": str(offset),
                "length": str(page_length),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
   }

    headers =   {
                "start": "2023-12-31T00:00",
                "end": "2024-12-31T00:00",
                }

    count = 0

    while True:
        params['offset'] = str(offset)
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()

            if data['response']['data']:
                #print(data['response']['data'])
                for item in data['response']['data']:
                    #print(item)
                    results.append({'period':item.get('period'),'Megawatthours':item.get('value')})
                #print('success') #test line ---------delete-----------
                #test line ---------delete-----------
        else:
            print(f"Details: {response.status_code}, {response.text}")
            continue
        
        if len(data['response']['data']) < page_length:
            break
        
        #Increment the offset for the next batch
        offset = int(offset)
        offset += page_length
        
        
        print(count)
        count += 1

    df = pd.DataFrame(results)

    return df

energy_data = getEnergyData()
energy_data.to_csv('historical_energy_24.csv')


