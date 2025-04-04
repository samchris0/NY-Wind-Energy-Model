import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def dateparser(date_str):
    # Adjust the timezone offset format to ISO 8601 (from -05 to -05:00)
    date_str = str(date_str).strip()
    date_str = date_str + ":00"  # Insert the colon in the timezone offset
    date = pd.to_datetime(date_str, format="%Y-%m-%dT%H%z").tz_convert("UTC-05:00")
    return date

makeDataFrame = False
exploreData = False
trainForestModel = False
trainLinReg = True

if makeDataFrame == True:
    df_weather = pd.read_csv('historical_weather.csv', parse_dates=["date"])
    df_energy = pd.read_csv('historical_energy.csv')

    df_weather.drop('Unnamed: 0', axis=1, inplace=True)
    df_energy.drop('Unnamed: 0', axis=1, inplace=True)

    df_weather["date"] = df_weather["date"].dt.tz_convert("UTC-05:00")
    df_energy["period"] = df_energy["period"].astype(str).str.strip().apply(dateparser)

    #df_energy["period"] = pd.to_datetime(df_energy["period"])

    print(df_weather["date"].dtypes)
    print(df_energy["period"].dtypes)

    #Confirm date parsing and uniformity of format
    #print('Len of df_weather', len(df_weather))
    #print('Len of df_energy', len(df_energy))
    #print("Shared dates:", len(set(df_weather["date"]).intersection(set(df_energy["period"]))))
    #inter = set(df_weather["date"]).intersection(set(df_energy["period"]))

    df = pd.merge(df_energy, df_weather, left_on = 'period', right_on = 'date')
    df.set_index('period', inplace=True)
    df = df.dropna(axis=0, how='any')
    df.drop('date', axis=1).to_csv('EnergyWeatherCombined.csv')


if exploreData == True:
    if makeDataFrame == False:
        df = pd.read_csv('EnergyWeatherCombined.csv')
    
    df = df[df['period']>'2023-12-31']
    #sns.histplot(data=df[df['period']>'2023-12-31'], x="Megawatthours")
    print(df[df['period']>'2023-12-31']['Megawatthours'].mean())
    #plt.show()

    sns.pairplot(df, x_vars=df.columns[30:37], y_vars=['Megawatthours'])
    plt.show()


if trainForestModel == True:
    df = pd.read_csv('EnergyWeatherCombined.csv')
    
    df = df[df['period']>'2023-12-31']

    y = df['Megawatthours']
    
    #all variables
    X = df.drop(['period','Megawatthours'], axis=1)

    #just windspeed at 100m
    #X = df.loc[:, df.columns.isin(df.columns[30:37])]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Regression Mean Squared Error: {mse:.2f}")

    # save model
    joblib.dump(rf, "randomForestWind.joblib")
    print('Model saved')

if trainLinReg == True:
    if makeDataFrame == False:
        df = pd.read_csv('EnergyWeatherCombined.csv')
    
    y = df['Megawatthours']
    
    #just windspeed at 100m
    X_wind = df.loc[:, df.columns.isin(df.columns[30:37])]

    #all variables
    X_all = df.drop(['period','Megawatthours'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
    X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X_wind, y, test_size=0.2, random_state=42)

    model_wind = LinearRegression()
    model_all = LinearRegression()

    model_wind.fit(X_train_wind, y_train_wind)
    model_all.fit(X_train, y_train)

    y_pred_wind = model_wind.predict(X_test_wind)
    y_pred_all = model_all.predict(X_test)

    mse_wind = mean_squared_error(y_test_wind, y_pred_wind)
    mse_all = mean_squared_error(y_test, y_pred_all)

    print(f"Mean Squared Error (wind): {mse_wind}")
    print(f"Mean Squared Error (all): {mse_all}")