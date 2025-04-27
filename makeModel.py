import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.models import load_model 
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

def df_to_X_y(df_X, df_y, window_size=24):
    data_X = df_X.to_numpy()
    data_y = df_y.to_numpy()
    X = []
    y = []
    for i in range(len(data_X)-window_size):
        row = [[x] for x in X[i:i+window_size]]
        X.append(row)
        y.append(data_y[i+window_size])
    return np.array(X), np.array(y)

def buildModel(X):
    data_shape = (X.shape[1],X.shape[2])

    model = Sequential()
    model.add(InputLayer(data_shape))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    return model

def trainModel(model,model_name,X,y):
    _80 = int(len(X)*(0.8))
    _10 = int(len(X)*(0.8))
    X_train, y_train = X[:_80], y[:_80]
    X_val, y_val = X[_80:_80+_10], y[_80:_80+_10]
    X_test, y_test = X[_80+_10:], y[_80+_10:]
    
    cp1 = ModelCheckpoint(f'{model_name}/', save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp1])

    test_predictions = model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})  

    MSE = mse(test_predictions, y_test)

    plt.plot(test_results['Predictions'])
    plt.plot(test_results['Actuals'])
    plt.show()

    return test_results, MSE

