import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# Convert DataFrame input/output into 3D array for LSTM
def df_to_X_y(data_X, data_y, window_size=24):
    """
    Converts DataFrame input/output into 3D array for LSTM.

    Parameters:
        data_X (np.ndarray): Features (already in numpy array format)
        data_y (np.ndarray): Targets (already in numpy array format)
        window_size (int): Time window for sequences

    Returns:
        X (np.ndarray): 3D input for LSTM
        y (np.ndarray): Target values
    """
    X, y = [], []
    for i in range(len(data_X) - window_size):
        X.append(data_X[i:i + window_size])
        y.append(data_y[i + window_size])
    return np.array(X), np.array(y)

# Build LSTM model
def buildModel(X):
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))  # Input layer for defining input shape
    model.add(LSTM(64))  # Reduced units in the first LSTM layer
    model.add(Dense(32, activation='relu'))  # Dense layer with fewer units
    model.add(Dense(1, activation='linear'))  # Output layer for regression task
    return model

# Train model with early stopping and model checkpoint
def trainModel(model, model_name, X, y):
    _80 = int(len(X) * 0.8)
    _10 = int(len(X) * 0.1)
    
    X_train, y_train = X[:_80], y[:_80]
    X_val, y_val = X[_80:_80 + _10], y[_80:_80 + _10]
    X_test, y_test = X[_80 + _10:], y[_80 + _10:]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    # Save best model
    cp1 = ModelCheckpoint(model_name, save_best_only=True, verbose=1)
    
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.001),
        metrics=[RootMeanSquaredError()]
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[cp1, early_stopping],
        verbose=1
    )
    
    # Reload best model
    best_model = load_model(model_name)
    
    # Predict on test
    test_predictions = best_model.predict(X_test, verbose=1).flatten()  # Ensure predictions are 1D
    
    # Flatten y_test if it's not already 1D
    y_test_flat = y_test.flatten()
    
    # Create a DataFrame with the results
    test_results = pd.DataFrame({
        'Test Predictions': test_predictions,
        'Actuals': y_test_flat  # Use flattened y_test
    })
    
    # Evaluate
    mse_score = mse(test_predictions, y_test_flat)
    mae_score = mean_absolute_error(y_test_flat, test_predictions)
    r2 = r2_score(y_test_flat, test_predictions)
    
    print(f"\n✅ Final Test MSE: {mse_score:.6f}")
    print(f"✅ Final Test MAE: {mae_score:.6f}")
    print(f"✅ Final Test R²: {r2:.6f}")
    
    # Plot results
    plt.plot(test_results['Test Predictions'], label='Predictions')
    plt.plot(test_results['Actuals'], label='Actuals')
    plt.legend()
    plt.title("Random Forest Regression Result")
    plt.show()
    
    return test_results, mse_score