import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def make_kernel(df):
    """
    Creates a kernel covariance matrix for the given DataFrame using Gaussian Process Regression.
        
    Parameters:
        df (pd.DataFrame): DataFrame containing sensor data over time with 1 column per sensor 
        
    Returns:
        K (np.ndarray): Covariance matrix of the kernel
    """
    
    vals = df.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vals)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_scaled)

    # Heuristic for distance
    dists = pdist(X_scaled.T)  # pairwise distances
    median_dist = np.median(dists)

    kernel = RBF(length_scale=median_dist/2)
    gp = GaussianProcessRegressor(kernel=kernel)
    K = gp.kernel(X_imputed, X_imputed)
    return K


def mutual_info_gain(K_sub, sigma2):
    """
    Computes the mutual information gain for a subset of the covariance matrix
    Args:
        K_sub: Subset of the covariance matrix
        sigma2: Variance of observed values
    Returns:
        mutual_info: Mutual information gain
    """
    n = K_sub.shape[0]
    return 0.5 * np.linalg.slogdet(np.eye(n) + (1 / sigma2) * K_sub)[1]


def greedy_select(K_full, sensor_id_to_index, k=10, sigma2=1e-5):
    """
    Greedy Mutual Information Optimization to select points of maximum information

    Args:
        K_full: kernel matrix (N x N)
        sensor_id_to_index: dict mapping sensor_id to kernel index (0-based)
        k: number of sensors to select
        sigma2: observation noise variance

    Returns:
        selected_ids: list of selected sensor IDs
    """
    remaining_ids = list(sensor_id_to_index.keys())
    selected_ids = []

    for _ in range(k):
        best_gain = -np.inf
        best_sensor = None

        for sensor_id in remaining_ids:
            candidate_ids = selected_ids + [sensor_id]
            candidate_indices = [sensor_id_to_index[sid] for sid in candidate_ids]

            K_sub = K_full[np.ix_(candidate_indices, candidate_indices)]
            gain = mutual_info_gain(K_sub, sigma2)

            if gain > best_gain:
                best_gain = gain
                best_sensor = sensor_id

        selected_ids.append(best_sensor)
        remaining_ids.remove(best_sensor)

    return selected_ids


def get_optimized_sensors(df_mag_sqrt,k):
    """
    Optimizes sensor selection using greedy mutual information optimization.
    
    Args:
        df_mag_sqrt: DataFrame containing square root of sensor data over time with 1 column per sensor
        k: number of sensors to select

    Returns:
        selected_ids: list of selected sensor IDs
    """
    
    K = make_kernel(df_mag_sqrt)
    sensor_id_to_index = {sensor_id: idx for idx, sensor_id in enumerate(df_mag_sqrt.columns.astype(int))}
    return greedy_select(K_full=K,sensor_id_to_index=sensor_id_to_index,k=k)
    