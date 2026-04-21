import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(X, y, save_processed = False, processed_dir = "data/processed"): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

    if save_processed: 
        os.makedirs(processed_dir, exist_ok = True)
        np.save(os.path.join(processed_dir, "X_train.npy"), X_train.scaled)
        np.save(os.path.join(processed_dir, "X_test.npy"), X_test_scaled)
        np.save(os.path.join(processed_dir, "y_train.npy"), y_train_scaled)
        np.save(os.path.join(processed_dir, "y_test.npy"), y_test_scaled)
        joblib.dump(scaler, os.path.join(processed_dir, scaler.pkl))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler