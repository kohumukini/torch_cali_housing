import pandas as pd
import os

from sklearn.datasets import fetch_california_housing

def load_california_housing(save_raw = False, raw_path = "data/raw/california_housing.csv"): 
    df = fetch_california_housing(as_frame = True).frame

    if save_raw: 
        os.makedirs(os.path.dirname(raw_path), exist_ok = True)
        df.to_csv(raw_path, index = False)

    X = df.drop(columns = ["MedHousingVal"])
    y = df["MedHousingVal"]

    return X, y