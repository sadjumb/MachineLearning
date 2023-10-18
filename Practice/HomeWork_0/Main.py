
import pandas as pd
# import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("../../DataSet/weatherAUS.csv")
    print(f"Shape: {df.shape}")
    print(f"The presence of missing values {df.isnull().sum().sum() * 100.0 / (len(df) * len(df.keys()))}")
