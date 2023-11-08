import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("../../DataSet/weatherAUS.csv")
    print(f"Shape: {df.shape}")
    print(f"The presence of missing values {df.isnull().sum().sum() * 100.0 / (len(df) * len(df.keys()))}")
    print(f"Rain tomorrow: yes({len(df[df['RainTomorrow'] == 'Yes'])}) / no({len(df[df['RainTomorrow'] == 'No'])}) / NaN({len(df[df['RainTomorrow'].isna()])})")
    print(f"Rain tomorrow(percent): yes({len(df[df['RainTomorrow'] == 'Yes']) / len(df['RainTomorrow']) * 100}%) / no({len(df[df['RainTomorrow'] == 'No']) / len(df['RainTomorrow']) * 100}%)\
    NA ~{len(df[df['RainTomorrow'].isna()]) / len(df['RainTomorrow']) * 100}")