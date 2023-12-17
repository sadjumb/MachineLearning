from telnetlib import DM
import matplotlib.pyplot as plt
from networkx import cartesian_product
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pandas as pd
from dateutil.parser import parse
import datetime as dt


def dropCategoricalFeatures(df):
    binary = {'Yes': 1, 'No' : 0}
    data = df.replace({'RainTomorrow' : binary, 'RainToday' : binary})
    categorical = categoricalV(df)
    data = data.drop(categorical, axis=1)
    return data


def plotRainTomorrow(df: pd.DataFrame, PATH_IMAGES: str):
    plt.close()
    fig, ax = plt.subplots()
    ax.bar(x=['No', 'Yes'], height=[df[df["RainTomorrow"] == "No"]["RainTomorrow"].count(), df[df["RainTomorrow"] == "Yes"]["RainTomorrow"].count()],\
           color=['red', 'green'])
    ax.set_title("Rain Tomorrow")
    fig.savefig(PATH_IMAGES + 'RainTomorrow.png')
    plt.close()


def plotOutliers(features: pd.DataFrame, PATH_IMAGES: str, NAMEFILE: str):
    plt.close()

    fig, ax = plt.subplots(figsize=(20,10))

    sns.boxenplot(data = features)
    plt.xticks(rotation=90)
    ax.set_title("Outliers")

    fig.savefig(PATH_IMAGES + f'{NAMEFILE}.png')
    plt.close()

def corrMatrix(data: pd.DataFrame, PATH_IMAGES: str):
    corrmat = data.corr()
    cmap = sns.diverging_palette(260,-10,s=50, l=75, n=6, as_cmap=True)
    plt.subplots(figsize=(15,15))
    sns.heatmap(corrmat,cmap= cmap,annot=True, square=True)
    plt.title('Correlation matrix')
    plt.savefig(PATH_IMAGES + 'corrMatrix.png')
    plt.close()


def parseDate(df: pd.DataFrame):
    dMonth = [parse(i).month for i in df['Date']]
    dDay = [parse(i).day for i in df['Date']]
    dYear = [parse(i).year for i in df['Date']]
    df.insert(0, 'Year', dYear)
    df.insert(0, 'Month', dMonth)
    df.insert(0, 'Day', dDay)
    df = df.drop('Date', axis=1)
    return df


def numericV(data):# Get list of numeric variables
    s = (data.dtypes == "float64")
    object_cols = list(s[s].index)
    return object_cols

def categoricalV(data):# Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)
    return object_cols


def scatterMatrix(data: pd.DataFrame, PATH_IMAGES: str):
    pd.plotting.scatter_matrix(data, alpha=0.01, figsize=(10,10))
    plt.savefig(PATH_IMAGES + 'scatterMatrix.png')
    plt.close()


def exploreCategoricalV(df: pd.DataFrame, missingValues: bool = False):
    categorical = categoricalV(df)

    if (missingValues):
        print(df[categorical].isnull().sum())

    print('The categorical variables are :', categorical)
    
    # Location
    location = dict()
    for k, v in df.Location.value_counts().items():
        location[k] = v
    print('Location contains', location, 'labels')

    pd.get_dummies(df.Location, drop_first=True)
    
    #binary = {'Yes': 1, 'No' : 0}
    #data = df.replace({'RainTomorrow' : binary, 'RainToday' : binary})


def missingValuesCategorical(df):
    # Missing values in categorical variables
    object_cols = categoricalV(df)
    print()
    print("Before filling missing values:")
    for i in object_cols:
        print(f"|{i:13}\t | {df[i].isnull().sum():7d}\t|")

    # Filling missing values with mode of the column in value
    for i in object_cols:
        df[i].fillna(df[i].mode()[0], inplace=True)

    print()
    print("After filling missing values:")
    for i in object_cols:
        print(i, df[i].isnull().sum())
    print()
    return df


def missingValuesNumerical(df):
    # Missing values in numeric variables
    num_cols = numericV(df)
    print()
    print("Before filling missing values:")
    for i in num_cols:
        print(f"|{i:13}\t | {df[i].isnull().sum():7d}\t|")
    
    # Filling missing values with median of the column in value
    for i in num_cols:
        df[i].fillna(df[i].median(), inplace=True)
    
    print()
    print("After filling missing values:")
    for i in num_cols:
        print(i, df[i].isnull().sum())
    print()
    return df


#Dropping with outlier
def droppingOutlier(features, target):
    features["RainTomorrow"] = target
    
    features = features[(features["MinTemp"]<2.3)&(features["MinTemp"]>-2.3)]
    features = features[(features["MaxTemp"]<2.3)&(features["MaxTemp"]>-2)]
    features = features[(features["Rainfall"]<4.5)]
    features = features[(features["Evaporation"]<2.8)]
    features = features[(features["Sunshine"]<2.1)]
    features = features[(features["WindGustSpeed"]<4)&(features["WindGustSpeed"]>-4)]
    features = features[(features["WindSpeed9am"]<4)]
    features = features[(features["WindSpeed3pm"]<2.5)]
    features = features[(features["Humidity9am"]>-3)]
    features = features[(features["Humidity3pm"]>-2.2)]
    features = features[(features["Pressure9am"]< 2)&(features["Pressure9am"]>-2.7)]
    features = features[(features["Pressure3pm"]< 2)&(features["Pressure3pm"]>-2.7)]
    features = features[(features["Cloud9am"]<1.8)]
    features = features[(features["Cloud3pm"]<2)]
    features = features[(features["Temp9am"]<2.3)&(features["Temp9am"]>-2)]
    features = features[(features["Temp3pm"]<2.3)&(features["Temp3pm"]>-2)]
    return features

# Prepairing attributes of scale data 
def prepairingAttributes(df, PATH_IMAGES):
    # Apply label encoder to each column with categorical data
    object_cols = categoricalV(df)
    label_encoder = LabelEncoder()
    for i in object_cols:
        df[i] = label_encoder.fit_transform(df[i])

    features = df.drop(['RainTomorrow', 'Year','Month', 'Day'], axis=1)
    target = df['RainTomorrow']
    col_names = list(features.columns)
    s_scaler = preprocessing.StandardScaler()
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names) 

    print(features.describe().T)
    #Detecting outliers
    plotOutliers(features, PATH_IMAGES, 'withOutliers')

    features = droppingOutlier(features, target)
    plotOutliers(features, PATH_IMAGES, 'withoutOutliers')
    


if __name__ == "__main__":
    pd.options.display.max_columns = 15

    PATH_IMAGES = '/home/makar/Python/MachineLearning/Images/'
    df = pd.read_csv("../../DataSet/weatherAUS.csv", engine='python')
    
    #plotRainTomorrow(df, PATH_IMAGES)

    df = parseDate(df)
    dfWithoutCategorical = dropCategoricalFeatures(df)

    #exploreCategoricalV(df)
    
    #corrMatrix(dfWithoutCategorical, PATH_IMAGES)

    #df.describe(include = ['object']).to_csv('describeCategorical.csv')
    #df.describe(include = 'all').to_csv('describeAll.csv')
    #dfWithoutCategorical.describe(include='all').to_csv('describeWithoutCategorical.csv')

    #scatterMatrix(df, PATH_IMAGES)
    
    df = missingValuesCategorical(df)
    df = missingValuesNumerical(df)

    prepairingAttributes(df, PATH_IMAGES)

    
    
