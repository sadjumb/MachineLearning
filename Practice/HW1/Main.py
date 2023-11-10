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
#from keras.layers import Dense, BatchNormalization, Dropout, LSTM
#from keras.models import Sequential
#from keras.utils import to_categorical
#from keras.optimizers import Adam
#from tensorflow.keras import regularizers
#from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
#from keras import callbacks
from dateutil.parser import parse
import datetime as dt


def dropCategoricalFeatures(df):
    binary = {'Yes': 1, 'No' : 0}
    data = df.replace({'RainTomorrow' : binary, 'RainToday' : binary})
    categorical = categoricalV(df)
    data = data.drop(categorical, axis=1)
    return data


def plotRainTomorrow(df: pd.DataFrame, PATH_IMAGES: str):
    fig, ax = plt.subplots()
    ax.bar(x=['No', 'Yes'], height=[df[df["RainTomorrow"] == "No"]["RainTomorrow"].count(), df[df["RainTomorrow"] == "Yes"]["RainTomorrow"].count()],\
           color=['red', 'green'])
    ax.set_title("Rain Tomorrow")
    fig.savefig(PATH_IMAGES + 'RainTomorrow.png')
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

    

if __name__ == "__main__":
    PATH_IMAGES = '/home/makar/Python/MachineLearning/Images/'
    df = pd.read_csv("../../DataSet/weatherAUS.csv", engine='python')
    
    #plotRainTomorrow(df, PATH_IMAGES)

    df = parseDate(df)
    dfWithoutCategorical = dropCategoricalFeatures(df)

    exploreCategoricalV(df)
    
    #corrMatrix(dfWithoutCategorical, PATH_IMAGES)
#
    #df.describe(include = ['object']).to_csv('describeCategorical.csv')
    #df.describe(include = 'all').to_csv('describeAll.csv')
    #dfWithoutCategorical.describe(include='all').to_csv('describeWithoutCategorical.csv')

    #scatterMatrix(df, PATH_IMAGES)
    
    
