import matplotlib.pyplot as plt
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


def dropCategoricalFeatures(df):
    binary = {'Yes': 1, 'No' : 0}
    data = df.replace({'RainTomorrow' : binary, 'RainToday' : binary})
    data = data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)
    return data


def plotRainTomorrow(df: pd.DataFrame, PATH_IMAGES: str):
    fig, ax = plt.subplots()
    ax.bar(x=['No', 'Yes'], height=[df[df["RainTomorrow"] == "No"]["RainTomorrow"].count(), df[df["RainTomorrow"] == "Yes"]["RainTomorrow"].count()],\
           color=['red', 'green'])
    ax.set_title("Rain Tomorrow")
    fig.savefig(PATH_IMAGES + 'RainTomorrow.png')
    plt.close()


def getTypeVar(df):
    for i in df:
        print(type(df[i][0]))


def corrMatrix(data: pd.DataFrame, PATH_IMAGES: str):
    corrmat = data.corr()
    cmap = sns.diverging_palette(260,-10,s=50, l=75, n=6, as_cmap=True)
    plt.subplots(figsize=(15,15))
    sns.heatmap(corrmat,cmap= cmap,annot=True, square=True)
    plt.title('Correlation matrix')
    plt.savefig(PATH_IMAGES + 'corrMatrix.png')
    plt.close()


if __name__ == "__main__":
    PATH_IMAGES = '/home/makar/Python/MachineLearning/Images/'
    df = pd.read_csv("../../DataSet/weatherAUS.csv", engine='python')
    data = dropCategoricalFeatures(df)

    #print(df.info())
    #getTypeVar(df)
    #plotRainTomorrow(df, PATH_IMAGES)

    corrMatrix(data, PATH_IMAGES)

