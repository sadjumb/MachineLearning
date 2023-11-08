import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotRainTomorrow(data, PATH_IMAGES):
    fig, ax = plt.subplots()
    ax.bar(x=['No', 'Yes'], height=[df[df["RainTomorrow"] == "No"]["RainTomorrow"].count(), df[df["RainTomorrow"] == "Yes"]["RainTomorrow"].count()])
    ax.set_title("Rain Tomorrow")
    fig.savefig(PATH_IMAGES + 'RainTomorrow.png')

def getTypeVar(df):
    for i in df:
        print(type(df[i][0]))

if __name__ == "__main__":
    PATH_IMAGES = '/home/makar/Python/MachineLearning/Images/'
    df = pd.read_csv("../../DataSet/weatherAUS.csv", engine='python')
    getTypeVar(df)

