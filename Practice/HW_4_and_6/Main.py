import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dateutil.parser import parse
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def plotRainTomorrow(df: pd.DataFrame, PATH_IMAGES: str):
    plt.close()
    fig, ax = plt.subplots(figsize=(20,10))
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


def plotConfusionMatrix(cm, PATH_IMAGES: str, NAMEFILE: str):
    plt.close()

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots(figsize=(20,10))
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    ax.set_title("Confusion Matrix")

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


def dropCategoricalFeatures(df):
    binary = {'Yes': 1, 'No' : 0}
    data = df.replace({'RainTomorrow' : binary, 'RainToday' : binary})
    categorical = categoricalV(df)
    data = data.drop(categorical, axis=1)
    return data


def scatterMatrix(data: pd.DataFrame, PATH_IMAGES: str):
    pd.plotting.scatter_matrix(data, alpha=0.01, figsize=(10,10))
    plt.savefig(PATH_IMAGES + 'scatterMatrix.png')
    plt.close()


def describe(df, dfWithoutCategorical, PATH_IMAGES):
    plotRainTomorrow(df, PATH_IMAGES)
    exploreCategoricalV(df)
    corrMatrix(dfWithoutCategorical, PATH_IMAGES)

    df.describe(include = ['object']).to_csv('describeCategorical.csv')
    df.describe(include = 'all').to_csv('describeAll.csv')
    dfWithoutCategorical.describe(include='all').to_csv('describeWithoutCategorical.csv')

    scatterMatrix(df, PATH_IMAGES)


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

    return features


def createTrainTest(features):
    X = features.drop(["RainTomorrow"], axis=1)
    y = features["RainTomorrow"]

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    print(f"X.shape - {X.shape}")
    return X, y, X_train, X_test, y_train, y_test


def selectionOfHyperparametersKNN(X_train, y_train):
    nnb = np.arange(1, 30, 2)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid = {'n_neighbors': nnb}, cv=10)
    grid.fit(X_train, y_train)

    #best_cv_err = 1 - grid.best_score_
    #best_n_neighbors = grid.best_estimator_.n_neighbors 
    #print(f"best_cv_err: {best_cv_err}")

    print(f"best_n_neighbors KNN: {grid.best_estimator_.n_neighbors}") # type: ignore
    return grid.best_estimator_.n_neighbors # type: ignore


def selectionOfHyperparametersLR(X_train, y_train):
    parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 5, 10, 15, 20, 30, 40, 50, 100]}]

    logreg = LogisticRegression(solver='liblinear')
    grid = GridSearchCV(logreg, param_grid = parameters, cv = 5)

    grid.fit(X_train, y_train)

    print('GridSearch CV best score : {:.4f}\n\n'.format(grid.best_score_))
    print('Parameters that give the best results :','\n\n', (grid.best_params_))
    print()
    print('Estimator that was chosen by the search :','\n\n', (grid.best_estimator_))

    return grid.best_estimator_.penalty, grid.best_estimator_.C # type: ignore


def checkAccuracy(model, X_train, X_test, y_train, y_test, y_test_predict):
    #err_test  = np.mean(y_test  != knn.predict(X_test))
    err_test = accuracy_score(y_test, y_test_predict)

    #err_train = np.mean(y_train != knn.predict(X_train))
    err_train = accuracy_score(y_train, model.predict(X_train))

    print('Model accuracy score: {0:0.4f}'. format(err_test))
    print('Training-set accuracy score: {0:0.4f}'. format(err_train))


#Check for overfitting and underfitting
def checkOverfitting(model, X_train, X_test, y_train, y_test, y_test_predict):
    print("\nCheck for overfitting and underfitting: ")
    print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
    print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))


def confusionMatrix(y_test, y_test_predict, PATH_IMAGES, NAMEFILE):
    cm = confusion_matrix(y_test, y_test_predict)
    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0,0])
    print('\nTrue Negatives(TN) = ', cm[1,1])
    print('\nFalse Positives(FP) = ', cm[0,1])
    print('\nFalse Negatives(FN) = ', cm[1,0])
    plotConfusionMatrix(cm, PATH_IMAGES, NAMEFILE)


def runKNN(features, PATH_IMAGES):
    _, _, X_train, X_test, y_train, y_test = createTrainTest(features)

    best_n_neighbors = selectionOfHyperparametersKNN(X_train, y_train)
    
    knn = KNeighborsClassifier(n_neighbors=best_n_neighbors).fit(X_train, y_train)
    y_test_predict = knn.predict(X_test)

    checkAccuracy(knn, X_train, X_test, y_train, y_test, y_test_predict)
    checkOverfitting(knn, X_train, X_test, y_train, y_test, y_test_predict)
    confusionMatrix(y_test, y_test_predict, PATH_IMAGES, 'confusionMatrixKNN')


def runLogisticRegression(features, PATH_IMAGES):
    _, _, X_train, X_test, y_train, y_test = createTrainTest(features)

    penalty, c = selectionOfHyperparametersLR(X_train, y_train)

    logreg = LogisticRegression(penalty=penalty, C=c, solver='liblinear').fit(X_train, y_train)

    y_test_predict = logreg.predict(X_test)
    checkAccuracy(logreg, X_train, X_test, y_train, y_test, y_test_predict)
    checkOverfitting(logreg, X_train, X_test, y_train, y_test, y_test_predict)
    confusionMatrix(y_test, y_test_predict, PATH_IMAGES, 'confusionMatrixLR')


def runRandomForest(features, PATH_IMAGES):
    _, _, X_train, X_test, y_train, y_test = createTrainTest(features)

    rf = RandomForestClassifier(n_estimators=20).fit(X_train, y_train)

    y_test_predict = rf.predict(X_test)
    checkAccuracy(rf, X_train, X_test, y_train, y_test, y_test_predict)
    checkOverfitting(rf, X_train, X_test, y_train, y_test, y_test_predict)
    confusionMatrix(y_test, y_test_predict, PATH_IMAGES, 'confusionMatrixRF')


if __name__ == "__main__":
    pd.options.display.max_columns = 15

    PATH_IMAGES = '/home/makar/Python/MachineLearning/Images/'
    df = pd.read_csv("../../DataSet/weatherAUS.csv", engine='python')

    df = parseDate(df)
    dfWithoutCategorical = dropCategoricalFeatures(df)

    describe(df, dfWithoutCategorical, PATH_IMAGES)

    df = missingValuesCategorical(df)
    df = missingValuesNumerical(df)

    df = prepairingAttributes(df, PATH_IMAGES)
    runKNN(df, PATH_IMAGES)
    #runLogisticRegression(df, PATH_IMAGES)
    #runRandomForest(df, PATH_IMAGES)
