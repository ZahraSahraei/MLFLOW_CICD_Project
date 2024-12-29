import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import pickle 
import numpy as np

# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's - Target variable.

def train_and_save_model():
    # print ('sklearn.__version__ ' , sklearn.__version__)
    # print ('pandas.__version__ ' , pd.__version__)
    # print ('numpy.__version__ ' , np.__version__)

    data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
    X = data.drop(columns=["medv"])
    y = data['medv'] # target 

    X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_Train, y_train)

    with open("model.pkl" , 'wb') as file:
        pickle.dump(model, file)

    print(' Pickle file saved. ')

if __name__ == "__main__":
    train_and_save_model()