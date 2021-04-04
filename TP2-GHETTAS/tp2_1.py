
import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense

text = open("Dataset_ozone2.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Est", "1")
x = open("Dataset_ozone2.csv","w")
x.writelines(text)
x.close()

text = open("Dataset_ozone2.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Sud", "2")
x = open("Dataset_ozone2.csv","w")
x.writelines(text)
x.close()

text = open("Dataset_ozone2.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Ouest", "3")
x = open("Dataset_ozone2.csv","w")
x.writelines(text)
x.close()

text = open("Dataset_ozone2.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Nord", "4")
x = open("Dataset_ozone2.csv","w")
x.writelines(text)
x.close()

text = open("Dataset_ozone2.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Pluie", "1")
x = open("Dataset_ozone2.csv","w")
x.writelines(text)
x.close()

text = open("Dataset_ozone2.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Sec", "2")
x = open("Dataset_ozone2.csv","w")
x.writelines(text)
x.close()

text = open("Dataset_ozone2.csv", "r")
text = ''.join([i for i in text]) \
    .replace(";", ",")
x = open("Dataset_ozone2.csv","w")
x.writelines(text)
x.close()

def regression_model(): #define regression model
    model = Sequential() # create model
    model.add(Dense(14, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') # compile model
    return model

concrete_data = pd.read_csv('Dataset_ozone2.csv') #téléchargement des données et lecture avec la bibliothèque pandas
concrete_data.head()

concrete_data.shape # vérifier combien de points de données contient le jeu de donnée avec la fonction shape

concrete_data.describe()
concrete_data.isnull().sum()

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'maxO3']] # all columns except Strength
target = concrete_data['maxO3'] # Strength column

predictors.head() #rapide contrôle de validité des prédicteurs et des données cibles
target.head()

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
n_cols = predictors_norm.shape[1] # number of predictors

model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
