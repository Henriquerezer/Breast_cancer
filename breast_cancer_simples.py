# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:25:22 2022

@author: Henrique
"""

import pandas as pd 

previsores = pd.read_csv('./entradas_breast.csv')
classe = pd.read_csv('./saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
classificador.add(Dense(units = 16, activation= 'relu',
                        kernel_initializer= 'random_uniform', input_dim= 30))
classificador.add(Dense(units = 16, activation= 'relu',
                        kernel_initializer= 'random_uniform'))
classificador.add(Dense(units = 1, activation = 'sigmoid',))

opt = keras.optimizers.Adam(learning_rate=0.01, decay = 0.0001, clipvalue = 0.5)

classificador.compile(opt, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)

