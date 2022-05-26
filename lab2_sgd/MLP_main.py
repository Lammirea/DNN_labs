# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import MLP


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
#было y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)
#cтало:
yf = np.eye(100,3)
for i in range(0,100):
    if y[i] == "Iris-setosa":
        yf[i] = np.array([1,0,0])
    if y[i] == "Iris-versicolor":
        yf[i] = np.array([0,1,0])
    if y[i] == "Iris-virginica":
        yf[i] = np.array([0,0,1])
X = df.iloc[0:100, 0:4].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = yf.shape[1] # количество выходных сигналов равно количеству классов задачи

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# обучаем сеть (фактически сеть это вектор весов weights)
for i in range(iterations):
    net.train(X, yf)

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(yf - net.predict(X)))))

# считаем ошибку на обучающей выборке
pr = net.predict(X)
print(sum(abs(yf-(pr>0.5))))
