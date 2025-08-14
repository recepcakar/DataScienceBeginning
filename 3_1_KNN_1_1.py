# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
""" 

from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#(1) Veri seti incelemesi
canser = load_breast_cancer()
df=pd.DataFrame(data=canser.data,columns=canser.feature_names)
df["target"]=canser.target

#EXPLORER DATA ANALYİZE YAP

#(2) Makine öğrenim Modeli seçilmesi -KNN Olarak seçtik.

#(3) Modeli Train etmek


knn=KNeighborsClassifier()
X=canser.data
y=canser.target
knn.fit(X,y) #verimizi kullanarak knn algoritmasını eğitiyor.




#(4) Modeli test etmek

y_predict=knn.predict(X) 
accuracy_score(y,y_predict)
#(5) Hiperparametreleri ayarlamak


