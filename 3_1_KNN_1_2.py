# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:52:23 2025

@author: Recep Çakar
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
#(1) Veri seti incelemesi

cancer=load_breast_cancer()

df=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
df["target"]=cancer.target

#EXPLORER DATA ANALYİZE YAP

#(2) Makine öğrenim Modeli seçilmesi -KNN Olarak seçtik.

#(3) Modeli Train etmek
knn=KNeighborsClassifier()
X=cancer.data
y=cancer.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
#olceklendirme

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train) #fit öğrenir ,transform öğrendiğini dönüştürür
X_test=scaler.transform(X_test) # öğrenmiş olduğu için birdaha öğretmiyoruz.



knn.fit(X_train,y_train)



#(4) Modeli test etmek
y_pred=knn.predict(X_test)
score=accuracy_score(y_test, y_pred)
print(score)
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)


#(5) Hiperparametreleri ayarlamak

