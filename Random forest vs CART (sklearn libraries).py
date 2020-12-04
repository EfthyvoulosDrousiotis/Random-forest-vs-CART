# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:50:50 2020

@author: efthi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# We are reading our data
df = pd.read_csv(r"C:\Users\efthi\Downloads/heart.csv")

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)


df = df.drop(columns = ['cp', 'thal', 'slope'])
y = df.target.values
x_data = df.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

accuracies = {}
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10000, random_state = 5,criterion= "entropy")
rf.fit(x_train.T, y_train.T)

acc = rf.score(x_test.T,y_test.T)*100
accuracies['Random Forest'] = acc
print("random forest : {:.2f}%".format(acc))

#cart
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#critirion works better than entropy
a="entropy"
b="gini"
clf = tree.DecisionTreeClassifier(criterion=b)
cart_sum=0

for i in range (15):
    clf = clf.fit(x_train.T, y_train.T)
    acc_cart= clf.score(x_test.T,y_test.T)*100
    #print("cart:", acc_cart)
    cart_sum+= acc_cart
    



print("Cart: ", cart_sum/15)

