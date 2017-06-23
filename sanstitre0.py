#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:56:53 2017

@author: ryad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:47:26 2017

@author: ryad
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Country = LabelEncoder()
labelencoder_X_Gender = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])
onhotencoder = OneHotEncoder(categorical_features= [1])
X = onhotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from xgboost import XGBClassifier 
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= classifier,X= X_train, y = y_train, cv=10)
   