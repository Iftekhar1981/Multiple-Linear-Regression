# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:03:27 2022

@author: 47483
"""

# Multiple Linear Regression 

# Import libraries
import pandas as pd

# Import the dataset
dataset = pd.read_csv(r'C:\Study\Data Science\Fifty_Startups.csv')


#Matrix of features (Independent Variables) 
X = dataset.iloc[:, :-1].values
#Vector of the dependent variable
y = dataset.iloc[:, -1].values


# Encod the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)


# split the modelling dataset into the training and testing sets 
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
linear_regressor_obj = LinearRegression()
linear_regressor_obj.fit(X_train, y_train)


# Predicting the Testing set results
y_predict = linear_regressor_obj.predict(X_test)
