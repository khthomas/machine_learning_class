#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:34:08 2019

@author: kyle

Following an example from Kaggle. 
Application of One Class SVM to detect anomaly
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Read in the data
df_credit = pd.read_csv("creditcard.csv")

# This data 
df_credit.columns
# Time - time of transaction
# V1 - V28 - Principle components
# Amount - transaction amount
# Class - fraudulent transactions (1 = fraud)

# What percent of the data is fraud?
# We can see that 492 of the 284,315 transactions are fraud
# less than .2 percent of the data is fraud
df_credit['Class'].value_counts()
len(df_credit[df_credit['Class'] == 1]) / len(df_credit[df_credit['Class'] == 0])

# Lets make training data
# lets say from historical pusposes we know that fradulent data makes up 0.3 
# percent of fradulent claims
#X = df_credit.drop('Class', 1)
X = df_credit[["V1", "V2", "V3", "V4", "V5"]]
y = df_credit['Class']
true_fraud = df_credit[df_credit['Class'] == 1]
normal_act = df_credit[df_credit['Class'] == 0]

# Since this is an unsupervised approach we do not need labels. Also, since this
# method is bounding "normal data" we do not need to feed in a balanced amount 
# of fraudulent and non-fradulent data.
X_train2 = X.sample(n=100000)
X_train3 = X.sample(1000)

fraud_sample = true_fraud[["V1", "V2", "V3", "V4", "V5", "Class"]].sample(10)
normal_sample = normal_act[["V1", "V2", "V3", "V4", "V5", "Class"]].sample(10)

testing_data = pd.concat([fraud_sample, normal_sample])
testing_class = testing_data.pop('Class')
testing_data
testing_class


# Make the SVDD
historical_fraud_rate = 0.05
svdd = svm.OneClassSVM(kernel='rbf', nu=historical_fraud_rate)
# Takes some time to trian
svdd.fit(X_train2)

svdd_preds = svdd.predict(testing_data)
unique, counts = np.unique(svdd_preds, return_counts=True)
print(np.asarray((unique, counts)).T)

# Predicitons: -1 = Outlier, 1 = Normal
# in the example I just ran I got 9 outliers and 11 


# SVDD Example 3
# Since most of our data is normal we do not need a huge training set like
# I just demoed. In fact I can use only a small protion of the data and get 
# very good results as well. 
svdd2 = svm.OneClassSVM(kernel='rbf', nu=0.05)
svdd2.fit(X_train3)

svdd2_preds = svdd2.predict(testing_data)
unique2, counts2 = np.unique(svdd2_preds, return_counts=True)
print(np.asarray((unique2, counts2)).T)