#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
import numpy as np
import pandas as pd


X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')


train_features = np.array(X_train)

train_labels= np.array(y_train)

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=15,
           forward=False,
           verbose=2,
           scoring='roc_auc',
           cv=4)

features1 = feature_selector.fit(train_features, train_labels.ravel())
filtered_features= [train_features.columns[list(features1.k_feature_idx_)]]

with open("forward.txt", "w") as output:
    output.write(str(filtered_features))
    
    

    


feature_selector2 = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=15,
           forward=False,
           verbose=2,
           scoring='roc_auc',
           cv=4)

features2 = feature_selector2.fit(train_features, train_labels.ravel())
filtered_features2= [train_features.columns[list(features2.k_feature_idx_)]]

with open("backward.txt", "w") as output:
    output.write(str(filtered_features2))

