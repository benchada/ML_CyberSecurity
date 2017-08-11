#This file reads the logs activity from a file or a DB. 
#This data is used in order to train a machine learning model,
# in order to predict the action taken for future activities (Allowed Vs. Blocked)

#Libraries Import
from __future__ import print_function
import os
import pandas as pd
import numpy as np
import sklearn


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Reading the logs activity
data_path = ['zscaler_logs']
filepath = os.sep.join(data_path + ['zscaler_logs.csv'])
data = pd.read_csv(filepath, sep=',')

#Dropping all empty columns
cleaned_data = data.dropna(axis ='columns', how='all')

#Resizing the data to a smaller set
sample_data = cleaned_data.head(100000)

for col in sample_data.columns.values.tolist():
    if (len (sample_data[col].value_counts()) == 1) :
        sample_data.drop(col, axis=1, inplace=True)
#len (sample_data.columns) #Now we only have 59 columns left

cleaned_sample_data = sample_data

#_________Preparing variables for the model____________
#Separating the feature columns from the target columns
feature_cols =  ['bytes', 'bytes_in', 'bytes_out',  'ctime', 'reqsize', 'respsize', 'riskscore',  'stime', 'url_length']

# Split the data into two parts with 1500 points in the test data
strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=20000, random_state=42) # This creates a generator

# Get the index values from the generator
train_idx, test_idx = next(strat_shuff_split.split(cleaned_sample_data[feature_cols], cleaned_sample_data['action']))

# Create the data sets for training & testing
X_train = cleaned_sample_data.loc[train_idx, feature_cols]
y_train = cleaned_sample_data.loc[train_idx, 'action']

X_test = cleaned_sample_data.loc[test_idx, feature_cols]
y_test = cleaned_sample_data.loc[test_idx, 'action']

#Trying with the KNN model for classification
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print(accuracy_score(y_test, knn_pred))