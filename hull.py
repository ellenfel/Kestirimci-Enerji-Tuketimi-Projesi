#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

import folium
from folium.plugins import HeatMap
import plotly.express as px

# plotting configurations
plt.style.use('fivethirtyeight')
%matplotlib inline
pd.set_option('display.max_columns', 32)

# reading data
df = pd.read_csv('../data/sample-set-energy-consumption-forecast-noUG-67.csv',
                 on_bad_lines='skip', sep='|',names=["id", "device-name", "key", "ts", "value"])
"""
column_index_to_rename=0
df.rename(columns={df.columns[column_index_to_rename]: '111'}, inplace=True)
df[['device_name', 'key', 'ts', 'telemetry']] = df['111'].str.split('|', expand=True)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.drop('111', axis=1, inplace=True)
df = df[df[df.columns[0]] != 'UG-67'] #Bidakine sqlde yap

"""
#df.to_csv('../data/newest-clean-data.csv', index=False)
df.drop('id', axis=1, inplace=True)


unique_values = df['key'].unique()
unique_values = [value for value in unique_values if 'error' not in str(value)]

df = df[~df['key'].str.contains('error', case=False, na=False)]


# Assuming the column name is 'key'
column_name_to_check = 'key'

# List of values to exclude
values_to_exclude = ['devName', 'rssi',  'devEUI', 
                      'snr', 'Active', 'time', 'Asset', 'delta', 'currentDelta_yss', 'currentDelta_ei',
                      'load', 'maneuver', 'a', 
                      'state', 'currentDelta_ecs', 'hardware_version', 'ipso_version', 'power', 'sn',
                      'software_version']

# Filter out rows where the value in the 'key' column is not in the list of values to exclude
df = df[~df[column_name_to_check].isin(values_to_exclude)]
df['value'] = df['value'].astype('float64')


#for sampling
#df_sample = df.head(10000)
#unique_values = df_sample['key'].unique()


#dummy encoding
df = pd.concat([df, pd.get_dummies(df['key'].str.lower())], axis=1)
df = df.fillna(0)

# Assign the values of the 'value' column to the corresponding columns
for value in unique_values:
    df[value.lower()] = df.apply(lambda row: row['value'] if row['key'].lower() == value.lower() else 0, axis=1)


# Columns that include the word 'Active'
active_columns = df.filter(like='active')

# Create a new column 'Active Import' with the maximum value across 'Active' columns
df['active import'] = active_columns.max(axis=1, skipna=True)

columns_to_drop = df.filter(like='Active').columns.to_list() + [
    'devName', 'rssi', 'devEUI', 'snr', 'time', 'Asset', 'delta', 'currentDelta_yss',
    'currentDelta_ei', 'load', 'maneuver', 'a', 'state', 'currentDelta_ecs',
    'hardware_version', 'ipso_version', 'power', 'sn', 'software_version'
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Drop the original 'key' and 'value' columns
df = df.drop(['key', 'value'], axis=1)


#merging
merged_df = df.groupby(['device-name', 'ts'], as_index=False).agg(lambda x: x.max() if x.dtype == 'float64' else x.max())
df = merged_df


df = df[df['active energy'].notna() & (df['active energy'] != 0)]

df.to_csv('../data/newest-clean-merged-data.csv', index=False)


#data analysis
df.info()





X = df.drop(columns=['active energy'], errors='ignore')
y = df['active energy']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


def evaluate_model(model, X_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


###  Decision Tree model ###

from sklearn import tree

def sklearn_eval(X_train,y_train):
    dtc = tree.DecisionTreeClassifier(random_state=0)
    dtc.fit(X_train, y_train)
    
    # Evaluate Model
    dtc_eval = evaluate_model(dtc, X_test, y_test)
    
    # Print result
    print('Accuracy:', dtc_eval['acc'])
    print('Precision:', dtc_eval['prec'])
    print('Recall:', dtc_eval['rec'])
    print('F1 Score:', dtc_eval['f1'])
    print('Cohens Kappa Score:', dtc_eval['kappa'])
    print('Area Under Curve:', dtc_eval['auc'])
    print('Confusion Matrix:\n', dtc_eval['cm'])

################################################

### Randdom Forest ###

from sklearn.ensemble import RandomForestClassifier

def randomforest_eval(X_train,y_train):
    rf_clf = RandomForestClassifier(criterion='entropy')   
    rf_clf.fit(X_train,y_train)
    
    rf_eval = evaluate_model(rf_clf, X_test, y_test)
    print('Accuracy:', rf_eval['acc'])
    print('Precision:', rf_eval['prec'])
    print('Recall:', rf_eval['rec'])
    print('F1 Score:', rf_eval['f1'])
    print('Cohens Kappa Score:', rf_eval['kappa'])
    print('Area Under Curve:', rf_eval['auc'])
    print('Confusion Matrix:\n', rf_eval['cm'])

################################################

### Naive Bayes ###

from sklearn.naive_bayes import GaussianNB

def naivebayes_eval(X_train,y_train):
    #Calling the Class
    naive_bayes = GaussianNB()
     
    #Fitting the data to the classifier
    naive_bayes.fit(X_train , y_train)
     
    #Predict on test data
    y_pred = naive_bayes.predict(X_test)
    naive_eval = evaluate_model(naive_bayes, X_test, y_test)
    print('Accuracy:', naive_eval['acc'])
    print('Precision:', naive_eval['prec'])
    print('Recall:', naive_eval['rec'])
    print('F1 Score:', naive_eval['f1'])
    print('Cohens Kappa Score:', naive_eval['kappa'])
    print('Area Under Curve:', naive_eval['auc'])
    print('Confusion Matrix:\n', naive_eval['cm'])

################################################

### Ada Boost ###

from sklearn.ensemble import AdaBoostClassifier

def adaboost(X_train,y_train):
    abc = AdaBoostClassifier(n_estimators=50,
             learning_rate=1)
    ada_boost = abc.fit(X_train, y_train)
    y_pred = ada_boost.predict(X_test)
    ada_beval = evaluate_model(ada_boost, X_test, y_test)
    print('Accuracy:', ada_beval['acc'])
    print('Precision:', ada_beval['prec'])
    print('Recall:', ada_beval['rec'])
    print('F1 Score:', ada_beval['f1'])
    print('Cohens Kappa Score:', ada_beval['kappa'])
    print('Area Under Curve:', ada_beval['auc'])
    print('Confusion Matrix:\n', ada_beval['cm'])


# acc of models
sklearn_eval(X_train,y_train)
randomforest_eval(X_train,y_train)
naivebayes_eval(X_train,y_train)
adaboost(X_train,y_train)

plt.figure(figsize = (24, 12))

corr = df.corr()
sns.heatmap(corr, annot = True, linewidths = 1)
plt.show()

correlation = df.corr()['active import'].abs().sort_values(ascending = False)
correlation