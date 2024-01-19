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
values_to_exclude = ['devName', 'rssi', 'Active Enegy-CL', 'Active Enegy-CH', 'devEUI', 'Active Enegy-Ex',
                      'snr', 'Active', 'time', 'Asset', 'delta', 'currentDelta_yss', 'currentDelta_ei',
                      'load', 'maneuver', 'Active Energy-Import', 'a', 'Active Enegy-DL', 'Active Enegy-DH',
                      'state', 'currentDelta_ecs', 'hardware_version', 'ipso_version', 'power', 'sn',
                      'software_version','Active Enegy-Im','Active Energy-DL','Active Energy-DH']

# Filter out rows where the value in the 'key' column is not in the list of values to exclude
df = df[~df[column_name_to_check].isin(values_to_exclude)]
df['value'] = df['value'].astype('float64')


#for sampling
#df_sample = df.head(10000)
#unique_values = df_sample['key'].unique()



df = pd.concat([df, pd.get_dummies(df['key'].str.lower())], axis=1)
df = df.fillna(0)

# Assign the values of the 'value' column to the corresponding columns
for value in unique_values:
    df[value.lower()] = df.apply(lambda row: row['value'] if row['key'].lower() == value.lower() else 0, axis=1)

# Drop the original 'key' and 'value' columns
df = df.drop(['key', 'value'], axis=1)



merged_df = df.groupby(['device-name', 'ts'], as_index=False).agg(lambda x: x.max() if x.dtype == 'float64' else x.max())
merged_df = df.groupby('device-name', as_index=False).max()

merged_df.to_csv('../data/newest-clean-merged-data.csv', index=False)