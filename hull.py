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
df = pd.read_csv('../data/energy-consumption-forecast-all.csv', on_bad_lines='skip')

column_index_to_rename=0
df.rename(columns={df.columns[column_index_to_rename]: '111'}, inplace=True)
df[['device_name', 'key', 'ts', 'telemetry']] = df['111'].str.split('|', expand=True)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.drop('111', axis=1, inplace=True)

df = df[df[df.columns[0]] != 'UG-67'] #Bidakine sqlde yap
df.to_csv('../data/newest-clean-data.csv', index=False)



df = pd.read_csv('../data/newest-clean-data.csv')




















