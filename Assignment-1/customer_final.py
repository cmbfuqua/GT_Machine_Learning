# basic packages
from configparser import MAX_INTERPOLATION_DEPTH
import pandas as pd
import altair as alt 
alt.data_transformers.disable_max_rows()
import numpy as np 
# models
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt 
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.ensemble import GradientBoostingClassifier as xgboost
from sklearn.svm import SVC as svc
# cleaning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# %%
data = pd.read_csv('Bank Customer Churn Prediction.csv')
data.head()
# %%
#import pandas_profiling as pr
#pr.ProfileReport(data)
# %%
clean = data[['credit_score','age','tenure',
              'balance','credit_card','active_member',
              'estimated_salary','churn']]
# one hot encode country
countries = pd.get_dummies(data.country,drop_first=True)
# one hot encode gender
clean['gender'] = pd.get_dummies(data.gender,drop_first=True)
# one hot encode the 4 products
products = pd.get_dummies(data.products_number,drop_first=True,prefix='product')

cleanT = pd.concat([clean,countries,products],axis = 1)
#%%
decision_tree2 = dt(crit = 'entropy',split = 'random',depth = 6)
MLP2 = mlp(activation = 'relu',solver = 'sgd',layers = 2,iter = 400)
KNN2 = knn(n_neighbors = 250, algorithm = 'ball_tree',metric = 'manhattan')
boost2 = xgboost(loss = 'exponential',learning_rate=.1,n_estimators=10)