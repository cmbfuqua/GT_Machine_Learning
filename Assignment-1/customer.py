#%%
# basic packages
import pandas as pd
import altair as alt 
import numpy as np 
# models
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt 
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.ensemble import GradientBoostingClassifier as xgboost
from sklearn.svm import SVC as svc
# cleaning
from sklearn.model_selection import train_test_split
# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
# %%
data = pd.read_csv('Bank Customer Churn Prediction.csv')
data.head()
# %%
import pandas_profiling as pr
pr.ProfileReport(data)
# %%
# one hot encode country
# one hot encode gender
