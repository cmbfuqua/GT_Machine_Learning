#%%
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
from sklearn.model_selection import cross_val_score
# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# %%
data1 = pd.read_csv('subscribed.csv')
data1.head()
# %%
data1.loc[data1.marital.isna(),'marital'] = 'Unknown'
data1.loc[data1.customer_age.isna(),'customer_age'] = data1.customer_age.mean()
data1.loc[data1.balance.isna(),'balance'] = data1.balance.mean()
data1.loc[data1.personal_loan.isna(),'personal_loan'] = 'no'
data1.loc[data1.last_contact_duration.isna(),'last_contact_duration'] = data1.last_contact_duration.mean()
data1.loc[data1.num_contacts_in_campaign.isna(),'num_contacts_in_campaign'] = data1.num_contacts_in_campaign.mean()

clean1 = pd.DataFrame(data1[['customer_age',
                    'balance',
                    'last_contact_duration',
                    'num_contacts_in_campaign',
                    'num_contacts_prev_campaign']])
clean1['subscribed'] = data1.term_deposit_subscribed
#%%
marital_status = pd.get_dummies(data1.marital,drop_first = True,prefix='married')
jobs = pd.get_dummies(data1.job_type,drop_first = True,prefix='jt')
edu = pd.get_dummies(data1.education,drop_first=True,prefix='edu')
clean1['default'] = pd.get_dummies(data1.default,drop_first=True)
clean1['housing_loan'] = pd.get_dummies(data1.housing_loan)['yes']
clean1['personal_loan'] = pd.get_dummies(data1.personal_loan,drop_first=True)
comm_type = pd.get_dummies(data1.communication_type,drop_first=True,prefix='comm_type')
outcomes = pd.get_dummies(data1.prev_campaign_outcome,drop_first=True,prefix='out')

cleanT1 = pd.concat([jobs,edu,comm_type,marital_status,clean1],axis = 1)
#%%
decision_tree1 = dt(crit = 'gini',split = 'best',depth = 4)
MLP1 = mlp(activation = 'logistic',solver = 'sgd',layers = 2,iter = 550)
KNN1 = knn(n_neighbors = 100, algorithm = 'ball_tree',metric = 'manhattan')
boost1 = xgboost(loss = 'exponential',learning_rate=.1,n_estimators=30)