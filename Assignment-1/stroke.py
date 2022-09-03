#%%
import pandas as pd
import altair as alt 
import numpy as np 

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt 
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.ensemble import GradientBoostingClassifier as xgboost
from sklearn.svm import SVC as svc

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import OneHotEncoder

# %%
data = pd.read_csv('full_data_stroke.csv')
data.head()
# %%
clean = pd.DataFrame(data[['hypertension','heart_disease','avg_glucose_level','bmi','stroke','smoking_status']])
#%%
## Clean up age because it has decimal ages
data.loc[data.age < 1,'age'] = 0
#%%
## one hot encode gender(2 values), work_type(4 values),ever_married(2 values) and residence_type(2)
clean['isMale'] = pd.get_dummies(data.gender,drop_first=True)
clean['beenMarried'] = pd.get_dummies(data.ever_married,drop_first = True)
clean['isUrban'] = pd.get_dummies(data.Residence_type, drop_first = True)
clean['isPrivate'] = pd.get_dummies(data.work_type,drop_first = True)['Private']
clean['isSelf'] = pd.get_dummies(data.work_type,drop_first = True)['Self-employed']
clean['isChildren'] = pd.get_dummies(data.work_type,drop_first = True)['children']

clean['has_smoked'] = 0
clean.loc[clean.smoking_status.isin(['formerly smoked','smokes']),'has_smoked']=1
clean['not_currently_smoking'] = 0
clean.loc[clean.smoking_status.isin(['formerly_smoked','never smoked']),'not_currently_smoking']=1
## Data is severly unbalanced so we need to balance the target variable (248 had strokes 4733 didn't have strokes)
strokes = clean.loc[data.stroke == 1]
nostroke = clean.loc[data.stroke == 0].head(248)

clean = pd.concat([strokes,nostroke])
clean.head()
# %%
X = clean.drop(columns = 'stroke')
Y = clean.stroke
# %%
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = .2,random_state=42)

# %%
# Decision Tree
tree_model = dt()