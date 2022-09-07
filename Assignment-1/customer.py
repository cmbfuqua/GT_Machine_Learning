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
# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# %%
data = pd.read_csv('Bank Customer Churn Prediction.csv')
data.head()
# %%
import pandas_profiling as pr
pr.ProfileReport(data)
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
X = cleanT.drop(columns = 'churn')
Y = cleanT.churn
#%%
x_train,x_test,y_train,y_test = train_test_split(X,Y,
                                                 test_size=.2,
                                                 random_state=43)

scaler = ss().fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# %%
# Decision Tree
import time
critdf = []
splitdf = []
depthdf = []
timedf = []
accuracy = []
precision = []
recall = []
crit = ['gini','entropy']
spli = ['best','random']
dep = [1,4,6,8,10,12,14,16,18,20]

for c in crit:
    for s in spli:
        for d in dep:
            startt = time.time()
            model_dt = dt(criterion=c
                        ,splitter = s
                        ,max_depth=d
                        ,random_state=42)

            model_dt.fit(x_train,y_train)

            endt = time.time()
            ellapsed_time = endt - startt

            pred_dt = model_dt.predict(x_test)

            critdf.append(c)
            splitdf.append(s)
            depthdf.append(d)
            timedf.append(ellapsed_time)
            accuracy.append(accuracy_score(y_test,pred_dt))
            precision.append(precision_score(y_test,pred_dt))
            recall.append(recall_score(y_test,pred_dt))
# %%
results_dt = pd.DataFrame({'crit':critdf,
                           'split':splitdf,
                           'depth':depthdf,
                           'time':timedf,
                           'accuracy':accuracy,
                           'precision':precision,
                           'recall':recall})
results_dt.to_csv('dt_results_d2.csv')
# %%
# Neural Network or MLP in sklearn terms
import time
activationdf = []
solverdf = []
layersdf = []
itterdf = []
timedf = []
accuracy = []
precision = []
recall = []
layers = [2,4,6,8,10]
solver = ['lbfgs','sgd','adam']
activation = ['identity','logistic','tanh','relu']
itter = [500,1000,1500,2000]
count = 0
for l in layers:
    for s in solver:
        for a in activation:
            for i in itter:
                print('row: {}'.format(count))
                startt = time.time()
                model_mlp = mlp(hidden_layer_sizes=l,
                                activation= a,
                                solver= s,
                                max_iter=i)

                model_mlp.fit(x_train,y_train)

                endt = time.time()
                ellapsed_time = endt - startt

                pred = model_dt.predict(x_test)

                activationdf.append(a)
                solverdf.append(s)
                layersdf.append(l)
                itterdf.append(i)
                timedf.append(ellapsed_time)
                accuracy.append(accuracy_score(y_test,pred))
                precision.append(precision_score(y_test,pred))
                recall.append(recall_score(y_test,pred))
                print('\n\n')

                count = count + 1
# %%
results_mlp = pd.DataFrame({'activation':activationdf,
                           'solver':solverdf,
                           'layers':layersdf,
                           'iterations':itterdf,
                           'time':timedf,
                           'accuracy':accuracy,
                           'precision':precision,
                           'recall':recall})
#%%
results_mlp.to_csv('mlp_results_d2.csv')
# %%
# knn
import time
timedf = []
accuracydf = []
precisiondf = []
recalldf = []


neighborsdf = []
algorithmdf = []
metricdf = []

neighbors = [10,50,100,250,400,500]
algorithm = ['ball_tree','kd_tree','brute','auto']
metric = ['euclidean','manhattan']

for n in neighbors:
    for a in algorithm:
        for m in metric:
            print('moving on')
            startt = time.time()
            model_knn = knn(n_neighbors=n,
                            algorithm=a,
                            metric=m)

            model_knn.fit(x_train,y_train)

            endt = time.time()
            ellapsed_time = endt - startt

            pred = model_knn.predict(x_test)

            timedf.append(ellapsed_time)
            accuracydf.append(accuracy_score(y_test,pred))
            precisiondf.append(precision_score(y_test,pred))
            recalldf.append(recall_score(y_test,pred))

            neighborsdf.append(n)
            algorithmdf.append(a)
            metricdf.append(m)
# %%
results_knn = pd.DataFrame({'neighbors':neighborsdf,
                           'algorithm':algorithmdf,
                           'metric':metricdf,
                           'time':timedf,
                           'accuracy':accuracydf,
                           'precision':precisiondf,
                           'recall':recalldf})
#%%
results_knn.to_csv('knn_results_d2.csv')