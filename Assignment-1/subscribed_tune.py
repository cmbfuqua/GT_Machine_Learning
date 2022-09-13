#%%
#plot_confusion_matrix(model_dt,x_train_dt,y_train_dt)
#plot_confusion_matrix(model_dt,x_test_dt,y_test_dt)

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
# %%
X1 = cleanT1.drop(columns = ['subscribed'])
Y1 = cleanT1.subscribed
# %%
x_train1,x_test1,y_train1,y_test1 = train_test_split(X1,
                                                Y1,
                                                test_size = .2,
                                                random_state=42)

scaler = ss().fit(x_train1)

x_train1 = scaler.transform(x_train1)
x_test1 = scaler.transform(x_test1)

# %%
########################################################
# Decision Tree
########################################################
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
dep = [1,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

for c in crit:
    for s in spli:
        for d in dep:
            startt = time.time()
            model_dt = dt(criterion=c
                        ,splitter = s
                        ,max_depth=d
                        ,random_state=42)

            model_dt.fit(x_train1,y_train1)

            endt = time.time()
            ellapsed_time = endt - startt

            pred_dt = model_dt.predict(x_test1)

            critdf.append(c)
            splitdf.append(s)
            depthdf.append(d)
            timedf.append(ellapsed_time)
            accuracy.append(accuracy_score(y_test1,pred_dt))
            precision.append(precision_score(y_test1,pred_dt))
            recall.append(recall_score(y_test1,pred_dt))
# %%
results_dt = pd.DataFrame({'crit':critdf,
                           'split':splitdf,
                           'depth':depthdf,
                           'time':timedf,
                           'accuracy':accuracy,
                           'precision':precision,
                           'recall':recall})
results_dt.to_csv('dt_results_d1.csv')

# %%
########################################################
# Neural Network or MLP in sklearn terms
########################################################
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
solver = ['sgd','adam']
activation = ['identity','logistic','relu']
itter = [200,250,300,350,400,450,500,550,600,650,700]
c= 0
for l in layers:
    for s in solver:
        for a in activation:
            for i in itter:
                print('{} out of 150'.format(c))
                c = c + 1
                startt = time.time()
                model_mlp = mlp(hidden_layer_sizes=l,
                                activation= a,
                                solver= s,
                                max_iter=i)

                model_mlp.fit(x_train1,y_train1)

                endt = time.time()
                ellapsed_time = endt - startt

                pred = model_mlp.predict(x_test1)

                activationdf.append(a)
                solverdf.append(s)
                layersdf.append(l)
                itterdf.append(i)
                timedf.append(ellapsed_time)
                accuracy.append(accuracy_score(y_test1,pred))
                precision.append(precision_score(y_test1,pred))
                recall.append(recall_score(y_test1,pred))
# %%
results_mlp = pd.DataFrame({'activation':activationdf,
                           'solver':solverdf,
                           'layers':layersdf,
                           'iterations':itterdf,
                           'time':timedf,
                           'accuracy':accuracy,
                           'precision':precision,
                           'recall':recall})
results_mlp.to_csv('mlp_results_d1.csv')
# %%
########################################################
# knn
########################################################
import time
timedf = []
accuracydf = []
precisiondf = []
recalldf = []


neighborsdf = []
algorithmdf = []
metricdf = []

neighbors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,50,75,100,150,200,250]
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

            model_knn.fit(x_train1,y_train1)

            endt = time.time()
            ellapsed_time = endt - startt

            pred = model_knn.predict(x_test1)

            timedf.append(ellapsed_time)
            accuracydf.append(accuracy_score(y_test1,pred))
            precisiondf.append(precision_score(y_test1,pred))
            recalldf.append(recall_score(y_test1,pred))

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

results_knn.to_csv('knn_results_d1.csv')
# %%
########################################################
# xgboost
########################################################
import time
timedf = []
accuracydf = []
precisiondf = []
recalldf = []


loss = ['log_loss','deviance','exponential']
learn_rate = [.1,.5,1,3]
n_est = [10,20,30,40,50,75,100,125,150,200,250,300,350,400,500]

lossdf = []
learn_ratedf = []
n_estdf = []

for l in loss:
    for lr in learn_rate:
        for ne in n_est:
            print('moving on')
            startt = time.time()
            model_boost = xgboost(
                                    loss=l,
                                    learning_rate=lr,
                                    n_estimators=ne
                                )

            model_boost.fit(x_train1,y_train1)

            endt = time.time()
            ellapsed_time = endt - startt

            pred = model_boost.predict(x_test1)

            timedf.append(ellapsed_time)
            accuracydf.append(accuracy_score(y_test1,pred))
            precisiondf.append(precision_score(y_test1,pred))
            recalldf.append(recall_score(y_test1,pred))

            lossdf.append(l)
            learn_ratedf.append(lr)
            n_estdf.append(ne)
# %%
results_boost = pd.DataFrame({'loss':lossdf,
                           'learning_rate':learn_ratedf,
                           'n_estimators':n_estdf,
                           'time':timedf,
                           'accuracy':accuracydf,
                           'precision':precisiondf,
                           'recall':recalldf})

results_boost.to_csv('boost_results_d1.csv')
# %%
########################################################
# SVM
########################################################
import time
timedf = []
accuracydf = []
precisiondf = []
recalldf = []


kernel = ['linear','poly','rbf','sigmoid']
probability = [True, False]

Kerneldf = []
probabilitydf = []


for k in kernel:
    for p in probability:
        print('moving on')
        startt = time.time()
        model_svm = svc(kernel = k, probability = p)

        model_svm.fit(x_train1,y_train1)

        endt = time.time()
        ellapsed_time = endt - startt

        pred = model_svm.predict(x_test1)

        timedf.append(ellapsed_time)
        accuracydf.append(accuracy_score(y_test1,pred))
        precisiondf.append(precision_score(y_test1,pred))
        recalldf.append(recall_score(y_test1,pred))

        Kerneldf.append(k)
        probabilitydf.append(str(p))
# %%
results_svm = pd.DataFrame({'kernel':Kerneldf,
                            'probability':probabilitydf,
                           'time':timedf,
                           'accuracy':accuracydf,
                           'precision':precisiondf,
                           'recall':recalldf})

results_svm.to_csv('svm_results_d1.csv')
# %%
