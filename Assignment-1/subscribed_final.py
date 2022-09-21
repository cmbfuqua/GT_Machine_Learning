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
from sklearn.metrics import plot_confusion_matrix as plot_cm
from sklearn.metrics import plot_precision_recall_curve as plot_prc
from sklearn.metrics import plot_roc_curve as plot_roc

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

yes = cleanT1.loc[cleanT1.subscribed == 1]
no = cleanT1.loc[cleanT1.subscribed == 0].head(3394) #Balance Data

cleanT1 = pd.concat([yes,no],axis = 0)
#%%
decision_tree1 = dt(criterion = 'gini',splitter = 'best',max_depth = 4)
MLP1 = mlp(activation = 'logistic',solver = 'adam',hidden_layer_sizes = 10,max_iter = 650)
KNN1 = knn(n_neighbors = 2, algorithm = 'ball_tree',metric = 'euclidean')
boost1 = xgboost(loss = 'log_loss',learning_rate=.1,n_estimators=6)
svm1 = svc(kernel = 'poly',probability=True,degree=1)
#%%
X1 = cleanT1.drop(columns= ['subscribed'])
Y1 = cleanT1.subscribed

x_train1,x_test1,y_train1,y_test1 = train_test_split(X1,
                                                Y1,
                                                test_size = .2,
                                                random_state=42)

scaler = ss().fit(x_train1)

x_train1 = scaler.transform(x_train1)
x_test1 = scaler.transform(x_test1)

mins = []
means = []
maxs = []
stds = []

models = [decision_tree1,MLP1,KNN1,boost1,svm1]
models_names = ['DecisionTree1','MLP1','KNN1','boost1','SVM1']
for i in range(len(models)):
    cvscores = cross_val_score(models[i]
                              ,x_train1
                              ,y_train1
                              ,cv = 10
                              ,scoring = 'precision')
    models[i].fit(x_train1,y_train1)
    
    #print('before removing 0s \nmin: {}  max:{} \n'.format(cvscores.min(),cvscores.max()))

    cvscores = cvscores[np.greater_equal(cvscores,.01)] # This gets rid of the 0s

    print("{}: {} precision with a std of {}".format(models_names[i],cvscores.mean().round(2), cvscores.std().round(4)))
    print('min: {}  max:{} \n'.format(cvscores.min().round(2),cvscores.max().round(2)))

    mins.append(cvscores.min().round(3))
    means.append(cvscores.mean().round(3))
    maxs.append(cvscores.max().round(3))
    stds.append(cvscores.std().round(3))
#%%
scores = pd.DataFrame({'model':models_names,
                       'min':mins,
                       'mean':means,
                       'max':maxs,
                       'std':stds})
scores.to_csv('final_model_precision_validation_scores1.csv',index = False)
scores.head()
# %%
best_MetPar = pd.read_csv('best_metrics_parameters.csv')
#%%
# Decision Tree
plot_cm(decision_tree1,x_test1,y_test1)
plot_roc(decision_tree1,x_test1,y_test1)
plot_prc(decision_tree1,x_test1,y_test1)
#%%
# Decision Tree
plot_cm(boost1,x_test1,y_test1)
plot_roc(boost1,x_test1,y_test1)
plot_prc(boost1,x_test1,y_test1)

#%%
# Decision Tree
plot_cm(MLP1,x_test1,y_test1)
plot_roc(MLP1,x_test1,y_test1)
plot_prc(MLP1,x_test1,y_test1)
#%%
# Decision Tree
plot_cm(svm1,x_test1,y_test1)
plot_roc(svm1,x_test1,y_test1)
plot_prc(svm1,x_test1,y_test1)
#%%
# Decision Tree
plot_cm(KNN1,x_test1,y_test1)
plot_roc(KNN1,x_test1,y_test1)
plot_prc(KNN1,x_test1,y_test1)
#plot_roc(decision_tree1,X1,Y1)
# %%
best = best_MetPar
best.loc[(best.metric_name == 'time') & (best.model.str.endswith('1'))]
#%%
best.loc[(best.metric_name == 'precision') & (best.model.str.endswith('1'))].sort_values(by = 'metric',ascending = False)
# %%
