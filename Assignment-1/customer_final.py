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

yes = cleanT.loc[cleanT.churn == 1]
no = cleanT.loc[cleanT.churn == 0].head(2037) #Balance Data

cleanT = pd.concat([yes,no],axis = 0)
#%%
decision_tree2 = dt(criterion = 'entropy',splitter = 'random',max_depth = 6)
MLP2 = mlp(activation = 'relu',solver = 'sgd',hidden_layer_sizes = 2,max_iter = 500)
KNN2 = knn(n_neighbors = 250, algorithm = 'ball_tree',metric = 'manhattan')
boost2 = xgboost(loss = 'exponential',learning_rate=.1,n_estimators=10)

#%%
X = cleanT.drop(columns= ['churn'])
Y = cleanT.churn

x_train,x_test,y_train,y_test = train_test_split(X,
                                                Y,
                                                test_size = .2,
                                                random_state=42)

scaler = ss().fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

mins = []
means = []
maxs = []
stds = []

models = [decision_tree2,MLP2,KNN2,boost2]
models_names = ['decision_tree2','MLP2','KNN2','boost2']
for i in range(len(models)):
    cvscores = cross_val_score(models[i]
                              ,x_train
                              ,y_train
                              ,cv = 10
                              ,scoring = 'precision')
    
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
scores.to_csv('final_model_precision_validation_scores2.csv')
scores.head()
# %%
