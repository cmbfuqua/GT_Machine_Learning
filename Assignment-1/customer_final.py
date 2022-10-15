#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best")

    return plt




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
decision_tree2 = dt(criterion = 'gini',splitter = 'best',max_depth = 2)
MLP2 = mlp(activation = 'logistic',max_iter = 600,solver = 'sgd',hidden_layer_sizes = 2)
KNN2 = knn(n_neighbors = 4, algorithm = 'ball_tree',metric = 'euclidean')
boost2 = xgboost(loss = 'log_loss',learning_rate=.1,n_estimators=6)
svm2 = svc(kernel = 'poly',probability=True,degree=10)

#%%
X = cleanT.drop(columns= ['churn'])
Y = cleanT.churn

scaler = ss().fit(X)
#%%
plot_learning_curve(decision_tree2,title = 'DT',X = X, y = Y)
#%%
plot_learning_curve(MLP2,title = 'MLP',X = X, y = Y)
#%%
plot_learning_curve(KNN2,title = 'KNN',X = X, y = Y)
#%%
plot_learning_curve(boost2,title = 'Boost',X = X, y = Y)
#%%
plot_learning_curve(svm2,title = 'SVM',X = X, y = Y)
#%%


#%%
X = cleanT.drop(columns= ['churn'])
Y = cleanT.churn

x_train,x_test,y_train,y_test = train_test_split(X,
                                                Y,
                                                test_size = .2,
                                                random_state=43)

scaler = ss().fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

mins = []
means = []
maxs = []
stds = []

models = [decision_tree2,MLP2,KNN2,boost2,svm2]
models_names = ['decision_tree2','MLP2','KNN2','boost2','SVM2']
for i in range(len(models)):
    cvscores = cross_val_score(models[i]
                              ,x_train
                              ,y_train
                              ,cv = 10
                              ,scoring = 'precision')
    
    #print('before removing 0s \nmin: {}  max:{} \n'.format(cvscores.min(),cvscores.max()))
    models[i].fit(x_train,y_train)
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
scores.to_csv('final_model_precision_validation_scores2.csv',index = False)
scores.head()

# %%
best_MetPar = pd.read_csv('best_metrics_parameters.csv')
data2_best = best_MetPar.loc[(best_MetPar.metric_name == 'precision') & (best_MetPar.model.str.endswith('2') )].sort_values(by = 'metric',ascending=False)

#%%
# Decision Tree
plot_cm(decision_tree2,x_test,y_test)
plot_roc(decision_tree2,x_test,y_test)
#plot_prc(decision_tree2,x_test,y_test)
#%%
# Decision Tree
plot_cm(boost2,x_test,y_test)
plot_roc(boost2,x_test,y_test)
#plot_prc(boost2,x_test,y_test)

#%%
# Decision Tree
plot_cm(MLP2,x_test,y_test)
plot_roc(MLP2,x_test,y_test)
#plot_prc(MLP2,x_test,y_test)
#%%
# Decision Tree
plot_cm(svm2,x_test,y_test)
plot_roc(svm2,x_test,y_test)
#plot_prc(svm2,x_test,y_test)
#%%
# Decision Tree
plot_cm(KNN2,x_test,y_test)
plot_roc(KNN2,x_test,y_test)
#plot_prc(KNN2,x_test,y_test)
#plot_roc(decision_tree1,X1,Y1)
# %%
best = best_MetPar
best.loc[(best.metric_name == 'time') & (best.model.str.endswith('2'))]
#%%
best.loc[(best.metric_name == 'accuracy') & (best.model.str.endswith('2'))].sort_values(by = 'metric',ascending = False)

# %%
