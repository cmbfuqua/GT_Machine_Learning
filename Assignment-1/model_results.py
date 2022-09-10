#%%
import pandas as pd
import altair as alt 
alt.data_transformers.disable_max_rows()

# %%
b1 = pd.read_csv('boost_results_d1.csv')
b2 = pd.read_csv('boost_results_d2.csv')
d1 = pd.read_csv('dt_results_d1.csv')
d2 = pd.read_csv('dt_results_d2.csv')
k1 = pd.read_csv('knn_results_d1.csv')
k2 = pd.read_csv('knn_results_d2.csv')
m1 = pd.read_csv('mlp_results_d1.csv')
m2 = pd.read_csv('mlp_results_d2.csv')

dfs = [b1,b2,d1,d2,k1,k2,m1,m2]
dfsname = ['Boost1','Boost2','DecisionTree1','DecisionTree2','KNN1','KNN2','MLP1','MLP2']

for i in range(len(dfs)):
    dfs[i] = dfs[i].drop(columns = 'Unnamed: 0')
# %%
#######################################
# get best precision
#######################################
model = []
parameter = []
parameter_name = []
metric = []
metric_name = []

for val in ['time','accuracy','recall','precision']:
    for i in range(len(dfs)):
        if val == 'time':
            temp = dfs[i].loc[dfs[i][val].isin([dfs[i][val].min()])].head(1)
        else:
            temp = dfs[i].loc[dfs[i][val].isin([dfs[i][val].max()])].head(1)
        #print('{} \n {}'.format(dfsname[i], temp.drop(columns = ['time','accuracy','recall']).head(1)))
        for col in temp.columns:
            if col not in ['time','accuracy','precision','recall']:
                #print(temp)
                model.append(dfsname[i])
                metric.append(temp[val].reset_index(drop = True)[0])
                metric_name.append(val)
                print(temp[col])
                parameter.append(temp[col].reset_index(drop = True)[0])
                parameter_name.append(col)
            

best = pd.DataFrame({'model':model,
                     'parameter_name':parameter_name,
                     'parameter':parameter,
                     'metric_name':metric_name,
                     'metric':metric
                     })
best.head()
#%%
best.to_csv('best_metrics_parameters.csv')
# %%
'''
Optimizing for precision
Decision Tree D1
crit = gini split = best depth = 4

Decision Tree D2
crit = entropy split = random depth = 6

MLP D1
activation = logistic solver = sgd layers = 2 iterations = 550

MLP D2
activation = relu solver = sgd layers = 2 itterations = 400

KNN D1
neighbors = 100 algorithm = ball_tree metric = manhattan

KNN D2
neighbors = 250 algorithm = ball_tree metric = manhattan

Boost D1
loss = exponential learning_rate = .1 n_estimators = 30

Boost D2
loss = exponential learning_rate = .1 n_estimators = 10
'''

# %%
