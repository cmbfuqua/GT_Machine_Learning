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
dfsname = ['b1','b2','d1','d2','k1','k2','m1','m2']

for i in range(len(dfs)):
    dfs[i] = dfs[i].drop(columns = 'Unnamed: 0')
# %%
#######################################3
# First Dataset
#######################################3
for i in range(len(dfs)):
    temp = dfs[i].loc[dfs[i].precision.isin([dfs[i].precision.max()])].head(1)
    print('{} \n {}'.format(dfsname[i], temp.drop(columns = ['time','accuracy','recall']).head(1)))
# %%
'''
Decision Tree D1
crit = entropy split = best depth = 14

Decision Tree D2
crit = gini split = best depth = 20

MLP D1
activation = relu solver = adam layers = 2 iterations = 350
MLP D2

KNN D1

KNN D2

Boost D1

Boost D2
'''