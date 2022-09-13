#%% 
import pandas as pd 
import altair as alt 
alt.data_transformers.enable(maxrows = None)
# %%
# Dataframes containing the grid search values and the associated metrics
b1 = pd.read_csv('boost_results_d1.csv').drop(columns='Unnamed: 0')
b2 = pd.read_csv('boost_results_d2.csv').drop(columns='Unnamed: 0')
d1 = pd.read_csv('dt_results_d1.csv').drop(columns='Unnamed: 0')
d2 = pd.read_csv('dt_results_d2.csv').drop(columns='Unnamed: 0')
k1 = pd.read_csv('knn_results_d1.csv').drop(columns='Unnamed: 0')
k2 = pd.read_csv('knn_results_d2.csv').drop(columns='Unnamed: 0')
m1 = pd.read_csv('mlp_results_d1.csv').drop(columns='Unnamed: 0')
m2 = pd.read_csv('mlp_results_d2.csv').drop(columns='Unnamed: 0')
s1 = pd.read_csv('svm_results_d2.csv').drop(columns='Unnamed: 0')
s2 = pd.read_csv('svm_results_d2.csv').drop(columns='Unnamed: 0')

# Dataframe containing the model name and the best combination of metrics to
# obtain the best result of either time, precision, accuracy or recall
best_MetPar = pd.read_csv('best_metrics_parameters.csv').drop(columns='Unnamed: 0')

# Dataframes containing the further tuned svm models with the degree metric
# Note, this data was created on an arbitrary train_test_split and not with
# cross validation like unto the final_val{1,2} datasets. 
# This was in effort to see how the parameter performed due to the poly
# kernel being the best
svm1 = pd.read_csv('svm2_results_d1.csv').drop(columns='Unnamed: 0')
svm2 = pd.read_csv('svm2_results_d1.csv').drop(columns='Unnamed: 0')

# Dataframe containing the min, mean, max and std for the precision metric 
# across all models for each dataset respectively
final_val1 = pd.read_csv('final_model_precision_validation_scores1.csv').drop(columns = 'Unnamed: 0')
final_val2 = pd.read_csv('final_model_precision_validation_scores2.csv').drop(columns='Unnamed: 0')

# %%
