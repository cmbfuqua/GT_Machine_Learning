#%% 
import pandas as pd 
import altair as alt 
alt.data_transformers.enable(max_rows = None)
# %%
# Dataframes containing the grid search values and the associated metrics
b1 = pd.read_csv('boost_results_d1.csv')
b2 = pd.read_csv('boost_results_d2.csv')
d1 = pd.read_csv('dt_results_d1.csv')
d2 = pd.read_csv('dt_results_d2.csv')
k1 = pd.read_csv('knn_results_d1.csv')
k2 = pd.read_csv('knn_results_d2.csv')
m1 = pd.read_csv('mlp_results_d1.csv')
m2 = pd.read_csv('mlp_results_d2.csv')
s1 = pd.read_csv('svm_results_d2.csv')
s2 = pd.read_csv('svm_results_d2.csv')
b21 = pd.read_csv('boost2_results_d1.csv')
b22 = pd.read_csv('boost2_results_d2.csv')
svm1 = pd.read_csv('svm2_results_d1.csv')
svm2 = pd.read_csv('svm2_results_d2.csv')

b1['dataset_name'] = 'boost1'.upper()
b2['dataset_name'] = 'boost2'.upper()
d1['dataset_name'] = 'decision_tree1'.upper()
d2['dataset_name'] = 'decision_tree2'.upper()
k1['dataset_name'] = 'knn1'.upper()
k2['dataset_name'] = 'knn2'.upper()
m1['dataset_name'] = 'mlp1'.upper()
m2['dataset_name'] = 'mlp2'.upper()
s1['dataset_name'] = 'svm1'.upper()
s2['dataset_name'] = 'svm2'.upper()
b21['dataset_name'] = 'boost21'.upper()
b22['dataset_name'] = 'boost22'.upper()
svm1['dataset_name'] = 'svm1'.upper()
svm2['dataset_name'] = 'svm2'.upper()
# Dataframe containing the model name and the best combination of metrics to
# obtain the best result of either time, precision, accuracy or recall
best_MetPar = pd.read_csv('best_metrics_parameters.csv')

# Dataframes containing the further tuned svm models with the degree metric
# Note, this data was created on an arbitrary train_test_split and not with
# cross validation like unto the final_val{1,2} datasets. 
# This was in effort to see how the parameter performed due to the poly
# kernel being the best

# Dataframe containing the min, mean, max and std for the precision metric 
# across all models for each dataset respectively
final_val1 = pd.read_csv('final_model_precision_validation_scores1.csv')
final_val2 = pd.read_csv('final_model_precision_validation_scores2.csv')

decision_tree = pd.concat([d1,d2])
knn = pd.concat([k1,k2])
boost = pd.concat([b1,b2])
mlps = pd.concat([m1,m2])
mlps = mlps.loc[mlps.solver == 'adam'] # removing the adam values because it is too much to demonstrate
svm = pd.concat([s1,s2])
refined_svm = pd.concat([svm1,svm2])
boost2 = pd.concat([b21,b22])

# %%
gb = alt.Chart(
    decision_tree.loc[(decision_tree.crit == 'gini') & (decision_tree.split == 'best')],
    ).mark_line().encode(
        alt.X('depth',title = 'Tree Depth'),
        alt.Y('precision', title = 'Precision Score',scale = alt.Scale(zero = False)),
        alt.Color('dataset_name',title = 'Dataset Type',legend = None)
    ).properties(
        title = {
            'text': ['Gini:Best:1-30'],
            'subtitle':['dt_results_d[1,2]'],
            'subtitleColor':'gray'
        }
    )
gb.configure_axis(labelFontSize=20,titleFontSize=20).save('graphs/dt_gb.png')

gr = alt.Chart(
    decision_tree.loc[(decision_tree.crit == 'gini') & (decision_tree.split == 'random')],
    ).mark_point().encode(
        alt.X('depth',title = 'Tree Depth'),
        alt.Y('precision', title = 'Precision Score',scale = alt.Scale(zero = False)),
        alt.Color('dataset_name',title = 'Dataset Type',legend = None)
    ).properties(
        title = {
            'text': ['Gini:Random:1-30'],
            'subtitle':['dt_results_d[1,2]'],
            'subtitleColor':'gray'
        }
    )
(gr+gb).configure_axis(labelFontSize=20,titleFontSize=20).save('graphs/dt_gr.png')

eb = alt.Chart(
    decision_tree.loc[(decision_tree.crit == 'entropy') & (decision_tree.split == 'best')],
    ).mark_point().encode(
        alt.X('depth',title = 'Tree Depth'),
        alt.Y('precision', title = 'Precision Score',scale = alt.Scale(zero = False)),
        alt.Color('dataset_name',title = 'Dataset Type',legend = None)
    ).properties(
        title = {
            'text': ['Entorpy:Best:1-30'],
            'subtitle':['dt_results_d[1,2]'],
            'subtitleColor':'gray'
        }
    )
(eb+gb).configure_axis(labelFontSize=20,titleFontSize=20).save('graphs/dt_eb.png')

er = alt.Chart(
    decision_tree.loc[(decision_tree.crit == 'entropy') & (decision_tree.split == 'random')],
    ).mark_point().encode(
        alt.X('depth',title = 'Tree Depth'),
        alt.Y('precision', title = 'Precision Score',scale = alt.Scale(zero = False)),
        alt.Color('dataset_name',title = 'Dataset Type',legend = None)
    ).properties(
        title = {
            'text': ['Entropy:Random:1-30'],
            'subtitle':['dt_results_d[1,2]'],
            'subtitleColor':'gray'
        }
    )
(er+gb).configure_axis(labelFontSize=20,titleFontSize=20).save('graphs/dt_er.png')
# %%
#################################
# run the code in this comment to see all of the plateaus
# alt.Chart(knn).mark_line().encode(
#    alt.X('neighbors'),
 #   alt.Y('precision'),
 #   alt.Color('dataset_name'),
 #   alt.Row('algorithm'),
 #   alt.Column('metric')
#)
################################
beoge = alt.Chart(knn.loc[(knn.algorithm == 'ball_tree') & (knn.metric == 'euclidean')]
                ).mark_line().encode(
                    alt.X('neighbors', title = 'Number of Neighbors'),
                    alt.Y('precision',title = 'Precision Score'),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None)
                ).properties(
                    title = {
                        'text':['Ball_Tree'],
                        'subtitle':['knn_results_d[1,2]'],
                        'subtitleColor':'gray'
                    }
                )
beogm = alt.Chart(knn.loc[(knn.algorithm == 'ball_tree') & (knn.metric == 'manhattan')]
                ).mark_line().encode(
                    alt.X('neighbors', title = 'Number of Neighbors'),
                    alt.Y('precision',title = 'Precision Score'),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None, scale=alt.Scale(range = ['red','black']))
                ).properties(
                    title = {
                        'text':['Ball_Tree'],
                        'subtitle':['knn_results_d[1,2]'],
                        'subtitleColor':'gray'
                    }
                )


be = alt.Chart(knn.loc[(knn.algorithm == 'ball_tree') & (knn.metric == 'euclidean')]
                ).mark_point(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None)
                )

bm = alt.Chart(knn.loc[(knn.algorithm == 'ball_tree') & (knn.metric == 'manhattan')]
                ).mark_point(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None, scale=alt.Scale(range = ['red','black']))
                )

bre = alt.Chart(knn.loc[(knn.algorithm == 'brute') & (knn.metric == 'euclidean')]
                ).mark_point(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None)
                )

brm = alt.Chart(knn.loc[(knn.algorithm == 'brute') & (knn.metric == 'manhattan')]
                ).mark_point(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None, scale=alt.Scale(range = ['red','black']))
                )

ke = alt.Chart(knn.loc[(knn.algorithm == 'kd_tree') & (knn.metric == 'euclidean')]
                ).mark_point(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None)
                )

km = alt.Chart(knn.loc[(knn.algorithm == 'kd_tree') & (knn.metric == 'manhattan')]
                ).mark_point(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None, scale=alt.Scale(range = ['red','black']))
                )

ae = alt.Chart(knn.loc[(knn.algorithm == 'auto') & (knn.metric == 'euclidean')]
                ).mark_line(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type',legend = None)
                )

am = alt.Chart(knn.loc[(knn.algorithm == 'auto') & (knn.metric == 'manhattan')]
                ).mark_line(clip=True).encode(
                    alt.X('neighbors', title = 'Number of Neighbors',scale = alt.Scale(domain=(0,25))),
                    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
                    alt.Color('dataset_name',title = 'Dataset type', scale=alt.Scale(range = ['red','black']),legend = None)
                )

base = (am+ae)
(beoge + beogm).resolve_scale(color = 'independent').save('graphs/tail.png')
big_chart = ((base+km+ke).resolve_scale(color = 'independent').properties(title = {'text':['Kd_Tree']})|
(base+brm+bre).resolve_scale(color = 'independent').properties(title = {'text':['Brute']})|
(base+bm+be).resolve_scale(color = 'independent').properties(title = {'text':['Ball_Tree']}))
big_chart.configure_axis(labelFontSize=20,titleFontSize=20).save('graphs/knn_big_chart.png')
# %%
mlp_chart = alt.Chart(mlps).mark_line().encode(
    alt.X('iterations'),
    alt.Y('precision',scale = alt.Scale(zero = False)),
    alt.Column('activation'),
    alt.Color('layers:N'),
    alt.Row('dataset_name')
)
mlp_chart.configure_axis(labelFontSize=20,titleFontSize=20).save('graphs/mlpAllchart.png')

mlp_chart = alt.Chart(mlps).mark_line().encode(
    alt.X('iterations'),
    alt.Y('time',scale = alt.Scale(zero = False)),
    alt.Column('activation'),
    alt.Color('layers:N'),
    alt.Row('dataset_name')
)
mlp_chart.configure_axis(labelFontSize=20,titleFontSize=20).save('graphs/mlpAllchart_time.png')
# %%
plateau_chart = alt.Chart(boost).mark_point().encode(
    alt.X('n_estimators',title = 'Number of Estimators'),
    alt.Y('precision',title = 'Precision Score',scale = alt.Scale(zero = False)),
    alt.Color('loss',title = 'Loss Function'),
    alt.Column('learning_rate',title = 'Learning Rate'),
    alt.Row('dataset_name',title = 'Dataset')
).configure_axis(labelFontSize=20,titleFontSize=20)
plateau_chart.save('graphs/boost_plateau.png')
# %%
precision_chart = alt.Chart(boost2).mark_point().encode(
    alt.X('n_estimators'),
    alt.Y('precision'),
    alt.Color('loss'),
    alt.Column('learning_rate'),
    alt.Row('dataset_name')
)
precision_chart
# %%
recall_chart = alt.Chart(boost2).mark_point().encode(
    alt.X('n_estimators'),
    alt.Y('recall'),
    alt.Color('loss'),
    alt.Column('learning_rate'),
    alt.Row('dataset_name')
)
recall_chart
# %%
f = alt.Chart(svm.loc[svm.probability == False]).mark_bar().encode(
    alt.X('kernel',title = 'Kernel Type'),
    alt.Y('accuracy',title = 'Precision Score'),
    alt.Color('kernel'),
    alt.Column('dataset_name')
).configure_axis(labelFontSize=20,titleFontSize=20)

f.save('graphs/svm_bar.png')
# %%
degree = alt.Chart(refined_svm).mark_line(point = True).encode(
    alt.X('degree',title = 'Polynomial Degree'),
    alt.Y('precision',title = 'Precision Score'),
    alt.Color('dataset_name',legend = None)
).configure_axis(labelFontSize=20,titleFontSize=20)
degree.save('graphs/svm_degree.png')
# %%
