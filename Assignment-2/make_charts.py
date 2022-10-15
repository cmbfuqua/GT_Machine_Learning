#%%
import pandas as pd 
import altair as alt 
alt.data_transformers.enable(max_rows = None)

tspt = pd.read_csv('tsp_time_data.csv')
tspc = pd.read_csv('tsp_curve_data.csv')
kst = pd.read_csv('knapsack_time_data.csv')
ksc = pd.read_csv('knapsack_curve_data.csv')
p4t = pd.read_csv('peaks4_time_data.csv')
p4c = pd.read_csv('peaks4_curve_data.csv')


# %%
tspc
sa_show_fit = alt.Chart(tspc.loc[tspc['size'].isin([5,25,45])]).mark_line(clip = True).encode(
    alt.X('iteration', title = 'Iterations', scale = alt.Scale(domain = (0,2000))),
    alt.Y('score', title = 'Fitness Score'),
    alt.Color('algorithm', title = 'Algorithm'),
    alt.Column('size', title = 'Size')
)
#sa_show_fit.save('SA_showcase_fit.png')
sa_show_fit
#%%
sa_show_time = alt.Chart(tspt).mark_line().encode(
    alt.X('sample_size', title = 'Samples'),
    alt.Y('time', title = 'Time'),
    alt.Color('algorithm_name', title = 'Algorithm'),
)
sa_show_time.save('SA_showcase_time.png')
sa_show_time
# %%
y_axis = [0,9,0,3,0,5,0,7,0]
x_axis = [0,1,2,6,10,13,16,18,20]
fake_data = pd.DataFrame({'x':x_axis,'y':y_axis})
fake = alt.Chart(fake_data).mark_line(point = True).encode(
    alt.X('x'),
    alt.Y('y')
)
fake.save('peak4_fake.png')
fake
# %%
# getting p4 data clean
ga_show_fit = alt.Chart(p4c.loc[p4c['size'].isin([30,45])]).mark_line().encode(
    alt.X('iteration'),
    alt.Y('score'),
    alt.Color('algorithm'),
    alt.Column('size')
)
ga_show_fit.save('ga_show_fit.png')
ga_show_fit
# %%
ga_show_time = alt.Chart(p4t).mark_line().encode(
    alt.X('sample_size'),
    alt.Y('time'),
    alt.Color('algorithm_name'),
)
ga_show_time.save('ga_show_time.png')
ga_show_time
# %%
mimic_show_fit = alt.Chart(ksc.loc[(ksc['size'].isin([20,35,45]))]).mark_line(clip = True).encode(
    alt.X('iteration', scale = alt.Scale(domain = (0,24))),
    alt.Y('score'),
    alt.Color('algorithm'),
    alt.Column('size')
)
mimic_show_fit.save('mimic_show_fit.png')
mimic_show_fit
#%%
show_rhc = alt.Chart(ksc.loc[(ksc['size'].isin([35]))]).mark_line(clip = True).encode(
    alt.X('iteration'),
    alt.Y('score'),
    alt.Color('algorithm', legend = None),
)
show_rhc.save('mimic_showRHC_fit.png')
show_rhc
# %%
mimic_show_time = alt.Chart(kst).mark_line().encode(
    alt.X('sample_size'),
    alt.Y('time'),
    alt.Color('algorithm_name', legend = None)
)
mimic_show_time.save('mimic_show_time.png')
mimic_show_time
# %%
#######################
# ANN adjustment
#######################
rhc_ann = pd.read_csv('Random_Hill_Climb_Ann.csv')
ga_ann = pd.read_csv('Genetic_ANN.csv')
sa_ann = pd.read_csv('Simulated_Annealing_ANN.csv')

base = pd.DataFrame({'time':[.15],
                     'precision':[.53],
                     'accuracy':[.52]})
                     
# %%
rhc_long = rhc_ann[['restart','time','precision','accuracy']].melt(id_vars = ['restart'],value_vars = ['time','precision','accuracy'],var_name = 'metric',value_name = 'value')
rhc_best = alt.Chart(rhc_long.loc[rhc_long.metric.isin(['precision','accuracy'])],title = 'Random Hill Climb').mark_line().encode(
    alt.X('restart', title = 'Restarts'),
    alt.Y('value', title = 'Value', scale = alt.Scale(zero = False)),
    alt.Color('metric',title = 'Metric Type')
)
rhc_best.save('rhc_best_ann.png')
rhc_best
# %%
ga_long = ga_ann[['size','mutation','time','precision','accuracy']].melt(id_vars = ['size','mutation'],value_vars = ['time','precision','accuracy'],var_name = 'metric',value_name = 'value')
ga_best = alt.Chart(ga_long.loc[(ga_long.metric.isin(['precision','accuracy'])& (ga_long.mutation.isin([1,6])))],title = 'Genetic').mark_line().encode(
    alt.X('size', title = 'Size'),
    alt.Y('value', title = 'Value', scale = alt.Scale(zero = False)),
    alt.Color('metric',title = 'Metric Type'),
    alt.Column('mutation')
)
ga_best.save('ga_best_ann.png')
ga_best
# %%
sa_long = sa_ann[['schedule','temp','time','precision','accuracy']].melt(id_vars = ['schedule','temp'],value_vars = ['time','precision','accuracy'],var_name = 'metric',value_name = 'value')
sa_best = alt.Chart(sa_long.loc[sa_long.metric.isin(['precision','accuracy'])],title = 'Simulated Annealing').mark_line().encode(
    alt.X('temp',title = 'Temperature'),
    alt.Y('value',title = 'Value'),
    alt.Color('metric'),
    alt.Column('schedule',title = 'Schedule')
)
sa_best.save('sa_best_ann.png')
sa_best
# %%
