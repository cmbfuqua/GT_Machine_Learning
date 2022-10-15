#%%
import pandas as pd 
import altair as alt 
import numpy as np 
import mlrose_hiive as mlhive
import random
import time

from mlrose_hiive import genetic_alg as gen
from mlrose_hiive import random_hill_climb as rhc
from mlrose_hiive import mimic 
from mlrose_hiive import simulated_annealing as sa

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import StandardScaler as ss


#peak4 = mlhive.FourPeaks()
#sack = mlhive.Knapsack()
#sales = mlhive.TravellingSales()


#%% Ready for full run: This takes all day to run started at 9 am and it finishes
# Traveling sales
# set 'grid search' values
num_samples = list(range(5,50,5))
# establish empty arrays to append data
time_list = []
samp_list = []
fit_score = []
algo_list = []
curve_list = []

count_samp = 1
for sample in num_samples:

    print('{} samples out of {}'.format(count_samp,len(num_samples)))

    x_cor = random.sample(range(1,1000),sample)
    y_cor = random.sample(range(1,1000),sample)

    coordinates = list(zip(x_cor,y_cor))

    fitness = mlhive.TravellingSales(coords=coordinates)

    traveler_fit = mlhive.TSPOpt(length=len(x_cor),fitness_fn=fitness,maximize=True)

    # Genetic Algorithm
    start = time.time()
    gen_state, gen_fit, gen_curve = gen(traveler_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    gen_time = stop - start

    # Random Hill Climb
    start = time.time()
    rhc_state, rhc_fit,rhc_curve = rhc(traveler_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    rhc_time = stop - start

    # MIMIC
    start = time.time()
    mimic_state,mimic_fit,mimic_curve = mimic(traveler_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    mimic_time = stop - start

    # Simulated Annealing
    start = time.time()
    sa_state,sa_fit,sa_curve = sa(traveler_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    sa_time = stop - start

    # Append values
    # Genetic
    time_list.append(gen_time)
    fit_score.append(gen_fit)
    samp_list.append(sample)
    curve_list.append(gen_curve)
    algo_list.append('Genetic')
    # Random Hill Climb     
    time_list.append(rhc_time)
    fit_score.append(rhc_fit)
    samp_list.append(sample)
    curve_list.append(rhc_curve)
    algo_list.append('Random Hill Climb')
    # MIMIC
    time_list.append(mimic_time)
    fit_score.append(mimic_fit)
    samp_list.append(sample)
    curve_list.append(mimic_curve)
    algo_list.append('MIMIC')
    # Simulated Annealing
    time_list.append(sa_time)
    fit_score.append(sa_fit)
    samp_list.append(sample)
    curve_list.append(sa_curve)
    algo_list.append('Simulated Annealing')
    count_samp += 1

tsp_data = pd.DataFrame({'time':time_list,
                         'fit_score':fit_score,
                         'sample_size':samp_list,
                         'curves':curve_list,
                         'algorithm_name':algo_list})


tsp_data.to_csv('tsp_time_data.csv')

# unpacking curve values
import numpy as np
x = []
y = []
s = []
a = []
index = 0
for size in tsp_data.sample_size.drop_duplicates():
    for algo in tsp_data.algorithm_name.drop_duplicates():
        val = tsp_data.loc[(tsp_data.sample_size == size) & (tsp_data.algorithm_name == algo)].curves
        for i in range(len(val[index])):
            #print(val[index][i])
            x.append(val[index][i][1])
            y.append(val[index][i][0])
            s.append(size)
            a.append(algo)
        index += 1

tsp_curves = pd.DataFrame({'iteration':x,
                              'score':y,
                              'size':s,
                              'algorithm':a})
tsp_curves.to_csv('tsp_curve_data.csv')
# %% Ready for full run
# four peaks

lengths = list(range(5,50,5))

time_list = []
samp_list = []
fit_score = []
algo_list = []
curve_list = []


count2 = 1
for lngth in lengths:
    print('{} length out of {}'.format(count2,len(lengths)))
    fitness = mlhive.FourPeaks()

    peaks_fit = mlhive.DiscreteOpt(length=lngth,fitness_fn=fitness,maximize=True,max_val=2)

    # Genetic Algorithm
    start = time.time()
    gen_state, gen_fit, gen_curve = gen(peaks_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    gen_time = stop - start

    # Random Hill Climb
    start = time.time()
    rhc_state, rhc_fit,rhc_curve = rhc(peaks_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    rhc_time = stop - start

    # MIMIC
    start = time.time()
    mimic_state,mimic_fit,mimic_curve = mimic(peaks_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    mimic_time = stop - start

    # Simulated Annealing
    start = time.time()
    sa_state,sa_fit,sa_curve = sa(peaks_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    sa_time = stop - start

    # Append values
    # Genetic
    time_list.append(gen_time)
    fit_score.append(gen_fit)
    samp_list.append(lngth)
    curve_list.append(gen_curve)
    algo_list.append('Genetic')
    # Random Hill Climb     
    time_list.append(rhc_time)
    fit_score.append(rhc_fit)
    samp_list.append(lngth)
    curve_list.append(rhc_curve)
    algo_list.append('Random Hill Climb')
    # MIMIC
    time_list.append(mimic_time)
    fit_score.append(mimic_fit)
    samp_list.append(lngth)
    curve_list.append(mimic_curve)
    algo_list.append('MIMIC')
    # Simulated Annealing
    time_list.append(sa_time)
    fit_score.append(sa_fit)
    samp_list.append(lngth)
    curve_list.append(sa_curve)
    algo_list.append('Simulated Annealing')
    count2 += 1
              

peaks4_data = pd.DataFrame({'time':time_list,
                         'fit_score':fit_score,
                         'sample_size':samp_list,
                         'curves':curve_list,
                         'algorithm_name':algo_list})
peaks4_data.to_csv('peaks4_time_data.csv')

# unpacking curve values
import numpy as np
x = []
y = []
s = []
a = []
index = 0
for size in peaks4_data.sample_size.drop_duplicates():
    for algo in peaks4_data.algorithm_name.drop_duplicates():
        val = peaks4_data.loc[(peaks4_data.sample_size == size) & (peaks4_data.algorithm_name == algo)].curves
        for i in range(len(val[index])):
            #print(val[index][i])
            x.append(val[index][i][1])
            y.append(val[index][i][0])
            s.append(size)
            a.append(algo)
        index += 1

peaks4_curves = pd.DataFrame({'iteration':x,
                              'score':y,
                              'size':s,
                              'algorithm':a})
peaks4_curves.to_csv('peaks4_curve_data.csv')

# %% Ready for full run
# knapsack
num_samples = list(range(5,50,5))

time_list = []
samp_list = []
fit_score = []
algo_list = []
curve_list = []

 
count2 = 1
for samples in num_samples:
    print('{} samples out of {}'.format(count2,len(num_samples)))

    weigh = random.choices(population = range(0,1),k=samples)
    value = list(range(1,samples+1))

    fitness = mlhive.FlipFlop()

    knapsack_fit = mlhive.FlipFlopOpt(length = samples, fitness_fn=fitness,maximize=True)

     # Genetic Algorithm
    start = time.time()
    gen_state, gen_fit, gen_curve = gen(knapsack_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    gen_time = stop - start

    # Random Hill Climb
    start = time.time()
    rhc_state, rhc_fit,rhc_curve = rhc(knapsack_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    rhc_time = stop - start

    # MIMIC
    start = time.time()
    mimic_state,mimic_fit,mimic_curve = mimic(knapsack_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    mimic_time = stop - start

    # Simulated Annealing
    start = time.time()
    sa_state,sa_fit,sa_curve = sa(knapsack_fit,
                                random_state = 42,
                                curve=True)
    stop = time.time()
    sa_time = stop - start

    # Append values
    # Genetic
    time_list.append(gen_time)
    fit_score.append(gen_fit)
    samp_list.append(samples)
    curve_list.append(gen_curve)
    algo_list.append('Genetic')
    # Random Hill Climb     
    time_list.append(rhc_time)
    fit_score.append(rhc_fit)
    samp_list.append(samples)
    curve_list.append(rhc_curve)
    algo_list.append('Random Hill Climb')
    # MIMIC
    time_list.append(mimic_time)
    fit_score.append(mimic_fit)
    samp_list.append(samples)
    curve_list.append(mimic_curve)
    algo_list.append('MIMIC')
    # Simulated Annealing
    time_list.append(sa_time)
    fit_score.append(sa_fit)
    samp_list.append(samples)
    curve_list.append(sa_curve)
    algo_list.append('Simulated Annealing')
    count2 += 1
            

knapsack_data = pd.DataFrame({'time':time_list,
                         'fit_score':fit_score,
                         'sample_size':samp_list,
                         'curves':curve_list,
                         'algorithm_name':algo_list})

# unpacking curve values
import numpy as np
x = []
y = []
s = []
a = []
index = 0
for size in knapsack_data.sample_size.drop_duplicates():
    for algo in knapsack_data.algorithm_name.drop_duplicates():
        val = knapsack_data.loc[(knapsack_data.sample_size == size) & (knapsack_data.algorithm_name == algo)].curves
        for i in range(len(val[index])):
            #print(val[index][i])
            x.append(val[index][i][1])
            y.append(val[index][i][0])
            s.append(size)
            a.append(algo)
        index += 1

knapsack_curves = pd.DataFrame({'iteration':x,
                              'score':y,
                              'size':s,
                              'algorithm':a})
knapsack_curves.to_csv('knapsack_curve_data.csv')
knapsack_data.to_csv('knapsack_time_data.csv')
    
# %%
###################################################
# Part 2, The ANN
###################################################
# go through cleaning process
# %%
'''
data1 = pd.read_csv('subscribed.csv')
data1.head()

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

X = cleanT1.drop(columns= ['subscribed'])
y = cleanT1.subscribed

x_train,x_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = .2,
                                                random_state=42)

scaler = ss().fit(x_train)

#x_train = pd.DataFrame(scaler.transform(x_train))
#x_test = pd.DataFrame(scaler.transform(x_test))
'''
#%%
#Brandon cleaning
# Data cleaning
data_raw = pd.read_csv('https://raw.githubusercontent.com/brandonritchie/SupervisedLearningProject1/master/Train.csv')
data_raw = pd.get_dummies(data_raw, columns = ['job_type','marital','education','default', 'prev_campaign_outcome'])
# Transformation of date time: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
data_raw['sin_time'] = np.sin(2*np.pi*data_raw.day_of_month/365)
data_raw['cos_time'] = np.cos(2*np.pi*data_raw.day_of_month/365)
data_raw['housing_loan'] = np.where(data_raw['housing_loan'] == 'yes', 1,0)
data_raw['personal_loan'] = np.where(data_raw['personal_loan'] == 'yes', 1,0)
data_raw = data_raw.fillna(data_raw.mean())
data_cleaned = data_raw.drop(columns=['communication_type', 'day_of_month', 'month', 'id','days_since_prev_campaign_contact'])
# Downsample to balanced data
sub = data_cleaned.loc[data_cleaned.term_deposit_subscribed == 1]
nsub = data_cleaned.loc[data_cleaned.term_deposit_subscribed == 0].head(len(sub))
data_cleaned = pd.concat([sub,nsub])
X = data_cleaned.drop(['term_deposit_subscribed'], axis = 1)
y = data_cleaned[['term_deposit_subscribed']]
X_train1,X_test1,y_train,y_test = train_test_split(X,y,test_size=0.2)
#X_train2, X_val1, y_train, y_val = train_test_split(X_train1,y_train,test_size=0.3)
y_train = y_train.reset_index().drop('index', axis = 1)
y_test = y_test.reset_index().drop('index', axis = 1)
sc_train = ss()
sc_val = ss()
sc_test = ss()
x_train = pd.DataFrame(sc_train.fit_transform(X_train1.values), columns = X_train1.columns)
#X_val = pd.DataFrame(sc_train.fit_transform(X_val1.values), columns = X_val1.columns)
x_test = pd.DataFrame(sc_test.fit_transform(X_test1.values), columns = X_test1.columns)
#%%
from mlrose_hiive import NeuralNetwork as ann
#####################
# Need to create base ANN because activation function of LOGISTIC doesn't exist
# switching to identity as precision was .86 and accuracy was .84
################

restart = list(range(1,20,1))

restart_list = []

time_list = []
precision_list = []
accuracy_list = []

for start in restart:
    hill = ann(activation='identity',
        hidden_nodes=[10],
        restarts= start,
        max_iters=650,
        algorithm = 'random_hill_climb',
        early_stopping=True,
        random_state = 42)

    s = time.time()
    hill.fit(x_train,y_train)
    st = time.time()
    clock_time = st-s

    preds = hill.predict(x_test)

    precision = precision_score(y_test,preds)
    accuracy = accuracy_score(y_test,preds)

    restart_list.append(start)

    time_list.append(clock_time)
    precision_list.append(precision)
    accuracy_list.append(accuracy)

hill_data = pd.DataFrame({
    'restart':restart_list,
    'time':time_list,
    'precision':precision_list,
    'accuracy':accuracy_list
})

hill_data.to_csv('Random_Hill_Climb_ANN.csv')

#%%
schedule_name_list = []
t_list = []

time_list = []
precision_list = []
accuracy_list = []

for t in range(1,22,2):
    t = t/100
    schedule_name = ['Geom','Exp','Arith']
    schedule_ob = [mlhive.GeomDecay(init_temp = t),mlhive.ExpDecay(init_temp = t),mlhive.ArithDecay(init_temp = t)]

    for i in [0,1,2]:

        sima = ann(activation='identity',
            hidden_nodes=[10],
            max_iters=650,
            algorithm = 'simulated_annealing',
            early_stopping=True,
            schedule = schedule_ob[i])

        s = time.time()
        sima.fit(x_train,y_train)
        st = time.time()
        clock_time = st-s

        preds = sima.predict(x_test)

        precision = precision_score(y_test,preds)
        accuracy = accuracy_score(y_test,preds)

        schedule_name_list.append(schedule_name[i])
        t_list.append(t)

        time_list.append(clock_time)
        precision_list.append(precision)
        accuracy_list.append(accuracy)

sima_data = pd.DataFrame({
    'schedule':schedule_name_list,
    'temp':t_list,
    'time':time_list,
    'precision':precision_list,
    'accuracy':accuracy_list
})

sima_data.to_csv('Simulated_Annealing_ANN.csv')

#%%
popSize = list(range(5,200,20))
mutation = list(range(1,21,5))

size_list = []
mutation_list = []

time_list = []
precision_list = []
accuracy_list = []

count1 = 1
for size in popSize:
    count2 = 1
    for mut in mutation:
        print('{} out of {}'.format(count1,len(popSize)))
        print('{} out of {}\n'.format(count2,len(mutation)))
        count2 += 1
        ga = ann(activation='identity',
            hidden_nodes=[10],
            max_iters=650,
            algorithm='genetic_alg',
            pop_size=size,
            mutation_prob=mut/100,
            early_stopping=True,
            random_state = 42)
        
        s = time.time()
        ga.fit(x_train,y_train)
        st = time.time()
        clock_time = st-s

        preds = ga.predict(x_test)

        precision = precision_score(y_test,preds)
        accuracy = accuracy_score(y_test,preds)

        size_list.append(size)
        mutation_list.append(mut)

        time_list.append(clock_time)
        precision_list.append(precision)
        accuracy_list.append(accuracy)
    count1 += 1

ga_data = pd.DataFrame({
    'size':size_list,
    'mutation':mutation_list,
    'time':time_list,
    'precision':precision_list,
    'accuracy':accuracy_list
})

ga_data.to_csv('Genetic_ANN.csv')        

# %%
from mlrose_hiive import NeuralNetwork as ann


# Run this cell to see the base line for the new ANN
gd = ann(activation='identity',
        hidden_nodes=[20],
        max_iters=650,
        algorithm='gradient_descent',
        early_stopping=True,
        random_state = 42)
s = time.time()
gd.fit(x_train,y_train)
st = time.time()
clock_time = st-s
preds = gd.predict(x_test)
precision = precision_score(y_test,preds).round(2)
accuracy = accuracy_score(y_test,preds).round(2)

print('Time: {}\nPrecision: {}\nAccuracy: {}\n\n'.format(round(clock_time,2),precision,accuracy))
# %%
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

# %%
s = time.time()
sima = ann(activation='identity',
            hidden_nodes=[10],
            max_iters=650,
            algorithm = 'simulated_annealing',
            early_stopping=True,
            schedule = mlhive.GeomDecay(init_temp = .15))
sima.fit(x_train,y_train)
st = time.time()
sima_time = st-s
print('done Sim')
s = time.time()
hill = ann(activation='identity',
        hidden_nodes=[10],
        restarts= 6,
        max_iters=650,
        algorithm = 'random_hill_climb',
        early_stopping=True,
        random_state = 42)
hill.fit(x_train,y_train)
st = time.time()
hill_time = st-s
print('done hill')
s = time.time()
ga = ann(activation='identity',
            hidden_nodes=[10],
            max_iters=650,
            algorithm='genetic_alg',
            pop_size= 120,
            mutation_prob= 1/100,
            early_stopping=True,
            random_state = 42)
ga.fit(x_train,y_train)
st = time.time()
ga_time = st-s
print('done GA')
#%%
sima_pred = sima.predict(x_test)
hill_pred = hill.predict(x_test)
ga_pred = ga.predict(x_test)

ga_curve = plot_learning_curve(ga,'Genetic Algorithm',X,y)
sa_curve = plot_learning_curve(sima,'Simulated Annealing',X,y)
hill_curve = plot_learning_curve(hill,'Random Hill Climb',X,y)

# %%
