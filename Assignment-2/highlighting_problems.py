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
# at 9 pm
# Traveling sales
# set 'grid search' values
num_samples = list(range(5,50,5))
max_itterations = list(range(5,100,5))
max_attmpts = list(range(2,22,2))

# establish empty arrays to append data
time_list = []
samp_list = []
itter_list = []
attempt_list = []
fit_score = []
algo_list = []
count_itter = 1


for itteration in max_itterations:
    count_samp = 1

    for sample in num_samples:
        count_attmpt = 1

        for attmpt in max_attmpts:
            print('{} itterations out of {}'.format(count_itter,len(max_itterations)))
            print('{} samples out of {}'.format(count_samp,len(num_samples)))
            print('{} attempts out of {}\n'.format(count_attmpt,len(max_attmpts)))

            x_cor = random.sample(range(1,1000),sample)
            y_cor = random.sample(range(1,1000),sample)

            coordinates = list(zip(x_cor,y_cor))

            fitness = mlhive.TravellingSales(coords=coordinates)

            traveler_fit = mlhive.TSPOpt(length=len(x_cor),fitness_fn=fitness,maximize=True)

            # Genetic Algorithm
            start = time.time()
            gen_state, gen_fit, gen_curve = gen(traveler_fit,
                                     max_iters=itteration,
                                     max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            gen_time = stop - start

            # Random Hill Climb
            start = time.time()
            rhc_state, rhc_fit,curve = rhc(traveler_fit,
                                     max_iters=itteration,
                                     max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            rhc_time = stop - start

            # MIMIC
            start = time.time()
            mimic_state,mimic_fit,curve = mimic(traveler_fit,
                                         max_iters=itteration,
                                         max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            mimic_time = stop - start

            # Simulated Annealing
            start = time.time()
            sa_state,sa_fit,curve = sa(traveler_fit,
                                 max_iters=itteration,
                                 max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            sa_time = stop - start

            # Append values
            # Genetic
            time_list.append(gen_time)
            fit_score.append(gen_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Genetic')
            # Random Hill Climb     
            time_list.append(rhc_time)
            fit_score.append(rhc_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Random Hill Climb')
            # MIMIC
            time_list.append(mimic_time)
            fit_score.append(mimic_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('MIMIC')
            # Simulated Annealing
            time_list.append(sa_time)
            fit_score.append(sa_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Simulated Annealing')
            count_attmpt += 1
        count_samp += 1
    count_itter += 1

tsp_data = pd.DataFrame({'time':time_list,
                         'fit_score':fit_score,
                         'sample_size':samp_list,
                         'itterations':itter_list,
                         'attempts':attempt_list,
                         'algorithm_name':algo_list})


tsp_data.to_csv('tsp_data.csv')
# %% Ready for full run
# four peaks

lengths = list(range(5,50,5))
max_itterations = list(range(5,100,5))
max_attmpts = list(range(2,22,2))

time_list = []
samp_list = []
itter_list = []
attempt_list = []
fit_score = []
algo_list = []

count1 = 1
for itteration in max_itterations:
    
    count2 = 1
    for lngth in lengths:
        
        count3 = 1
        for attmpt in max_attmpts:
            print('{} itteration out of {}'.format(count1,len(max_itterations)))
            print('{} length out of {}'.format(count2,len(lengths)))
            print('{} attempts out of {}\n'.format(count3,len(max_attmpts)))
            fitness = mlhive.FourPeaks()

            peaks_fit = mlhive.DiscreteOpt(length=lngth,fitness_fn=fitness,maximize=True,max_val=2)

            # Genetic Algorithm
            start = time.time()
            gen_state, gen_fit, gen_curve = gen(traveler_fit,
                                     max_iters=itteration,
                                     max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            gen_time = stop - start

            # Random Hill Climb
            start = time.time()
            rhc_state, rhc_fit,curve = rhc(traveler_fit,
                                     max_iters=itteration,
                                     max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            rhc_time = stop - start

            # MIMIC
            start = time.time()
            mimic_state,mimic_fit,curve = mimic(traveler_fit,
                                         max_iters=itteration,
                                         max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            mimic_time = stop - start

            # Simulated Annealing
            start = time.time()
            sa_state,sa_fit,curve = sa(traveler_fit,
                                 max_iters=itteration,
                                 max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            sa_time = stop - start

            # Append values
            # Genetic
            time_list.append(gen_time)
            fit_score.append(gen_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Genetic')
            # Random Hill Climb     
            time_list.append(rhc_time)
            fit_score.append(rhc_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Random Hill Climb')
            # MIMIC
            time_list.append(mimic_time)
            fit_score.append(mimic_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('MIMIC')
            # Simulated Annealing
            time_list.append(sa_time)
            fit_score.append(sa_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Simulated Annealing')
            count3 += 1
        count2 += 1
    count1 += 1
              

peaks4_data = pd.DataFrame({'time':time_list,
                         'fit_score':fit_score,
                         'sample_size':samp_list,
                         'itterations':itter_list,
                         'attempts':attempt_list,
                         'algorithm_name':algo_list})


peaks4_data.to_csv('peaks4_data.csv')

# %% Ready for full run
# knapsack
max_itterations = list(range(5,100,5))
max_attmpts = list(range(2,22,2))
num_samples = list(range(5,50,5))

time_list = []
samp_list = []
itter_list = []
attempt_list = []
fit_score = []
algo_list = []

count1 = 1
for itteration in max_itterations:   
    count2 = 1

    for samples in num_samples:
        count3 = 1

        for attmpt in max_attmpts:
            print('{} itterations out of {}'.format(count1,len(max_itterations)))
            print('{} samples out of {}'.format(count2,len(num_samples)))
            print('{} attempts out of {}\n'.format(count3,len(max_attmpts)))

            weigh = random.choices(population = range(1,10),k=samples)
            value = list(range(1,samples+1))

            fitness = mlhive.Knapsack(weights=weigh,values=value)

            knapsack_fit = mlhive.KnapsackOpt(length=samples,fitness_fn=fitness,maximize=True)

            # Genetic Algorithm
            start = time.time()
            gen_state, gen_fit, gen_curve = gen(traveler_fit,
                                     max_iters=itteration,
                                     max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            gen_time = stop - start

            # Random Hill Climb
            start = time.time()
            rhc_state, rhc_fit,curve = rhc(traveler_fit,
                                     max_iters=itteration,
                                     max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            rhc_time = stop - start

            # MIMIC
            start = time.time()
            mimic_state,mimic_fit,curve = mimic(traveler_fit,
                                         max_iters=itteration,
                                         max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            mimic_time = stop - start

            # Simulated Annealing
            start = time.time()
            sa_state,sa_fit,curve = sa(traveler_fit,
                                 max_iters=itteration,
                                 max_attempts=attmpt,
                                     random_state = 42)
            stop = time.time()
            sa_time = stop - start

            # Append values
            # Genetic
            time_list.append(gen_time)
            fit_score.append(gen_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Genetic')
            # Random Hill Climb     
            time_list.append(rhc_time)
            fit_score.append(rhc_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Random Hill Climb')
            # MIMIC
            time_list.append(mimic_time)
            fit_score.append(mimic_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('MIMIC')
            # Simulated Annealing
            time_list.append(sa_time)
            fit_score.append(sa_fit)
            samp_list.append(sample)
            itter_list.append(itteration)
            attempt_list.append(attmpt)
            algo_list.append('Simulated Annealing')
            count3 += 1
        count2 += 1
    count1 += 1
            
#%%
knapsack_data = pd.DataFrame({'time':time_list,
                         'fit_score':fit_score,
                         'sample_size':samp_list,
                         'itterations':itter_list,
                         'attempts':attempt_list,
                         'algorithm_name':algo_list})


knapsack_data.to_csv('knapsack_data.csv')
    
# %%
###################################################
# Part 2, The ANN
###################################################
# go through cleaning process
# %%
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

X1 = cleanT1.drop(columns= ['subscribed'])
Y1 = cleanT1.subscribed

x_train,x_test,y_train,y_test = train_test_split(X1,
                                                Y1,
                                                test_size = .2,
                                                random_state=42)

scaler = ss().fit(x_train)

#x_train = list(scaler.transform(x_train))
#x_test = list(scaler.transform(x_test))
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

schedule_name = ['Geom','Exp','Arith']
schedule_ob = [mlhive.GeomDecay,mlhive.ExpDecay,mlhive.ArithDecay]

schedule_name_list = []

time_list = []
precision_list = []
accuracy_list = []

for i in [0,1,2]:

    sima = ann(activation='identity',
        hidden_nodes=[10],
        max_iters=650,
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

    time_list.append(clock_time)
    precision_list.append(precision)
    accuracy_list.append(accuracy)

sima_data = pd.DataFrame({
    'schedule':schedule_name_list,
    'time':time_list,
    'precision':precision_list,
    'accuracy':accuracy_list
})

sima_data.to_csv('Simulated_Annealing_ANN.csv')

#%%
popSize = list(range(5,420,20))
mutation = list(range(1,10,1))

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
            mutation_prob=mut/10,
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
