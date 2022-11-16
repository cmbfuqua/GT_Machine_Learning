#%% 
import numpy as np 
import pandas as pd
import altair as alt
alt.data_transformers.enable(max_rows = None)
from matplotlib import pyplot as plt 
import time
import gym

from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.mdp import QLearning
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive.mdptoolbox.example import openai
from hiive.mdptoolbox.example import forest
import re
#%%
def visualize_policy(policy, shape, name, title=None):
                M = shape[0]
                N = shape[1]
                actions = np.asarray(policy).reshape(shape)
                mapping = {
                    0: '←',
                    1: '↓',
                    2: '→',
                    3: '↑'
                }
                arr = np.zeros(shape)
                for i in range(M):
                    for j in range(N):
                        if (N * i + j) in TERM_STATE_MAP[name]:
                            arr[i, j] = 0.25
                        elif (N * i + j) in GOAL_STATE_MAP[name]:
                            arr[i, j] = 1.0
                fig, ax = plt.subplots(figsize=(10,10))
                im = ax.imshow(arr, cmap='cool')
                ax.set_xticks(np.arange(M))
                ax.set_yticks(np.arange(N))
                ax.set_xticklabels(np.arange(M))
                ax.set_yticklabels(np.arange(N))
                ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
                ax.grid(False)
                ax.grid(which='minor', color='w', linewidth=2)

                for i in range(M):
                    for j in range(N):
                        if (N * i + j) in TERM_STATE_MAP[name]:
                            ax.text(j, i, 'H', ha='center', va='center', color='k', size=18)
                        elif (N * i + j) in GOAL_STATE_MAP[name]:
                            ax.text(j, i, 'G', ha='center', va='center', color='k', size=18)
                        else:
                            ax.text(j, i, mapping[actions[i, j]], ha='center', va='center', color='k', size=18)
                # fig.tight_layout()
                if title:
                    ax.set_title(title)
                save_value = title + '.png'
                plt.savefig(save_value)
                plt.show()
# %%
######################################
# Frozen Lake visualize map
######################################
tile_frozen = .8
discount = [.3,.9]
map_size = [15,50]
global TERM_STATE_MAP
global GOAL_STATE_MAP

######################
# for tracking metrics
######################
reward_list = []
iterations_list = []
policy_list = []
time_list = []

algorithm_list = []
discount_list = []
mapsize_list = []

qmatrix_list = []

for discnt in discount:
    for msize in map_size:
        maps = generate_random_map(msize,p = tile_frozen)
        
        P, R = openai('FrozenLake-v1',desc = maps)
        

        algos = [ValueIteration(P,R,discnt),
                PolicyIteration(P,R,discnt)]
        algos_name = ['ValueIteration','Policy_Iteration']
        # fill out 
        bigmap = ''
        for i in range(len(maps)):
            bigmap = bigmap + maps[i]
            temp_list = [int(i.start()) for i in re.finditer('H',bigmap)]
            TERM_STATE_MAP = {f"{msize}x{msize}":temp_list}
        GOAL_STATE_MAP = {f"{msize}x{msize}": [msize**2-1]}

        for i in range(len(algos_name)):
            print('{}-{}-{}-{}'.format(algos_name[i],tile_frozen,discnt,msize))
            test = algos[i].run()  

            # append metrics
            for step in range(len(test)):
                print('{} out of {}'.format(step,len(test)))
                reward_list.append(test[step]['Reward'])
                time_list.append(test[step]['Time'])
                iterations_list.append(test[step]['Iteration'])

                policy_list.append(algos[i].policy)
                algorithm_list.append(algos_name[i])
                discount_list.append(discnt)
                mapsize_list.append(msize)

            visualize_policy(algos[i].policy,
                                shape = (msize,msize),
                                name = f"{msize}x{msize}",
                                title = f'{algos_name[i]}-{msize}x{msize}-Discount={discnt}')
#%%
frozen_lake = pd.DataFrame({'algorithm':algorithm_list,
                            'reward':reward_list,
                            'time':time_list,
                            'iterations':iterations_list,
                            'policy':policy_list,
                            'discount':discount_list,
                            'map_size':mapsize_list})

#%%
##################
#Convergence Plots
##################
chart1 = alt.Chart(frozen_lake.loc[frozen_lake.algorithm != 'QLearning']).mark_line(clip = True,point = True).encode(
    alt.X('iterations',title = 'Num of Iterations',scale = alt.Scale(domain = (0,12))),
    alt.Y('reward',title = 'Reward'),
    alt.Color('map_size:N',title = 'NxN Map Size'),
    alt.Column('algorithm',title = 'MDP Solver'),
    alt.Row('discount',title = 'Gamma value')
)
chart1
#%%
chart1.save('Convergence_vi_pi.png')
#%%
###################
# QL: Apparently doesn't do well with larget sizes. Test small 4x4 and then show 15x15
###################
import gym
import random
import numpy as np 
import matplotlib.pyplot as plt

reward_list = []
iteration_list = []
time_list = []
mapsize_list = []

map_sizes = [4,7,10]

for map_size in map_sizes:

    maps = generate_random_map(map_size,.8)

    bigmap = ''
    for i in range(len(maps)):
        bigmap = bigmap + maps[i]
    temp_list = [int(i.start()) for i in re.finditer('H',bigmap)]
    TERM_STATE_MAP = {f"{map_size}x{map_size}":temp_list}
    GOAL_STATE_MAP = {f"{map_size}x{map_size}": [map_size**2-1]}

    environment = gym.make('FrozenLake-v1',is_slippery = False,desc = maps,map_name = '{}x{}'.format(map_size,map_size))
    environment.reset()
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 17})

    # We re-initialize the Q-table
    qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

    # Hyperparameters
    episodes = 10000        # Total number of episodes
    alpha = 0.5            # Learning rate
    gamma = 0.9            # Discount factor

    # List of outcomes to plot
    outcomes = []

    # Training
    for i in range(episodes):
        state = environment.reset()[0]
        done = False

        # By default, we consider our outcome to be a failure
        outcomes.append("Failure")

        # Until the agent gets stuck in a hole or reaches the goal, keep training it
        start = time.time()
        while not done:
            # Choose the action with the highest value in the current state
            if np.max(qtable[state]) > 0:
                action = np.argmax(qtable[state])

            # If there's no best action (only zeros), take a random one
            else:
                action = environment.action_space.sample()
                
            # Implement this action and move the agent in the desired direction
            new_state, reward, done, info = environment.step(action)

            # Update Q(s,a)
            qtable[state, action] = qtable[state, action] + \
                                    alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
            
            # Update our current state
            state = new_state

        end = time.time()
        
        total_reward = np.mean(qtable)
        total_time = end-start

        reward_list.append(total_reward)
        iteration_list.append(i)
        time_list.append(total_time)
        mapsize_list.append(map_size)


    policy = qtable.argmax(axis = 1)
    visualize_policy(policy,
                        shape = (map_size,map_size),
                        name = f"{map_size}x{map_size}",
                        title = f'QLearning-{map_size}x{map_size}-Discount={gamma}')
#%%
frozen_lakeq = pd.DataFrame({'reward':reward_list,
                            'time':time_list,
                            'iterations':iteration_list,
                            'map_size':mapsize_list})

#%%
##################
#Convergence Plots
##################
chart1q = alt.Chart(frozen_lakeq).mark_line(clip = True,point = True).encode(
    alt.X('iterations',title = 'Num of Iterations',scale = alt.Scale(domain = (0,2500))),
    alt.Y('reward',title = 'Reward'),
    alt.Color('map_size:N',title = 'NxN Map Size'),
)

chartf = chart1q
chartf
#%%
chartf.save('Convergence_ql.png')

# %%
########################################
# Forest
########################################
state_size = [225,2500] #Same state size as Frozen Lake
fire_prob = [.2,.5,.8]

reward_list = []
time_list = []
iterations_list = []
policy_list = []
algorithm_list = []
size_list = []
fire_list = []


for size in state_size:
    for fire in fire_prob:        
        P, R = forest(S = size, # Number of states
                    r1 = 3, # reward when old and wait, 
                    r2 = 100, # reward when old and cut,
                    p = fire) # probability of wildfire

        algos = [ValueIteration(P,R,.8),
                PolicyIteration(P,R,.8),
                QLearning(P,R,.8)]
        algos_name = ['ValueIteration','Policy_Iteration','QLearning']

        for i in range(len(algos_name)):
            print('{}-{}-{}'.format(algos_name[i],size,fire))
            test = algos[i].run()  
        
            for step in range(len(test)):
                print('{} out of {}'.format(step,len(test)))
                reward_list.append(test[step]['Reward'])
                time_list.append(test[step]['Time'])
                iterations_list.append(test[step]['Iteration'])

                policy_list.append(algos[i].policy)
                algorithm_list.append(algos_name[i])
                size_list.append(size)
                fire_list.append(fire)

# %%
fire_data = pd.DataFrame({
    'reward':reward_list,
    'time':time_list,
    'iteration':iterations_list,
    'policy':policy_list,
    'algorithm':algorithm_list,
    'size':size_list,
    'fire':fire_list
})
# %%
chart2 = alt.Chart(fire_data.loc[fire_data.algorithm == 'QLearning']).mark_line(clip = True,point = True).encode(
    alt.X('iteration',title = 'Num of Iterations', scale = alt.Scale(domain = (0,50))),
    alt.Y('reward',title = 'Reward',scale = alt.Scale(domain = (0,10))),
    alt.Color('size:N',title = 'NxN Map Size'),
    alt.Column('fire',title = 'MDP Solver')
)
chart2
# %%
