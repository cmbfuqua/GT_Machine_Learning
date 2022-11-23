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
def visualize_policy(policy, shape, name, title=None, show = False):
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
                if show:
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
    alt.X('time',title = 'Time (S)',scale = alt.Scale(domain = (-1,50))),
    alt.Y('reward',title = 'Reward',scale = alt.Scale(domain = (.3,.8))),
    alt.Color('map_size:N',title = 'NxN Map Size'),
    alt.Column('algorithm',title = 'MDP Solver'),
    alt.Row('discount',title = 'Gamma value')
)
chart1
#%%
chart1.save('ConvergenceTS_vi_pi.png')
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

map_sizes = [15,20]
pfroze = 1

for map_size in map_sizes:

    maps = generate_random_map(map_size,pfroze)

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

    # Training
    for i in range(episodes):
        state = environment.reset()
        done = False

        # By default, we consider our outcome to be a failure

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
        
        total_reward = np.max(qtable)
        total_time = end-start

        reward_list.append(total_reward)
        iteration_list.append(i)
        time_list.append(total_time)
        mapsize_list.append(map_size)


    policy = qtable.argmax(axis = 1)
    visualize_policy(policy,
                        shape = (map_size,map_size),
                        name = f"{map_size}x{map_size}",
                        title = f'QLearning frozen={pfroze}-{map_size}x{map_size}-Discount={gamma}')
#%%
frozen_lakeq = pd.DataFrame({'reward':reward_list,
                            'time':time_list,
                            'iterations':iteration_list,
                            'map_size':mapsize_list})

#%%
##################
#Convergence Plots
##################
chart1q = alt.Chart(frozen_lakeq.loc[frozen_lakeq.reward != 0]).mark_line(clip = True,point = True).encode(
    alt.X('iterations',title = 'Iterations'),
    alt.Y('reward',title = 'Reward'),
    alt.Color('map_size:N',title = 'NxN Map Size'),
)

chartf = chart1q
chartf
#%%
chartf.save('ConvergenceT_ql.png')

# %%
########################################
# Black Jack https://gist.github.com/iiLaurens/ba9c479e71ee4ceef816ad50b87d9ebd
########################################

# Code to set up environment
from itertools import product
from functools import reduce

ACTIONLIST = {
    0: 'skip',
    1: 'draw'
}

CARDS = np.array([2,3,4,5,6,7,8,9,10,10,10,10,11])
BLACKJACK = 21
DEALER_SKIP = 17
STARTING_CARDS_PLAYER = 2
STARTING_CARDS_DEALER = 1

STATELIST = {0: (0,0,0)} # Game start state
STATELIST = {**STATELIST, **{nr+1:state for nr, state in enumerate(product(range(2), range(CARDS.min()*STARTING_CARDS_PLAYER,BLACKJACK + 2), range(CARDS.min()*STARTING_CARDS_DEALER, BLACKJACK+2)))}}


def cartesian(x,y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2).sum(axis=1)

def deal_card_probability(count_now, count_next, take=1):
    if take > 1:
        cards = reduce(cartesian, [CARDS]*take)
    else:
        cards = CARDS
        
    return (np.minimum(count_now + cards, BLACKJACK + 1) == count_next).sum() / len(cards)

def is_gameover(skipped, player, dealer):
    return any([
        dealer >= DEALER_SKIP and skipped == 1,
        dealer > BLACKJACK and skipped == 1,
        player > BLACKJACK
     ])

def blackjack_probability(action, stateid_now, stateid_next):
    skipped_now, player_now, dealer_now  = STATELIST[stateid_now]
    skipped_next, player_next, dealer_next = STATELIST[stateid_next]
    
    if stateid_now == stateid_next:
        # Game cannot stay in current state
        return 0.0
    
    if stateid_now == 0:
        if skipped_next == 1:
            # After start of the game the game cannot be in a skipped state
            return 0
        else:
            # State lower or equal than 1 is a start of a new game
            dealer_prob = deal_card_probability(0, dealer_next, take=STARTING_CARDS_DEALER)
            player_prob = deal_card_probability(0, player_next, take=STARTING_CARDS_PLAYER)

            return dealer_prob * player_prob
    
    if is_gameover(skipped_now, player_now, dealer_now):
        # We arrived at end state, now reset game
        return 1.0 if stateid_next == 0 else 0.0
    
    if skipped_now == 1:
        if skipped_next == 0 or player_next != player_now:
            # Once you skip you keep on skipping in blackjack
            # Also player cards cannot increase once in a skipped state
            return 0.0
    
    if ACTIONLIST[action] == 'skip' or skipped_now == 1:
        # If willingly skipped or in forced skip (attempted draw in already skipped game):
        if skipped_next != 1 or player_now != player_next:
            # Next state must be a skipped state with same card count for player
            return 0.0
    
    if ACTIONLIST[action] == 'draw' and skipped_now == 0 and skipped_next != 0:
        # Next state must be a drawable state
        return 0.0
    
    if dealer_now != dealer_next and player_now != player_next:
        # Only the player or the dealer can draw a card. Not both simultaneously!
        return 0.0

    # Now either the dealer or the player draws a card
    if ACTIONLIST[action] == 'draw' and skipped_now == 0:
        # Player draws a card
        prob = deal_card_probability(player_now, player_next, take=1)
    else:
        # Dealer draws a card
        if dealer_now >= DEALER_SKIP:
            if dealer_now != dealer_next:
                # Dealer always stands once it has a card count higher than set amount
                return 0.0
            else:
                # Dealer stands
                return 1.0

        prob = deal_card_probability(dealer_now, dealer_next, take=1)

    return prob

def blackjack_rewards(action, stateid,reward_dict):
    skipped, player, dealer  = STATELIST[stateid]
    
    if not is_gameover(skipped, player, dealer):
        reward_key = 'draw'
        return reward_dict[reward_key]
    elif player > BLACKJACK or (player <= dealer and dealer <= BLACKJACK):
        reward_key = 'loss'
        return reward_dict[reward_key]
    elif player == BLACKJACK and dealer < BLACKJACK:
        reward_key = 'perfect_win'
        return reward_dict[reward_key]
    elif player > dealer or dealer > BLACKJACK:
        reward_key = 'win'
        return reward_dict[reward_key]
    else:
        raise Exception(f'Undefined reward: {skipped}, {player}, {dealer}')
    
def print_blackjack_policy(policy):
    idx = pd.MultiIndex.from_tuples(list(STATELIST.values()), names=['Skipped', 'Player','Dealer'])
    S = pd.Series(['D' if i == 1 else '.' for i in policy], index=idx)
    S = S.loc[S.index.get_level_values('Skipped')==0].reset_index('Skipped', drop=True)
    S = S.loc[S.index.get_level_values('Dealer')>0]
    S = S.loc[S.index.get_level_values('Player')>0]
    return S.unstack(-1)

def print_blackjack_rewards():
    idx = pd.MultiIndex.from_tuples(list(STATELIST.values()), names=['Skipped', 'Player', 'Dealer'])
    S = pd.Series(R[:,0], index=idx)
    S = S.loc[S.index.get_level_values('Skipped')==1].reset_index('Skipped', drop=True)
    S = S.loc[S.index.get_level_values('Player')>0]
    S = S.loc[S.index.get_level_values('Dealer')>0]
    return S.unstack(-1)
# %%
reward_list = []
time_list = []
iteration_list = []
policy_list = []
algorithm_list = []
discount_list = []
dictionary_type_list = []

#Create different reward environments by factors of 10
reward_dict1 = {
    'draw':0,
    'loss':-1,
    'perfect_win':1.5,
    'win':1
}

reward_dict2 = {
    'draw':0,
    'loss':-10,
    'perfect_win':1.5,
    'win':1
}

reward_dict3 = {
    'draw':0,
    'loss':-1,
    'perfect_win':1.5,
    'win':10
}

dictionary_list = [reward_dict1,reward_dict2,reward_dict3]

dictionary_type = ['Base','Higher Risk','Higher Reward']

for dl in range(len(dictionary_type)):
    # Define transition matrix
    T = np.zeros((len(ACTIONLIST), len(STATELIST), len(STATELIST)))
    for a, i, j in product(ACTIONLIST.keys(), STATELIST.keys(), STATELIST.keys()):
        T[a,i,j] = blackjack_probability(a, i, j)
        
    # Define reward matrix
    R = np.zeros((len(STATELIST), len(ACTIONLIST)))
    for a, s in product(ACTIONLIST.keys(), STATELIST.keys()):
        R[s, a] = blackjack_rewards(a, s,dictionary_list[dl])

    # Check that we have a valid transition matrix with transition probabilities summing to 1
    assert (T.sum(axis=2).round(10) == 1).all()



    discount = [.9]
    for d in discount:
        algos = [ValueIteration(T,R,.9),
                PolicyIteration(T,R,.9)]

        algos_name = ['ValueIteration','Policy_Iteration']

        for i in range(len(algos_name)):
            test = algos[i].run()

            for step in range(len(test)):
                print('{} out of {}'.format(step,len(test)))
                reward_list.append(test[step]['Reward'])
                time_list.append(test[step]['Time'])
                iteration_list.append(test[step]['Iteration'])
                policy_list.append(algos[i].policy)

                algorithm_list.append(algos_name[i])
                discount_list.append(d)
                dictionary_type_list.append(dictionary_type[dl])

blackjack = pd.DataFrame({
    'reward':reward_list,
    'time':time_list,
    'iteration':iteration_list,
    'policy':policy_list,
    'algorithm':algorithm_list,
    'dictionary_type':dictionary_type_list
})
# %%
#Scale the rewards back down to /10 so we can compare behavior
blackjack.loc[blackjack.dictionary_type != 'Base','reward'] = blackjack.loc[blackjack.dictionary_type != 'Base'].reward.div(10).abs()
#%%
bj_chart = alt.Chart(blackjack).mark_line(point = True).encode(
    alt.X('time',title = 'Time'),
    alt.Y('reward',title = 'Reward (Absolute Value)'),
    alt.Column('algorithm'),
    alt.Color('dictionary_type',title = 'Reward Method')
)
bj_chart.save('bj_convergenceT_vi_pi.png')
bj_chart
# %% Create data to visualize policy
distinct_policy = blackjack[['algorithm','dictionary_type','policy']].drop_duplicates().reset_index(drop = True)
# 1 or x = draw 0 or . = pass
dealer_list = []
player_list = []
value_list = []
algorithm_list = []
dic_type = []
for i in range(len(distinct_policy)):
    temppol = distinct_policy['policy'][i]
    temppol = print_blackjack_policy(temppol)
    for player in temppol:
        if len(temppol[player].value_counts()) == 1:
            #print('no values')
            value_list.append(0)
            dealer_list.append(player)
            algorithm_list.append(distinct_policy['algorithm'][i])
            dic_type.append(distinct_policy['dictionary_type'][i])
        else:
            #print('has values')
            xvals = temppol[player][temppol[player] == 'D'].index.values
            for x in range(len(xvals)):
                print(xvals[x])
                value_list.append(xvals[x])
                dealer_list.append(player)
                algorithm_list.append(distinct_policy['algorithm'][i])
                dic_type.append(distinct_policy['dictionary_type'][i])
policy_data = pd.DataFrame({
    'dealer':dealer_list,
    'draw_til':value_list,
    'algorithm':algorithm_list,
    'dictionary_type':dic_type
})
# %%
bj_policy_vi_pi = alt.Chart(policy_data).mark_point(clip = True).encode(
    alt.X('dealer',title = 'When Dealer Has:'),
    alt.Y('draw_til',title = 'Draw If I Have',scale = alt.Scale(domain = (3,22))),
    alt.Row('algorithm'),
    alt.Column('dictionary_type',title = 'Reward Method')
)
bj_policy_vi_pi
# %%
bj_policy_vi_pi.save('bj_policy_vi_pi.png')
# %%
#######################
# QLEARNING
########################
reward_list = []
time_list = []
iteration_list = []
policy_list = []
algorithm_list = []
discount_list = []
dictionary_type_list = []

reward_dict1 = {
    'draw':0,
    'loss':-1,
    'perfect_win':1.5,
    'win':1
}

reward_dict2 = {
    'draw':0,
    'loss':-10,
    'perfect_win':1.5,
    'win':1
}

reward_dict3 = {
    'draw':0,
    'loss':-1,
    'perfect_win':1.5,
    'win':10
}

dictionary_list = [reward_dict1,reward_dict2,reward_dict3]

dictionary_type = ['Base','Higher Risk','Higher Reward']


for dl in range(len(dictionary_type)):
    # Define transition matrix
    T = np.zeros((len(ACTIONLIST), len(STATELIST), len(STATELIST)))
    for a, i, j in product(ACTIONLIST.keys(), STATELIST.keys(), STATELIST.keys()):
        T[a,i,j] = blackjack_probability(a, i, j)
        
    # Define reward matrix
    R = np.zeros((len(STATELIST), len(ACTIONLIST)))
    for a, s in product(ACTIONLIST.keys(), STATELIST.keys()):
        R[s, a] = blackjack_rewards(a, s,dictionary_list[dl])

    # Check that we have a valid transition matrix with transition probabilities summing to 1
    assert (T.sum(axis=2).round(10) == 1).all()



    discount = [.9]
    for d in discount:
        algos = [QLearning(T,R,.9,n_iter=30000)]

        algos_name = ['QLearning']

        for i in range(len(algos_name)):
            test = algos[i].run()

            for step in range(len(test)):
                #print('{} out of {}'.format(step,len(test)))
                reward_list.append(test[step]['Reward'])
                time_list.append(test[step]['Time'])
                iteration_list.append(test[step]['Iteration'])
                policy_list.append(algos[i].policy)

                algorithm_list.append(algos_name[i])
                discount_list.append(d)
                dictionary_type_list.append(dictionary_type[dl])

blackjackQL = pd.DataFrame({
    'reward':reward_list,
    'time':time_list,
    'iteration':iteration_list,
    'policy':policy_list,
    'algorithm':algorithm_list,
    'dictionary_type':dictionary_type_list
})
#blackjack.loc[blackjack.dictionary_type != 'Base','reward'] = blackjack.loc[blackjack.dictionary_type != 'Base'].reward.div(10).abs()
#%%
bj_chartQL = alt.Chart(blackjackQL).mark_point().encode(
    alt.X('iteration',title = 'Iteration'),
    alt.Y('reward',title = 'Reward (Absolute Value)'),
    alt.Column('dictionary_type',title = 'Reward Method')
)
bj_chartQL.save('bj_convergence_ql.png')
bj_chartQL
# %%
distinct_policy = blackjackQL[['algorithm','dictionary_type','policy']].drop_duplicates().reset_index(drop = True)
# 1 or x = draw 0 or . = pass
dealer_list = []
player_list = []
value_list = []
algorithm_list = []
dic_type = []
for i in range(len(distinct_policy)):
    temppol = distinct_policy['policy'][i]
    temppol = print_blackjack_policy(temppol)
    for player in temppol:
        if len(temppol[player].value_counts()) == 1:
            #print('no values')
            value_list.append(0)
            dealer_list.append(player)
            algorithm_list.append(distinct_policy['algorithm'][i])
            dic_type.append(distinct_policy['dictionary_type'][i])
        else:
            #print('has values')
            xvals = temppol[player][temppol[player] == 'D'].index.values
            for x in range(len(xvals)):
                print(xvals[x])
                value_list.append(xvals[x])
                dealer_list.append(player)
                algorithm_list.append(distinct_policy['algorithm'][i])
                dic_type.append(distinct_policy['dictionary_type'][i])
policy_dataQL = pd.DataFrame({
    'dealer':dealer_list,
    'draw_til':value_list,
    'algorithm':algorithm_list,
    'dictionary_type':dic_type
})
# %%
bj_policy_ql = alt.Chart(policy_dataQL).mark_point(clip = True).encode(
    alt.X('dealer',title = 'When Dealer Has:'),
    alt.Y('draw_til',title = 'Draw If I Have',scale = alt.Scale(domain = (3,22))),
    alt.Color('algorithm'),
    alt.Column('dictionary_type',title = 'Reward Method')
)

bj_policy_ql
#%%
bj_policy_ql.save('bj_policy_ql.png')

# %%
bj_chartQL = alt.Chart(blackjackQL.loc[blackjackQL.dictionary_type == 'Base']).mark_boxplot().encode(
   alt.X('dictionary_type'),
   alt.Y('time')
)
bj_chartQL
bj_chartQL.save('bj_ql_bxplt.png')

# %%
