# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import math

benchmark_1_data_file = "output/benchmark_1.txt"
benchmark_2_data_file = "output/benchmark_2.txt"
benchmark_3_data_file = "output/benchmark_3.txt"

question_header = 'q_'
user_header = 'u_'
tag_header = 't_'

# model_file = 


# %%

import random
from collections import Counter

def load_benchmark_data(benchmark_file, print_first_5_data=True):
    benchmark_data = []
    with open(benchmark_file, 'r') as input_file:
        for line in input_file:
            benchmark_data.append(line.strip().split())
    print(np.array(benchmark_data).shape)
    if print_first_5_data:
        print(benchmark_data[:5])
    return benchmark_data

# returns whether x2 - y2 edge is the same as x1 - y1 edge
# is_same_edge(1, 2, 2, 1) == is_same_edge(1, 2, 1, 2) == True
def is_same_edge(x1, y1, x2, y2):
    if x1 == x2:
        return y1 == y2
    elif x1 == y2:
        return y1 == x2
    return False

# a random walk on the user_node (x or y)
# pretend the direct edge (x, y) does not exist
# returns a distribution of questions nodes
def random_walk_on_edge(curr_model, x, y, teleportation_alpha=0.3, early_stop_threshold=20, n_steps=1000, top_n=None):
    # curr_pos is either user nodes or tag_node
    starting_pos = x if x[:2] == user_header else y
    # print('starting at', starting_pos)
    curr_pos = starting_pos
    reacheable_count = Counter()
    for s in range(n_steps):
        try:
            potential_questions_nodes = random.sample(set(curr_model[curr_pos]), 2)
            question_node = potential_questions_nodes[0] if not is_same_edge(x, y, potential_questions_nodes[0], curr_pos) else potential_questions_nodes[1]
            # diff from pinterest algorithm in that we care about questions, not users
            reacheable_count[question_node] += 1
            if sum(reacheable_count.values()) / len(reacheable_count.values()) >= early_stop_threshold:
                # if average > early_stop_threshold, stop
                print('early stopping!')
                break
            potential_user_nodes = random.sample(set(curr_model[question_node]), 2)
            user_node = potential_user_nodes[0] if not is_same_edge(x, y, potential_user_nodes[0], question_node) else potential_user_nodes[1]
            if random.random() < teleportation_alpha:
                curr_pos = starting_pos
            else:
                curr_pos = user_node
        except ValueError:
            # encouter valueError during random.sample when population is smaller than 2
            # This only happens if we reached a deadend
            # simply teleport back
            curr_pos = starting_pos
    # calculate distribution
    tot_visits = sum(reacheable_count.values())
    # sort visits by counts
    all_visits = sorted(reacheable_count.items(), key=lambda x: x[1], reverse = True)
    
    visit_distribution = [(i[0], i[1] / tot_visits) for i in all_visits]
    if top_n is not None:
        visit_distribution = visit_distribution[:top_n]

    return visit_distribution

# uses curr_model & teleportation_alpha from global variables
def run_benchmark(curr_bench_data):
    count = 0
    for question, user in tqdm(curr_bench_data):
        curr_distr = random_walk_on_edge(curr_model, user, question, teleportation_alpha=teleportation_alpha, top_n=100)
        count += len([i for i in curr_distr if i[0] == question])>=1
    return count / len(curr_bench_data)

def run_benchmark_parallel(curr_bench_data, n_core=8):
    # split data into chucks
    chuncks_of_data = np.array_split(curr_bench_data, n_core)

    with Pool(n_core) as p:
        final_accus = p.map(run_benchmark, chuncks_of_data)
    print(final_accus)
    return final_accus


# %%
# load benchmark file
b1_data = load_benchmark_data(benchmark_1_data_file)
b2_data = load_benchmark_data(benchmark_2_data_file)
b3_data = load_benchmark_data(benchmark_3_data_file)


# %%
# load model
# specify parameters here
teleportation_alpha = 0.3
curr_model = nx.read_edgelist('output/original_edges.txt', nodetype=str, data=(('weight',float),))
print(nx.info(curr_model), flush=True)

all_accu = []
for curr_bench_data in [b1_data, b2_data, b3_data]:
    all_accu.append(run_benchmark_parallel(curr_bench_data))

with open('original_model_result.pkl', 'wb') as handle:
    pickle.dump(all_accu, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
# with open('original_model_result.pkl', 'rb') as handle:
#     all_accu = pickle.load(handle)
# plt.plot([np.average(i) for i in all_accu])


# %%
# load model
# specify parameters here
teleportation_alpha = 0.3
curr_model = nx.read_edgelist('output/tripartite_edges.txt', nodetype=str, data=(('weight',float),))
print(nx.info(curr_model), flush=True)

all_accu = []
for curr_bench_data in [b1_data, b2_data, b3_data]:
    all_accu.append(run_benchmark_parallel(curr_bench_data))

with open('tag_model_result.pkl', 'wb') as handle:
    pickle.dump(all_accu, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%


