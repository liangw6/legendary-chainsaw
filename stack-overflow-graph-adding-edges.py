
# coding: utf-8

# In[13]:

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle
from tqdm import tqdm
from multiprocessing import Pool
# This uses the Answers.csv file from the 10% Stack Overflow data
answer_file = "Answers.csv"
# This edge list is the intermediate file used for graph building
edges_list_file = "answer_edges.txt"


# In[14]:

benchmark_1_data_file = "output/benchmark_1.txt"
benchmark_2_data_file = "output/benchmark_2.txt"
benchmark_3_data_file = "output/benchmark_3.txt"

question_header = 'q_'
user_header = 'u_'
tag_header = 't_'


# ## Pre-processing

# In[ ]:

# loads data with pands, it eats up memory, but parsing with pyspark is much more work
df = pd.read_csv("Answers.csv", encoding="ISO-8859-1")
df.head(5)


# In[ ]:

df.shape


# In[ ]:

# Question_ids and user_ids may overlap, but that does not mean questions are users!!!
# Soln: each question_id += max_user_id
max_user_id = df[['OwnerUserId']].max()
max_user_id


# In[ ]:

edge_df = df[['OwnerUserId', 'ParentId']]
# 1. drop null values
edge_df = edge_df.dropna()
# 2. make parentIds unique
edge_df = edge_df.assign(newParentId=lambda x: x.ParentId + max(max_user_id))
edge_df = edge_df.drop(['ParentId'], axis=1)
# 3. add weights to edges
edge_df['EdgeWeight'] = 1
# 4. cast the datafraem to int type
edge_df = edge_df.astype('int32')
edge_df.head(2)


# ## Build Graph

# In[ ]:

# by default, nx creates undirected edges, exactly what we want
G = nx.read_edgelist(edges_list_file, nodetype=int, data=(('weight',float),))
print(nx.info(G))


# In[ ]:

all_user_ids = set()
all_question_ids = set()
with open(edges_list_file, 'r') as read_file:
    for line in read_file.readlines():
        user_id, question_id, weight = line.strip().split(' ')
        all_user_ids.add(int(user_id))
        all_question_ids.add(int(question_id))
print(list(all_user_ids)[:10])
print(list(all_question_ids)[:10])
# should be no intersection between user_ids and question_ids
print(len(all_user_ids.intersection(all_question_ids)))


# In[ ]:

# General Data Analysis
islands = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
print("connected components", islands[:10])


# In[ ]:

# analyze how connected the graph is
# connectivity of 0..... oh well
from networkx.algorithms import approximation as approx
approx.node_connectivity(G)


# ## Loading edges from similarity score 

# In[ ]:

import pickle
with open("neighbors-10.pickle", 'rb') as data:
    similarity_data = pickle.load(data)


# In[ ]:

similarity_data


# In[ ]:

questionID = list(similarity_data.keys())
unique_questionID = set(questionID)
unique_newParentID = set(list(edge_df["newParentId"]))
print("unique_newParentID", len(unique_newParentID))
print("unique_questionID", len(unique_questionID))
print("difference:", len(unique_questionID.difference(unique_newParentID)))


# ## Benchmark on similar edges

# In[ ]:

import random
from collections import Counter
# parameters
n_test_edge = 1000
n_steps = 1000
teleportation_alpha = 0.3
origin_teleport_alpha = 0.45
early_stop_threshold = 20

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
def random_walk_on_edge(curr_model, x, y, teleportation_alpha, origin_teleport_alpha, top_n=100):
    # curr_pos is always on user nodes
#     starting_pos = x if x in all_user_ids else y
    y = int(y[2:])
    starting_pos = int(x[2:]) if x[:2] == user_header else y
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
            
            #### get similar question node from dict
            if question_node in unique_questionID:
                for cand in similarity_data[question_node][0]:
                    if cand in unique_newParentID:
                        similar_question_node = cand
                    else:
                        similar_question_node = question_node
            else:
                similar_question_node = question_node
                
            ##### trace back user node of the similar question from G[similar_question_node]
            potential_similar_user_nodes = random.sample(set(curr_model[similar_question_node]), 2)

            ### choice from similar questions (i.e artifical edges)
            similar_user_node = potential_similar_user_nodes[0] if not                                     is_same_edge(x, y, potential_similar_user_nodes[0], similar_question_node)                                     else potential_similar_user_nodes[1]
                
            ### choice returned from random walk
            user_node = potential_user_nodes[0] if not is_same_edge(x, y, potential_user_nodes[0], question_node) else potential_user_nodes[1]
            if random.random() < teleportation_alpha:
                curr_pos = starting_pos
            elif teleportation_alpha <= random.random() < (teleportation_alpha + origin_teleport_alpha):
                ### choice from teleporting back to origin
                curr_pos = user_node
            else:
                curr_pos = similar_user_node
                ### best threshold is teleportation_alpha = 0.3, origin_teleport_alpha = 0.45, and 0.25 will teleport to 
                ### similar questions
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

    return x, visit_distribution
    
# uses curr_model & teleportation_alpha from global variables
def run_benchmark(curr_bench_data):
    count = 0
    recommendation_list = []
    for question, user in tqdm(curr_bench_data):
        _,curr_distr = random_walk_on_edge(curr_model, user, question, teleportation_alpha=teleportation_alpha,                                           origin_teleport_alpha = origin_teleport_alpha, top_n=100)
        count += len([i for i in curr_distr if i[0] == question])>=1
        recommendation_list.append((user, question, curr_distr))
        
    percentage = count / len(curr_bench_data)
    print("Correct percentage", percentage)
    return recommendation_list

def run_benchmark_parallel(curr_bench_data, n_core=8):
    # split data into chucks
    chuncks_of_data = np.array_split(curr_bench_data, n_core)

    with Pool(n_core) as p:
        final_accus = p.map(run_benchmark, chuncks_of_data)
#     print(final_accus)
    return final_accus


# #### Next step 
# 1. load all user from benchmark, join with all dataset 
# run random walk on all user and questions and calculate the correct percentage 
# while running the random walk, store user with top 100 recommendations from random walk 
# then check if targeted question is within the top 100 questions and if targeted question is within the top 100 questions of the baseline  
# 
# 2. Each line of benchmark is a user - question pair. the random walker 
#  tries to recommend top 100 questions to the user and we check to see 
#  if targeted question is within the top 100 questions and if targeted question is within the top 100 questions
#  of the baseline 
# 3. we need to store user and 100 top recommendations from random walk 
# and calculate the correct percentage for all users at the end 

# In[ ]:

# load benchmark file
b1_data = load_benchmark_data(benchmark_1_data_file)
b2_data = load_benchmark_data(benchmark_2_data_file)
b3_data = load_benchmark_data(benchmark_3_data_file)


# In[ ]:

curr_model = G
print(nx.info(curr_model), flush=True)

for idx, curr_bench_data in enumerate([b1_data, b2_data,b3_data]):
    recommendation = run_benchmark_parallel(curr_bench_data)
    save_to = "benchmark" + str(idx +1) + "_add_eges.pk1"
    with open(save_to, 'wb') as handle:
        pickle.dump(recommendation, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # Baseline
# - return top 100 randomly questions for each user

# In[ ]:

question_user_df = edge_df.groupby(["newParentId"])["OwnerUserId"].apply(list).reset_index(name="user_list")
question_user_df["num_user"] = question_user_df["user_list"].apply(lambda x: len(x))
question_user_df["user"] = question_user_df["user_list"].apply(lambda x: random.sample(x,1)[0])
question_user_df.head(5)


# In[ ]:



