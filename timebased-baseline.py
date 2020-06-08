# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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


# %%
benchmark_1_data_file = "output/benchmark_1.txt"
benchmark_2_data_file = "output/benchmark_2.txt"
benchmark_3_data_file = "output/benchmark_3.txt"

question_header = 'q_'
user_header = 'u_'
tag_header = 't_'

# %% [markdown]
# ## Pre-processing

# %%
# loads data with pands, it eats up memory, but parsing with pyspark is much more work
df = pd.read_csv("Answers.csv", encoding="ISO-8859-1")
df.head(5)


# %%
df.shape


# %%
# Question_ids and user_ids may overlap, but that does not mean questions are users!!!
# Soln: each question_id += max_user_id
max_user_id = df[['OwnerUserId']].max()
max_user_id


# %%
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

# %% [markdown]
# ## Benchmark on similar edges

# %%
import random
from collections import Counter
# parameters
n_test_edge = 1000
n_steps = 1000
teleportation_alpha = 0.3
origin_teleport_alpha = 0.7
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


# %%
# load benchmark file
b1_data = load_benchmark_data(benchmark_1_data_file)
b2_data = load_benchmark_data(benchmark_2_data_file)
b3_data = load_benchmark_data(benchmark_3_data_file)

# %% [markdown]
# ## Baseline in time

# %%
df["CreationDate"] = pd.to_datetime(df["CreationDate"])
df["ParentId"] = df["ParentId"] + 7053078
df.head(2)


# %%

def run_benchmark(curr_bench_data):
    result = []
    df_copy = df.copy()
    correct_count = 0.0
    for question, user in tqdm(curr_bench_data):
        suggestions = []
        question_temp = int(question[2:]) + 7053078
        user_temp = int(user[2:])

        df_user = df[(df.OwnerUserId == user_temp)].sort_values(by=['CreationDate'], ascending = False)
        curr_df = df[(df.OwnerUserId == user_temp) & (df.ParentId == question_temp)]
        curr_time = list(curr_df["CreationDate"])[0]

        df_user["filter_time"] = curr_time
        df_user = df_user[df_user.filter_time > df_user.CreationDate]
        if df_user.shape[0] >= 1:
            pre_question = list(df_user["ParentId"])[0]
            pre_time = list(df_user["CreationDate"])[0]
            
            df_copy["filter_time"] = pre_time 
            df_copy = df_copy[df_copy.CreationDate > df_copy.filter_time]
            df_copy["diff_time"] = df_copy["CreationDate"]  - df_copy["filter_time"]
            df_result = df_copy.drop_duplicates(["ParentId"])
            df_result = df_result.sort_values(by=["diff_time"])
            suggestions = list(df_result["ParentId"])[1:101]

        correct_count += 1 if question in suggestions else 0
        result.append((user, question, suggestions))
    
    return correct_count / len(curr_bench_data)

n_cores = 8
for idx, curr_bench_data in enumerate([b1_data, b2_data, b3_data]):
    chuncks_of_data = np.array_split(curr_bench_data, n_cores)

    with Pool(n_cores) as p:
        final_accus = p.map(run_benchmark, chuncks_of_data)

    save_to = "arthur_time_based_baseline" + str(idx +1) + ".pk1"
    with open(save_to, 'wb') as handle:
        pickle.dump(final_accus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    avg_accu = [np.average(i) for i in final_accus]
    print('for benchmark {}, the accu is {}'.format(i, avg_accu))


# %%
# count = 0
# for e in result:
#     if int(e[1][2:]) + 7053078 in e[2]:
#         count += 1


# # %%
# count/1000


# # %%
# result[4][2] == result[5][2] 


# # %%
# result 

