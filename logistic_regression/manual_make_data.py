from make_logistic_data import make_si_table, make_data
from make_graph import make_graph

import os, time
from math import ceil
import pandas as pd
import numpy as np

pop_file = '../sample_data/va_person.csv'
disease_file = '../sample_data/va_disease_outcome_training.csv'
graph_file = '../sample_data/va_population_network.csv'
n_jobs = 4
pid_partition = 0 #123
min_date = 0 # 50 for eval
is_eval = False
out_file = f'../sample_data/logistic_regression/train/train_{pid_partition}.csv'
# out_file = f'../sample_data/logistic_regression/eval/train_{pid_partition}.csv'


pop = pd.read_csv(pop_file, engine="pyarrow")
disease_data = pd.read_csv(disease_file, engine="pyarrow")
si_table = make_si_table(disease_data)
pop.set_index("pid", inplace=True)

graph = make_graph(graph_file)
data = []

disease = si_table[si_table.index.get_level_values("infected") >= (min_date - 3 )]

vertex_names = list(graph.vs["name"])
vertex_names.sort()

part_size = ceil(len(vertex_names) / n_jobs)
pid_part = vertex_names[pid_partition * part_size:(pid_partition + 1) * part_size]

for pid in pid_part:
    pid_data = make_data(pid, graph, disease, pop, 3,)
    data.append(pid_data)

    # if is_eval and len(pid_data) > 0:
    #     data.append(pid_data[pid_data["s_t+d"] != 1.0])
    # else:
    #     data.append(pid_data)


train = pd.concat(data)
train.to_csv( out_file, index=False)
