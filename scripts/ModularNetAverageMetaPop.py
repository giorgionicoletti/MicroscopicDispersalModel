import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import igraph
import scipy.optimize as optimize

import sys
sys.path.insert(0, '../modules/')
import model 
import utils

import copy

N_modules = 6
Nodes_per_module = [100]*N_modules

N = sum(Nodes_per_module)
Nconn_per_node = 1

cumsum = np.concatenate([[0], np.cumsum(Nodes_per_module)])

f_array = np.geomspace(0.001, 1000, 50)
xi_array = np.linspace(0.5, 10, 5)

p_intra_arr = np.linspace(0.05, 0.5, 10)
p_conn_arr = np.linspace(0.05, 0.3, 10)

Nrep = 10
NBatch = 100

np.random.seed(42)
seeds = np.random.randint(int(1e9), size = NBatch*Nrep*p_conn_arr.size*p_intra_arr.size)
seeds = seeds.reshape(NBatch, Nrep, p_conn_arr.size, p_intra_arr.size)

for idx_batch in range(NBatch):
    print('Batch:', idx_batch, '/', NBatch)
    LambdaMax_all = np.zeros((p_intra_arr.size, p_conn_arr.size, Nrep, xi_array.size, f_array.size))
    average_degree_all = np.zeros((p_intra_arr.size, p_conn_arr.size, Nrep))
    for idx_int, p_intra in enumerate(p_intra_arr):
        for idx_conn, p_connect in enumerate(p_conn_arr):
            print('\t Generating network with p_intra =', p_intra, 'p_connect =', p_connect)
            for idx_rep in range(Nrep):
                seed = seeds[idx_batch, idx_rep, idx_conn, idx_int]

                print('\t\t Repetition:', idx_rep, '/', Nrep)
                net, edges_modular = utils.generate_modular_ER(**{'N': Nodes_per_module, 'Nmod': N_modules,
                                                                  'p_intra': p_intra, 'p_connect': p_connect,
                                                                  'Nconn_per_node': Nconn_per_node, 'seed': seed})
                
                average_degree_all[idx_int, idx_conn, idx_rep] = utils.get_average_degree(net)
                
                LambdaMax = model.find_all_metapop(f_array, xi_array, net, undirected = True,
                                                  guaranteed_connected = True)
                LambdaMax_all[idx_int, idx_conn, idx_rep] = LambdaMax
    np.save('../data/AverageModular/LambdaMax_all_batch_' + str(idx_batch) + '.npy', LambdaMax_all)
    np.save('../data/AverageModular/average_degree_all_batch_' + str(idx_batch) + '.npy', average_degree_all)
