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

N = 200

path = '../data/AverageSmallWorld/'

f_array = np.geomspace(0.001, 1000, 100)
xi_array = np.linspace(0.5, 10, 5)
p_SW_arr = np.linspace(0.05, 1, 20)

np.save(path + 'f_array.npy', f_array)
np.save(path + 'xi_array.npy', xi_array)
np.save(path + 'p_SW_arr.npy', p_SW_arr)

Nrep = 10
NBatch = 20

np.random.seed(42)
seeds = np.random.randint(int(1e9), size = NBatch*Nrep*p_SW_arr.size)
seeds = seeds.reshape(NBatch, Nrep, p_SW_arr.size)

for idx_batch in range(NBatch):
    print('Batch:', idx_batch, '/', NBatch)
    LambdaMax_all = np.zeros((p_SW_arr.size, Nrep, xi_array.size, f_array.size))
    for idx_p, p_SW in enumerate(p_SW_arr):
        print('\t Generating network with p_SW =', p_SW)
        for idx_rep in range(Nrep):
            seed = seeds[idx_batch, idx_rep, idx_p]

            print('\t\t Repetition:', idx_rep, '/', Nrep)
            net = nx.connected_watts_strogatz_graph(N, 2, p_SW)

            LambdaMax = model.find_all_metapop(f_array, xi_array, net, undirected = True,
                                                guaranteed_connected = True)
            LambdaMax_all[idx_p, idx_rep] = LambdaMax
    np.save(path + 'LambdaMax_all_batch_' + str(idx_batch) + '_SmallWorld_N' + str(N) + '.npy', LambdaMax_all)