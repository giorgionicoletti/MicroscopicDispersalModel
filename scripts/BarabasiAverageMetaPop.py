import numpy as np
import networkx as nx

import sys
sys.path.insert(0, '../modules/')
import model 
import utils

import copy

N = 200

path = '../data/AverageBarabasi/'

f_array = np.geomspace(0.001, 1000, 100)
xi_array = np.linspace(0.5, 10, 5)

np.save(path + 'f_array.npy', f_array)
np.save(path + 'xi_array.npy', xi_array)

Nrep = 10
NBatch = 20

for idx_batch in range(NBatch):
    print('Batch:', idx_batch, '/', NBatch)
    LambdaMax_all = np.zeros((Nrep, xi_array.size, f_array.size))
    average_degree_all = np.zeros(Nrep)
    average_kernel = np.zeros((N, N))

    for idx_rep in range(Nrep):
        print('\t\t Repetition:', idx_rep, '/', Nrep)
        net = nx.barabasi_albert_graph(N, m = 1)
        LambdaMax = model.find_all_metapop(f_array, xi_array, net, undirected = True,
                                            guaranteed_connected = True)
        LambdaMax_all[idx_rep] = LambdaMax
        average_degree_all[idx_rep] = utils.get_average_degree(net)

    np.save(path + 'LambdaMax_all_batch_' + str(idx_batch) + '_Barabasi_N' + str(N) + '.npy', LambdaMax_all)
    np.save(path + 'average_degree_all_batch_' + str(idx_batch) + '_Barabasi_N' + str(N) + '.npy', average_degree_all)