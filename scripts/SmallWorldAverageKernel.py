import numpy as np
import networkx as nx

import sys
sys.path.insert(0, '../modules/')
import model 
import utils

import copy

N = 200

path = '../data/AverageSmallWorld/'

f_kernel = [0.1, 1, 10]
xi = 1

Nrep = 500

p_SW = 0.3

for idx_f, f in enumerate(f_kernel):
    np.random.seed(42)
    avg_kernel = []
    min_kernel = []
    max_kernel = []
    all_distances = []

    for idx_rep in range(Nrep):
        print(f'Repetition ({idx_f}):', idx_rep, '/', Nrep)
        net = nx.connected_watts_strogatz_graph(N, 2, p_SW)

        kernel = model.find_effective_kernel(f, xi, net)
        avg_K, std_K, dist_K, distances = model.dist_dependence(net, kernel)
        avg_kernel.append(avg_K)
        min_kernel.append(np.array([np.min(dist_K[idx]) for idx in range(len(dist_K))]))
        max_kernel.append(np.array([np.max(dist_K[idx]) for idx in range(len(dist_K))]))
        all_distances.append(distances)

    MaxDist = np.max([np.max(i) for i in all_distances])
    print(MaxDist)

    AverageKernel = np.ones((Nrep, MaxDist + 1))*np.nan
    MinKernel = np.ones((Nrep, MaxDist + 1))*np.nan
    MaxKernel = np.ones((Nrep, MaxDist + 1))*np.nan
    for idx_rep in range(Nrep):
        for dist in all_distances[idx_rep]:
            AverageKernel[idx_rep, dist] = avg_kernel[idx_rep][dist]
            MinKernel[idx_rep, dist] = min_kernel[idx_rep][dist]
            MaxKernel[idx_rep, dist] = max_kernel[idx_rep][dist]

    np.save(path + 'AverageKernel_SmallWorld_N' + str(N) + '_f' + str(f) + '_xi' + str(xi) + '.npy', AverageKernel)
    np.save(path + 'MinKernel_SmallWorld_N' + str(N) + '_f' + str(f) + '_xi' + str(xi) + '.npy', MinKernel)
    np.save(path + 'MaxKernel_SmallWorld_N' + str(N) + '_f' + str(f) + '_xi' + str(xi) + '.npy', MaxKernel)


