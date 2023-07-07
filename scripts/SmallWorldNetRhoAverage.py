import numpy as np
import networkx as nx

import sys
sys.path.insert(0, '../modules/')
import model 
import utils

import copy

NDyn = 50

Nsteps = 100000
dt = 1e-2

Time = np.arange(0, Nsteps*dt, dt)

rho_0 = np.zeros(NDyn)
rho_0[0] = 0.001

f_dyn = 0.5
xi_dyn = 3.5

path = '../data/AverageSmallWorld/'

cvec = np.ones(NDyn)
e_array = np.linspace(1, 3., 200)

Nrep = 1000

p_SW = 0.3

rho_average = np.zeros((Nrep, e_array.size))

for idx_rep in range(Nrep):
    net = nx.connected_watts_strogatz_graph(NDyn, 2, p_SW)
    for idx_e, eval in enumerate(e_array):
        print(f'Repetition ({idx_rep} / {Nrep}):', idx_e, '/', e_array.size)
        evec = np.ones(NDyn)*eval

        kernel = model.find_effective_kernel(f_dyn, xi_dyn, net)
        rho = model.simulate(NDyn, Nsteps, dt, kernel, cvec, evec, rho0 = rho_0, check_stationary = True)
        rho_average[idx_rep, idx_e] = np.mean(rho[-1])
        

np.save(path + 'AverageStatPop_SmallWorld_N' + str(NDyn) + '_f' + str(f_dyn) + '_xi' + str(xi_dyn) + '.npy', rho_average)
np.save(path + 'e_over_c_array_SmallWorld_N' + str(NDyn) + '_f' + str(f_dyn) + '_xi' + str(xi_dyn) + '.npy', e_array)

