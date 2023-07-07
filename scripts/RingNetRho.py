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

path = '../data/Ring/'

cvec = np.ones(NDyn)
e_array = np.linspace(1, 2.5, 200)

rho_average = np.zeros(e_array.size)

for idx_e, eval in enumerate(e_array):
    evec = np.ones(NDyn)*eval
    print(f'Repetition ({idx_e} / {e_array.size})')
    net = nx.cycle_graph(NDyn)

    kernel = model.find_effective_kernel(f_dyn, xi_dyn, net)
    rho = model.simulate(NDyn, Nsteps, dt, kernel, cvec, evec, rho0 = rho_0, check_stationary = True)
    rho_average[idx_e] = np.mean(rho[-1])
        

np.save(path + 'StatPop_Ring_N' + str(NDyn) + '_f' + str(f_dyn) + '_xi' + str(xi_dyn) + '.npy', rho_average)
np.save(path + 'e_over_c_array_Ring_N' + str(NDyn) + '_f' + str(f_dyn) + '_xi' + str(xi_dyn) + '.npy', e_array)
